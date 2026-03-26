import asyncio
import json
import re
import time
import pickle
import tempfile
import os
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.logger import log_tool_call, log_error

MCP_SERVERS = {
    "drive":  Path(__file__).parent.parent / "mcp_servers" / "drive_server.py",
    "gmail":  Path(__file__).parent.parent / "mcp_servers" / "gmail_server.py",
    "github": Path(__file__).parent.parent / "mcp_servers" / "github_server.py",
}

# Track tools called in current request
_tools_called_this_request = []
_current_creds_file        = None
_current_github_creds_file = None
_last_tool_results         = {}  # stores last result per tool name for smart reply formatting


def reset_tools_called():
    global _tools_called_this_request, _last_tool_results
    _tools_called_this_request = []
    _last_tool_results = {}


def get_tools_called():
    return list(_tools_called_this_request)


def get_last_tool_result(tool_name: str) -> str:
    """Get the last result from a specific tool call this request."""
    return _last_tool_results.get(tool_name, "")


def set_current_creds(creds):
    """Save Google credentials to temp file for MCP server processes to use."""
    global _current_creds_file
    cleanup_creds_file()
    if creds:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pickle", prefix="mcp_creds_"
        )
        with open(tmp.name, "wb") as f:
            pickle.dump(creds, f)
        _current_creds_file = tmp.name


def set_current_github_creds(token_data: dict):
    """Save GitHub token to temp file for MCP server processes to use."""
    global _current_github_creds_file
    cleanup_github_creds_file()
    if token_data:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", prefix="mcp_github_creds_"
        )
        with open(tmp.name, "w") as f:
            json.dump(token_data, f)
        _current_github_creds_file = tmp.name


def cleanup_creds_file():
    """Delete temp Google credentials file."""
    global _current_creds_file
    if _current_creds_file and os.path.exists(_current_creds_file):
        try:
            os.unlink(_current_creds_file)
        except Exception:
            pass
    _current_creds_file = None


def cleanup_github_creds_file():
    """Delete temp GitHub credentials file."""
    global _current_github_creds_file
    if _current_github_creds_file and os.path.exists(_current_github_creds_file):
        try:
            os.unlink(_current_github_creds_file)
        except Exception:
            pass
    _current_github_creds_file = None


async def execute_tool_async(server_name: str, tool_name: str, tool_args: dict) -> str:
    server_path = MCP_SERVERS.get(server_name)
    if not server_path:
        return f"Error: Unknown server '{server_name}'"

    env = os.environ.copy()

    # Pass Google creds
    if _current_creds_file and os.path.exists(_current_creds_file):
        env["MCP_CREDS_FILE"] = _current_creds_file

    # Pass GitHub creds
    if _current_github_creds_file and os.path.exists(_current_github_creds_file):
        env["GITHUB_CREDS_FILE"] = _current_github_creds_file

    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
        env=env,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            if result.content:
                return "\n".join(
                    item.text for item in result.content if hasattr(item, "text")
                )
            return "Tool executed but returned no output."


# ── Sanitizers ────────────────────────────────────────────────────────────────

def _sanitize_search_query(query: str) -> str:
    """
    Clean search query for Groq compatibility.
    - Remove file extensions (.docx, .pdf etc)
    - Replace underscores with spaces
    - Keep only first 3 meaningful words
    """
    # Remove file extension
    query = re.sub(r'\.[a-zA-Z0-9]{2,5}$', '', query)
    # Replace underscores
    query = query.replace('_', ' ').strip()
    # Keep first 3 words — short keyword search works better
    words = [w for w in query.split() if len(w) > 1]  # filter single chars
    return ' '.join(words[:3]) if len(words) > 3 else ' '.join(words)


def _sanitize_path(path: str) -> str:
    """
    Clean file path for Groq compatibility.
    Only replace underscores in Drive search paths.
    NEVER sanitize local filesystem paths like /Users/bala/Downloads/
    """
    if not path:
        return path

    if path.startswith("/") or path.startswith("~"):
        return path

    if '/' in path:
        folder, filename = path.rsplit('/', 1)
        filename = filename.replace('_', ' ')
        return f"{folder}/{filename}"

    return path.replace('_', ' ')


def _sanitize_email_arg(value: str) -> str:
    """
    Clean email-related arguments.
    - Strip whitespace
    - Remove accidental quotes around email addresses
    """
    value = value.strip().strip('"').strip("'")
    return value


def _sanitize_arg(tool_name: str, key: str, value: str) -> str:
    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return value

    # GitHub tools — never sanitize repo names, file paths, or branch names
    github_tools = {
        "list_repos", "create_repo", "search_repos", "list_repo_files",
        "read_file_from_repo", "list_issues", "create_issue",
        "read_issue", "list_pull_requests", "create_pull_request",
    }
    if tool_name in github_tools:
        return value

    # Search query — short keyword, no extension, no underscores
    if key == "query":
        return _sanitize_search_query(value)

    # Drive paths — sanitize underscores for Drive search
    # BUT not file_path — that's a local filesystem path, never touch it
    if key in ("path", "source_path", "destination_folder"):
        return _sanitize_path(value)

    # file_path is always a local path — never sanitize
    if key == "file_path":
        return value  # return as-is

    # Email addresses
    if key in ("to", "cc", "bcc", "from", "to_email", "from_email"):
        return _sanitize_email_arg(value)

    # Max results
    if key == "max_results":
        digits = re.sub(r'\D', '', value)
        return digits if digits else "5"

    return value


# ── Main Executor ─────────────────────────────────────────────────────────────

def execute_tool(
    server_name: str,
    tool_name: str,
    tool_args,
    creds=None,
    github_token: dict = None,
) -> str:
    global _tools_called_this_request, _last_tool_results

    # Set Google creds if provided
    if creds and not _current_creds_file:
        set_current_creds(creds)

    # Set GitHub creds if provided
    if github_token and not _current_github_creds_file:
        set_current_github_creds(github_token)

    # ── Parse args ────────────────────────────────────────────────────────────
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except Exception:
            tool_args = {}
    if not tool_args:
        tool_args = {}
    if not isinstance(tool_args, dict):
        tool_args = {}

    # Unwrap nested kwargs if LLM wraps them
    while "kwargs" in tool_args and isinstance(tool_args["kwargs"], dict):
        tool_args = tool_args["kwargs"]

    # Unwrap Ollama-style nested tool calls — handles all patterns
    if "parameters" in tool_args and isinstance(tool_args["parameters"], dict):
        tool_args = tool_args["parameters"]
    elif "params" in tool_args and isinstance(tool_args["params"], dict):
        tool_args = tool_args["params"]
    elif "arguments" in tool_args and isinstance(tool_args["arguments"], dict):
        tool_args = tool_args["arguments"]
    elif "args" in tool_args and isinstance(tool_args["args"], dict):
        tool_args = tool_args["args"]

    # Drop Ollama meta keys
    # NOTE: do NOT pop "name" for GitHub tools — create_repo uses "name" as a real arg
    _github_tool_names = {
        "list_repos", "create_repo", "search_repos", "list_repo_files",
        "read_file_from_repo", "list_issues", "create_issue",
        "read_issue", "list_pull_requests", "create_pull_request",
    }
    tool_args.pop("function", None)
    tool_args.pop("type", None)
    if tool_name not in _github_tool_names:
        tool_args.pop("name", None)

    # ── Clean each arg value ──────────────────────────────────────────────────
    # GitHub tools — never drop any args, even empty strings
    # because required fields like 'name' must always be preserved
    github_tool_names = {
        "list_repos", "create_repo", "search_repos", "list_repo_files",
        "read_file_from_repo", "list_issues", "create_issue",
        "read_issue", "list_pull_requests", "create_pull_request",
    }
    is_github_tool = tool_name in github_tool_names

    cleaned_args = {}
    for k, v in tool_args.items():
        if isinstance(v, str):
            v = v.strip()
            if not v and not is_github_tool:
                continue  # drop empty strings for non-GitHub tools only

            # Handle "1 (some label)" style replies from LLM
            num_match = re.match(r'^\s*(\d+)\s+\(.*\)\s*$', v)
            if num_match:
                cleaned_args[k] = num_match.group(1)
                continue

            cleaned_args[k] = v
        else:
            cleaned_args[k] = v

    tool_args = cleaned_args

    # ── Sanitize all args through central router ──────────────────────────────
    sanitized_args = {}
    for k, v in tool_args.items():
        sanitized_args[k] = _sanitize_arg(tool_name, k, v)
    tool_args = sanitized_args

    # ── Execute ───────────────────────────────────────────────────────────────
    start = time.time()
    try:
        result = asyncio.run(execute_tool_async(server_name, tool_name, tool_args))
        duration = time.time() - start
        _tools_called_this_request.append(tool_name)
        _last_tool_results[tool_name] = result  # store for smart reply formatting
        log_tool_call(tool_name, tool_args, result, duration)
        return result
    except Exception as e:
        duration = time.time() - start
        log_error(str(e), context=f"tool={tool_name} args={tool_args}")
        return f"Error executing tool {tool_name}: {str(e)}"