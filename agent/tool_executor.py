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
    "drive": Path(__file__).parent.parent / "mcp_servers" / "drive_server.py",
    "gmail": Path(__file__).parent.parent / "mcp_servers" / "gmail_server.py",
}

# Track tools called in current request
_tools_called_this_request = []
_current_creds_file = None


def reset_tools_called():
    global _tools_called_this_request
    _tools_called_this_request = []


def get_tools_called():
    return list(_tools_called_this_request)


def set_current_creds(creds):
    """Save credentials to temp file for MCP server processes to use."""
    global _current_creds_file
    cleanup_creds_file()
    if creds:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pickle", prefix="mcp_creds_"
        )
        with open(tmp.name, "wb") as f:
            pickle.dump(creds, f)
        _current_creds_file = tmp.name


def cleanup_creds_file():
    """Delete temp credentials file after request."""
    global _current_creds_file
    if _current_creds_file and os.path.exists(_current_creds_file):
        try:
            os.unlink(_current_creds_file)
        except Exception:
            pass
    _current_creds_file = None


async def execute_tool_async(server_name: str, tool_name: str, tool_args: dict) -> str:
    server_path = MCP_SERVERS.get(server_name)
    if not server_path:
        return f"Error: Unknown server '{server_name}'"

    env = os.environ.copy()
    if _current_creds_file and os.path.exists(_current_creds_file):
        env["MCP_CREDS_FILE"] = _current_creds_file

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

def execute_tool(server_name: str, tool_name: str, tool_args, creds=None) -> str:
    global _tools_called_this_request

    # Set creds if provided — only once per session
    if creds and not _current_creds_file:
        set_current_creds(creds)

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
    tool_args.pop("function", None)
    tool_args.pop("name", None)
    tool_args.pop("type", None)

    # ── Clean each arg value ──────────────────────────────────────────────────
    cleaned_args = {}
    for k, v in tool_args.items():
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue  # drop empty strings

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
        log_tool_call(tool_name, tool_args, result, duration)
        return result
    except Exception as e:
        duration = time.time() - start
        log_error(str(e), context=f"tool={tool_name} args={tool_args}")
        return f"Error executing tool {tool_name}: {str(e)}"