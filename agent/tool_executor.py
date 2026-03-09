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

    # Cleanup previous temp file first
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

    # Pass creds file path as environment variable to MCP server process
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


def execute_tool(server_name: str, tool_name: str, tool_args, creds=None) -> str:
    global _tools_called_this_request

    # Set creds if provided — only set once per session, not every tool call
    if creds and not _current_creds_file:
        set_current_creds(creds)

    # Clean args
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except Exception:
            tool_args = {}
    if not tool_args:
        tool_args = {}
    if not isinstance(tool_args, dict):
        tool_args = {}
    while "kwargs" in tool_args and isinstance(tool_args["kwargs"], dict):
        tool_args = tool_args["kwargs"]

    # Clean each value safely
    cleaned_args = {}
    for k, v in tool_args.items():
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                continue
            # Extract number only if value is EXACTLY "5 (some explanation)"
            # Never truncate real IDs like "19cb259aebd5212d"
            num_match = re.match(r'^\s*(\d+)\s+\(.*\)\s*$', v)
            if num_match:
                cleaned_args[k] = num_match.group(1)
            else:
                cleaned_args[k] = v
        else:
            cleaned_args[k] = v

    tool_args = cleaned_args

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