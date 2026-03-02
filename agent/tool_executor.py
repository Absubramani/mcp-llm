import asyncio
import json
import time
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.logger import log_tool_call, log_error

MCP_SERVERS = {
    "drive": Path(__file__).parent.parent / "mcp_servers" / "drive_server.py",
    "gmail": Path(__file__).parent.parent / "mcp_servers" / "gmail_server.py",
}


async def execute_tool_async(server_name: str, tool_name: str, tool_args: dict) -> str:
    server_path = MCP_SERVERS[server_name]
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
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


def execute_tool(server_name: str, tool_name: str, tool_args) -> str:
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

    # Execute and log
    start = time.time()
    try:
        result = asyncio.run(execute_tool_async(server_name, tool_name, tool_args))
        duration = time.time() - start
        log_tool_call(tool_name, tool_args, result, duration)
        return result
    except Exception as e:
        duration = time.time() - start
        log_error(str(e), context=f"tool={tool_name} args={tool_args}")
        return f"Error executing tool {tool_name}: {str(e)}"