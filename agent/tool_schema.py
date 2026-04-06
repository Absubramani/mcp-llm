import asyncio
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVERS = {
    "drive":  Path(__file__).parent.parent / "mcp_servers" / "drive_server.py",
    "gmail":  Path(__file__).parent.parent / "mcp_servers" / "gmail_server.py",
    "github": Path(__file__).parent.parent / "mcp_servers" / "github_server.py",
}


async def get_tools_from_server(server_name: str, server_path: Path) -> list[dict]:
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()

            result = []
            for tool in tools_result.tools:
                schema = dict(tool.inputSchema)
                schema.pop("title", None)

                clean_properties = {}
                for prop_name, prop_val in schema.get("properties", {}).items():
                    clean_prop = dict(prop_val)
                    clean_prop.pop("title", None)
                    clean_prop["type"] = "string"
                    if "default" in clean_prop:
                        if clean_prop["default"] is None:
                            clean_prop["default"] = ""
                        else:
                            clean_prop["default"] = str(clean_prop["default"])
                    clean_properties[prop_name] = clean_prop

                schema["properties"] = clean_properties

                result.append({
                    "type": "function",
                    "function": {
                        "name":        tool.name,
                        "description": tool.description,
                        "parameters":  schema,
                    }
                })
            return result


async def get_all_tools() -> tuple[list[dict], dict[str, str]]:
    all_tools      = []
    tool_server_map = {}

    for server_name, server_path in MCP_SERVERS.items():
        try:
            tools = await get_tools_from_server(server_name, server_path)
            for tool in tools:
                tool_name = tool["function"]["name"]
                all_tools.append(tool)
                tool_server_map[tool_name] = server_name
        except Exception as e:
            print(f"[ERROR] Failed to load tools from {server_name}: {e}")

    return all_tools, tool_server_map


def fetch_tools() -> tuple[list[dict], dict[str, str]]:
    return asyncio.run(get_all_tools())