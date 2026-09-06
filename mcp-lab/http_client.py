"""
Standalone MCP client -- connects to the DEPLOYED http_server.py by URL.
=======================================================================
This is what an MCP host does internally, shown in the open so you can see the
protocol: connect, initialize, list tools, call a tool.

Start the server first (in another terminal):
    py http_server.py

Then run this:
    py http_client.py
"""

import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = "http://localhost:8000/mcp"


async def main() -> None:
    # open an HTTP connection to the deployed server
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()  # the MCP handshake

            # 1) discover what the server offers
            tools = await session.list_tools()
            print("tools on server:", [t.name for t in tools.tools])

            # 2) call a couple of them
            r1 = await session.call_tool("add", {"a": 21, "b": 21})
            print("add(21, 21)      ->", r1.content[0].text)

            r2 = await session.call_tool(
                "createTask", {"assignee": "Sagar", "taskName": "Ship MCP demo"}
            )
            print("createTask(...)  ->", r2.content[0].text)

            r3 = await session.call_tool("word_count", {"text": "one two three four"})
            print("word_count(...)  ->", r3.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
