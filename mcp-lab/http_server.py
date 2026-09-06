"""
Deployable MCP server (HTTP transport).
=======================================
The stdio server (simple_server.py) is launched as a subprocess by the host on
YOUR machine. To *deploy* a server -- run it once and let many clients connect
over the network by URL -- you switch the transport to "streamable-http".

Same tools, same @mcp.tool() decorators. The ONLY difference is how it's served.

Run it (locally or on a VM/container):
    py http_server.py
    # -> serves MCP at  http://0.0.0.0:8000/mcp

Then a client connects to that URL instead of spawning a process.
See http_client.py for a standalone client, and README.md section
"Deploying over HTTP" for Claude Code / Docker / cloud steps.
"""

import os

from mcp.server.fastmcp import FastMCP

# host/port come from env so the same file works locally and in a container.
HOST = os.environ.get("MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_PORT", "8000"))

# stateless_http=True => no per-client session state kept on the server, which
# is what you want behind a load balancer / for a simple deployment.
mcp = FastMCP("deployable-demo", host=HOST, port=PORT, stateless_http=True)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@mcp.tool()
def createTask(assignee: str, taskName: str) -> str:
    """Create a task for the given assignee and return a confirmation."""
    return f"Task '{taskName}' created and assigned to {assignee}."


@mcp.tool()
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


if __name__ == "__main__":
    print(f"Serving MCP over HTTP at http://{HOST}:{PORT}/mcp  (Ctrl+C to stop)")
    # The transport is the whole point: HTTP instead of stdio.
    mcp.run(transport="streamable-http")
