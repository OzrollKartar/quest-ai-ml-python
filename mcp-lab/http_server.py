import os
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
import httpx

# Initialize FastMCP server
mcp = FastMCP("GitHub Manager")

@mcp.tool()
async def search_github_repos(query: str, github_pat: str = "") -> str:
    """Search public or private GitHub repositories.
    
    Args:
        query: Search query string (e.g., 'fastapi language:python')
        github_pat: Optional GitHub Personal Access Token. Falls back to environment variable if empty.
    """
    token = github_pat or os.getenv("GITHUB_PAT", "")
    headers = {"Accept": "vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query},
            headers=headers
        )
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
            
        data = response.json()
        repos = [f"- [{repo['full_name']}]({repo['html_url']}): {repo['description']}" for repo in data.get("items", [])[:5]]
        return "\n".join(repos) if repos else "No repositories found."

# Bridge MCP SSE transport with Starlette
def create_sse_app(mcp_server: FastMCP) -> Starlette:
    transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
            await mcp_server._mcp_server.run(
                streams[0], streams[1], mcp_server._mcp_server.create_initialization_options()
            )

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=transport.handle_post_message),
        ]
    )

# Main FastAPI application for Render
app = FastAPI()
app.mount("/", create_sse_app(mcp))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))