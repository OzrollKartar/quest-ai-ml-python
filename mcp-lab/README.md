# Lab 22 — MCP Servers

Two runnable [Model Context Protocol](https://modelcontextprotocol.io) servers:

| File | What it is |
|------|------------|
| `simple_server.py` | Smallest useful demo (stdio): `add`, `createTask`, `greet` tools + a `clock://now` resource |
| `quest_labs_server.py` | **Usable custom server** — exposes this repo's `Lab*.py` files as tools (`list_labs`, `read_lab`, `search_labs`) |
| `http_server.py` | **Deployable server** — same tools served over HTTP so clients connect by URL |
| `http_client.py` | Standalone client that connects to the deployed HTTP server and calls tools |

Read-along notes: [`../Lab22_MCP_Server_And_Client.py`](../Lab22_MCP_Server_And_Client.py)

## 1. Install

```bash
pip install -r requirements.txt
```

(Windows note: this repo's Python launcher is `py`. Swap `py` for `python`/`python3` on other machines.)

## 2. Poke at it without any AI — the Inspector

The fastest feedback loop. Opens a browser UI where you can list and call tools by hand:

```bash
mcp dev quest_labs_server.py
```

## 3. Use it from Claude Code

```bash
claude mcp add quest-labs -- py "C:/Users/Sagar/Desktop/Everything/quest-ai-ml-python/mcp-lab/quest_labs_server.py"
claude mcp list
```

Then in a session just ask: *"use quest-labs to search the labs for autograd"*.

Remove it later with `claude mcp remove quest-labs`.

## 4. Use it from Claude Desktop

Settings → Developer → **Edit Config**, then add (merge into any existing `mcpServers`):

```json
{
  "mcpServers": {
    "quest-labs": {
      "command": "py",
      "args": ["C:/Users/Sagar/Desktop/Everything/quest-ai-ml-python/mcp-lab/quest_labs_server.py"]
    },
    "simple-demo": {
      "command": "py",
      "args": ["C:/Users/Sagar/Desktop/Everything/quest-ai-ml-python/mcp-lab/simple_server.py"]
    }
  }
}
```

Restart Claude Desktop — the tools appear under the tools/plug icon.

## Deploying a server over HTTP

`stdio` servers run as a local subprocess — great on your own machine, but they
can't be shared. To **deploy** a server (run it once, let many clients connect by
URL), switch the transport to `streamable-http`. That's the only change;
`http_server.py` is the same tools as the stdio demo, served over HTTP.

**Run the server** (locally, on a VM, or in a container):

```bash
py http_server.py          # serves MCP at http://0.0.0.0:8000/mcp
```

**Connect a raw client** (shows the protocol in the open — connect, initialize,
list tools, call tool):

```bash
py http_client.py
# tools on server: ['add', 'createTask', 'word_count']
# add(21, 21)      -> 42.0
# createTask(...)  -> Task 'Ship MCP demo' created and assigned to Sagar.
```

**Connect Claude Code** to the deployed URL:

```bash
claude mcp add --transport http deployable-demo http://localhost:8000/mcp
```

**Connect Claude Desktop** — in `claude_desktop_config.json`:

```json
{ "mcpServers": { "deployable-demo": { "url": "http://localhost:8000/mcp" } } }
```

**Containerize it** (minimal Dockerfile):

```dockerfile
FROM python:3.13-slim
WORKDIR /app
RUN pip install "mcp[cli]"
COPY http_server.py .
EXPOSE 8000
CMD ["python", "http_server.py"]
```

```bash
docker build -t mcp-demo . && docker run -p 8000:8000 mcp-demo
```

Host/port are read from `MCP_HOST` / `MCP_PORT` env vars. For a public
deployment put it behind a reverse proxy (TLS) and add auth — the server is
stateless (`stateless_http=True`), so it scales horizontally behind a load
balancer.

## How it works (one paragraph)

The host launches the script as a subprocess and speaks **JSON-RPC over stdio**.
`FastMCP` turns your Python type hints into each tool's input schema and your
docstrings into the descriptions the model reads to decide when to call a tool.
The model never sees the JSON — it sees clean tool names and calls them; the
client handles the wire format.
