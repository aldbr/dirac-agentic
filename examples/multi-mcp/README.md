# Multi-MCP Composition Example

Demonstrates how to connect a single agent to multiple MCP servers simultaneously. The agent receives tools from all servers and can orchestrate cross-server workflows.

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
```

## Running

```bash
python compose_servers.py
```

## Adding More Servers

Edit `compose_servers.py` to add additional MCP servers. Each server is defined as a `StdioServerParameters` instance, and their tools are merged into a single agent.

Common MCP servers to compose with:
- **Filesystem**: `npx @anthropic-ai/mcp-filesystem /allowed/path`
- **GitHub**: `npx @anthropic-ai/mcp-github`
- **Custom**: Any MCP-compatible server via stdio or HTTP
