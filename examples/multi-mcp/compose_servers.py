"""Multi-MCP server composition example.

Demonstrates connecting a Smolagents agent to multiple MCP servers
simultaneously. The agent can use tools from all connected servers.

Usage:
    pip install -r requirements.txt
    export HF_TOKEN=your_token
    python compose_servers.py
"""

from mcp.client.stdio import StdioServerParameters
from smolagents import CodeAgent, InferenceClientModel, ToolCollection

model = InferenceClientModel()

# Define multiple MCP servers to connect to
diracx_server = StdioServerParameters(
    command="pixi",
    args=["run", "-e", "dirac-mcp", "dirac-mcp"],
)

# Example: connect to a second MCP server (e.g., a filesystem server)
# Uncomment and adjust for your setup:
# filesystem_server = StdioServerParameters(
#     command="npx",
#     args=["-y", "@anthropic-ai/mcp-filesystem", "/path/to/allowed/dir"],
# )


if __name__ == "__main__":
    # Compose tools from multiple MCP servers
    with ToolCollection.from_mcp(diracx_server, trust_remote_code=True) as diracx_tools:
        # Combine tools from all servers
        all_tools = [*diracx_tools.tools]

        # To add tools from another server:
        # with ToolCollection.from_mcp(filesystem_server, trust_remote_code=True) as fs_tools:
        #     all_tools.extend(fs_tools.tools)

        agent = CodeAgent(tools=all_tools, model=model)

        # The agent can now use tools from all connected servers
        agent.run("Search for my recent failed jobs and provide a summary of the failures.")
