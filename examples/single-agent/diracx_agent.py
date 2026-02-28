"""Simple MCP client agent connecting to the DiracX MCP server.

This example shows how to connect a Smolagents CodeAgent to the
dirac-mcp server via the MCP protocol over stdio.

Usage:
    pip install -r requirements.txt
    python diracx_agent.py
"""

from mcp.client.stdio import StdioServerParameters
from smolagents import CodeAgent, InferenceClientModel, ToolCollection

model = InferenceClientModel()

# Connect to the DiracX MCP server.
# Adjust the command/args to match your dirac-mcp installation.
server_parameters = StdioServerParameters(
    command="pixi",
    args=["run", "-e", "dirac-mcp", "dirac-mcp"],
)

if __name__ == "__main__":
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        agent = CodeAgent(tools=[*tool_collection.tools], model=model)
        agent.run("Can you submit a hello world job on diracx for me?")
