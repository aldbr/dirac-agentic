"""Simple MCP client agent connecting to the DiracX MCP server.

This example shows how to connect a Smolagents CodeAgent to the
dirac-mcp server via the MCP protocol over stdio.

Configuration via environment variables:
    LLM_BASE_URL:  OpenAI-compatible API base URL
                   (default: https://router.huggingface.co/v1)
    LLM_API_KEY:   API key for the LLM provider (default: $HF_TOKEN)
    LLM_MODEL:     Model name (default: Qwen/Qwen2.5-Coder-32B-Instruct)

Usage:
    pip install -r requirements.txt
    export HF_TOKEN=your_token   # or set LLM_API_KEY directly
    python diracx_agent.py
"""

import os

from mcp.client.stdio import StdioServerParameters
from smolagents import CodeAgent, OpenAIServerModel, ToolCollection

model = OpenAIServerModel(
    model_id=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    api_base=os.environ.get("LLM_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.environ.get("LLM_API_KEY", os.environ.get("HF_TOKEN", "unused")),
)

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
