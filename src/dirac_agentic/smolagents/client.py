from smolagents import ToolCollection, CodeAgent
from mcp.client.stdio import StdioServerParameters
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()

server_parameters = StdioServerParameters(command="uv", args=["run", "server.py"])
#server_parameters = StdioServerParameters(
#    command="uv",
#    args=["--quiet", "pubmedmcp@0.1.3"],
#    env={"UV_PYTHON": "3.12", **os.environ},
#)

with ToolCollection.from_mcp(
    server_parameters, trust_remote_code=True
) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model)
    agent.run("What's the weather in Tokyo?")
