<p align="center">
  <img alt="Dirac HF Logo" src="public/dirac_agentic.png" width="300" >
</p>

# Dirac Agentic

This repository is now split into three subprojects:

- **dirac-mcp**: MCP server and related code
- **dirac-agents**: Agents and tool code
- **dirac-model**: Model finetuning tools (based on papers)
- **dirac-dataset**: Tools to load papers, code, documentation into a vector DB


```mermaid
---
config:
  layout: elk
---
flowchart TD
 subgraph Models["Models"]
        any-model["Qwen, Mistral, OpenAI, DeepSeek, ..."]
        dirac-model["**dirac-model** based on long term data - research papers"]
  end
 subgraph Agents["Agents"]
        frameworks["Frameworks - *Smolagents, LlamaIndex, LangGraph, ...*"]
        dirac-agent1["**dirac-operator** leverages Dirac, Grafana tools"]
        dirac-agent2["**dirac-developer** leverages Dirac, Github tools"]
        dirac-agent3["**dirac-doc** leverages Dirac, DuckDuckGo tools"]
  end
 subgraph MCP_Servers["MCP Servers"]
        sdk["Anthropic MCP SDK"]
        dirac-mcp["**dirac-mcp** exposes tools to interact with Dirac"]
        k8s-mcp["K8s-mcp"]
        duckduckgo-mcp["DuckDuckGo-mcp"]
  end
 subgraph Datasets["Datasets"]
        dirac-dataset["Dirac Dataset [papers & documentation]"]
  end
    dirac-model -. based on .-> any-model
    Models -- trained on/finetuned with --> Datasets
    dirac-agent1 -. based on .-> frameworks
    dirac-agent2 -. based on .-> frameworks
    dirac-agent3 -. based on .-> frameworks
    Models -- Provides domain knowledge, LLM completions --> Agents
    Agents -- Connects to, orchestrates, and calls tools/resources on --> MCP_Servers
    MCP_Servers -- provide data as resources --> Datasets
    dirac-mcp -. uses .-> dirac["DiracX Client"]
    dirac-mcp -. based on .-> sdk
    duckduckgo-mcp -. based on .-> sdk
    k8s-mcp -. based on .-> sdk
     dirac-model:::Peach
     dirac-agent1:::Peach
     dirac-agent2:::Peach
     dirac-agent3:::Peach
     dirac-mcp:::Peach
     dirac-dataset:::Peach
     dirac:::Rose
    classDef Rose stroke-width:1px, stroke-dasharray:none, stroke:#FF5978, fill:#FFDFE5, color:#8E2236
    classDef Peach stroke-width:1px, stroke-dasharray:none, stroke:#FBB35A, fill:#FFEFDB, color:#8F632D

```



## Development Setup

Each subproject has its own environment. We recommend using [pixi](https://prefix.dev/docs/pixi/) for environment management.

### 1. dirac-mcp

This subproject contains the MCP server and related code. **To interact with the MCP server, you must have a valid `diracx` access token** (usually set in your environment as `DIRACX_CREDENTIALS_PATH` or via your DiracX login). This token is required for authentication and to access the underlying DiracX instance.

#### Contribute

1. Open the `dirac-mcp` folder.
2. Run `pixi install` to create the environment.
4. Run your MCP server as needed, e.g.:

   ```bash
   pixi run dirac-mcp
   ```


#### Run in Copilot Chat

- Open the chat
- Select the `Agent` mode
- Click on `Select tools` and:
  - `Add more tools`
  - `Add MCP Server`
  - `STDIO`

- Copy the following content to `settings.json`:

  ```json
  {
    "mcp": {
      "servers": {
        "diracx": {
          "type": "stdio",
          "command": "docker",
          "args": [
            "run",
            "-i",
            "--rm",
            "-e",
            "DIRACX_URL",
            "-e",
            "DIRACX_CREDENTIALS_PATH",
            "-v",
            "/path/to/.cache/diracx/credentials.json:/tmp/credentials.json",
            "dirac-mcp:latest"
          ],
          "env": {
            "DIRACX_URL": "https://diracx-cert.app.cern.ch",
            "DIRACX_CREDENTIALS_PATH": "/tmp/credentials.json"
          }
        }
      }
    }
  }
  ```

- Log in your diracx instance
- Start chatting about diracx

**Examples:**
- I want to create a diracx job that executes `echo "hello world". Can you do it for me?
- Can you give me the latest failed jobs from diracx?

### 2. dirac-agents

This subproject is focused on building agents that interact with existing MCP servers (such as `dirac_mcp`) and other components. It provides agent logic, tool integration, and the ability to connect to and orchestrate workflows using the MCP protocol. You can use it to build advanced agents that leverage both the MCP server and additional tools or APIs.

#### Contribute

1. Open the `dirac-agents` folder.
2. Run `pixi install` to create the environment.

#### Leverage `dirac_mcp` in your agent

The `dirac_agents` code can connect to an MCP server running from `dirac_mcp` using the MCP protocol. For example, see `client.py` in `dirac_agents`:

```python
from smolagents import ToolCollection, CodeAgent, InferenceClientModel
from mcp.client.stdio import StdioServerParameters

model = InferenceClientModel()
server_parameters = StdioServerParameters(command="uv", args=["run", "server.py"])

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model)
    agent.run("Can you submit a hello world job on diracx for me?")
```

This allows you to develop and run your MCP server and agents independently, but use them together for end-to-end workflows.

### 3. dirac-model

This subproject contains code and tools for finetuning language models, with the goal of integrating general scientific knowledge (from papers and documentation) into the models. The resulting models can be used to enhance agent capabilities and provide more domain-specific knowledge. (Deployment instructions and advanced usage will be added in the future.)

1. Open the `dirac-agents` folder.
2. Run `pixi install` to create the environment.

### 4. dirac-dataset

This subproject provides tools to load research papers, documentation, and code from various sources (such as GitHub repositories and PDFs) into a vector database using LangChain. It supports downloading, parsing, and chunking documents, making them ready for retrieval-augmented generation (RAG) and other AI workflows. The resulting vector DB can be used to enhance search, question answering, and agent capabilities with domain-specific knowledge.
