# Single-Agent Examples

Examples of single-agent workflows using [Smolagents](https://huggingface.co/docs/smolagents) with the DiracX MCP server.

## Examples

### `diracx_agent.py`
Connects a Smolagents `CodeAgent` to the DiracX MCP server via stdio. Demonstrates the basic MCP client pattern for submitting and managing DIRAC jobs.

## Setup

```bash
# With pixi (recommended)
pixi run -e example-single-agent diracx-agent

# Or with pip
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
python diracx_agent.py
```
