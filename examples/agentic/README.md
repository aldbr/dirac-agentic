# Agentic Examples

Examples of agent workflows using the [Smolagents](https://huggingface.co/docs/smolagents) framework.

## Examples

### `simple_agent.py`
Connects a Smolagents `CodeAgent` to the DiracX MCP server via stdio. Demonstrates the basic MCP client pattern.

### `smolagents_example.py`
Shows custom tool creation (`suggest_menu`) combined with built-in tools (`DuckDuckGoSearchTool`). Runs multiple agent tasks sequentially.

### `multiagent_example.py`
Advanced multi-agent setup with a manager agent delegating to a specialized web search agent. Includes:
- Custom haversine-based travel time calculator
- Geographic visualization with Plotly
- Output validation using a multimodal model (gpt-4o)

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
```

## Running

```bash
python simple_agent.py
python smolagents_example.py
python multiagent_example.py
```
