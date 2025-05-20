<p align="center">
  <img alt="Dirac HF Logo" src="public/dirac_agentic.png" width="300" >
</p>

# Dirac Agent

This Python prototype is an attempt to produce a Dirac AI Agent.

## Installation

**Development requirements**

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- (optional) [pre-commit](https://pre-commit.com/)

1. Clone the repository
2. Install the dependencies:

  ```bash
  uv pip install -r pyproject.toml
  ```
3. (Optional) Install the pre-commit hooks:

```bash
pre-commit install
```

## Set Up

### Starting the MCP Inspector:

```bash
mcp dev src/dirac_agentic/mcp/diracx.py
```

### Interacting with diracx through Github Copilot:

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
                  "command": "uv",
                  "args": [
                      "run",
                      "--with",
                      "mcp",
                      "mcp",
                      "run",
                      "Documents/dirac-agentic/src/dirac_agentic/mcp/diracx.py"
                  ],
                  "env": {
                      "DIRACX_URL": "<diracx instance>"
                  }

              }
          }
      }
  }
  ```

- Log in your diracx instance
- Start chatting about diracx

## Usage

**Examples**

- I want to create a diracx job that executes `echo "hello world". Can you do it for me?
- Can you give me the latest failed jobs from diracx?
