# dirac-mcp

MCP server exposing DiracX functionality to AI agents via the [Model Context Protocol](https://modelcontextprotocol.io/).

Built with [FastMCP 3.0](https://gofastmcp.com/). Works standalone or as a future DiracX extension.

## Quick Start

```bash
# stdio mode (for IDE integration — VS Code, Claude Desktop, etc.)
pixi run -e dirac-mcp dirac-mcp

# Streamable HTTP mode (for web deployment)
pixi run -e dirac-mcp mcp-http

# Development inspector (FastMCP dev UI)
pixi run -e dirac-mcp mcp-dev
```

## Prerequisites

Authenticate with a DiracX instance:

```bash
export DIRACX_URL=https://diracx-cert.app.cern.ch
export DIRACX_CREDENTIALS_PATH=/path/to/credentials.json
```

## Available Tools

| Tool | Description | Annotations |
|------|-------------|-------------|
| `search_jobs` | Search and filter DIRAC jobs | read-only |
| `get_job` | Get detailed job information | read-only |
| `get_job_metadata` | Get full metadata for one or more jobs | read-only |
| `get_job_sandboxes` | Get sandbox download URLs for a job | read-only |
| `get_job_status_summary` | Get job status overview (server-side aggregation) | read-only |
| `create_basic_jdl` | Generate a JDL file from parameters | read-only |
| `submit_job` | Submit a job using JDL | |
| `set_job_statuses` | Kill or delete jobs | destructive |
| `reschedule_jobs` | Reschedule failed or killed jobs | |

## Skills

Workflow guidance for agents is provided as vendor-neutral skills in `.agents/skills/`:

| Skill | Description |
|-------|-------------|
| `submit-job` | Step-by-step job submission workflow |
| `debug-job` | Job debugging and diagnosis guide |
| `search-jobs` | Job search and filtering guide |

## Architecture

```
src/dirac_mcp/
├── app.py              # Shared FastMCP instance
├── server.py           # Entry point (stdio + streamable HTTP)
├── tools/
│   └── jobs.py         # Tool implementations
└── resources/
    └── jobs.py         # Resource definitions
```

All modules register against the shared `mcp` instance in `app.py`. Transport is controlled by the `MCP_TRANSPORT` env var:

| Variable | Default | Options |
|----------|---------|---------|
| `MCP_TRANSPORT` | `stdio` | `stdio`, `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Any host (HTTP mode only) |
| `MCP_PORT` | `8080` | Any port (HTTP mode only) |

## Standalone vs DiracX Extension

This package is designed to work in two modes:

- **Standalone** (current): Runs its own transport, calls DiracX over HTTP via `AsyncDiracClient()`, authenticates via env vars. Use for development, testing, and direct IDE integration.
- **DiracX extension** (future): Mounted into DiracX via Python entrypoints, calls DiracX's service layer directly (no HTTP overhead), inherits JWT authentication.

The core value of dirac-mcp is the MCP protocol adapter (tool schemas, resources), not the business logic itself — that lives in DiracX.

See [ADR 002](../docs/adr/002-dirac-mcp.md) for the full architecture rationale.

## Integration with Claude

### Claude Code

Add a `.mcp.json` file at the root of your project:

```json
{
  "mcpServers": {
    "dirac-mcp": {
      "command": "pixi",
      "args": ["run", "-e", "dirac-mcp", "python", "-m", "dirac_mcp"],
      "env": {
        "DIRACX_URL": "https://diracx-cert.app.cern.ch",
        "DIRACX_CREDENTIALS_PATH": "/path/to/credentials.json"
      }
    }
  }
}
```

Or via CLI:

```bash
claude mcp add dirac-mcp -- pixi run -e dirac-mcp python -m dirac_mcp
```

For a remote server running streamable HTTP:

```bash
claude mcp add --transport http dirac-mcp https://your-server:8080/mcp
```

### Claude Desktop

Add to your `claude_desktop_config.json` (Claude Desktop supports stdio only):

```json
{
  "mcpServers": {
    "dirac-mcp": {
      "command": "pixi",
      "args": ["run", "-e", "dirac-mcp", "python", "-m", "dirac_mcp"],
      "env": {
        "DIRACX_URL": "https://diracx-cert.app.cern.ch",
        "DIRACX_CREDENTIALS_PATH": "/path/to/credentials.json"
      }
    }
  }
}
```

The config file is located at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

## Docker

```bash
docker build -f Dockerfile ..
docker run -p 8080:8080 \
  -e DIRACX_URL=https://diracx-cert.app.cern.ch \
  -e DIRACX_CREDENTIALS_PATH=/tmp/credentials.json \
  -v /path/to/credentials.json:/tmp/credentials.json \
  dirac-mcp:latest
```

## Development

```bash
pixi run -e dirac-mcp test-mcp      # Run tests
pixi run -e dirac-mcp mypy-mcp      # Type checking
```
