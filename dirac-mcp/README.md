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

| Tool | Description |
|------|-------------|
| `search_jobs` | Search and filter DIRAC jobs |
| `get_job` | Get detailed job information |
| `submit_job` | Submit a job using JDL |
| `create_basic_jdl` | Generate a JDL file from parameters |
| `get_job_status_summary` | Get job status overview |

## Available Prompts

| Prompt | Description |
|--------|-------------|
| `job_analysis_prompt` | Guide for analyzing job failures |
| `job_search_prompt` | Guide for constructing job searches |
| `jdl_creation_prompt` | Guide for creating JDL files |

## Architecture

```
src/dirac_mcp/
├── app.py              # Shared FastMCP instance
├── server.py           # Entry point (stdio + streamable HTTP)
├── tools/
│   └── jobs.py         # Tool implementations
├── prompts/
│   └── jobs.py         # Prompt definitions
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

The core value of dirac-mcp is the MCP protocol adapter (tool schemas, prompts, resources), not the business logic itself — that lives in DiracX.

See [ADR 002](../docs/adr/002-dirac-mcp.md) for the full architecture rationale.

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
