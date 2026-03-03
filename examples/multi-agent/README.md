# Multi-Agent Production Debugging

A supervisor agent that coordinates two specialized agents for debugging LHCb analysis productions:

- **DiracX Agent** — connects to `dirac-mcp` (stdio) for job-level operations: search, inspect sandboxes, reschedule, change statuses.
- **LbAPI Agent** — connects to `LbAPI MCP` (streamable-http) for production-level context: sample progress, MaxReset files, pipeline logs, debugging guides.

> **Note:** This example requires CERN infrastructure — LbAPI uses CERN SSO for authentication, and the multi-model setup is designed for the CERN ML platform.

## Architecture

```
Supervisor (routes queries)
├── DiracX Agent  →  dirac-mcp (stdio)
│   └── search_jobs, get_job, get_job_sandboxes, get_job_metadata,
│       set_job_statuses, reschedule_jobs, ...
└── LbAPI Agent   →  LbAPI MCP (streamable-http)
    └── get_production_summary, get_sample_progress, get_maxreset_files,
        get_job_log_url, explain_sample_state, get_debugging_guide, ...
```

The supervisor starts with LbAPI to get production context, then drills into DiracX for job-level details.

## Setup

```bash
# With pixi (recommended)
export LLM_API_KEY=your_api_key   # or set HF_TOKEN for HuggingFace Inference API
pixi run -e example-multi-agent debug-production

# Or with pip
pip install -r requirements.txt
export LLM_API_KEY=your_api_key
python debug_production.py
```

## Configuration

### Common defaults

These apply to all agents unless overridden with per-agent variables (see below):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible API base URL |
| `LLM_API_KEY` | `$HF_TOKEN` | API key for the LLM provider |
| `LLM_MODEL` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Model with tool-calling support |
| `LLM_AUTH_COOKIE` | *(unset)* | Session cookie for cookie-based auth (e.g. CERN ML `authservice_session`) |
| `DIRACX_MCP_COMMAND` | `pixi run -e dirac-mcp dirac-mcp` | Command to start dirac-mcp (stdio) |
| `LBAPI_MCP_URL` | `https://lbap.app.cern.ch/mcp` | LbAPI MCP endpoint (streamable-http) |

Other good model choices: `Qwen/Qwen2.5-32B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`.

### Per-agent model overrides

Each agent (supervisor, DiracX, LbAPI) can use a different model and endpoint. Set `{PREFIX}_BASE_URL`, `{PREFIX}_API_KEY`, `{PREFIX}_MODEL`, or `{PREFIX}_AUTH_COOKIE` to override the common `LLM_*` defaults:

| Prefix | Agent |
|--------|-------|
| `SUPERVISOR_` | Supervisor (routing) |
| `DIRACX_AGENT_` | DiracX job specialist |
| `LBAPI_AGENT_` | LbAPI production specialist |

Example — large model for the supervisor, smaller model for the sub-agents:

```bash
# Common defaults (used by DiracX and LbAPI agents)
export LLM_BASE_URL=https://router.huggingface.co/v1
export LLM_API_KEY=$HF_TOKEN
export LLM_MODEL=mistralai/Mistral-Small-3.1-24B-Instruct-2503

# Supervisor: use a larger model from a different endpoint
export SUPERVISOR_BASE_URL=https://api.openai.com/v1
export SUPERVISOR_API_KEY=sk-...
export SUPERVISOR_MODEL=gpt-4o
```

## CERN ML Deployment

To run against models deployed on the CERN ML platform (see [`deploy/cern-ml/`](../../deploy/cern-ml/README.md)):

```bash
# Common CERN ML settings (shared cookie, API key unused)
export LLM_API_KEY=unused
export LLM_AUTH_COOKIE=<your-authservice-session-cookie>

# Supervisor: large model for routing
export SUPERVISOR_BASE_URL=https://ml.cern.ch/serving/<namespace>/<large-service>/openai/v1
export SUPERVISOR_MODEL=qwen-72b

# Sub-agents: smaller model for tool-calling
export LLM_BASE_URL=https://ml.cern.ch/serving/<namespace>/<small-service>/openai/v1
export LLM_MODEL=qwen-14b

# DiracX MCP (requires DIRACX_URL and DIRACX_CREDENTIALS_PATH)
export DIRACX_MCP_COMMAND="pixi run -e dirac-mcp dirac-mcp"

# LbAPI MCP (OAuth via CERN SSO — browser will open automatically)
export LBAPI_MCP_URL=https://lbap.app.cern.ch/mcp

pixi run -e example-multi-agent debug-production
```

The CERN ML models must have vLLM tool-calling enabled (`--enable-auto-tool-choice --tool-call-parser=hermes`), otherwise the supervisor cannot delegate to sub-agents. See the [deployment guide](../../deploy/cern-ml/README.md) for details.
