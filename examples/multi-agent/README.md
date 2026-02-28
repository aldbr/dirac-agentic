# Multi-Agent Production Debugging

A supervisor agent that coordinates two specialized agents for debugging LHCb analysis productions:

- **DiracX Agent** — connects to `dirac-mcp` (stdio) for job-level operations: search, inspect sandboxes, reschedule, change statuses.
- **LbAPI Agent** — connects to `LbAPI MCP` (streamable-http) for production-level context: sample progress, MaxReset files, pipeline logs, debugging guides.

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
export HF_TOKEN=your_huggingface_token
pixi run -e example-multi-agent debug-production

# Or with pip
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
python debug_production.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace API token |
| `MODEL_ID` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | HF model with tool-calling support |
| `DIRACX_MCP_COMMAND` | `pixi run -e dirac-mcp dirac-mcp` | Command to start dirac-mcp (stdio) |
| `LBAPI_MCP_URL` | `https://lbap.app.cern.ch/mcp` | LbAPI MCP endpoint (streamable-http) |

Other good model choices: `Qwen/Qwen2.5-32B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`.
