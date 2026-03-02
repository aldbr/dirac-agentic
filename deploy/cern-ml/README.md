# CERN ML Platform Deployment

Deploy vLLM inference services on the CERN ML platform (Kubeflow + KServe) for use with dirac-agentic.

## Overview

This directory contains KServe `InferenceService` manifests that deploy Qwen models on CERN's GPU-enabled Kubernetes clusters via vLLM:

| Manifest | Model | Use Case | Resources |
|----------|-------|----------|-----------|
| `orchestrator.yaml` | `Qwen/Qwen2.5-72B-Instruct-AWQ` | Supervisor routing (multi-agent) | 1 GPU, 64 Gi RAM |
| `agent-14b.yaml` | `Qwen/Qwen2.5-14B-Instruct` | Tool-calling agent | 1 GPU, 48 Gi RAM |

## Prerequisites

- CERN account with access to the [ML platform](https://ml.cern.ch)
- `kubectl` configured for your Kubeflow namespace
- A GPU-enabled namespace (requires `nvidia-drivers: "true"` label)

## Deploy

```bash
kubectl apply -f deploy/cern-ml/orchestrator.yaml
kubectl apply -f deploy/cern-ml/agent-14b.yaml

# Wait for pods to become Ready
kubectl get inferenceservices -w
```

Once Ready, endpoints follow this pattern:

```
https://ml.cern.ch/serving/<namespace>/<service-name>/openai/v1
```

For example:

```
https://ml.cern.ch/serving/lhcb-dirac-dpa-mlops/lhcb-dirac-dpa-mlops-llm-orchestrator/openai/v1
https://ml.cern.ch/serving/lhcb-dirac-dpa-mlops/lhcb-dirac-dpa-mlops-llm-agent-14b/openai/v1
```

## Authentication

The CERN ML platform uses cookie-based authentication via `authservice_session`. To obtain the cookie:

1. Open the model endpoint URL in your browser (you'll be redirected to CERN SSO)
2. After login, extract the `authservice_session` cookie from your browser's developer tools
3. Pass it as `LLM_AUTH_COOKIE` when running dirac-agentic examples

## Verify

```bash
# List available models (replace with your endpoint and cookie)
curl -s \
  -H "Cookie: authservice_session=<your-cookie>" \
  https://ml.cern.ch/serving/<namespace>/<service>/openai/v1/models | python -m json.tool

# Tool-calling smoke test
curl -s \
  -H "Cookie: authservice_session=<your-cookie>" \
  -H "Content-Type: application/json" \
  https://ml.cern.ch/serving/<namespace>/<service>/openai/v1/chat/completions \
  -d '{
    "model": "agent",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "tools": [{"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}}]
  }' | python -m json.tool
```

Verify that `tool_calls` in the response is a list (not `null`). If it's `null`, the tool-calling flags are not active.

## Tool Calling

Both manifests include vLLM flags required for tool calling with Qwen models:

```yaml
- --enable-auto-tool-choice
- --tool-call-parser=hermes
```

Without these flags, vLLM returns `tool_calls: null` in chat completion responses even when tools are provided. This breaks LangGraph supervisor delegation and any agent loop that relies on tool calls. The `hermes` parser is the correct choice for Qwen2.5 models.

## Use with dirac-agentic

Point any dirac-agentic example at your CERN ML endpoints:

```bash
export LLM_BASE_URL=https://ml.cern.ch/serving/<namespace>/<service>/openai/v1
export LLM_MODEL=orchestrator   # or "agent" for the 14B model
export LLM_API_KEY=unused
export LLM_AUTH_COOKIE=<your-authservice-session-cookie>
```

See [`examples/multi-agent/README.md`](../../examples/multi-agent/README.md) for the full multi-agent setup.
