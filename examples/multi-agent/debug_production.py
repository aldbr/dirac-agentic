"""Multi-agent production debugging with DiracX + LbAPI MCP servers.

A supervisor agent routes debugging queries between two specialized agents:

- **DiracX Agent** (dirac-mcp via stdio): job-level operations — search, inspect,
  reschedule, and manage individual DIRAC jobs.
- **LbAPI Agent** (LbAPI MCP via streamable-http): production-level context —
  sample progress, MaxReset files, pipeline logs, and debugging guides.

Architecture::

    Supervisor (routes queries)
    ├── DiracX Agent  →  dirac-mcp (stdio)
    └── LbAPI Agent   →  LbAPI MCP (streamable-http)

Uses any OpenAI-compatible inference endpoint (HuggingFace by default).

Environment variables (common defaults):
    LLM_BASE_URL:        OpenAI-compatible API base URL
                         (default: https://router.huggingface.co/v1)
    LLM_API_KEY:         API key for the LLM provider (default: $HF_TOKEN)
    LLM_AUTH_COOKIE:     Session cookie for providers using cookie auth (e.g. CERN ML)
    LLM_MODEL:           Model name
                         (default: mistralai/Mistral-Small-3.1-24B-Instruct-2503)
    DIRACX_MCP_COMMAND:  Command to start dirac-mcp
                         (default: pixi run -e dirac-mcp dirac-mcp)
    LBAPI_MCP_URL:       LbAPI MCP endpoint
                         (default: https://lbap.app.cern.ch/mcp)

Per-agent overrides (fall back to the common LLM_* values above):
    SUPERVISOR_BASE_URL, SUPERVISOR_API_KEY, SUPERVISOR_MODEL, SUPERVISOR_AUTH_COOKIE
    DIRACX_AGENT_BASE_URL, DIRACX_AGENT_API_KEY, DIRACX_AGENT_MODEL, DIRACX_AGENT_AUTH_COOKIE
    LBAPI_AGENT_BASE_URL, LBAPI_AGENT_API_KEY, LBAPI_AGENT_MODEL, LBAPI_AGENT_AUTH_COOKIE

Usage:
    pip install -r requirements.txt
    export HF_TOKEN=your_token   # or set LLM_API_KEY directly
    python debug_production.py
"""

import asyncio
import os
import shlex
import webbrowser
from contextlib import AsyncExitStack
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
from urllib.parse import parse_qs, urlparse

from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://router.huggingface.co/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("HF_TOKEN", ""))
LLM_MODEL = os.environ.get("LLM_MODEL", "mistralai/Mistral-Small-3.1-24B-Instruct-2503")

DIRACX_MCP_COMMAND = os.environ.get("DIRACX_MCP_COMMAND", "pixi run -e dirac-mcp dirac-mcp")
LBAPI_MCP_URL = os.environ.get("LBAPI_MCP_URL", "https://lbap.app.cern.ch/mcp")

OAUTH_CALLBACK_PORT = 19823
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_CALLBACK_PORT}/callback"


def _make_model(prefix: str) -> ChatOpenAI:
    """Create a ChatOpenAI from ``{prefix}_*`` env vars, falling back to ``LLM_*``."""
    base_url = os.environ.get(f"{prefix}_BASE_URL", LLM_BASE_URL)
    api_key = os.environ.get(f"{prefix}_API_KEY", LLM_API_KEY)
    model = os.environ.get(f"{prefix}_MODEL", LLM_MODEL)
    cookie = os.environ.get(f"{prefix}_AUTH_COOKIE", os.environ.get("LLM_AUTH_COOKIE"))

    kwargs: dict = {}
    if cookie:
        import httpx

        kwargs["http_client"] = httpx.Client(headers={"Cookie": f"authservice_session={cookie}"})

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key or "unused",
        max_tokens=1024,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """\
You are a routing supervisor for LHCb production debugging. Your job is to \
delegate the user's request to the right specialist agent by calling a transfer tool.

Routing rules:
- CI pipeline failures, production status, sample progress, MaxReset files, \
debugging guides -> call transfer_to_lbapi_agent
- Individual DIRAC job operations (search, inspect, reschedule, sandboxes) \
-> call transfer_to_diracx_agent

When the user message contains a URL, pass the full message including the URL \
to the agent. The agent will extract parameters from it.

Start by delegating to lbapi_agent for production context. Call the transfer \
tool now.
"""

DIRACX_AGENT_PROMPT = """\
You are a DIRAC job specialist. Use your tools to answer questions.

Tool selection:
- Find jobs by status, owner, site -> search_jobs(parameter="Status", operator="eq", value="Failed")
- Get details of one job -> get_job(job_id=12345)
- Inspect input/output sandboxes -> get_job_sandboxes(job_id=12345)
- Job status overview -> get_job_status_summary()
- Submit a job -> submit_job(executable="/bin/echo", arguments="hello world")
- Kill or delete jobs -> set_job_statuses(job_ids=[12345], status="Killed")
- Reschedule failed jobs -> reschedule_jobs(job_ids=[12345])

Call the tool first, then summarize the results.
"""

LBAPI_AGENT_PROMPT = """\
You are an LHCb production specialist. Use your tools to answer questions.

When the user provides a pipeline URL like \
https://lhcb-productions.web.cern.ch/ana-prod/pipelines/?id=26806&ci_run=Lb2D0Dsp_Run3 \
extract the analysis name from the ci_run parameter: "Lb2D0Dsp_Run3". \
Then ask yourself which working group (wg) owns that analysis. \
Example: Lb2D0Dsp_Run3 belongs to wg="B2OC".

Tool selection:
- Production summary or CI pipeline status -> get_production_summary(wg=..., analysis=...)
- Sample progress -> get_sample_progress(wg=..., analysis=..., version=..., name=...)
- MaxReset files -> get_maxreset_files(wg=..., analysis=..., version=..., name=..., transformation_id=...)
- Job log URL -> get_job_log_url(job_id=...)
- Explain a sample state string -> explain_sample_state(state=...)
- Debugging guidance -> get_debugging_guide(scenario=...)

Call the tool first, then summarize the results.
"""

# Tools to keep per agent (None = keep all). Filtering reduces token usage
# so that small-context models (e.g. Qwen-14B, 16K) don't overflow.
DIRACX_TOOL_ALLOWLIST: set[str] | None = None  # dirac-mcp has few tools; keep all
LBAPI_TOOL_ALLOWLIST: set[str] = {
    "get_production_summary",
    "get_sample_progress",
    "get_maxreset_files",
    "get_job_log_url",
    "explain_sample_state",
    "get_debugging_guide",
}


# ---------------------------------------------------------------------------
# OAuth helpers for LbAPI MCP
# ---------------------------------------------------------------------------


class _InMemoryTokenStorage(TokenStorage):
    """Simple in-memory token storage for the OAuth flow."""

    def __init__(self) -> None:
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


def _create_oauth_provider(server_url: str) -> OAuthClientProvider:
    """Create an MCP OAuthClientProvider that opens the browser for login.

    Uses the MCP SDK's built-in OAuth 2.0 Authorization Code + PKCE flow:
    1. Dynamic client registration with the MCP server
    2. Opens browser for CERN SSO login
    3. Captures callback on localhost
    4. Exchanges auth code for access/refresh tokens
    """
    code_result: dict[str, str | None] = {"code": None, "state": None}
    callback_ready = Event()
    callback_received = Event()

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            code_result["code"] = qs.get("code", [None])[0]
            code_result["state"] = qs.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authorization successful! You can close this tab.</h1>")
            callback_received.set()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: HTTPServer | None = None

    def _start_server() -> None:
        nonlocal server
        server = HTTPServer(("localhost", OAUTH_CALLBACK_PORT), _CallbackHandler)
        callback_ready.set()
        server.handle_request()

    async def redirect_handler(url: str) -> None:
        """Start callback server and open browser for auth."""
        thread = Thread(target=_start_server, daemon=True)
        thread.start()
        callback_ready.wait(timeout=5)
        print("\nOpening browser for LbAPI authorization...")
        print(f"If it doesn't open, visit:\n{url}\n")
        webbrowser.open(url)

    async def callback_handler() -> tuple[str, str | None]:
        """Wait for the OAuth callback and return (code, state)."""
        callback_received.wait(timeout=300)
        if not code_result["code"]:
            raise RuntimeError("OAuth callback timed out or failed")
        return code_result["code"], code_result["state"]

    metadata = OAuthClientMetadata(
        redirect_uris=[OAUTH_REDIRECT_URI],
        client_name="dirac-agentic-multi-agent",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
        scope="openid profile email offline_access",
    )

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=_InMemoryTokenStorage(),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def amain() -> None:
    if not LLM_API_KEY:
        raise SystemExit(
            "No API key configured.\n"
            "Set LLM_API_KEY (or HF_TOKEN for HuggingFace Inference API).\n"
            "Get a HF token at https://huggingface.co/settings/tokens"
        )

    # -- Models (per-agent, falling back to common LLM_* defaults) -------------
    supervisor_model = _make_model("SUPERVISOR")
    diracx_model = _make_model("DIRACX_AGENT")
    lbapi_model = _make_model("LBAPI_AGENT")

    # -- MCP sessions (kept alive via AsyncExitStack) --------------------------
    async with AsyncExitStack() as stack:
        # DiracX MCP (stdio)
        parts = shlex.split(DIRACX_MCP_COMMAND)
        diracx_params = StdioServerParameters(command=parts[0], args=parts[1:])

        diracx_transport = await stack.enter_async_context(stdio_client(diracx_params))
        diracx_session = await stack.enter_async_context(ClientSession(*diracx_transport))
        await diracx_session.initialize()
        diracx_tools = await load_mcp_tools(diracx_session)
        if DIRACX_TOOL_ALLOWLIST is not None:
            diracx_tools = [t for t in diracx_tools if t.name in DIRACX_TOOL_ALLOWLIST]

        # LbAPI MCP (streamable-http with OAuth)
        oauth_provider = _create_oauth_provider(LBAPI_MCP_URL)
        lbapi_transport = await stack.enter_async_context(
            streamablehttp_client(LBAPI_MCP_URL, auth=oauth_provider)
        )
        lbapi_session = await stack.enter_async_context(
            ClientSession(lbapi_transport[0], lbapi_transport[1])
        )
        await lbapi_session.initialize()
        lbapi_tools = await load_mcp_tools(lbapi_session)
        if LBAPI_TOOL_ALLOWLIST is not None:
            lbapi_tools = [t for t in lbapi_tools if t.name in LBAPI_TOOL_ALLOWLIST]

        print(f"  diracx tools: {[t.name for t in diracx_tools]}")
        print(f"  lbapi tools:  {[t.name for t in lbapi_tools]}")

        # -- Agents ----------------------------------------------------------------
        diracx_agent = create_agent(
            model=diracx_model,
            tools=diracx_tools,
            name="diracx_agent",
            system_prompt=DIRACX_AGENT_PROMPT,
        )

        lbapi_agent = create_agent(
            model=lbapi_model,
            tools=lbapi_tools,
            name="lbapi_agent",
            system_prompt=LBAPI_AGENT_PROMPT,
        )

        # -- Supervisor ------------------------------------------------------------
        workflow = create_supervisor(
            [diracx_agent, lbapi_agent],
            model=supervisor_model,
            prompt=SUPERVISOR_PROMPT,
            output_mode="full_history",
        )
        app = workflow.compile()

        # -- Interactive loop ------------------------------------------------------
        print(
            f"Production Debugging Agent\n"
            f"  supervisor : {supervisor_model.model_name}\n"
            f"  diracx     : {diracx_model.model_name}\n"
            f"  lbapi      : {lbapi_model.model_name}\n"
            f"Type a query or 'quit' to exit.\n"
        )

        while True:
            query = input("You: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            result = await app.ainvoke({"messages": [{"role": "user", "content": query}]})
            # Print the final assistant message
            for msg in result["messages"]:
                if msg.type == "ai" and msg.content:
                    print(f"\nAssistant: {msg.content}\n")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
