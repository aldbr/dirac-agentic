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

Environment variables:
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

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
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

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """\
You are a routing supervisor. Your ONLY job is to delegate tasks to specialist \
agents by calling the transfer tool. NEVER answer questions yourself. NEVER \
describe what you would do. ALWAYS immediately call one of these agents:

- transfer_to_lbapi_agent: For anything about productions, samples, CI pipelines, \
  pipeline failures, MaxReset files, debugging guides, or production-level context.
- transfer_to_diracx_agent: For anything about individual DIRAC/DiracX jobs \
  (search, inspect, reschedule, sandboxes, metadata, statuses).

Workflow: delegate to lbapi_agent first for production context, then diracx_agent \
for job-level details if needed. After receiving results from agents, provide a \
clear summary to the user.

IMPORTANT: Do NOT write text before delegating. Immediately call the transfer tool.
"""

DIRACX_AGENT_PROMPT = """\
You are a DIRAC job specialist. You MUST use your tools to answer questions. \
Do NOT guess or make up information. Call the appropriate tool for every request:
- search_jobs: find jobs by status, owner, or other filters
- get_job: get details of a specific job
- get_job_sandboxes: inspect input/output sandboxes
- get_job_metadata: get job metadata
- set_job_statuses: change job statuses
- reschedule_jobs: reschedule failed jobs
Always call a tool first, then summarize the results.
"""

LBAPI_AGENT_PROMPT = """\
You are an LHCb production specialist. You MUST use your tools to answer \
questions. Do NOT guess or make up information. Call the appropriate tool for \
every request — inspect CI pipeline results, sample progress, MaxReset files, \
debugging guides, or production status. Always call a tool first, then \
summarize the results.
"""


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

    # -- Model -----------------------------------------------------------------
    kwargs: dict = {}
    cookie = os.environ.get("LLM_AUTH_COOKIE")
    if cookie:
        import httpx

        kwargs["http_client"] = httpx.Client(headers={"Cookie": f"authservice_session={cookie}"})
    model = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY or "unused",
        max_tokens=1024,
        **kwargs,
    )

    # -- MCP sessions (kept alive via AsyncExitStack) --------------------------
    async with AsyncExitStack() as stack:
        # DiracX MCP (stdio)
        parts = shlex.split(DIRACX_MCP_COMMAND)
        diracx_params = StdioServerParameters(command=parts[0], args=parts[1:])

        diracx_transport = await stack.enter_async_context(stdio_client(diracx_params))
        diracx_session = await stack.enter_async_context(ClientSession(*diracx_transport))
        await diracx_session.initialize()
        diracx_tools = await load_mcp_tools(diracx_session)

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

        # -- Agents ----------------------------------------------------------------
        diracx_agent = create_react_agent(
            model=model,
            tools=diracx_tools,
            name="diracx_agent",
            prompt=DIRACX_AGENT_PROMPT,
        )

        lbapi_agent = create_react_agent(
            model=model,
            tools=lbapi_tools,
            name="lbapi_agent",
            prompt=LBAPI_AGENT_PROMPT,
        )

        # -- Supervisor ------------------------------------------------------------
        workflow = create_supervisor(
            [diracx_agent, lbapi_agent],
            model=model,
            prompt=SUPERVISOR_PROMPT,
            output_mode="full_history",
        )
        app = workflow.compile()

        # -- Interactive loop ------------------------------------------------------
        print(
            f"Production Debugging Agent  (model: {LLM_MODEL})\nType a query or 'quit' to exit.\n"
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
