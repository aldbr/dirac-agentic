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

Uses open-source models via the HuggingFace Inference API.

Environment variables:
    MODEL_ID:            HF model (default: mistralai/Mistral-Small-3.1-24B-Instruct-2503)
    HF_TOKEN:            HuggingFace API token (required)
    DIRACX_MCP_COMMAND:  Command to start dirac-mcp (default: pixi run -e dirac-mcp dirac-mcp)
    LBAPI_MCP_URL:       LbAPI MCP endpoint (default: https://lbap.app.cern.ch/mcp)

Usage:
    pip install -r requirements.txt
    export HF_TOKEN=your_token
    python debug_production.py
"""

import asyncio
import os
import shlex
from contextlib import AsyncExitStack

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-Small-3.1-24B-Instruct-2503")
HF_TOKEN = os.environ.get("HF_TOKEN")

DIRACX_MCP_COMMAND = os.environ.get("DIRACX_MCP_COMMAND", "pixi run -e dirac-mcp dirac-mcp")
LBAPI_MCP_URL = os.environ.get("LBAPI_MCP_URL", "https://lbap.app.cern.ch/mcp")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """\
You are a supervisor agent for debugging LHCb analysis productions.
You coordinate two specialist agents:

- **lbapi_agent**: Knows about productions, samples, transformations, CI pipelines,
  and debugging workflows. Use it first to get production-level context (sample states,
  MaxReset files, pipeline validation reports, debugging guides).

- **diracx_agent**: Knows about individual DIRAC jobs. Use it to drill into specific
  jobs (inspect sandboxes, check metadata, reschedule, change statuses).

**Debugging workflow**: Start with lbapi_agent to understand the production context,
then use diracx_agent for job-level details when needed. Synthesize findings from
both agents into a clear summary for the user.
"""

DIRACX_AGENT_PROMPT = """\
You are a DIRAC job specialist. Use your tools to search, inspect, and manage
individual DiracX jobs. You can look up job details, sandboxes, metadata, and
change job statuses or reschedule them.
"""

LBAPI_AGENT_PROMPT = """\
You are an LHCb production specialist. Use your tools to inspect analysis
productions, sample progress, MaxReset files, CI pipeline logs, and debugging
guides. You provide the production-level context needed to diagnose issues.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def amain() -> None:
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN environment variable is required.\n"
            "Get a token at https://huggingface.co/settings/tokens"
        )

    # -- Model -----------------------------------------------------------------
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        max_new_tokens=1024,
        huggingfacehub_api_token=HF_TOKEN,
    )
    model = ChatHuggingFace(llm=llm)

    # -- MCP sessions (kept alive via AsyncExitStack) --------------------------
    async with AsyncExitStack() as stack:
        # DiracX MCP (stdio)
        parts = shlex.split(DIRACX_MCP_COMMAND)
        diracx_params = StdioServerParameters(command=parts[0], args=parts[1:])

        diracx_transport = await stack.enter_async_context(stdio_client(diracx_params))
        diracx_session = await stack.enter_async_context(ClientSession(*diracx_transport))
        await diracx_session.initialize()
        diracx_tools = await load_mcp_tools(diracx_session)

        # LbAPI MCP (streamable-http)
        lbapi_transport = await stack.enter_async_context(streamablehttp_client(LBAPI_MCP_URL))
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
        print(f"Production Debugging Agent  (model: {MODEL_ID})\nType a query or 'quit' to exit.\n")

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
