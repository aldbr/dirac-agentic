"""Layer 2: RAGAS agent evaluation with LLM-driven tool calling.

An LLM agent receives the scenario's ``user_input``, has access to all
dirac-mcp tools (formatted as function-calling schemas), picks and calls
tools against the mocked DiracX client, and the resulting trace is scored
with RAGAS ``ToolCallAccuracy`` and ``AgentGoalAccuracy``.

Requires an API key: set ``LLM_API_KEY`` (or ``HF_TOKEN`` for HuggingFace).
Skipped automatically if neither is set.

Optionally sends traces and scores to Langfuse when ``LANGFUSE_SECRET_KEY``
is set. See ``dirac_eval.langfuse_utils`` for details.

Environment variables
---------------------
LLM_BASE_URL          OpenAI-compatible API base URL
                      (default: https://router.huggingface.co/v1).
LLM_API_KEY           API key for the LLM provider (default: $HF_TOKEN).
EVAL_AGENT_MODEL      Model for the agent loop (needs tool calling).
                      Default: ``Qwen/Qwen3-14B``.
EVAL_JUDGE_MODEL      Model for the RAGAS judge (needs structured output).
                      Default: same as EVAL_AGENT_MODEL.
LANGFUSE_SECRET_KEY   Langfuse secret key (optional, enables tracing).
LANGFUSE_PUBLIC_KEY   Langfuse public key (required if above is set).
LANGFUSE_BASE_URL     Langfuse server URL (default: cloud.langfuse.com).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from dirac_eval.tool_registry import TOOL_REGISTRY

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"
BASELINES_PATH = Path(__file__).resolve().parent.parent / "baselines.json"


def _load_baselines() -> dict[str, Any]:
    """Load baseline scores from baselines.json."""
    with open(BASELINES_PATH) as f:
        return json.load(f)


def _get_baseline(scenario_stem: str, metric: str) -> float:
    """Return the baseline score for a scenario/metric, with tolerance applied."""
    data = _load_baselines()
    tolerance = data.get("tolerance", 0.1)
    scenario_baselines = data.get("scenarios", {}).get(scenario_stem, {})
    baseline = scenario_baselines.get(metric, 0.5)
    return baseline - tolerance


# The agent model must support tool/function calling. The judge model needs
# structured output (instructor) and can be a different (possibly larger) model.
# Qwen3-14B handles complex nested tool schemas well at a fraction of the cost
# of 70B+ models. Available on HF Inference Providers (nscale).
DEFAULT_AGENT_MODEL = "Qwen/Qwen3-14B"

DEFAULT_BASE_URL = "https://router.huggingface.co/v1"


def _get_api_key() -> str | None:
    """Return the LLM API key, falling back to HF_TOKEN."""
    return os.environ.get("LLM_API_KEY", os.environ.get("HF_TOKEN"))


def _make_openai_client(
    base_url: str,
    api_key: str,
    *,
    async_client: bool = False,
):
    """Build an OpenAI client, injecting a session cookie when needed.

    CERN ML service uses ``authservice_session`` cookies for auth.
    Set ``LLM_AUTH_COOKIE`` to the cookie value, or leave unset for
    providers that accept a standard ``Authorization: Bearer`` header.
    """
    import httpx

    kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}

    cookie = os.environ.get("LLM_AUTH_COOKIE")
    if cookie:
        headers = {"Cookie": f"authservice_session={cookie}"}
        if async_client:
            kwargs["http_client"] = httpx.AsyncClient(headers=headers)
        else:
            kwargs["http_client"] = httpx.Client(headers=headers)

    if async_client:
        from openai import AsyncOpenAI

        return AsyncOpenAI(**kwargs)

    from openai import OpenAI

    return OpenAI(**kwargs)


# Skip the entire module if no API key is available
pytestmark = [
    pytest.mark.agent_eval,
    pytest.mark.skipif(
        not _get_api_key(),
        reason="LLM_API_KEY or HF_TOKEN not set",
    ),
]

# --- Tool schema generation from the FastMCP server ---

_tool_schemas_cache: list[dict[str, Any]] | None = None


async def _get_tool_schemas() -> list[dict[str, Any]]:
    """Generate OpenAI-compatible tool schemas from the FastMCP server.

    Converts each registered MCP tool into the OpenAI function-calling
    format so the eval agent loop stays in sync with dirac-mcp automatically.
    """
    global _tool_schemas_cache
    if _tool_schemas_cache is not None:
        return _tool_schemas_cache

    from dirac_mcp.app import mcp

    tools = await mcp.list_tools(run_middleware=False)
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        parameters = dict(tool.parameters)
        if "type" not in parameters:
            parameters["type"] = "object"
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": parameters,
                },
            }
        )

    _tool_schemas_cache = schemas
    return schemas


async def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool and return the result as a JSON string."""
    fn = TOOL_REGISTRY[name]
    result = fn(**args)
    if hasattr(result, "__await__"):
        result = await result
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)


async def _run_agent_loop(
    user_input: str,
    *,
    model: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    max_turns: int = 10,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Run a function-calling agent loop using an OpenAI-compatible API.

    Args:
        model: Model name to use for completions.
        api_key: API key for the inference provider.
        base_url: OpenAI-compatible API base URL.

    Returns:
        Tuple of (RAGAS message trace, list of actual tool calls).
    """
    from dirac_eval.langfuse_utils import get_langfuse_client
    from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    client = _make_openai_client(base_url, api_key)
    langfuse = get_langfuse_client()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a DIRAC grid computing assistant. "
                "Use the provided tools to help the user with their request. "
                "When you have completed the task, provide a summary."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    # Build RAGAS trace (excludes system message — RAGAS starts with HumanMessage)
    ragas_trace: list[Any] = [HumanMessage(content=user_input)]
    actual_tool_calls: list[dict[str, Any]] = []

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=await _get_tool_schemas(),
            tool_choice="auto",
        )

        choice = response.choices[0]
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": choice.message.content}

        # Log LLM generation to Langfuse
        if langfuse is not None:
            with langfuse.start_as_current_observation(
                name=f"turn-{turn}",
                as_type="generation",
                model=model,
                input=messages,
                output=choice.message.content or "",
            ):
                pass

        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
            messages.append(assistant_msg)

            # Build RAGAS AIMessage with tool calls
            ragas_tool_calls = []
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                ragas_tool_calls.append(ToolCall(name=name, args=args))
                actual_tool_calls.append({"name": name, "args": args})

            ragas_trace.append(
                AIMessage(
                    content=choice.message.content or "",
                    tool_calls=ragas_tool_calls,
                )
            )

            # Execute each tool call and feed results back
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_result = await _execute_tool(name, args)

                # Log tool execution to Langfuse
                if langfuse is not None:
                    with langfuse.start_as_current_observation(
                        name=f"tool:{name}",
                        as_type="span",
                        input=args,
                        output=tool_result,
                    ):
                        pass

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
                ragas_trace.append(ToolMessage(content=tool_result))
        else:
            # No tool calls — agent is done
            final_content = choice.message.content or ""
            ragas_trace.append(AIMessage(content=final_content))
            break
    else:
        # Reached max turns — add final AI message from last response
        ragas_trace.append(AIMessage(content="[max turns reached]"))

    return ragas_trace, actual_tool_calls


def _build_ragas_dataset(
    scenario: Any,
    ragas_trace: list[Any],
) -> Any:
    """Build a RAGAS MultiTurnSample evaluation dataset from the agent trace."""
    from ragas import EvaluationDataset
    from ragas.dataset_schema import MultiTurnSample
    from ragas.messages import ToolCall

    reference_tool_calls = [
        ToolCall(name=tc.name, args=tc.args) for tc in scenario.expected_tool_calls
    ]

    sample = MultiTurnSample(
        user_input=ragas_trace,
        reference=scenario.expected_goal,
        reference_tool_calls=reference_tool_calls,
    )
    return EvaluationDataset(samples=[sample])


def _scenario_files() -> list[str]:
    return [p.name for p in sorted(SCENARIOS_DIR.glob("*.yaml"))]


def _make_judge_llm(
    model: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    *,
    async_client: bool = False,
):
    """Create a RAGAS-compatible judge LLM via an OpenAI-compatible API.

    Args:
        async_client: If True, use AsyncOpenAI (required for collections metrics
                      like AgentGoalAccuracyWithReference).
    """
    from ragas.llms import llm_factory

    client = _make_openai_client(base_url, api_key, async_client=async_client)
    return llm_factory(model, client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", _scenario_files())
async def test_agent_tool_call_accuracy(scenario_file: str) -> None:
    """Score the agent's tool-calling trace with RAGAS ToolCallAccuracy."""
    from dirac_eval.langfuse_utils import langfuse_trace, push_score
    from dirac_eval.mock_client import patch_diracx_client
    from dirac_eval.scenario import Scenario
    from ragas import evaluate
    from ragas.metrics._tool_call_accuracy import ToolCallAccuracy

    api_key = _get_api_key()
    assert api_key, "LLM_API_KEY or HF_TOKEN must be set"
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    agent_model = os.environ.get("EVAL_AGENT_MODEL", DEFAULT_AGENT_MODEL)
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", agent_model)

    scenario = Scenario.from_yaml(SCENARIOS_DIR / scenario_file)

    with langfuse_trace(
        name="test_agent_tool_call_accuracy",
        metadata={
            "scenario": scenario.name,
            "skill": scenario.skill,
            "agent_model": agent_model,
            "judge_model": judge_model,
            "metric": "ToolCallAccuracy",
        },
    ) as trace:
        with patch_diracx_client(scenario):
            ragas_trace, actual_tool_calls = await _run_agent_loop(
                scenario.user_input,
                model=agent_model,
                api_key=api_key,
                base_url=base_url,
            )

        # The agent must actually call tools when the scenario expects them
        if scenario.expected_tool_calls:
            assert actual_tool_calls, (
                f"[{scenario.name}] Agent made 0 tool calls but scenario expects "
                f"{len(scenario.expected_tool_calls)}. "
                "Is the model configured for tool/function calling?"
            )

        dataset = _build_ragas_dataset(scenario, ragas_trace)
        judge_llm = _make_judge_llm(judge_model, api_key, base_url)

        result = evaluate(
            dataset=dataset,
            metrics=[ToolCallAccuracy()],
            llm=judge_llm,
        )

        scores = result["tool_call_accuracy"]  # list of per-sample scores
        score = scores[0]
        print(f"\n[{scenario.name}] ToolCallAccuracy: {score:.2f}")

        push_score(
            trace_id=trace.trace_id if trace else None,
            name="ToolCallAccuracy",
            value=score,
            comment=f"scenario={scenario.name}",
        )

    threshold = _get_baseline(scenario.name, "tool_call_accuracy")
    assert score >= threshold, (
        f"[{scenario.name}] ToolCallAccuracy {score:.2f} < baseline threshold {threshold:.2f}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", _scenario_files())
async def test_agent_goal_accuracy(scenario_file: str) -> None:
    """Score whether the agent achieved the stated goal with RAGAS AgentGoalAccuracy.

    Uses the collections API directly (metric.ascore) since
    AgentGoalAccuracyWithReference is a BaseMetric, not a legacy Metric,
    and is incompatible with ragas.evaluate().
    """
    from dirac_eval.langfuse_utils import langfuse_trace, push_score
    from dirac_eval.mock_client import patch_diracx_client
    from dirac_eval.scenario import Scenario
    from ragas.metrics.collections.agent_goal_accuracy import (
        AgentGoalAccuracyWithReference,
    )

    api_key = _get_api_key()
    assert api_key, "LLM_API_KEY or HF_TOKEN must be set"
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    agent_model = os.environ.get("EVAL_AGENT_MODEL", DEFAULT_AGENT_MODEL)
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", agent_model)

    scenario = Scenario.from_yaml(SCENARIOS_DIR / scenario_file)

    with langfuse_trace(
        name="test_agent_goal_accuracy",
        metadata={
            "scenario": scenario.name,
            "skill": scenario.skill,
            "agent_model": agent_model,
            "judge_model": judge_model,
            "metric": "AgentGoalAccuracy",
        },
    ) as trace:
        with patch_diracx_client(scenario):
            ragas_trace, actual_tool_calls = await _run_agent_loop(
                scenario.user_input,
                model=agent_model,
                api_key=api_key,
                base_url=base_url,
            )

        # The agent must actually call tools when the scenario expects them
        if scenario.expected_tool_calls:
            assert actual_tool_calls, (
                f"[{scenario.name}] Agent made 0 tool calls but scenario expects "
                f"{len(scenario.expected_tool_calls)}. "
                "Is the model configured for tool/function calling?"
            )

        judge_llm = _make_judge_llm(judge_model, api_key, base_url, async_client=True)
        metric = AgentGoalAccuracyWithReference(llm=judge_llm)

        result = await metric.ascore(ragas_trace, reference=scenario.expected_goal)
        score = float(result)
        print(f"\n[{scenario.name}] AgentGoalAccuracy: {score:.2f}")

        push_score(
            trace_id=trace.trace_id if trace else None,
            name="AgentGoalAccuracy",
            value=score,
            comment=f"scenario={scenario.name}",
        )

    threshold = _get_baseline(scenario.name, "agent_goal_accuracy")
    assert score >= threshold, (
        f"[{scenario.name}] AgentGoalAccuracy {score:.2f} < baseline threshold {threshold:.2f}"
    )
