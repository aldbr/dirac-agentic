"""Layer 1: Tool execution tests with mocked AsyncDiracClient.

Each scenario defines expected tool calls and mock responses. This test
parametrizes over every scenario YAML file, patches the DiracX client,
invokes each tool function directly, and asserts the response is successful.

No LLM required — runs in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dirac_eval.mock_client import patch_diracx_client
from dirac_eval.scenario import Scenario
from dirac_mcp.tools.jobs import (
    create_basic_jdl,
    get_job,
    get_job_metadata,
    get_job_sandboxes,
    get_job_status_summary,
    reschedule_jobs,
    search_jobs,
    set_job_statuses,
    submit_job,
)

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"

# Registry mapping tool name → callable
TOOL_REGISTRY: dict[str, Any] = {
    "search_jobs": search_jobs,
    "get_job": get_job,
    "submit_job": submit_job,
    "create_basic_jdl": create_basic_jdl,
    "get_job_status_summary": get_job_status_summary,
    "get_job_sandboxes": get_job_sandboxes,
    "set_job_statuses": set_job_statuses,
    "reschedule_jobs": reschedule_jobs,
    "get_job_metadata": get_job_metadata,
}


def _scenario_files() -> list[str]:
    return [p.name for p in sorted(SCENARIOS_DIR.glob("*.yaml"))]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", _scenario_files())
async def test_tool_execution(scenario_file: str) -> None:
    """Execute each expected tool call against the mocked client and verify success."""
    scenario = Scenario.from_yaml(SCENARIOS_DIR / scenario_file)

    previous_result: Any = None

    with patch_diracx_client(scenario):
        for tool_call in scenario.expected_tool_calls:
            tool_fn = TOOL_REGISTRY[tool_call.name]

            # Resolve sentinel: if args reference output from a previous tool
            args = dict(tool_call.args)
            for key, val in args.items():
                if val == "__FROM_PREVIOUS__" and previous_result is not None:
                    args[key] = previous_result

            result = tool_fn(**args)
            # Await if coroutine
            if hasattr(result, "__await__"):
                result = await result

            # create_basic_jdl returns a string (JDL), not a dict
            if tool_call.name == "create_basic_jdl":
                assert isinstance(result, str)
                assert "Executable" in result
                previous_result = result
            else:
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"
                assert result["success"] is tool_call.expected_success, (
                    f"Tool {tool_call.name}: expected success={tool_call.expected_success}, "
                    f"got success={result.get('success')}, error={result.get('error')}"
                )
                previous_result = result


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", _scenario_files())
async def test_tool_response_structure(scenario_file: str) -> None:
    """Verify that tool responses have the expected structure per tool type."""
    scenario = Scenario.from_yaml(SCENARIOS_DIR / scenario_file)

    with patch_diracx_client(scenario):
        for tool_call in scenario.expected_tool_calls:
            tool_fn = TOOL_REGISTRY[tool_call.name]

            args = dict(tool_call.args)
            # For chained calls, provide a minimal valid JDL
            for key, val in args.items():
                if val == "__FROM_PREVIOUS__":
                    args[key] = create_basic_jdl(executable="/bin/echo")

            result = tool_fn(**args)
            if hasattr(result, "__await__"):
                result = await result

            if tool_call.name == "create_basic_jdl":
                assert isinstance(result, str)
                continue

            assert isinstance(result, dict)

            # Error responses only have success + error keys
            if not result.get("success"):
                assert "error" in result
                continue

            # Verify expected keys per tool on success
            if tool_call.name == "search_jobs":
                assert "data" in result
                assert "content_range" in result
                assert "pagination" in result
            elif tool_call.name == "get_job":
                assert "data" in result
            elif tool_call.name == "submit_job":
                assert "job_ids" in result
                assert len(result["job_ids"]) > 0
            elif tool_call.name == "get_job_status_summary":
                assert "total_jobs" in result
                assert "status_summary" in result
            elif tool_call.name == "get_job_sandboxes":
                assert "sandboxes" in result
                assert "job_id" in result
            elif tool_call.name == "set_job_statuses":
                assert "data" in result
            elif tool_call.name == "reschedule_jobs":
                assert "data" in result
            elif tool_call.name == "get_job_metadata":
                assert "data" in result
