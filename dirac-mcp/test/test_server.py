"""Tests for MCP server configuration."""

import asyncio

from dirac_mcp.app import mcp


def _run(coro):
    """Helper to run async code in tests."""
    return asyncio.new_event_loop().run_until_complete(coro)


class TestServerInstantiation:
    """Tests for server setup and tool registration."""

    def test_mcp_instance_exists(self):
        assert mcp is not None
        assert mcp.name == "DiracX Services"

    def test_job_tools_registered(self):
        # Import to trigger registration
        import dirac_mcp.tools.jobs  # noqa: F401

        tools = _run(mcp.list_tools())
        tool_names = [t.name for t in tools]
        assert "search_jobs" in tool_names
        assert "get_job" in tool_names
        assert "submit_job" in tool_names
        assert "create_basic_jdl" in tool_names
        assert "get_job_status_summary" in tool_names

    def test_job_prompts_registered(self):
        import dirac_mcp.prompts.jobs  # noqa: F401

        prompts = _run(mcp.list_prompts())
        prompt_names = [p.name for p in prompts]
        assert "job_analysis_prompt" in prompt_names
        assert "job_search_prompt" in prompt_names
        assert "jdl_creation_prompt" in prompt_names
