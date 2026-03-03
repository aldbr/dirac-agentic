"""Shared tool registry mapping MCP tool names to their callables."""

from __future__ import annotations

from typing import Any

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
