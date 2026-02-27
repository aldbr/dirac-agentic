"""Job-related resource definitions for the DiracX MCP server."""

from typing import Any

from dirac_mcp.app import mcp
from dirac_mcp.tools.jobs import get_job, get_job_status_summary


@mcp.resource("dirac-job://{job_id}")
async def job_resource(job_id: int) -> dict[str, Any]:
    """
    Provide job information as a resource.

    Args:
        job_id: The ID of the job to retrieve
    """
    result = await get_job(job_id)
    if result.get("success") and "data" in result:
        return result["data"]
    else:
        return {"error": f"Job {job_id} not found or couldn't be retrieved"}


@mcp.resource("dirac-dashboard://jobs")
async def dashboard_resource() -> dict[str, Any]:
    """
    Provide a job dashboard overview as a resource.
    """
    return await get_job_status_summary()
