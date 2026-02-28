"""Job management tools for the DiracX MCP server."""

from datetime import UTC, datetime
from typing import Any

from diracx.client.aio import AsyncDiracClient
from diracx.client.models import (
    BodyJobsRescheduleJobs,
    JobStatusUpdate,
)
from diracx.core.models.search import (  # type: ignore[no-redef]
    ScalarSearchOperator,
    VectorSearchOperator,
)
from mcp.types import ToolAnnotations

from dirac_mcp.app import mcp

VALID_USER_STATUSES = {"Killed", "Deleted"}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def search_jobs(
    conditions: list[dict[str, str]],
    parameters: list[str] | None = None,
    page: int = 1,
    per_page: int = 10,
) -> dict[str, Any]:
    """
    Search for jobs using the DIRACX API.

    Args:
        conditions: List of search conditions, each with format:
                   {"parameter": "Status", "operator": "eq", "value": "Failed"}
        parameters: Job attributes to return (defaults to standard set if None)
        page: Page number for pagination
        per_page: Items per page

    Returns:
        Dictionary with job search results
    """
    if parameters is None:
        parameters = [
            "JobID",
            "Status",
            "MinorStatus",
            "ApplicationStatus",
            "JobGroup",
            "Site",
            "JobName",
            "Owner",
            "LastUpdateTime",
        ]

    # Convert conditions to SearchSpec format
    search_specs = []
    for condition in conditions:
        param = condition.get("parameter")
        op = condition.get("operator")
        value = condition.get("value")
        values = condition.get("values")

        if not param or not op:
            continue

        # Handle scalar operators
        if value is not None:
            try:
                search_specs.append(
                    {
                        "parameter": param,
                        "operator": ScalarSearchOperator(op),
                        "value": value,
                    }
                )
            except ValueError:
                pass

        # Handle vector operators
        elif values is not None:
            try:
                search_specs.append(
                    {
                        "parameter": param,
                        "operator": VectorSearchOperator(op),
                        "values": values,
                    }
                )
            except ValueError:
                pass

    # Execute the search
    try:
        async with AsyncDiracClient() as client:
            jobs, content_range = await client.jobs.search(
                parameters=parameters,
                search=search_specs,  # type: ignore[arg-type]
                page=page,
                per_page=per_page,
                cls=lambda _, jobs, headers: (jobs, headers.get("Content-Range", "")),
            )

            return {
                "success": True,
                "data": jobs,
                "content_range": content_range,
                "search_specs": search_specs,
            }
    except Exception as e:
        return {"success": False, "error": str(e), "search_specs": search_specs}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_job(job_id: int) -> dict[str, Any]:
    """
    Get detailed information about a specific job.

    Args:
        job_id: The ID of the job to retrieve details for
    """
    try:
        async with AsyncDiracClient() as client:
            jobs, _ = await client.jobs.search(
                search=[
                    {
                        "parameter": "JobID",
                        "operator": ScalarSearchOperator.EQUAL,
                        "value": str(job_id),
                    }
                ],
                cls=lambda _, jobs, headers: (jobs, headers.get("Content-Range", "")),
            )

            if not jobs:
                return {"success": False, "error": f"Job {job_id} not found"}

            return {"success": True, "data": jobs[0]}  # type: ignore[index]
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    )
)
async def submit_job(jdl_content: str) -> dict[str, Any]:
    """
    Submit a new job using the provided JDL description.

    Args:
        jdl_content: The full JDL content defining the job
    """
    try:
        async with AsyncDiracClient() as client:
            jobs = await client.jobs.submit_jdl_jobs([jdl_content])
            return {
                "success": True,
                "job_ids": [job.job_id for job in jobs],
                "jdl": jdl_content,
            }
    except Exception as e:
        return {"success": False, "error": str(e), "jdl": jdl_content}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def create_basic_jdl(
    executable: str,
    job_name: str = "Auto-generated Job",
    arguments: str | None = None,
    input_sandbox: list[str] | None = None,
    output_sandbox: list[str] | None = None,
    site: str | None = None,
    memory: int | None = None,
    max_cpu_time: int | None = None,
) -> str:
    """
    Create a basic JDL file with the given parameters.

    Args:
        executable: The executable to run
        job_name: Name of the job
        arguments: Command-line arguments for the executable
        input_sandbox: List of files to include in the job
        output_sandbox: List of files to retrieve after job completion
        site: Specific site to run the job on
        memory: Memory requirement in MB
        max_cpu_time: Maximum CPU time in seconds

    Returns:
        A complete JDL string ready to submit
    """
    if output_sandbox is None:
        output_sandbox = ["StdOut", "StdErr"]

    if input_sandbox is None:
        input_sandbox = []

    jdl = [
        f'JobName = "{job_name}";',
        f'Executable = "{executable}";',
        'StdOutput = "StdOut";',
        'StdError = "StdErr";',
    ]

    # Add optional parameters
    if arguments:
        jdl.append(f'Arguments = "{arguments}";')

    if input_sandbox:
        input_sandbox_files = ", ".join(f'"{file}"' for file in input_sandbox)
        jdl.append(f"InputSandbox = {{{input_sandbox_files}}};")
    if output_sandbox:
        output_sandbox_files = ", ".join(f'"{file}"' for file in output_sandbox)
        jdl.append(f"OutputSandbox = {{{output_sandbox_files}}};")
    if site:
        jdl.append(f'Site = "{site}";')

    if memory:
        jdl.append(f"Memory = {memory};")

    if max_cpu_time:
        jdl.append(f"MaxCPUTime = {max_cpu_time};")

    return "\n".join(jdl) + "\n"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_job_status_summary() -> dict[str, Any]:
    """
    Get a summary of job statuses for the current user.

    Uses server-side aggregation for accurate counts across all jobs.
    """
    try:
        async with AsyncDiracClient() as client:
            result = await client.jobs.summary(grouping=["Status"])

            # Convert the summary result into a status_summary dict
            status_counts: dict[str, int] = {}
            total = 0
            for entry in result:
                status = entry.get("Status", "Unknown")
                count = entry.get("count", 0)
                status_counts[status] = count
                total += count

            return {
                "success": True,
                "total_jobs": total,
                "status_summary": status_counts,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_job_sandboxes(job_id: int) -> dict[str, Any]:
    """
    Get sandbox download URLs for a job.

    Retrieves input and output sandbox file references for the given job,
    then resolves each to a presigned download URL.

    Args:
        job_id: The ID of the job to get sandboxes for
    """
    try:
        async with AsyncDiracClient() as client:
            sandboxes = await client.jobs.get_job_sandboxes(job_id)

            result: dict[str, list[dict[str, Any]]] = {"input": [], "output": []}
            for sandbox_type in ("input", "output"):
                for pfn in sandboxes.get(sandbox_type, []):
                    try:
                        download_info = await client.jobs.get_sandbox_file(pfn=pfn)
                        result[sandbox_type].append(
                            {
                                "pfn": pfn,
                                "url": download_info.url,
                                "expires_in": download_info.expires_in,
                            }
                        )
                    except Exception as e:
                        result[sandbox_type].append({"pfn": pfn, "error": str(e)})

            return {"success": True, "job_id": job_id, "sandboxes": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True))
async def set_job_statuses(
    job_ids: list[int],
    status: str,
    minor_status: str | None = None,
) -> dict[str, Any]:
    """
    Set the status of one or more jobs (e.g. kill or delete).

    Only user-settable statuses are allowed: Killed, Deleted.

    Args:
        job_ids: List of job IDs to update
        status: Target status (must be 'Killed' or 'Deleted')
        minor_status: Optional minor status message
    """
    if status not in VALID_USER_STATUSES:
        return {
            "success": False,
            "error": f"Invalid status '{status}'. Allowed: {sorted(VALID_USER_STATUSES)}",
        }

    try:
        now = datetime.now(UTC).isoformat()
        body: dict[str, dict[str, JobStatusUpdate]] = {
            str(job_id): {
                now: JobStatusUpdate(
                    status=status,
                    minor_status=minor_status,
                    source="MCP",
                )
            }
            for job_id in job_ids
        }

        async with AsyncDiracClient() as client:
            result = await client.jobs.set_job_statuses(body=body, force=False)
            return {"success": True, "data": result.as_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    )
)
async def reschedule_jobs(
    job_ids: list[int],
    reset_counter: bool = False,
) -> dict[str, Any]:
    """
    Reschedule failed or killed jobs for re-execution.

    Args:
        job_ids: List of job IDs to reschedule
        reset_counter: If True, reset the reschedule counter for these jobs
    """
    try:
        async with AsyncDiracClient() as client:
            result = await client.jobs.reschedule_jobs(
                body=BodyJobsRescheduleJobs(job_ids=job_ids),
                reset_jobs=reset_counter,
            )
            return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_job_metadata(job_ids: list[int]) -> dict[str, Any]:
    """
    Get full metadata for one or more jobs.

    Retrieves all available parameters for the specified jobs.

    Args:
        job_ids: List of job IDs to get metadata for
    """
    try:
        async with AsyncDiracClient() as client:
            jobs, _ = await client.jobs.search(
                search=[
                    {
                        "parameter": "JobID",
                        "operator": VectorSearchOperator.IN,
                        "values": [str(jid) for jid in job_ids],
                    }
                ],
                cls=lambda _, jobs, headers: (jobs, headers.get("Content-Range", "")),
            )

            return {"success": True, "data": jobs}
    except Exception as e:
        return {"success": False, "error": str(e)}
