"""Job management tools for the DiracX MCP server."""

from typing import Any

from diracx.client.aio import AsyncDiracClient

from diracx.core.models.search import ScalarSearchOperator, VectorSearchOperator  # type: ignore[no-redef]

from dirac_mcp.app import mcp


@mcp.tool()
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
                search=search_specs,
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


@mcp.tool()
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

            return {"success": True, "data": jobs[0]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
async def get_job_status_summary() -> dict[str, Any]:
    """
    Get a summary of job statuses for the current user.
    """
    try:
        async with AsyncDiracClient() as client:
            jobs, _ = await client.jobs.search(
                parameters=["JobID", "Status"],
                per_page=1000,  # Get a larger sample for summary
                cls=lambda _, jobs, headers: (jobs, headers.get("Content-Range", "")),
            )

            # Group by status
            status_counts: dict[str, int] = {}
            for job in jobs:
                status = job.get("Status", "Unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "success": True,
                "total_jobs": len(jobs),
                "status_summary": status_counts,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
