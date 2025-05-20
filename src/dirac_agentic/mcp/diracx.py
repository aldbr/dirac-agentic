from typing import List, Dict, Optional, Any
from mcp.server.fastmcp import FastMCP

from diracx.client.aio import AsyncDiracClient
from diracx.core.models import ScalarSearchOperator, VectorSearchOperator

# Create an MCP server
mcp = FastMCP("DiracX Services")

# =========================================================
# Tools
# =========================================================


@mcp.tool()
async def search_jobs(
    conditions: List[Dict[str, str]],
    parameters: Optional[List[str]] = None,
    page: int = 1,
    per_page: int = 10,
) -> Dict[str, Any]:
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
async def get_job(job_id: int) -> Dict[str, Any]:
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
async def submit_job(jdl_content: str) -> Dict[str, Any]:
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
    arguments: Optional[str] = None,
    input_sandbox: Optional[List[str]] = None,
    output_sandbox: Optional[List[str]] = None,
    site: Optional[str] = None,
    memory: Optional[int] = None,
    max_cpu_time: Optional[int] = None,
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
async def get_job_status_summary() -> Dict[str, Any]:
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
            status_counts: Dict[str, int] = {}
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


# =========================================================
# Prompts
# =========================================================


@mcp.prompt()
def job_analysis_prompt(job_id: int) -> str:
    """
    Create a prompt to analyze job status and potential issues.

    Args:
        job_id: The ID of the job to analyze
    """
    return f"""You are a DIRAC job analysis assistant. Please analyze the following job with ID {job_id}.

First, retrieve the job details using the get_job tool. Look for:
1. The current Status and MinorStatus
2. Any error messages in ApplicationStatus
3. The job's execution site

Based on this information:
- If the job failed, explain the most likely cause of failure and suggest corrective actions
- If the job is running, estimate when it might complete based on submission time
- If the job is stuck or stalled, recommend troubleshooting steps

Your goal is to help the user understand what's happening with their job and what actions they should take."""


@mcp.prompt()
def job_search_prompt(query: str) -> str:
    """
    Create a prompt to help translate natural language queries into structured job searches.

    Args:
        query: Natural language query describing the jobs to find
    """
    return f"""You are a DIRAC job search assistant. Transform this natural language query into structured search parameters:

Query: "{query}"

First, identify key search parameters like:
- Job status (Running, Done, Failed, etc.)
- Time frame (yesterday, last week, etc.)
- Site information
- Job name or ID

Then, construct a valid list of search conditions in this format:
[
  {{"parameter": "Status", "operator": "eq", "value": "Failed"}},
  {{"parameter": "Site", "operator": "eq", "value": "LCG.CERN.ch"}}
]

Valid operators include:
- Scalar: eq, ne, gt, ge, lt, le, like
- Vector: in, nin

Use the search_jobs tool with your structured conditions to perform the search.
Analyze the results and summarize the key findings for the user in a clear, tabular format."""


@mcp.prompt()
def jdl_creation_prompt(user_requirements: str) -> str:
    """
    Create a prompt to help generate a JDL file from user requirements.

    Args:
        user_requirements: Natural language description of the job requirements
    """
    return f"""You are a DIRAC JDL generation assistant. Create a valid JDL file based on the following requirements:

"{user_requirements}"

The JDL should include:
1. Appropriate JobName
2. Executable and arguments
3. Input and output sandboxes
4. Any specific site requirements
5. Resource requirements (CPU, memory)
6. Any other parameters needed for the job

First, analyze the requirements and identify key parameters. Then use the create_basic_jdl tool to create a starting JDL,
and modify it as needed to fully match the requirements.

Your goal is to create a valid, optimized JDL that follows best practices for DIRAC job submission."""


# =========================================================
# Resources
# =========================================================


@mcp.resource("dirac-job://{job_id}")
async def job_resource(job_id: int) -> Dict[str, Any]:
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
async def dashboard_resource() -> Dict[str, Any]:
    """
    Provide a job dashboard overview as a resource.
    """
    return await get_job_status_summary()
