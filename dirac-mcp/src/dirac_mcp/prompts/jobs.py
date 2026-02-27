"""Job-related prompt definitions for the DiracX MCP server."""

from dirac_mcp.app import mcp


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
