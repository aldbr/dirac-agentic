"""Job management tools for the DiracX MCP server."""

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any, Literal

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
from pydantic import BaseModel, ValidationError, model_validator

from dirac_mcp.app import mcp

VALID_USER_STATUSES = {"Killed", "Deleted"}

SCALAR_OPERATORS = {"eq", "neq", "gt", "lt", "like", "not like", "regex"}
VECTOR_OPERATORS = {"in", "not in"}


logger = logging.getLogger(__name__)


class SearchCondition(BaseModel):
    """A single search filter for job queries (internal helper)."""

    parameter: str
    operator: Literal["eq", "neq", "gt", "lt", "like", "not like", "regex", "in", "not in"]
    value: str | None = None
    values: list[str] | None = None

    @model_validator(mode="after")
    def check_value_matches_operator(self) -> "SearchCondition":
        if self.operator in SCALAR_OPERATORS:
            if self.value is None:
                raise ValueError(
                    f"Scalar operator '{self.operator}' requires 'value', not 'values'"
                )
            if self.values is not None:
                raise ValueError(f"Scalar operator '{self.operator}' does not accept 'values'")
        elif self.operator in VECTOR_OPERATORS:
            if self.values is None:
                raise ValueError(
                    f"Vector operator '{self.operator}' requires 'values', not 'value'"
                )
            if self.value is not None:
                raise ValueError(f"Vector operator '{self.operator}' does not accept 'value'")
        return self


def _parse_content_range(header: str) -> dict[str, Any]:
    """Parse a Content-Range header like 'jobs 0-9/42' into structured pagination info."""
    match = re.match(r"\w+\s+(\d+)-(\d+)/(\d+)", header)
    if not match:
        return {"raw": header}
    start, end, total = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return {"start": start, "end": end, "total": total, "has_more": end < total - 1}


def _condition_to_search_spec(condition: SearchCondition) -> dict[str, Any] | None:
    """Convert a SearchCondition to a DiracX search spec dict."""
    if condition.value is not None:
        try:
            return {
                "parameter": condition.parameter,
                "operator": ScalarSearchOperator(condition.operator),
                "value": condition.value,
            }
        except ValueError:
            return None
    elif condition.values is not None:
        try:
            return {
                "parameter": condition.parameter,
                "operator": VectorSearchOperator(condition.operator),
                "values": condition.values,
            }
        except ValueError:
            return None
    return None


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def search_jobs(
    parameter: str,
    operator: Literal["eq", "neq", "gt", "lt", "like", "not like", "regex", "in", "not in"],
    value: str | None = None,
    values: list[str] | None = None,
    extra_conditions: str | None = None,
    result_fields: list[str] | None = None,
    page: int = 1,
    per_page: int = 10,
) -> dict[str, Any]:
    """Search for jobs using the DIRACX API.

    Provide a primary condition with parameter + operator + value/values.
    For additional filters, pass extra_conditions as a JSON array string.

    Examples:
      - Find failed jobs: parameter="Status", operator="eq", value="Failed"
      - Find jobs at CERN: parameter="Site", operator="eq", value="LCG.CERN.ch"
      - Find jobs by IDs: parameter="JobID", operator="in", values=["100", "101"]
      - Multi-filter: parameter="Status", operator="eq", value="Failed",
          extra_conditions='[{"parameter": "Site", "operator": "eq", "value": "LCG.CERN.ch"}]'

    Args:
        parameter: Job attribute to filter on (e.g. Status, JobID, Site, Owner, MinorStatus).
        operator: Comparison operator. Use eq/neq/gt/lt/like/regex with 'value'.
            Use 'in'/'not in' with 'values'.
        value: Single value for scalar operators (eq, neq, gt, lt, like, not like, regex).
        values: List of values for vector operators (in, not in).
        extra_conditions: Optional JSON array string with additional conditions.
            Example: '[{"parameter": "Site", "operator": "eq", "value": "LCG.CERN.ch"}]'
        result_fields: Job attributes to return (defaults to standard set if None).
        page: Page number for pagination.
        per_page: Items per page.
    """
    if result_fields is None:
        result_fields = [
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

    # Build primary condition
    try:
        primary = SearchCondition(
            parameter=parameter, operator=operator, value=value, values=values
        )
    except ValidationError as e:
        return {"success": False, "error": f"Invalid primary condition: {e}"}
    conditions = [primary]

    # Parse extra_conditions if provided
    if extra_conditions is not None:
        try:
            extra = json.loads(extra_conditions)
            for entry in extra:
                conditions.append(SearchCondition.model_validate(entry))
        except (json.JSONDecodeError, ValidationError) as e:
            return {"success": False, "error": f"Invalid extra_conditions: {e}"}

    # Convert conditions to SearchSpec format
    search_specs = []
    for condition in conditions:
        spec = _condition_to_search_spec(condition)
        if spec is not None:
            search_specs.append(spec)

    # Execute the search
    try:
        async with AsyncDiracClient() as client:
            jobs, content_range = await client.jobs.search(
                parameters=result_fields,
                search=search_specs,  # type: ignore[arg-type]
                page=page,
                per_page=per_page,
                cls=lambda _, jobs, headers: (jobs, headers.get("Content-Range", "")),
            )

            return {
                "success": True,
                "data": jobs,
                "content_range": content_range,
                "pagination": _parse_content_range(content_range),
                "search_specs": search_specs,
            }
    except Exception as e:
        logger.exception("search_jobs failed")
        return {"success": False, "error": str(e), "search_specs": search_specs}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_job(job_id: int) -> dict[str, Any]:
    """Get detailed information about a specific job.

    Example: get_job(job_id=12345)

    Args:
        job_id: The numeric ID of the job to retrieve details for.
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
        logger.exception("get_job failed for job_id=%s", job_id)
        return {"success": False, "error": str(e)}


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    )
)
async def submit_job(jdl_content: str) -> dict[str, Any]:
    """Submit a new job using the provided JDL description.

    Use create_basic_jdl first to generate a valid JDL string, then pass it here.

    Example: submit_job(jdl_content=create_basic_jdl(executable="/bin/echo", arguments="hello"))

    Args:
        jdl_content: The full JDL content defining the job.
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
        logger.exception("submit_job failed")
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
    """Create a basic JDL file with the given parameters.

    Example: create_basic_jdl(executable="/bin/echo", arguments="hello world", job_name="TestJob")

    Args:
        executable: The executable to run.
        job_name: Name of the job.
        arguments: Command-line arguments for the executable.
        input_sandbox: List of files to include in the job.
        output_sandbox: List of files to retrieve after job completion.
        site: Specific site to run the job on.
        memory: Memory requirement in MB.
        max_cpu_time: Maximum CPU time in seconds.

    Returns:
        A complete JDL string ready to pass to submit_job.
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
        logger.exception("get_job_status_summary failed")
        return {"success": False, "error": str(e)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_job_sandboxes(job_id: int) -> dict[str, Any]:
    """Get sandbox download URLs for a job.

    Retrieves input and output sandbox file references for the given job,
    then resolves each to a presigned download URL.

    Example: get_job_sandboxes(job_id=12345)

    Args:
        job_id: The numeric ID of the job to get sandboxes for.
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
                        logger.exception("get_job_sandboxes: failed to resolve pfn=%s", pfn)
                        result[sandbox_type].append({"pfn": pfn, "error": str(e)})

            return {"success": True, "job_id": job_id, "sandboxes": result}
    except Exception as e:
        logger.exception("get_job_sandboxes failed for job_id=%s", job_id)
        return {"success": False, "error": str(e)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True))
async def set_job_statuses(
    job_ids: list[int],
    status: Literal["Killed", "Deleted"],
    minor_status: str | None = None,
) -> dict[str, Any]:
    """Set the status of one or more jobs (e.g. kill or delete).

    Only user-settable statuses are allowed: Killed, Deleted.

    Example: set_job_statuses(job_ids=[123, 456], status="Killed")

    Args:
        job_ids: List of job IDs to update.
        status: Target status (Killed or Deleted).
        minor_status: Optional minor status message.
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
        logger.exception("set_job_statuses failed for job_ids=%s", job_ids)
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
    """Reschedule failed or killed jobs for re-execution.

    Example: reschedule_jobs(job_ids=[123, 456])

    Args:
        job_ids: List of job IDs to reschedule.
        reset_counter: If True, reset the reschedule counter for these jobs.
    """
    try:
        async with AsyncDiracClient() as client:
            result = await client.jobs.reschedule_jobs(
                body=BodyJobsRescheduleJobs(job_ids=job_ids),
                reset_jobs=reset_counter,
            )
            return {"success": True, "data": result}
    except Exception as e:
        logger.exception("reschedule_jobs failed for job_ids=%s", job_ids)
        return {"success": False, "error": str(e)}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_job_metadata(job_ids: list[int]) -> dict[str, Any]:
    """Get full metadata for one or more jobs.

    Retrieves all available parameters for the specified jobs.

    Example: get_job_metadata(job_ids=[123, 456])

    Args:
        job_ids: List of job IDs to get metadata for.
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
        logger.exception("get_job_metadata failed for job_ids=%s", job_ids)
        return {"success": False, "error": str(e)}
