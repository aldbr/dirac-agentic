"""Tests for MCP tool functions."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dirac_mcp.tools.jobs import (
    _parse_content_range,
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

# ---------------------------------------------------------------------------
# Helpers: mock AsyncDiracClient
# ---------------------------------------------------------------------------


def _make_mock_client(jobs_ns: MagicMock) -> MagicMock:
    """Build a mock AsyncDiracClient context manager with the given jobs namespace."""
    mock_client = AsyncMock()
    mock_client.jobs = jobs_ns

    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_cls


def _search_side_effect(job_list: list[dict[str, Any]], content_range: str = "jobs 0-0/0"):
    """Return an async side_effect for client.jobs.search that honours the cls kwarg."""

    async def _search(*args: Any, **kwargs: Any) -> Any:
        cls_fn = kwargs.get("cls")
        if cls_fn:
            return cls_fn(None, job_list, {"Content-Range": content_range})
        return job_list

    return _search


def _search_raises(exc: Exception):
    """Return an async side_effect for client.jobs.search that raises."""

    async def _search(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return _search


class TestCreateBasicJdl:
    """Tests for the create_basic_jdl pure function."""

    def test_minimal_jdl(self):
        result = create_basic_jdl(executable="/bin/echo")
        assert 'Executable = "/bin/echo";' in result
        assert 'JobName = "Auto-generated Job";' in result
        assert 'StdOutput = "StdOut";' in result
        assert 'StdError = "StdErr";' in result
        assert 'OutputSandbox = {"StdOut", "StdErr"};' in result

    def test_custom_job_name(self):
        result = create_basic_jdl(executable="/bin/ls", job_name="My Job")
        assert 'JobName = "My Job";' in result

    def test_with_arguments(self):
        result = create_basic_jdl(executable="/bin/echo", arguments="hello world")
        assert 'Arguments = "hello world";' in result

    def test_without_arguments(self):
        result = create_basic_jdl(executable="/bin/echo")
        assert "Arguments" not in result

    def test_with_input_sandbox(self):
        result = create_basic_jdl(executable="/bin/run.sh", input_sandbox=["run.sh", "data.txt"])
        assert 'InputSandbox = {"run.sh", "data.txt"};' in result

    def test_with_output_sandbox(self):
        result = create_basic_jdl(executable="/bin/echo", output_sandbox=["result.txt", "StdOut"])
        assert 'OutputSandbox = {"result.txt", "StdOut"};' in result

    def test_with_site(self):
        result = create_basic_jdl(executable="/bin/echo", site="LCG.CERN.ch")
        assert 'Site = "LCG.CERN.ch";' in result

    def test_without_site(self):
        result = create_basic_jdl(executable="/bin/echo")
        assert "Site" not in result

    def test_with_memory(self):
        result = create_basic_jdl(executable="/bin/echo", memory=2048)
        assert "Memory = 2048;" in result

    def test_with_max_cpu_time(self):
        result = create_basic_jdl(executable="/bin/echo", max_cpu_time=3600)
        assert "MaxCPUTime = 3600;" in result

    def test_full_jdl(self):
        result = create_basic_jdl(
            executable="/bin/run.sh",
            job_name="Full Test Job",
            arguments="--verbose",
            input_sandbox=["run.sh"],
            output_sandbox=["output.dat", "StdOut", "StdErr"],
            site="LCG.CERN.ch",
            memory=4096,
            max_cpu_time=7200,
        )
        assert 'JobName = "Full Test Job";' in result
        assert 'Executable = "/bin/run.sh";' in result
        assert 'Arguments = "--verbose";' in result
        assert 'InputSandbox = {"run.sh"};' in result
        assert 'Site = "LCG.CERN.ch";' in result
        assert "Memory = 4096;" in result
        assert "MaxCPUTime = 7200;" in result


# ---------------------------------------------------------------------------
# Content-Range parser
# ---------------------------------------------------------------------------


class TestParseContentRange:
    def test_valid_range(self):
        result = _parse_content_range("jobs 0-9/42")
        assert result == {"start": 0, "end": 9, "total": 42, "has_more": True}

    def test_last_page(self):
        result = _parse_content_range("jobs 40-41/42")
        assert result == {"start": 40, "end": 41, "total": 42, "has_more": False}

    def test_single_item(self):
        result = _parse_content_range("jobs 0-0/1")
        assert result == {"start": 0, "end": 0, "total": 1, "has_more": False}

    def test_unparseable(self):
        result = _parse_content_range("garbage")
        assert result == {"raw": "garbage"}

    def test_empty_string(self):
        result = _parse_content_range("")
        assert result == {"raw": ""}


# ---------------------------------------------------------------------------
# SearchCondition validation
# ---------------------------------------------------------------------------


class TestSearchConditionValidation:
    def test_scalar_operator_requires_value(self):
        """Scalar operator without value should fail."""
        from dirac_mcp.tools.jobs import SearchCondition

        with pytest.raises(Exception, match="requires 'value'"):
            SearchCondition(parameter="Status", operator="eq", values=["a"])

    def test_vector_operator_requires_values(self):
        from dirac_mcp.tools.jobs import SearchCondition

        with pytest.raises(Exception, match="requires 'values'"):
            SearchCondition(parameter="JobID", operator="in", value="100")

    def test_scalar_operator_rejects_values(self):
        from dirac_mcp.tools.jobs import SearchCondition

        with pytest.raises(Exception):
            SearchCondition(parameter="Status", operator="eq", value="Failed", values=["x"])

    def test_valid_scalar(self):
        from dirac_mcp.tools.jobs import SearchCondition

        c = SearchCondition(parameter="Status", operator="eq", value="Failed")
        assert c.value == "Failed"

    def test_valid_vector(self):
        from dirac_mcp.tools.jobs import SearchCondition

        c = SearchCondition(parameter="JobID", operator="in", values=["100", "101"])
        assert c.values == ["100", "101"]


# ---------------------------------------------------------------------------
# Async tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSearchJobs:
    async def test_basic_success(self):
        jobs_ns = MagicMock()
        jobs_ns.search = AsyncMock(
            side_effect=_search_side_effect([{"JobID": 100, "Status": "Failed"}], "jobs 0-0/1")
        )
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await search_jobs(parameter="Status", operator="eq", value="Failed")

        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["pagination"]["total"] == 1

    async def test_api_error(self):
        jobs_ns = MagicMock()
        jobs_ns.search = AsyncMock(side_effect=_search_raises(RuntimeError("connection refused")))
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await search_jobs(parameter="Status", operator="eq", value="Failed")

        assert result["success"] is False
        assert "connection refused" in result["error"]

    async def test_invalid_extra_conditions_json(self):
        jobs_ns = MagicMock()
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await search_jobs(
                parameter="Status",
                operator="eq",
                value="Failed",
                extra_conditions="not valid json",
            )

        assert result["success"] is False
        assert "Invalid extra_conditions" in result["error"]

    async def test_invalid_extra_conditions_schema(self):
        jobs_ns = MagicMock()
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await search_jobs(
                parameter="Status",
                operator="eq",
                value="Failed",
                extra_conditions='[{"parameter": "Site", "operator": "eq"}]',
            )

        assert result["success"] is False
        assert "Invalid extra_conditions" in result["error"]

    async def test_mismatched_operator_value(self):
        """Scalar operator with values= instead of value= should fail validation."""
        jobs_ns = MagicMock()
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await search_jobs(
                parameter="Status", operator="eq", value=None, values=["Failed"]
            )

        assert result["success"] is False
        assert "Invalid primary condition" in result["error"]

    async def test_with_extra_conditions(self):
        jobs_ns = MagicMock()
        jobs_ns.search = AsyncMock(
            side_effect=_search_side_effect(
                [{"JobID": 100, "Status": "Failed", "Site": "LCG.CERN.ch"}],
                "jobs 0-0/1",
            )
        )
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await search_jobs(
                parameter="Status",
                operator="eq",
                value="Failed",
                extra_conditions='[{"parameter": "Site", "operator": "eq", "value": "LCG.CERN.ch"}]',
            )

        assert result["success"] is True
        assert len(result["search_specs"]) == 2


@pytest.mark.asyncio
class TestGetJob:
    async def test_found(self):
        jobs_ns = MagicMock()
        jobs_ns.search = AsyncMock(
            side_effect=_search_side_effect([{"JobID": 123, "Status": "Done"}], "jobs 0-0/1")
        )
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await get_job(job_id=123)

        assert result["success"] is True
        assert result["data"]["JobID"] == 123

    async def test_not_found(self):
        jobs_ns = MagicMock()
        jobs_ns.search = AsyncMock(side_effect=_search_side_effect([], "jobs 0-0/0"))
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await get_job(job_id=99999)

        assert result["success"] is False
        assert "not found" in result["error"]


@pytest.mark.asyncio
class TestSubmitJob:
    async def test_success(self):
        jobs_ns = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = 42

        async def _submit(*args: Any, **kwargs: Any) -> list:
            return [mock_job]

        jobs_ns.submit_jdl_jobs = AsyncMock(side_effect=_submit)
        mock_cls = _make_mock_client(jobs_ns)

        jdl = create_basic_jdl(executable="/bin/echo")
        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await submit_job(jdl_content=jdl)

        assert result["success"] is True
        assert 42 in result["job_ids"]

    async def test_api_error(self):
        jobs_ns = MagicMock()
        jobs_ns.submit_jdl_jobs = AsyncMock(side_effect=RuntimeError("quota exceeded"))
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await submit_job(jdl_content="bogus jdl")

        assert result["success"] is False
        assert "quota exceeded" in result["error"]


@pytest.mark.asyncio
class TestGetJobStatusSummary:
    async def test_success(self):
        jobs_ns = MagicMock()

        async def _summary(*args: Any, **kwargs: Any) -> list:
            return [
                {"Status": "Done", "count": 10},
                {"Status": "Failed", "count": 3},
            ]

        jobs_ns.summary = AsyncMock(side_effect=_summary)
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await get_job_status_summary()

        assert result["success"] is True
        assert result["total_jobs"] == 13
        assert result["status_summary"]["Done"] == 10
        assert result["status_summary"]["Failed"] == 3


@pytest.mark.asyncio
class TestGetJobSandboxes:
    async def test_success(self):
        jobs_ns = MagicMock()

        async def _get_sandboxes(*args: Any, **kwargs: Any) -> dict:
            return {"input": ["/sb/in.tar.gz"], "output": ["/sb/out.tar.gz"]}

        jobs_ns.get_job_sandboxes = AsyncMock(side_effect=_get_sandboxes)

        mock_file = MagicMock()
        mock_file.url = "https://example.com/dl"
        mock_file.expires_in = 3600

        async def _get_file(*args: Any, **kwargs: Any) -> Any:
            return mock_file

        jobs_ns.get_sandbox_file = AsyncMock(side_effect=_get_file)
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await get_job_sandboxes(job_id=123)

        assert result["success"] is True
        assert len(result["sandboxes"]["input"]) == 1
        assert result["sandboxes"]["input"][0]["url"] == "https://example.com/dl"

    async def test_individual_sandbox_error(self):
        jobs_ns = MagicMock()

        async def _get_sandboxes(*args: Any, **kwargs: Any) -> dict:
            return {"input": [], "output": ["/sb/out.tar.gz"]}

        jobs_ns.get_job_sandboxes = AsyncMock(side_effect=_get_sandboxes)
        jobs_ns.get_sandbox_file = AsyncMock(side_effect=RuntimeError("expired"))
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await get_job_sandboxes(job_id=123)

        assert result["success"] is True
        assert "error" in result["sandboxes"]["output"][0]


@pytest.mark.asyncio
class TestSetJobStatuses:
    async def test_success_killed(self):
        jobs_ns = MagicMock()
        mock_result = MagicMock()
        mock_result.as_dict.return_value = {"123": "Killed"}

        async def _set(*args: Any, **kwargs: Any) -> Any:
            return mock_result

        jobs_ns.set_job_statuses = AsyncMock(side_effect=_set)
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await set_job_statuses(job_ids=[123], status="Killed")

        assert result["success"] is True
        assert result["data"]["123"] == "Killed"

    async def test_invalid_status_rejected(self):
        """VALID_USER_STATUSES guard rejects statuses outside the whitelist."""
        result = await set_job_statuses(job_ids=[123], status="Running")  # type: ignore[arg-type]
        assert result["success"] is False
        assert "Invalid status" in result["error"]


@pytest.mark.asyncio
class TestRescheduleJobs:
    async def test_success(self):
        jobs_ns = MagicMock()

        async def _reschedule(*args: Any, **kwargs: Any) -> list:
            return [123, 456]

        jobs_ns.reschedule_jobs = AsyncMock(side_effect=_reschedule)
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await reschedule_jobs(job_ids=[123, 456])

        assert result["success"] is True
        assert result["data"] == [123, 456]


@pytest.mark.asyncio
class TestGetJobMetadata:
    async def test_success(self):
        jobs_ns = MagicMock()
        jobs_ns.search = AsyncMock(
            side_effect=_search_side_effect(
                [{"JobID": 100, "Status": "Done"}, {"JobID": 101, "Status": "Failed"}],
                "jobs 0-1/2",
            )
        )
        mock_cls = _make_mock_client(jobs_ns)

        with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
            result = await get_job_metadata(job_ids=[100, 101])

        assert result["success"] is True
        assert len(result["data"]) == 2
