"""AsyncDiracClient mock factory for evaluation scenarios.

Patches ``dirac_mcp.tools.jobs.AsyncDiracClient`` so that every tool
function runs against scripted return values instead of a live DiracX
instance.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from dirac_eval.scenario import MockResponseSpec, Scenario


def _make_search_response(spec: MockResponseSpec, kwargs: dict[str, Any]) -> Any:
    """Return mock search data, applying the ``cls`` transform if present."""
    if spec.side_effect:
        raise RuntimeError(spec.side_effect)
    data = spec.return_value
    cls_fn = kwargs.get("cls")
    job_list = data.get("jobs", [])
    content_range = data.get("content_range", "")
    if cls_fn:
        return cls_fn(None, job_list, {"Content-Range": content_range})
    return job_list


def _build_mock_jobs(mock_responses: dict[str, MockResponseSpec]) -> MagicMock:
    """Build a mock ``client.jobs`` namespace from scenario responses."""
    jobs = MagicMock()

    # --- search (used by search_jobs, get_job, get_job_metadata) ---
    # Build a single dispatch function that inspects the ``search=`` kwarg
    # to route to the correct mock data.  This avoids the previous bug where
    # each block overwrote ``jobs.search`` and only the last one won.
    search_specs = {
        k: mock_responses[k]
        for k in ("search_jobs", "get_job", "get_job_metadata")
        if k in mock_responses
    }

    if search_specs:

        async def _search_dispatch(*args: Any, **kwargs: Any) -> Any:
            search_arg = kwargs.get("search", [])

            # Detect get_job: single condition, JobID + scalar operator
            if (
                "get_job" in search_specs
                and len(search_arg) == 1
                and search_arg[0].get("parameter") == "JobID"
                and hasattr(search_arg[0].get("operator"), "value")
                and search_arg[0]["operator"].value == "eq"
            ):
                return _make_search_response(search_specs["get_job"], kwargs)

            # Detect get_job_metadata: single condition, JobID + vector IN operator
            if (
                "get_job_metadata" in search_specs
                and len(search_arg) == 1
                and search_arg[0].get("parameter") == "JobID"
                and hasattr(search_arg[0].get("operator"), "value")
                and search_arg[0]["operator"].value == "in"
            ):
                return _make_search_response(search_specs["get_job_metadata"], kwargs)

            # Default: search_jobs mock (general search)
            if "search_jobs" in search_specs:
                return _make_search_response(search_specs["search_jobs"], kwargs)

            # Fallback: use whichever single mock is defined
            fallback = next(iter(search_specs.values()))
            return _make_search_response(fallback, kwargs)

        jobs.search = AsyncMock(side_effect=_search_dispatch)
    else:
        jobs.search = AsyncMock(return_value=[])

    # --- submit_jdl_jobs (used by submit_job) ---
    if "submit_job" in mock_responses:
        spec = mock_responses["submit_job"]
        submit_data = spec.return_value

        async def _submit(*args: Any, **kwargs: Any) -> Any:
            if spec.side_effect:
                raise RuntimeError(spec.side_effect)
            # Return objects with a .job_id attribute
            result = []
            for jid in submit_data.get("job_ids", []):
                obj = MagicMock()
                obj.job_id = jid
                result.append(obj)
            return result

        jobs.submit_jdl_jobs = AsyncMock(side_effect=_submit)
    else:
        jobs.submit_jdl_jobs = AsyncMock(return_value=[])

    # --- summary (used by get_job_status_summary) ---
    if "get_job_status_summary" in mock_responses:
        spec = mock_responses["get_job_status_summary"]
        summary_data = spec.return_value

        async def _summary(*args: Any, **kwargs: Any) -> Any:
            if spec.side_effect:
                raise RuntimeError(spec.side_effect)
            return summary_data

        jobs.summary = AsyncMock(side_effect=_summary)
    else:
        jobs.summary = AsyncMock(return_value=[])

    # --- get_job_sandboxes + get_sandbox_file (used by get_job_sandboxes) ---
    if "get_job_sandboxes" in mock_responses:
        spec = mock_responses["get_job_sandboxes"]
        sandbox_data = spec.return_value

        async def _get_sandboxes(*args: Any, **kwargs: Any) -> Any:
            if spec.side_effect:
                raise RuntimeError(spec.side_effect)
            return sandbox_data.get("sandboxes", {})

        jobs.get_job_sandboxes = AsyncMock(side_effect=_get_sandboxes)

        # Build sandbox file mock
        sandbox_files = sandbox_data.get("sandbox_files", {})

        async def _get_sandbox_file(*args: Any, **kwargs: Any) -> Any:
            pfn = kwargs.get("pfn", args[0] if args else "")
            file_info = sandbox_files.get(pfn, {"url": f"https://mock/{pfn}", "expires_in": 3600})
            obj = MagicMock()
            obj.url = file_info["url"]
            obj.expires_in = file_info["expires_in"]
            return obj

        jobs.get_sandbox_file = AsyncMock(side_effect=_get_sandbox_file)
    else:
        jobs.get_job_sandboxes = AsyncMock(return_value={})
        jobs.get_sandbox_file = AsyncMock(return_value=MagicMock(url="", expires_in=0))

    # --- set_job_statuses (used by set_job_statuses) ---
    if "set_job_statuses" in mock_responses:
        spec = mock_responses["set_job_statuses"]
        status_data = spec.return_value

        async def _set_statuses(*args: Any, **kwargs: Any) -> Any:
            if spec.side_effect:
                raise RuntimeError(spec.side_effect)
            obj = MagicMock()
            obj.as_dict.return_value = status_data
            return obj

        jobs.set_job_statuses = AsyncMock(side_effect=_set_statuses)
    else:
        empty = MagicMock()
        empty.as_dict.return_value = {}
        jobs.set_job_statuses = AsyncMock(return_value=empty)

    # --- reschedule_jobs (used by reschedule_jobs) ---
    if "reschedule_jobs" in mock_responses:
        spec = mock_responses["reschedule_jobs"]
        resched_data = spec.return_value

        async def _reschedule(*args: Any, **kwargs: Any) -> Any:
            if spec.side_effect:
                raise RuntimeError(spec.side_effect)
            return resched_data

        jobs.reschedule_jobs = AsyncMock(side_effect=_reschedule)
    else:
        jobs.reschedule_jobs = AsyncMock(return_value=[])

    return jobs


@contextmanager
def patch_diracx_client(scenario: Scenario):
    """Context manager that patches AsyncDiracClient for a given scenario.

    Usage::

        with patch_diracx_client(scenario):
            result = await search_jobs(...)
    """
    mock_jobs = _build_mock_jobs(scenario.mock_responses)

    mock_client = AsyncMock()
    mock_client.jobs = mock_jobs

    # AsyncDiracClient is used as `async with AsyncDiracClient() as client:`
    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("dirac_mcp.tools.jobs.AsyncDiracClient", mock_cls):
        yield mock_client
