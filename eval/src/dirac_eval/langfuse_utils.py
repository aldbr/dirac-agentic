"""Conditional Langfuse integration for eval tracing and score persistence.

All functions degrade to no-ops when ``LANGFUSE_SECRET_KEY`` is not set
or when the ``langfuse`` package is not installed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Module-level cached client (avoids creating a new client per call)
_langfuse_client: Any = None


def is_langfuse_enabled() -> bool:
    """Return True if Langfuse credentials are configured."""
    return bool(os.environ.get("LANGFUSE_SECRET_KEY"))


def get_langfuse_client() -> Any:
    """Return a Langfuse client if credentials are set and package is installed, else None.

    The ``langfuse`` import is deferred so it only happens when credentials
    are configured. CI never triggers it. Returns a cached singleton.
    """
    from langfuse import Langfuse

    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if not is_langfuse_enabled():
        return None

    _langfuse_client = Langfuse()
    return _langfuse_client


@contextmanager
def langfuse_trace(name: str, metadata: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context manager that opens a Langfuse trace (or yields None).

    Degrades gracefully on errors (wrong credentials, network issues).

    Usage::

        with langfuse_trace("test_tool_call_accuracy", metadata={...}) as trace:
            # trace is an Observation or None
            ...
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    with client.start_as_current_observation(
        name=name,
        metadata=metadata or {},
    ) as trace:
        try:
            yield trace
        finally:
            client.flush()


def push_score(
    trace_id: str | None,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """Push a score to Langfuse. No-op if Langfuse is disabled or trace_id is None."""
    if trace_id is None:
        return
    client = get_langfuse_client()
    if client is None:
        return
    client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
    client.flush()
