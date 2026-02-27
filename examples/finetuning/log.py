from __future__ import annotations

import logging

from rich.logging import RichHandler

_LOG_FORMAT = "%(name)s | %(levelname)s | %(message)s"


def configure(level: str = "INFO") -> None:
    """Call once (from CLI) to prettify logs with Rich."""
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )


# Library modules just do:
logger = logging.getLogger(__name__)
