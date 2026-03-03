"""Shared fixtures for the evaluation test suite."""

from __future__ import annotations

from pathlib import Path

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def scenario_ids() -> list[str]:
    """Return scenario file stems for pytest parametrize IDs."""
    return [p.stem for p in sorted(SCENARIOS_DIR.glob("*.yaml"))]


def scenario_files() -> list[Path]:
    """Return sorted list of scenario YAML paths."""
    return sorted(SCENARIOS_DIR.glob("*.yaml"))
