"""Shared fixtures for the evaluation test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dirac_eval.scenario import Scenario, load_scenarios

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def scenario_ids() -> list[str]:
    """Return scenario file stems for pytest parametrize IDs."""
    return [p.stem for p in sorted(SCENARIOS_DIR.glob("*.yaml"))]


def scenario_files() -> list[Path]:
    """Return sorted list of scenario YAML paths."""
    return sorted(SCENARIOS_DIR.glob("*.yaml"))


@pytest.fixture(scope="session")
def all_scenarios() -> list[Scenario]:
    """Load every scenario once per session."""
    return load_scenarios(SCENARIOS_DIR)


@pytest.fixture
def hf_token() -> str | None:
    """Return the HuggingFace token or None if not set."""
    return os.environ.get("HF_TOKEN")
