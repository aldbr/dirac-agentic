"""Pydantic models for evaluation scenario definitions and YAML loader."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ToolCallSpec(BaseModel):
    """Expected tool call in a scenario."""

    name: str
    args: dict[str, Any]


class MockResponseSpec(BaseModel):
    """Mock response configuration for a tool's underlying client method."""

    return_value: Any
    side_effect: str | None = None


class Scenario(BaseModel):
    """A single evaluation scenario loaded from YAML."""

    name: str
    skill: str
    description: str
    user_input: str
    expected_tool_calls: list[ToolCallSpec]
    expected_goal: str
    mock_responses: dict[str, MockResponseSpec]

    @classmethod
    def from_yaml(cls, path: Path) -> "Scenario":
        """Load a scenario from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_scenarios(directory: Path) -> list[Scenario]:
    """Load all scenario YAML files from a directory."""
    scenarios = []
    for path in sorted(directory.glob("*.yaml")):
        scenarios.append(Scenario.from_yaml(path))
    return scenarios
