from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from autosaddler.v2.core.ports import ScenarioComponents

SCENARIO_PLUGIN_API_VERSION = "1"
SCENARIO_PLUGIN_ENTRY_POINT_GROUP = "autosaddler.scenarios"
_SCENARIO_NAME = re.compile(r"[a-z][a-z0-9_]*")

ScenarioFactory: TypeAlias = Callable[..., ScenarioComponents]


@dataclass(frozen=True, slots=True)
class ScenarioPlugin:
    name: str
    api_version: str
    factory: ScenarioFactory

    def __post_init__(self) -> None:
        if _SCENARIO_NAME.fullmatch(self.name) is None:
            raise ValueError("Scenario plugin name must match [a-z][a-z0-9_]*")
        if not self.api_version:
            raise ValueError("Scenario plugin API version must be non-empty")
        if not callable(self.factory):
            raise TypeError("Scenario plugin factory must be callable")


def validate_scenario_plugin(value: object, *, entry_point_name: str) -> ScenarioPlugin:
    if not isinstance(value, ScenarioPlugin):
        raise TypeError(
            f"Scenario entry point {entry_point_name!r} must load a ScenarioPlugin"
        )
    if value.name != entry_point_name:
        raise ValueError(
            f"Scenario plugin name {value.name!r} does not match entry point name {entry_point_name!r}"
        )
    if value.api_version != SCENARIO_PLUGIN_API_VERSION:
        raise ValueError(
            f"Scenario plugin {value.name!r} uses unsupported API version {value.api_version!r}; "
            f"expected {SCENARIO_PLUGIN_API_VERSION!r}"
        )
    return value