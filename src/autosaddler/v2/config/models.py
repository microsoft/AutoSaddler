from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from autosaddler.v2.core.domain import JsonValue, to_json_value
from autosaddler.v2.prompting.models import Capability

CONFIG_SCHEMA_VERSION = "autosaddler/v2"
KNOWN_CAPABILITIES = frozenset({"read_workspace", "edit_workspace", "run_commands", "load_skills", "network"})


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    type: str
    settings: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class TaskSelectionConfig:
    type: str
    batch_size: int
    seed: int


@dataclass(frozen=True, slots=True)
class NamedPolicyConfig:
    type: str


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_rollouts: int
    max_iterations: int


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    task_selection: TaskSelectionConfig
    acceptance: NamedPolicyConfig
    development: NamedPolicyConfig
    ranking: NamedPolicyConfig
    budget: BudgetConfig
    diagnosis_patch_timeout_seconds: float
    selection_timeout_seconds: float
    reflection_timeout_seconds: float
    session_retries: int
    session_retry_backoff_seconds: float


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    type: str
    capabilities: frozenset[Capability]
    settings: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class StorageConfig:
    type: str
    run_root: Path


@dataclass(frozen=True, slots=True)
class RunConfig:
    schema_version: str
    scenario: ScenarioConfig
    optimization: OptimizationConfig
    provider: ProviderConfig
    storage: StorageConfig

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _mapping(raw, "<root>")
        _exact(root, {"schema_version", "scenario", "optimization", "provider", "storage"}, "<root>")
        schema_version = _required_string(root, "schema_version", "<root>")
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unknown configuration schema: {schema_version}")
        scenario_value = _mapping(root["scenario"], "scenario")
        _exact(scenario_value, {"type", "settings"}, "scenario")
        optimization_value = _mapping(root["optimization"], "optimization")
        required_optimization = {
            "task_selection",
            "acceptance",
            "development",
            "ranking",
            "budget",
            "diagnosis_patch_timeout_seconds",
        }
        optional_optimization = {
            "selection_timeout_seconds",
            "reflection_timeout_seconds",
            "session_retries",
            "session_retry_backoff_seconds",
        }
        _required_and_allowed(
            optimization_value,
            required_optimization,
            required_optimization | optional_optimization,
            "optimization",
        )
        provider_value = _mapping(root["provider"], "provider")
        _exact(provider_value, {"type", "capabilities", "settings"}, "provider")
        storage_value = _mapping(root["storage"], "storage")
        _exact(storage_value, {"type", "run_root"}, "storage")

        capabilities_value = provider_value["capabilities"]
        if not isinstance(capabilities_value, list) or any(not isinstance(item, str) for item in capabilities_value):
            raise TypeError("provider.capabilities must be a list of strings")
        unknown_capabilities = sorted(set(capabilities_value) - KNOWN_CAPABILITIES)
        if unknown_capabilities:
            raise ValueError(f"Unknown provider capabilities: {unknown_capabilities}")
        capabilities = frozenset(cast(Capability, item) for item in capabilities_value)
        if not capabilities:
            raise ValueError("provider.capabilities cannot be empty")

        task_selection = _mapping(optimization_value["task_selection"], "optimization.task_selection")
        _required_and_allowed(
            task_selection,
            {"type", "batch_size"},
            {"type", "batch_size", "seed"},
            "optimization.task_selection",
        )
        acceptance = _named_policy(optimization_value["acceptance"], "optimization.acceptance")
        development = _named_policy(optimization_value["development"], "optimization.development")
        ranking = _named_policy(optimization_value["ranking"], "optimization.ranking")
        budget_value = _mapping(optimization_value["budget"], "optimization.budget")
        _exact(budget_value, {"max_rollouts", "max_iterations"}, "optimization.budget")
        diagnosis_patch_timeout = _positive_number(
            optimization_value["diagnosis_patch_timeout_seconds"],
            "optimization.diagnosis_patch_timeout_seconds",
        )
        selection_timeout = _positive_number(
            optimization_value.get("selection_timeout_seconds", diagnosis_patch_timeout),
            "optimization.selection_timeout_seconds",
        )
        reflection_timeout = _positive_number(
            optimization_value.get("reflection_timeout_seconds", diagnosis_patch_timeout),
            "optimization.reflection_timeout_seconds",
        )
        session_retries = _nonnegative_int(
            optimization_value.get("session_retries", 2),
            "optimization.session_retries",
        )
        retry_backoff = _nonnegative_number(
            optimization_value.get("session_retry_backoff_seconds", 0.0),
            "optimization.session_retry_backoff_seconds",
        )
        run_root_value = _required_string(storage_value, "run_root", "storage")
        run_root = Path(run_root_value)
        if not run_root.is_absolute():
            run_root = (path.parent / run_root).resolve()
        settings = _json_mapping(scenario_value["settings"], "scenario.settings")
        provider_settings = _json_mapping(provider_value["settings"], "provider.settings")
        return cls(
            schema_version=schema_version,
            scenario=ScenarioConfig(type=_required_string(scenario_value, "type", "scenario"), settings=settings),
            optimization=OptimizationConfig(
                task_selection=TaskSelectionConfig(
                    type=_required_string(task_selection, "type", "optimization.task_selection"),
                    batch_size=_positive_int(task_selection["batch_size"], "optimization.task_selection.batch_size"),
                    seed=_nonnegative_int(task_selection.get("seed", 0), "optimization.task_selection.seed"),
                ),
                acceptance=acceptance,
                development=development,
                ranking=ranking,
                budget=BudgetConfig(
                    max_rollouts=_positive_int(budget_value["max_rollouts"], "optimization.budget.max_rollouts"),
                    max_iterations=_positive_int(budget_value["max_iterations"], "optimization.budget.max_iterations"),
                ),
                diagnosis_patch_timeout_seconds=diagnosis_patch_timeout,
                selection_timeout_seconds=selection_timeout,
                reflection_timeout_seconds=reflection_timeout,
                session_retries=session_retries,
                session_retry_backoff_seconds=retry_backoff,
            ),
            provider=ProviderConfig(
                type=_required_string(provider_value, "type", "provider"),
                capabilities=capabilities,
                settings=provider_settings,
            ),
            storage=StorageConfig(
                type=_required_string(storage_value, "type", "storage"),
                run_root=run_root,
            ),
        )

    def as_mapping(self) -> Mapping[str, JsonValue]:
        converted = to_json_value(self)
        assert isinstance(converted, dict)
        return converted


def _named_policy(value: object, path: str) -> NamedPolicyConfig:
    mapping = _mapping(value, path)
    _exact(mapping, {"type"}, path)
    return NamedPolicyConfig(type=_required_string(mapping, "type", path))


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _json_mapping(value: object, path: str) -> Mapping[str, JsonValue]:
    mapping = _mapping(value, path)
    converted = to_json_value(mapping)
    assert isinstance(converted, dict)
    return converted


def _exact(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing or extra:
        raise ValueError(f"Invalid keys at {path}: missing={missing}, extra={extra}")


def _required_and_allowed(
    value: Mapping[str, Any],
    required: set[str],
    allowed: set[str],
    path: str,
) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - allowed)
    if missing or extra:
        raise ValueError(f"Invalid keys at {path}: missing={missing}, extra={extra}")


def _required_string(value: Mapping[str, Any], key: str, path: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise TypeError(f"{path}.{key} must be a non-empty string")
    return item


def _positive_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a nonnegative integer")
    return value


def _positive_number(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be positive")
    return float(value)


def _nonnegative_number(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be nonnegative")
    return float(value)