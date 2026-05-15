from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import yaml

from autosaddler.v2.config.models import RunConfig
from autosaddler.v2.config import registry as registry_module
from autosaddler.v2.config.registry import build_runtime, default_registry
from autosaddler.v2.plugins.api import (
    SCENARIO_PLUGIN_API_VERSION,
    SCENARIO_PLUGIN_ENTRY_POINT_GROUP,
    ScenarioPlugin,
)


class FixtureDistribution:
    def __init__(self, name: str = "fixture-scenarios", distribution_version: str = "1.2.3") -> None:
        self.metadata = {"Name": name}
        self.version = distribution_version


class FixtureEntryPoint:
    group = SCENARIO_PLUGIN_ENTRY_POINT_GROUP

    def __init__(
        self,
        name: str,
        loaded: object,
        *,
        value: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.value = value or f"fixture_plugins:{name}"
        self.dist = FixtureDistribution()
        self._loaded = loaded
        self._error = error

    def load(self) -> object:
        if self._error is not None:
            raise self._error
        return self._loaded


def external_fake_factory(**kwargs: Any):
    return replace(registry_module._fake_scenario(**kwargs), name="external_fake")


def config_value(run_root: Path) -> dict:
    return {
        "schema_version": "autosaddler/v2",
        "scenario": {
            "type": "fake",
            "settings": {
                "baseline": {"instruction": "baseline"},
                "target_component": "instruction",
                "improved_text": "improved",
                "train_case_ids": ["train-a", "train-b"],
                "development_case_ids": ["dev-a", "dev-b"],
            },
        },
        "optimization": {
            "task_selection": {"type": "fixed", "batch_size": 2},
            "acceptance": {"type": "matched_valid_strict_improvement"},
            "development": {"type": "full_on_accept"},
            "ranking": {"type": "mean_development_score"},
            "budget": {"max_rollouts": 100, "max_iterations": 1},
            "diagnosis_patch_timeout_seconds": 10,
        },
        "provider": {
            "type": "fake",
            "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
            "settings": {},
        },
        "storage": {"type": "local", "run_root": str(run_root)},
    }


def write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_registry_builds_complete_fake_runtime_and_artifact_layout(tmp_path: Path) -> None:
    runtime = build_runtime(write_config(tmp_path, config_value(tmp_path / "runs")), run_id="registry-run")
    result = runtime.engine.run()
    run_dir = runtime.store.run_dir

    assert result.development_score == 1.0
    for relative in (
        "resolved_config.yaml",
        "resolved/component_graph.json",
        "resolved/scenario_runtime.json",
        "resolved/provider_runtime.json",
        "resolved/policies.json",
        "resolved/sources/harness.json",
        "resolved/sources/dataset.json",
        "resolved/prompts/evolve.md",
        "resolved/prompts/diagnosis_patch.md",
        "resolved/prompts/reflect.md",
        "resolved/schemas/observations.json",
        "resolved/schemas/session_outputs.json",
        "manifest.json",
        "events.jsonl",
        "snapshot.json",
        "evolution_dag.json",
        "strategy/lessons.json",
        "result.json",
    ):
        assert (run_dir / relative).is_file(), relative
    assert list((run_dir / "quarantine/dev").glob("*/evaluation.json"))
    assert list((run_dir / "evaluations").glob("*/evaluation.json"))
    assert list((run_dir / "sessions").glob("*/request.json"))
    assert list((run_dir / "evidence").glob("*/evidence.json"))

    session_text = "\n".join(path.read_text() for path in (run_dir / "sessions").glob("*/request.json"))
    assert "train-a" in session_text
    assert "dev-a" not in session_text and "dev-b" not in session_text


def test_registry_discovers_external_scenario_and_records_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = ScenarioPlugin(
        name="external_fake",
        api_version=SCENARIO_PLUGIN_API_VERSION,
        factory=external_fake_factory,
    )
    entry_point = FixtureEntryPoint("external_fake", plugin)
    monkeypatch.setattr(registry_module, "_scenario_entry_points", lambda: (entry_point,))
    registry = default_registry()
    value = config_value(tmp_path / "runs")
    value["scenario"]["type"] = "external_fake"

    runtime = build_runtime(
        write_config(tmp_path, value),
        run_id="external-scenario",
        registry=registry,
    )

    assert "external_fake" in registry.scenarios
    assert runtime.store.read_json("resolved/scenario_runtime.json") == {
        "schema_version": "autosaddler-scenario-runtime/v1",
        "scenario": {"name": "external_fake", "version": "1"},
        "plugin": {
            "api_version": SCENARIO_PLUGIN_API_VERSION,
            "source": "entry_point",
            "entry_point": {
                "group": SCENARIO_PLUGIN_ENTRY_POINT_GROUP,
                "name": "external_fake",
                "value": "fixture_plugins:external_fake",
            },
            "distribution": {"name": "fixture-scenarios", "version": "1.2.3"},
        },
    }


@pytest.mark.parametrize(
    ("entry_point", "error_type", "message"),
    [
        (
            FixtureEntryPoint("broken", object()),
            TypeError,
            "ScenarioPlugin",
        ),
        (
            FixtureEntryPoint(
                "wrong_api",
                ScenarioPlugin(name="wrong_api", api_version="unsupported", factory=external_fake_factory),
            ),
            ValueError,
            "API version",
        ),
        (
            FixtureEntryPoint(
                "entry_point_name",
                ScenarioPlugin(
                    name="descriptor_name",
                    api_version=SCENARIO_PLUGIN_API_VERSION,
                    factory=external_fake_factory,
                ),
            ),
            ValueError,
            "name",
        ),
    ],
)
def test_registry_rejects_invalid_external_scenario_plugins(
    monkeypatch: pytest.MonkeyPatch,
    entry_point: FixtureEntryPoint,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(registry_module, "_scenario_entry_points", lambda: (entry_point,))

    with pytest.raises(error_type, match=message):
        default_registry()


def test_registry_rejects_duplicate_and_failed_external_scenario_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = ScenarioPlugin(
        name="fake",
        api_version=SCENARIO_PLUGIN_API_VERSION,
        factory=external_fake_factory,
    )
    monkeypatch.setattr(
        registry_module,
        "_scenario_entry_points",
        lambda: (FixtureEntryPoint("fake", duplicate),),
    )
    with pytest.raises(ValueError, match="Duplicate scenario plugin"):
        default_registry()

    monkeypatch.setattr(
        registry_module,
        "_scenario_entry_points",
        lambda: (FixtureEntryPoint("load_failure", object(), error=RuntimeError("boom")),),
    )
    with pytest.raises(RuntimeError, match="load_failure"):
        default_registry()


def test_registry_loads_external_scenario_plugins_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    class RecordingEntryPoint(FixtureEntryPoint):
        def load(self) -> object:
            loaded.append(self.name)
            return super().load()

    entry_points = tuple(
        RecordingEntryPoint(
            name,
            ScenarioPlugin(
                name=name,
                api_version=SCENARIO_PLUGIN_API_VERSION,
                factory=external_fake_factory,
            ),
        )
        for name in ("zeta", "alpha")
    )
    monkeypatch.setattr(registry_module, "_scenario_entry_points", lambda: entry_points)

    default_registry()

    assert loaded == ["alpha", "zeta"]


def test_registry_records_copilot_sdk_version_in_resolved_provenance_and_manifest(tmp_path: Path) -> None:
    value = config_value(tmp_path / "runs")
    value["provider"] = {
        "type": "copilot",
        "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
        "settings": {"model": "test-model", "reasoning_effort": None},
    }

    runtime = build_runtime(write_config(tmp_path, value), run_id="copilot-runtime")

    expected = {
        "schema_version": "autosaddler-provider-runtime/v1",
        "provider_type": "copilot",
        "sdk": {
            "distribution": "github-copilot-sdk",
            "version": version("github-copilot-sdk"),
        },
    }
    resolved = json.loads((runtime.store.run_dir / "resolved/provider_runtime.json").read_text())
    manifest_path = runtime.store.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert resolved == expected
    assert manifest["provider_runtime"] == expected
    runtime.store.validate_integrity()

    manifest["provider_runtime"]["sdk"]["version"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provider_runtime differs"):
        runtime.store.validate_integrity()


def test_registry_rejects_unknown_names_and_extra_keys(tmp_path: Path) -> None:
    unknown = config_value(tmp_path / "runs")
    unknown["provider"]["type"] = "mystery"
    with pytest.raises(ValueError, match="providers.mystery"):
        build_runtime(write_config(tmp_path, unknown), run_id="unknown")

    extra = config_value(tmp_path / "runs")
    extra["optimization"]["acceptance"]["threshold"] = 0.1
    with pytest.raises(ValueError, match=r"extra=\['threshold'\]"):
        build_runtime(write_config(tmp_path, extra), run_id="extra")


def test_config_rejects_removed_generic_session_timeout_key(tmp_path: Path) -> None:
    removed_only = config_value(tmp_path / "runs")
    removed_only["optimization"].pop("diagnosis_patch_timeout_seconds")
    removed_only["optimization"]["session_timeout_seconds"] = 10
    with pytest.raises(ValueError, match="diagnosis_patch_timeout_seconds"):
        RunConfig.load(write_config(tmp_path, removed_only))

    conflicting = config_value(tmp_path / "runs")
    conflicting["optimization"]["session_timeout_seconds"] = 10
    with pytest.raises(ValueError, match="session_timeout_seconds"):
        RunConfig.load(write_config(tmp_path, conflicting))
