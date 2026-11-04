from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import yaml

from autosaddler.v2.config.models import NamedPolicyConfig, ParetoAcceptanceConfig, RunConfig
from autosaddler.v2.core.domain import ParetoScreenParameters
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


def pareto_config_value(run_root: Path, repetitions: int = 2) -> dict:
    value = config_value(run_root)
    value["scenario"] = {
        "type": "meta_are",
        "settings": {
            "repetitions": repetitions,
            "capability_phase_iterations": 1,
            "task_agent_pricing": {
                "input_usd_per_million": 2.0,
                "cached_input_usd_per_million": 0.5,
                "output_usd_per_million": 8.0,
            },
        },
    }
    value["optimization"]["acceptance"] = {"type": "pareto", "parameters": {"draws": 100}}
    value["optimization"]["ranking"] = {"type": "pareto"}
    return value


@pytest.mark.parametrize("repetitions", [1, 2, 3, 5, 10])
def test_pareto_config_reuses_screen_parameters_and_scenario_repetitions(tmp_path: Path, repetitions: int) -> None:
    value = pareto_config_value(tmp_path / "runs", repetitions)
    config = RunConfig.load(write_config(tmp_path, value))
    acceptance = config.optimization.acceptance
    assert isinstance(acceptance, ParetoAcceptanceConfig)
    assert acceptance.parameters.repetitions == repetitions
    assert acceptance.parameters.draws == 100
    assert acceptance.parameters.rng_seed == 42
    assert config.optimization.ranking == NamedPolicyConfig(type="pareto")
    resolved = dict(config.as_mapping())
    assert "repetitions" not in resolved["optimization"]["acceptance"]["parameters"]
    assert RunConfig.load(write_config(tmp_path, resolved)) == config


@pytest.mark.parametrize("example", ["meta_are_smoke.yaml", "meta_are_full.yaml"])
def test_documented_pareto_fragment_parses_with_existing_examples(tmp_path: Path, example: str) -> None:
    root = Path(__file__).resolve().parents[3]
    guide = (root / "README.md").read_text()
    section = guide.split("## Configuration\n", 1)[1]
    fragment = yaml.safe_load(section.split("```yaml\n", 1)[1].split("```", 1)[0])
    value = yaml.safe_load((root / "configs/v2" / example).read_text())
    value["scenario"]["settings"].update(fragment["scenario"]["settings"])
    value["optimization"].update(fragment["optimization"])
    config = RunConfig.load(write_config(tmp_path, value))
    assert config.scenario.type == "meta_are"
    assert config.scenario.settings["repetitions"] == 3
    assert config.scenario.settings["task_agent_pricing"] == fragment["scenario"]["settings"]["task_agent_pricing"]
    assert config.optimization.acceptance == ParetoAcceptanceConfig(
        type="pareto", parameters=ParetoScreenParameters(
            repetitions=3, draws=10000, rng_seed=42,
            max_accuracy_drop=0.10, joint_support_threshold=0.70,
        ),
    )
    assert config.optimization.ranking == NamedPolicyConfig(type="pareto")
    assert config.optimization.final_selection == "pareto"


@pytest.mark.parametrize("explicit_parameters", [False, True])
def test_pareto_named_policy_uses_default_screen_parameters(tmp_path: Path, explicit_parameters: bool) -> None:
    value = pareto_config_value(tmp_path / "runs")
    value["optimization"]["acceptance"] = {"type": "pareto"}
    if explicit_parameters:
        value["optimization"]["acceptance"]["parameters"] = {}
    config = RunConfig.load(write_config(tmp_path, value))
    assert config.optimization.acceptance == ParetoAcceptanceConfig(
        type="pareto", parameters=ParetoScreenParameters(repetitions=2),
    )


def test_pareto_smoke_config_preserves_accuracy_example_and_v5_settings() -> None:
    root = Path(__file__).resolve().parents[3]
    config = RunConfig.load(root / "configs/v2/meta_are_pareto_smoke.yaml")
    accuracy = RunConfig.load(root / "configs/v2/meta_are_smoke.yaml")
    settings = config.scenario.settings
    for key in (
        "source_repo", "base_revision", "train_manifest", "development_manifest", "test_manifest",
        "writable_paths", "forbidden_paths",
    ):
        assert settings[key] == accuracy.scenario.settings[key]
    assert settings["repetitions"] == 3
    assert settings["model"] == "gpt-5.5-20260424"
    assert settings["reasoning_effort"] == "xhigh"
    assert settings["judge_model"] == "Llama-3.3-70B-Instruct"
    assert settings["task_agent_pricing"] == {
        "input_usd_per_million": 5.0, "cached_input_usd_per_million": 0.5, "output_usd_per_million": 30.0,
    }
    assert config.optimization.acceptance == ParetoAcceptanceConfig(
        type="pareto", parameters=ParetoScreenParameters(repetitions=3),
    )
    assert config.optimization.ranking == NamedPolicyConfig(type="pareto")
    assert config.optimization.final_selection == "pareto"
    assert config.optimization.task_selection.batch_size == 3
    assert config.optimization.task_selection.seed == 42
    assert config.optimization.budget.max_iterations == 2
    assert config.optimization.budget.max_rollouts == 45
    assert config.provider.type == "copilot"
    assert config.provider.settings == {"model": "gpt-5.5", "reasoning_effort": "xhigh"}
    provider = default_registry().providers[config.provider.type](ledger=None, settings=config.provider.settings)
    assert isinstance(provider, registry_module.CopilotAgentProvider)
    assert accuracy.optimization.acceptance.type == "matched_valid_strict_improvement"
    assert accuracy.scenario.settings["repetitions"] == 1


@pytest.mark.parametrize("policy", ["acceptance", "development", "ranking"])
def test_ordinary_named_policies_still_reject_parameters(tmp_path: Path, policy: str) -> None:
    value = config_value(tmp_path / "runs")
    value["optimization"][policy]["parameters"] = {}
    with pytest.raises(ValueError, match=f"Invalid keys at optimization.{policy}"):
        RunConfig.load(write_config(tmp_path, value))


def test_pareto_acceptance_requires_scenario_repetitions(tmp_path: Path) -> None:
    value = pareto_config_value(tmp_path / "runs")
    del value["scenario"]["settings"]["repetitions"]
    with pytest.raises(ValueError, match="scenario.settings.repetitions"):
        RunConfig.load(write_config(tmp_path, value))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("settings", "repetitions", 0),
        ("settings", "repetitions", True),
        ("optimization", "ranking", {"type": "pareto", "parameters": {}}),
        ("parameters", "repetitions", 2),
        ("parameters", "draws", 0),
        ("parameters", "rng_seed", -1),
        ("parameters", "joint_support_threshold", 1.1),
        ("parameters", "max_accuracy_drop", -0.1),
        ("parameters", "method", "other"),
    ],
)
def test_pareto_config_rejects_invalid_parameters(tmp_path: Path, section: str, key: str, value: object) -> None:
    config = pareto_config_value(tmp_path / "runs")
    target = (
        config["scenario"]["settings"] if section == "settings" else
        config["optimization"]["acceptance"]["parameters"] if section == "parameters" else
        config["optimization"]
    )
    target[key] = value
    with pytest.raises((ValueError, TypeError)):
        RunConfig.load(write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("settings", "task_agent_pricing", None),
        ("settings", "capability_phase_iterations", 0),
        ("optimization", "acceptance", {"type": "matched_valid_strict_improvement"}),
        ("optimization", "ranking", {"type": "mean_development_score"}),
        ("optimization", "development", {"type": "other"}),
    ],
)
def test_config_parsing_does_not_enforce_pareto_combinations(
    tmp_path: Path, section: str, key: str, value: object,
) -> None:
    config = pareto_config_value(tmp_path / "runs")
    target = config["scenario"]["settings"] if section == "settings" else config["optimization"]
    target[key] = value
    resolved = RunConfig.load(write_config(tmp_path, config)).as_mapping()
    actual = resolved["scenario"]["settings"] if section == "settings" else resolved["optimization"]
    assert actual[key] == value


def test_registry_builds_complete_fake_runtime_and_artifact_layout(tmp_path: Path) -> None:
    runtime = build_runtime(write_config(tmp_path, config_value(tmp_path / "runs")), run_id="registry-run")
    result = runtime.engine.run()
    run_dir = runtime.store.run_dir

    assert result.development_score == 1.0
    assert "final_selection" not in runtime.config.as_mapping()["optimization"]
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


@pytest.mark.parametrize("preference", [None, "score", "cost", "pareto", "accuracy"])
def test_registry_assembles_pareto_policies_without_starting_work(tmp_path: Path, preference: str | None) -> None:
    from autosaddler.v2.core.policies import ParetoAcceptance, ParetoDevelopmentRanking

    registry = default_registry()
    fake_settings = config_value(tmp_path / "runs")["scenario"]["settings"]

    def scenario_factory(**kwargs):
        return replace(registry_module._fake_scenario(**{**kwargs, "settings": fake_settings}), name="meta_are")

    registry.scenarios["meta_are"] = scenario_factory
    value = pareto_config_value(tmp_path / "runs")
    if preference is not None:
        value["optimization"]["final_selection"] = preference
    runtime = build_runtime(
        write_config(tmp_path, value), run_id="pareto-config", registry=registry,
    )
    expected_preference = "score" if preference is None else preference
    assert runtime.engine.final_selection == expected_preference
    assert runtime.config.as_mapping()["optimization"]["final_selection"] == expected_preference
    assert RunConfig.load(write_config(tmp_path, dict(runtime.config.as_mapping()))) == runtime.config
    assert isinstance(runtime.policies.acceptance, ParetoAcceptance)
    assert isinstance(runtime.policies.ranking, ParetoDevelopmentRanking)
    assert runtime.policies.ranking.repetitions == runtime.policies.acceptance.parameters.repetitions == 2
    resolved = runtime.store.read_json("resolved/policies.json")["pareto"]
    assert resolved["method"] == "bayesian_bootstrap"
    assert resolved["parameters"]["draws"] == 100
    assert resolved["parameters"]["repetitions"] == 2
    assert runtime.store.events() == ()


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
