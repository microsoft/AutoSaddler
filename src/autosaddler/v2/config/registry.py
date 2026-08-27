from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, PackageNotFoundError, entry_points, version
from pathlib import Path
from typing import Any, Literal, cast

from autosaddler.v2.config.models import RunConfig
from autosaddler.v2.core.domain import JsonValue
from autosaddler.v2.core.engine import AutoSaddlerEngine
from autosaddler.v2.core.policies import (
    BudgetPolicy,
    EpochShuffledTaskSelectionPolicy,
    FixedTaskSelectionPolicy,
    FullOnAcceptDevelopment,
    MatchedValidStrictImprovement,
    MeanDevelopmentRanking,
    PolicyBundle,
)
from autosaddler.v2.core.ports import AgentProvider, ScenarioComponents
from autosaddler.v2.plugins.api import (
    SCENARIO_PLUGIN_API_VERSION,
    SCENARIO_PLUGIN_ENTRY_POINT_GROUP,
    ScenarioFactory,
    ScenarioPlugin,
    validate_scenario_plugin,
)
from autosaddler.v2.plugins.fake import FakeScenarioSettings, build_fake_components
from autosaddler.v2.plugins.meta_are.plugin import build_meta_are_components
from autosaddler.v2.providers.claude import ClaudeAgentProvider, ClaudeProviderConfig
from autosaddler.v2.providers.copilot import (
    CopilotAgentProvider,
    CopilotCustomProviderConfig,
    CopilotProviderConfig,
)
from autosaddler.v2.providers.fake import FakeAgentProvider, PaidWorkLedger
from autosaddler.v2.providers.taskferry import TaskferryAgentProvider, TaskferryProviderConfig
from autosaddler.v2.storage.local import LocalRunStore, TransitionHook

_PROVIDER_SDK_DISTRIBUTIONS = {
    "claude": "claude-agent-sdk",
    "copilot": "github-copilot-sdk",
}


@dataclass(frozen=True, slots=True)
class Runtime:
    config: RunConfig
    store: LocalRunStore
    scenario: ScenarioComponents
    provider: AgentProvider
    policies: PolicyBundle
    engine: AutoSaddlerEngine
    ledger: PaidWorkLedger


@dataclass(frozen=True, slots=True)
class ScenarioRegistration:
    plugin: ScenarioPlugin
    source: Literal["builtin", "entry_point"]
    entry_point_group: str | None = None
    entry_point_name: str | None = None
    entry_point_value: str | None = None
    distribution_name: str | None = None
    distribution_version: str | None = None

    def resolved_entity(self, scenario: ScenarioComponents) -> Mapping[str, JsonValue]:
        if scenario.name != self.plugin.name:
            raise ValueError(
                f"Scenario factory for {self.plugin.name!r} returned components named {scenario.name!r}"
            )
        entry_point_record: Mapping[str, JsonValue] | None = None
        distribution_record: Mapping[str, JsonValue] | None = None
        if self.source == "entry_point":
            entry_point_record = {
                "group": self.entry_point_group,
                "name": self.entry_point_name,
                "value": self.entry_point_value,
            }
            distribution_record = {
                "name": self.distribution_name,
                "version": self.distribution_version,
            }
        return {
            "schema_version": "autosaddler-scenario-runtime/v1",
            "scenario": {"name": scenario.name, "version": scenario.version},
            "plugin": {
                "api_version": self.plugin.api_version,
                "source": self.source,
                "entry_point": entry_point_record,
                "distribution": distribution_record,
            },
        }


class Registry:
    def __init__(self) -> None:
        self.scenarios: dict[str, ScenarioFactory] = {}
        self.scenario_registrations: dict[str, ScenarioRegistration] = {}
        self.providers: dict[str, Callable[..., AgentProvider]] = {}
        self.task_selection: dict[
            str,
            Callable[..., FixedTaskSelectionPolicy | EpochShuffledTaskSelectionPolicy],
        ] = {}
        self.acceptance: dict[str, Callable[[], MatchedValidStrictImprovement]] = {}
        self.development: dict[str, Callable[[], FullOnAcceptDevelopment]] = {}
        self.ranking: dict[str, Callable[[], MeanDevelopmentRanking]] = {}

    def resolve(self, family: str, name: str):
        values = cast(dict[str, Any], getattr(self, family))
        if name not in values:
            raise ValueError(f"Unknown registry name {family}.{name}")
        return values[name]

    def register_scenario(self, registration: ScenarioRegistration) -> None:
        name = registration.plugin.name
        if name in self.scenarios:
            raise ValueError(f"Duplicate scenario plugin name: {name}")
        self.scenarios[name] = registration.plugin.factory
        self.scenario_registrations[name] = registration

    def scenario_registration(self, name: str) -> ScenarioRegistration:
        if name not in self.scenario_registrations:
            raise ValueError(f"Unknown registry name scenarios.{name}")
        return self.scenario_registrations[name]


def default_registry() -> Registry:
    registry = Registry()
    registry.register_scenario(_builtin_scenario("fake", _fake_scenario))
    registry.register_scenario(_builtin_scenario("meta_are", build_meta_are_components))
    _register_external_scenarios(registry)
    registry.providers.update(
        {
            "fake": lambda *, ledger, settings: _fake_provider(ledger, settings),
            "claude": _registered_claude_provider,
            "copilot": _registered_copilot_provider,
            "taskferry": _registered_taskferry_provider,
        }
    )
    registry.task_selection["fixed"] = lambda *, batch_size, seed: FixedTaskSelectionPolicy(batch_size=batch_size)
    registry.task_selection["epoch_shuffled"] = lambda *, batch_size, seed: EpochShuffledTaskSelectionPolicy(
        batch_size=batch_size,
        seed=seed,
    )
    registry.acceptance["matched_valid_strict_improvement"] = MatchedValidStrictImprovement
    registry.development["full_on_accept"] = FullOnAcceptDevelopment
    registry.ranking["mean_development_score"] = MeanDevelopmentRanking
    return registry


def _builtin_scenario(name: str, factory: ScenarioFactory) -> ScenarioRegistration:
    return ScenarioRegistration(
        plugin=ScenarioPlugin(
            name=name,
            api_version=SCENARIO_PLUGIN_API_VERSION,
            factory=factory,
        ),
        source="builtin",
    )


def _scenario_entry_points() -> tuple[EntryPoint, ...]:
    return tuple(entry_points(group=SCENARIO_PLUGIN_ENTRY_POINT_GROUP))


def _register_external_scenarios(registry: Registry) -> None:
    discovered = sorted(
        _scenario_entry_points(),
        key=lambda item: (item.name, item.value),
    )
    for entry_point in discovered:
        try:
            loaded = entry_point.load()
        except Exception as error:
            raise RuntimeError(
                f"Failed to load scenario plugin entry point {entry_point.name!r}"
            ) from error
        plugin = validate_scenario_plugin(loaded, entry_point_name=entry_point.name)
        distribution = entry_point.dist
        if distribution is None:
            raise RuntimeError(
                f"Scenario plugin entry point {entry_point.name!r} has no owning distribution"
            )
        distribution_name = distribution.metadata.get("Name")
        if not distribution_name or not distribution.version:
            raise RuntimeError(
                f"Scenario plugin entry point {entry_point.name!r} has incomplete distribution metadata"
            )
        registry.register_scenario(
            ScenarioRegistration(
                plugin=plugin,
                source="entry_point",
                entry_point_group=SCENARIO_PLUGIN_ENTRY_POINT_GROUP,
                entry_point_name=entry_point.name,
                entry_point_value=entry_point.value,
                distribution_name=distribution_name,
                distribution_version=distribution.version,
            )
        )


def _fake_scenario(*, settings, base_dir, run_dir, store, ledger) -> ScenarioComponents:
    del base_dir
    return build_fake_components(
        settings=_fake_settings(settings),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )


def _registered_claude_provider(*, ledger, settings) -> ClaudeAgentProvider:
    del ledger
    return _claude_provider(settings)


def _registered_copilot_provider(*, ledger, settings) -> CopilotAgentProvider:
    del ledger
    return _copilot_provider(settings)


def _registered_taskferry_provider(*, ledger, settings) -> TaskferryAgentProvider:
    del ledger
    return _taskferry_provider(settings)


def build_runtime(
    config_path: Path,
    *,
    run_id: str,
    transition_hook: TransitionHook | None = None,
    registry: Registry | None = None,
) -> Runtime:
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError("run_id must be one safe path segment")
    config = RunConfig.load(config_path.resolve())
    if config.storage.type != "local":
        raise ValueError(f"Unknown registry name storage.{config.storage.type}")
    registry = registry or default_registry()
    scenario_factory = registry.resolve("scenarios", config.scenario.type)
    scenario_registration = registry.scenario_registration(config.scenario.type)
    provider_factory = registry.resolve("providers", config.provider.type)
    task_selection_factory = registry.resolve("task_selection", config.optimization.task_selection.type)
    acceptance_factory = registry.resolve("acceptance", config.optimization.acceptance.type)
    development_factory = registry.resolve("development", config.optimization.development.type)
    ranking_factory = registry.resolve("ranking", config.optimization.ranking.type)

    run_dir = config.storage.run_root / run_id
    store = LocalRunStore(run_dir=run_dir, run_id=run_id, transition_hook=transition_hook)
    ledger = PaidWorkLedger(run_dir / "audit/fake_paid_work.jsonl")
    scenario = scenario_factory(
        settings=config.scenario.settings,
        base_dir=config_path.parent,
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )
    if scenario.name != scenario_registration.plugin.name:
        raise ValueError(
            f"Scenario factory for {scenario_registration.plugin.name!r} "
            f"returned components named {scenario.name!r}"
        )
    missing_capabilities = sorted(scenario.required_capabilities - config.provider.capabilities)
    if missing_capabilities:
        raise ValueError(f"Configured provider lacks required capabilities: {missing_capabilities}")
    provider = provider_factory(ledger=ledger, settings=config.provider.settings)
    policies = PolicyBundle(
        task_selection=task_selection_factory(
            batch_size=config.optimization.task_selection.batch_size,
            seed=config.optimization.task_selection.seed,
        ),
        acceptance=acceptance_factory(),
        development=development_factory(),
        ranking=ranking_factory(),
        budget=BudgetPolicy(
            max_rollouts=config.optimization.budget.max_rollouts,
            max_iterations=config.optimization.budget.max_iterations,
        ),
    )
    resolved_entities = _resolved_entities(
        config,
        scenario,
        policies,
        scenario_registration,
    )
    store.initialize(resolved_config=config.as_mapping(), resolved_entities=resolved_entities)
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=policies,
        diagnosis_patch_timeout_seconds=config.optimization.diagnosis_patch_timeout_seconds,
        selection_timeout_seconds=config.optimization.selection_timeout_seconds,
        reflection_timeout_seconds=config.optimization.reflection_timeout_seconds,
        session_retries=config.optimization.session_retries,
        session_retry_backoff_seconds=config.optimization.session_retry_backoff_seconds,
    )
    return Runtime(config, store, scenario, provider, policies, engine, ledger)


def _fake_settings(value: Mapping[str, JsonValue]) -> FakeScenarioSettings:
    expected = {"baseline", "target_component", "improved_text", "train_case_ids", "development_case_ids"}
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing or extra:
        raise ValueError(f"Invalid keys at scenario.settings: missing={missing}, extra={extra}")
    baseline = value["baseline"]
    train = value["train_case_ids"]
    development = value["development_case_ids"]
    if not isinstance(baseline, Mapping) or any(
        not isinstance(key, str) or not isinstance(text, str) for key, text in baseline.items()
    ):
        raise TypeError("scenario.settings.baseline must map strings to strings")
    if not isinstance(train, list) or any(not isinstance(item, str) for item in train):
        raise TypeError("scenario.settings.train_case_ids must be a list of strings")
    if not isinstance(development, list) or any(not isinstance(item, str) for item in development):
        raise TypeError("scenario.settings.development_case_ids must be a list of strings")
    return FakeScenarioSettings(
        baseline=cast(Mapping[str, str], baseline),
        target_component=_string(value["target_component"], "scenario.settings.target_component"),
        improved_text=_string(value["improved_text"], "scenario.settings.improved_text"),
        train_case_ids=tuple(train),
        development_case_ids=tuple(development),
    )


def _fake_provider(ledger: PaidWorkLedger, settings: Mapping[str, JsonValue]) -> FakeAgentProvider:
    if settings:
        raise ValueError(f"Invalid keys at provider.settings for fake: {sorted(settings)}")
    return FakeAgentProvider(ledger)


def _claude_provider(settings: Mapping[str, JsonValue]) -> ClaudeAgentProvider:
    expected = {"model", "effort", "permission_mode", "base_url"}
    _exact_settings(settings, expected, "claude")
    return ClaudeAgentProvider(
        ClaudeProviderConfig(
            model=_string(settings["model"], "provider.settings.model"),
            effort=_optional_string(settings["effort"], "provider.settings.effort"),
            permission_mode=_string(settings["permission_mode"], "provider.settings.permission_mode"),
            base_url=_string(settings["base_url"], "provider.settings.base_url"),
        )
    )


def _copilot_provider(settings: Mapping[str, JsonValue]) -> CopilotAgentProvider:
    required = {"model", "reasoning_effort"}
    missing = sorted(required - settings.keys())
    extra = sorted(settings.keys() - required - {"provider"})
    if missing or extra:
        raise ValueError(f"Invalid keys at provider.settings for copilot: missing={missing}, extra={extra}")
    return CopilotAgentProvider(
        CopilotProviderConfig(
            model=_string(settings["model"], "provider.settings.model"),
            reasoning_effort=_optional_string(settings["reasoning_effort"], "provider.settings.reasoning_effort"),
            provider=_copilot_custom_provider(settings.get("provider")),
        )
    )


def _copilot_custom_provider(value: JsonValue | None) -> CopilotCustomProviderConfig | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("provider.settings.provider must be a mapping")
    expected = {"type", "base_url", "wire_api", "model_id", "wire_model"}
    _exact_settings(value, expected, "copilot custom provider")
    return CopilotCustomProviderConfig(
        type=_string(value["type"], "provider.settings.provider.type"),
        base_url=_string(value["base_url"], "provider.settings.provider.base_url"),
        wire_api=_string(value["wire_api"], "provider.settings.provider.wire_api"),
        model_id=_string(value["model_id"], "provider.settings.provider.model_id"),
        wire_model=_string(value["wire_model"], "provider.settings.provider.wire_model"),
    )


def _taskferry_provider(settings: Mapping[str, JsonValue]) -> TaskferryAgentProvider:
    expected = {"model", "variant", "executor", "sandboxed"}
    missing = sorted({"model"} - settings.keys())
    extra = sorted(settings.keys() - expected)
    if missing or extra:
        raise ValueError(f"Invalid keys at provider.settings for taskferry: missing={missing}, extra={extra}")
    return TaskferryAgentProvider(
        TaskferryProviderConfig(
            model=_string(settings["model"], "provider.settings.model"),
            variant=_optional_string(settings.get("variant"), "provider.settings.variant"),
            executor=_optional_string(settings.get("executor"), "provider.settings.executor"),
            sandboxed=_optional_bool(settings.get("sandboxed"), "provider.settings.sandboxed", default=True),
        )
    )


def _resolved_entities(
    config: RunConfig,
    scenario: ScenarioComponents,
    policies: PolicyBundle,
    scenario_registration: ScenarioRegistration,
) -> Mapping[str, str | Mapping[str, JsonValue]]:
    common: dict[str, str | Mapping[str, JsonValue]] = {
        "resolved/component_graph.json": {
            "schema_version": "autosaddler-component-graph/v1",
            "scenario": {"name": scenario.name, "version": scenario.version},
            "provider": config.provider.type,
            "components": ["harness_space", "evaluator", "evidence_builder", "prompt_pack"],
        },
        "resolved/scenario_runtime.json": scenario_registration.resolved_entity(scenario),
        "resolved/provider_runtime.json": _provider_runtime(config.provider.type),
        "resolved/policies.json": {
            "task_selection": config.optimization.task_selection.type,
            "acceptance": config.optimization.acceptance.type,
            "development": config.optimization.development.type,
            "ranking": config.optimization.ranking.type,
            "budget": {
                "max_rollouts": policies.budget.max_rollouts,
                "max_iterations": policies.budget.max_iterations,
            },
        },
        "resolved/schemas/observations.json": {
            "$id": "autosaddler-observation/v1",
            "type": "object",
            "required": ["observation_id", "candidate_id", "case_id", "split", "disposition"],
        },
        "resolved/schemas/session_outputs.json": {
            "$id": "autosaddler-session-outputs/v1",
            "kinds": ["evolve", "diagnose_patch", "reflect"],
        },
    }
    overlap = sorted(common.keys() & scenario.resolved_entities.keys())
    if overlap:
        raise ValueError(f"Scenario resolved entities collide with common entities: {overlap}")
    return {**common, **scenario.resolved_entities}


def _provider_runtime(provider_type: str) -> Mapping[str, JsonValue]:
    distribution = _PROVIDER_SDK_DISTRIBUTIONS.get(provider_type)
    sdk: Mapping[str, JsonValue] | None = None
    if distribution is not None:
        try:
            sdk_version = version(distribution)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Provider SDK distribution is not installed: {distribution}") from error
        sdk = {"distribution": distribution, "version": sdk_version}
    return {
        "schema_version": "autosaddler-provider-runtime/v1",
        "provider_type": provider_type,
        "sdk": sdk,
    }


def _exact_settings(value: Mapping[str, JsonValue], expected: set[str], provider: str) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing or extra:
        raise ValueError(f"Invalid keys at provider.settings for {provider}: missing={missing}, extra={extra}")


def _string(value: JsonValue, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _optional_string(value: JsonValue, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _optional_bool(value: JsonValue, path: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value
