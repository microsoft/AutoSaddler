from __future__ import annotations

import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosaddler.v1.core.data_loader import ListDataLoader
from autosaddler.v1.proposer.autosaddler.pattern_registry import PatternRegistry
from autosaddler.v1.proposer.autosaddler.prompt_builder import install_pattern_cli
from autosaddler.v1.proposer.autosaddler.strategy import resolve_strategy
from autosaddler.v1.proposer.autosaddler.strategy_settings import resolve_strategy_settings
from autosaddler.v1.strategies.batch_sampler import ActiveSaddlerBanditSampler

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_strategy_specs_gate_activesaddler_sessions() -> None:
    passive = resolve_strategy("autosaddler")
    active = resolve_strategy(" ActiveSaddler ")

    assert passive.enabled_sessions == (0, 1, 2)
    assert not passive.pattern_sampling
    assert active.enabled_sessions == (0, 1, 2, 3, 3.5, 4)
    assert active.pattern_sampling and active.agent_scoring and active.agent_arm_creation
    assert active.pattern_cli_capabilities_for_session(4) >= {"rate", "list"}
    assert "rate" not in active.pattern_cli_capabilities_for_session(3.5)
    with pytest.raises(ValueError, match="sampling_strategy"):
        resolve_strategy("bandit")


def test_strategy_settings_resolve_nested_values_and_reject_conflicts() -> None:
    settings = resolve_strategy_settings(
        {
            "sampling_strategy": "activesaddler",
            "pattern": {"extraction_timeout": 120.0},
            "bandit": {
                "selection": {"softmax_temperature": 0.2, "min_prob": 0.0},
                "scoring": {"ema_eta": 0.9, "agent_timeout": 60.0},
            },
        }
    )

    assert (settings.eta, settings.softmax_temperature, settings.min_prob) == (0.9, 0.2, 0.0)
    assert (settings.pattern_extraction_timeout, settings.arm_scoring_timeout) == (120.0, 60.0)
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_strategy_settings({"eta": 0.5, "bandit": {"scoring": {"ema_eta": 0.9}}})
    with pytest.raises(ValueError, match="min_prob"):
        resolve_strategy_settings({"bandit": {"selection": {"min_prob": 1.0}}})


def registry_with_two_arms(tmp_path: Path) -> tuple[PatternRegistry, str, str]:
    registry = PatternRegistry(str(tmp_path))
    strong = registry.register("strong", created_iteration=1)
    weak = registry.register("weak", created_iteration=1)
    for scenario in ("s0", "s1", "s2"):
        registry.tag(strong, harness_idx=1, trace_dir="before", scenario_id=scenario, root_cause="r")
    registry.tag(weak, harness_idx=1, trace_dir="before", scenario_id="s3", root_cause="r")
    registry.record_agent_score(strong, 2, severity=1.0, fixability=1.0, breadth=1.0, side_effect=0.0, rationale="r")
    registry.record_agent_score(weak, 2, severity=0.0, fixability=0.0, breadth=0.0, side_effect=1.0, rationale="r")
    return registry, strong, weak


def sampler(tmp_path: Path, registry: PatternRegistry) -> ActiveSaddlerBanditSampler:
    return ActiveSaddlerBanditSampler(
        minibatch_size=2,
        pattern_registry=registry,
        eta=0.9,
        temperature=0.05,
        min_prob=0.0,
        scenario_to_idx={f"s{index}": index for index in range(6)},
        state_path=tmp_path / "bandit_state.json",
        rng=random.Random(42),
    )


def test_bandit_draws_unseen_then_pulls_the_best_scored_arm(tmp_path: Path) -> None:
    registry, strong, _ = registry_with_two_arms(tmp_path)
    loader = ListDataLoader([f"s{index}" for index in range(6)])
    bandit = sampler(tmp_path, registry)
    state = SimpleNamespace(i=1)

    drawn = bandit.next_minibatch_ids(loader, state, forced_action="draw")
    bandit.mark_executed(drawn)
    pulled = bandit.next_minibatch_ids(loader, state, forced_action="pull")

    assert len(drawn) == 2 and bandit.unseen_pool_size(loader) == 4
    assert bandit.last_score_snapshot["action"] == "arm_pull"
    assert bandit.last_score_snapshot["chosen_arm"] == strong
    assert set(pulled) <= {0, 1, 2}
    with pytest.raises(ValueError, match="forced_action"):
        bandit.next_minibatch_ids(loader, state, forced_action=None)


def test_bandit_state_round_trips_for_deterministic_resume(tmp_path: Path) -> None:
    registry, _, _ = registry_with_two_arms(tmp_path)
    loader = ListDataLoader([f"s{index}" for index in range(6)])
    first = sampler(tmp_path, registry)
    first.mark_executed(first.next_minibatch_ids(loader, SimpleNamespace(i=0), forced_action="draw"))
    first.record_probe_points(["s0"], "commit-a")

    resumed = sampler(tmp_path, registry)
    expected = first.next_minibatch_ids(loader, SimpleNamespace(i=1), forced_action="pull")

    assert resumed.n_probes == 1
    assert resumed.next_minibatch_ids(loader, SimpleNamespace(i=1), forced_action="pull") == expected


def test_softmax_floor_applies_minimum_probability(tmp_path: Path) -> None:
    registry, _, _ = registry_with_two_arms(tmp_path)
    bandit = sampler(tmp_path, registry)
    bandit.min_prob = 0.2

    probabilities = bandit._softmax_floor({"a": 1.0, "b": 0.0})

    assert probabilities["b"] >= 0.2
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_pattern_cli_wrapper_enforces_session_capabilities(tmp_path: Path) -> None:
    registry, _, _ = registry_with_two_arms(tmp_path)
    registry.save()
    environment = install_pattern_cli(str(tmp_path), str(tmp_path), current_iteration=2, eta=0.9, session=4)
    wrapper = Path(environment["PATH"].split(":", 1)[0]) / "pattern"

    listed = subprocess.run([str(wrapper), "list"], capture_output=True, text=True, check=False)
    denied = subprocess.run(
        [str(wrapper), "decide", "--action", "pull", "--rationale", "r"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert f'export PYTHONPATH="{REPO_ROOT / "src"}' in wrapper.read_text(encoding="utf-8")
    assert listed.returncode == 0, listed.stderr
    assert "strong" in listed.stdout
    assert denied.returncode == 2
    assert "not available" in denied.stderr


@pytest.mark.parametrize("config", ["meta_are_activesaddler.yaml", "meta_are_activesaddler_smoke.yaml"])
def test_activesaddler_configs_dry_run_without_building_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: str,
) -> None:
    from autosaddler.v1.adapters.meta_are_adapter import meta_are_adapter, optimize as optimize_cli

    def unexpected_adapter(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run must not construct a Meta-ARE adapter")

    monkeypatch.setenv("META_ARE_REPO", str(tmp_path / "meta-are"))
    monkeypatch.setattr(meta_are_adapter, "MetaAREAdapter", unexpected_adapter)
    monkeypatch.setattr(
        "sys.argv",
        ["autosaddler-v1", "--config", str(REPO_ROOT / "configs/v1" / config), "--dry-run"],
    )

    optimize_cli.main()

    loaded = optimize_cli.load_config(str(REPO_ROOT / "configs/v1" / config))
    assert resolve_strategy_settings(loaded["autosaddler"]).strategy.name.value == "activesaddler"
    assert not (tmp_path / "meta-are").exists()
