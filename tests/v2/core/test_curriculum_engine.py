from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from autosaddler.v2.config.registry import build_runtime, default_registry
from autosaddler.v2.core.curriculum import CURRICULUM_NAMESPACE, CurriculumState
from autosaddler.v2.core.domain import Cost, canonical_json
from autosaddler.v2.core.engine import AutoSaddlerEngine, SessionRetriesExhausted
from autosaddler.v2.core.policies import (
    ActiveSaddlerTaskSelectionPolicy,
    BudgetPolicy,
    FullOnAcceptDevelopment,
    MatchedValidStrictImprovement,
    MeanDevelopmentRanking,
    PolicyBundle,
)
from autosaddler.v2.core.ports import BASE_SESSION_KINDS
from autosaddler.v2.plugins.fake import FakeScenarioSettings, build_fake_components
from autosaddler.v2.prompting.models import SessionResult
from autosaddler.v2.providers.fake import FakeAgentProvider, PaidWorkLedger
from autosaddler.v2.storage.local import LocalRunStore

TRAIN_CASES = ["train-a", "train-b", "train-c", "train-d"]


def curriculum_config(root: Path, *, max_iterations: int = 2) -> dict:
    return {
        "schema_version": "autosaddler/v2",
        "scenario": {
            "type": "fake",
            "settings": {
                "baseline": {"instruction": "baseline"},
                "target_component": "instruction",
                "improved_text": "improved",
                "train_case_ids": TRAIN_CASES,
                "development_case_ids": ["dev-a", "dev-b"],
            },
        },
        "optimization": {
            "task_selection": {
                "type": "activesaddler",
                "batch_size": 2,
                "seed": 0,
                "settings": {"softmax_temperature": 0.15, "min_prob": 0.0, "ema_eta": 0.9},
            },
            "acceptance": {"type": "matched_valid_strict_improvement"},
            "development": {"type": "full_on_accept"},
            "ranking": {"type": "mean_development_score"},
            "budget": {"max_rollouts": 100, "max_iterations": max_iterations},
            "diagnosis_patch_timeout_seconds": 10,
            "pattern_extraction_timeout_seconds": 11,
            "arm_scoring_timeout_seconds": 12,
        },
        "provider": {
            "type": "fake",
            "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
            "settings": {},
        },
        "storage": {"type": "local", "run_root": str(root / "runs")},
    }


def write_config(root: Path, value: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def curriculum_changes(store: LocalRunStore) -> list[dict]:
    return [
        dict(event.payload)
        for event in store.events_of_type("ExtensionStateChanged")
        if event.payload.get("namespace") == CURRICULUM_NAMESPACE
    ]


def session_stages(store: LocalRunStore) -> list[str]:
    return [str(event.payload.get("stage")) for event in store.events_of_type("SessionStarted")]


def test_activesaddler_runtime_orders_curriculum_sessions(tmp_path: Path) -> None:
    runtime = build_runtime(write_config(tmp_path, curriculum_config(tmp_path)), run_id="curriculum")
    result = runtime.engine.run()
    store = runtime.store

    assert result.development_score == 1.0
    assert session_stages(store) == [
        "proposal.selection",
        "proposal.patch",
        "proposal.reflection",
        "proposal.pattern_extraction",
        "proposal.selection",
        "proposal.arm_decision",
        "proposal.arm_scoring",
        "proposal.patch",
        "proposal.reflection",
        "proposal.pattern_extraction",
    ]
    batches = store.events_of_type("BatchSampled")
    assert [event.payload["provenance"]["action"] for event in batches] == ["unseen_draw", "arm_pull"]
    first_batch = list(batches[0].payload["case_ids"])
    assert batches[1].payload["case_ids"] == first_batch
    assert [change["change"] for change in curriculum_changes(store)] == [
        "arm_decision",
        "patterns_extracted",
        "observations_recorded",
        "arm_decision",
        "arm_scores",
        "patterns_extracted",
        "observations_recorded",
    ]
    first_decision = curriculum_changes(store)[0]
    assert first_decision["requested_action"] is None and first_decision["action"] == "unseen_draw"

    state = CurriculumState.replay(store.events())
    (pattern,) = state.patterns.values()
    assert pattern.case_ids == tuple(first_batch)
    assert {tag.source for tag in pattern.tags} == {"pre_patch"}
    assert [observation.active for observation in pattern.observations] == [0.0, 0.0]
    assert batches[1].payload["provenance"]["chosen_arm"] == pattern.pattern_id

    timeouts = {}
    for started in store.events_of_type("SessionStarted"):
        request = store.read_json(str(started.payload["request"]["uri"]))
        timeouts[request["spec"]["kind"]] = request["timeout_seconds"]
    assert timeouts == {
        "evolve": 10.0,
        "diagnose_patch": 10.0,
        "reflect": 10.0,
        "extract_patterns": 11.0,
        "decide_arm": 12.0,
        "score_arms": 12.0,
    }
    policies = json.loads((store.run_dir / "resolved/policies.json").read_text())
    assert policies["task_selection"] == "activesaddler"
    assert policies["task_selection_settings"] == {"softmax_temperature": 0.15, "min_prob": 0.0, "ema_eta": 0.9}
    kinds = json.loads((store.run_dir / "resolved/schemas/session_outputs.json").read_text())["kinds"]
    assert kinds == ["evolve", "diagnose_patch", "reflect", "decide_arm", "extract_patterns", "score_arms"]
    projection = json.loads((store.run_dir / "strategy/curriculum.json").read_text())
    assert projection["namespace"] == CURRICULUM_NAMESPACE
    assert len(projection["changes"]) == 7
    paid = runtime.ledger.entries()
    assert sum(entry["kind"] == "session" for entry in paid) == 10


class LatestParentProvider:
    """Fake provider whose evolution session always builds on the newest accepted candidate."""

    def __init__(self, ledger: PaidWorkLedger, *failing_kinds: str) -> None:
        self.delegate = FakeAgentProvider(ledger)
        self.failing_kinds = frozenset(failing_kinds)

    async def run(self, request):
        if request.spec.kind in self.failing_kinds:
            return SessionResult(
                status="failed",
                structured_output=None,
                raw_response="",
                tool_calls=(),
                usage=(),
                cost=Cost(sessions=1),
                error=f"persistent {request.spec.kind} failure",
            )
        if request.spec.kind == "evolve":
            context = json.loads(request.spec.workspace_files["session_context.json"])
            response = {
                "schema_version": "autosaddler-evolution/v1",
                "parent_ids": [context["candidate_ids"][-1]],
                "component_sources": {},
                "rationale": "Build on the newest accepted candidate.",
            }
            files = dict(request.spec.workspace_files)
            files[".autosaddler/fake_response.json"] = canonical_json(response) + "\n"
            request = replace(request, spec=replace(request.spec, workspace_files=files))
        return await self.delegate.run(request)


def direct_engine(tmp_path: Path, *failing_kinds: str, max_iterations: int = 2) -> tuple[AutoSaddlerEngine, LocalRunStore]:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="direct-curriculum")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "fake", "provider": "fake"}},
    )
    ledger = PaidWorkLedger(run_dir / "audit/fake_paid_work.jsonl")
    scenario = build_fake_components(
        settings=FakeScenarioSettings(
            baseline={"instruction": "baseline"},
            target_component="instruction",
            improved_text="improved",
            train_case_ids=tuple(TRAIN_CASES),
            development_case_ids=("dev-a", "dev-b"),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=LatestParentProvider(ledger, *failing_kinds),
        policies=PolicyBundle(
            task_selection=ActiveSaddlerTaskSelectionPolicy(
                batch_size=2,
                seed=0,
                softmax_temperature=0.15,
                min_prob=0.0,
                ema_eta=0.9,
            ),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=max_iterations),
        ),
        session_retries=0,
    )
    return engine, store


def test_all_pass_pull_records_inactive_observation_without_diagnosis(tmp_path: Path) -> None:
    engine, store = direct_engine(tmp_path)

    engine.run()

    outcomes = [event.payload["outcome"] for event in store.events_of_type("IterationCompleted")]
    assert outcomes == ["accepted", "no_training_failures"]
    observation_events = [
        change for change in curriculum_changes(store) if change["change"] == "observations_recorded"
    ]
    assert [change["iteration"] for change in observation_events] == [0, 1]
    assert observation_events[1]["observations"][0]["active"] == 0.0
    assert observation_events[1]["observations"][0]["tagged_case_ids"] == []
    assert session_stages(store).count("proposal.patch") == 1


def test_failed_arm_decision_defaults_to_pull(tmp_path: Path) -> None:
    engine, store = direct_engine(tmp_path, "decide_arm")

    engine.run()

    decision = next(
        change for change in curriculum_changes(store) if change["change"] == "arm_decision" and change["iteration"] == 1
    )
    assert decision["requested_action"] == "pull"
    assert decision["action"] == "arm_pull"
    assert "decide_arm failure" in decision["fallback_reason"]
    assert "proposal.arm_scoring" in session_stages(store)


def test_failed_arm_scoring_fails_the_run(tmp_path: Path) -> None:
    engine, store = direct_engine(tmp_path, "score_arms")

    with pytest.raises(SessionRetriesExhausted):
        engine.run()

    assert store.events()[-1].event_type == "RunFailed"
    assert len(store.events_of_type("BatchSampled")) == 1


def test_failed_pattern_extraction_is_abandoned_and_next_iteration_draws(tmp_path: Path) -> None:
    engine, store = direct_engine(tmp_path, "extract_patterns")

    engine.run()

    assert [event.payload["obligation_id"] for event in store.events_of_type("DeferredWorkAbandoned")]
    assert CurriculumState.replay(store.events()).patterns == {}
    actions = [event.payload["provenance"]["action"] for event in store.events_of_type("BatchSampled")]
    assert actions == ["unseen_draw", "unseen_draw"]
    assert "proposal.arm_decision" not in session_stages(store)


def test_runtime_rejects_scenarios_without_curriculum_session_kinds(tmp_path: Path) -> None:
    registry = default_registry()
    builtin = registry.scenarios["fake"]

    def base_only(**kwargs):
        return replace(builtin(**kwargs), supported_session_kinds=BASE_SESSION_KINDS)

    registry.scenarios["fake"] = base_only
    with pytest.raises(ValueError, match="does not support session kinds"):
        build_runtime(write_config(tmp_path, curriculum_config(tmp_path)), run_id="unsupported", registry=registry)
