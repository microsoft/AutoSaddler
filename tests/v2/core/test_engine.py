from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from autosaddler.v2.core.domain import ArtifactRef, Cost, canonical_json
from autosaddler.v2.core.engine import (
    AutoSaddlerEngine,
    SessionRetriesExhausted,
    _prepare_session_attempt_workspace,
    _selection_failure_reason,
    _selection_plan,
    _workspace_delta_failure_reason,
)
from autosaddler.v2.core.policies import (
    BudgetPolicy,
    FixedTaskSelectionPolicy,
    FullOnAcceptDevelopment,
    MatchedValidStrictImprovement,
    MeanDevelopmentRanking,
    PolicyBundle,
)
from autosaddler.v2.plugins.fake import FakePromptPack, FakeScenarioSettings, build_fake_components
from autosaddler.v2.providers.fake import FakeAgentProvider, PaidWorkLedger
from autosaddler.v2.core.ports import WorkspaceDelta
from autosaddler.v2.prompting.models import SessionResult, Usage
from autosaddler.v2.storage.local import LocalRunStore


class FailSessionKindsProvider:
    def __init__(self, ledger: PaidWorkLedger, *kinds: str) -> None:
        self.delegate = FakeAgentProvider(ledger)
        self.kinds = frozenset(kinds)
        self.calls = 0

    async def run(self, request):
        self.calls += 1
        if request.spec.kind in self.kinds:
            return SessionResult(
                status="failed",
                structured_output=None,
                raw_response="",
                tool_calls=(),
                usage=(),
                cost=Cost(sessions=1),
                error=f"persistent {request.spec.kind} failure",
            )
        return await self.delegate.run(request)


def test_prepare_source_less_session_attempt_workspace(tmp_path: Path) -> None:
    destination = tmp_path / "workspaces/.attempts/session"

    _prepare_session_attempt_workspace(None, destination)

    assert destination.is_dir()
    assert not any(destination.iterdir())


def test_selection_plan_ignores_components_sourced_from_base_parent() -> None:
    base = "sha256:base"
    result = SessionResult(
        status="completed",
        structured_output={
            "parent_ids": [base],
            "component_sources": {"instruction": base},
            "rationale": "Keep the base component.",
        },
        raw_response="{}",
        tool_calls=(),
        usage=(),
        cost=Cost(sessions=1),
    )

    plan = _selection_plan(result, (base,))

    assert plan.parents == (base,)
    assert plan.selections == {}


def test_selection_failure_reason_rejects_unused_additional_parent() -> None:
    base = "sha256:base"
    other = "sha256:other"
    result = SessionResult(
        status="completed",
        structured_output={
            "parent_ids": [base, other],
            "component_sources": {"instruction": base},
            "rationale": "The additional parent supplies nothing.",
        },
        raw_response="{}",
        tool_calls=(),
        usage=(),
        cost=Cost(sessions=1),
    )

    assert _selection_failure_reason(result, (base, other)) == (
        "Every additional composition parent must supply a component"
    )


def test_workspace_delta_failure_reason_requires_exact_declared_paths() -> None:
    result = SessionResult(
        status="completed",
        structured_output={"changed_paths": ["candidate/declared.txt"]},
        raw_response="{}",
        tool_calls=(),
        usage=(),
        cost=Cost(sessions=1),
    )
    delta = WorkspaceDelta(
        attempt_operation_id="attempt-1",
        changed_paths=("candidate/observed.txt",),
        artifact=ArtifactRef(uri="deltas/attempt-1.patch", kind="workspace-delta"),
    )

    reason = _workspace_delta_failure_reason(result, delta)

    assert reason is not None
    assert "candidate/declared.txt" in reason
    assert "candidate/observed.txt" in reason


def test_prepare_session_attempt_workspace_requires_existing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Session source workspace does not exist"):
        _prepare_session_attempt_workspace(
            tmp_path / "missing-source",
            tmp_path / "workspaces/.attempts/session",
        )


def test_prepare_session_attempt_workspace_excludes_source_provider_assets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "workspaces/.attempts/session"
    for directory in (".git", ".github", ".claude", ".copilot", ".autosaddler"):
        (source / directory).mkdir(parents=True, exist_ok=True)
    (source / ".github/copilot-instructions.md").write_text("source instructions\n")
    (source / ".claude/settings.json").write_text("{}\n")
    (source / ".copilot/instructions.md").write_text("source instructions\n")
    (source / ".autosaddler/session_output.json").write_text("{}\n")
    (source / ".autosaddler/session_output_schema.json").write_text("{}\n")
    (source / ".autosaddler/session.json").write_text("{}\n")
    (source / "AGENTS.md").write_text("source instructions\n")
    (source / "CLAUDE.md").write_text("source instructions\n")
    (source / "candidate.txt").write_text("candidate content\n")

    _prepare_session_attempt_workspace(source, destination)

    assert (destination / "candidate.txt").read_text() == "candidate content\n"
    assert (destination / ".autosaddler/session.json").is_file()
    assert not (destination / ".git").exists()
    assert not (destination / ".github").exists()
    assert not (destination / ".claude").exists()
    assert not (destination / ".copilot").exists()
    assert not (destination / "AGENTS.md").exists()
    assert not (destination / "CLAUDE.md").exists()
    assert not (destination / ".autosaddler/session_output.json").exists()
    assert not (destination / ".autosaddler/session_output_schema.json").exists()


def test_event_engine_runs_complete_fake_optimization(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="fake-run")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={
            "resolved/component_graph.json": {"scenario": "fake", "provider": "fake"},
            "resolved/policies.json": {"acceptance": "matched_valid_strict_improvement"},
        },
    )
    ledger = PaidWorkLedger(run_dir / "audit/fake_paid_work.jsonl")
    scenario = replace(
        build_fake_components(
            settings=FakeScenarioSettings(
                baseline={"instruction": "baseline"},
                target_component="instruction",
                improved_text="improved",
                train_case_ids=("train-a", "train-b"),
                development_case_ids=("dev-a", "dev-b"),
            ),
            run_dir=run_dir,
            store=store,
            ledger=ledger,
        ),
        evaluation_repetitions=3,
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=FakeAgentProvider(ledger),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
        diagnosis_patch_timeout_seconds=11,
        selection_timeout_seconds=12,
        reflection_timeout_seconds=13,
    )

    assert engine._session_timeout("diagnose_patch") == 11
    assert engine._session_timeout("evolve") == 12
    assert engine._session_timeout("reflect") == 13
    with pytest.raises(ValueError, match="Unknown session kind"):
        engine._session_timeout("other")

    result = engine.run()

    assert result.development_score == 1.0
    assert result.selected_candidate_id != engine._state().accepted_candidate_ids[0]
    completed = store.events_of_type("IterationCompleted")[0]
    assert completed.payload["schema_version"] == "autosaddler-iteration-completion/v1"
    assert completed.payload["accepted"] is True
    assert completed.payload["outcome"] == "accepted"
    assert completed.payload["resulting_candidate_id"] == result.selected_candidate_id
    assert completed.payload["evaluated_candidate_ids"][-1] == result.selected_candidate_id
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "completed"
    assert json.loads((run_dir / "evolution_dag.json").read_text())["nodes"][-1]["status"] == "selected"
    assert (run_dir / "result.json").is_file()
    rollouts = [entry for entry in ledger.entries() if entry["kind"] == "rollout"]
    assert len(rollouts) == 24
    assert len({entry["key"] for entry in rollouts}) == 24
    assert len([entry for entry in ledger.entries() if entry["kind"] == "session"]) == 3
    metrics = json.loads((run_dir / "metrics-summary.json").read_text())
    assert metrics["model_usage"]["model_calls"] == 3
    assert metrics["model_usage"]["total_tokens"] == 24
    assert metrics["model_usage_by_role"]["optimizer"]["total_tokens"] == 24
    assert metrics["attempted_rollouts"] == 24
    assert metrics["valid_rollouts"] == 24
    assert metrics["accepted_candidate_count"] == 1
    assert metrics["new_best_candidate_count"] == 1
    assert metrics["active_wall_seconds"] > 0
    assert metrics["per_accepted_candidate"]["tokens"] == 24
    reflection_start = next(
        event
        for event in store.events_of_type("SessionStarted")
        if ":deferred:" in str(event.payload["logical_operation_id"])
    )
    request = store.read_json(reflection_start.payload["request"]["uri"])
    reflection_context = json.loads(request["spec"]["workspace_files"]["session_context.json"])
    assert reflection_context["parent_candidate_id"] == engine._state().accepted_candidate_ids[0]
    assert reflection_context["working_parent_candidate_id"] == engine._state().accepted_candidate_ids[0]
    assert reflection_context["selection_parent_ids"] == [engine._state().accepted_candidate_ids[0]]
    assert reflection_context["component_sources"] == {}
    assert reflection_context["selection_rationale"]
    assert reflection_context["train_before_aggregate"] == 0.0
    assert reflection_context["train_after_aggregate"] == 1.0
    assert reflection_context["train_before_case_scores"] == {"train-a": 0.0, "train-b": 0.0}
    assert reflection_context["train_after_case_scores"] == {"train-a": 1.0, "train-b": 1.0}
    assert reflection_context["accepted_by_minibatch_gate"] is True
    assert reflection_context["parent_development_aggregate"] == 0.0
    assert reflection_context["candidate_development_aggregate"] == 1.0
    assert reflection_context["changed_components"] == ["instruction"]
    assert reflection_context["updates"] == {"instruction": "improved"}
    assert reflection_context["diagnosis"]
    diagnosis_start = next(
        event
        for event in store.events_of_type("SessionStarted")
        if str(event.payload["logical_operation_id"]).endswith(":diagnose-patch")
    )
    diagnosis_request = store.read_json(diagnosis_start.payload["request"]["uri"])
    diagnosis_context = json.loads(diagnosis_request["spec"]["workspace_files"]["session_context.json"])
    assert "selection_rationale" not in diagnosis_context
    assert "component_sources" not in diagnosis_context
    lesson_change = store.events_of_type("ExtensionStateChanged")[0]
    assert lesson_change.payload["owning_iteration"] == 0
    assert not engine._state().pending_obligations


def test_declined_child_keeps_parent_as_iteration_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="declined-child-run")
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
            train_case_ids=("train-a", "train-b"),
            development_case_ids=("dev-a",),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class DeclineImprovement(MatchedValidStrictImprovement):
        def compare(self, parent, child):
            verdict = super().compare(parent, child)
            return replace(verdict, accepted=False, reason="deliberate test decline")

    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=FakeAgentProvider(ledger),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=DeclineImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
    )

    result = engine.run()

    parent_id = engine._state().accepted_candidate_ids[0]
    finalized = store.events_of_type("CandidateFinalized")[-1]
    child_id = finalized.payload["candidate"]["candidate_id"]
    acceptance = store.events_of_type("AcceptanceDecided")[0]
    completed = store.events_of_type("IterationCompleted")[0]
    assert child_id != parent_id
    assert acceptance.payload["child_candidate_id"] == child_id
    assert acceptance.payload["verdict"]["accepted"] is False
    assert completed.payload["accepted"] is False
    assert completed.payload["schema_version"] == "autosaddler-iteration-completion/v1"
    assert completed.payload["outcome"] == "declined"
    assert completed.payload["resulting_candidate_id"] == parent_id
    assert completed.payload["evaluated_candidate_ids"] == [parent_id, child_id]
    assert result.selected_candidate_id == parent_id


def test_event_engine_recovers_each_missing_result_usage_row(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    class InterruptAfterFirstUsage:
        def __call__(self, event) -> None:
            if event.event_type == "ModelUsageObserved":
                raise RuntimeError("interrupt after first usage row")

    store = LocalRunStore(
        run_dir=run_dir,
        run_id="usage-recovery-run",
        transition_hook=InterruptAfterFirstUsage(),
    )
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
            train_case_ids=("train-a", "train-b"),
            development_case_ids=("dev-a", "dev-b"),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class TwoUsageProvider:
        def __init__(self) -> None:
            self.delegate = FakeAgentProvider(ledger)

        async def run(self, request):
            result = await self.delegate.run(request)
            usage = (
                Usage(input_tokens=2, output_tokens=1, model="fake-deterministic-v1"),
                Usage(input_tokens=4, output_tokens=3, model="fake-deterministic-v1"),
            )
            return replace(result, usage=usage, cost=Cost(sessions=1, input_tokens=6, output_tokens=4))

    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=TwoUsageProvider(),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
    )

    with pytest.raises(RuntimeError, match="interrupt after first usage row"):
        engine.run()
    first_attempt = store.events_of_type("SessionStarted")[0].operation_id
    assert list((run_dir / "sessions").glob("*/result.json"))
    assert (
        len(
            [
                event
                for event in store.events_of_type("ModelUsageObserved")
                if event.payload["attempt_operation_id"] == first_attempt
            ]
        )
        == 1
    )

    store.transition_hook = None
    resumed_engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=TwoUsageProvider(),
        policies=engine.policies,
    )
    result = resumed_engine.run()

    assert result.development_score == 1.0
    recovered = [
        event
        for event in store.events_of_type("ModelUsageObserved")
        if event.payload["attempt_operation_id"] == first_attempt
    ]
    assert [event.payload["usage_sequence"] for event in recovered] == [0, 1]
    assert len({event.payload["run_invocation_id"] for event in recovered}) == 1
    assert len([entry for entry in ledger.entries() if entry["kind"] == "session"]) == 3


def test_event_engine_records_degraded_observability_when_recovered_usage_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"

    class InterruptAfterFirstUsage:
        def __call__(self, event) -> None:
            if event.event_type == "ModelUsageObserved":
                raise RuntimeError("interrupt after first usage row")

    store = LocalRunStore(
        run_dir=run_dir,
        run_id="usage-recovery-degraded-run",
        transition_hook=InterruptAfterFirstUsage(),
    )
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
            train_case_ids=("train-a", "train-b"),
            development_case_ids=("dev-a", "dev-b"),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class TwoUsageProvider:
        def __init__(self) -> None:
            self.delegate = FakeAgentProvider(ledger)

        async def run(self, request):
            result = await self.delegate.run(request)
            usage = (
                Usage(input_tokens=2, output_tokens=1, model="fake-deterministic-v1"),
                Usage(input_tokens=4, output_tokens=3, model="fake-deterministic-v1"),
            )
            return replace(result, usage=usage, cost=Cost(sessions=1, input_tokens=6, output_tokens=4))

    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=TwoUsageProvider(),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
    )

    with pytest.raises(RuntimeError, match="interrupt after first usage row"):
        engine.run()
    first_attempt = store.events_of_type("SessionStarted")[0].operation_id
    store.transition_hook = None
    original_append = store.append
    fail_next_usage_write = True

    def append_with_one_usage_failure(event_type, operation_id, payload):
        nonlocal fail_next_usage_write
        if event_type == "ModelUsageObserved" and fail_next_usage_write:
            fail_next_usage_write = False
            raise OSError("simulated recovered usage write failure")
        return original_append(event_type, operation_id, payload)

    monkeypatch.setattr(store, "append", append_with_one_usage_failure)
    result = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=TwoUsageProvider(),
        policies=engine.policies,
    ).run()

    assert result.development_score == 1.0
    degraded = store.events_of_type("ObservabilityDegraded")
    assert len(degraded) == 1
    assert degraded[0].payload["attempt_operation_id"] == first_attempt
    assert degraded[0].payload["errors"] == ["OSError: simulated recovered usage write failure"]


def test_event_engine_settles_reflection_before_next_iteration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="deferred-order-run")
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
            train_case_ids=("train-a", "train-b", "train-c", "train-d"),
            development_case_ids=("dev-a",),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class LatestParentPromptPack(FakePromptPack):
        def session(self, kind, context):
            spec = super().session(kind, context)
            if kind != "evolve":
                return spec
            candidate_ids = context["candidate_ids"]
            if len(candidate_ids) > 1:
                assert context["component_source_options"] == {"instruction": [candidate_ids[-1]]}
            response = (
                {
                    "schema_version": "autosaddler-evolution/v1",
                    "parent_ids": [candidate_ids[0]],
                    "component_sources": {},
                    "rationale": "Only the seed candidate exists.",
                }
                if len(candidate_ids) == 1
                else {
                    "schema_version": "autosaddler-evolution/v1",
                    "parent_ids": [candidate_ids[0], candidate_ids[-1]],
                    "component_sources": {"instruction": candidate_ids[-1]},
                    "rationale": "Graft the measured instruction improvement onto the seed base.",
                }
            )
            return replace(
                spec,
                workspace_files={
                    **spec.workspace_files,
                    ".autosaddler/fake_response.json": canonical_json(response) + "\n",
                },
            )

    scenario = replace(
        scenario,
        prompt_pack=LatestParentPromptPack(target_component="instruction", improved_text="improved"),
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=FakeAgentProvider(ledger),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=2),
        ),
    )

    engine.run()

    iteration_one_started = next(
        event for event in store.events_of_type("IterationStarted") if event.payload["iteration"] == 1
    )
    iteration_zero_reflected = next(
        event
        for event in store.events_of_type("DeferredWorkCompleted")
        if event.payload["obligation_id"]
        == next(
            scheduled.payload["obligation_id"]
            for scheduled in store.events_of_type("DeferredWorkScheduled")
            if scheduled.payload["owning_iteration"] == 0
        )
    )
    assert iteration_zero_reflected.sequence < iteration_one_started.sequence
    assert not engine._state().pending_obligations


def test_event_engine_records_deliberate_empty_mutation_without_child(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="no-proposal-run")
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
            train_case_ids=("train-a", "train-b"),
            development_case_ids=("dev-a", "dev-b"),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class NoProposalPromptPack(FakePromptPack):
        def session(self, kind, context):
            spec = super().session(kind, context)
            if kind != "diagnose_patch":
                return spec
            response = {
                "schema_version": "autosaddler-diagnosis-patch/v1",
                "updates": {},
                "diagnosis": "No defensible edit exists.",
            }
            return replace(
                spec,
                workspace_files={
                    **spec.workspace_files,
                    ".autosaddler/fake_response.json": canonical_json(response) + "\n",
                },
            )

    scenario = replace(
        scenario,
        prompt_pack=NoProposalPromptPack(target_component="instruction", improved_text="improved"),
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=FakeAgentProvider(ledger),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
    )

    result = engine.run()

    assert result.development_score == 0.0
    assert len(store.events_of_type("MutationRejected")) == 1
    assert not store.events_of_type("CandidateFinalized")
    assert len([entry for entry in ledger.entries() if entry["kind"] == "rollout"]) == 4
    assert len([entry for entry in ledger.entries() if entry["kind"] == "session"]) == 2
    assert not store.events_of_type("DeferredWorkScheduled")
    assert not engine._state().pending_obligations


def test_event_engine_retries_provider_exception_with_failed_usage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="session-retry-run")
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
            train_case_ids=("train-a", "train-b"),
            development_case_ids=("dev-a", "dev-b"),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class FailOnceProvider:
        def __init__(self) -> None:
            self.delegate = FakeAgentProvider(ledger)
            self.calls = 0

        async def run(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient provider failure")
            return await self.delegate.run(request)

    provider = FailOnceProvider()
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
    )

    result = engine.run()

    assert result.development_score == 1.0
    assert provider.calls == 4
    assert len(store.events_of_type("SessionFailed")) == 1
    assert len(store.events_of_type("SessionCompleted")) == 3
    starts = store.events_of_type("SessionStarted")
    assert starts[0].operation_id.endswith(":attempt:1")
    assert starts[1].operation_id.endswith(":attempt:2")
    failed_usage = [
        event
        for event in store.events_of_type("ModelUsageObserved")
        if event.payload["attempt_operation_id"] == starts[0].operation_id
    ]
    assert len(failed_usage) == 1
    assert failed_usage[0].payload["usage"]["status"] == "failed"
    assert failed_usage[0].payload["usage"]["error_type"] == "RuntimeError"
    assert failed_usage[0].payload["usage"]["usage_incomplete"] is True
    failed_session = store.find("SessionFailed", starts[0].operation_id)
    assert failed_session is not None
    assert failed_session.payload["usage_incomplete"] is True


def test_event_engine_retries_provider_in_pristine_workspace(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="session-workspace-retry-run")
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
            train_case_ids=("train-a",),
            development_case_ids=("dev-a",),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class DirtyFailedAttemptProvider:
        def __init__(self) -> None:
            self.delegate = FakeAgentProvider(ledger)
            self.failed_diagnosis = False
            self.diagnosis_workspaces: list[Path] = []

        async def run(self, request):
            if request.spec.kind == "diagnose_patch":
                self.diagnosis_workspaces.append(request.workspace)
                candidate_path = request.workspace / "candidate.json"
                if not self.failed_diagnosis:
                    self.failed_diagnosis = True
                    candidate_path.write_text('{"instruction":"contaminated"}\n', encoding="utf-8")
                    return SessionResult(
                        status="failed",
                        structured_output=None,
                        raw_response="",
                        tool_calls=(),
                        usage=(),
                        cost=Cost(sessions=1),
                        error="transient provider failure after editing",
                    )
                assert json.loads(candidate_path.read_text(encoding="utf-8")) == {"instruction": "baseline"}
            return await self.delegate.run(request)

    provider = DirtyFailedAttemptProvider()
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=1),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
    )

    result = engine.run()

    assert result.development_score == 1.0
    assert len(provider.diagnosis_workspaces) == 2
    assert provider.diagnosis_workspaces[0] != provider.diagnosis_workspaces[1]


def test_event_engine_skips_diagnosis_when_batch_has_no_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="all-pass-run")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "fake", "provider": "fake"}},
    )
    ledger = PaidWorkLedger(run_dir / "audit/fake_paid_work.jsonl")
    scenario = build_fake_components(
        settings=FakeScenarioSettings(
            baseline={"instruction": "improved"},
            target_component="instruction",
            improved_text="improved",
            train_case_ids=("train-a", "train-b"),
            development_case_ids=("dev-a", "dev-b"),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=FakeAgentProvider(ledger),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
    )

    result = engine.run()

    assert result.development_score == 1.0
    assert len([entry for entry in ledger.entries() if entry["kind"] == "session"]) == 1
    assert len([entry for entry in ledger.entries() if entry["kind"] == "rollout"]) == 4
    assert not store.events_of_type("CandidateFinalized")
    assert not store.events_of_type("DeferredWorkScheduled")
    completed = store.events_of_type("IterationCompleted")
    assert completed[0].payload["outcome"] == "no_training_failures"
    assert completed[0].payload["resulting_candidate_id"] == engine._state().accepted_candidate_ids[0]
    assert completed[0].payload["evaluated_candidate_ids"] == [engine._state().accepted_candidate_ids[0]]


def test_event_engine_records_novel_all_pass_composition(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="composition-pass-run")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "fake", "provider": "fake"}},
    )
    ledger = PaidWorkLedger(run_dir / "audit/fake_paid_work.jsonl")
    scenario = build_fake_components(
        settings=FakeScenarioSettings(
            baseline={"instruction": "baseline", "detail": "baseline detail"},
            target_component="instruction",
            improved_text="improved",
            train_case_ids=("train-a", "train-b", "train-c", "train-d"),
            development_case_ids=("dev-a",),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class CompositionPromptPack(FakePromptPack):
        def session(self, kind, context):
            spec = super().session(kind, context)
            candidate_ids = context.get("candidate_ids", [])
            if kind == "diagnose_patch":
                response = {
                    "schema_version": "autosaddler-diagnosis-patch/v1",
                    "updates": {"instruction": "improved", "detail": "candidate detail"},
                    "diagnosis": "Both baseline components need clarification.",
                }
            elif kind == "evolve" and len(candidate_ids) > 1:
                response = {
                    "schema_version": "autosaddler-evolution/v1",
                    "parent_ids": [candidate_ids[0], candidate_ids[-1]],
                    "component_sources": {"instruction": candidate_ids[-1]},
                    "rationale": "Graft only the measured instruction improvement.",
                }
            else:
                return spec
            return replace(
                spec,
                workspace_files={
                    **spec.workspace_files,
                    ".autosaddler/fake_response.json": canonical_json(response) + "\n",
                },
            )

    scenario = replace(
        scenario,
        prompt_pack=CompositionPromptPack(target_component="instruction", improved_text="improved"),
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=FakeAgentProvider(ledger),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=2),
        ),
    )

    engine.run()

    completed = next(event for event in store.events_of_type("IterationCompleted") if event.payload["iteration"] == 1)
    state = engine._state()
    composition = next(
        event for event in store.events_of_type("CandidateFinalized") if event.payload.get("kind") == "composition"
    )
    composed_id = str(composition.payload["candidate"]["candidate_id"])
    assert completed.payload["outcome"] == "no_training_failures"
    assert completed.payload["resulting_candidate_id"] == completed.payload["selection_parent_ids"][0]
    assert completed.payload["resulting_candidate_id"] in state.accepted_candidate_ids
    assert completed.payload["evaluated_candidate_ids"] == [composed_id]
    assert composed_id in state.candidates
    assert composed_id not in state.accepted_candidate_ids
    assert state.candidates[composed_id].change.labels == ("composition",)


def test_event_engine_reports_zero_iterations_when_seed_exhausts_budget(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="seed-budget-run")
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
            train_case_ids=("train-a",),
            development_case_ids=("dev-a",),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=FakeAgentProvider(ledger),
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=1),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=1, max_iterations=5),
        ),
    )

    result = engine.run()

    assert result.iterations == 0
    assert not store.events_of_type("IterationStarted")


def test_event_engine_fails_after_evolve_retries_exhausted(tmp_path: Path) -> None:
    engine, store, provider = _engine_with_failed_session_kind(tmp_path, "evolve")

    with pytest.raises(SessionRetriesExhausted, match="persistent evolve failure"):
        engine.run()

    assert provider.calls == 3
    assert len(store.events_of_type("SessionFailed")) == 3
    assert not store.events_of_type("CandidateFinalized")
    assert not store.events_of_type("IterationCompleted")
    failed_attempts = {event.operation_id for event in store.events_of_type("SessionFailed")}
    failed_usage = [
        event
        for event in store.events_of_type("ModelUsageObserved")
        if event.payload["attempt_operation_id"] in failed_attempts
    ]
    assert len(failed_usage) == 3
    assert all(event.payload["usage"]["status"] == "failed" for event in failed_usage)


def test_event_engine_records_no_proposal_after_diagnosis_retries_exhausted(tmp_path: Path) -> None:
    engine, store, provider = _engine_with_failed_session_kind(tmp_path, "diagnose_patch")

    result = engine.run()

    assert result.development_score == 0.0
    assert provider.calls == 4
    assert len(store.events_of_type("SessionFailed")) == 3
    assert len(store.events_of_type("MutationRejected")) == 1
    assert not store.events_of_type("CandidateFinalized")
    assert not store.events_of_type("DeferredWorkScheduled")


def test_event_engine_abandons_reflection_after_retries_exhausted(tmp_path: Path) -> None:
    engine, store, provider = _engine_with_failed_session_kind(tmp_path, "reflect")

    result = engine.run()

    assert result.development_score == 1.0
    assert provider.calls == 5
    assert len(store.events_of_type("SessionFailed")) == 3
    abandoned = store.events_of_type("DeferredWorkAbandoned")
    assert len(abandoned) == 1
    assert "persistent reflect failure" in abandoned[0].payload["reason"]
    assert not store.events_of_type("ExtensionStateChanged")
    assert not engine._state().pending_obligations


def test_event_engine_retries_reflection_with_non_training_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="reflection-provenance-run")
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
            train_case_ids=("train-a",),
            development_case_ids=("dev-a",),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )

    class InvalidThenValidReflectionProvider:
        def __init__(self) -> None:
            self.delegate = FakeAgentProvider(ledger)
            self.reflection_calls = 0

        async def run(self, request):
            result = await self.delegate.run(request)
            if request.spec.kind != "reflect":
                return result
            self.reflection_calls += 1
            if self.reflection_calls != 1:
                return result
            output = dict(result.structured_output or {})
            output["lessons"] = [
                {
                    "scope": "global",
                    "statement": "Invalid development provenance.",
                    "evidence_case_ids": ["dev-a"],
                }
            ]
            return replace(result, structured_output=output, raw_response=canonical_json(output))

    provider = InvalidThenValidReflectionProvider()
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=1),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
        session_retries=1,
    )

    engine.run()

    reflection_failures = [
        event
        for event in store.events_of_type("SessionFailed")
        if event.payload.get("logical_operation_id", "").endswith(":reflect")
    ]
    reflection_completions = [
        event
        for event in store.events_of_type("SessionCompleted")
        if event.payload.get("logical_operation_id", "").endswith(":reflect")
    ]
    assert provider.reflection_calls == 2
    assert len(reflection_failures) == 1
    assert "non-training case IDs" in reflection_failures[0].payload["error"]
    assert len(reflection_completions) == 1
    assert store.events_of_type("DeferredWorkCompleted")


def _engine_with_failed_session_kind(
    tmp_path: Path,
    kind: str,
) -> tuple[AutoSaddlerEngine, LocalRunStore, FailSessionKindsProvider]:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id=f"{kind}-failure-run")
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
            train_case_ids=("train-a", "train-b"),
            development_case_ids=("dev-a", "dev-b"),
        ),
        run_dir=run_dir,
        store=store,
        ledger=ledger,
    )
    provider = FailSessionKindsProvider(ledger, kind)
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=2),
            acceptance=MatchedValidStrictImprovement(),
            development=FullOnAcceptDevelopment(),
            ranking=MeanDevelopmentRanking(),
            budget=BudgetPolicy(max_rollouts=100, max_iterations=1),
        ),
    )
    return engine, store, provider
