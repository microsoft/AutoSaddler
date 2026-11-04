from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import pytest

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Case,
    Cost,
    Evaluation,
    JsonValue,
    Observation,
    ParetoScreenParameters,
    canonical_json,
    sha256_digest,
)
from autosaddler.v2.core.engine import AutoSaddlerEngine, _validate_lesson_case_ids
from autosaddler.v2.core.policies import (
    BudgetPolicy,
    FixedTaskSelectionPolicy,
    FullOnAcceptDevelopment,
    MatchedValidStrictImprovement,
    MeanDevelopmentRanking,
    ParetoAcceptance,
    ParetoDevelopmentRanking,
    PolicyBundle,
)
from autosaddler.v2.core.ports import EvaluationContext, ScenarioComponents
from autosaddler.v2.harness.git import GitHarnessSpace
from autosaddler.v2.prompting.models import SessionRequest, SessionResult, SessionSpec
from autosaddler.v2.storage.local import LocalRunStore
from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack
from autosaddler.v2.plugins.meta_are.verification import MetaAREVerifier
from autosaddler.v2.prompting.history import build_history_bundle, HISTORY_ROOT


class InterruptAfterDiagnosisCompleted:
    def __init__(self) -> None:
        self.triggered = False

    def __call__(self, event) -> None:
        if event.event_type == "SessionCompleted" and str(event.payload.get("logical_operation_id", "")).endswith(
            ":diagnose-patch"
        ):
            self.triggered = True
            raise RuntimeError("interrupt after durable diagnosis result")


class InterruptAfterDiagnosisFailureBudget:
    def __init__(self) -> None:
        self.triggered = False

    def __call__(self, event) -> None:
        if event.event_type == "BudgetUpdated" and str(event.payload.get("logical_operation_id", "")).endswith(
            ":diagnose-patch:attempt:1"
        ):
            self.triggered = True
            raise RuntimeError("interrupt after failed diagnosis budget")


class EditingProvider:
    def __init__(self, edit_path: str = "candidate/instructions.txt") -> None:
        self.calls: list[str] = []
        self.edit_path = edit_path

    async def run(self, request: SessionRequest) -> SessionResult:
        self.calls.append(request.spec.kind)
        context = json.loads(request.spec.workspace_files[".autosaddler/session_context.json"])
        if request.spec.kind == "evolve":
            output: Mapping[str, JsonValue] = {
                "schema_version": "autosaddler-evolution/v1",
                "parent_ids": [context["candidate_ids"][0]],
                "component_sources": {},
                "rationale": "Use the seed.",
            }
        elif request.spec.kind == "diagnose_patch":
            path = request.workspace / self.edit_path
            path.write_text("improved\n", encoding="utf-8")
            output = {
                "schema_version": "autosaddler-diagnosis-patch/v1",
                "intent": "Improve task behavior.",
                "rationale": "The baseline is incomplete.",
                "expected_effect": "The fixture evaluator passes.",
                "changed_paths": [self.edit_path],
            }
        elif request.spec.kind == "reflect":
            output = {
                "schema_version": "autosaddler-reflection/v1",
                "lessons": [
                    {
                        "scope": "component",
                        "statement": "The direct edit improved the matched case.",
                        "evidence_case_ids": context["train_case_ids"],
                    }
                ],
            }
        else:
            raise AssertionError(request.spec.kind)
        return SessionResult(
            status="completed",
            structured_output=output,
            raw_response=canonical_json(output),
            tool_calls=(),
            usage=(),
            cost=Cost(sessions=1),
        )


class MisdeclaringOnceProvider(EditingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.diagnosis_attempts = 0

    async def run(self, request: SessionRequest) -> SessionResult:
        result = await super().run(request)
        if request.spec.kind != "diagnose_patch":
            return result
        self.diagnosis_attempts += 1
        if self.diagnosis_attempts > 1:
            return result
        assert isinstance(result.structured_output, Mapping)
        output = {**result.structured_output, "changed_paths": ["candidate/wrong.txt"]}
        return SessionResult(
            status=result.status,
            structured_output=output,
            raw_response=canonical_json(output),
            tool_calls=result.tool_calls,
            usage=result.usage,
            cost=result.cost,
            error=result.error,
        )


class GitFixtureEvaluator:
    def __init__(self, harness: GitHarnessSpace) -> None:
        self.harness = harness
        self.calls = 0

    async def evaluate(self, candidate, cases, context: EvaluationContext) -> Evaluation:
        self.calls += 1
        materialized = self.harness.materialize(candidate, "evaluate")
        try:
            improved = (materialized.root / "candidate/instructions.txt").read_text() == "improved\n"
        finally:
            materialized.release()
        observations = []
        for case in cases:
            for repetition in range(context.repetitions):
                cached = context.attempt_sink.completed(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                if cached is not None:
                    observations.append(cached)
                    continue
                attempt_id, attempt_number = context.attempt_sink.start(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                observation = Observation.create(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    split=case.split,
                    repetition=repetition,
                    disposition="success" if improved else "task_failure",
                    score=1.0 if improved else 0.0,
                    evaluator_fingerprint="git-fixture/v1",
                    attempts=attempt_number,
                )
                context.attempt_sink.complete(attempt_id, observation, Cost(rollouts=1))
                observations.append(observation)
        return Evaluation(
            evaluation_id=sha256_digest(context.operation_id),
            candidate_id=candidate.candidate_id,
            split=context.split,
            purpose=context.purpose,
            iteration=context.iteration,
            requested_case_ids=tuple(case.case_id for case in cases),
            observations=tuple(observations),
            artifact_dir=ArtifactRef(
                uri=f"evaluations/{sha256_digest(context.operation_id)[7:]}", kind="evaluation-directory"
            ),
        )


class GitFixtureEvidence:
    def __init__(self, store: LocalRunStore) -> None:
        self.store = store

    def build(self, evaluation: Evaluation) -> ArtifactRef:
        return self.store.write_json(
            "evidence/git-fixture/evidence.json",
            {"evaluation_id": evaluation.evaluation_id},
            kind="training-evidence",
        )


class GitFixturePromptPack:
    def session(self, kind: str, context: Mapping[str, JsonValue]) -> SessionSpec:
        if kind == "evolve":
            required = ("schema_version", "parent_ids", "component_sources", "rationale")
        elif kind == "diagnose_patch":
            required = ("schema_version", "intent", "rationale", "expected_effect", "changed_paths")
        elif kind == "reflect":
            required = ("schema_version", "lessons")
        else:
            raise AssertionError(kind)
        return SessionSpec(
            kind=kind,
            system_context="Use the shared engine contract.",
            task_prompt=f"Run {kind}.",
            skills={
                "fixture": (
                    "---\n"
                    "name: fixture\n"
                    'description: "Exercise the Git mutation engine fixture."\n'
                    "---\n\n"
                    "# Fixture\n"
                )
            },
            output_schema={
                "type": "object",
                "required": list(required),
                "properties": {name: {} for name in required},
                "additionalProperties": False,
            },
            workspace_files={".autosaddler/session_context.json": canonical_json(context) + "\n"},
            capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
        )


def test_git_edit_survives_session_completion_interruption_without_provider_rerun(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    interruption = InterruptAfterDiagnosisCompleted()
    store = LocalRunStore(run_dir=run_dir, run_id="git-resume", transition_hook=interruption)
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "git-fixture"}},
    )
    harness = _harness(tmp_path, run_dir)
    provider = EditingProvider()
    scenario = ScenarioComponents(
        name="git-fixture",
        version="1",
        harness_space=harness,
        evaluator=GitFixtureEvaluator(harness),
        evidence_builder=GitFixtureEvidence(store),
        prompt_pack=GitFixturePromptPack(),
        train_cases=(Case("train-a", "train", {}),),
        development_cases=(Case("dev-a", "development", {}),),
        required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=_policies(),
    )

    with pytest.raises(RuntimeError, match="interrupt after durable diagnosis result"):
        engine.run()
    assert interruption.triggered
    assert provider.calls == ["evolve", "diagnose_patch"]
    assert len(store.events_of_type("SessionCompleted")) == 2

    store.transition_hook = None
    resumed = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=_policies(),
    )
    result = resumed.run()

    assert result.development_score == 1.0
    assert provider.calls == ["evolve", "diagnose_patch", "reflect"]
    assert len(store.events_of_type("CandidateFinalized")) == 1
    candidate = resumed._state().candidates[result.selected_candidate_id]
    materialized = harness.materialize(candidate, "inspect")
    try:
        assert (materialized.root / "candidate/instructions.txt").read_text() == "improved\n"
    finally:
        materialized.release()


def test_workspace_delta_mismatch_retries_diagnosis_in_pristine_workspace(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="git-delta-retry")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "git-fixture"}},
    )
    harness = _harness(tmp_path, run_dir)
    provider = MisdeclaringOnceProvider()
    scenario = ScenarioComponents(
        name="git-fixture",
        version="1",
        harness_space=harness,
        evaluator=GitFixtureEvaluator(harness),
        evidence_builder=GitFixtureEvidence(store),
        prompt_pack=GitFixturePromptPack(),
        train_cases=(Case("train-a", "train", {}),),
        development_cases=(Case("dev-a", "development", {}),),
        required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=_policies(),
    )

    result = engine.run()

    assert result.development_score == 1.0
    assert provider.calls == ["evolve", "diagnose_patch", "diagnose_patch", "reflect"]
    failed = store.events_of_type("SessionFailed")
    assert len(failed) == 1
    assert "changed_paths do not match workspace delta" in failed[0].payload["error"]
    diagnosis_starts = [
        event
        for event in store.events_of_type("SessionStarted")
        if str(event.payload.get("logical_operation_id", "")).endswith(":diagnose-patch")
    ]
    assert len(diagnosis_starts) == 2
    assert diagnosis_starts[0].payload["session_id"] != diagnosis_starts[1].payload["session_id"]


def test_workspace_delta_mismatch_recovers_result_and_failed_budget(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    interruption = InterruptAfterDiagnosisFailureBudget()
    store = LocalRunStore(run_dir=run_dir, run_id="git-delta-budget-resume", transition_hook=interruption)
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "git-fixture"}},
    )
    harness = _harness(tmp_path, run_dir)
    provider = MisdeclaringOnceProvider()
    scenario = ScenarioComponents(
        name="git-fixture",
        version="1",
        harness_space=harness,
        evaluator=GitFixtureEvaluator(harness),
        evidence_builder=GitFixtureEvidence(store),
        prompt_pack=GitFixturePromptPack(),
        train_cases=(Case("train-a", "train", {}),),
        development_cases=(Case("dev-a", "development", {}),),
        required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
    )

    with pytest.raises(RuntimeError, match="interrupt after failed diagnosis budget"):
        AutoSaddlerEngine(store=store, scenario=scenario, provider=provider, policies=_policies()).run()
    assert interruption.triggered
    assert not [
        event
        for event in store.events_of_type("SessionFailed")
        if str(event.payload.get("logical_operation_id", "")).endswith(":diagnose-patch")
    ]

    store.transition_hook = None
    result = AutoSaddlerEngine(store=store, scenario=scenario, provider=provider, policies=_policies()).run()

    assert result.development_score == 1.0
    assert provider.calls == ["evolve", "diagnose_patch", "diagnose_patch", "reflect"]
    diagnosis_budget_ids = [
        str(event.payload.get("logical_operation_id", ""))
        for event in store.events_of_type("BudgetUpdated")
        if ":diagnose-patch" in str(event.payload.get("logical_operation_id", ""))
    ]
    assert diagnosis_budget_ids == [
        "git-delta-budget-resume:iteration:0:diagnose-patch:attempt:1",
        "git-delta-budget-resume:iteration:0:diagnose-patch",
    ]


def test_protected_provider_edit_completes_as_rejected_mutation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="git-protected-edit")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "git-fixture"}},
    )
    harness = _harness(tmp_path, run_dir)
    provider = EditingProvider(edit_path="benchmark.py")
    scenario = ScenarioComponents(
        name="git-fixture",
        version="1",
        harness_space=harness,
        evaluator=GitFixtureEvaluator(harness),
        evidence_builder=GitFixtureEvidence(store),
        prompt_pack=GitFixturePromptPack(),
        train_cases=(Case("train-a", "train", {}),),
        development_cases=(Case("dev-a", "development", {}),),
        required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
    )
    engine = AutoSaddlerEngine(
        store=store,
        scenario=scenario,
        provider=provider,
        policies=_policies(),
    )

    result = engine.run()

    assert result.development_score == 0.0
    assert provider.calls == ["evolve", "diagnose_patch"]
    assert len(store.events_of_type("SessionCompleted")) == 2
    rejected = store.events_of_type("MutationRejected")
    assert len(rejected) == 1
    assert "benchmark.py" in rejected[0].payload["verification_failure"]
    assert not store.events_of_type("CandidateFinalized")
    completed = store.events_of_type("IterationCompleted")[0]
    assert completed.payload["schema_version"] == "autosaddler-iteration-completion/v1"
    assert completed.payload["outcome"] == "mutation_rejected"
    assert completed.payload["resulting_candidate_id"] == result.selected_candidate_id
    assert completed.payload["evaluated_candidate_ids"] == [result.selected_candidate_id]


def test_reflection_lessons_may_reference_only_owning_train_cases() -> None:
    valid = [
        {
            "scope": "case",
            "statement": "The matched change improved this case.",
            "evidence_case_ids": ["train-a"],
        }
    ]
    _validate_lesson_case_ids(valid, ("train-a", "train-b"))

    invalid = [
        {
            "scope": "case",
            "statement": "Unsupported development attribution.",
            "evidence_case_ids": ["dev-a"],
        }
    ]
    with pytest.raises(ValueError, match="non-training case IDs.*dev-a"):
        _validate_lesson_case_ids(invalid, ("train-a", "train-b"))


def _harness(tmp_path: Path, run_dir: Path) -> GitHarnessSpace:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "candidate").mkdir()
    (repo / "candidate/instructions.txt").write_text("baseline\n", encoding="utf-8")
    (repo / "benchmark.py").write_text("fixed\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=repo,
        check=True,
    )
    return GitHarnessSpace(
        source_repo=repo,
        base_revision="HEAD",
        store_root=run_dir / "candidates",
        worktree_root=run_dir / "worktrees",
        writable_paths=(PurePosixPath("candidate"),),
        forbidden_paths=(PurePosixPath("benchmark.py"),),
    )


def _policies() -> PolicyBundle:
    return PolicyBundle(
        task_selection=FixedTaskSelectionPolicy(batch_size=1),
        acceptance=MatchedValidStrictImprovement(),
        development=FullOnAcceptDevelopment(),
        ranking=MeanDevelopmentRanking(),
        budget=BudgetPolicy(max_rollouts=20, max_iterations=1),
    )


class PreparationProvider:
    def __init__(self, behavior="edit"):
        self.behavior = behavior
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        context = json.loads(request.spec.workspace_files[".autosaddler/session_context.json"])
        if request.spec.kind == "evolve":
            if self.behavior in {"edit", "failed_edit", "exception_edit", "invalid_base", "protected", "python", "interrupted_edit", "revert"}:
                path = "benchmark.py" if self.behavior == "protected" else (
                    "candidate/new.py" if self.behavior == "python" else "candidate/instructions.txt"
                )
                (request.workspace / path).write_text("# prepared\n" if path.endswith(".py") else "baseline\n" if self.behavior == "revert" else "prepared\n")
                changed = [path]
            else:
                changed = []
            if self.behavior in {"interrupted_edit", "interrupted_noop"}:
                raise AbruptPreparationStop()
            if self.behavior == "exception_edit":
                raise RuntimeError("provider failed after edit")
            output = {
                "schema_version": "autosaddler-meta-are-evolution/v1",
                "parent_ids": [context["fixed_current_candidate_id"]], "component_sources": {},
                "changed_paths": changed, "donor_candidate_ids": context["donor_candidate_ids"] if changed else [],
                "rationale": "Transfer a focused archive mechanism." if changed else "Keep the current fork.",
            }
            if self.behavior == "invalid_base":
                output["parent_ids"] = [context["donor_candidate_ids"][0]]
            if self.behavior in {"failed_edit", "failed_unchanged"}:
                return SessionResult(status="failed", structured_output=None, raw_response="", tool_calls=(),
                                     usage=(), cost=Cost(sessions=1), error="preparation failed")
        elif request.spec.kind == "diagnose_patch":
            (request.workspace / "candidate/instructions.txt").write_text(
                "improved\n" if context["iteration"] == 0 else "patched\n",
            )
            output = {
                "schema_version": "autosaddler-meta-are-diagnosis/v1", "intent": "Remove wasted work.",
                "diagnosis": "Unnecessary steps.", "expected_effect": "Preserve quality at lower cost.",
                "changed_paths": ["candidate/instructions.txt"],
            }
        else:
            output = {"schema_version": "autosaddler-meta-are-reflection/v1", "lessons": []}
        return SessionResult(status="completed", structured_output=output, raw_response=canonical_json(output),
                             tool_calls=(), usage=(), cost=Cost(sessions=1))


def make_git_preparation_engine(tmp_path, behavior="edit", capability_iterations=25):
    run_dir = tmp_path / "run"
    store = LocalRunStore(run_dir=run_dir, run_id="preparation")
    store.initialize(resolved_config={}, resolved_entities={})
    harness = _harness(tmp_path, run_dir)
    harness.verifier = MetaAREVerifier(import_check="", verification_timeout_seconds=1, train_case_ids=("train-a",))
    delegate = GitFixtureEvaluator(harness)
    seen = []

    class PricedEvaluator:
        async def evaluate(self, candidate, cases, context):
            materialized = harness.materialize(candidate, "inspect")
            try:
                content = (materialized.root / "candidate/instructions.txt").read_text().strip()
            finally:
                materialized.release()
            seen.append((context.iteration, context.purpose, candidate.candidate_id, content))
            measured = await delegate.evaluate(candidate, cases, context)
            cost = {"baseline": 4.0, "improved": 3.0, "prepared": 2.0, "patched": 1.0}[content]
            score = 1.0 if content == "baseline" else 0.95
            return replace(measured, observations=tuple(
                replace(item, score=score, disposition="success", cost=Cost(rollouts=1, currency_amount=cost))
                for item in measured.observations
            ))

    class Evidence:
        def build(self, evaluation):
            return store.write_json("evidence/preparation.json", {
                "schema_version": "autosaddler-meta-are-evidence/v1", "case_records": [],
            }, kind="training-evidence")

    provider = PreparationProvider(behavior)
    scenario = ScenarioComponents(
        name="preparation-fixture", version="1", harness_space=harness, evaluator=PricedEvaluator(),
        evidence_builder=Evidence(), prompt_pack=MetaAREPromptPack(
            store=store, writable_paths=(PurePosixPath("candidate"),),
            capability_phase_iterations=capability_iterations,
        ), train_cases=(Case("train-a", "train", {}),), development_cases=(Case("dev-a", "development", {}),),
        required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
    )
    engine = AutoSaddlerEngine(store=store, scenario=scenario, provider=provider, session_retries=1,
                             session_retry_backoff_seconds=0,
                             policies=PolicyBundle(
                                 task_selection=FixedTaskSelectionPolicy(batch_size=1),
                                 acceptance=ParetoAcceptance(ParetoScreenParameters(1, draws=100)),
                                 development=FullOnAcceptDevelopment(), ranking=ParetoDevelopmentRanking(1),
                                 budget=BudgetPolicy(max_rollouts=100, max_iterations=2),
                             ))
    return engine, provider, seen


@pytest.mark.parametrize("behavior,expected", [("edit", "prepared"), ("noop", "unchanged"),
                                               ("failed_unchanged", "failed_unchanged")])
def test_pareto_preparation_uses_exact_content_and_keeps_base_lineage(tmp_path, behavior, expected):
    engine, provider, seen = make_git_preparation_engine(tmp_path, behavior)
    engine.run()
    preparations = preparation_events(engine)
    assert len(preparations) == 1 and preparations[0].payload["iteration"] == 1
    preparation = preparations[0].payload
    assert preparation["status"] == expected
    before = next(item for item in seen if item[:2] == (1, "train_before"))
    assert before[2] == preparation["prepared_candidate_id"]
    assert before[3] == ("prepared" if behavior == "edit" else "improved")
    assert next(item for item in seen if item[:2] == (1, "train_after"))[3] == "patched"
    evolve = next(request for request in provider.requests if request.spec.kind == "evolve")
    context = json.loads(evolve.spec.workspace_files[".autosaddler/session_context.json"])
    assert context["candidate_ids"] == [preparation["base_candidate_id"]]
    assert "dev-a" not in "\n".join(evolve.spec.workspace_files.values())
    state = engine._state()
    if behavior == "edit":
        prepared = state.candidates[preparation["prepared_candidate_id"]]
        assert prepared.parent_ids == (preparation["base_candidate_id"],)
        decision = engine.store.events_of_type("AcceptanceDecided")[-1].payload
        assert state.candidates[decision["candidate_id"]].parent_ids == (prepared.candidate_id,)
        assert preparation["donor_candidate_ids"] == context["donor_candidate_ids"]
    else:
        assert preparation["prepared_candidate_id"] == preparation["base_candidate_id"]
        assert not any(event.payload.get("kind") == "preparation"
                       for event in engine.store.events_of_type("CandidateFinalized"))
    history = build_history_bundle(engine.store, {})
    iteration = json.loads(history.workspace_files[f"{HISTORY_ROOT}/iterations/0001.json"])
    assert iteration["preparation"] == dict(preparation)
    dag = engine.store.read_json("evolution_dag.json")
    nodes = {node["candidate_id"]: node for node in dag["nodes"]}
    if behavior == "edit":
        assert nodes[preparation["prepared_candidate_id"]]["parent_ids"] == [preparation["base_candidate_id"]]
    assert engine.store.events_of_type("DeferredWorkScheduled")[-1].payload["preparation"] == preparation


@pytest.mark.parametrize("behavior", ["failed_edit", "exception_edit", "invalid_base", "protected"])
def test_failed_or_invalid_preparation_never_changes_current(tmp_path, behavior):
    engine, provider, seen = make_git_preparation_engine(tmp_path, behavior)
    engine.run()
    preparation = preparation_events(engine)[0].payload
    assert preparation["status"] == "rejected" and preparation["prepared_candidate_id"] is None
    assert not any(item[0] == 1 for item in seen)
    assert engine._state().current_candidate_id == preparation["base_candidate_id"]
    assert sum(request.spec.kind == "evolve" for request in provider.requests) == 1
    assert engine.store.events_of_type("IterationCompleted")[-1].payload["outcome"] == "mutation_rejected"


def test_preparation_preserves_steering_verification(tmp_path):
    engine, _, seen = make_git_preparation_engine(tmp_path, "python", capability_iterations=1)
    engine.run()
    preparation = preparation_events(engine)[0].payload
    assert preparation["status"] == "rejected"
    assert "steering_scope" in preparation["rationale"]
    assert not any(item[0] == 1 for item in seen)


@pytest.mark.parametrize("behavior,boundary", [
    ("edit", "SessionCompleted"), ("edit", "CandidateFinalized"), ("edit", "ExtensionStateChanged"),
    ("noop", "ExtensionStateChanged"), ("failed_edit", "SessionFailed"),
    ("failed_unchanged", "SessionFailed"), ("protected", "ExtensionStateChanged"),
])
def test_preparation_recovery_reuses_completed_edits_and_failure_decisions(tmp_path, behavior, boundary):
    engine, provider, seen = make_git_preparation_engine(tmp_path, behavior)

    def interrupt(event):
        if event.event_type != boundary:
            return
        if boundary == "ExtensionStateChanged" and event.payload.get("namespace") != "autosaddler.preparation":
            return
        preparation = (
            event.payload.get("kind") == "preparation" if boundary == "CandidateFinalized"
            else str(event.payload.get("logical_operation_id", "")).endswith(":preparation")
            if boundary in {"SessionCompleted", "SessionFailed"} else True
        )
        if preparation:
            raise RuntimeError("interrupt preparation boundary")

    engine.store.transition_hook = interrupt
    with pytest.raises(RuntimeError, match="interrupt preparation boundary"):
        engine.run()
    completed_evaluations = list(seen)
    prep_calls = sum(request.spec.kind == "evolve" for request in provider.requests)
    engine.store.transition_hook = None
    engine.run()
    assert seen[:len(completed_evaluations)] == completed_evaluations
    # Only an unchanged failure may consume its remaining retry on resume.
    expected_calls = 2 if behavior == "failed_unchanged" else prep_calls
    assert sum(request.spec.kind == "evolve" for request in provider.requests) == expected_calls
    assert len(preparation_events(engine)) == 1
    assert len([item for item in seen if item[:2] == (0, "train_before")]) == 1
    assert len([item for item in seen if item[:2] == (1, "train_before")]) == (behavior not in {"failed_edit", "protected"})
    engine.store.validate_integrity()


class AbruptPreparationStop(BaseException):
    pass


@pytest.mark.parametrize("behavior,expected", [("interrupted_edit", "rejected"),
                                               ("interrupted_noop", "unchanged")])
def test_preparation_recovers_attempt_without_result_artifact(tmp_path, behavior, expected):
    engine, provider, seen = make_git_preparation_engine(tmp_path, behavior)
    with pytest.raises(AbruptPreparationStop):
        engine.run()
    provider.behavior = "noop"
    engine.run()
    preparation = preparation_events(engine)[0].payload
    assert preparation["status"] == expected
    assert any(item[:2] == (1, "train_before") for item in seen) is (expected == "unchanged")
    assert sum(request.spec.kind == "evolve" for request in provider.requests) == (1 if expected == "rejected" else 2)


@pytest.mark.parametrize("behavior", ["edit", "failed_edit", "failed_unchanged"])
def test_preparation_recovers_result_artifact_before_terminal_event(tmp_path, behavior, monkeypatch):
    engine, provider, seen = make_git_preparation_engine(tmp_path, behavior)
    write = engine.store.write_json

    def interrupt_result(path, value, **kwargs):
        result = write(path, value, **kwargs)
        if str(path).endswith("/result.json") and provider.requests[-1].spec.kind == "evolve":
            raise RuntimeError("interrupt durable result artifact")
        return result

    monkeypatch.setattr(engine.store, "write_json", interrupt_result)
    with pytest.raises(RuntimeError, match="interrupt durable result artifact"):
        engine.run()
    monkeypatch.setattr(engine.store, "write_json", write)
    engine.run()
    preparation = preparation_events(engine)[0].payload
    assert preparation["status"] == {"edit": "prepared", "failed_edit": "rejected",
                                     "failed_unchanged": "failed_unchanged"}[behavior]
    assert sum(request.spec.kind == "evolve" for request in provider.requests) == (2 if behavior == "failed_unchanged" else 1)
    assert len([item for item in seen if item[:2] == (0, "train_before")]) == 1


def test_preparation_repeated_content_preserves_original_identity_and_order(tmp_path):
    engine, _, seen = make_git_preparation_engine(tmp_path, "revert")
    engine.run()
    preparation = preparation_events(engine)[0].payload
    seed_id = engine.store.events_of_type("ParetoArchiveUpdated")[0].payload["current_candidate_id"]
    assert preparation["prepared_candidate_id"] == seed_id
    assert next(item for item in seen if item[:2] == (1, "train_before"))[3] == "baseline"
    assert list(engine._state().candidates)[0] == seed_id
    assert engine._state().candidates[seed_id].parent_ids == ()


def preparation_events(engine):
    return tuple(event for event in engine.store.events_of_type("ExtensionStateChanged")
                 if event.payload.get("namespace") == "autosaddler.preparation")
