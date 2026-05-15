from __future__ import annotations

import json
import subprocess
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
    PolicyBundle,
)
from autosaddler.v2.core.ports import EvaluationContext, ScenarioComponents
from autosaddler.v2.harness.git import GitHarnessSpace
from autosaddler.v2.prompting.models import SessionRequest, SessionResult, SessionSpec
from autosaddler.v2.storage.local import LocalRunStore


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
