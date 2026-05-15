from __future__ import annotations

import pytest

from autosaddler.v2.core.domain import ArtifactRef, Candidate, Cost, Evaluation, Observation


def artifact(uri: str = "artifacts/candidate.json") -> ArtifactRef:
    return ArtifactRef(uri=uri, kind="json", sha256="sha256:" + "a" * 64, bytes=2)


def observation(
    *,
    case_id: str,
    disposition: str = "success",
    score: float | None = 1.0,
    repetition: int = 0,
) -> Observation:
    return Observation.create(
        candidate_id="sha256:" + "b" * 64,
        case_id=case_id,
        split="train",
        repetition=repetition,
        disposition=disposition,  # type: ignore[arg-type]
        score=score,
        evaluator_fingerprint="fake:v1:config-a",
    )


def test_candidate_identity_and_parent_change_invariants() -> None:
    seed = Candidate(candidate_id="sha256:" + "b" * 64, parent_ids=(), space="component-map", artifact=artifact())
    assert seed.parent_ids == ()

    with pytest.raises(ValueError, match="content-derived"):
        Candidate(candidate_id="candidate-1", parent_ids=(), space="component-map", artifact=artifact())
    with pytest.raises(ValueError, match="change summary"):
        Candidate(
            candidate_id="sha256:" + "c" * 64,
            parent_ids=(seed.candidate_id,),
            space="component-map",
            artifact=artifact(),
        )


def test_execution_error_is_invalid_and_never_scored_zero() -> None:
    invalid = observation(case_id="case-a", disposition="execution_error", score=None)
    assert invalid.is_valid is False
    assert invalid.score is None

    with pytest.raises(ValueError, match="score=None"):
        observation(case_id="case-a", disposition="execution_error", score=0.0)


def test_evaluation_excludes_invalid_results_from_aggregate() -> None:
    valid = observation(case_id="case-a", score=1.0)
    invalid = observation(case_id="case-b", disposition="execution_error", score=None)
    evaluation = Evaluation(
        evaluation_id="eval-1",
        candidate_id=valid.candidate_id,
        split="train",
        purpose="train_before",
        iteration=1,
        requested_case_ids=("case-a", "case-b"),
        observations=(invalid, valid),
        artifact_dir=artifact("evaluations/eval-1"),
    )

    assert evaluation.aggregate_score == 1.0
    assert evaluation.attempted_rollouts == 2
    assert evaluation.valid_rollouts == 1


def test_observation_identity_binds_candidate_case_repetition_and_evaluator() -> None:
    baseline = observation(case_id="case-a")
    changed_candidate = Observation.create(
        candidate_id="sha256:" + "c" * 64,
        case_id="case-a",
        split="train",
        repetition=0,
        disposition="success",
        score=1.0,
        evaluator_fingerprint="fake:v1:config-a",
    )
    repeated = Observation.create(
        candidate_id=baseline.candidate_id,
        case_id="case-a",
        split="train",
        repetition=1,
        disposition="success",
        score=1.0,
        evaluator_fingerprint="fake:v1:config-a",
        cost=Cost(rollouts=1),
    )
    changed_evaluator = Observation.create(
        candidate_id=baseline.candidate_id,
        case_id="case-a",
        split="train",
        repetition=0,
        disposition="success",
        score=1.0,
        evaluator_fingerprint="fake:v2:config-a",
    )

    assert len(
        {
            baseline.observation_id,
            changed_candidate.observation_id,
            repeated.observation_id,
            changed_evaluator.observation_id,
        }
    ) == 4


def test_evaluation_pairs_by_identity_not_result_order() -> None:
    first = observation(case_id="case-a")
    second = observation(case_id="case-b", score=0.0, disposition="task_failure")
    evaluation = Evaluation(
        evaluation_id="eval-order",
        candidate_id=first.candidate_id,
        split="train",
        purpose="train_after",
        iteration=1,
        requested_case_ids=("case-a", "case-b"),
        observations=(second, first),
        artifact_dir=artifact("evaluations/eval-order"),
    )
    assert {(item.case_id, item.repetition) for item in evaluation.observations} == {
        ("case-a", 0),
        ("case-b", 0),
    }