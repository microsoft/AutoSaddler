from __future__ import annotations

import pytest

from autosaddler.v2.core.domain import ArtifactRef, Candidate, Case, Evaluation, Observation
from autosaddler.v2.core.policies import MatchedValidStrictImprovement, MeanDevelopmentRanking

CANDIDATE = "sha256:" + "a" * 64


def observation(case_id: str, score: float | None, disposition: str, candidate_id: str) -> Observation:
    return Observation.create(
        candidate_id=candidate_id,
        case_id=case_id,
        split="train",
        repetition=0,
        disposition=disposition,  # type: ignore[arg-type]
        score=score,
        evaluator_fingerprint="fake:v1",
    )


def evaluation(candidate_id: str, values: tuple[Observation, ...], purpose: str) -> Evaluation:
    return Evaluation(
        evaluation_id=f"eval-{purpose}-{candidate_id[-1]}",
        candidate_id=candidate_id,
        split="train",
        purpose=purpose,  # type: ignore[arg-type]
        iteration=0,
        requested_case_ids=("case-a", "case-b", "case-c"),
        observations=values,
        artifact_dir=ArtifactRef(uri=f"evaluations/{purpose}", kind="evaluation"),
    )


def test_matched_policy_pairs_by_identity_and_excludes_invalid_results() -> None:
    child_id = "sha256:" + "b" * 64
    parent = evaluation(
        CANDIDATE,
        (
            observation("case-a", 0.0, "task_failure", CANDIDATE),
            observation("case-b", 1.0, "success", CANDIDATE),
            observation("case-c", None, "execution_error", CANDIDATE),
        ),
        "train_before",
    )
    child = evaluation(
        child_id,
        (
            observation("case-b", 1.0, "success", child_id),
            observation("case-c", 1.0, "success", child_id),
            observation("case-a", 1.0, "success", child_id),
        ),
        "train_after",
    )

    verdict = MatchedValidStrictImprovement().compare(parent, child)

    assert verdict.compared_case_ids == ("case-a", "case-b")
    assert verdict.before_score == 0.5
    assert verdict.after_score == 1.0
    assert verdict.accepted


def test_matched_policy_rejects_different_batches() -> None:
    child_id = "sha256:" + "b" * 64
    parent = evaluation(
        CANDIDATE,
        tuple(observation(case, 0.0, "task_failure", CANDIDATE) for case in ("case-a", "case-b", "case-c")),
        "train_before",
    )
    child = Evaluation(
        evaluation_id="different",
        candidate_id=child_id,
        split="train",
        purpose="train_after",
        iteration=0,
        requested_case_ids=("case-a",),
        observations=(observation("case-a", 1.0, "success", child_id),),
        artifact_dir=ArtifactRef(uri="evaluations/different", kind="evaluation"),
    )

    with pytest.raises(ValueError, match="identical ordered case set"):
        MatchedValidStrictImprovement().compare(parent, child)


def test_development_ranking_prefers_latest_candidate_on_tie() -> None:
    candidate_ids = ("sha256:" + "a" * 64, "sha256:" + "b" * 64)
    ranked = []
    for iteration, candidate_id in enumerate(candidate_ids):
        candidate = Candidate(
            candidate_id=candidate_id,
            parent_ids=(),
            space="fake",
            artifact=ArtifactRef(uri=f"candidates/{candidate_id[-1]}", kind="candidate"),
        )
        evaluation = Evaluation(
            evaluation_id=f"dev-{iteration}",
            candidate_id=candidate_id,
            split="development",
            purpose="development",
            iteration=iteration,
            requested_case_ids=("dev-a",),
            observations=(
                Observation.create(
                    candidate_id=candidate_id,
                    case_id="dev-a",
                    split="development",
                    repetition=0,
                    disposition="success",
                    score=1.0,
                    evaluator_fingerprint="fake:v1",
                ),
            ),
            artifact_dir=ArtifactRef(uri=f"evaluations/dev-{iteration}", kind="evaluation"),
        )
        ranked.append((candidate, evaluation))

    selected, _ = MeanDevelopmentRanking().select(ranked)

    assert selected.candidate_id == candidate_ids[-1]


def test_epoch_shuffled_is_seeded_replayable_without_consuming_boundary_duplicates() -> None:
    from autosaddler.v2.core.policies import EpochShuffledTaskSelectionPolicy

    cases = tuple(Case(f"case-{index}", "train", {}) for index in range(5))
    first = EpochShuffledTaskSelectionPolicy(batch_size=3, seed=17)
    replay = EpochShuffledTaskSelectionPolicy(batch_size=3, seed=17)

    selections = [first.select(cases, iteration) for iteration in range(4)]
    replayed = [replay.select(cases, iteration) for iteration in range(4)]

    assert selections == replayed
    assert all(len(selection.case_ids) == 3 for selection in selections)
    assert all(len(set(selection.case_ids)) == 3 for selection in selections)
    assert selections[0].provenance["policy"] == "epoch_shuffled"
    assert selections[0].provenance["seed"] == 17
    assert selections[0].provenance["epoch"] == 0
    assert "epoch_order" in selections[0].provenance
    consumed_by_epoch: dict[int, list[str]] = {}
    for selection in selections:
        for segment in selection.provenance["epoch_segments"]:
            consumed_by_epoch.setdefault(segment["epoch"], []).extend(segment["case_ids"])
    assert consumed_by_epoch[0] == selections[0].provenance["epoch_order"]
    assert set(consumed_by_epoch[1]) == {case.case_id for case in cases}
    assert len(consumed_by_epoch[1]) == len(cases)


def test_epoch_shuffled_uses_complete_small_training_set_without_duplicates() -> None:
    from autosaddler.v2.core.policies import EpochShuffledTaskSelectionPolicy

    cases = tuple(Case(f"case-{index}", "train", {}) for index in range(2))
    selection = EpochShuffledTaskSelectionPolicy(batch_size=4, seed=3).select(cases, 0)

    assert set(selection.case_ids) == {"case-0", "case-1"}
    assert len(selection.case_ids) == 2