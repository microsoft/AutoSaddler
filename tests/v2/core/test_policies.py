from __future__ import annotations

from dataclasses import replace

import pytest

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Candidate,
    Case,
    Cost,
    Evaluation,
    Observation,
    ParetoScreenParameters,
    PatchVerdict,
)
from autosaddler.v2.core.policies import (
    FullOnAcceptDevelopment,
    MatchedValidStrictImprovement,
    MeanDevelopmentRanking,
    ParetoAcceptance,
    ParetoDevelopmentRanking,
    dominates,
    pareto_front,
)

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


def objective_evaluation(
    candidate_id: str,
    runs: tuple[tuple[tuple[float, float], ...], ...],
    purpose: str = "train_before",
) -> Evaluation:
    split = "development" if purpose == "development" else "train"
    case_ids = tuple(f"case-{index}" for index in range(len(runs)))
    return Evaluation(
        evaluation_id=f"{purpose}-{candidate_id}",
        candidate_id=candidate_id,
        split=split,
        purpose=purpose,
        iteration=0,
        requested_case_ids=case_ids,
        observations=tuple(
            Observation.create(
                candidate_id=candidate_id,
                case_id=case_id,
                split=split,
                repetition=repetition,
                disposition="success" if accuracy == 1.0 else "task_failure",
                score=accuracy,
                cost=Cost(rollouts=1, currency_amount=cost),
                evaluator_fingerprint="pareto-fixture/v1",
            )
            for case_id, task in zip(case_ids, runs, strict=True)
            for repetition, (accuracy, cost) in enumerate(task)
        ),
        artifact_dir=ArtifactRef(uri=f"evaluations/{purpose}", kind="evaluation"),
    )


def development_candidate(letter: str, accuracy: float, cost: float) -> tuple[Candidate, Evaluation]:
    candidate_id = "sha256:" + letter * 64
    return (
        Candidate(candidate_id, (), "fake", ArtifactRef(uri=f"candidates/{letter}", kind="candidate")),
        objective_evaluation(candidate_id, (((accuracy, cost),),), "development"),
    )


@pytest.mark.parametrize("repetitions", [1, 2, 3, 5, 10])
def test_pareto_screen_promotes_equal_accuracy_lower_cost(repetitions: int) -> None:
    parent = objective_evaluation(CANDIDATE, (((1.0, 2.0),) * repetitions,))
    child = objective_evaluation("sha256:" + "b" * 64, (((1.0, 1.0),) * repetitions,), "train_after")
    parameters = ParetoScreenParameters(repetitions=repetitions, draws=200)

    verdict = ParetoAcceptance(parameters).compare(parent, child)

    assert isinstance(verdict, PatchVerdict)
    assert verdict.parameters == parameters
    assert verdict.accepted
    assert verdict.before_score == verdict.after_score == 1.0
    assert verdict.before_cost == 2.0
    assert verdict.after_cost == 1.0
    assert verdict.samples_per_arm == repetitions
    assert verdict.joint_support == 1.0
    assert FullOnAcceptDevelopment().decide(verdict).evaluate
    assert not MatchedValidStrictImprovement().compare(parent, child).accepted


def test_pareto_aggregation_aligns_repetition_ids_and_ignores_retry_count() -> None:
    evaluation = objective_evaluation(CANDIDATE, (((0.0, 2.0), (1.0, 4.0)), ((0.5, 1.0), (0.5, 1.0))))
    evaluation = replace(
        evaluation,
        observations=tuple(replace(item, attempts=5) for item in evaluation.observations[::-1]),
    )

    verdict = ParetoAcceptance(ParetoScreenParameters(2)).compare(
        evaluation, replace(evaluation, purpose="train_after")
    )

    assert (verdict.before_score, verdict.before_cost) == (0.5, 2.0)
    assert (verdict.after_score, verdict.after_cost) == (0.5, 2.0)


def test_pareto_archive_keeps_tradeoffs_without_a_deployment_floor() -> None:
    seed = development_candidate("f", 0.90, 2.0)
    expensive = development_candidate("c", 0.95, 2.5)
    cheap = development_candidate("a", 0.50, 0.1)
    dominated = development_candidate("d", 0.49, 0.2)
    candidates = (seed, expensive, cheap, dominated, seed)

    assert pareto_front(candidates, repetitions=1) == (seed, expensive, cheap)
    assert ParetoDevelopmentRanking(1).select(candidates) == cheap
    assert len(candidates) == 5


def test_pareto_reuses_evaluation_aggregate_score() -> None:
    runs = (((0.1, 2.0), (0.2, 2.0), (0.3, 2.0)), ((0.6, 2.0), (0.7, 2.0), (0.8, 2.0)))
    parent = objective_evaluation(CANDIDATE, runs)
    child = replace(parent, purpose="train_after", observations=parent.observations[::-1])

    verdict = ParetoAcceptance(ParetoScreenParameters(3, draws=100)).compare(parent, child)

    assert verdict.before_score == parent.aggregate_score
    assert verdict.after_score == child.aggregate_score
    candidate, _ = development_candidate("a", 1.0, 2.0)
    development = objective_evaluation(CANDIDATE, runs, "development")
    assert development.aggregate_score is not None
    equal_candidate, _ = development_candidate("b", development.aggregate_score, 2.0)
    equal_evaluation = objective_evaluation(
        equal_candidate.candidate_id,
        tuple(((development.aggregate_score, 2.0),) * 3 for _ in runs),
        "development",
    )
    assert equal_evaluation.aggregate_score == development.aggregate_score
    candidates = ((candidate, development), (equal_candidate, equal_evaluation))
    assert pareto_front(candidates, repetitions=3) == candidates


def test_pareto_archive_strict_dominance_and_equal_vectors() -> None:
    seed = development_candidate("a", 0.8, 2.0)
    first = development_candidate("b", 0.9, 1.0)
    equal = development_candidate("c", 0.9, 1.0)

    assert dominates((0.9, 1.0), (0.8, 2.0))
    assert not dominates((0.9, 1.0), (0.9, 1.0))
    assert pareto_front((seed, first, equal, first), repetitions=1) == (first, equal)
    assert ParetoDevelopmentRanking(1).select((seed, first, equal, first)) == equal


@pytest.mark.parametrize(
    ("accuracy", "cost", "accepted"),
    [
        (0.9, 1.0, True),
        (0.899999, 1.0, False),
        (0.0, 0.01, False),
        (1.0, 2.0, False),
        (1.0, 2.0 - 1e-15, False),
        (1.0, 3.0, False),
    ],
)
def test_pareto_accuracy_safety_boundary_and_rejections(accuracy: float, cost: float, accepted: bool) -> None:
    parent = objective_evaluation(CANDIDATE, (((1.0, 2.0),) * 3,))
    child = objective_evaluation(CANDIDATE, (((accuracy, cost),) * 3,), "train_after")
    parameters = ParetoScreenParameters(3, draws=137, rng_seed=99)
    verdict = ParetoAcceptance(parameters).compare(parent, child)

    assert verdict.accepted is accepted
    assert verdict.joint_support == float(accepted)


@pytest.mark.parametrize("repetitions", [1, 3])
@pytest.mark.parametrize("threshold", [0.0, 0.7])
def test_pareto_screen_does_not_override_configured_threshold(repetitions: int, threshold: float) -> None:
    parent = objective_evaluation(CANDIDATE, (((1.0, 2.0),) * repetitions,))
    child = objective_evaluation(CANDIDATE, (((1.0, 2.0),) * repetitions,), "train_after")
    parameters = ParetoScreenParameters(repetitions, draws=137, rng_seed=99, joint_support_threshold=threshold)

    verdict = ParetoAcceptance(parameters).compare(parent, child)

    assert verdict.joint_support == 0.0
    assert verdict.accepted is (threshold == 0.0)
    assert verdict.parameters == parameters


@pytest.mark.parametrize("repetitions", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("cost", [1.0, 3.0])
def test_quality_improvement_can_trade_off_cost(repetitions: int, cost: float) -> None:
    parent = objective_evaluation(CANDIDATE, (((0.0, 2.0),) * repetitions,))
    child = objective_evaluation(CANDIDATE, (((1.0, cost),) * repetitions,), "train_after")
    verdict = ParetoAcceptance(ParetoScreenParameters(repetitions, draws=100)).compare(parent, child)

    assert verdict.accepted
    assert verdict.joint_support == 1.0


@pytest.mark.parametrize("threshold", [0.6609, 0.661, 0.70])
def test_three_run_bayesian_bootstrap_matches_reference_golden(threshold: float) -> None:
    parent = objective_evaluation(
        CANDIDATE,
        (
            ((1.0, 3.0), (0.0, 1.0), (1.0, 2.0)),
            ((0.0, 0.5), (1.0, 4.0), (1.0, 2.0)),
        ),
    )
    child = objective_evaluation(
        CANDIDATE,
        (
            ((1.0, 2.0), (1.0, 1.5), (0.0, 1.0)),
            ((1.0, 0.75), (1.0, 3.0), (0.0, 1.0)),
        ),
        "train_after",
    )
    parameters = ParetoScreenParameters(3, joint_support_threshold=threshold)
    verdict = ParetoAcceptance(parameters).compare(parent, child)

    assert verdict.samples_per_arm == 6
    assert (verdict.parameters.draws, verdict.parameters.rng_seed) == (10_000, 42)
    assert verdict.before_score == verdict.after_score == pytest.approx(2 / 3)
    assert verdict.before_cost == pytest.approx(25 / 12)
    assert verdict.after_cost == pytest.approx(37 / 24)
    assert verdict.joint_support == 0.6609
    assert verdict.accepted is (threshold <= 0.6609)


@pytest.mark.parametrize("repetitions", [2, 3, 5, 10])
def test_bootstrap_varied_samples_are_reproducible_and_order_independent(repetitions: int) -> None:
    task = tuple((float(index % 2), float(index + 1)) for index in range(repetitions))
    parent = objective_evaluation(CANDIDATE, (task, task[::-1]))
    child = objective_evaluation(CANDIDATE, (task[::-1], task), "train_after")
    parameters = ParetoScreenParameters(repetitions, draws=1_000)
    policy = ParetoAcceptance(parameters)
    verdict = policy.compare(parent, child)
    reordered = policy.compare(
        replace(parent, observations=parent.observations[::-1]),
        replace(child, observations=child.observations[::-1]),
    )
    different_seed = ParetoAcceptance(replace(parameters, rng_seed=43)).compare(parent, child)

    assert verdict == reordered
    assert verdict.joint_support != different_seed.joint_support
    assert verdict.samples_per_arm == 2 * repetitions
    assert verdict.compared_case_ids == parent.requested_case_ids
    assert verdict.samples_per_arm == len(verdict.compared_case_ids) * parameters.repetitions
    assert verdict.accepted is (verdict.joint_support >= parameters.joint_support_threshold)
    assert 0 <= verdict.joint_support <= 1


@pytest.mark.parametrize("repetitions", [0, -1, True, False, 1.5, "3", None])
def test_pareto_entry_points_reject_invalid_repetition_counts(repetitions) -> None:
    candidate = development_candidate("a", 1.0, 1.0)
    ranking = ParetoDevelopmentRanking(repetitions)
    with pytest.raises(ValueError, match="repetitions"):
        pareto_front((candidate,), repetitions=repetitions)
    with pytest.raises(ValueError, match="repetitions"):
        ranking.select((candidate,))


@pytest.mark.parametrize("repetition_ids", [(0,), (0, 1, 2), (0, 2), (1, 2), (False, 1)])
def test_pareto_requires_exact_repetition_coverage(repetition_ids: tuple[int, ...]) -> None:
    evaluation = objective_evaluation(CANDIDATE, (((1.0, 1.0),),))
    evaluation = replace(
        evaluation,
        observations=tuple(replace(evaluation.observations[0], repetition=value) for value in repetition_ids),
    )
    with pytest.raises(ValueError, match="repetition"):
        ParetoAcceptance(ParetoScreenParameters(2)).compare(evaluation, replace(evaluation, purpose="train_after"))


@pytest.mark.parametrize(
    "changes",
    [
        {"cost": Cost()},
        {"cost": Cost(currency_amount=0.0)},
        {"cost": Cost(currency_amount=float("nan"))},
        {"cost": Cost(currency_amount=float("inf"))},
        {"cost": Cost(currency_amount=True)},
        {"score": -0.1},
        {"score": 1.1},
        {"score": True},
        {"disposition": "execution_error", "score": None},
        {"disposition": "invalid", "score": None},
    ],
)
def test_pareto_never_discards_or_imputes_malformed_observations(changes: dict) -> None:
    evaluation = objective_evaluation(CANDIDATE, (((1.0, 1.0), (1.0, 1.0)),))
    invalid = replace(evaluation.observations[0], **changes)
    evaluation = replace(evaluation, observations=(invalid, evaluation.observations[1]))
    with pytest.raises(ValueError):
        ParetoAcceptance(ParetoScreenParameters(2)).compare(evaluation, replace(evaluation, purpose="train_after"))


def test_pareto_uses_existing_score_and_currency_fields_only() -> None:
    parent = objective_evaluation(CANDIDATE, (((0.0, 2.0),), ((1.0, 4.0),)))
    child = objective_evaluation(CANDIDATE, (((1.0, 1.0),), ((1.0, 1.0),)), "train_after")
    assert all(not item.objectives for item in (*parent.observations, *child.observations))
    policy = ParetoAcceptance(ParetoScreenParameters(1, draws=100))
    verdict = policy.compare(parent, child)

    assert (verdict.before_score, verdict.before_cost) == (0.5, 3.0)
    assert (verdict.after_score, verdict.after_cost) == (1.0, 1.0)
    assert MatchedValidStrictImprovement().compare(parent, child).before_score == verdict.before_score
    extra_objectives = replace(
        child,
        observations=tuple(replace(item, objectives={"accuracy": 0.0, "cost": 100.0}) for item in child.observations),
    )
    assert policy.compare(parent, extra_objectives) == verdict


def test_pareto_does_not_infer_missing_currency_from_objectives() -> None:
    evaluation = objective_evaluation(CANDIDATE, (((1.0, 1.0),),))
    missing_cost = replace(evaluation.observations[0], cost=Cost(), objectives={"accuracy": 1.0, "cost": 1.0})
    evaluation = replace(evaluation, observations=(missing_cost,))

    assert evaluation.aggregate_score == 1.0
    with pytest.raises(ValueError, match="cost.currency_amount"):
        ParetoAcceptance(ParetoScreenParameters(1)).compare(evaluation, replace(evaluation, purpose="train_after"))


def test_pareto_rejects_empty_evaluation() -> None:
    evaluation = objective_evaluation(CANDIDATE, ())
    with pytest.raises(ValueError, match="at least one case"):
        ParetoAcceptance(ParetoScreenParameters(1)).compare(evaluation, replace(evaluation, purpose="train_after"))


def test_pareto_screen_requires_matched_purposes_iteration_tasks_and_counts() -> None:
    parent = objective_evaluation(CANDIDATE, (((1.0, 2.0),), ((1.0, 2.0),)))
    child = objective_evaluation(CANDIDATE, (((1.0, 1.0),), ((1.0, 1.0),)), "train_after")
    invalid_children = (
        replace(child, purpose="train_before"),
        replace(child, iteration=1),
        replace(child, requested_case_ids=child.requested_case_ids[::-1]),
        objective_evaluation(CANDIDATE, (((1.0, 1.0),),), "train_after"),
        objective_evaluation(CANDIDATE, (((1.0, 1.0),) * 2, ((1.0, 1.0),) * 2), "train_after"),
        objective_evaluation(CANDIDATE, (((1.0, 1.0),), ((1.0, 1.0),)), "development"),
    )
    for invalid_child in invalid_children:
        with pytest.raises(ValueError):
            ParetoAcceptance(ParetoScreenParameters(1)).compare(parent, invalid_child)


def test_pareto_screen_rejects_nonfinite_cost_ratio() -> None:
    parent = objective_evaluation(CANDIDATE, (((1.0, 1e-300),),))
    child = objective_evaluation(CANDIDATE, (((1.0, 1e300),),), "train_after")
    with pytest.raises(ValueError, match="finite screening arithmetic"):
        ParetoAcceptance(ParetoScreenParameters(1)).compare(parent, child)


def test_pareto_archive_rejects_incompatible_evidence() -> None:
    seed = development_candidate("a", 0.9, 2.0)
    candidate, evaluation = development_candidate("b", 0.8, 1.0)
    invalid_pairs = (
        (candidate, seed[1]),
        (candidate, objective_evaluation(candidate.candidate_id, (((0.8, 1.0),),))),
        (candidate, objective_evaluation(candidate.candidate_id, (((0.8, 1.0),),) * 2, "development")),
        (candidate, replace(evaluation, observations=(replace(evaluation.observations[0], cost=Cost()),))),
    )
    for invalid_pair in invalid_pairs:
        with pytest.raises(ValueError):
            pareto_front((seed, invalid_pair), repetitions=1)
    with pytest.raises(ValueError, match="conflicting objectives"):
        pareto_front((seed, development_candidate("a", 0.8, 1.0)), repetitions=1)
    with pytest.raises(ValueError, match="requires development"):
        ParetoDevelopmentRanking(1).select(())