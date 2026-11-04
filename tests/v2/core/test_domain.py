from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Candidate,
    Cost,
    Evaluation,
    Observation,
    ParetoScreenParameters,
    ParetoScreenResult,
    PatchVerdict,
    to_json_value,
)


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


@pytest.mark.parametrize("cached,uncached", [(0, 8), (3, 5), (8, 0), (0, 0)])
def test_cost_input_breakdown_round_trips(cached: int, uncached: int) -> None:
    from autosaddler.v2.core.serde import cost_from, record

    cost = Cost(
        rollouts=1, input_tokens=cached + uncached, output_tokens=2,
        currency_amount=0.01, cached_input_tokens=cached, uncached_input_tokens=uncached,
    )
    payload = record(cost)
    assert payload["cached_input_tokens"] == cached
    assert payload["uncached_input_tokens"] == uncached
    assert cost_from(payload) == cost


def test_legacy_cost_loads_unknown_breakdown_and_serializes_nulls() -> None:
    from autosaddler.v2.core.serde import cost_from, record

    payload = {
        "rollouts": 1, "sessions": 0, "input_tokens": 8, "output_tokens": 2,
        "wall_seconds": 0.0, "currency_amount": None,
    }
    cost = cost_from(payload)
    assert cost.cached_input_tokens is None
    assert cost.uncached_input_tokens is None
    expected = {**payload, "cached_input_tokens": None, "uncached_input_tokens": None}
    assert record(cost) == expected
    assert record(Cost(rollouts=1, input_tokens=8, output_tokens=2)) == expected
    assert to_json_value({"cost": cost}) == {"cost": expected}
    assert cost_from(expected) == cost


@pytest.mark.parametrize("cached,uncached", [(None, 5), (3, None), (-1, 9), (9, -1), (True, 7), (3, False), (3.0, 5), (3, 5.0), (3, 4)])
def test_cost_rejects_invalid_input_breakdown(cached, uncached) -> None:
    with pytest.raises(ValueError, match="Cost.*input"):
        Cost(input_tokens=8, cached_input_tokens=cached, uncached_input_tokens=uncached)


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


def test_pareto_result_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(screen_result(), "after_cost", 0.0)


@pytest.mark.parametrize("repetitions", [1, 2, 3, 5, 10])
def test_pareto_screen_parameters_accept_arbitrary_positive_counts(repetitions: int) -> None:
    parameters = ParetoScreenParameters(repetitions)

    assert parameters.repetitions == repetitions
    assert parameters.draws == 10_000
    assert parameters.rng_seed == 42
    assert parameters.max_accuracy_drop == 0.1
    assert parameters.joint_support_threshold == 0.7


@pytest.mark.parametrize(
    ("field", "value"),
    [("repetitions", value) for value in (0, -1, True, False, 2.5, "3", None)]
    + [("draws", value) for value in (0, -1, True, 2.5, "3", None)]
    + [("rng_seed", value) for value in (-1, True, 2.5, "3", None)]
    + [("max_accuracy_drop", value) for value in (-0.1, float("nan"), float("inf"), True, None)]
    + [
        ("joint_support_threshold", value) for value in (-0.1, 1.1, float("nan"), float("inf"), True, None)
    ],
)
def test_pareto_screen_parameters_reject_invalid_values(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        ParetoScreenParameters(**{"repetitions": 3, field: value})


def screen_result() -> ParetoScreenResult:
    return ParetoScreenResult(
        before_score=1.0,
        after_score=1.0,
        compared_case_ids=("case-a",),
        accepted=True,
        reason="cost reduction",
        before_cost=2.0,
        after_cost=1.0,
        parameters=ParetoScreenParameters(repetitions=3),
        samples_per_arm=3,
        joint_support=1.0,
    )


def test_pareto_result_preserves_scalar_verdict_record_shape() -> None:
    scalar = PatchVerdict(1.0, 1.0, ("case-a",), True, "cost reduction")
    expected = {
        "before_score": 1.0,
        "after_score": 1.0,
        "compared_case_ids": ["case-a"],
        "accepted": True,
        "reason": "cost reduction",
    }
    assert to_json_value(scalar) == expected
    result = screen_result()
    payload = to_json_value(result)
    assert isinstance(payload, dict)
    assert {key: payload[key] for key in expected} == expected
    assert payload["before_cost"] == 2.0
    assert payload["after_cost"] == 1.0
    assert payload["parameters"]["repetitions"] == 3
    assert payload["parameters"]["draws"] == 10_000
    assert payload["parameters"]["rng_seed"] == 42
    assert payload["parameters"]["joint_support_threshold"] == 0.7
    assert not {"draws", "rng_seed", "joint_support_threshold"} & payload.keys()
    assert payload["samples_per_arm"] == 3
    assert payload.keys() == expected.keys() | {
        "before_cost",
        "after_cost",
        "parameters",
        "samples_per_arm",
        "joint_support",
    }