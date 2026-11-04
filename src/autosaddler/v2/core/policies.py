from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from autosaddler.v2.core.domain import (
    Candidate,
    Case,
    Evaluation,
    JsonValue,
    ParetoScreenParameters,
    ParetoScreenResult,
    PatchVerdict,
)


@dataclass(frozen=True, slots=True)
class TaskSelection:
    case_ids: tuple[str, ...]
    provenance: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Task selections must contain unique case IDs")


class FixedTaskSelectionPolicy:
    def __init__(self, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("Task-selection batch size must be positive")
        self.batch_size = batch_size

    def select(self, cases: Sequence[Case], iteration: int) -> TaskSelection:
        if not cases:
            raise ValueError("Cannot select from an empty training set")
        if iteration < 0:
            raise ValueError("Iteration cannot be negative")
        start = (iteration * self.batch_size) % len(cases)
        selected = tuple(cases[(start + offset) % len(cases)].case_id for offset in range(min(self.batch_size, len(cases))))
        return TaskSelection(
            case_ids=selected,
            provenance={"policy": "fixed", "iteration": iteration, "start": start},
        )


class EpochShuffledTaskSelectionPolicy:
    def __init__(self, *, batch_size: int, seed: int) -> None:
        if batch_size <= 0:
            raise ValueError("Task-selection batch size must be positive")
        self.batch_size = batch_size
        self.seed = seed

    def select(self, cases: Sequence[Case], iteration: int) -> TaskSelection:
        if not cases:
            raise ValueError("Cannot select from an empty training set")
        if iteration < 0:
            raise ValueError("Iteration cannot be negative")
        case_ids = tuple(case.case_id for case in cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Epoch-shuffled training cases must have unique IDs")
        epoch = 0
        remaining = list(_epoch_order(case_ids, seed=self.seed, epoch=epoch))
        selected: tuple[str, ...] = ()
        selected_epoch = 0
        selected_order: tuple[str, ...] = ()
        selected_segments: tuple[dict[str, JsonValue], ...] = ()
        for current_iteration in range(iteration + 1):
            batch: list[str] = []
            batch_epoch = epoch
            batch_order = _epoch_order(case_ids, seed=self.seed, epoch=epoch)
            segments: dict[int, list[str]] = {}
            while len(batch) < min(self.batch_size, len(case_ids)):
                if not remaining:
                    epoch += 1
                    remaining = list(_epoch_order(case_ids, seed=self.seed, epoch=epoch))
                candidate_index = next(
                    index for index, candidate_id in enumerate(remaining) if candidate_id not in batch
                )
                candidate_id = remaining.pop(candidate_index)
                batch.append(candidate_id)
                segments.setdefault(epoch, []).append(candidate_id)
            if current_iteration == iteration:
                selected = tuple(batch)
                selected_epoch = batch_epoch
                selected_order = batch_order
                selected_segments = tuple(
                    {"epoch": segment_epoch, "case_ids": segment_case_ids}
                    for segment_epoch, segment_case_ids in segments.items()
                )
        return TaskSelection(
            case_ids=selected,
            provenance={
                "policy": "epoch_shuffled",
                "iteration": iteration,
                "seed": self.seed,
                "epoch": selected_epoch,
                "epoch_order": list(selected_order),
                "epoch_segments": list(selected_segments),
            },
        )


def _epoch_order(case_ids: tuple[str, ...], *, seed: int, epoch: int) -> tuple[str, ...]:
    order = list(case_ids)
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return tuple(order)


class MatchedValidStrictImprovement:
    def compare(self, parent: Evaluation, child: Evaluation) -> PatchVerdict:
        if parent.split != "train" or child.split != "train":
            raise ValueError("Acceptance may compare training evaluations only")
        if parent.requested_case_ids != child.requested_case_ids:
            raise ValueError("Parent and child must be evaluated on the identical ordered case set")
        parent_by_key = {(item.case_id, item.repetition): item for item in parent.observations}
        child_by_key = {(item.case_id, item.repetition): item for item in child.observations}
        if parent_by_key.keys() != child_by_key.keys():
            raise ValueError("Parent and child observations must have identical case/repetition keys")
        matched = [
            (key, parent_by_key[key], child_by_key[key])
            for key in sorted(parent_by_key)
            if parent_by_key[key].is_valid and child_by_key[key].is_valid
        ]
        if not matched:
            raise ValueError("Acceptance has no matched valid observations")
        before = sum(item.score for _, item, _ in matched if item.score is not None) / len(matched)
        after = sum(item.score for _, _, item in matched if item.score is not None) / len(matched)
        case_ids = tuple(dict.fromkeys(key[0] for key, _, _ in matched))
        accepted = after > before
        return PatchVerdict(
            before_score=before,
            after_score=after,
            compared_case_ids=case_ids,
            accepted=accepted,
            reason="strict matched-valid improvement" if accepted else "no strict matched-valid improvement",
        )


@dataclass(frozen=True, slots=True)
class DevelopmentDecision:
    evaluate: bool
    reason: str


class FullOnAcceptDevelopment:
    def decide(self, verdict: PatchVerdict) -> DevelopmentDecision:
        return DevelopmentDecision(
            evaluate=verdict.accepted,
            reason="candidate accepted on training" if verdict.accepted else "candidate declined on training",
        )


class MeanDevelopmentRanking:
    def select(self, candidates: Sequence[tuple[Candidate, Evaluation]]) -> tuple[Candidate, Evaluation]:
        if not candidates:
            raise ValueError("Ranking requires at least one development-evaluated candidate")
        for _, evaluation in candidates:
            if evaluation.split != "development" or evaluation.aggregate_score is None:
                raise ValueError("Ranking requires valid development aggregates")
        _, selected = max(
            enumerate(candidates),
            key=lambda indexed: (
                indexed[1][1].aggregate_score
                if indexed[1][1].aggregate_score is not None
                else float("-inf"),
                indexed[0],
            ),
        )
        return selected


def _objective_runs(evaluation: Evaluation, repetitions: int) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not evaluation.requested_case_ids:
        raise ValueError("Pareto evaluation requires at least one case")
    expected = {(case_id, repetition) for case_id in evaluation.requested_case_ids for repetition in range(repetitions)}
    by_key: dict[tuple[str, int], tuple[float, float]] = {}
    for observation in evaluation.observations:
        if not observation.is_valid:
            raise ValueError("Pareto evaluations require every observation to be valid")
        if type(observation.repetition) is not int:
            raise ValueError("Pareto repetition IDs must be integers")
        score = observation.score
        cost = observation.cost.currency_amount
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
        ):
            raise ValueError("Pareto observation score must be a number in [0, 1]")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost <= 0
        ):
            raise ValueError("Pareto observation cost.currency_amount must be finite and positive")
        by_key[(observation.case_id, observation.repetition)] = (score, cost)
    if by_key.keys() != expected:
        raise ValueError("Pareto evaluation must cover exactly the configured case/repetition IDs")
    return tuple(
        tuple(by_key[(case_id, repetition)] for repetition in range(repetitions))
        for case_id in evaluation.requested_case_ids
    )


def _mean_cost(runs: tuple[tuple[tuple[float, float], ...], ...]) -> float:
    cost = math.fsum(math.fsum(cost / len(task) for _, cost in task) / len(runs) for task in runs)
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("Pareto mean cost must be finite and positive")
    return cost


def dominates(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Compare validated (mean score, mean USD cost) pairs."""
    first_score, first_cost = first
    second_score, second_cost = second
    return (
        first_score >= second_score
        and first_cost <= second_cost
        and (first_score > second_score or first_cost < second_cost)
    )


def pareto_front(
    candidates: Sequence[tuple[Candidate, Evaluation]], *, repetitions: int
) -> tuple[tuple[Candidate, Evaluation], ...]:
    """Return nondominated candidates in caller-supplied durable creation order.

    Repeated identities retain their first position and must agree on objectives.
    Distinct identities with equal vectors remain separate archive members.
    """
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("Pareto repetitions must be a positive integer")
    if not candidates:
        raise ValueError("Pareto archive requires development-evaluated candidates")
    case_ids = candidates[0][1].requested_case_ids
    unique: dict[str, tuple[Candidate, Evaluation, tuple[float, float]]] = {}
    for candidate, evaluation in candidates:
        if evaluation.split != "development" or evaluation.candidate_id != candidate.candidate_id:
            raise ValueError("Pareto archive requires matching candidate development evaluations")
        if evaluation.requested_case_ids != case_ids:
            raise ValueError("Pareto archive requires identical ordered development case sets")
        runs = _objective_runs(evaluation, repetitions)
        score = evaluation.aggregate_score
        assert score is not None
        summary = (score, _mean_cost(runs))
        if candidate.candidate_id in unique:
            if unique[candidate.candidate_id][2] != summary:
                raise ValueError("Repeated Pareto candidate identity has conflicting objectives")
            continue
        unique[candidate.candidate_id] = (candidate, evaluation, summary)
    return tuple(
        (candidate, evaluation)
        for candidate, evaluation, summary in unique.values()
        if not any(dominates(other_summary, summary) for _, _, other_summary in unique.values())
    )


def _bootstrap_arm(
    runs: tuple[tuple[tuple[float, float], ...], ...],
    parameters: ParetoScreenParameters,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    task_draws = []
    for task in runs:
        weights = rng.dirichlet(np.ones(parameters.repetitions), parameters.draws)
        task_draws.append(weights @ np.asarray(task))
    mean_draws = np.mean(task_draws, axis=0)
    return mean_draws[:, 0], mean_draws[:, 1]


@dataclass(frozen=True, slots=True)
class ParetoAcceptance:
    parameters: ParetoScreenParameters

    def compare(self, parent: Evaluation, child: Evaluation) -> ParetoScreenResult:
        if parent.purpose != "train_before" or child.purpose != "train_after":
            raise ValueError("Pareto screening requires train_before and train_after evaluations")
        if parent.iteration != child.iteration or parent.requested_case_ids != child.requested_case_ids:
            raise ValueError("Pareto screening requires the same iteration and identical ordered case sets")
        parameters = self.parameters
        before_runs = _objective_runs(parent, parameters.repetitions)
        after_runs = _objective_runs(child, parameters.repetitions)
        before_score = parent.aggregate_score
        after_score = child.aggregate_score
        assert before_score is not None and after_score is not None
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                rng = np.random.default_rng(parameters.rng_seed)
                before_accuracy, before_cost = _bootstrap_arm(before_runs, parameters, rng)
                after_accuracy, after_cost = _bootstrap_arm(after_runs, parameters, rng)
                accuracy_delta = after_accuracy - before_accuracy
                cost_delta_ratio = (after_cost - before_cost) / before_cost
        except FloatingPointError as error:
            raise ValueError("Pareto objective magnitudes exceed finite screening arithmetic") from error
        tolerance = np.finfo(float).eps * 8
        promising = (accuracy_delta > tolerance) | (cost_delta_ratio < -tolerance)
        safe = accuracy_delta >= -parameters.max_accuracy_drop - tolerance * max(1.0, parameters.max_accuracy_drop)
        joint_support = float(np.mean(promising & safe))
        threshold = parameters.joint_support_threshold
        accepted = joint_support >= threshold
        return ParetoScreenResult(
            before_score=before_score,
            after_score=after_score,
            compared_case_ids=parent.requested_case_ids,
            accepted=accepted,
            reason=(
                f"joint support {joint_support:.4f} "
                f"{'meets' if accepted else 'is below'} threshold {threshold:.4f} "
                f"for promising movement with max accuracy drop {parameters.max_accuracy_drop:.4f}"
            ),
            before_cost=_mean_cost(before_runs),
            after_cost=_mean_cost(after_runs),
            parameters=parameters,
            samples_per_arm=len(parent.observations),
            joint_support=joint_support,
        )


@dataclass(frozen=True, slots=True)
class ParetoDevelopmentRanking:
    repetitions: int

    def select(self, candidates: Sequence[tuple[Candidate, Evaluation]]) -> tuple[Candidate, Evaluation]:
        """Select the newest nondominated candidate in the supplied creation order."""
        return pareto_front(candidates, repetitions=self.repetitions)[-1]


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    max_rollouts: int
    max_iterations: int

    def __post_init__(self) -> None:
        if self.max_rollouts <= 0 or self.max_iterations <= 0:
            raise ValueError("Budget limits must be positive")

    def allows_iteration(self, *, iteration: int, attempted_rollouts: int) -> bool:
        return iteration < self.max_iterations and attempted_rollouts < self.max_rollouts


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    task_selection: FixedTaskSelectionPolicy | EpochShuffledTaskSelectionPolicy
    acceptance: MatchedValidStrictImprovement | ParetoAcceptance
    development: FullOnAcceptDevelopment
    ranking: MeanDevelopmentRanking | ParetoDevelopmentRanking
    budget: BudgetPolicy