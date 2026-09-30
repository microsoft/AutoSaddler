from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from autosaddler.v2.core.domain import Candidate, Case, Evaluation, JsonValue, PatchVerdict


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
    acceptance: MatchedValidStrictImprovement
    development: FullOnAcceptDevelopment
    ranking: MeanDevelopmentRanking
    budget: BudgetPolicy