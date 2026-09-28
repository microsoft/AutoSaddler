from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from autosaddler.v2.core.curriculum import (
    CURRICULUM_SESSION_KINDS,
    SCORE_FORMULA,
    CurriculumState,
    SamplingAction,
    activity_ema,
    softmax_floor,
)
from autosaddler.v2.core.domain import Candidate, Case, Evaluation, JsonValue, PatchVerdict


@dataclass(frozen=True, slots=True)
class TaskSelection:
    case_ids: tuple[str, ...]
    provenance: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Task selections must contain unique case IDs")


class TaskSelectionPolicy(Protocol):
    required_session_kinds: frozenset[str]

    def settings_record(self) -> dict[str, JsonValue]: ...


class FixedTaskSelectionPolicy:
    required_session_kinds: frozenset[str] = frozenset()

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

    def settings_record(self) -> dict[str, JsonValue]:
        return {}


class EpochShuffledTaskSelectionPolicy:
    required_session_kinds: frozenset[str] = frozenset()

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

    def settings_record(self) -> dict[str, JsonValue]:
        return {}


def _epoch_order(case_ids: tuple[str, ...], *, seed: int, epoch: int) -> tuple[str, ...]:
    order = list(case_ids)
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return tuple(order)


class ActiveSaddlerTaskSelectionPolicy:
    """Agent-driven infinite-armed bandit curriculum over failure patterns.

    Each iteration performs exactly one action. An unseen draw takes the next
    never-executed cases from a fixed seeded permutation of the training set.
    An arm pull samples one failure pattern with probability given by a floored
    softmax over the agent's current-iteration learning-progress scores and then
    evaluates up to ``batch_size`` of that pattern's cases. The pull/draw choice
    and the arm scores come from optimizer sessions recorded before selection;
    this policy is a pure function of the replayed curriculum state.
    """

    required_session_kinds: frozenset[str] = CURRICULUM_SESSION_KINDS

    def __init__(
        self,
        *,
        batch_size: int,
        seed: int,
        softmax_temperature: float,
        min_prob: float,
        ema_eta: float,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Task-selection batch size must be positive")
        if softmax_temperature <= 0.0:
            raise ValueError("ActiveSaddler softmax_temperature must be positive")
        if not 0.0 <= min_prob < 1.0:
            raise ValueError("ActiveSaddler min_prob must be in [0, 1)")
        if not 0.0 < ema_eta <= 1.0:
            raise ValueError("ActiveSaddler ema_eta must be in (0, 1]")
        self.batch_size = batch_size
        self.seed = seed
        self.softmax_temperature = softmax_temperature
        self.min_prob = min_prob
        self.ema_eta = ema_eta

    def settings_record(self) -> dict[str, JsonValue]:
        return {
            "softmax_temperature": self.softmax_temperature,
            "min_prob": self.min_prob,
            "ema_eta": self.ema_eta,
        }

    def curriculum_context(self) -> dict[str, JsonValue]:
        return {"policy": "activesaddler", "batch_size": self.batch_size, **self.settings_record()}

    def draw_order(self, cases: Sequence[Case]) -> tuple[str, ...]:
        return _epoch_order(_unique_case_ids(cases), seed=self.seed, epoch=0)

    def select_curriculum(
        self,
        cases: Sequence[Case],
        iteration: int,
        *,
        state: CurriculumState,
        action: SamplingAction,
    ) -> TaskSelection:
        if iteration < 0:
            raise ValueError("Iteration cannot be negative")
        case_ids = _unique_case_ids(cases)
        arms = state.arms(case_ids)
        unseen = state.unseen_case_ids(self.draw_order(cases))
        scores = {pattern_id: state.arm_score(pattern_id, iteration) for pattern_id in arms}
        probabilities: dict[str, float] = {}
        chosen_arm: str | None = None
        if action == "unseen_draw":
            if not unseen:
                raise ValueError("An unseen draw requires never-executed training cases")
            selected = unseen[: self.batch_size]
        elif action == "arm_pull":
            if not arms:
                raise ValueError("An arm pull requires at least one instantiated arm")
            probabilities = softmax_floor(scores, temperature=self.softmax_temperature, min_prob=self.min_prob)
            rng = random.Random(f"{self.seed}:{iteration}:activesaddler")
            pattern_ids = list(probabilities)
            chosen_arm = rng.choices(pattern_ids, weights=[probabilities[item] for item in pattern_ids], k=1)[0]
            candidates = list(arms[chosen_arm])
            selected = tuple(candidates if len(candidates) <= self.batch_size else rng.sample(candidates, self.batch_size))
        else:
            raise ValueError(f"ActiveSaddler cannot select a batch for action {action!r}")
        arm_records: list[dict[str, JsonValue]] = []
        for pattern_id, arm_case_ids in arms.items():
            pattern = state.patterns[pattern_id]
            score = pattern.score_at(iteration)
            arm_records.append(
                {
                    "pattern_id": pattern_id,
                    "label": pattern.label,
                    "case_ids": list(arm_case_ids),
                    "num_cases": len(arm_case_ids),
                    "num_observations": len(pattern.observations),
                    "ema": activity_ema(pattern.observations, self.ema_eta),
                    "severity": score.severity if score is not None else None,
                    "fixability": score.fixability if score is not None else None,
                    "breadth": score.breadth if score is not None else None,
                    "side_effect": score.side_effect if score is not None else None,
                    "rationale": score.rationale if score is not None else None,
                    "score": scores[pattern_id],
                    "prob": probabilities.get(pattern_id),
                    "selected": pattern_id == chosen_arm,
                }
            )
        arm_records.sort(key=lambda item: (-cast(float, item["score"]), cast(str, item["pattern_id"])))
        return TaskSelection(
            case_ids=selected,
            provenance={
                "policy": "activesaddler",
                "iteration": iteration,
                "seed": self.seed,
                "action": action,
                "chosen_arm": chosen_arm,
                "batch_size": self.batch_size,
                **self.settings_record(),
                "score_formula": SCORE_FORMULA,
                "num_arms": len(arms),
                "n_probes": len(state.probe_points),
                "num_unseen_before": len(unseen),
                "unseen_case_ids": list(unseen),
                "arms": cast(JsonValue, arm_records),
            },
        )


def _unique_case_ids(cases: Sequence[Case]) -> tuple[str, ...]:
    if not cases:
        raise ValueError("Cannot select from an empty training set")
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Training cases must have unique IDs")
    return case_ids


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
    task_selection: FixedTaskSelectionPolicy | EpochShuffledTaskSelectionPolicy | ActiveSaddlerTaskSelectionPolicy
    acceptance: MatchedValidStrictImprovement
    development: FullOnAcceptDevelopment
    ranking: MeanDevelopmentRanking
    budget: BudgetPolicy