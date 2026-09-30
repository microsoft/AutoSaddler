"""ActiveSaddler: an adaptive task-selection policy over failure patterns.

The curriculum treats every failure pattern that owns at least one training
case as a bandit arm. All state is folded from the append-only event log:
pattern registrations, tags, observations, arm decisions, and arm scores are
recorded as ``ExtensionStateChanged`` events in the ``autosaddler.curriculum``
namespace, executed cases come from ``BatchSampled`` events, and probe points
come from completed training evaluations.

``ActiveSaddlerTaskSelectionPolicy`` implements the adaptive task-selection
interface in ``autosaddler.v2.core.scheduling``: it asks the engine for an arm
decision and arm scores before sampling a batch, and for a pattern-extraction
session after reflection. Nothing here performs I/O.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from autosaddler.v2.core.domain import Case, JsonValue, canonical_json, sha256_digest, to_json_value
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.core.policies import TaskSelection
from autosaddler.v2.core.scheduling import (
    DeferredRequest,
    IterationFeedback,
    NoSelection,
    SelectionRequest,
    SelectionStep,
    SessionStep,
    StatePayload,
    StateStep,
)
from autosaddler.v2.core.serde import evaluation_from
from autosaddler.v2.prompting.models import SessionResult

CURRICULUM_NAMESPACE = "autosaddler.curriculum"
CURRICULUM_SCHEMA_VERSION = "autosaddler-curriculum/v1"
CURRICULUM_SESSION_KINDS = frozenset({"extract_patterns", "decide_arm", "score_arms"})
FAILURE_SCORE_THRESHOLD = 0.5
SCORE_FORMULA = (
    "phi(p) = mean(severity, fixability, breadth, 1 - side_effect); "
    "P(pull p) = softmax(phi / tau) floored at min_prob"
)

PatternSource: TypeAlias = Literal["pre_patch", "post_patch"]
ArmAction: TypeAlias = Literal["pull", "draw"]
SamplingAction: TypeAlias = Literal["unseen_draw", "arm_pull", "empty"]
CurriculumChange: TypeAlias = Literal["patterns_extracted", "observations_recorded", "arm_decision", "arm_scores"]

_CHANGES = frozenset({"patterns_extracted", "observations_recorded", "arm_decision", "arm_scores"})


@dataclass(frozen=True, slots=True)
class PatternTag:
    case_id: str
    candidate_id: str
    evaluation_id: str
    source: PatternSource
    iteration: int
    root_cause: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.candidate_id or not self.evaluation_id:
            raise ValueError("Pattern tags require case, candidate, and evaluation IDs")
        if self.source not in {"pre_patch", "post_patch"}:
            raise ValueError(f"Unknown pattern tag source: {self.source}")
        if self.iteration < 0:
            raise ValueError("Pattern tag iteration cannot be negative")


@dataclass(frozen=True, slots=True)
class PatternObservation:
    iteration: int
    active: float
    evaluated_case_ids: tuple[str, ...]
    tagged_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.active <= 1.0:
            raise ValueError("Pattern observation activity must be in [0, 1]")
        if not self.evaluated_case_ids:
            raise ValueError("Pattern observations require evaluated case IDs")
        if not set(self.tagged_case_ids) <= set(self.evaluated_case_ids):
            raise ValueError("Tagged case IDs must be a subset of evaluated case IDs")


@dataclass(frozen=True, slots=True)
class ArmScore:
    iteration: int
    severity: float
    fixability: float
    breadth: float
    side_effect: float
    rationale: str

    def __post_init__(self) -> None:
        for name in ("severity", "fixability", "breadth", "side_effect"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Arm score {name} must be in [0, 1]")
        if not self.rationale:
            raise ValueError("Arm scores require a rationale")

    @property
    def value(self) -> float:
        return (self.severity + self.fixability + self.breadth + (1.0 - self.side_effect)) / 4.0


@dataclass(frozen=True, slots=True)
class FailurePattern:
    pattern_id: str
    label: str
    created_iteration: int
    tags: tuple[PatternTag, ...] = ()
    observations: tuple[PatternObservation, ...] = ()
    scores: tuple[ArmScore, ...] = ()

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(tag.case_id for tag in self.tags))

    @property
    def last_observed_iteration(self) -> int | None:
        return max((item.iteration for item in self.observations), default=None)

    def score_at(self, iteration: int) -> ArmScore | None:
        return next((score for score in self.scores if score.iteration == iteration), None)


@dataclass(frozen=True, slots=True)
class ArmPull:
    iteration: int
    pattern_id: str
    case_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CurriculumState:
    patterns: Mapping[str, FailurePattern] = field(default_factory=dict)
    executed_case_ids: frozenset[str] = frozenset()
    decisions: Mapping[int, Mapping[str, JsonValue]] = field(default_factory=dict)
    pulls: tuple[ArmPull, ...] = ()
    probe_points: frozenset[tuple[str, str]] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", MappingProxyType(dict(self.patterns)))
        object.__setattr__(self, "decisions", MappingProxyType(dict(self.decisions)))

    @classmethod
    def replay(cls, events: Iterable[RunEvent]) -> "CurriculumState":
        patterns: dict[str, FailurePattern] = {}
        executed: set[str] = set()
        decisions: dict[int, Mapping[str, JsonValue]] = {}
        pulls: list[ArmPull] = []
        probe_points: set[tuple[str, str]] = set()
        for event in events:
            if event.event_type == "BatchSampled":
                case_ids = _strings(event.payload.get("case_ids"), "BatchSampled case_ids")
                executed.update(case_ids)
                provenance = event.payload.get("provenance")
                if isinstance(provenance, Mapping) and isinstance(provenance.get("chosen_arm"), str):
                    pulls.append(
                        ArmPull(
                            iteration=_integer(event.payload.get("iteration"), "BatchSampled iteration"),
                            pattern_id=cast(str, provenance["chosen_arm"]),
                            case_ids=case_ids,
                        )
                    )
            elif event.event_type == "EvaluationCompleted":
                evaluation = evaluation_from(event.payload.get("evaluation"))
                if evaluation.split == "train":
                    probe_points.update((case_id, evaluation.candidate_id) for case_id in evaluation.requested_case_ids)
            elif event.event_type == "ExtensionStateChanged" and event.payload.get("namespace") == CURRICULUM_NAMESPACE:
                _apply_change(event.payload, patterns, decisions)
        return cls(
            patterns=patterns,
            executed_case_ids=frozenset(executed),
            decisions=decisions,
            pulls=tuple(pulls),
            probe_points=frozenset(probe_points),
        )

    def applied(self, change: Mapping[str, JsonValue]) -> "CurriculumState":
        """Return the state after one curriculum change payload."""
        patterns = dict(self.patterns)
        decisions = dict(self.decisions)
        _apply_change(change, patterns, decisions)
        return replace(self, patterns=patterns, decisions=decisions)

    def arms(self, train_case_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """Instantiated arms: patterns owning at least one available training case, in creation order."""
        arms: dict[str, tuple[str, ...]] = {}
        for pattern_id, pattern in self.patterns.items():
            owned = set(pattern.case_ids)
            cases = tuple(case_id for case_id in train_case_ids if case_id in owned)
            if cases:
                arms[pattern_id] = cases
        return arms

    def unseen_case_ids(self, ordered_case_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(case_id for case_id in ordered_case_ids if case_id not in self.executed_case_ids)

    def arm_score(self, pattern_id: str, iteration: int) -> float:
        """Agent score for exactly ``iteration``; an unrated arm scores zero."""
        score = self.patterns[pattern_id].score_at(iteration)
        return score.value if score is not None else 0.0


def activity_ema(observations: Sequence[PatternObservation], eta: float) -> float:
    """Diagnostic failure-activity EMA seeded at 1.0; it never drives arm selection."""
    if not 0.0 < eta <= 1.0:
        raise ValueError("EMA eta must be in (0, 1]")
    value = 1.0
    for observation in sorted(observations, key=lambda item: item.iteration):
        value = (1.0 - eta) * value + eta * observation.active
    return value


def softmax_floor(scores: Mapping[str, float], *, temperature: float, min_prob: float) -> dict[str, float]:
    """Softmax over arm scores with a per-arm probability floor."""
    if temperature <= 0.0:
        raise ValueError("Softmax temperature must be positive")
    if not 0.0 <= min_prob < 1.0:
        raise ValueError("Arm minimum probability must be in [0, 1)")
    pattern_ids = list(scores)
    count = len(pattern_ids)
    if count == 0:
        return {}
    if count == 1:
        return {pattern_ids[0]: 1.0}
    maximum = max(scores.values())
    exponentials = {pattern_id: math.exp((scores[pattern_id] - maximum) / temperature) for pattern_id in pattern_ids}
    total = sum(exponentials.values())
    soft = {pattern_id: exponentials[pattern_id] / total for pattern_id in pattern_ids}
    if min_prob > 0.0 and min_prob * count < 1.0:
        return {pattern_id: min_prob + (1.0 - min_prob * count) * soft[pattern_id] for pattern_id in pattern_ids}
    if min_prob > 0.0:
        return {pattern_id: 1.0 / count for pattern_id in pattern_ids}
    return soft


def resolve_action(*, num_arms: int, unseen_count: int, requested: ArmAction | None) -> SamplingAction:
    """Combine the agent's pull/draw request with feasibility; a cold start always draws."""
    if num_arms == 0:
        want_unseen = True
    elif requested == "draw":
        want_unseen = True
    elif requested == "pull":
        want_unseen = False
    else:
        raise ValueError(f"Arm action must be 'pull' or 'draw', got {requested!r}")
    if want_unseen and unseen_count > 0:
        return "unseen_draw"
    if num_arms > 0:
        return "arm_pull"
    return "empty"


def derive_pattern_id(*, iteration: int, key: str, label: str) -> str:
    digest = sha256_digest(canonical_json({"iteration": iteration, "key": key, "label": label}))
    return f"pattern-{digest.removeprefix('sha256:')[:12]}"


def failing_case_ids(case_scores: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(case_id for case_id, score in case_scores.items() if score < FAILURE_SCORE_THRESHOLD)


def pattern_observations(
    state: CurriculumState,
    *,
    iteration: int,
    batch_case_ids: Sequence[str],
    after_evaluation_id: str | None,
) -> list[dict[str, JsonValue]]:
    """Post-patch activity observations for every pattern overlapping the batch.

    A pattern whose overlapping tags all come from this iteration's post-patch
    evaluation was created by that analysis; arm creation is not an arm pull, so
    it receives no observation. Without a post-patch evaluation (all cases
    already passed) every overlapping pattern is observed as inactive.
    """
    batch = set(batch_case_ids)
    observations: list[dict[str, JsonValue]] = []
    for pattern_id, pattern in state.patterns.items():
        overlapping_tags = [tag for tag in pattern.tags if tag.case_id in batch]
        if not overlapping_tags:
            continue
        if all(tag.evaluation_id == after_evaluation_id for tag in overlapping_tags):
            continue
        evaluated = sorted({tag.case_id for tag in overlapping_tags})
        tagged = (
            sorted({tag.case_id for tag in overlapping_tags if tag.evaluation_id == after_evaluation_id})
            if after_evaluation_id is not None
            else []
        )
        observations.append(
            {
                "pattern_id": pattern_id,
                "active": len(tagged) / len(evaluated),
                "evaluated_case_ids": cast(JsonValue, evaluated),
                "tagged_case_ids": cast(JsonValue, tagged),
            }
        )
    return observations


class ActiveSaddlerTaskSelectionPolicy:
    """Agent-driven infinite-armed bandit curriculum over failure patterns.

    Each iteration performs exactly one action. An unseen draw takes the next
    never-executed cases from a fixed seeded permutation of the training set.
    An arm pull samples one failure pattern with probability given by a floored
    softmax over the agent's current-iteration learning-progress scores and then
    evaluates up to ``batch_size`` of that pattern's cases. The pull/draw choice
    and the arm scores come from optimizer sessions recorded before selection,
    and failure patterns come from a deferred extraction session after
    reflection; every decision is a pure function of the replayed events.
    """

    namespace = CURRICULUM_NAMESPACE
    required_session_kinds: frozenset[str] = CURRICULUM_SESSION_KINDS

    def __init__(
        self,
        *,
        batch_size: int,
        seed: int,
        softmax_temperature: float,
        min_prob: float,
        ema_eta: float,
        pattern_extraction_timeout_seconds: float,
        arm_scoring_timeout_seconds: float,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Task-selection batch size must be positive")
        if softmax_temperature <= 0.0:
            raise ValueError("ActiveSaddler softmax_temperature must be positive")
        if not 0.0 <= min_prob < 1.0:
            raise ValueError("ActiveSaddler min_prob must be in [0, 1)")
        if not 0.0 < ema_eta <= 1.0:
            raise ValueError("ActiveSaddler ema_eta must be in (0, 1]")
        if pattern_extraction_timeout_seconds <= 0.0 or arm_scoring_timeout_seconds <= 0.0:
            raise ValueError("ActiveSaddler session timeouts must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.softmax_temperature = softmax_temperature
        self.min_prob = min_prob
        self.ema_eta = ema_eta
        self.pattern_extraction_timeout_seconds = pattern_extraction_timeout_seconds
        self.arm_scoring_timeout_seconds = arm_scoring_timeout_seconds

    def settings_record(self) -> dict[str, JsonValue]:
        return {
            "softmax_temperature": self.softmax_temperature,
            "min_prob": self.min_prob,
            "ema_eta": self.ema_eta,
            "pattern_extraction_timeout_seconds": self.pattern_extraction_timeout_seconds,
            "arm_scoring_timeout_seconds": self.arm_scoring_timeout_seconds,
        }

    def session_timeouts(self) -> Mapping[str, float]:
        return {
            "extract_patterns": self.pattern_extraction_timeout_seconds,
            "decide_arm": self.arm_scoring_timeout_seconds,
            "score_arms": self.arm_scoring_timeout_seconds,
        }

    def curriculum_context(self) -> dict[str, JsonValue]:
        return {
            "policy": "activesaddler",
            "batch_size": self.batch_size,
            "softmax_temperature": self.softmax_temperature,
            "min_prob": self.min_prob,
            "ema_eta": self.ema_eta,
        }

    def prompt_context(self, events: Sequence[RunEvent], iteration: int) -> Mapping[str, JsonValue]:
        context = self.curriculum_context()
        for event in events:
            if event.event_type == "BatchSampled" and event.payload.get("iteration") == iteration:
                provenance = event.payload.get("provenance")
                if isinstance(provenance, Mapping):
                    context["sampling_action"] = provenance.get("action")
                    context["pulled_arm_id"] = provenance.get("chosen_arm")
        return context

    def draw_order(self, cases: Sequence[Case]) -> tuple[str, ...]:
        order = list(_unique_case_ids(cases))
        random.Random(f"{self.seed}:0").shuffle(order)
        return tuple(order)

    def next_selection_step(self, events: Sequence[RunEvent], request: SelectionRequest) -> SelectionStep:
        state = CurriculumState.replay(events)
        iteration = request.iteration
        case_ids = _unique_case_ids(request.train_cases)
        arms = state.arms(case_ids)
        unseen = state.unseen_case_ids(self.draw_order(request.train_cases))
        decision = state.decisions.get(iteration)
        if decision is None:
            if not arms:
                return StateStep(
                    name="arm-decision",
                    payload=self._decision(
                        iteration,
                        requested=None,
                        arms=arms,
                        unseen=unseen,
                        rationale="No failure-pattern arm exists yet; draw unseen training cases.",
                    ),
                )
            return SessionStep(
                name="decide-arm",
                kind="decide_arm",
                context=self._session_context(request, arms, unseen),
                stage="proposal.arm_decision",
                validate=_arm_decision_failure_reason,
                record=lambda result: self._decision(
                    iteration,
                    requested=cast(ArmAction, _output_string(result, "action")),
                    arms=arms,
                    unseen=unseen,
                    rationale=_output_string(result, "rationale"),
                ),
                # ActiveSaddler treats a missing or failed decision as an arm pull.
                record_exhausted=lambda error: self._decision(
                    iteration,
                    requested="pull",
                    arms=arms,
                    unseen=unseen,
                    rationale="Arm decision session failed; defaulting to an arm pull.",
                    fallback_reason=error,
                ),
            )
        action = decision.get("action")
        if action == "empty":
            return NoSelection(reason="No failure-pattern arm and no unseen training case remain.")
        if action not in {"unseen_draw", "arm_pull"}:
            raise ValueError(f"Unknown curriculum action: {action}")
        if action == "arm_pull" and all(state.patterns[arm_id].score_at(iteration) is None for arm_id in arms):
            arm_ids = tuple(arms)
            return SessionStep(
                name="score-arms",
                kind="score_arms",
                context=self._session_context(request, arms, unseen),
                stage="proposal.arm_scoring",
                validate=lambda result: _arm_scoring_failure_reason(result, arm_ids),
                record=lambda result: _change(
                    "arm_scores",
                    iteration,
                    scores=[
                        {
                            key: item[key]
                            for key in ("pattern_id", "severity", "fixability", "breadth", "side_effect", "rationale")
                        }
                        for item in cast(list[Mapping[str, JsonValue]], _output_list(result, "scores"))
                    ],
                ),
                # A missing arm score would silently zero that arm, so exhausted retries fail the run.
                record_exhausted=None,
            )
        return self.select_curriculum(request.train_cases, iteration, state=state, action=cast(SamplingAction, action))

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
                "softmax_temperature": self.softmax_temperature,
                "min_prob": self.min_prob,
                "ema_eta": self.ema_eta,
                "score_formula": SCORE_FORMULA,
                "num_arms": len(arms),
                "n_probes": len(state.probe_points),
                "num_unseen_before": len(unseen),
                "unseen_case_ids": list(unseen),
                "arms": cast(JsonValue, arm_records),
            },
        )

    def iteration_changes(self, events: Sequence[RunEvent], feedback: IterationFeedback) -> Sequence[StatePayload]:
        if feedback.outcome != "no_training_failures":
            return ()
        # An all-pass pull is still an arm observation: every overlapping arm was inactive.
        return _observation_changes(CurriculumState.replay(events), feedback.iteration, feedback.case_ids, None)

    def after_reflection(
        self,
        events: Sequence[RunEvent],
        feedback: IterationFeedback,
        lessons: Sequence[JsonValue],
    ) -> DeferredRequest | None:
        del events
        after_scores = dict(feedback.train_after_case_scores or {})
        pre_patch = failing_case_ids(feedback.train_before_case_scores)
        post_patch = failing_case_ids(after_scores)
        if not pre_patch and not post_patch:
            return None
        if feedback.child_id is None or feedback.train_after_evaluation_id is None:
            raise ValueError("Pattern extraction requires a patched candidate and its training evaluation")
        return DeferredRequest(
            kind="extract_patterns",
            stage="proposal.pattern_extraction",
            payload={
                "iteration": feedback.iteration,
                "candidate_id": feedback.child_id,
                "working_parent_candidate_id": feedback.working_parent_id,
                "train_case_ids": list(feedback.case_ids),
                "train_before_evaluation_id": feedback.train_before_evaluation_id,
                "train_after_evaluation_id": feedback.train_after_evaluation_id,
                "train_before_evidence": to_json_value(feedback.train_before_evidence),
                "train_after_evidence": to_json_value(feedback.train_after_evidence),
                "pre_patch_failures": [
                    {"case_id": case_id, "train_before_score": feedback.train_before_case_scores[case_id]}
                    for case_id in pre_patch
                ],
                "post_patch_failures": [
                    {"case_id": case_id, "train_after_score": after_scores[case_id]} for case_id in post_patch
                ],
                "train_before_case_scores": dict(feedback.train_before_case_scores),
                "train_after_case_scores": after_scores,
                "diagnosis": feedback.diagnosis,
                "lessons": list(lessons),
            },
        )

    def deferred_context(self, events: Sequence[RunEvent], request: DeferredRequest) -> Mapping[str, JsonValue]:
        payload = request.payload
        return {
            "iteration": payload["iteration"],
            "candidate_ids": [payload["candidate_id"]],
            "train_case_ids": payload["train_case_ids"],
            "task_selection": {
                **self.curriculum_context(),
                **{
                    key: payload[key]
                    for key in (
                        "working_parent_candidate_id",
                        "train_before_evaluation_id",
                        "train_after_evaluation_id",
                        "train_before_evidence",
                        "train_after_evidence",
                    )
                },
            },
            **{
                key: payload[key]
                for key in (
                    "pre_patch_failures",
                    "post_patch_failures",
                    "train_before_case_scores",
                    "train_after_case_scores",
                    "diagnosis",
                    "lessons",
                )
            },
            "existing_pattern_ids": list(CurriculumState.replay(events).patterns),
        }

    def deferred_failure_reason(
        self,
        events: Sequence[RunEvent],
        request: DeferredRequest,
        result: SessionResult,
    ) -> str | None:
        return _pattern_extraction_failure_reason(
            result,
            _case_ids(request.payload.get("pre_patch_failures"), "pre_patch_failures"),
            _case_ids(request.payload.get("post_patch_failures"), "post_patch_failures"),
            tuple(CurriculumState.replay(events).patterns),
        )

    def deferred_changes(
        self,
        events: Sequence[RunEvent],
        request: DeferredRequest,
        result: SessionResult,
    ) -> Sequence[StatePayload]:
        patterns = _pattern_extraction_change(result, request.payload)
        state = CurriculumState.replay(events).applied(patterns)
        return (patterns, *self._post_patch_observations(state, request))

    def deferred_exhausted_changes(
        self,
        events: Sequence[RunEvent],
        request: DeferredRequest,
        error: str,
    ) -> Sequence[StatePayload]:
        del error
        # ActiveSaddler still observes the sampled arms when extraction fails.
        return self._post_patch_observations(CurriculumState.replay(events), request)

    def _post_patch_observations(self, state: CurriculumState, request: DeferredRequest) -> tuple[StatePayload, ...]:
        payload = request.payload
        return _observation_changes(
            state,
            _integer(payload.get("iteration"), "extraction iteration"),
            _strings(payload.get("train_case_ids"), "extraction train_case_ids"),
            _string(payload.get("train_after_evaluation_id"), "train_after_evaluation_id"),
        )

    def _decision(
        self,
        iteration: int,
        *,
        requested: ArmAction | None,
        arms: Mapping[str, tuple[str, ...]],
        unseen: Sequence[str],
        rationale: str,
        fallback_reason: str | None = None,
    ) -> StatePayload:
        return _change(
            "arm_decision",
            iteration,
            requested_action=requested,
            action=resolve_action(num_arms=len(arms), unseen_count=len(unseen), requested=requested),
            rationale=rationale,
            fallback_reason=fallback_reason,
            num_arms=len(arms),
            num_unseen=len(unseen),
        )

    def _session_context(
        self,
        request: SelectionRequest,
        arms: Mapping[str, tuple[str, ...]],
        unseen: Sequence[str],
    ) -> dict[str, JsonValue]:
        return {
            "iteration": request.iteration,
            "candidate_ids": [request.working_parent_id],
            "selected_parent_candidate_id": request.selected_parent_id,
            "selection_parent_ids": list(request.selection_parent_ids),
            "component_sources": dict(request.component_sources),
            "selection_rationale": request.selection_rationale,
            "task_selection": {
                **self.curriculum_context(),
                "arm_ids": list(arms),
                "num_arms": len(arms),
                "num_unseen": len(unseen),
            },
        }


def activesaddler_task_selection(
    *,
    batch_size: int,
    seed: int,
    settings: Mapping[str, JsonValue] | None = None,
) -> ActiveSaddlerTaskSelectionPolicy:
    """Build the policy from ``optimization.task_selection.settings``; registered in the default registry."""
    if settings is None:
        raise ValueError("optimization.task_selection.settings is required for 'activesaddler'")
    expected = {
        "softmax_temperature",
        "min_prob",
        "ema_eta",
        "pattern_extraction_timeout_seconds",
        "arm_scoring_timeout_seconds",
    }
    missing = sorted(expected - settings.keys())
    extra = sorted(settings.keys() - expected)
    if missing or extra:
        raise ValueError(
            f"Invalid keys at optimization.task_selection.settings for activesaddler: missing={missing}, extra={extra}"
        )
    path = "optimization.task_selection.settings"
    return ActiveSaddlerTaskSelectionPolicy(
        batch_size=batch_size,
        seed=seed,
        softmax_temperature=_setting_number(settings["softmax_temperature"], f"{path}.softmax_temperature"),
        min_prob=_setting_number(settings["min_prob"], f"{path}.min_prob"),
        ema_eta=_setting_number(settings["ema_eta"], f"{path}.ema_eta"),
        pattern_extraction_timeout_seconds=_setting_number(
            settings["pattern_extraction_timeout_seconds"],
            f"{path}.pattern_extraction_timeout_seconds",
        ),
        arm_scoring_timeout_seconds=_setting_number(settings["arm_scoring_timeout_seconds"], f"{path}.arm_scoring_timeout_seconds"),
    )


def _setting_number(value: JsonValue, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    return float(value)


def _change(change: CurriculumChange, iteration: int, **payload: object) -> dict[str, JsonValue]:
    return {
        "schema_version": CURRICULUM_SCHEMA_VERSION,
        "change": change,
        "iteration": iteration,
        **cast(dict[str, JsonValue], payload),
    }


def _observation_changes(
    state: CurriculumState,
    iteration: int,
    batch_case_ids: Sequence[str],
    after_evaluation_id: str | None,
) -> tuple[StatePayload, ...]:
    observations = pattern_observations(
        state,
        iteration=iteration,
        batch_case_ids=batch_case_ids,
        after_evaluation_id=after_evaluation_id,
    )
    if not observations:
        return ()
    return (_change("observations_recorded", iteration, observations=observations),)


def _apply_change(
    payload: Mapping[str, JsonValue],
    patterns: dict[str, FailurePattern],
    decisions: dict[int, Mapping[str, JsonValue]],
) -> None:
    if payload.get("schema_version") != CURRICULUM_SCHEMA_VERSION:
        raise ValueError(f"Unknown curriculum schema: {payload.get('schema_version')}")
    change = payload.get("change")
    if change not in _CHANGES:
        raise ValueError(f"Unknown curriculum change: {change}")
    iteration = _integer(payload.get("iteration"), "curriculum iteration")
    if change == "patterns_extracted":
        for raw in _objects(payload.get("new_patterns"), "new_patterns"):
            pattern_id = _string(raw.get("pattern_id"), "pattern_id")
            if pattern_id in patterns:
                raise ValueError(f"Duplicate curriculum pattern: {pattern_id}")
            patterns[pattern_id] = FailurePattern(
                pattern_id=pattern_id,
                label=_string(raw.get("label"), "pattern label"),
                created_iteration=iteration,
            )
        for raw in _objects(payload.get("tags"), "tags"):
            pattern_id = _string(raw.get("pattern_id"), "tag pattern_id")
            if pattern_id not in patterns:
                raise ValueError(f"Curriculum tag references an unknown pattern: {pattern_id}")
            tag = PatternTag(
                case_id=_string(raw.get("case_id"), "tag case_id"),
                candidate_id=_string(raw.get("candidate_id"), "tag candidate_id"),
                evaluation_id=_string(raw.get("evaluation_id"), "tag evaluation_id"),
                source=cast(PatternSource, _string(raw.get("source"), "tag source")),
                iteration=iteration,
                root_cause=_string(raw.get("root_cause"), "tag root_cause"),
            )
            pattern = patterns[pattern_id]
            key = (tag.case_id, tag.candidate_id, tag.evaluation_id)
            if any((item.case_id, item.candidate_id, item.evaluation_id) == key for item in pattern.tags):
                continue
            patterns[pattern_id] = replace(pattern, tags=(*pattern.tags, tag))
    elif change == "observations_recorded":
        for raw in _objects(payload.get("observations"), "observations"):
            pattern_id = _string(raw.get("pattern_id"), "observation pattern_id")
            if pattern_id not in patterns:
                raise ValueError(f"Curriculum observation references an unknown pattern: {pattern_id}")
            active = raw.get("active")
            if isinstance(active, bool) or not isinstance(active, (int, float)):
                raise TypeError("Curriculum observation activity must be numeric")
            observation = PatternObservation(
                iteration=iteration,
                active=float(active),
                evaluated_case_ids=_strings(raw.get("evaluated_case_ids"), "evaluated_case_ids"),
                tagged_case_ids=_optional_strings(raw.get("tagged_case_ids"), "tagged_case_ids"),
            )
            pattern = patterns[pattern_id]
            patterns[pattern_id] = replace(pattern, observations=(*pattern.observations, observation))
    elif change == "arm_decision":
        decisions[iteration] = dict(payload)
    else:
        for raw in _objects(payload.get("scores"), "scores"):
            pattern_id = _string(raw.get("pattern_id"), "score pattern_id")
            if pattern_id not in patterns:
                raise ValueError(f"Curriculum score references an unknown pattern: {pattern_id}")
            score = ArmScore(
                iteration=iteration,
                severity=_number(raw.get("severity"), "severity"),
                fixability=_number(raw.get("fixability"), "fixability"),
                breadth=_number(raw.get("breadth"), "breadth"),
                side_effect=_number(raw.get("side_effect"), "side_effect"),
                rationale=_string(raw.get("rationale"), "score rationale"),
            )
            pattern = patterns[pattern_id]
            retained = tuple(item for item in pattern.scores if item.iteration != iteration)
            patterns[pattern_id] = replace(pattern, scores=(*retained, score))


def _objects(value: object, label: str) -> list[Mapping[str, JsonValue]]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f"Curriculum {label} must be a list of objects")
    return [cast(Mapping[str, JsonValue], item) for item in value]


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"Curriculum {label} must be a non-empty string")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Curriculum {label} must be a non-empty list of strings")
    return tuple(cast(Sequence[str], value))


def _optional_strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Curriculum {label} must be a list of strings")
    return tuple(cast(Sequence[str], value))


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{label} must be a nonnegative integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Curriculum {label} must be numeric")
    return float(value)


def _arm_decision_failure_reason(result: SessionResult) -> str | None:
    output = result.structured_output
    if output is None:
        return "Arm decision session produced no structured output"
    if output.get("action") not in {"pull", "draw"}:
        return "Arm decision action must be 'pull' or 'draw'"
    rationale = output.get("rationale")
    if not isinstance(rationale, str) or not rationale:
        return "Arm decision rationale must be a non-empty string"
    return None


def _arm_scoring_failure_reason(result: SessionResult, arm_ids: Sequence[str]) -> str | None:
    try:
        scores = _output_list(result, "scores")
    except TypeError as error:
        return str(error)
    scored: list[str] = []
    for index, item in enumerate(scores):
        if not isinstance(item, Mapping):
            return f"Arm score {index} must be an object"
        pattern_id = item.get("pattern_id")
        if not isinstance(pattern_id, str):
            return f"Arm score {index} requires a pattern_id"
        for axis in ("severity", "fixability", "breadth", "side_effect"):
            value = item.get(axis)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                return f"Arm score {index} {axis} must be a number in [0, 1]"
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale:
            return f"Arm score {index} requires a rationale"
        scored.append(pattern_id)
    if len(set(scored)) != len(scored):
        return "Every arm must be scored exactly once"
    if set(scored) != set(arm_ids):
        missing = sorted(set(arm_ids) - set(scored))
        unknown = sorted(set(scored) - set(arm_ids))
        return f"Arm scores must cover exactly the current arms: missing={missing}, unknown={unknown}"
    return None


def _pattern_extraction_failure_reason(
    result: SessionResult,
    pre_patch: Sequence[str],
    post_patch: Sequence[str],
    existing_pattern_ids: Sequence[str],
) -> str | None:
    output = result.structured_output
    if output is None:
        return "Pattern extraction session produced no structured output"
    new_patterns = output.get("new_patterns")
    tags = output.get("tags")
    if not isinstance(new_patterns, list) or not isinstance(tags, list):
        return "Pattern extraction output requires new_patterns and tags lists"
    keys: list[str] = []
    for index, item in enumerate(new_patterns):
        if not isinstance(item, Mapping):
            return f"New pattern {index} must be an object"
        key = item.get("key")
        label = item.get("label")
        if not isinstance(key, str) or not key or not isinstance(label, str) or not label:
            return f"New pattern {index} requires a non-empty key and label"
        keys.append(key)
    if len(set(keys)) != len(keys):
        return "New pattern keys must be unique"
    if set(keys) & set(existing_pattern_ids):
        return "New pattern keys must not reuse existing pattern IDs"
    allowed_refs = set(keys) | set(existing_pattern_ids)
    failures = {"pre_patch": set(pre_patch), "post_patch": set(post_patch)}
    referenced: set[str] = set()
    for index, item in enumerate(tags):
        if not isinstance(item, Mapping):
            return f"Pattern tag {index} must be an object"
        source = item.get("source")
        case_id = item.get("case_id")
        if source not in failures:
            return f"Pattern tag {index} source must be pre_patch or post_patch"
        if case_id not in failures[cast(str, source)]:
            return f"Pattern tag {index} case {case_id!r} is not a {source} failure of this iteration"
        refs = item.get("pattern_refs")
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs):
            return f"Pattern tag {index} pattern_refs must be a non-empty list of strings"
        if len(set(refs)) != len(refs):
            return f"Pattern tag {index} pattern_refs must be unique"
        unknown = sorted(set(cast(list[str], refs)) - allowed_refs)
        if unknown:
            return f"Pattern tag {index} references unknown patterns: {unknown}"
        root_cause = item.get("root_cause")
        if not isinstance(root_cause, str) or not root_cause:
            return f"Pattern tag {index} requires a root_cause"
        referenced.update(cast(list[str], refs))
    untagged = sorted(set(keys) - referenced)
    if untagged:
        return f"New patterns must tag at least one failure: {untagged}"
    return None


def _pattern_extraction_change(result: SessionResult, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    output = result.structured_output
    if output is None:
        raise ValueError("Pattern extraction session produced no structured output")
    iteration = _integer(payload.get("iteration"), "extraction iteration")
    candidate_id = _string(payload.get("candidate_id"), "extraction candidate_id")
    key_to_id: dict[str, str] = {}
    new_patterns: list[JsonValue] = []
    for item in cast(list[Mapping[str, str]], output.get("new_patterns")):
        pattern_id = derive_pattern_id(iteration=iteration, key=item["key"], label=item["label"])
        key_to_id[item["key"]] = pattern_id
        new_patterns.append({"pattern_id": pattern_id, "key": item["key"], "label": item["label"]})
    provenance = {
        "pre_patch": (
            _string(payload.get("working_parent_candidate_id"), "working_parent_candidate_id"),
            _string(payload.get("train_before_evaluation_id"), "train_before_evaluation_id"),
        ),
        "post_patch": (candidate_id, _string(payload.get("train_after_evaluation_id"), "train_after_evaluation_id")),
    }
    tags: list[JsonValue] = []
    for item in cast(list[Mapping[str, JsonValue]], output.get("tags")):
        source = cast(str, item["source"])
        tagged_candidate_id, evaluation_id = provenance[source]
        for ref in cast(list[str], item["pattern_refs"]):
            tags.append(
                {
                    "pattern_id": key_to_id.get(ref, ref),
                    "case_id": item["case_id"],
                    "candidate_id": tagged_candidate_id,
                    "evaluation_id": evaluation_id,
                    "source": source,
                    "root_cause": item["root_cause"],
                }
            )
    return _change("patterns_extracted", iteration, candidate_id=candidate_id, new_patterns=new_patterns, tags=tags)


def _case_ids(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, Mapping) or not isinstance(item.get("case_id"), str) for item in value
    ):
        raise TypeError(f"Curriculum {label} must be a list of case records")
    return tuple(cast(str, item["case_id"]) for item in value)


def _unique_case_ids(cases: Sequence[Case]) -> tuple[str, ...]:
    if not cases:
        raise ValueError("Cannot select from an empty training set")
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Training cases must have unique IDs")
    return case_ids


def _output_string(result: SessionResult, key: str) -> str:
    output = result.structured_output
    value = output.get(key) if output is not None else None
    if not isinstance(value, str) or not value:
        raise TypeError(f"Session output {key!r} must be a non-empty string")
    return value


def _output_list(result: SessionResult, key: str) -> list[JsonValue]:
    output = result.structured_output
    if output is None or not isinstance(output.get(key), list):
        raise TypeError(f"Session output {key!r} must be a list")
    return cast(list[JsonValue], output[key])
