"""Failure-pattern curriculum state for the ActiveSaddler task-selection policy.

The curriculum treats every failure pattern that owns at least one training
case as a bandit arm. All state is folded from the append-only event log:
pattern registrations, tags, observations, arm decisions, and arm scores are
recorded as ``ExtensionStateChanged`` events in the ``autosaddler.curriculum``
namespace, executed cases come from ``BatchSampled`` events, and probe points
come from completed training evaluations. Nothing here performs I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from autosaddler.v2.core.domain import JsonValue, canonical_json, sha256_digest
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.core.serde import evaluation_from

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
