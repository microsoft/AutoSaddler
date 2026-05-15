from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Split: TypeAlias = Literal["train", "development", "test"]
Disposition: TypeAlias = Literal["success", "task_failure", "execution_error", "invalid"]
EvaluationPurpose: TypeAlias = Literal["train_before", "train_after", "development", "test"]
CandidateStatus: TypeAlias = Literal["proposed", "evaluated", "declined", "accepted", "selected"]


def to_json_value(value: Any) -> JsonValue:
    """Convert a v0.2 record into JSON-compatible values without lossy repr fallbacks."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON records cannot contain non-finite floats")
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        converted = [to_json_value(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_value(item) for item in value]
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(to_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_digest(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def freeze_json_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    """Copy a JSON mapping so callers cannot mutate a frozen record through an alias."""
    copied = json.loads(canonical_json(value))
    assert isinstance(copied, dict)
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    uri: str
    kind: str
    sha256: str | None = None
    bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.uri or Path(self.uri).is_absolute():
            raise ValueError("ArtifactRef.uri must be a non-empty run-relative path")
        if not self.kind:
            raise ValueError("ArtifactRef.kind must be non-empty")
        if self.sha256 is not None and not self.sha256.startswith("sha256:"):
            raise ValueError("ArtifactRef.sha256 must use the sha256:<hex> form")
        if self.bytes is not None and self.bytes < 0:
            raise ValueError("ArtifactRef.bytes cannot be negative")


@dataclass(frozen=True, slots=True)
class Cost:
    rollouts: int = 0
    sessions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    currency_amount: float | None = None

    def __post_init__(self) -> None:
        numeric = (self.rollouts, self.sessions, self.input_tokens, self.output_tokens, self.wall_seconds)
        if any(value < 0 for value in numeric):
            raise ValueError("Cost fields cannot be negative")
        if self.currency_amount is not None and self.currency_amount < 0:
            raise ValueError("Cost.currency_amount cannot be negative")


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    split: Split
    payload: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("Case.case_id must be non-empty")
        object.__setattr__(self, "payload", freeze_json_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class ChangeSummary:
    changed_units: tuple[str, ...]
    added: int
    removed: int
    labels: tuple[str, ...] = ()
    diff: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if not self.changed_units:
            raise ValueError("A finalized child must contain at least one changed unit")
        if self.added < 0 or self.removed < 0:
            raise ValueError("ChangeSummary line counts cannot be negative")


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    parent_ids: tuple[str, ...]
    space: str
    artifact: ArtifactRef
    change: ChangeSummary | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("sha256:"):
            raise ValueError("Candidate identity must be content-derived sha256:<hex>")
        if not self.space:
            raise ValueError("Candidate.space must be non-empty")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("Candidate.parent_ids must be unique")
        if self.parent_ids and self.change is None:
            raise ValueError("A child candidate requires a change summary")
        if not self.parent_ids and self.change is not None:
            raise ValueError("A seed candidate cannot have a change summary")


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    candidate_id: str
    case_id: str
    split: Split
    repetition: int
    disposition: Disposition
    score: float | None
    objectives: Mapping[str, float]
    output: ArtifactRef | None
    trace: ArtifactRef | None
    attempts: int
    cost: Cost
    metadata: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.observation_id.startswith("sha256:"):
            raise ValueError("Observation.observation_id must be content-derived")
        if not self.candidate_id.startswith("sha256:"):
            raise ValueError("Observation.candidate_id must be content-derived")
        if not self.case_id:
            raise ValueError("Observation.case_id must be non-empty")
        if self.repetition < 0:
            raise ValueError("Observation.repetition cannot be negative")
        if self.attempts < 1:
            raise ValueError("Observation.attempts must be at least one")
        valid = self.disposition in ("success", "task_failure")
        if valid and self.score is None:
            raise ValueError("Valid observations require a score")
        if not valid and self.score is not None:
            raise ValueError("Invalid observations must use score=None")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("Observation.score must be finite")
        if any(not math.isfinite(value) for value in self.objectives.values()):
            raise ValueError("Observation objectives must be finite")
        object.__setattr__(self, "objectives", MappingProxyType(dict(self.objectives)))
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    @property
    def is_valid(self) -> bool:
        return self.disposition in ("success", "task_failure")

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        case_id: str,
        split: Split,
        repetition: int,
        disposition: Disposition,
        score: float | None,
        evaluator_fingerprint: str,
        objectives: Mapping[str, float] | None = None,
        output: ArtifactRef | None = None,
        trace: ArtifactRef | None = None,
        attempts: int = 1,
        cost: Cost | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> "Observation":
        identity = sha256_digest(
            canonical_json(
                {
                    "candidate_id": candidate_id,
                    "case_id": case_id,
                    "repetition": repetition,
                    "evaluator_fingerprint": evaluator_fingerprint,
                }
            )
        )
        return cls(
            observation_id=identity,
            candidate_id=candidate_id,
            case_id=case_id,
            split=split,
            repetition=repetition,
            disposition=disposition,
            score=score,
            objectives=objectives or {},
            output=output,
            trace=trace,
            attempts=attempts,
            cost=cost or Cost(rollouts=attempts),
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class Evaluation:
    evaluation_id: str
    candidate_id: str
    split: Split
    purpose: EvaluationPurpose
    iteration: int | None
    requested_case_ids: tuple[str, ...]
    observations: tuple[Observation, ...]
    artifact_dir: ArtifactRef

    def __post_init__(self) -> None:
        if not self.evaluation_id:
            raise ValueError("Evaluation.evaluation_id must be non-empty")
        expected_split = {
            "train_before": "train",
            "train_after": "train",
            "development": "development",
            "test": "test",
        }[self.purpose]
        if self.split != expected_split:
            raise ValueError(f"Evaluation purpose {self.purpose!r} requires split {expected_split!r}")
        if self.purpose == "test" and self.iteration is not None:
            raise ValueError("Post-optimization test evaluations cannot belong to an optimization iteration")
        if self.purpose != "test" and self.iteration is None:
            raise ValueError("Optimization evaluations require an iteration")
        if len(set(self.requested_case_ids)) != len(self.requested_case_ids):
            raise ValueError("Evaluation.requested_case_ids must be unique")
        keys = [(observation.case_id, observation.repetition) for observation in self.observations]
        if len(set(keys)) != len(keys):
            raise ValueError("Evaluation observations must be unique by case_id and repetition")
        observed_cases = {observation.case_id for observation in self.observations}
        if observed_cases != set(self.requested_case_ids):
            raise ValueError("Evaluation observations must cover exactly the requested cases")
        for observation in self.observations:
            if observation.candidate_id != self.candidate_id or observation.split != self.split:
                raise ValueError("Evaluation observations must match its candidate and split")

    @property
    def aggregate_score(self) -> float | None:
        scores = [observation.score for observation in self.observations if observation.is_valid]
        valid_scores = [score for score in scores if score is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None

    @property
    def attempted_rollouts(self) -> int:
        return sum(observation.attempts for observation in self.observations)

    @property
    def valid_rollouts(self) -> int:
        return sum(1 for observation in self.observations if observation.is_valid)


@dataclass(frozen=True, slots=True)
class EvaluationRef:
    evaluation_id: str
    split: Split
    purpose: EvaluationPurpose
    aggregate_score: float | None
    valid_case_count: int
    attempted_case_count: int
    executed_observations: int


@dataclass(frozen=True, slots=True)
class PatchIntent:
    target_case_ids: tuple[str, ...]
    approach: str
    patch_labels: tuple[str, ...]
    changed_units: tuple[str, ...]
    diagnosis: str


@dataclass(frozen=True, slots=True)
class PatchVerdict:
    before_score: float
    after_score: float
    compared_case_ids: tuple[str, ...]
    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class Lesson:
    scope: Literal["global", "case", "component"]
    statement: str
    evidence_case_ids: tuple[str, ...]
    source_candidate_id: str


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: str
    parent_ids: tuple[str, ...]
    batch_case_ids: tuple[str, ...]
    patch_intent: PatchIntent | None
    verification_failure: str
    session_id: str