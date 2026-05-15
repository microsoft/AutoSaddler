from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from autosaddler.v2.core.domain import JsonValue, freeze_json_mapping

EVENT_SCHEMA_VERSION = "autosaddler-event/v1"
ITERATION_COMPLETION_SCHEMA_VERSION = "autosaddler-iteration-completion/v1"

KNOWN_EVENT_TYPES = frozenset(
    {
        "RunStarted",
        "RunForked",
        "RunResumed",
        "IterationStarted",
        "IterationCompleted",
        "BatchSampled",
        "DeferredWorkScheduled",
        "DeferredWorkCompleted",
        "DeferredWorkAbandoned",
        "SessionStarted",
        "SessionCompleted",
        "SessionFailed",
        "ModelUsageObserved",
        "ObservabilityDegraded",
        "CandidateFinalized",
        "MutationRejected",
        "EvaluationStarted",
        "EvaluationCompleted",
        "EvaluationFailed",
        "EvaluationAttemptStarted",
        "EvaluationAttemptCompleted",
        "EvaluationAttemptFailed",
        "EvidenceProviderStarted",
        "EvidenceProviderCompleted",
        "EvidenceProviderFailed",
        "AcceptanceDecided",
        "DevelopmentGateDecided",
        "EvolutionRecorded",
        "BudgetUpdated",
        "BestCandidateSelected",
        "OptimizationCompleted",
        "TestEvaluationStarted",
        "TestEvaluationCompleted",
        "RunInterrupted",
        "RunCompleted",
        "RunFailed",
        "ExtensionStateChanged",
    }
)


@dataclass(frozen=True, slots=True)
class RunEvent:
    schema_version: str
    sequence: int
    timestamp: str
    run_id: str
    event_type: str
    operation_id: str
    payload: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(f"Unknown event schema: {self.schema_version}")
        if self.sequence < 1:
            raise ValueError("Event sequence must be positive")
        if not self.run_id or not self.operation_id:
            raise ValueError("Event run_id and operation_id must be non-empty")
        if self.event_type not in KNOWN_EVENT_TYPES:
            raise ValueError(f"Unknown event type: {self.event_type}")
        object.__setattr__(self, "payload", freeze_json_mapping(self.payload))

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        run_id: str,
        event_type: str,
        operation_id: str,
        payload: Mapping[str, JsonValue],
    ) -> "RunEvent":
        return cls(
            schema_version=EVENT_SCHEMA_VERSION,
            sequence=sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            event_type=event_type,
            operation_id=operation_id,
            payload=payload,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunEvent":
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("Event payload must be a mapping")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            sequence=_integer(value.get("sequence"), "sequence"),
            timestamp=str(value.get("timestamp", "")),
            run_id=str(value.get("run_id", "")),
            event_type=str(value.get("event_type", "")),
            operation_id=str(value.get("operation_id", "")),
            payload=payload,
        )


def operation_id(run_id: str, *parts: object) -> str:
    if not run_id or not parts:
        raise ValueError("Operation IDs require a run ID and at least one part")
    encoded = ":".join(str(part).replace(":", "_") for part in parts)
    return f"{run_id}:{encoded}"


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Event {label} must be an integer")
    return value
