from __future__ import annotations

import time
from collections.abc import Mapping

from autosaddler.v2.core.domain import Cost, Observation, canonical_json, sha256_digest
from autosaddler.v2.core.metrics import EventModelUsageSink
from autosaddler.v2.core.serde import observation_from, record
from autosaddler.v2.prompting.models import Usage
from autosaddler.v2.storage.local import LocalRunStore


class EventEvaluationAttemptSink:
    def __init__(
        self,
        *,
        store: LocalRunStore,
        evaluation_operation_id: str,
        metrics_context: Mapping[str, object] | None = None,
    ) -> None:
        self.store = store
        self.evaluation_operation_id = evaluation_operation_id
        self.metrics_context = dict(metrics_context or {})
        self._started_at: dict[str, float] = {}
        self._incomplete_timing: set[str] = set()
        self._usage_sinks: dict[str, EventModelUsageSink] = {}

    def pending(
        self,
        *,
        candidate_id: str,
        case_id: str,
        repetition: int,
    ) -> tuple[str, int] | None:
        for event in reversed(self.store.events_of_type("EvaluationAttemptStarted")):
            if self._matches(event.payload, candidate_id, case_id, repetition) and self._terminal(event.operation_id) is None:
                attempt_number = event.payload.get("attempt_number")
                if not isinstance(attempt_number, int) or isinstance(attempt_number, bool):
                    raise TypeError("Evaluation attempt_number must be an integer")
                self._started_at[event.operation_id] = time.monotonic()
                self._incomplete_timing.add(event.operation_id)
                return event.operation_id, attempt_number
        return None

    def completed(
        self,
        *,
        candidate_id: str,
        case_id: str,
        repetition: int,
    ) -> Observation | None:
        for event in reversed(self.store.events_of_type("EvaluationAttemptCompleted")):
            if self._matches(event.payload, candidate_id, case_id, repetition):
                return observation_from(event.payload.get("observation"))
        return None

    def start(
        self,
        *,
        candidate_id: str,
        case_id: str,
        repetition: int,
    ) -> tuple[str, int]:
        starts = [
            event
            for event in self.store.events_of_type("EvaluationAttemptStarted")
            if self._matches(event.payload, candidate_id, case_id, repetition)
        ]
        for event in starts:
            if self._terminal(event.operation_id) is None:
                timing = self._timing(event.operation_id)
                self.store.append(
                    "EvaluationAttemptFailed",
                    event.operation_id,
                    {
                        **dict(event.payload),
                        "error_kind": "interrupted",
                        "cost": record(Cost()),
                        **timing,
                    },
                )
        attempt_number = len(starts) + 1
        attempt_id = sha256_digest(
            canonical_json(
                {
                    "evaluation_operation_id": self.evaluation_operation_id,
                    "candidate_id": candidate_id,
                    "case_id": case_id,
                    "repetition": repetition,
                    "attempt_number": attempt_number,
                }
            )
        )
        self.store.append(
            "EvaluationAttemptStarted",
            attempt_id,
            {
                "evaluation_operation_id": self.evaluation_operation_id,
                "candidate_id": candidate_id,
                "case_id": case_id,
                "repetition": repetition,
                "attempt_number": attempt_number,
            },
        )
        self._started_at[attempt_id] = time.monotonic()
        self._incomplete_timing.discard(attempt_id)
        return attempt_id, attempt_number

    def complete(self, attempt_id: str, observation: Observation, cost: Cost) -> None:
        started = self.store.find("EvaluationAttemptStarted", attempt_id)
        if started is None:
            raise RuntimeError(f"Evaluation attempt was not started: {attempt_id}")
        timing = self._timing(attempt_id)
        self.store.append(
            "EvaluationAttemptCompleted",
            attempt_id,
            {
                **dict(started.payload),
                "observation": record(observation),
                "cost": record(cost),
                **timing,
            },
        )

    def fail(self, attempt_id: str, error_kind: str, cost: Cost) -> None:
        if not error_kind:
            raise ValueError("Evaluation attempt failure requires an error kind")
        started = self.store.find("EvaluationAttemptStarted", attempt_id)
        if started is None:
            raise RuntimeError(f"Evaluation attempt was not started: {attempt_id}")
        timing = self._timing(attempt_id)
        self.store.append(
            "EvaluationAttemptFailed",
            attempt_id,
            {
                **dict(started.payload),
                "error_kind": error_kind,
                "cost": record(cost),
                **timing,
            },
        )

    def observe_usage(self, attempt_id: str, usage: Usage) -> None:
        started = self.store.find("EvaluationAttemptStarted", attempt_id)
        if started is None:
            raise RuntimeError(f"Evaluation attempt was not started: {attempt_id}")
        sink = self._usage_sinks.get(attempt_id)
        if sink is None:
            context = {
                **self.metrics_context,
                **dict(started.payload),
                "stage": "evaluation.task_agent",
            }
            existing = [
                event
                for event in self.store.events_of_type("ModelUsageObserved")
                if event.payload.get("attempt_operation_id") == attempt_id
            ]
            if existing and isinstance(existing[0].payload.get("run_invocation_id"), str):
                context["run_invocation_id"] = existing[0].payload["run_invocation_id"]
            sink = EventModelUsageSink(
                store=self.store,
                attempt_operation_id=attempt_id,
                context=context,
            )
            self._usage_sinks[attempt_id] = sink
        sink.observe(usage)

    def _matches(
        self,
        payload,
        candidate_id: str,
        case_id: str,
        repetition: int,
    ) -> bool:
        return (
            payload.get("evaluation_operation_id") == self.evaluation_operation_id
            and payload.get("candidate_id") == candidate_id
            and payload.get("case_id") == case_id
            and payload.get("repetition") == repetition
        )

    def _terminal(self, attempt_id: str):
        return self.store.find("EvaluationAttemptCompleted", attempt_id) or self.store.find(
            "EvaluationAttemptFailed", attempt_id
        )

    def _timing(self, attempt_id: str) -> dict[str, float | bool]:
        started = self._started_at.pop(attempt_id, None)
        incomplete = attempt_id in self._incomplete_timing or started is None
        self._incomplete_timing.discard(attempt_id)
        return {
            "wall_seconds": time.monotonic() - started if started is not None else 0.0,
            "wall_seconds_incomplete": incomplete,
        }