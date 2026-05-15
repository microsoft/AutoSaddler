from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol

from autosaddler.v2.core.domain import JsonValue, to_json_value
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.core.serde import record
from autosaddler.v2.prompting.models import Usage

logger = logging.getLogger(__name__)

METRICS_RECORD_SCHEMA_VERSION = "autosaddler-metrics-record/v2"
METRICS_SUMMARY_SCHEMA_VERSION = "autosaddler-metrics-summary/v2"
_STAGE_METADATA_FIELDS = (
    "run_invocation_id",
    "logical_operation_id",
    "session_id",
    "session_kind",
    "provider",
    "attempt",
    "iteration",
    "candidate_id",
    "evaluation_operation_id",
    "evaluation_purpose",
    "split",
    "case_id",
    "repetition",
    "attempt_number",
    "error_type",
    "error_kind",
    "wall_seconds_incomplete",
)


class EventAppender(Protocol):
    def append(self, event_type: str, operation_id: str, payload: Mapping[str, JsonValue]) -> RunEvent: ...


class EventModelUsageSink:
    """Durably append attempt-scoped model usage without affecting the paid call."""

    def __init__(
        self,
        *,
        store: EventAppender,
        attempt_operation_id: str,
        context: Mapping[str, JsonValue],
    ) -> None:
        self.store = store
        self.attempt_operation_id = attempt_operation_id
        self.context = dict(context)
        self.observed_count = 0
        self.write_errors: list[str] = []

    def observe(self, usage: Usage) -> None:
        sequence = self.observed_count
        self.observe_at(sequence, usage)

    def observe_at(self, sequence: int, usage: Usage) -> None:
        if sequence < 0:
            raise ValueError("Model usage sequence cannot be negative")
        self.observed_count = max(self.observed_count, sequence + 1)
        try:
            self.store.append(
                "ModelUsageObserved",
                f"{self.attempt_operation_id}:model-usage:{sequence}",
                {
                    **self.context,
                    "attempt_operation_id": self.attempt_operation_id,
                    "usage_sequence": sequence,
                    "usage": record(usage),
                },
            )
        except Exception as error:
            if getattr(self.store, "transition_hook_error", None) is error:
                raise
            self.write_errors.append(f"{type(error).__name__}: {error}")
            logger.warning("Failed to record v0.2 model usage", exc_info=True)


def metrics_records(events: Sequence[RunEvent]) -> list[dict[str, JsonValue]]:
    failed_sessions = {
        event.operation_id: event
        for event in events
        if event.event_type == "SessionFailed"
    }
    records: list[dict[str, JsonValue]] = []
    best_development_score: float | None = None
    for event in events:
        if event.event_type == "ModelUsageObserved":
            usage_value = event.payload.get("usage")
            if not isinstance(usage_value, Mapping):
                raise TypeError("ModelUsageObserved usage must be a mapping")
            usage = dict(usage_value)
            usage = _canonical_usage(usage)
            usage["nonreasoning_output_tokens"] = (
                int(usage.get("output_tokens") or 0) - int(usage.get("reasoning_tokens") or 0)
            )
            attempt_operation_id = str(event.payload.get("attempt_operation_id") or "")
            failed = failed_sessions.get(attempt_operation_id)
            if failed is not None and usage.get("status") == "success":
                usage["status"] = str(failed.payload.get("status") or "failed")
                usage["error_type"] = _error_type(failed.payload)
                usage["usage_incomplete"] = True
            records.append(
                _json_record(
                    {
                        "schema_version": METRICS_RECORD_SCHEMA_VERSION,
                        "source_event_sequence": event.sequence,
                        "source_event_type": event.event_type,
                        "recorded_at": event.timestamp,
                        "run_id": event.run_id,
                        "type": "model_call",
                        **_without(event.payload, "usage"),
                        **usage,
                    }
                )
            )
            continue
        stage = _stage_record(event)
        if stage is not None:
            records.append(stage)
        if event.event_type == "EvaluationCompleted":
            evaluation = event.payload.get("evaluation")
            if isinstance(evaluation, Mapping) and evaluation.get("purpose") == "development":
                score = _evaluation_score(evaluation)
                if score is not None:
                    if best_development_score is None:
                        best_development_score = score
                    elif score > best_development_score:
                        best_development_score = score
                        records.append(
                            _json_record(
                                {
                                    "schema_version": METRICS_RECORD_SCHEMA_VERSION,
                                    "source_event_sequence": event.sequence,
                                    "source_event_type": event.event_type,
                                    "recorded_at": event.timestamp,
                                    "run_id": event.run_id,
                                    "type": "new_best",
                                    "candidate_id": evaluation.get("candidate_id"),
                                    "iteration": evaluation.get("iteration"),
                                    "development_score": score,
                                }
                            )
                        )
        if event.event_type in {"EvolutionRecorded", "ObservabilityDegraded"}:
            records.append(
                _json_record(
                    {
                        "schema_version": METRICS_RECORD_SCHEMA_VERSION,
                        "source_event_sequence": event.sequence,
                        "source_event_type": event.event_type,
                        "recorded_at": event.timestamp,
                        "run_id": event.run_id,
                        "type": _record_type(event),
                        **event.payload,
                    }
                )
            )
    return records


def summarize_metrics(events: Sequence[RunEvent]) -> dict[str, JsonValue]:
    records = metrics_records(events)
    totals = _new_totals()
    by_role: dict[str, dict[str, JsonValue]] = defaultdict(_new_totals)
    by_model: dict[str, dict[str, JsonValue]] = defaultdict(_new_totals)
    by_stage: dict[str, dict[str, JsonValue]] = defaultdict(_new_totals)
    by_iteration: dict[str, dict[str, JsonValue]] = defaultdict(_new_totals)
    by_candidate: dict[str, dict[str, JsonValue]] = defaultdict(_new_totals)
    by_purpose: dict[str, dict[str, JsonValue]] = defaultdict(_new_totals)
    by_agent_scope: dict[str, dict[str, JsonValue]] = defaultdict(_new_totals)
    provider_costs: dict[str, float] = defaultdict(float)
    active_wall_seconds = 0.0
    accepted_candidates: set[str] = set()
    new_best_candidates: set[str] = set()
    attempted_rollouts = 0
    valid_rollouts = 0
    degraded_count = 0

    for record_value in records:
        record_type = record_value.get("type")
        if record_type == "model_call":
            _add_model_call(totals, record_value)
            _add_model_call(by_role[str(record_value.get("role") or "unknown")], record_value)
            _add_model_call(by_model[str(record_value.get("model") or "unknown")], record_value)
            _add_model_call(by_stage[str(record_value.get("stage") or "unknown")], record_value)
            _add_model_call(by_iteration[_group_key(record_value.get("iteration"), "post_loop")], record_value)
            _add_model_call(by_candidate[_group_key(record_value.get("candidate_id"), "unattributed")], record_value)
            _add_model_call(by_purpose[_group_key(record_value.get("evaluation_purpose"), "not_evaluation")], record_value)
            _add_model_call(by_agent_scope[str(record_value.get("agent_scope") or "unknown")], record_value)
            cost = record_value.get("provider_cost")
            unit = record_value.get("provider_cost_unit")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and isinstance(unit, str):
                provider_costs[unit] += float(cost)
            nano_aiu = record_value.get("provider_nano_aiu")
            if isinstance(nano_aiu, (int, float)) and not isinstance(nano_aiu, bool):
                provider_costs["nano_aiu"] += float(nano_aiu)
                provider_costs["ai_credit"] += float(nano_aiu) / 1_000_000_000
            else:
                ai_credits = record_value.get("provider_ai_credits")
                if isinstance(ai_credits, (int, float)) and not isinstance(ai_credits, bool):
                    provider_costs["ai_credit"] += float(ai_credits)
        elif record_type == "stage":
            stage = str(record_value.get("stage") or "unknown")
            wall_seconds = _number(record_value.get("wall_seconds"))
            by_stage[stage]["wall_seconds"] = _number(by_stage[stage]["wall_seconds"]) + wall_seconds
            if stage == "run":
                active_wall_seconds += wall_seconds
        elif record_type == "candidate_outcome":
            candidate_id = record_value.get("candidate_id")
            if record_value.get("accepted") is True and isinstance(candidate_id, str):
                accepted_candidates.add(candidate_id)
        elif record_type == "new_best":
            candidate_id = record_value.get("candidate_id")
            if isinstance(candidate_id, str):
                new_best_candidates.add(candidate_id)
        elif record_type == "observability_degraded":
            degraded_count += 1

    for event in events:
        if event.event_type == "EvaluationAttemptStarted":
            attempted_rollouts += 1
        elif event.event_type == "EvaluationAttemptCompleted":
            observation = event.payload.get("observation")
            if isinstance(observation, Mapping) and observation.get("score") is not None:
                valid_rollouts += 1

    return _json_record(
        {
            "schema_version": METRICS_SUMMARY_SCHEMA_VERSION,
            "source_event_count": len(events),
            "model_usage": totals,
            "model_usage_by_role": dict(by_role),
            "model_usage_by_model": dict(by_model),
            "metrics_by_stage": dict(by_stage),
            "model_usage_by_iteration": dict(by_iteration),
            "model_usage_by_candidate": dict(by_candidate),
            "model_usage_by_evaluation_purpose": dict(by_purpose),
            "model_usage_by_agent_scope": dict(by_agent_scope),
            "provider_reported_cost_by_unit": dict(provider_costs),
            "active_wall_seconds": active_wall_seconds,
            "attempted_rollouts": attempted_rollouts,
            "valid_rollouts": valid_rollouts,
            "accepted_candidate_count": len(accepted_candidates),
            "new_best_candidate_count": len(new_best_candidates),
            "observability_degraded_count": degraded_count,
            "per_accepted_candidate": _per_success(
                int(totals["total_tokens"]), provider_costs, active_wall_seconds, len(accepted_candidates)
            ),
            "per_new_best_candidate": _per_success(
                int(totals["total_tokens"]), provider_costs, active_wall_seconds, len(new_best_candidates)
            ),
        }
    )


def _stage_record(event: RunEvent) -> dict[str, JsonValue] | None:
    stages = {
        "RunCompleted": "run",
        "RunFailed": "run",
        "RunInterrupted": "run",
        "IterationCompleted": "iteration",
        "SessionCompleted": "provider_session",
        "SessionFailed": "provider_session",
        "EvaluationAttemptCompleted": "evaluation_attempt",
        "EvaluationAttemptFailed": "evaluation_attempt",
        "EvidenceProviderCompleted": "evidence_provider",
        "EvidenceProviderFailed": "evidence_provider",
        "OptimizationCompleted": "finalization",
    }
    stage = stages.get(event.event_type)
    wall_seconds = event.payload.get("wall_seconds")
    if stage is None or not isinstance(wall_seconds, (int, float)) or isinstance(wall_seconds, bool):
        return None
    payload_stage = event.payload.get("stage")
    if isinstance(payload_stage, str):
        stage = payload_stage
    payload_status = event.payload.get("status")
    status = (
        payload_status
        if isinstance(payload_status, str)
        else "success" if event.event_type.endswith("Completed") else "failed"
    )
    return _json_record(
        {
            "schema_version": METRICS_RECORD_SCHEMA_VERSION,
            "source_event_sequence": event.sequence,
            "source_event_type": event.event_type,
            "recorded_at": event.timestamp,
            "run_id": event.run_id,
            "type": "stage",
            "stage": stage,
            "wall_seconds": float(wall_seconds),
            "status": status,
            **{
                key: event.payload[key]
                for key in _STAGE_METADATA_FIELDS
                if key in event.payload
            },
        }
    )


def _record_type(event: RunEvent) -> str:
    if event.event_type == "EvolutionRecorded":
        return "candidate_outcome"
    return "observability_degraded"


def _evaluation_score(evaluation: Mapping[str, JsonValue]) -> float | None:
    observations = evaluation.get("observations")
    if not isinstance(observations, list):
        return None
    scores = [
        float(observation["score"])
        for observation in observations
        if isinstance(observation, dict)
        and isinstance(observation.get("score"), (int, float))
        and not isinstance(observation.get("score"), bool)
    ]
    return sum(scores) / len(scores) if scores else None


def _new_totals() -> dict[str, JsonValue]:
    return {
        "model_calls": 0,
        "successful_model_calls": 0,
        "failed_model_calls": 0,
        "input_tokens": 0,
        "uncached_input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "nonreasoning_output_tokens": 0,
        "total_tokens": 0,
        "duration_seconds": 0.0,
        "wall_seconds": 0.0,
        "provider_cost_by_unit": {},
    }


def _add_model_call(target: dict[str, JsonValue], record_value: Mapping[str, JsonValue]) -> None:
    target["model_calls"] = int(target["model_calls"]) + 1
    success = record_value.get("status") == "success"
    status_key = "successful_model_calls" if success else "failed_model_calls"
    target[status_key] = int(target[status_key]) + 1
    for key in (
        "input_tokens",
        "uncached_input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "nonreasoning_output_tokens",
        "total_tokens",
    ):
        target[key] = int(target[key]) + int(record_value.get(key) or 0)
    target["duration_seconds"] = _number(target["duration_seconds"]) + _number(record_value.get("duration_seconds"))
    provider_cost = record_value.get("provider_cost")
    provider_cost_unit = record_value.get("provider_cost_unit")
    if isinstance(provider_cost, (int, float)) and not isinstance(provider_cost, bool) and isinstance(
        provider_cost_unit, str
    ):
        costs = target["provider_cost_by_unit"]
        assert isinstance(costs, dict)
        costs[provider_cost_unit] = _number(costs.get(provider_cost_unit)) + float(provider_cost)
    nano_aiu = record_value.get("provider_nano_aiu")
    if isinstance(nano_aiu, (int, float)) and not isinstance(nano_aiu, bool):
        costs = target["provider_cost_by_unit"]
        assert isinstance(costs, dict)
        costs["nano_aiu"] = _number(costs.get("nano_aiu")) + float(nano_aiu)
        costs["ai_credit"] = _number(costs.get("ai_credit")) + float(nano_aiu) / 1_000_000_000
    else:
        ai_credits = record_value.get("provider_ai_credits")
        if isinstance(ai_credits, (int, float)) and not isinstance(ai_credits, bool):
            costs = target["provider_cost_by_unit"]
            assert isinstance(costs, dict)
            costs["ai_credit"] = _number(costs.get("ai_credit")) + float(ai_credits)


def _canonical_usage(usage: dict[str, JsonValue]) -> dict[str, JsonValue]:
    raw_input = int(usage.get("input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or raw_input + output)
    cached = int(usage.get("cached_input_tokens") or usage.get("cache_read_tokens") or 0)
    if "uncached_input_tokens" in usage:
        uncached = int(usage.get("uncached_input_tokens") or 0)
        canonical_input = raw_input
    elif total == raw_input + output:
        canonical_input = raw_input
        uncached = canonical_input - cached
    else:
        cache_creation = int(usage.get("cache_write_tokens") or 0)
        canonical_input = raw_input + cached + cache_creation
        uncached = raw_input + cache_creation
    usage.update(
        {
            "input_tokens": canonical_input,
            "cached_input_tokens": cached,
            "uncached_input_tokens": uncached,
            "total_tokens": canonical_input + output,
        }
    )
    usage.pop("cache_read_tokens", None)
    usage.pop("cache_write_tokens", None)
    return usage


def _per_success(
    total_tokens: int,
    provider_costs: Mapping[str, float],
    active_wall_seconds: float,
    count: int,
) -> dict[str, JsonValue]:
    if count == 0:
        return {
            "tokens": None,
            "provider_cost_by_unit": {unit: None for unit in provider_costs},
            "wall_seconds": None,
        }
    return {
        "tokens": total_tokens / count,
        "provider_cost_by_unit": {unit: value / count for unit, value in provider_costs.items()},
        "wall_seconds": active_wall_seconds / count,
    }


def _without(payload: Mapping[str, JsonValue], excluded: str) -> dict[str, JsonValue]:
    return {key: value for key, value in payload.items() if key != excluded}


def _error_type(payload: Mapping[str, JsonValue]) -> str:
    error = str(payload.get("error") or payload.get("status") or "failed")
    return error.split(":", maxsplit=1)[0]


def _group_key(value: JsonValue, missing: str) -> str:
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else missing


def _number(value: JsonValue) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _json_record(value: Mapping[str, object]) -> dict[str, JsonValue]:
    converted = to_json_value(value)
    assert isinstance(converted, dict)
    return converted