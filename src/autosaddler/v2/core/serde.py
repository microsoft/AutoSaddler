from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Candidate,
    ChangeSummary,
    Cost,
    Evaluation,
    JsonValue,
    Observation,
    to_json_value,
)
from autosaddler.v2.prompting.models import SessionResult, ToolCall, Usage


def record(value: Any) -> dict[str, JsonValue]:
    converted = to_json_value(value)
    if not isinstance(converted, dict):
        raise TypeError(f"Expected record-like value, found {type(value).__name__}")
    return converted


def artifact_from(value: object) -> ArtifactRef:
    item = _mapping(value, "artifact")
    return ArtifactRef(
        uri=_string(item, "uri"),
        kind=_string(item, "kind"),
        sha256=_optional_string(item, "sha256"),
        bytes=_optional_int(item, "bytes"),
    )


def cost_from(value: object) -> Cost:
    item = _mapping(value, "cost")
    return Cost(
        rollouts=_integer(item, "rollouts"),
        sessions=_integer(item, "sessions"),
        input_tokens=_integer(item, "input_tokens"),
        output_tokens=_integer(item, "output_tokens"),
        wall_seconds=_number(item, "wall_seconds"),
        currency_amount=_optional_number(item, "currency_amount"),
    )


def change_from(value: object) -> ChangeSummary:
    item = _mapping(value, "change")
    diff_value = item.get("diff")
    return ChangeSummary(
        changed_units=_strings(item, "changed_units"),
        added=_integer(item, "added"),
        removed=_integer(item, "removed"),
        labels=_strings(item, "labels"),
        diff=artifact_from(diff_value) if diff_value is not None else None,
    )


def candidate_from(value: object) -> Candidate:
    item = _mapping(value, "candidate")
    change_value = item.get("change")
    return Candidate(
        candidate_id=_string(item, "candidate_id"),
        parent_ids=_strings(item, "parent_ids"),
        space=_string(item, "space"),
        artifact=artifact_from(item.get("artifact")),
        change=change_from(change_value) if change_value is not None else None,
    )


def observation_from(value: object) -> Observation:
    item = _mapping(value, "observation")
    objectives = _mapping(item.get("objectives"), "observation.objectives")
    metadata = _mapping(item.get("metadata"), "observation.metadata")
    output = item.get("output")
    trace = item.get("trace")
    return Observation(
        observation_id=_string(item, "observation_id"),
        candidate_id=_string(item, "candidate_id"),
        case_id=_string(item, "case_id"),
        split=cast(Any, _string(item, "split")),
        repetition=_integer(item, "repetition"),
        disposition=cast(Any, _string(item, "disposition")),
        score=_optional_number(item, "score"),
        objectives={str(key): _finite_number(number, f"objective {key}") for key, number in objectives.items()},
        output=artifact_from(output) if output is not None else None,
        trace=artifact_from(trace) if trace is not None else None,
        attempts=_integer(item, "attempts"),
        cost=cost_from(item.get("cost")),
        metadata=cast(Mapping[str, JsonValue], metadata),
    )


def evaluation_from(value: object) -> Evaluation:
    item = _mapping(value, "evaluation")
    observations = item.get("observations")
    if not isinstance(observations, list):
        raise TypeError("evaluation.observations must be a list")
    return Evaluation(
        evaluation_id=_string(item, "evaluation_id"),
        candidate_id=_string(item, "candidate_id"),
        split=cast(Any, _string(item, "split")),
        purpose=cast(Any, _string(item, "purpose")),
        iteration=_optional_int(item, "iteration"),
        requested_case_ids=_strings(item, "requested_case_ids"),
        observations=tuple(observation_from(observation) for observation in observations),
        artifact_dir=artifact_from(item.get("artifact_dir")),
    )


def session_result_from(value: object) -> SessionResult:
    item = _mapping(value, "session_result")
    structured = item.get("structured_output")
    tool_values = item.get("tool_calls")
    usage_values = item.get("usage")
    if not isinstance(tool_values, list) or not isinstance(usage_values, list):
        raise TypeError("Session tool_calls and usage must be lists")
    tools = []
    for value_item in tool_values:
        tool = _mapping(value_item, "tool_call")
        tools.append(
            ToolCall(
                tool=_string(tool, "tool"),
                arguments=cast(Mapping[str, JsonValue], _mapping(tool.get("arguments"), "tool arguments")),
                result_preview=_optional_string(tool, "result_preview"),
            )
        )
    usages = []
    for value_item in usage_values:
        usage = _mapping(value_item, "usage")
        raw_input_tokens = _integer(usage, "input_tokens")
        output_tokens = _integer(usage, "output_tokens")
        cached_input_tokens = (
            _optional_int(usage, "cached_input_tokens")
            if "cached_input_tokens" in usage
            else _optional_int(usage, "cache_read_tokens")
        ) or 0
        legacy_cache_creation_tokens = _optional_int(usage, "cache_write_tokens") or 0
        provider_total_tokens = _optional_int(usage, "total_tokens")
        legacy_excluded_cache = (
            provider_total_tokens is not None
            and provider_total_tokens != raw_input_tokens + output_tokens
            and provider_total_tokens
            == raw_input_tokens + cached_input_tokens + legacy_cache_creation_tokens + output_tokens
        )
        input_tokens = (
            raw_input_tokens + cached_input_tokens + legacy_cache_creation_tokens
            if legacy_excluded_cache
            else raw_input_tokens
        )
        uncached_input_tokens = (
            _optional_int(usage, "uncached_input_tokens")
            if "uncached_input_tokens" in usage
            else input_tokens - cached_input_tokens
        )
        provider_metadata = dict(
            _mapping(usage.get("provider_metadata", {}), "usage provider metadata")
        )
        if legacy_cache_creation_tokens and "cache_creation_input_tokens" not in provider_metadata:
            provider_metadata["cache_creation_input_tokens"] = legacy_cache_creation_tokens
        usages.append(
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=_optional_string(usage, "model"),
                role=_optional_string(usage, "role") or "optimizer",
                cached_input_tokens=cached_input_tokens,
                uncached_input_tokens=uncached_input_tokens,
                reasoning_tokens=_optional_int(usage, "reasoning_tokens") or 0,
                total_tokens=input_tokens + output_tokens,
                provider_cost=_optional_number(usage, "provider_cost"),
                provider_cost_unit=_optional_string(usage, "provider_cost_unit"),
                provider_nano_aiu=_optional_number(usage, "provider_nano_aiu"),
                provider_ai_credits=_optional_number(usage, "provider_ai_credits"),
                duration_seconds=_optional_number(usage, "duration_seconds"),
                status=cast(Any, _optional_string(usage, "status") or "success"),
                error_type=_optional_string(usage, "error_type"),
                usage_incomplete=bool(usage.get("usage_incomplete", False)),
                provider_correlation_id=_optional_string(usage, "provider_correlation_id"),
                agent_id=_optional_string(usage, "agent_id"),
                agent_scope=cast(Any, _optional_string(usage, "agent_scope") or "unknown"),
                input_token_semantics=cast(
                    Any,
                    "excludes_cached_tokens"
                    if legacy_excluded_cache
                    else _optional_string(usage, "input_token_semantics") or "unknown",
                ),
                provider_reported_input_tokens=(
                    _optional_int(usage, "provider_reported_input_tokens")
                    if "provider_reported_input_tokens" in usage
                    else raw_input_tokens
                ),
                provider_reported_total_tokens=(
                    _optional_int(usage, "provider_reported_total_tokens")
                    if "provider_reported_total_tokens" in usage
                    else provider_total_tokens
                ),
                total_tokens_is_inferred=bool(usage.get("total_tokens_is_inferred", False)),
                configured_settings=cast(
                    Mapping[str, JsonValue],
                    _mapping(usage.get("configured_settings", {}), "usage configured settings"),
                ),
                reported_settings=cast(
                    Mapping[str, JsonValue],
                    _mapping(usage.get("reported_settings", {}), "usage reported settings"),
                ),
                provider_metadata=cast(Mapping[str, JsonValue], provider_metadata),
            )
        )
    return SessionResult(
        status=cast(Any, _string(item, "status")),
        structured_output=cast(Mapping[str, JsonValue], _mapping(structured, "structured output"))
        if structured is not None
        else None,
        raw_response=_string(item, "raw_response", allow_empty=True),
        tool_calls=tuple(tools),
        usage=tuple(usages),
        cost=cost_from(item.get("cost")),
        error=_optional_string(item, "error"),
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _string(value: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item):
        raise TypeError(f"{key} must be a {'string' if allow_empty else 'non-empty string'}")
    return item


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        raise TypeError(f"{key} must be a string or null")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise TypeError(f"{key} must be an integer")
    return item


def _optional_int(value: Mapping[str, Any], key: str) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, int) or isinstance(item, bool):
        raise TypeError(f"{key} must be an integer or null")
    return item


def _number(value: Mapping[str, Any], key: str) -> float:
    return _finite_number(value.get(key), key)


def _optional_number(value: Mapping[str, Any], key: str) -> float | None:
    item = value.get(key)
    return None if item is None else _finite_number(item, key)


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    converted = float(value)
    if converted != converted or converted in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return converted


def _strings(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(element, str) for element in item):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(item)