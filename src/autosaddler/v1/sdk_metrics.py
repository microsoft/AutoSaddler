"""Resume-safe aggregation of persisted SDK session metrics."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
)
COST_FIELDS = (
    "reported_cost_usd",
    "metered_cost_usd",
    "estimated_cost_usd",
    "total_cost_usd",
    "billing_net_cost_usd",
)


def _number(value: Any) -> int | float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return None


def _usage_total(usage: list[Any], field: str, *aliases: str) -> int:
    total = 0
    for item in usage:
        if not isinstance(item, dict):
            continue
        for key in (field, *aliases):
            value = _number(item.get(key))
            if value is not None:
                total += int(value)
                break
    return total


def _normalized_session(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or []
    attempts = [
        attempt
        for attempt in payload.get("attempts") or []
        if isinstance(attempt, dict)
    ]
    attempts_complete = bool(attempts) and all(
        attempt.get("accounting_complete", False) for attempt in attempts
    )
    token_aliases = {
        "input_tokens": ("promptTokens", "prompt_tokens"),
        "output_tokens": ("completionTokens", "completion_tokens"),
        "cache_read_input_tokens": ("cache_read_tokens",),
        "cache_creation_input_tokens": ("cache_write_tokens",),
        "reasoning_tokens": (),
    }
    attempt_llm_calls = sum(
        int(attempt.get("llm_call_count", 0) or 0) for attempt in attempts
    )
    record: dict[str, Any] = {
        "path": str(path),
        "session_type": payload.get("session_type") or "unknown",
        "session_id": payload.get("session_id"),
        "model": payload.get("model") or "unknown",
        "wall_clock_s": _number(payload.get("wall_clock_s"))
        or sum(float(attempt.get("wall_clock_s", 0.0) or 0.0) for attempt in attempts),
        "llm_call_count": int(
            _number(payload.get("llm_call_count"))
            or _number(payload.get("usage_event_count"))
            or attempt_llm_calls
            or len(usage)
        ),
        "cost_source": payload.get("cost_source"),
        "cost_is_estimate": bool(payload.get("cost_is_estimate", False)),
        "copilot_nano_aiu": _number(payload.get("copilot_nano_aiu"))
        or (
            sum(
                float(attempt.get("copilot_nano_aiu", 0.0) or 0.0)
                for attempt in attempts
            )
            if attempts
            else None
        ),
        "attempt_accounting_complete": (
            payload.get("attempt_accounting_complete")
            if "attempt_accounting_complete" in payload
            else attempts_complete if attempts else True
        ),
    }
    for field, aliases in token_aliases.items():
        value = _number(payload.get(field))
        record[field] = (
            int(value)
            if value is not None
            else sum(int(attempt.get(field, 0) or 0) for attempt in attempts)
            if attempts
            else _usage_total(usage, field, *aliases)
        )
    for field in COST_FIELDS:
        value = _number(payload.get(field))
        if value is None and attempts and attempts_complete:
            attempt_values = [
                float(attempt[field])
                for attempt in attempts
                if _number(attempt.get(field)) is not None
            ]
            value = sum(attempt_values) if attempt_values else None
        record[field] = value
    record["model_usage"] = payload.get("model_usage")
    record["attempts"] = attempts
    return record


def _empty_totals() -> dict[str, Any]:
    return {
        **{field: 0 for field in TOKEN_FIELDS},
        "wall_clock_s": 0.0,
        "llm_call_count": 0,
        "copilot_nano_aiu": 0.0,
        **{field: None for field in COST_FIELDS},
    }


def _sum_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _empty_totals()
    for record in records:
        for field in TOKEN_FIELDS:
            totals[field] += record[field]
        totals["wall_clock_s"] += record["wall_clock_s"]
        totals["llm_call_count"] += record["llm_call_count"]
        nano_aiu = _number(record.get("copilot_nano_aiu"))
        if nano_aiu is not None:
            totals["copilot_nano_aiu"] += float(nano_aiu)
    for field in COST_FIELDS:
        values = [
            float(value)
            for record in records
            if (value := _number(record.get(field))) is not None
        ]
        totals[field] = sum(values) if values else None
    return totals


def _model_entry_values(entry: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "cache_read_input_tokens": (
            "cache_read_input_tokens",
            "cacheReadInputTokens",
        ),
        "cache_creation_input_tokens": (
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
        ),
        "reasoning_tokens": ("reasoning_tokens", "reasoningTokens"),
        "llm_call_count": ("llm_calls",),
        "copilot_nano_aiu": ("copilot_nano_aiu",),
        "reported_cost_usd": ("reported_cost_usd",),
        "metered_cost_usd": ("metered_cost_usd",),
        "estimated_cost_usd": ("estimated_cost_usd",),
        "total_cost_usd": (
            "total_cost_usd",
            "metered_cost_usd",
            "estimated_cost_usd",
            "costUSD",
        ),
    }
    values: dict[str, Any] = {}
    for output_name, source_names in aliases.items():
        values[output_name] = next(
            (
                value
                for source_name in source_names
                if (value := _number(entry.get(source_name))) is not None
            ),
            0,
        )
    return values


def aggregate_session_artifacts(session_root: str | Path) -> dict[str, Any]:
    """Aggregate every canonical SDK session artifact under one run root."""
    root = Path(session_root)
    records: list[dict[str, Any]] = []
    for path in sorted((root / "cycles").glob("*/iter*_c*_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("session_type"):
            continue
        records.append(_normalized_session(path, payload))

    by_session_type: dict[str, Any] = {}
    for session_type in sorted({record["session_type"] for record in records}):
        selected = [
            record for record in records if record["session_type"] == session_type
        ]
        by_session_type[session_type] = {
            "num_sessions": len(selected),
            **_sum_records(selected),
        }

    by_model: dict[str, dict[str, Any]] = {}
    for record in records:
        attempt_model_usage = [
            attempt.get("model_usage")
            for attempt in record.get("attempts") or []
            if isinstance(attempt, dict)
            and isinstance(attempt.get("model_usage"), dict)
            and attempt.get("model_usage")
        ]
        if attempt_model_usage:
            entries = [
                item
                for model_usage in attempt_model_usage
                for item in model_usage.items()
            ]
        else:
            model_usage = record.get("model_usage")
            entries = (
                list(model_usage.items())
                if isinstance(model_usage, dict) and model_usage
                else [(record["model"], record)]
            )
        for model, raw_entry in entries:
            if not isinstance(raw_entry, dict):
                continue
            values = _model_entry_values(raw_entry)
            target = by_model.setdefault(
                str(model),
                {
                    "num_sessions": 0,
                    **{field: 0 for field in TOKEN_FIELDS},
                    "llm_call_count": 0,
                    "copilot_nano_aiu": 0.0,
                    "reported_cost_usd": 0.0,
                    "metered_cost_usd": 0.0,
                    "estimated_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                },
            )
            target["num_sessions"] += 1
            for field, value in values.items():
                target[field] += value

    missing_cost_sessions = [
        record["path"]
        for record in records
        if record["total_cost_usd"] is None
        or not record["attempt_accounting_complete"]
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "num_sessions": len(records),
        "accounting_complete": bool(records) and not missing_cost_sessions,
        "missing_cost_sessions": missing_cost_sessions,
        "aggregate": _sum_records(records),
        "by_session_type": by_session_type,
        "by_model": by_model,
        "sessions": records,
    }


def session_root_from_artifact_dir(artifact_dir: str | Path) -> Path | None:
    """Resolve ``<session_root>`` only from a canonical ``cycles/<cycle>`` path."""
    path = Path(artifact_dir).resolve()
    for parent in (path, *path.parents):
        if parent.name == "cycles":
            return parent.parent
    return None


def write_run_sdk_metrics(session_root: str | Path) -> Path:
    """Atomically rebuild ``sdk_metrics.json`` from persisted session files."""
    root = Path(session_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "sdk_metrics.json"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(aggregate_session_artifacts(root), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path