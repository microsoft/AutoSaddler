"""Session contracts and read-only workspace views for the failure-pattern curriculum."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from autosaddler.v2.core.curriculum import CurriculumState, FailurePattern, activity_ema
from autosaddler.v2.core.domain import JsonValue, canonical_json, to_json_value
from autosaddler.v2.prompting.history import HISTORY_ROOT, iteration_records
from autosaddler.v2.storage.local import LocalRunStore

CURRICULUM_ROOT = ".autosaddler/curriculum"

_PULL_KINDS = {
    "accepted": "patched",
    "declined": "patched",
    "no_training_failures": "all_pass_skip",
    "mutation_rejected": "failed_attempt",
}


def pattern_extraction_schema(schema_version: str, failing_case_ids: Sequence[str]) -> Mapping[str, JsonValue]:
    case_id: dict[str, JsonValue] = {"type": "string", "enum": list(dict.fromkeys(failing_case_ids))}
    source: dict[str, JsonValue] = {"type": "string", "enum": ["pre_patch", "post_patch"]}
    text: dict[str, JsonValue] = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "required": ["schema_version", "symptoms", "new_patterns", "tags"],
        "properties": {
            "schema_version": {"const": schema_version},
            "symptoms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["case_id", "source", "root_cause", "symptom", "rationale"],
                    "properties": {
                        "case_id": case_id,
                        "source": source,
                        "root_cause": text,
                        "symptom": text,
                        "rationale": text,
                    },
                    "additionalProperties": False,
                },
            },
            "new_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["key", "label"],
                    "properties": {"key": text, "label": text},
                    "additionalProperties": False,
                },
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["case_id", "source", "pattern_refs", "root_cause"],
                    "properties": {
                        "case_id": case_id,
                        "source": source,
                        "pattern_refs": {"type": "array", "items": text, "minItems": 1, "uniqueItems": True},
                        "root_cause": text,
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def arm_decision_schema(schema_version: str) -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "required": ["schema_version", "action", "rationale"],
        "properties": {
            "schema_version": {"const": schema_version},
            "action": {"type": "string", "enum": ["pull", "draw"]},
            "rationale": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


def arm_scoring_schema(schema_version: str, arm_ids: Sequence[str]) -> Mapping[str, JsonValue]:
    unit: dict[str, JsonValue] = {"type": "number", "minimum": 0, "maximum": 1}
    return {
        "type": "object",
        "required": ["schema_version", "scores"],
        "properties": {
            "schema_version": {"const": schema_version},
            "scores": {
                "type": "array",
                "minItems": len(arm_ids),
                "maxItems": len(arm_ids),
                "items": {
                    "type": "object",
                    "required": ["pattern_id", "severity", "fixability", "breadth", "side_effect", "rationale"],
                    "properties": {
                        "pattern_id": {"type": "string", "enum": list(arm_ids)},
                        "severity": unit,
                        "fixability": unit,
                        "breadth": unit,
                        "side_effect": unit,
                        "rationale": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def build_curriculum_bundle(store: LocalRunStore, context: Mapping[str, JsonValue]) -> dict[str, str]:
    """Render the replayed failure-pattern registry as read-only workspace files."""
    curriculum = context.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise TypeError("Curriculum sessions require a curriculum context object")
    eta = curriculum.get("ema_eta")
    if isinstance(eta, bool) or not isinstance(eta, (int, float)):
        raise TypeError("Curriculum context requires a numeric ema_eta")
    events = store.events()
    state = CurriculumState.replay(events)
    iterations = iteration_records(events)
    files: dict[str, str] = {}
    table: list[dict[str, JsonValue]] = []
    for pattern_id, pattern in state.patterns.items():
        name = _safe_name(pattern_id)
        detail_path = f"{CURRICULUM_ROOT}/patterns/{name}.json"
        pull_path = f"{CURRICULUM_ROOT}/pull_history/{name}.json"
        activity = activity_ema(pattern.observations, float(eta))
        latest = max(pattern.scores, key=lambda item: item.iteration, default=None)
        table.append(
            {
                "pattern_id": pattern_id,
                "label": pattern.label,
                "activity": activity,
                "num_observations": len(pattern.observations),
                "num_cases": len(pattern.case_ids),
                "last_observed_iteration": pattern.last_observed_iteration,
                "created_iteration": pattern.created_iteration,
                "case_ids": list(pattern.case_ids),
                "latest_score": to_json_value(latest),
                "latest_score_value": latest.value if latest is not None else None,
                "detail_path": detail_path,
                "pull_history_path": pull_path,
            }
        )
        files[detail_path] = _json(
            {
                "schema_version": "autosaddler-curriculum-pattern/v1",
                "pattern_id": pattern_id,
                "label": pattern.label,
                "created_iteration": pattern.created_iteration,
                "activity": activity,
                "tags": to_json_value(pattern.tags),
                "observations": to_json_value(pattern.observations),
                "scores": to_json_value(pattern.scores),
            }
        )
        files[pull_path] = _json(
            {
                "schema_version": "autosaddler-curriculum-pull-history/v1",
                "pattern_id": pattern_id,
                "pulls": _pull_records(state, pattern, iterations),
            }
        )
    table.sort(key=lambda item: (-_float(item["activity"]), str(item["pattern_id"])))
    patterns_path = f"{CURRICULUM_ROOT}/patterns.json"
    files[patterns_path] = _json({"schema_version": "autosaddler-curriculum-patterns/v1", "patterns": table})
    decisions_path = f"{CURRICULUM_ROOT}/decisions.json"
    files[decisions_path] = _json(
        {
            "schema_version": "autosaddler-curriculum-decisions/v1",
            "decisions": [dict(state.decisions[key]) for key in sorted(state.decisions)],
        }
    )
    files[f"{CURRICULUM_ROOT}/manifest.json"] = _json(
        {
            "schema_version": "autosaddler-curriculum-manifest/v1",
            "iteration": context.get("iteration"),
            "settings": to_json_value(curriculum),
            "num_patterns": len(state.patterns),
            "num_executed_cases": len(state.executed_case_ids),
            "num_probe_points": len(state.probe_points),
            "entry_points": {"patterns": patterns_path, "decisions": decisions_path},
            "files": sorted(files),
        }
    )
    return files


def _pull_records(
    state: CurriculumState,
    pattern: FailurePattern,
    iterations: Mapping[int, Mapping[str, JsonValue]],
) -> list[JsonValue]:
    records: list[JsonValue] = []
    for pull in state.pulls:
        if pull.pattern_id != pattern.pattern_id:
            continue
        record = iterations.get(pull.iteration, {})
        outcome = record.get("outcome")
        records.append(
            {
                "iteration": pull.iteration,
                "case_ids": list(pull.case_ids),
                "kind": _PULL_KINDS.get(str(outcome), "in_progress"),
                "outcome": outcome,
                "diagnosis": record.get("diagnosis"),
                "train_before_case_scores": record.get("train_before_case_scores"),
                "train_after_case_scores": record.get("train_after_case_scores"),
                "candidate_development_aggregate": record.get("candidate_development_aggregate"),
                "parent_development_aggregate": record.get("parent_development_aggregate"),
                "history_iteration_path": f"{HISTORY_ROOT}/iterations/{pull.iteration:04d}.json",
            }
        )
    return records


def _json(value: Mapping[str, JsonValue]) -> str:
    return canonical_json(value) + "\n"


def _float(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _safe_name(pattern_id: str) -> str:
    if not pattern_id or PurePosixPath(pattern_id).name != pattern_id:
        raise ValueError(f"Unsafe curriculum pattern ID: {pattern_id}")
    return pattern_id
