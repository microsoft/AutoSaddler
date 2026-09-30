"""Session contracts and read-only workspace views for the failure-pattern curriculum."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

from autosaddler.v2.core.curriculum import CurriculumState, FailurePattern, activity_ema, case_status
from autosaddler.v2.core.domain import JsonValue, canonical_json, sha256_digest, to_json_value
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.prompting.assets import extension_prompt_source_entities
from autosaddler.v2.prompting.history import HISTORY_ROOT
from autosaddler.v2.storage.local import LocalRunStore

CURRICULUM_ROOT = ".autosaddler/curriculum"
CURRICULUM_METHODOLOGY_ROOT = Path(__file__).parent / "curriculum_methodology"

_PULL_KINDS = {
    "accepted": "patched",
    "declined": "patched",
    "no_training_failures": "all_pass_skip",
    "mutation_rejected": "failed_attempt",
}


def curriculum_prompt_source_entities(*, plugin_root: Path, plugin_name: str) -> dict[str, str | Mapping[str, JsonValue]]:
    """Record the curriculum prompt sources; a scenario keeps its own under ``<plugin_root>/curriculum``."""
    return extension_prompt_source_entities(
        extension="curriculum",
        shared_root=CURRICULUM_METHODOLOGY_ROOT,
        plugin_root=plugin_root,
        plugin_name=plugin_name,
    )


def pattern_extraction_schema(schema_version: str, failing_case_ids: Sequence[str]) -> Mapping[str, JsonValue]:
    case_id: dict[str, JsonValue] = {"type": "string", "enum": list(dict.fromkeys(failing_case_ids))}
    source: dict[str, JsonValue] = {"type": "string", "enum": ["pre_patch", "post_patch"]}
    text: dict[str, JsonValue] = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "required": ["schema_version", "new_patterns", "tags"],
        "properties": {
            "schema_version": {"const": schema_version},
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
    curriculum = context.get("task_selection")
    if not isinstance(curriculum, Mapping):
        raise TypeError("Curriculum sessions require a task_selection context object")
    eta = curriculum.get("ema_eta")
    if isinstance(eta, bool) or not isinstance(eta, (int, float)):
        raise TypeError("Curriculum context requires a numeric ema_eta")
    events = store.events()
    state = CurriculumState.replay(events)
    iterations = _iteration_outcomes(events, state)
    files: dict[str, str] = {}
    case_paths = {
        case_id: f"{CURRICULUM_ROOT}/cases/{sha256_digest(case_id).removeprefix('sha256:')}.json"
        for case_id in sorted(state.executed_case_ids)
    }
    for case_id, case_path in case_paths.items():
        files[case_path] = _json(_case_history(case_id, state, iterations))
    cases_path = f"{CURRICULUM_ROOT}/cases.json"
    files[cases_path] = _json({"schema_version": "autosaddler-curriculum-cases/v1", "case_history_paths": case_paths})
    table: list[dict[str, JsonValue]] = []
    for pattern_id, pattern in state.patterns.items():
        name = _safe_name(pattern_id)
        detail_path = f"{CURRICULUM_ROOT}/patterns/{name}.json"
        pull_path = f"{CURRICULUM_ROOT}/pull_history/{name}.json"
        pattern_case_paths = {case_id: case_paths[case_id] for case_id in pattern.case_ids if case_id in case_paths}
        activity = activity_ema(pattern.observations, float(eta))
        latest = max(pattern.scores, key=lambda item: item.iteration, default=None)
        table.append(
            {
                "pattern_id": pattern_id,
                "label": pattern.label,
                "activity": activity,
                "observations": to_json_value(pattern.observations),
                "num_observations": len(pattern.observations),
                "num_cases": len(pattern.case_ids),
                "last_observed_iteration": pattern.last_observed_iteration,
                "created_iteration": pattern.created_iteration,
                "case_ids": list(pattern.case_ids),
                "latest_score": to_json_value(latest),
                "latest_score_value": latest.value if latest is not None else None,
                "detail_path": detail_path,
                "pull_history_path": pull_path,
                "case_history_paths": cast(JsonValue, pattern_case_paths),
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
                "case_history_paths": cast(JsonValue, pattern_case_paths),
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
            "entry_points": {"patterns": patterns_path, "decisions": decisions_path, "cases": cases_path},
            "files": sorted(files),
        }
    )
    return files


def _pull_records(
    state: CurriculumState,
    pattern: FailurePattern,
    iterations: Mapping[int, Mapping[str, JsonValue]],
) -> list[JsonValue]:
    """Every pull of one arm: patch intent, development impact, per-case results, and lessons."""
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
                "working_parent_candidate_id": record.get("working_parent_candidate_id"),
                "candidate_id": record.get("candidate_id"),
                "diagnosis": record.get("diagnosis"),
                "patch_intent": record.get("patch_intent"),
                "parent_development_aggregate": record.get("parent_development_aggregate"),
                "candidate_development_aggregate": record.get("candidate_development_aggregate"),
                "case_outcomes": record.get("case_outcomes", []),
                "lessons": record.get("lessons", []),
                "history_iteration_path": f"{HISTORY_ROOT}/iterations/{pull.iteration:04d}.json",
            }
        )
    return records


def _case_history(
    case_id: str,
    state: CurriculumState,
    iterations: Mapping[int, Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    """Per-case history across iterations, like the original scenario registry."""
    evaluations: list[JsonValue] = []
    for iteration in sorted(iterations):
        record = iterations[iteration]
        case_outcome = next(
            (
                item
                for item in cast(list[Mapping[str, JsonValue]], record.get("case_outcomes", []))
                if item.get("case_id") == case_id
            ),
            None,
        )
        if case_outcome is None:
            continue
        evaluations.append(
            {
                "iteration": iteration,
                "sampling_action": record.get("sampling_action"),
                "pulled_arm_id": record.get("pulled_arm_id"),
                "outcome": record.get("outcome"),
                "working_parent_candidate_id": record.get("working_parent_candidate_id"),
                "candidate_id": record.get("candidate_id"),
                "diagnosis": record.get("diagnosis"),
                "patch_intent": record.get("patch_intent"),
                **dict(case_outcome),
                "lessons": [
                    lesson
                    for lesson in cast(list[Mapping[str, JsonValue]], record.get("lessons", []))
                    if case_id in cast(list[JsonValue], lesson.get("evidence_case_ids") or [])
                ],
            }
        )
    return {
        "schema_version": "autosaddler-curriculum-case/v1",
        "case_id": case_id,
        "pattern_ids": [pattern_id for pattern_id, pattern in state.patterns.items() if case_id in pattern.case_ids],
        "evaluations": evaluations,
    }


def _iteration_outcomes(events: Sequence[RunEvent], state: CurriculumState) -> dict[int, dict[str, JsonValue]]:
    """Per-iteration batch, outcome, patch intent, per-case results, lessons, and development aggregates."""
    records: dict[int, dict[str, JsonValue]] = {}
    for event in events:
        payload = event.payload
        if event.event_type == "BatchSampled":
            key = payload.get("iteration")
            provenance = payload.get("provenance")
            provenance = provenance if isinstance(provenance, Mapping) else {}
            fields = {
                "case_ids": payload.get("case_ids"),
                "sampling_action": provenance.get("action"),
                "pulled_arm_id": provenance.get("chosen_arm"),
            }
        elif event.event_type == "IterationCompleted":
            key = payload.get("iteration")
            fields = {"outcome": payload.get("outcome")}
        elif event.event_type == "ExtensionStateChanged" and payload.get("namespace") == "autosaddler.lessons":
            key = payload.get("owning_iteration")
            fields = {"lessons": payload.get("lessons", [])}
        elif event.event_type == "DeferredWorkScheduled" and payload.get("session_kind") == "reflect":
            key = payload.get("owning_iteration")
            feedback = payload.get("task_selection_feedback")
            fields = {
                name: payload.get(name)
                for name in (
                    "working_parent_candidate_id",
                    "candidate_id",
                    "diagnosis",
                    "train_before_case_scores",
                    "train_after_case_scores",
                    "candidate_development_aggregate",
                    "parent_development_aggregate",
                )
            }
            fields["patch_intent"] = feedback.get("patch_intent") if isinstance(feedback, Mapping) else None
        else:
            continue
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(f"{event.event_type} event must contain an integer iteration")
        records.setdefault(key, {}).update(fields)
    for iteration, record in records.items():
        record["case_outcomes"] = _case_outcomes(iteration, record, state)
    return records


def _case_outcomes(iteration: int, record: Mapping[str, JsonValue], state: CurriculumState) -> list[JsonValue]:
    """Per-case status, scores, and the failure-pattern tags (with root causes) of one iteration."""
    before = record.get("train_before_case_scores")
    after = record.get("train_after_case_scores")
    outcomes: list[JsonValue] = []
    for case_id in cast(list[str], record.get("case_ids") or []):
        before_score = before.get(case_id) if isinstance(before, Mapping) else None
        after_score = after.get(case_id) if isinstance(after, Mapping) else None
        if isinstance(before_score, (int, float)) and isinstance(after_score, (int, float)):
            status: str | None = case_status(float(before_score), float(after_score))
        elif record.get("outcome") == "no_training_failures":
            status = "passing_before_patch"
        else:
            status = None
        outcomes.append(
            {
                "case_id": case_id,
                "status": status,
                "train_before_score": before_score,
                "train_after_score": after_score,
                "pattern_tags": [
                    {"pattern_id": pattern_id, "source": tag.source, "root_cause": tag.root_cause}
                    for pattern_id, pattern in state.patterns.items()
                    for tag in pattern.tags
                    if tag.iteration == iteration and tag.case_id == case_id
                ],
            }
        )
    return outcomes


def _json(value: Mapping[str, JsonValue]) -> str:
    return canonical_json(value) + "\n"


def _float(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _safe_name(pattern_id: str) -> str:
    if not pattern_id or PurePosixPath(pattern_id).name != pattern_id:
        raise ValueError(f"Unsafe curriculum pattern ID: {pattern_id}")
    return pattern_id
