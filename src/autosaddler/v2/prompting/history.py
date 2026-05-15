from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from autosaddler.v2.core.domain import JsonValue, canonical_json, sha256_digest, to_json_value
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.core.run_state import RunState
from autosaddler.v2.storage.local import LocalRunStore

HISTORY_ROOT = ".autosaddler/history"


@dataclass(frozen=True, slots=True)
class HistoryBundle:
    workspace_files: Mapping[str, str]
    compatibility_history: Mapping[str, JsonValue]


def build_history_bundle(
    store: LocalRunStore,
    context: Mapping[str, JsonValue],
) -> HistoryBundle:
    events = store.events()
    state = RunState.replay(events)
    all_iterations = _iteration_records(events)
    iteration_ids = sorted(all_iterations)
    iterations = {iteration: all_iterations[iteration] for iteration in iteration_ids}
    all_lessons = _lessons(events)
    lessons = all_lessons
    all_current_case_ids = _current_case_ids(context)
    current_case_ids = set(all_current_case_ids)
    all_candidate_ids = list(state.candidates)
    candidate_ids = all_candidate_ids
    development_scores = {
        evaluation.candidate_id: evaluation.aggregate_score
        for evaluation in state.evaluations.values()
        if evaluation.split == "development" and evaluation.aggregate_score is not None
    }
    accepted_ids = set(state.accepted_candidate_ids)
    evolution_status = {
        candidate_id: "accepted" if event.payload.get("accepted") is True else "declined"
        for event in events
        if event.event_type == "EvolutionRecorded"
        and isinstance(candidate_id := event.payload.get("candidate_id"), str)
    }

    files: dict[str, str] = {}
    candidate_summaries: list[dict[str, JsonValue]] = []
    edge_paths: list[str] = []
    omitted_diffs = 0
    included_diff_bytes = 0
    for candidate_id in sorted(candidate_ids):
        candidate = state.candidates[candidate_id]
        status = "accepted" if candidate_id in accepted_ids else evolution_status.get(candidate_id, "proposed")
        if state.selected_candidate_id == candidate_id:
            status = "selected"
        candidate_path = f"{HISTORY_ROOT}/candidates/{_digest_name(candidate_id)}.json"
        changed_units = list(candidate.change.changed_units) if candidate.change is not None else []
        detail: dict[str, JsonValue] = {
            "schema_version": "autosaddler-history-candidate/v1",
            "candidate_id": candidate_id,
            "parent_ids": list(candidate.parent_ids),
            "space": candidate.space,
            "status": status,
            "accepted": candidate_id in accepted_ids,
            "development_aggregate": development_scores.get(candidate_id),
            "changed_units": changed_units,
            "change": to_json_value(candidate.change),
        }
        if candidate.change is not None and candidate.change.diff is not None:
            diff = candidate.change.diff
            source = store.run_dir / diff.uri
            payload = source.read_bytes()
            if len(payload) != diff.bytes or sha256_digest(payload) != diff.sha256:
                raise ValueError(f"History diff artifact drift: {diff.uri}")
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                detail["diff_omitted_reason"] = "not_utf8"
                omitted_diffs += 1
            else:
                diff_path = f"{HISTORY_ROOT}/diffs/{_digest_name(diff.sha256)}.patch"
                files[diff_path] = text
                detail["diff_path"] = diff_path
                detail["diff_sha256"] = diff.sha256
                included_diff_bytes += len(payload)
        files[candidate_path] = canonical_json(detail) + "\n"
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "parent_ids": list(candidate.parent_ids),
                "status": status,
                "changed_units": changed_units,
                "development_aggregate": development_scores.get(candidate_id),
                "detail_path": candidate_path,
            }
        )
        for parent_id in candidate.parent_ids:
            edge_path = f"{HISTORY_ROOT}/edges/{_digest_name(parent_id)}--{_digest_name(candidate_id)}.json"
            files[edge_path] = (
                canonical_json(
                    {
                        "schema_version": "autosaddler-history-edge/v1",
                        "parent_candidate_id": parent_id,
                        "child_candidate_id": candidate_id,
                        "child_detail_path": candidate_path,
                        "changed_units": changed_units,
                        "change": to_json_value(candidate.change),
                        "diff_path": detail.get("diff_path"),
                    }
                )
                + "\n"
            )
            edge_paths.append(edge_path)

    iteration_paths: list[str] = []
    relevant_iterations: list[dict[str, JsonValue]] = []
    for iteration in sorted(iterations):
        record = iterations[iteration]
        path = f"{HISTORY_ROOT}/iterations/{iteration:04d}.json"
        files[path] = canonical_json({"schema_version": "autosaddler-history-iteration/v1", **record}) + "\n"
        iteration_paths.append(path)
        if current_case_ids.intersection(_record_case_ids(record)):
            relevant_iterations.append(record)

    relevant_case_paths: dict[str, str] = {}
    for case_id in sorted(current_case_ids):
        case_path = f"{HISTORY_ROOT}/relevant_cases/{_digest_name(sha256_digest(case_id))}.json"
        matching = [record for record in iterations.values() if case_id in _record_case_ids(record)]
        files[case_path] = (
            canonical_json(
                {
                    "schema_version": "autosaddler-history-relevant-case/v1",
                    "case_id": case_id,
                    "prior_iterations": matching,
                }
            )
            + "\n"
        )
        relevant_case_paths[case_id] = case_path

    lessons_path = f"{HISTORY_ROOT}/lessons.json"
    files[lessons_path] = (
        canonical_json({"schema_version": "autosaddler-history-lessons/v1", "records": lessons}) + "\n"
    )
    current_batch_path = f"{HISTORY_ROOT}/current_batch.json"
    files[current_batch_path] = (
        canonical_json(
            {
                "schema_version": "autosaddler-history-current-batch/v1",
                "train_case_ids": sorted(current_case_ids),
                "relevant_case_paths": relevant_case_paths,
                "relevant_iteration_ids": [record["iteration"] for record in relevant_iterations],
            }
        )
        + "\n"
    )
    summary_path = f"{HISTORY_ROOT}/summary.json"
    files[summary_path] = (
        canonical_json(
            {
                "schema_version": "autosaddler-history-summary/v1",
                "accepted_candidate_ids": list(state.accepted_candidate_ids),
                "selected_candidate_id": state.selected_candidate_id,
                "candidates": candidate_summaries,
                "edge_paths": edge_paths,
                "iteration_paths": iteration_paths,
                "lessons_path": lessons_path,
                "current_batch_path": current_batch_path,
            }
        )
        + "\n"
    )

    event_snapshot = "".join(canonical_json(event) + "\n" for event in events)
    manifest_path = f"{HISTORY_ROOT}/manifest.json"
    manifest_files = {
        path: {"sha256": sha256_digest(text), "bytes": len(text.encode("utf-8"))}
        for path, text in sorted(files.items())
    }
    files[manifest_path] = (
        canonical_json(
            {
                "schema_version": "autosaddler-history-manifest/v1",
                "event_snapshot_sha256": sha256_digest(event_snapshot),
                "through_sequence": events[-1].sequence if events else 0,
                "entry_points": {
                    "summary": summary_path,
                    "lessons": lessons_path,
                    "current_batch": current_batch_path,
                },
                "limits": {
                    "max_candidates": None,
                    "max_iterations": None,
                    "max_lesson_records": None,
                    "max_current_cases": None,
                    "max_diff_bytes": None,
                    "max_total_diff_bytes": None,
                },
                "included": {
                    "candidate_ids": candidate_ids,
                    "iteration_ids": iteration_ids,
                    "lesson_records": len(lessons),
                    "current_case_ids": sorted(current_case_ids),
                    "diff_bytes": included_diff_bytes,
                },
                "totals": {
                    "candidates": len(all_candidate_ids),
                    "iterations": len(all_iterations),
                    "lesson_records": len(all_lessons),
                    "current_cases": len(all_current_case_ids),
                },
                "omission_counts": {
                    "candidates": len(all_candidate_ids) - len(candidate_ids),
                    "iterations": len(all_iterations) - len(iterations),
                    "lesson_records": len(all_lessons) - len(lessons),
                    "current_cases": len(all_current_case_ids) - len(current_case_ids),
                    "diffs": omitted_diffs,
                },
                "files": manifest_files,
            }
        )
        + "\n"
    )

    summaries_by_id = {item["candidate_id"]: item for item in candidate_summaries}
    compatibility_candidates = []
    for candidate_id in state.accepted_candidate_ids:
        if candidate_id not in summaries_by_id:
            continue
        item = summaries_by_id[candidate_id]
        compatibility_candidates.append(
            {
                "candidate_id": item["candidate_id"],
                "parent_ids": item["parent_ids"],
                "changed_components": item["changed_units"],
                "development_aggregate": item["development_aggregate"],
            }
        )
    compatibility: dict[str, JsonValue] = {
        "schema_version": "autosaddler-optimization-history/v1",
        "candidates": compatibility_candidates,
        "lessons": lessons,
        "prior_iterations": [iterations[index] for index in sorted(iterations)],
        "relevant_prior_iterations": relevant_iterations,
        "limits": {
            "max_candidates": None,
            "max_iterations": None,
            "max_lesson_records": None,
        },
        "omission_counts": {
            "candidates": len(all_candidate_ids) - len(candidate_ids),
            "iterations": len(all_iterations) - len(iterations),
            "lesson_records": len(all_lessons) - len(lessons),
        },
    }
    return HistoryBundle(workspace_files=files, compatibility_history=compatibility)


def _iteration_records(events: tuple[RunEvent, ...]) -> dict[int, dict[str, JsonValue]]:
    records: dict[int, dict[str, JsonValue]] = {}
    for event in events:
        if event.event_type not in {
            "BatchSampled",
            "MutationRejected",
            "AcceptanceDecided",
            "DeferredWorkScheduled",
            "EvolutionRecorded",
            "IterationCompleted",
        }:
            continue
        payload = event.payload
        iteration = (
            payload.get("owning_iteration") if event.event_type == "DeferredWorkScheduled" else payload.get("iteration")
        )
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise TypeError(f"{event.event_type} event must contain an integer iteration")
        record = records.setdefault(iteration, {"iteration": iteration})
        if event.event_type == "BatchSampled":
            record["train_case_ids"] = to_json_value(payload.get("case_ids", []))
        elif event.event_type == "MutationRejected":
            record["mutation_rejected"] = True
            record["verification_failure"] = to_json_value(payload.get("verification_failure"))
        elif event.event_type == "AcceptanceDecided":
            record["verdict"] = to_json_value(payload.get("verdict"))
        elif event.event_type == "DeferredWorkScheduled":
            for key in (
                "candidate_id",
                "parent_candidate_id",
                "working_parent_candidate_id",
                "selection_parent_ids",
                "component_sources",
                "selection_rationale",
                "train_case_ids",
                "train_before_aggregate",
                "train_after_aggregate",
                "train_before_case_scores",
                "train_after_case_scores",
                "accepted_by_minibatch_gate",
                "parent_development_aggregate",
                "candidate_development_aggregate",
                "changed_components",
                "updates",
                "diagnosis",
                "patch_phase",
                "acceptance_reason",
            ):
                record[key] = to_json_value(payload.get(key))
        elif event.event_type == "EvolutionRecorded":
            record["candidate_id"] = to_json_value(payload.get("candidate_id"))
            record["accepted"] = payload.get("accepted") is True
            record["development_aggregate"] = to_json_value(payload.get("development_aggregate"))
        elif event.event_type == "IterationCompleted":
            record["completion_schema_version"] = to_json_value(payload.get("schema_version"))
            record["outcome"] = to_json_value(payload.get("outcome"))
            record["resulting_candidate_id"] = to_json_value(payload.get("resulting_candidate_id"))
            record["evaluated_candidate_ids"] = to_json_value(payload.get("evaluated_candidate_ids", []))
    return records


def _lessons(events: tuple[RunEvent, ...]) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    for event in events:
        if event.event_type != "ExtensionStateChanged" or event.payload.get("namespace") != "autosaddler.lessons":
            continue
        records.append(
            {
                "candidate_id": to_json_value(event.payload.get("candidate_id")),
                "owning_iteration": to_json_value(event.payload.get("owning_iteration")),
                "lessons": to_json_value(event.payload.get("lessons", [])),
            }
        )
    return records


def _current_case_ids(context: Mapping[str, JsonValue]) -> tuple[str, ...]:
    value = context.get("train_case_ids", [])
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(case_id for case_id in value if isinstance(case_id, str)))


def _record_case_ids(record: Mapping[str, JsonValue]) -> set[str]:
    value = record.get("train_case_ids", [])
    if not isinstance(value, list):
        return set()
    return {case_id for case_id in value if isinstance(case_id, str)}


def _digest_name(value: str) -> str:
    name = value.removeprefix("sha256:")
    if not name or PurePosixPath(name).name != name:
        raise ValueError(f"Unsafe history digest: {value}")
    return name
