"""Lesson manager: automated AccumulatedLessons and ScenarioRegistry updates.

All logic in this module runs in the outer loop (Python), not the Agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autosaddler.v1.proposer.autosaddler.models import (
    AttemptedFix,
    LessonEntry,
    ScenarioImpact,
    ScenarioSnapshot,
)

if TYPE_CHECKING:
    from autosaddler.v1.proposer.autosaddler.dag import EvolutionDAG
    from autosaddler.v1.proposer.autosaddler.models import (
        EvolutionNode,
        PatchVerdict,
        ScenarioEntry,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AccumulatedLessons
# ---------------------------------------------------------------------------


def update_lessons(dag: EvolutionDAG, node: EvolutionNode, verdict: PatchVerdict) -> None:
    """Extract lessons from reflections and add to AccumulatedLessons.

    1. ``fixed`` reflections → ``good_patterns``
    2. ``regressed`` reflections → ``bad_patterns``
    3. ``still_failing`` reflections → ``bad_patterns`` (ineffective attempts)
    """
    for refl in verdict.reflections:
        if refl.status_change == "fixed":
            dag.accumulated_lessons.good_patterns.append(
                LessonEntry(
                    pattern=refl.explanation,
                    evidence=f"C{node.idx} (iter {node.iteration}): {refl.scenario_id} fixed",
                    source_iteration=node.iteration,
                )
            )
        elif refl.status_change == "regressed":
            pattern = refl.explanation
            if refl.prevention_or_next:
                pattern += f" | Prevention: {refl.prevention_or_next}"
            dag.accumulated_lessons.bad_patterns.append(
                LessonEntry(
                    pattern=pattern,
                    evidence=f"C{node.idx} (iter {node.iteration}): {refl.scenario_id} regressed",
                    source_iteration=node.iteration,
                )
            )
        elif refl.status_change == "still_failing":
            pattern = f"[INEFFECTIVE] {refl.explanation}"
            if refl.prevention_or_next:
                pattern += f" | Next: {refl.prevention_or_next}"
            dag.accumulated_lessons.bad_patterns.append(
                LessonEntry(
                    pattern=pattern,
                    evidence=f"C{node.idx} (iter {node.iteration}): {refl.scenario_id} still_failing",
                    source_iteration=node.iteration,
                )
            )


# ---------------------------------------------------------------------------
# ScenarioRegistry
# ---------------------------------------------------------------------------


def _extract_task_description(cycle_dir: str, scenario_id: str) -> str | None:
    """Extract task description from evaluation trace files.

    NOTE: The trace file parsing below is adapter-specific (Meta-ARE format).
    If the trace format doesn't match, the function returns None gracefully.
    The directory path (run/lite/) is documented in CLAUDE.md's Iteration
    Output Structure section.
    """
    import json as _json
    from pathlib import Path as _Path

    lite_dir = _Path(cycle_dir) / "run" / "lite"
    if not lite_dir.exists():
        return None

    for trace_file in lite_dir.glob("*.json"):
        try:
            with open(trace_file, encoding="utf-8") as f:
                data = _json.load(f)
            if data.get("scenario_id") != scenario_id:
                continue
            histories = data.get("per_agent_interaction_histories", {})
            for _agent_name, events in histories.items():
                for evt in events:
                    if evt.get("role") != "user":
                        continue
                    content = evt.get("content", "")
                    # Parse [TASK] format: extract after "Message:"
                    if "Message:" in content:
                        msg_idx = content.index("Message:") + len("Message:")
                        task_msg = content[msg_idx:].strip()
                        if len(task_msg) > 500:
                            task_msg = task_msg[:497] + "..."
                        return task_msg
                    # Fallback: return raw content (truncated)
                    if len(content) > 500:
                        return content[:497] + "..."
                    return content
                break  # only check first agent
        except Exception:
            continue
    return None


def _classify_scenario(entry: ScenarioEntry) -> str:
    """Re-compute the category for a scenario based on its history.

    Categories:
    - ``consistently_failing``: never passed
    - ``volatile``: pass↔fail transitions >= 2
    - ``stable_passing``: last 3+ consecutive passes
    - ``recently_fixed``: most recent result is pass, previous was fail
    - ``recently_broken``: most recent result is fail, previous was pass
    """
    if not entry.history:
        return "consistently_failing"

    statuses = [s.status for s in entry.history]

    # Count transitions
    transitions = sum(
        1 for i in range(1, len(statuses)) if statuses[i] != statuses[i - 1]
    )

    latest = statuses[-1]

    if latest == "pass" and len(statuses) >= 2 and statuses[-2] == "fail":
        return "recently_fixed"
    if latest == "fail" and len(statuses) >= 2 and statuses[-2] == "pass":
        return "recently_broken"
    if transitions >= 2:
        return "volatile"

    # Check stable passing (last N consecutive passes)
    consecutive_pass = 0
    for s in reversed(statuses):
        if s == "pass":
            consecutive_pass += 1
        else:
            break
    if consecutive_pass >= 3:
        return "stable_passing"

    if all(s == "fail" for s in statuses):
        return "consistently_failing"

    if latest == "pass":
        return "recently_fixed"

    return "consistently_failing"


def update_scenario_registry(
    dag: EvolutionDAG,
    node: EvolutionNode,
    scenario_impacts: list[ScenarioImpact],
) -> None:
    """Update the ScenarioRegistry with results from this iteration.

    1. Add ScenarioSnapshot to each scenario's history
    2. Re-classify each scenario's category
    3. For regressed scenarios: record sensitive_to_files
    4. For target scenarios: record attempted_fixes
    """
    # Get files changed from the base edge
    files_changed: list[str] = []
    base_edge_key = None
    if node.base_parent_idx is not None:
        base_edge_key = f"{node.base_parent_idx}->{node.idx}"
        base_edge = dag.edges.get(base_edge_key)
        if base_edge and base_edge.files_changed:
            files_changed = base_edge.files_changed

    for impact in scenario_impacts:
        sid = impact.scenario_id

        # Ensure entry exists
        if sid not in dag.scenario_registry:
            from autosaddler.v1.proposer.autosaddler.models import ScenarioEntry

            dag.scenario_registry[sid] = ScenarioEntry(scenario_id=sid)

        entry = dag.scenario_registry[sid]

        # 0. Extract task description if not yet stored
        if not getattr(entry, "task_description", None):
            cycle_dir = node.train_before_cycle_dir or node.train_after_cycle_dir
            if cycle_dir:
                task_desc = _extract_task_description(cycle_dir, sid)
                if task_desc:
                    entry.task_description = task_desc

        # 1. Add snapshot
        status = "pass" if impact.score_after >= 0.5 else "fail"
        rationale = impact.rationale_after
        entry.history.append(
            ScenarioSnapshot(
                iteration=node.iteration,
                candidate_idx=node.idx,
                score=impact.score_after,
                status=status,
                rationale=rationale,
            )
        )

        # 2. Re-classify
        entry.category = _classify_scenario(entry)

        # 3. Regressed → record sensitive files
        if impact.status_change == "regressed" and files_changed:
            for f in files_changed:
                if f not in entry.sensitive_to_files:
                    entry.sensitive_to_files.append(f)

        # 4. Record attempted fixes for target scenarios
        if node.patch_intent and sid in node.patch_intent.target_scenarios:
            result = "fixed" if impact.status_change == "fixed" else "not_fixed"
            failure_reason = None
            next_suggestion = None
            if result == "not_fixed" and node.patch_verdict:
                for refl in node.patch_verdict.reflections:
                    if refl.scenario_id == sid and refl.status_change in ("still_failing", "regressed"):
                        failure_reason = refl.explanation
                        next_suggestion = refl.prevention_or_next
                        break
            entry.attempted_fixes.append(
                AttemptedFix(
                    iteration=node.iteration,
                    candidate_idx=node.idx,
                    approach=node.patch_intent.approach,
                    result=result,
                    failure_reason=failure_reason,
                    prevention_or_next=next_suggestion,
                )
            )

        # 5. Root causes are backfilled from reflections in
        #    update_scenario_registry_from_reflections() after Session 2,
        #    because at this point verdict.reflections is still empty
        #    (Session 2 is deferred).


def update_scenario_registry_from_reflections(
    dag: EvolutionDAG,
    node: EvolutionNode,
) -> None:
    """Backfill ScenarioRegistry with reflection data after Session 2.

    Called after reflections are populated (deferred session 2) to fill in
    fields that were unavailable when ``update_scenario_registry`` first ran
    with an empty reflections list:

    1. ``AttemptedFix.failure_reason`` and ``prevention_or_next``
    2. ``ScenarioEntry.known_root_causes``
    """
    if not node.patch_verdict or not node.patch_verdict.reflections:
        return

    for refl in node.patch_verdict.reflections:
        sid = refl.scenario_id
        if sid not in dag.scenario_registry:
            continue

        entry = dag.scenario_registry[sid]

        # 1. Backfill attempted_fixes with failure_reason / prevention_or_next
        if refl.status_change in ("still_failing", "regressed"):
            for fix in reversed(entry.attempted_fixes):
                if fix.iteration == node.iteration and fix.candidate_idx == node.idx:
                    fix.failure_reason = refl.explanation
                    fix.prevention_or_next = refl.prevention_or_next
                    break

        # 2. Backfill known_root_causes
        if refl.status_change in ("fixed", "still_failing"):
            root_cause = refl.root_cause
            if root_cause and len(root_cause) > 10 and root_cause not in entry.known_root_causes:
                entry.known_root_causes.append(root_cause)
                # Keep list manageable
                if len(entry.known_root_causes) > 10:
                    entry.known_root_causes = entry.known_root_causes[-10:]

        # 2. Extract root causes
        if refl.status_change in ("fixed", "still_failing"):
            explanation = refl.explanation
            if explanation and explanation not in entry.known_root_causes:
                if len(explanation) > 10:
                    entry.known_root_causes.append(explanation)
                    if len(entry.known_root_causes) > 10:
                        entry.known_root_causes = entry.known_root_causes[-10:]
