"""Initial/re-evaluation comparison and ScenarioImpact computation.

All computation is performed by the outer loop (Python), not the Agent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from autosaddler.v1.proposer.autosaddler.models import ScenarioImpact

logger = logging.getLogger(__name__)


def compute_scenario_impact(
    scenario_id: str,
    score_before: float,
    score_after: float,
    rationale_before: str | None = None,
    rationale_after: str | None = None,
) -> ScenarioImpact:
    """Compute the status change for a single scenario.

    Binary scores (0 or 1) yield exactly one of four transitions:
    - 0 → 1: "fixed"
    - 1 → 0: "regressed"
    - 0 → 0: "still_failing"
    - 1 → 1: "still_passing"
    """
    before_pass = score_before >= 0.5
    after_pass = score_after >= 0.5

    if not before_pass and after_pass:
        status_change = "fixed"
    elif before_pass and not after_pass:
        status_change = "regressed"
    elif not before_pass and not after_pass:
        status_change = "still_failing"
    else:
        status_change = "still_passing"

    return ScenarioImpact(
        scenario_id=scenario_id,
        score_before=score_before,
        score_after=score_after,
        status_change=status_change,
        rationale_before=rationale_before,
        rationale_after=rationale_after,
    )


def compute_all_scenario_impacts(
    before_results: dict[str, dict[str, Any]],
    after_results: dict[str, dict[str, Any]],
    mini_batch_ids: list[str],
) -> list[ScenarioImpact]:
    """Compute ScenarioImpact for all scenarios in the mini-batch.

    Parameters
    ----------
    before_results:
        Mapping of scenario_id → {score, rationale, ...} from initial evaluation.
    after_results:
        Mapping of scenario_id → {score, rationale, ...} from re-evaluation.
    mini_batch_ids:
        List of scenario IDs in this mini-batch.

    Returns
    -------
    List of ScenarioImpact, one per scenario in the mini-batch.
    """
    impacts: list[ScenarioImpact] = []

    for scenario_id in mini_batch_ids:
        before = before_results.get(scenario_id, {})
        after = after_results.get(scenario_id, {})

        score_before = float(before.get("score", 0))
        score_after = float(after.get("score", 0))

        rationale_before = before.get("rationale")
        rationale_after = after.get("rationale")

        impact = compute_scenario_impact(
            scenario_id=scenario_id,
            score_before=score_before,
            score_after=score_after,
            rationale_before=rationale_before,
            rationale_after=rationale_after,
        )
        impacts.append(impact)

    return impacts


def parse_evaluation_results(cycle_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Parse evaluation results from a cycle directory.

    Reads ``run/output.jsonl`` and extracts per-scenario scores and rationales.

    Parameters
    ----------
    cycle_dir:
        Path to the cycle directory containing ``run/output.jsonl``.

    Returns
    -------
    Mapping of scenario_id → {"score": float, "rationale": str | None, "status": str}.
    """
    output_path = Path(cycle_dir) / "run" / "output.jsonl"
    results: dict[str, dict[str, Any]] = {}

    if not output_path.exists():
        logger.warning("output.jsonl not found at %s", output_path)
        return results

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON line in %s: %s", output_path, line[:100])
                continue

            # scenario_id may be at top level, or inside metadata, or use task_id
            metadata = entry.get("metadata", {})
            scenario_id = (
                entry.get("scenario_id")
                or metadata.get("scenario_id")
                or entry.get("task_id", "")
            )
            if not scenario_id:
                continue

            # score: prefer top-level numeric score, fallback to status string
            if "score" in entry and entry["score"] is not None:
                score = float(entry["score"])
            else:
                status = metadata.get("status", entry.get("status", ""))
                score = 1.0 if status in ("passed", "success") else 0.0

            status = metadata.get("status", entry.get("status", ""))

            rationale = (
                metadata.get("rationale")
                or entry.get("rationale")
                or entry.get("failure_reason")
                or entry.get("reason")
            )

            results[scenario_id] = {
                "score": score,
                "status": status,
                "rationale": rationale,
                "has_exception": metadata.get("has_exception", entry.get("has_exception", False)),
            }

    return results


def compute_pass_rate(results: dict[str, dict[str, Any]], scenario_ids: list[str] | None = None) -> float:
    """Compute pass rate from evaluation results.

    Parameters
    ----------
    results:
        Parsed evaluation results.
    scenario_ids:
        If provided, only compute pass rate for these scenarios.
        Otherwise, compute for all scenarios in results.

    Returns
    -------
    Pass rate as a float in [0, 1].
    """
    if scenario_ids is not None:
        scores = [results.get(sid, {}).get("score", 0.0) for sid in scenario_ids]
    else:
        scores = [r.get("score", 0.0) for r in results.values()]

    if not scores:
        return 0.0
    return sum(scores) / len(scores)
