"""Shared rendering for failure-pattern tables and observation histories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autosaddler.v1.proposer.autosaddler.pattern_registry import (
        FailurePattern,
        PatternObservation,
        PatternRegistry,
    )


PATTERN_TABLE_HEADER = "| Pattern | Activity | Observations | #Scen | LastObs | Label | Scenarios |"
PATTERN_TABLE_SEPARATOR = "|---------|----------|--------------|-------|---------|-------|-----------|"


def _format_scenario_ids(scenario_ids: Sequence[str]) -> str:
    return "[" + ", ".join(sorted(set(scenario_ids))) + "]"


def format_pattern_observation(observation: PatternObservation) -> str:
    """Render one observation without discarding its scenario-level evidence."""
    evaluated = observation.evaluated_scenario_ids
    tagged = observation.tagged_scenario_ids
    if not evaluated and not tagged:
        return (
            f"{observation.iteration}:?/? (active={observation.active:.2f}), "
            "tagged=(not recorded), evaluated=(not recorded)"
        )
    return (
        f"{observation.iteration}:{len(set(tagged))}/{len(set(evaluated))} "
        f"(active={observation.active:.2f}), "
        f"tagged={_format_scenario_ids(tagged)}, "
        f"evaluated={_format_scenario_ids(evaluated)}"
    )


def format_pattern_observations(pattern: FailurePattern) -> str:
    """Render a pattern's complete chronological observation history."""
    observations = sorted(pattern.observations, key=lambda item: item.iteration)
    return "; ".join(format_pattern_observation(item) for item in observations) or "(none)"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_pattern_table(
    registry: PatternRegistry,
    scores: Mapping[str, float],
    *,
    include_untagged: bool,
) -> str:
    """Render the canonical pattern table used by prompts and ``pattern list``."""
    patterns = registry.list_patterns()
    if not include_untagged:
        patterns = [pattern for pattern in patterns if pattern.tuples]
    patterns.sort(
        key=lambda pattern: scores.get(pattern.pattern_id, 0.0),
        reverse=True,
    )

    rows = [PATTERN_TABLE_HEADER, PATTERN_TABLE_SEPARATOR]
    for pattern in patterns:
        scenario_ids = sorted(registry.get_scenarios_for_pattern(pattern.pattern_id))
        observations = _escape_table_cell(format_pattern_observations(pattern))
        label = _escape_table_cell(pattern.label)
        scenarios = _escape_table_cell(", ".join(scenario_ids) or "(none)")
        rows.append(
            f"| {pattern.pattern_id} "
            f"| {scores.get(pattern.pattern_id, 0.0):.2f} "
            f"| {observations} | {len(scenario_ids)} "
            f"| {pattern.last_observed_iteration} | {label} | {scenarios} |"
        )
    return "\n".join(rows)
