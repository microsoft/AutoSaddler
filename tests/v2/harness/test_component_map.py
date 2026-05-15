from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosaddler.v2.core.ports import CompositionPlan, MutationContext
from autosaddler.v2.harness.component_map import ComponentMapHarnessSpace


def space(tmp_path: Path) -> ComponentMapHarnessSpace:
    return ComponentMapHarnessSpace(
        baseline={"tool.search.description": "Search records", "tool.send.description": "Send a message"},
        store_root=tmp_path / "candidates",
    )


def context(tmp_path: Path, iteration: int = 0) -> MutationContext:
    return MutationContext(iteration=iteration, patch_label="description", evidence=None, workspace_root=tmp_path / "sessions")


def test_component_map_seed_mutate_diff_and_materialize(tmp_path: Path) -> None:
    harness = space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path))
    harness.apply_updates(session, {"tool.search.description": "Search records by exact identifiers"})
    child = harness.finalize(session)

    assert seed.candidate_id.startswith("sha256:")
    assert child.parent_ids == (seed.candidate_id,)
    assert child.candidate_id != seed.candidate_id
    assert child.change.changed_units == ("tool.search.description",)
    assert harness.diff(seed, child).changed_units == ("tool.search.description",)
    materialized = harness.materialize(child, "evaluate")
    try:
        assert json.loads((materialized.root / "candidate.json").read_text())["tool.search.description"].endswith(
            "identifiers"
        )
    finally:
        materialized.release()


def test_component_map_rejects_schema_changes_and_no_op(tmp_path: Path) -> None:
    harness = space(tmp_path)
    seed = harness.seed()
    unknown = harness.begin_mutation(seed, context(tmp_path))
    with pytest.raises(ValueError, match="frozen schema"):
        harness.apply_updates(unknown, {"tool.unknown.description": "Not allowed"})

    no_op = harness.begin_mutation(seed, context(tmp_path, iteration=1))
    harness.apply_updates(no_op, {"tool.search.description": "Search records"})
    with pytest.raises(ValueError, match="change at least one"):
        harness.finalize(no_op)


def test_component_map_composes_declared_parents(tmp_path: Path) -> None:
    harness = space(tmp_path)
    seed = harness.seed()
    first_session = harness.begin_mutation(seed, context(tmp_path, iteration=1))
    harness.apply_updates(first_session, {"tool.search.description": "Search precisely"})
    first = harness.finalize(first_session)
    second_session = harness.begin_mutation(seed, context(tmp_path, iteration=2))
    harness.apply_updates(second_session, {"tool.send.description": "Send safely"})
    second = harness.finalize(second_session)

    composed = harness.compose(
        CompositionPlan(
            parents=(first.candidate_id, second.candidate_id),
            selections={"tool.send.description": second.candidate_id},
            overrides={},
            rationale="Combine independent improvements",
        )
    )

    assert composed.parent_ids == (first.candidate_id, second.candidate_id)
    assert composed.change.labels == ("composition",)
    materialized = harness.materialize(composed, "inspect")
    try:
        value = json.loads((materialized.root / "candidate.json").read_text())
        assert value == {
            "tool.search.description": "Search precisely",
            "tool.send.description": "Send safely",
        }
    finally:
        materialized.release()


def test_component_map_mutation_preserves_composed_parent(tmp_path: Path) -> None:
    harness = space(tmp_path)
    seed = harness.seed()
    first_session = harness.begin_mutation(seed, context(tmp_path, iteration=1))
    harness.apply_updates(first_session, {"tool.search.description": "Search precisely"})
    first = harness.finalize(first_session)
    second_session = harness.begin_mutation(seed, context(tmp_path, iteration=2))
    harness.apply_updates(second_session, {"tool.send.description": "Send safely"})
    second = harness.finalize(second_session)
    composed = harness.compose(
        CompositionPlan(
            parents=(first.candidate_id, second.candidate_id),
            selections={"tool.send.description": second.candidate_id},
            overrides={},
            rationale="Combine independent improvements",
        )
    )

    mutation = harness.begin_mutation(composed, context(tmp_path, iteration=3))
    harness.apply_updates(mutation, {"tool.search.description": "Search exact identifiers"})
    child = harness.finalize(mutation)

    assert child.parent_ids == (composed.candidate_id,)