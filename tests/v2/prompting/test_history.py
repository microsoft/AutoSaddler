from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosaddler.v2.core.domain import ArtifactRef, Candidate, ChangeSummary, sha256_digest
from autosaddler.v2.core.serde import record
from autosaddler.v2.prompting.history import build_history_bundle
from autosaddler.v2.storage.local import LocalRunStore


def test_history_bundle_is_navigable_and_digest_audited(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seed = _seed(store)
    patch = "diff --git a/prompt.md b/prompt.md\n+" + ("x" * 262_144) + "\n"
    diff = store.write_text("candidates/child.patch", patch, kind="git-diff")
    child = Candidate(
        candidate_id=sha256_digest("child"),
        parent_ids=(seed.candidate_id,),
        space="git",
        artifact=ArtifactRef(uri="candidates/child.json", kind="git-candidate"),
        change=ChangeSummary(
            changed_units=("prompt.md",),
            added=1,
            removed=0,
            labels=("capability",),
            diff=diff,
        ),
    )
    store.append(
        "CandidateFinalized",
        "run:iteration:0:candidate",
        {"iteration": 0, "candidate": record(child)},
    )
    store.append("BatchSampled", "run:iteration:0:batch", {"iteration": 0, "case_ids": ["train-a"]})
    store.append(
        "EvolutionRecorded",
        "run:iteration:0:evolution",
        {"iteration": 0, "candidate_id": child.candidate_id, "accepted": True},
    )
    store.append(
        "IterationCompleted",
        "run:iteration:0",
        {
            "schema_version": "autosaddler-iteration-completion/v1",
            "iteration": 0,
            "resulting_candidate_id": child.candidate_id,
            "evaluated_candidate_ids": [seed.candidate_id, child.candidate_id],
            "accepted": True,
            "outcome": "accepted",
        },
    )
    store.append(
        "DeferredWorkScheduled",
        "run:iteration:0:reflect",
        {
            "owning_iteration": 0,
            "obligation_id": sha256_digest("reflect"),
            "candidate_id": child.candidate_id,
            "parent_candidate_id": seed.candidate_id,
            "working_parent_candidate_id": seed.candidate_id,
            "selection_parent_ids": [seed.candidate_id],
            "component_sources": {},
            "selection_rationale": "Use the measured seed.",
            "session_kind": "reflect",
            "train_case_ids": ["train-a"],
            "train_before_aggregate": 0.0,
            "train_after_aggregate": 1.0,
            "accepted_by_minibatch_gate": True,
            "changed_components": ["prompt.md"],
            "diagnosis": "The instruction omitted a general constraint.",
        },
    )
    store.append(
        "ExtensionStateChanged",
        "run:lessons:0",
        {
            "namespace": "autosaddler.lessons",
            "schema_version": "autosaddler-lessons/v1",
            "candidate_id": child.candidate_id,
            "owning_iteration": 0,
            "lessons": [
                {
                    "scope": "global",
                    "statement": "Check the constraint against current evidence.",
                    "evidence_case_ids": ["train-a"],
                }
            ],
        },
    )

    first = build_history_bundle(store, {"train_case_ids": ["train-a"]})
    second = build_history_bundle(store, {"train_case_ids": ["train-a"]})

    assert first.workspace_files == second.workspace_files
    manifest = json.loads(first.workspace_files[".autosaddler/history/manifest.json"])
    assert manifest["through_sequence"] == 7
    assert manifest["omission_counts"] == {
        "candidates": 0,
        "iterations": 0,
        "lesson_records": 0,
        "current_cases": 0,
        "diffs": 0,
    }
    assert manifest["event_snapshot_sha256"].startswith("sha256:")
    for path, metadata in manifest["files"].items():
        text = first.workspace_files[path]
        assert metadata == {
            "sha256": sha256_digest(text),
            "bytes": len(text.encode("utf-8")),
        }

    summary = json.loads(first.workspace_files[".autosaddler/history/summary.json"])
    child_summary = next(item for item in summary["candidates"] if item["candidate_id"] == child.candidate_id)
    child_detail = json.loads(first.workspace_files[child_summary["detail_path"]])
    assert child_detail["status"] == "accepted"
    assert first.workspace_files[child_detail["diff_path"]] == patch
    assert len(summary["edge_paths"]) == 1
    edge = json.loads(first.workspace_files[summary["edge_paths"][0]])
    assert edge["parent_candidate_id"] == seed.candidate_id
    assert edge["child_candidate_id"] == child.candidate_id
    assert edge["diff_path"] == child_detail["diff_path"]

    current = json.loads(first.workspace_files[".autosaddler/history/current_batch.json"])
    assert current["relevant_iteration_ids"] == [0]
    case_path = current["relevant_case_paths"]["train-a"]
    relevant = json.loads(first.workspace_files[case_path])
    assert relevant["prior_iterations"][0]["diagnosis"].startswith("The instruction")
    iteration = json.loads(first.workspace_files[".autosaddler/history/iterations/0000.json"])
    assert iteration["completion_schema_version"] == "autosaddler-iteration-completion/v1"
    assert iteration["resulting_candidate_id"] == child.candidate_id
    assert iteration["evaluated_candidate_ids"] == [seed.candidate_id, child.candidate_id]
    assert first.compatibility_history["relevant_prior_iterations"] == relevant["prior_iterations"]


def test_history_bundle_rejects_diff_artifact_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seed = _seed(store)
    diff = store.write_text("candidates/child.patch", "original\n", kind="git-diff")
    child = Candidate(
        candidate_id=sha256_digest("child"),
        parent_ids=(seed.candidate_id,),
        space="git",
        artifact=ArtifactRef(uri="candidates/child.json", kind="git-candidate"),
        change=ChangeSummary(
            changed_units=("prompt.md",),
            added=1,
            removed=0,
            diff=diff,
        ),
    )
    store.append(
        "CandidateFinalized",
        "run:iteration:0:candidate",
        {"iteration": 0, "candidate": record(child)},
    )
    (store.run_dir / diff.uri).write_text("drifted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="History diff artifact drift"):
        build_history_bundle(store, {"train_case_ids": ["train-a"]})


def test_history_bundle_preserves_complete_event_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seed = _seed(store)
    parent = seed
    for iteration in range(2):
        child = Candidate(
            candidate_id=sha256_digest(f"child-{iteration}"),
            parent_ids=(parent.candidate_id,),
            space="git",
            artifact=ArtifactRef(
                uri=f"candidates/child-{iteration}.json",
                kind="git-candidate",
            ),
            change=ChangeSummary(
                changed_units=(f"prompt-{iteration}.md",),
                added=1,
                removed=0,
            ),
        )
        store.append(
            "CandidateFinalized",
            f"run:iteration:{iteration}:candidate",
            {"iteration": iteration, "candidate": record(child)},
        )
        store.append(
            "BatchSampled",
            f"run:iteration:{iteration}:batch",
            {"iteration": iteration, "case_ids": [f"train-{iteration}"]},
        )
        store.append(
            "ExtensionStateChanged",
            f"run:lessons:{iteration}",
            {
                "namespace": "autosaddler.lessons",
                "candidate_id": child.candidate_id,
                "owning_iteration": iteration,
                "lessons": [],
            },
        )
        parent = child

    bundle = build_history_bundle(
        store,
        {"train_case_ids": ["train-0", "train-1"]},
    )
    manifest = json.loads(bundle.workspace_files[".autosaddler/history/manifest.json"])

    assert manifest["limits"] == {
        "max_candidates": None,
        "max_iterations": None,
        "max_lesson_records": None,
        "max_current_cases": None,
        "max_diff_bytes": None,
        "max_total_diff_bytes": None,
    }
    assert manifest["included"]["candidate_ids"] == [
        seed.candidate_id,
        sha256_digest("child-0"),
        parent.candidate_id,
    ]
    assert manifest["included"]["iteration_ids"] == [0, 1]
    assert manifest["included"]["lesson_records"] == 2
    assert manifest["included"]["current_case_ids"] == ["train-0", "train-1"]
    assert manifest["omission_counts"] == {
        "candidates": 0,
        "iterations": 0,
        "lesson_records": 0,
        "current_cases": 0,
        "diffs": 0,
    }


def test_history_bundle_rejects_malformed_iteration_events(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)
    store.append(
        "BatchSampled",
        "run:iteration:0:batch",
        {"iteration": "0", "case_ids": ["train-a"]},
    )

    with pytest.raises(TypeError, match="BatchSampled event must contain an integer iteration"):
        build_history_bundle(store, {"train_case_ids": ["train-a"]})


def test_history_bundle_keeps_accepted_status_after_identical_declined_proposal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seed = _seed(store)
    store.append(
        "EvolutionRecorded",
        "run:iteration:0:evolution",
        {"iteration": 0, "candidate_id": seed.candidate_id, "accepted": True},
    )
    store.append(
        "EvolutionRecorded",
        "run:iteration:1:evolution",
        {"iteration": 1, "candidate_id": seed.candidate_id, "accepted": False},
    )

    bundle = build_history_bundle(store, {"train_case_ids": []})
    summary = json.loads(bundle.workspace_files[".autosaddler/history/summary.json"])
    seed_summary = next(item for item in summary["candidates"] if item["candidate_id"] == seed.candidate_id)

    assert seed_summary["status"] == "accepted"


def _store(tmp_path: Path) -> LocalRunStore:
    store = LocalRunStore(run_dir=tmp_path / "run", run_id="run")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "test"}},
    )
    return store


def _seed(store: LocalRunStore) -> Candidate:
    seed = Candidate(
        candidate_id=sha256_digest("seed"),
        parent_ids=(),
        space="git",
        artifact=ArtifactRef(uri="candidates/seed.json", kind="git-candidate"),
    )
    store.append("RunStarted", "run:start", {"seed_candidate": record(seed)})
    return seed
