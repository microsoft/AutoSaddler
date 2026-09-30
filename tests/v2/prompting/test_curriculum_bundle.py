from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema.validators import validator_for

from autosaddler.v2.config.registry import build_runtime
from autosaddler.v2.core.curriculum import CurriculumState
from autosaddler.v2.prompting.curriculum import (
    CURRICULUM_ROOT,
    arm_decision_schema,
    arm_scoring_schema,
    build_curriculum_bundle,
    pattern_extraction_schema,
)
from autosaddler.v2.prompting.history import build_history_bundle
from autosaddler.v2.prompting.models import session_output_validation_error


def run_curriculum(tmp_path: Path, registry, config):
    value = config(
        tmp_path,
        train_case_ids=("train-a", "train-b", "train-c"),
        development_case_ids=("dev-a",),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    runtime = build_runtime(path, run_id="bundle", registry=registry)
    runtime.engine.run()
    return runtime.store


def test_curriculum_bundle_renders_patterns_pull_history_and_decisions(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    store = run_curriculum(tmp_path, curriculum_registry, activesaddler_config)
    (pattern_id,) = CurriculumState.replay(store.events()).patterns

    files = build_curriculum_bundle(store, {"iteration": 2, "task_selection": {"ema_eta": 0.9}})

    manifest = json.loads(files[f"{CURRICULUM_ROOT}/manifest.json"])
    assert manifest["num_patterns"] == 1
    assert manifest["entry_points"]["patterns"] == f"{CURRICULUM_ROOT}/patterns.json"
    (row,) = json.loads(files[f"{CURRICULUM_ROOT}/patterns.json"])["patterns"]
    assert row["pattern_id"] == pattern_id
    assert row["num_observations"] == 2
    assert row["activity"] == pytest.approx(0.01)
    assert row["latest_score"]["iteration"] == 1
    pulls = json.loads(files[row["pull_history_path"]])["pulls"]
    assert [(pull["iteration"], pull["kind"]) for pull in pulls] == [(1, "patched")]
    assert pulls[0]["history_iteration_path"] == ".autosaddler/history/iterations/0001.json"
    detail = json.loads(files[row["detail_path"]])
    assert {tag["source"] for tag in detail["tags"]} == {"pre_patch"}
    decisions = json.loads(files[f"{CURRICULUM_ROOT}/decisions.json"])["decisions"]
    assert [decision["action"] for decision in decisions] == ["unseen_draw", "arm_pull"]
    assert row["observations"] == detail["observations"]

    history = build_history_bundle(store, {"train_case_ids": []}).workspace_files
    iteration = json.loads(history[".autosaddler/history/iterations/0001.json"])
    assert "sampling_action" not in iteration


def test_pull_history_and_case_history_carry_patch_intent_results_and_lessons(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    store = run_curriculum(tmp_path, curriculum_registry, activesaddler_config)

    files = build_curriculum_bundle(store, {"iteration": 2, "task_selection": {"ema_eta": 0.9}})

    (row,) = json.loads(files[f"{CURRICULUM_ROOT}/patterns.json"])["patterns"]
    (pull,) = json.loads(files[row["pull_history_path"]])["pulls"]
    assert pull["patch_intent"] and "schema_version" not in pull["patch_intent"]
    assert pull["lessons"] and pull["candidate_id"] and pull["working_parent_candidate_id"]
    assert [item["case_id"] for item in pull["case_outcomes"]] == pull["case_ids"]
    outcome = pull["case_outcomes"][0]
    assert outcome["status"] in {"fixed", "still_failing"}
    assert outcome["pattern_tags"][0]["root_cause"] == "The instruction omits the required behavior."
    assert [tag["pattern_id"] for tag in outcome["pattern_tags"]] == [row["pattern_id"]]

    cases = json.loads(files[f"{CURRICULUM_ROOT}/cases.json"])["case_history_paths"]
    assert set(row["case_history_paths"]) == set(row["case_ids"])
    history = json.loads(files[cases[outcome["case_id"]]])
    assert history["pattern_ids"] == [row["pattern_id"]]
    assert [item["iteration"] for item in history["evaluations"]] == [0, 1]
    assert history["evaluations"][1]["status"] == outcome["status"]
    assert history["evaluations"][0]["sampling_action"] == "unseen_draw"


def test_curriculum_bundle_requires_task_selection_context(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    store = run_curriculum(tmp_path, curriculum_registry, activesaddler_config)

    with pytest.raises(TypeError, match="task_selection context"):
        build_curriculum_bundle(store, {"iteration": 0})


def test_curriculum_schemas_are_valid_and_constrain_outputs() -> None:
    extraction = pattern_extraction_schema("x/v1", ["case-a", "case-b"])
    decision = arm_decision_schema("y/v1")
    scoring = arm_scoring_schema("z/v1", ["pattern-a", "pattern-b"])
    for schema in (extraction, decision, scoring):
        validator_for(schema).check_schema(schema)

    assert session_output_validation_error(
        extraction,
        {
            "schema_version": "x/v1",
            "new_patterns": [{"key": "k", "label": "L"}],
            "tags": [{"case_id": "case-c", "source": "pre_patch", "pattern_refs": ["k"], "root_cause": "r"}],
        },
    )
    assert session_output_validation_error(decision, {"schema_version": "y/v1", "action": "pull", "rationale": "r"}) is None
    one_score = {
        "pattern_id": "pattern-a",
        "severity": 1,
        "fixability": 1,
        "breadth": 1,
        "side_effect": 0,
        "rationale": "r",
    }
    assert session_output_validation_error(scoring, {"schema_version": "z/v1", "scores": [one_score]})
