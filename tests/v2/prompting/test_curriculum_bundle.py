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


def run_curriculum(tmp_path: Path):
    value = {
        "schema_version": "autosaddler/v2",
        "scenario": {
            "type": "fake",
            "settings": {
                "baseline": {"instruction": "baseline"},
                "target_component": "instruction",
                "improved_text": "improved",
                "train_case_ids": ["train-a", "train-b", "train-c"],
                "development_case_ids": ["dev-a"],
            },
        },
        "optimization": {
            "task_selection": {
                "type": "activesaddler",
                "batch_size": 2,
                "seed": 0,
                "settings": {"softmax_temperature": 0.15, "min_prob": 0.0, "ema_eta": 0.9},
            },
            "acceptance": {"type": "matched_valid_strict_improvement"},
            "development": {"type": "full_on_accept"},
            "ranking": {"type": "mean_development_score"},
            "budget": {"max_rollouts": 100, "max_iterations": 2},
            "diagnosis_patch_timeout_seconds": 10,
        },
        "provider": {
            "type": "fake",
            "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
            "settings": {},
        },
        "storage": {"type": "local", "run_root": str(tmp_path / "runs")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    runtime = build_runtime(path, run_id="bundle")
    runtime.engine.run()
    return runtime.store


def test_curriculum_bundle_renders_patterns_pull_history_and_decisions(tmp_path: Path) -> None:
    store = run_curriculum(tmp_path)
    (pattern_id,) = CurriculumState.replay(store.events()).patterns

    files = build_curriculum_bundle(store, {"iteration": 2, "curriculum": {"ema_eta": 0.9}})

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

    history = build_history_bundle(store, {"train_case_ids": []}).workspace_files
    iteration = json.loads(history[".autosaddler/history/iterations/0001.json"])
    assert iteration["sampling_action"] == "arm_pull"
    assert iteration["pulled_arm_id"] == pattern_id


def test_curriculum_bundle_requires_curriculum_context(tmp_path: Path) -> None:
    store = run_curriculum(tmp_path)

    with pytest.raises(TypeError, match="curriculum context"):
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
            "symptoms": [],
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
