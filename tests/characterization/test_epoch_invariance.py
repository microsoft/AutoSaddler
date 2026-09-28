"""Pin the passive task-selection path to the inputs and events of the pre-curriculum engine.

The fixture was captured from origin/main at e841543 by running this module as a script:

    PYTHONPATH=<origin-main>/src python tests/characterization/test_epoch_invariance.py

Fixed and epoch-shuffled runs must keep byte-identical resolved inputs, so runs created
before the adaptive task-selection interface existed can still be resumed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import yaml

from autosaddler.v2.config.registry import build_runtime
from autosaddler.v2.plugins.meta_are import plugin as meta_are_plugin
from autosaddler.v2.plugins.meta_are.prompt_pack import meta_are_prompt_composition_entity
from autosaddler.v2.prompting.assets import prompt_source_entities

FIXTURE = Path(__file__).parent / "fixtures" / "epoch_invariance.json"
VOLATILE_KEYS = frozenset({"timestamp", "wall_seconds", "run_invocation_id"})
RUNS = {
    "fixed_one_iteration": {
        "task_selection": {"type": "fixed", "batch_size": 2},
        "train_case_ids": ["train-a", "train-b"],
    },
    "epoch_shuffled_one_iteration": {
        "task_selection": {"type": "epoch_shuffled", "batch_size": 2, "seed": 7},
        "train_case_ids": ["train-a", "train-b", "train-c", "train-d"],
    },
}


def _config(root: Path, spec: dict) -> dict:
    return {
        "schema_version": "autosaddler/v2",
        "scenario": {
            "type": "fake",
            "settings": {
                "baseline": {"instruction": "baseline"},
                "target_component": "instruction",
                "improved_text": "improved",
                "train_case_ids": spec["train_case_ids"],
                "development_case_ids": ["dev-a", "dev-b"],
            },
        },
        "optimization": {
            "task_selection": spec["task_selection"],
            "acceptance": {"type": "matched_valid_strict_improvement"},
            "development": {"type": "full_on_accept"},
            "ranking": {"type": "mean_development_score"},
            "budget": {"max_rollouts": 100, "max_iterations": 1},
            "diagnosis_patch_timeout_seconds": 10,
        },
        "provider": {
            "type": "fake",
            "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
            "settings": {},
        },
        "storage": {"type": "local", "run_root": str(root / "runs")},
    }


def _normalize(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarize(root: Path) -> dict:
    runs = {}
    for name, spec in RUNS.items():
        run_root = root / name
        run_root.mkdir(parents=True)
        config_path = run_root / "config.yaml"
        config_path.write_text(yaml.safe_dump(_config(run_root, spec), sort_keys=False), encoding="utf-8")
        runtime = build_runtime(config_path, run_id="golden")
        runtime.engine.run()
        run_dir = runtime.store.run_dir
        resolved_paths = [run_dir / "resolved_config.yaml", *sorted((run_dir / "resolved").rglob("*"))]
        resolved = {
            path.relative_to(run_dir).as_posix(): _digest(
                path.read_text(encoding="utf-8").replace(str(run_root), "<RUN_ROOT>")
            )
            for path in resolved_paths
            if path.is_file()
        }
        events = [
            {
                "event_type": event["event_type"],
                "operation_id": event["operation_id"],
                "payload_sha256": _digest(json.dumps(_normalize(event["payload"]), sort_keys=True)),
            }
            for event in (json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines())
        ]
        runs[name] = {"resolved": resolved, "events": events}
    meta_are_prompts = {
        path: _digest(value if isinstance(value, str) else json.dumps(value, sort_keys=True))
        for path, value in {
            **prompt_source_entities(plugin_root=Path(meta_are_plugin.__file__).parent, plugin_name="meta_are"),
            "resolved/prompts/compositions.json": meta_are_prompt_composition_entity(),
        }.items()
    }
    return {"runs": runs, "meta_are_prompts": meta_are_prompts}


def test_passive_task_selection_keeps_resolved_inputs_and_events(tmp_path: Path) -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = summarize(tmp_path)

    for name in RUNS:
        assert actual["runs"][name]["resolved"] == expected["runs"][name]["resolved"], name
        assert actual["runs"][name]["events"] == expected["runs"][name]["events"], name
    assert actual["meta_are_prompts"] == expected["meta_are_prompts"]


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        summary = summarize(Path(directory))
    FIXTURE.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {FIXTURE}\n")
