from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from autosaddler.v2.config.registry import build_runtime


class InjectedInterruption(RuntimeError):
    pass


class InterruptAfterTransition:
    def __init__(self, target: int) -> None:
        self.target = target
        self.count = 0

    def __call__(self, event) -> None:
        self.count += 1
        if self.count == self.target:
            raise InjectedInterruption(f"Interrupted after transition {self.target}: {event.event_type}")


def write_config(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "autosaddler/v2",
        "scenario": {
            "type": "fake",
            "settings": {
                "baseline": {"instruction": "baseline"},
                "target_component": "instruction",
                "improved_text": "improved",
                "train_case_ids": ["train-a", "train-b"],
                "development_case_ids": ["dev-a", "dev-b"],
            },
        },
        "optimization": {
            "task_selection": {"type": "fixed", "batch_size": 2},
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
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_resume_after_every_transition_never_duplicates_paid_work(tmp_path: Path) -> None:
    baseline = build_runtime(write_config(tmp_path / "baseline"), run_id="run")
    baseline_result = baseline.engine.run()
    transition_count = len(baseline.store.events())
    baseline_paid = baseline.ledger.entries()

    assert transition_count > 30
    assert sum(entry["kind"] == "rollout" for entry in baseline_paid) == 8
    assert sum(entry["kind"] == "session" for entry in baseline_paid) == 3

    def exercise_transition(transition: int) -> None:
        root = tmp_path / f"fault-{transition:03d}"
        config_path = write_config(root)
        injector = InterruptAfterTransition(transition)
        interrupted = build_runtime(config_path, run_id="run", transition_hook=injector)
        with pytest.raises(InjectedInterruption):
            interrupted.engine.run()

        resumed = build_runtime(config_path, run_id="run")
        result = resumed.engine.run()
        paid = resumed.ledger.entries()

        assert result.selected_candidate_id == baseline_result.selected_candidate_id, transition
        assert result.development_score == baseline_result.development_score, transition
        assert sum(entry["kind"] == "rollout" for entry in paid) == 8, transition
        assert sum(entry["kind"] == "session" for entry in paid) == 3, transition
        assert len({(entry["kind"], entry["key"]) for entry in paid}) == len(paid), transition
        resumed.store.validate_integrity()

        event_count = len(resumed.store.events())
        assert build_runtime(config_path, run_id="run").engine.run() == result
        assert len(resumed.store.events()) == event_count

    with ThreadPoolExecutor(max_workers=min(8, transition_count)) as executor:
        list(executor.map(exercise_transition, range(1, transition_count + 1)))