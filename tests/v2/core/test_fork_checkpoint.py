from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from autosaddler.v2.cli import main
from autosaddler.v2.config.registry import build_runtime
from autosaddler.v2.storage.local import LocalRunStore


class CheckpointReached(RuntimeError):
    pass


def _write_config(root: Path, *, max_iterations: int) -> Path:
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
            "budget": {"max_rollouts": 100, "max_iterations": max_iterations},
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_fork_checkpoint_runs_only_target_iteration_and_preserves_source(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, max_iterations=2)

    def interrupt_after_seed(event) -> None:
        if event.event_type == "BudgetUpdated" and event.operation_id.endswith(":seed:development:budget"):
            raise CheckpointReached

    source = build_runtime(config_path, run_id="source", transition_hook=interrupt_after_seed)
    with pytest.raises(CheckpointReached):
        source.engine.run()
    source_events = source.store.events_path.read_bytes()
    source_manifest = source.store.read_json("manifest.json")
    source_sequence = len(source.store.events())
    assert not source.store.events_of_type("IterationStarted")

    _write_config(tmp_path, max_iterations=1)
    target = build_runtime(config_path, run_id="slot-1-rerun")
    fork_event = target.store.fork_from(source.store, through_sequence=source_sequence)
    result = target.engine.run()

    assert fork_event.payload["source_run_id"] == "source"
    assert fork_event.payload["source_sequence"] == source_sequence
    assert len(target.store.events_of_type("RunForked")) == 1
    assert len(target.store.events_of_type("RunResumed")) == 1
    assert len(target.store.events_of_type("IterationCompleted")) == 1
    assert result.iterations == 1
    assert sum(entry["kind"] == "rollout" for entry in target.ledger.entries()) == 6
    assert sum(entry["kind"] == "session" for entry in target.ledger.entries()) == 3
    assert source.store.events_path.read_bytes() == source_events
    assert source.store.read_json("manifest.json") == source_manifest
    target.store.validate_integrity()


def test_cli_forks_from_nonterminal_source_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, max_iterations=1)
    source = build_runtime(config_path, run_id="source")
    source.engine.run()
    source_events = source.store.events_path.read_bytes()
    source_manifest = source.store.read_json("manifest.json")
    through_sequence = source.store.events_of_type("IterationStarted")[0].sequence - 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autosaddler.v2.cli",
            "--config",
            str(config_path),
            "--run-id",
            "slot-1-cli-rerun",
            "--fork-from-run-id",
            "source",
            "--fork-through-sequence",
            str(through_sequence),
        ],
    )
    main()

    target = LocalRunStore(run_dir=tmp_path / "runs" / "slot-1-cli-rerun", run_id="slot-1-cli-rerun")
    assert "selected_candidate_id=" in capsys.readouterr().out
    assert len(target.events_of_type("RunForked")) == 1
    assert len(target.events_of_type("IterationCompleted")) == 1
    assert target.events_of_type("RunCompleted")
    assert source.store.events_path.read_bytes() == source_events
    assert source.store.read_json("manifest.json") == source_manifest
    target.validate_integrity()