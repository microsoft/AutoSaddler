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


def curriculum_payloads(store) -> list[dict]:
    return [
        dict(event.payload)
        for event in store.events_of_type("ExtensionStateChanged")
        if event.payload.get("namespace") == "autosaddler.curriculum"
    ]


def is_curriculum_transition(event) -> bool:
    if event.event_type in {"BatchSampled", "DeferredWorkScheduled", "DeferredWorkCompleted"}:
        return True
    if event.event_type == "ExtensionStateChanged":
        return event.payload.get("namespace") == "autosaddler.curriculum"
    return str(event.payload.get("stage", "")) in {
        "proposal.pattern_extraction",
        "proposal.arm_decision",
        "proposal.arm_scoring",
    }


def test_curriculum_resume_after_curriculum_transitions_never_duplicates_paid_work(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    def write(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        value = activesaddler_config(
            root,
            train_case_ids=("train-a", "train-b"),
            development_case_ids=("dev-a",),
            batch_size=1,
        )
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        return path

    baseline = build_runtime(write(tmp_path / "baseline"), run_id="run", registry=curriculum_registry)
    baseline_result = baseline.engine.run()
    transition_count = len(baseline.store.events())
    baseline_paid = baseline.ledger.entries()
    baseline_rollouts = sum(entry["kind"] == "rollout" for entry in baseline_paid)
    baseline_sessions = sum(entry["kind"] == "session" for entry in baseline_paid)
    baseline_changes = curriculum_payloads(baseline.store)
    baseline_batches = [event.payload for event in baseline.store.events_of_type("BatchSampled")]

    assert baseline_sessions == 10
    assert len(baseline_changes) == 7

    def exercise_transition(transition: int) -> None:
        config_path = write(tmp_path / f"fault-{transition:03d}")
        interrupted = build_runtime(
            config_path,
            run_id="run",
            transition_hook=InterruptAfterTransition(transition),
            registry=curriculum_registry,
        )
        with pytest.raises(InjectedInterruption):
            interrupted.engine.run()

        resumed = build_runtime(config_path, run_id="run", registry=curriculum_registry)
        result = resumed.engine.run()
        paid = resumed.ledger.entries()

        assert result.selected_candidate_id == baseline_result.selected_candidate_id, transition
        assert sum(entry["kind"] == "rollout" for entry in paid) == baseline_rollouts, transition
        assert sum(entry["kind"] == "session" for entry in paid) == baseline_sessions, transition
        assert len({(entry["kind"], entry["key"]) for entry in paid}) == len(paid), transition
        assert curriculum_payloads(resumed.store) == baseline_changes, transition
        assert [event.payload for event in resumed.store.events_of_type("BatchSampled")] == baseline_batches, transition
        resumed.store.validate_integrity()

        event_count = len(resumed.store.events())
        assert build_runtime(config_path, run_id="run", registry=curriculum_registry).engine.run() == result
        assert len(resumed.store.events()) == event_count

    # The passive fault test interrupts every generic transition; interrupt around every
    # curriculum-specific transition here to keep the fault matrix bounded.
    targets = sorted(
        {
            neighbor
            for event in baseline.store.events()
            if is_curriculum_transition(event)
            for neighbor in (event.sequence - 1, event.sequence, event.sequence + 1)
            if 1 <= neighbor <= transition_count
        }
    )
    assert len(targets) < transition_count
    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as executor:
        list(executor.map(exercise_transition, targets))
