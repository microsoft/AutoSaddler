"""The engine runs any adaptive task-selection policy through the generic step interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from autosaddler.v2.config.registry import build_runtime, default_registry
from autosaddler.v2.core.domain import JsonValue, canonical_json
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.core.policies import TaskSelection
from autosaddler.v2.core.ports import BASE_SESSION_KINDS
from autosaddler.v2.core.scheduling import (
    AdaptiveTaskSelectionPolicy,
    DeferredRequest,
    IterationFeedback,
    NoSelection,
    SelectionRequest,
    SessionStep,
    StateStep,
)
from autosaddler.v2.prompting.models import SessionResult

NAMESPACE = "tests.probe"


def _probe_schema() -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "required": ["choice"],
        "properties": {"choice": {"type": "string"}},
        "additionalProperties": False,
    }


class ProbePromptPack:
    def __init__(self, base) -> None:
        self.base = base

    def session(self, kind: str, context: Mapping[str, JsonValue]) -> object:
        if kind != "probe":
            return self.base.session(kind, context)
        spec = self.base.session("reflect", {**context, "train_case_ids": []})
        return replace(
            spec,
            kind="probe",
            output_schema=_probe_schema(),
            workspace_files={
                "session_context.json": canonical_json(context) + "\n",
                ".autosaddler/fake_response.json": canonical_json({"choice": context["choice"]}) + "\n",
            },
        )


class ProbePolicy:
    """Asks one session for the case to train on, records a note, then selects it."""

    namespace = NAMESPACE
    required_session_kinds = frozenset({"probe"})

    def __init__(self, *, mode: str = "select") -> None:
        self.mode = mode

    def settings_record(self) -> dict[str, JsonValue]:
        return {"mode": self.mode}

    def session_timeouts(self) -> Mapping[str, float]:
        return {"probe": 7.0}

    def prompt_context(self, events: Sequence[RunEvent], iteration: int) -> Mapping[str, JsonValue]:
        return {"policy": "probe", "iteration": iteration}

    def next_selection_step(self, events: Sequence[RunEvent], request: SelectionRequest):
        if self.mode == "none":
            return NoSelection(reason="probe declined")
        if self.mode == "repeat":
            return StateStep(name="loop", payload={"iteration": request.iteration})
        states = [
            dict(event.payload)
            for event in events
            if event.event_type == "ExtensionStateChanged"
            and event.payload.get("namespace") == NAMESPACE
            and event.payload.get("iteration") == request.iteration
        ]
        if not states:
            return SessionStep(
                name="probe",
                kind="probe",
                context={"choice": request.train_cases[-1].case_id},
                stage="proposal.probe",
                validate=lambda result: None,
                record=lambda result: {
                    "iteration": request.iteration,
                    "choice": str((result.structured_output or {})["choice"]),
                },
            )
        if len(states) == 1:
            return StateStep(name="note", payload={"iteration": request.iteration, "note": "chosen"})
        return TaskSelection(case_ids=(str(states[0]["choice"]),), provenance={"policy": "probe"})

    def iteration_changes(self, events: Sequence[RunEvent], feedback: IterationFeedback):
        return ()

    def after_reflection(self, events: Sequence[RunEvent], feedback: IterationFeedback, lessons: Sequence[JsonValue]):
        return DeferredRequest(
            kind="probe",
            stage="proposal.probe",
            payload={"iteration": feedback.iteration, "choice": feedback.case_ids[0], "lessons": len(lessons)},
        )

    def deferred_context(self, events: Sequence[RunEvent], request: DeferredRequest) -> Mapping[str, JsonValue]:
        return {"choice": request.payload["choice"]}

    def deferred_failure_reason(self, events, request, result: SessionResult) -> str | None:
        return None

    def deferred_changes(self, events, request, result: SessionResult):
        return (
            {"iteration": request.payload["iteration"], "deferred": "first"},
            {"iteration": request.payload["iteration"], "deferred": "second"},
        )

    def deferred_exhausted_changes(self, events, request, error: str):
        return ({"iteration": request.payload["iteration"], "deferred": "exhausted"},)


def _runtime(tmp_path: Path, *, mode: str = "select", transition_hook=None):
    registry = default_registry()
    base_factory = registry.scenarios["fake"]
    registry.scenarios["fake"] = lambda **kwargs: replace(
        base := base_factory(**kwargs),
        prompt_pack=ProbePromptPack(base.prompt_pack),
        supported_session_kinds=BASE_SESSION_KINDS | {"probe"},
    )
    registry.task_selection["probe"] = lambda *, batch_size, seed: ProbePolicy(mode=mode)
    value = {
        "schema_version": "autosaddler/v2",
        "scenario": {
            "type": "fake",
            "settings": {
                "baseline": {"instruction": "baseline"},
                "target_component": "instruction",
                "improved_text": "improved",
                "train_case_ids": ["train-a", "train-b"],
                "development_case_ids": ["dev-a"],
            },
        },
        "optimization": {
            "task_selection": {"type": "probe", "batch_size": 1},
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
        "storage": {"type": "local", "run_root": str(tmp_path / "runs")},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return build_runtime(path, run_id="probe", registry=registry, transition_hook=transition_hook)


def _probe_operations(runtime) -> list[str]:
    return [
        event.operation_id.split(":", 1)[1]
        for event in runtime.store.events_of_type("ExtensionStateChanged")
        if event.payload.get("namespace") == NAMESPACE
    ]


def test_engine_executes_policy_steps_selection_and_deferred_work(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    runtime.engine.run()

    assert isinstance(runtime.policies.task_selection, AdaptiveTaskSelectionPolicy)
    (batch,) = runtime.store.events_of_type("BatchSampled")
    assert batch.payload["case_ids"] == ["train-b"]
    assert batch.payload["provenance"] == {"policy": "probe"}
    operations = _probe_operations(runtime)
    assert operations[:2] == ["iteration:0:selection:probe", "iteration:0:selection:note"]
    assert [operation.rsplit(":", 1)[1] for operation in operations[2:]] == ["0", "1"]
    assert all(":state:" in operation for operation in operations[2:])
    stages = [event.payload.get("stage") for event in runtime.store.events_of_type("SessionStarted")]
    assert stages == [
        "proposal.selection",
        "proposal.probe",
        "proposal.patch",
        "proposal.reflection",
        "proposal.probe",
    ]
    requests = [
        runtime.store.read_json(str(event.payload["request"]["uri"]))
        for event in runtime.store.events_of_type("SessionStarted")
    ]
    assert {request["spec"]["kind"]: request["timeout_seconds"] for request in requests}["probe"] == 7.0
    assert (runtime.store.run_dir / "strategy" / "tests.probe.json").is_file()


def test_engine_completes_iteration_without_selection(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, mode="none")

    runtime.engine.run()

    (completed,) = runtime.store.events_of_type("IterationCompleted")
    assert completed.payload["outcome"] == "no_selectable_cases"
    assert completed.payload["no_selection_reason"] == "probe declined"
    assert not runtime.store.events_of_type("BatchSampled")


def test_engine_rejects_policies_that_repeat_a_step(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, mode="repeat")

    with pytest.raises(RuntimeError, match="repeated step 'loop'"):
        runtime.engine.run()


class InterruptAfter:
    def __init__(self, target: int) -> None:
        self.target = target
        self.count = 0

    def __call__(self, event) -> None:
        self.count += 1
        if self.count == self.target:
            raise KeyboardInterrupt


def test_adaptive_steps_resume_without_repeating_work(tmp_path: Path) -> None:
    baseline = _runtime(tmp_path / "baseline")
    baseline.engine.run()
    expected = _probe_operations(baseline)
    total = len(baseline.store.events())

    for target in range(1, total + 1, 5):
        root = tmp_path / f"fault-{target:03d}"
        interrupted = _runtime(root, transition_hook=InterruptAfter(target))
        with pytest.raises(KeyboardInterrupt):
            interrupted.engine.run()
        resumed = _runtime(root)
        resumed.engine.run()
        assert _probe_operations(resumed) == expected, target
        sessions = [entry for entry in resumed.ledger.entries() if entry["kind"] == "session"]
        assert len(sessions) == len({entry["key"] for entry in sessions}) == 5, target
