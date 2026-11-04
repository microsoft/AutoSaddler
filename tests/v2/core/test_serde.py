from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from autosaddler.v2.core.domain import (
    ArtifactRef, Candidate, ChangeSummary, Cost, Evaluation, Observation, ParetoScreenParameters, PatchVerdict,
)
from autosaddler.v2.core.events import PARETO_DECISION_SCHEMA_VERSION, RunEvent, operation_id
from autosaddler.v2.core.policies import ParetoAcceptance
from autosaddler.v2.core.run_state import RunState
from autosaddler.v2.core.serde import record, session_result_from
from autosaddler.v2.storage.local import LocalRunStore, _evolution_dag


def test_legacy_cache_fields_migrate_to_canonical_input_buckets() -> None:
    result = session_result_from(
        {
            "status": "completed",
            "structured_output": {},
            "raw_response": "",
            "tool_calls": [],
            "usage": [
                {
                    "role": "optimizer",
                    "model": "claude-test",
                    "input_tokens": 10,
                    "cache_read_tokens": 3,
                    "cache_write_tokens": 2,
                    "output_tokens": 4,
                    "total_tokens": 19,
                }
            ],
            "cost": {
                "rollouts": 0,
                "sessions": 1,
                "input_tokens": 15,
                "output_tokens": 4,
                "wall_seconds": 1.0,
                "currency_amount": None,
            },
            "error": None,
        }
    )

    usage = result.usage[0]
    assert usage.input_tokens == 15
    assert usage.cached_input_tokens == 3
    assert usage.uncached_input_tokens == 12
    assert usage.total_tokens == 19
    assert usage.provider_metadata == {"cache_creation_input_tokens": 2}


def candidate(label: str, parents: tuple[str, ...] = ()) -> Candidate:
    return Candidate(
        candidate_id=f"sha256:{label}", parent_ids=parents, space="fixture", artifact=ArtifactRef(label, "candidate"),
        change=ChangeSummary(changed_units=("instruction",), added=1, removed=1, labels=()) if parents else None,
    )


def evaluation(item: Candidate, purpose: str, score: float, cost: float, repetitions: int = 2) -> Evaluation:
    split = "development" if purpose == "development" else "train"
    return Evaluation(
        evaluation_id=f"run:{item.candidate_id}:{purpose}", candidate_id=item.candidate_id,
        split=split, purpose=purpose, iteration=0, requested_case_ids=("case-a", "case-b"),
        observations=tuple(
            Observation.create(
                candidate_id=item.candidate_id, case_id=case_id, split=split, repetition=repetition,
                disposition="success", score=score, evaluator_fingerprint="priced-fixture",
                cost=Cost(rollouts=1, currency_amount=cost),
            )
            for case_id in ("case-a", "case-b") for repetition in range(repetitions)
        ),
        artifact_dir=ArtifactRef("evaluations/fixture", "evaluation"),
    )


def append(events: list[RunEvent], event_type: str, payload: dict) -> RunEvent:
    event = RunEvent.create(
        sequence=len(events) + 1, run_id="run", event_type=event_type,
        operation_id=operation_id("run", event_type, len(events)), payload=payload,
    )
    events.append(event)
    return event


@pytest.mark.parametrize("repetitions", [1, 2, 3, 5])
def test_pareto_decisions_round_trip_and_replay(repetitions: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from autosaddler.v2.core import policies

    seed = candidate("seed")
    child = candidate("child", (seed.candidate_id,))
    before = evaluation(seed, "train_before", 1.0, 2.0, repetitions)
    after = evaluation(child, "train_after", 1.0, 1.0, repetitions)
    seed_dev = evaluation(seed, "development", 1.0, 2.0, repetitions)
    child_dev = evaluation(child, "development", 1.0, 1.0, repetitions)
    events: list[RunEvent] = []
    append(events, "RunStarted", {"seed_candidate": record(seed)})
    append(events, "EvaluationCompleted", {"evaluation": record(seed_dev)})
    append(events, "ParetoArchiveUpdated", {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "repetitions": repetitions,
        "evaluation_ids": [seed_dev.evaluation_id],
        "archive_candidate_ids": [seed.candidate_id],
        "current_candidate_id": seed.candidate_id,
    })
    append(events, "CandidateFinalized", {"candidate": record(child)})
    for item in (before, after, child_dev):
        append(events, "EvaluationCompleted", {"evaluation": record(item)})
    verdict = ParetoAcceptance(ParetoScreenParameters(repetitions=repetitions, draws=100)).compare(before, after)
    payload = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "iteration": before.iteration,
        "candidate_id": child.candidate_id,
        "parent_evaluation_id": before.evaluation_id,
        "child_evaluation_id": after.evaluation_id,
        "verdict": record(verdict),
    }
    promotion = append(events, "AcceptanceDecided", payload)
    append(events, "ParetoArchiveUpdated", {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "repetitions": repetitions,
        "evaluation_ids": [seed_dev.evaluation_id, child_dev.evaluation_id],
        "archive_candidate_ids": [child.candidate_id],
        "current_candidate_id": child.candidate_id,
    })

    def unexpected_decision(*args, **kwargs):
        raise AssertionError("Replay must use persisted decisions")

    monkeypatch.setattr(ParetoAcceptance, "compare", unexpected_decision)
    monkeypatch.setattr(policies, "pareto_front", unexpected_decision)
    restored_events = tuple(RunEvent.from_mapping(record(event)) for event in events)
    assert restored_events[promotion.sequence - 1].payload["verdict"] == record(verdict)
    state = RunState.replay(restored_events)
    assert state.archive_candidate_ids == (child.candidate_id,)
    assert state.current_candidate_id == child.candidate_id
    assert state.selected_candidate_id is None
    nodes = {node["candidate_id"]: node for node in _evolution_dag(tuple(events), state)["nodes"]}
    assert {
        candidate_id: (node["development_aggregate"], node["development_cost"])
        for candidate_id, node in nodes.items()
    } == {seed.candidate_id: (1.0, 2.0), child.candidate_id: (1.0, 1.0)}
    assert state.attempted_rollouts == 0


def test_accuracy_verdict_and_state_remain_readable() -> None:
    verdict = PatchVerdict(0.0, 1.0, ("case-a",), True, "improved")
    seed = candidate("seed")
    events: list[RunEvent] = []
    append(events, "RunStarted", {"seed_candidate": record(seed)})
    append(events, "AcceptanceDecided", {"verdict": record(verdict)})
    state = RunState.replay(tuple(events))
    assert state.accepted_candidate_ids == (seed.candidate_id,)
    assert state == RunState.replay(tuple(events[:-1]))
    assert state.archive_candidate_ids == ()
    assert state.current_candidate_id is None


@pytest.mark.parametrize("mutation", [
    "version", "missing_version", "missing_evaluation", "current", "members", "repetitions", "incomplete",
    "member_type", "duplicate_member", "unknown_member", "wrong_split", "current_type", "duplicate_evaluation",
    "missing_current", "missing_repetitions",
])
def test_archive_replay_restores_recorded_membership(mutation: str) -> None:
    seed = candidate("seed")
    measured = evaluation(seed, "development", 1.0, 2.0)
    payload = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "repetitions": 2,
        "evaluation_ids": [measured.evaluation_id],
        "archive_candidate_ids": [seed.candidate_id],
        "current_candidate_id": seed.candidate_id,
    }
    if mutation == "version":
        payload["schema_version"] = "unsupported"
    elif mutation == "missing_version":
        del payload["schema_version"]
    elif mutation == "missing_evaluation":
        payload["evaluation_ids"] = ["unknown"]
    elif mutation == "current":
        payload["current_candidate_id"] = "unknown"
    elif mutation == "members":
        payload["archive_candidate_ids"] = []
    elif mutation == "repetitions":
        payload["repetitions"] = True
        measured = evaluation(seed, "development", 1.0, 2.0, repetitions=1)
    elif mutation == "incomplete":
        measured = replace(measured, observations=tuple(item for item in measured.observations if item.repetition == 0))
    elif mutation == "member_type":
        payload["archive_candidate_ids"] = seed.candidate_id
    elif mutation == "duplicate_member":
        payload["archive_candidate_ids"] = [seed.candidate_id, seed.candidate_id]
    elif mutation == "unknown_member":
        payload["archive_candidate_ids"] = ["sha256:unknown"]
    elif mutation == "wrong_split":
        measured = evaluation(seed, "train_before", 1.0, 2.0)
        payload["evaluation_ids"] = [measured.evaluation_id]
    elif mutation == "current_type":
        payload["current_candidate_id"] = None
    elif mutation == "duplicate_evaluation":
        payload["evaluation_ids"] = [measured.evaluation_id, measured.evaluation_id]
    elif mutation == "missing_current":
        del payload["current_candidate_id"]
    elif mutation == "missing_repetitions":
        del payload["repetitions"]
    events: list[RunEvent] = []
    append(events, "RunStarted", {"seed_candidate": record(seed)})
    append(events, "EvaluationCompleted", {"evaluation": record(measured)})
    append(events, "ParetoArchiveUpdated", payload)
    if mutation == "missing_current":
        with pytest.raises(KeyError):
            RunState.replay(tuple(events))
    else:
        state = RunState.replay(tuple(events))
        assert state.archive_candidate_ids == tuple(payload["archive_candidate_ids"])
        assert state.current_candidate_id == payload["current_candidate_id"]


@pytest.mark.parametrize("mutation", ["missing_evaluation", "future_evaluation", "missing_repetitions", "incomplete"])
def test_pareto_projection_requires_referenced_evaluation_inputs(mutation: str) -> None:
    seed = candidate("seed")
    measured = evaluation(seed, "development", 1.0, 2.0)
    payload = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "repetitions": 2,
        "evaluation_ids": [measured.evaluation_id],
        "archive_candidate_ids": [seed.candidate_id],
        "current_candidate_id": seed.candidate_id,
    }
    if mutation == "missing_evaluation":
        payload["evaluation_ids"] = ["unknown"]
    elif mutation == "missing_repetitions":
        del payload["repetitions"]
    elif mutation == "incomplete":
        measured = replace(measured, observations=tuple(item for item in measured.observations if item.repetition == 0))
    events: list[RunEvent] = []
    append(events, "RunStarted", {"seed_candidate": record(seed)})
    if mutation != "future_evaluation":
        append(events, "EvaluationCompleted", {"evaluation": record(measured)})
    append(events, "ParetoArchiveUpdated", payload)
    if mutation == "future_evaluation":
        append(events, "EvaluationCompleted", {"evaluation": record(measured)})
    state = RunState.replay(tuple(events))
    with pytest.raises(ValueError if mutation == "incomplete" else KeyError):
        _evolution_dag(tuple(events), state)


def test_pareto_projection_uses_archive_references_and_preserves_status_precedence() -> None:
    seed = candidate("seed")
    child = candidate("child", (seed.candidate_id,))
    seed_dev = evaluation(seed, "development", 1.0, 2.0)
    child_dev = evaluation(child, "development", 1.0, 1.0)
    newer = replace(evaluation(child, "development", 0.5, 9.0), evaluation_id="run:newer")
    events: list[RunEvent] = []
    append(events, "RunStarted", {"seed_candidate": record(seed)})
    append(events, "EvaluationCompleted", {"evaluation": record(seed_dev)})
    append(events, "ParetoArchiveUpdated", {
        "repetitions": 2, "evaluation_ids": [seed_dev.evaluation_id],
        "archive_candidate_ids": [seed.candidate_id], "current_candidate_id": seed.candidate_id,
    })
    append(events, "CandidateFinalized", {"candidate": record(child)})
    append(events, "EvaluationCompleted", {"evaluation": record(child_dev)})
    append(events, "ParetoArchiveUpdated", {
        "repetitions": 2, "evaluation_ids": [child_dev.evaluation_id],
        "archive_candidate_ids": [child.candidate_id], "current_candidate_id": child.candidate_id,
    })
    baseline = _evolution_dag(tuple(events), RunState.replay(tuple(events)))
    append(events, "EvaluationCompleted", {"evaluation": record(newer)})
    assert _evolution_dag(tuple(events), RunState.replay(tuple(events))) == baseline
    append(events, "EvolutionRecorded", {
        "candidate_id": child.candidate_id, "accepted": True,
        "development_aggregate": newer.aggregate_score, "iteration": 1,
    })
    nodes = {node["candidate_id"]: node for node in _evolution_dag(tuple(events), RunState.replay(tuple(events)))["nodes"]}
    assert len(nodes) == 2
    assert nodes[seed.candidate_id]["development_cost"] == 2.0
    assert nodes[seed.candidate_id]["status"] == "dominated"
    assert nodes[child.candidate_id]["development_aggregate"] == 1.0
    assert nodes[child.candidate_id]["development_cost"] == 1.0
    assert nodes[child.candidate_id]["status"] == "current"
    append(events, "BestCandidateSelected", {"candidate_id": child.candidate_id})
    nodes = {node["candidate_id"]: node for node in _evolution_dag(tuple(events), RunState.replay(tuple(events)))["nodes"]}
    assert nodes[child.candidate_id]["status"] == "selected"
    append(events, "ParetoArchiveUpdated", {
        "repetitions": 2, "evaluation_ids": [newer.evaluation_id],
        "archive_candidate_ids": [child.candidate_id], "current_candidate_id": child.candidate_id,
    })
    nodes = {node["candidate_id"]: node for node in _evolution_dag(tuple(events), RunState.replay(tuple(events)))["nodes"]}
    assert nodes[child.candidate_id]["development_aggregate"] == 0.5
    assert nodes[child.candidate_id]["development_cost"] == 9.0


@pytest.mark.parametrize("include_version", [True, False])
def test_pareto_projection_is_disposable_and_append_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_version: bool,
) -> None:
    store = LocalRunStore(run_dir=tmp_path / "run", run_id="run")
    store.initialize(resolved_config={}, resolved_entities={})
    seed = candidate("seed")
    child = candidate("child", (seed.candidate_id,))
    before = evaluation(seed, "train_before", 1, 2)
    after = evaluation(child, "train_after", 1, 1)
    seed_dev = evaluation(seed, "development", 1, 2)
    child_dev = evaluation(child, "development", 1, 1)
    store.append("RunStarted", "run:start", {"seed_candidate": record(seed)})
    store.append("CandidateFinalized", "run:child", {"candidate": record(child)})
    for item in (before, after, seed_dev, child_dev):
        store.append("EvaluationCompleted", item.evaluation_id, {"evaluation": record(item)})
    ordinary_decision = store.append("AcceptanceDecided", "run:ordinary-acceptance", {
        "candidate_id": seed.candidate_id,
        "verdict": record(PatchVerdict(0.0, 1.0, before.requested_case_ids, True, "improved")),
    })
    assert "pareto" not in store.read_json("snapshot.json")
    replay = RunState.replay
    replayed_sequences: list[int] = []

    def replay_persisted(events: tuple[RunEvent, ...]) -> RunState:
        assert events == store.events()
        replayed_sequences.append(events[-1].sequence)
        return replay(events)

    monkeypatch.setattr(RunState, "replay", replay_persisted)
    promotion = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "iteration": before.iteration,
        "candidate_id": child.candidate_id,
        "parent_evaluation_id": before.evaluation_id,
        "child_evaluation_id": after.evaluation_id,
        "verdict": record(ParetoAcceptance(ParetoScreenParameters(repetitions=2, draws=100)).compare(before, after)),
    }
    if not include_version:
        del promotion["schema_version"]
    decision = store.append("AcceptanceDecided", operation_id("run", "iteration", 0, "acceptance"), promotion)
    assert replayed_sequences == [store.events()[-1].sequence]
    completed = store.read_json("snapshot.json")["completed_operations"]
    assert decision.operation_id in completed
    assert ordinary_decision.operation_id not in completed
    nodes = {node["candidate_id"]: node for node in store.read_json("evolution_dag.json")["nodes"]}
    assert nodes[seed.candidate_id]["status"] == "evaluated"
    assert nodes[child.candidate_id]["status"] == "promoted"
    payload = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "repetitions": 2,
        "evaluation_ids": [seed_dev.evaluation_id, child_dev.evaluation_id],
        "archive_candidate_ids": [child.candidate_id],
        "current_candidate_id": child.candidate_id,
    }
    archive_operation = operation_id("run", "iteration", 0, "archive")
    event = store.append("ParetoArchiveUpdated", archive_operation, payload)
    assert replayed_sequences == [event.sequence - 1, event.sequence]
    assert store.append("ParetoArchiveUpdated", archive_operation, payload) == event
    assert replayed_sequences == [event.sequence - 1, event.sequence]
    with pytest.raises(RuntimeError, match="Idempotency conflict"):
        store.append("ParetoArchiveUpdated", archive_operation, {**payload, "current_candidate_id": "unknown"})
    event_bytes = store.events_path.read_bytes()
    snapshot = store.read_json("snapshot.json")
    dag = store.read_json("evolution_dag.json")
    assert snapshot["pareto"] == {"archive_candidate_ids": [child.candidate_id], "current_candidate_id": child.candidate_id}
    assert archive_operation in snapshot["completed_operations"]
    nodes = {node["candidate_id"]: node for node in dag["nodes"]}
    assert nodes[child.candidate_id]["status"] == "current"
    assert nodes[child.candidate_id]["development_cost"] == 1.0
    assert nodes[seed.candidate_id]["status"] == "dominated"
    for name in ("snapshot.json", "evolution_dag.json"):
        (store.run_dir / name).write_text("{}")
    store.refresh_projections()
    assert store.read_json("snapshot.json") == snapshot
    assert store.read_json("evolution_dag.json") == dag
    assert store.events_path.read_bytes() == event_bytes
    store.validate_integrity()


@pytest.mark.parametrize("mutation", [
    "version", "missing_version", "samples", "accepted", "cost", "input", "parameters", "missing_evaluation", "verdict_type",
    "verdict_shape", "invalid_parameters",
])
def test_screen_projection_reads_recorded_result_without_reconstructing_it(mutation: str) -> None:
    seed = candidate("seed")
    child = candidate("child", (seed.candidate_id,))
    before = evaluation(seed, "train_before", 1.0, 2.0)
    after = evaluation(child, "train_after", 1.0, 1.0)
    payload = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "iteration": before.iteration,
        "candidate_id": child.candidate_id,
        "parent_evaluation_id": before.evaluation_id,
        "child_evaluation_id": after.evaluation_id,
        "verdict": record(ParetoAcceptance(ParetoScreenParameters(repetitions=2, draws=100)).compare(before, after)),
    }
    if mutation == "version":
        payload["schema_version"] = "unsupported"
    elif mutation == "missing_version":
        del payload["schema_version"]
    elif mutation == "samples":
        payload["verdict"]["samples_per_arm"] = 1
    elif mutation == "accepted":
        payload["verdict"]["accepted"] = False
    elif mutation == "cost":
        payload["verdict"]["before_cost"] = 10.0
    elif mutation == "input":
        payload["child_evaluation_id"] = before.evaluation_id
    elif mutation == "parameters":
        payload["verdict"]["parameters"]["repetitions"] = 1
    elif mutation == "missing_evaluation":
        payload["parent_evaluation_id"] = "unknown"
    elif mutation == "verdict_type":
        payload["verdict"] = record(PatchVerdict(1.0, 1.0, before.requested_case_ids, True, "accepted"))
    elif mutation == "verdict_shape":
        payload["verdict"] = None
    elif mutation == "invalid_parameters":
        payload["verdict"]["parameters"]["repetitions"] = 0
    events: list[RunEvent] = []
    append(events, "RunStarted", {"seed_candidate": record(seed)})
    append(events, "CandidateFinalized", {"candidate": record(child)})
    for item in (before, after):
        append(events, "EvaluationCompleted", {"evaluation": record(item)})
    decision = append(events, "AcceptanceDecided", payload)
    state = RunState.replay(tuple(events))
    assert state == RunState.replay(tuple(events[:-1]))
    if mutation == "verdict_shape":
        with pytest.raises(TypeError):
            _evolution_dag(tuple(events), state)
    else:
        nodes = {node["candidate_id"]: node for node in _evolution_dag(tuple(events), state)["nodes"]}
        assert nodes[child.candidate_id]["status"] == ("declined" if mutation == "accepted" else "promoted")
    assert record(decision)["payload"]["verdict"] == payload["verdict"]


@pytest.mark.parametrize("cost", [1.0, 2.0])
def test_archive_replay_restores_recorded_order_and_current(cost: float) -> None:
    seed = candidate("z-seed")
    child = candidate("a-child", (seed.candidate_id,))
    seed_dev = evaluation(seed, "development", 1.0, 2.0)
    child_dev = evaluation(child, "development", 0.9 if cost == 1.0 else 1.0, cost)
    events: list[RunEvent] = []
    append(events, "RunStarted", {"seed_candidate": record(seed)})
    append(events, "CandidateFinalized", {"candidate": record(child)})
    append(events, "CandidateFinalized", {"candidate": record(seed)})
    for item in (child_dev, seed_dev):
        append(events, "EvaluationCompleted", {"evaluation": record(item)})
    prefix = tuple(events)
    payload = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "repetitions": 2,
        "evaluation_ids": [seed_dev.evaluation_id, child_dev.evaluation_id],
        "archive_candidate_ids": [seed.candidate_id, child.candidate_id],
        "current_candidate_id": child.candidate_id,
    }
    append(events, "ParetoArchiveUpdated", payload)
    state = RunState.replay(tuple(events))
    assert state.archive_candidate_ids == (seed.candidate_id, child.candidate_id)
    assert state.current_candidate_id == child.candidate_id
    reordered = {
        **payload,
        "evaluation_ids": [child_dev.evaluation_id, seed_dev.evaluation_id],
        "archive_candidate_ids": [child.candidate_id, seed.candidate_id],
        "current_candidate_id": seed.candidate_id,
    }
    restored = RunState.replay((*prefix, replace(events[-1], payload=reordered)))
    assert restored.archive_candidate_ids == (child.candidate_id, seed.candidate_id)
    assert restored.current_candidate_id == seed.candidate_id

    persisted = {**payload, "archive_candidate_ids": [seed.candidate_id], "current_candidate_id": seed.candidate_id}
    restored = RunState.replay((*prefix, replace(events[-1], payload=persisted)))
    assert restored.archive_candidate_ids == (seed.candidate_id,)
    assert restored.current_candidate_id == seed.candidate_id


@pytest.mark.parametrize("boundary", ["AcceptanceDecided", "ParetoArchiveUpdated"])
def test_pareto_decision_interruption_and_fork_preserve_replay(tmp_path: Path, boundary: str) -> None:
    store = LocalRunStore(run_dir=tmp_path / "run", run_id="run")
    store.initialize(resolved_config={"optimization": {"budget": {"max_iterations": 1}}}, resolved_entities={})
    seed = candidate("seed")
    child = candidate("child", (seed.candidate_id,))
    for item in (seed, child):
        store.write_json(item.artifact.uri, {"candidate_id": item.candidate_id}, kind="candidate")
    store.write_json("evaluations/fixture/results.json", {}, kind="evaluation")
    before = evaluation(seed, "train_before", 1.0, 2.0)
    after = evaluation(child, "train_after", 1.0, 1.0)
    seed_dev = evaluation(seed, "development", 1.0, 2.0)
    child_dev = evaluation(child, "development", 1.0, 1.0)
    store.append("RunStarted", "run:start", {"seed_candidate": record(seed)})
    store.append("CandidateFinalized", "run:child", {"candidate": record(child)})
    for item in (before, after, seed_dev, child_dev):
        store.append("EvaluationCompleted", item.evaluation_id, {"evaluation": record(item)})
    screen = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "iteration": before.iteration,
        "candidate_id": child.candidate_id,
        "parent_evaluation_id": before.evaluation_id,
        "child_evaluation_id": after.evaluation_id,
        "verdict": record(ParetoAcceptance(ParetoScreenParameters(repetitions=2, draws=100)).compare(before, after)),
    }
    archive = {
        "schema_version": PARETO_DECISION_SCHEMA_VERSION,
        "repetitions": 2,
        "evaluation_ids": [seed_dev.evaluation_id, child_dev.evaluation_id],
        "archive_candidate_ids": [child.candidate_id],
        "current_candidate_id": child.candidate_id,
    }
    if boundary == "ParetoArchiveUpdated":
        store.append("AcceptanceDecided", "run:iteration:0:acceptance", screen)
    payload = screen if boundary == "AcceptanceDecided" else archive
    decision_operation = operation_id("run", "iteration", 0, "acceptance" if boundary == "AcceptanceDecided" else "archive")

    def interrupt(event: RunEvent) -> None:
        if event.event_type == boundary:
            raise RuntimeError("interrupted after durable decision")

    store.transition_hook = interrupt
    with pytest.raises(RuntimeError, match="interrupted"):
        store.append(boundary, decision_operation, payload)
    durable_bytes = store.events_path.read_bytes()
    resumed = LocalRunStore(run_dir=store.run_dir, run_id="run")
    resumed.refresh_projections()
    assert resumed.append(boundary, decision_operation, payload) == resumed.events()[-1]
    assert resumed.events_path.read_bytes() == durable_bytes
    target = LocalRunStore(run_dir=tmp_path / "fork", run_id="fork")
    target.initialize(resolved_config={"optimization": {"budget": {"max_iterations": 2}}}, resolved_entities={})
    target.fork_from(resumed, through_sequence=len(resumed.events()))
    source_state = RunState.replay(resumed.events())
    target_state = RunState.replay(target.events())
    assert target_state.archive_candidate_ids == source_state.archive_candidate_ids
    assert target_state.current_candidate_id == source_state.current_candidate_id
    source_decisions = resumed.events_of_type("AcceptanceDecided")
    target_decisions = target.events_of_type("AcceptanceDecided")
    assert [event.payload["verdict"] for event in target_decisions] == [event.payload["verdict"] for event in source_decisions]
    expected_dag = resumed.read_json("evolution_dag.json")
    target_development = {
        evaluation.candidate_id: evaluation.evaluation_id
        for evaluation in target_state.evaluations.values() if evaluation.purpose == "development"
    }
    for node in expected_dag["nodes"]:
        if "development_evaluation_id" in node:
            node["development_evaluation_id"] = target_development[node["candidate_id"]]
    assert target.read_json("evolution_dag.json") == expected_dag
    assert all(event.operation_id.startswith("fork:") for event in target_decisions)
    assert resumed.events_path.read_bytes() == durable_bytes
    assert source_state.attempted_rollouts == target_state.attempted_rollouts == 0
    target.validate_integrity()