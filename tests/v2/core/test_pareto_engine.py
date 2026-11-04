import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosaddler.v2 import cli
from autosaddler.v2.core.domain import Cost, ParetoScreenParameters
from autosaddler.v2.core.engine import AutoSaddlerEngine
from autosaddler.v2.core.policies import (
    BudgetPolicy,
    FixedTaskSelectionPolicy,
    FullOnAcceptDevelopment,
    ParetoAcceptance,
    ParetoDevelopmentRanking,
    PolicyBundle,
)
from autosaddler.v2.plugins.fake import FakePromptPack, FakeScenarioSettings, build_fake_components
from autosaddler.v2.providers.fake import FakeAgentProvider, PaidWorkLedger
from autosaddler.v2.core.serde import record
from autosaddler.v2.prompting.history import HISTORY_ROOT, build_history_bundle
from autosaddler.v2.storage.local import LocalRunStore


def make_engine(
    tmp_path, *, repetitions=1, max_rollouts=100, max_iterations=1,
    child_score=1.0, child_cost=1.0, development_score=None, development_cost=None,
    patches=("improved",), objectives=None, final_selection="score",
):
    store = LocalRunStore(run_dir=tmp_path, run_id="pareto")
    store.initialize(resolved_config={}, resolved_entities={})
    ledger = PaidWorkLedger(tmp_path / "audit.jsonl")
    scenario = build_fake_components(
        settings=FakeScenarioSettings(baseline={"instruction": "baseline"}, target_component="instruction",
                                      improved_text="improved", train_case_ids=("train",), development_case_ids=("dev",)),
        run_dir=tmp_path, store=store, ledger=ledger,
    )
    delegate = scenario.evaluator
    seed_id = scenario.harness_space.seed().candidate_id

    class PricedEvaluator:
        async def evaluate(self, candidate, cases, context):
            delegate.fingerprint = f"pareto-fixture:{context.operation_id}"
            measured = await delegate.evaluate(candidate, cases, context)
            score, cost = (1.0, 2.0) if candidate.candidate_id == seed_id else (child_score, child_cost)
            if candidate.candidate_id != seed_id and context.purpose == "development":
                score = child_score if development_score is None else development_score
                cost = child_cost if development_cost is None else development_cost
            content = json.loads((tmp_path / candidate.artifact.uri).read_text())["instruction"]
            if objectives is not None and content in objectives:
                score, cost = objectives[content]
            return replace(measured, observations=tuple(
                replace(observation, score=score, disposition="success", cost=Cost(rollouts=1, currency_amount=cost))
                for observation in measured.observations
            ))

    class ScriptedPromptPack:
        def session(self, kind, context):
            text = patches[context["iteration"]]
            return FakePromptPack(target_component="instruction", improved_text=text).session(kind, context)

    scenario = replace(scenario, evaluator=PricedEvaluator(), evaluation_repetitions=repetitions, prompt_pack=ScriptedPromptPack())
    return AutoSaddlerEngine(
        store=store, scenario=scenario, provider=FakeAgentProvider(ledger),
        final_selection=final_selection,
        policies=PolicyBundle(
            task_selection=FixedTaskSelectionPolicy(batch_size=1),
            acceptance=ParetoAcceptance(ParetoScreenParameters(repetitions=repetitions, draws=100)),
            development=FullOnAcceptDevelopment(), ranking=ParetoDevelopmentRanking(repetitions),
            budget=BudgetPolicy(max_rollouts=max_rollouts, max_iterations=max_iterations),
        ),
    )


@pytest.mark.parametrize("repetitions", [1, 2, 3, 5, 10])
def test_cost_only_improvement_changes_current(tmp_path: Path, repetitions: int) -> None:
    engine = make_engine(tmp_path, repetitions=repetitions)
    result = engine.run()
    state = engine._state()
    assert state.archive_candidate_ids == (result.selected_candidate_id,)
    assert state.current_candidate_id != state.accepted_candidate_ids[0]
    assert state.attempted_rollouts == 4 * repetitions
    assert len(engine.store.events_of_type("ParetoArchiveUpdated")) == 2
    assert not any(event.payload.get("session_kind") == "evolve" for event in engine.store.events_of_type("SessionStarted"))
    assert engine.store.read_json("result.json")["current_candidate_id"] == result.selected_candidate_id


@pytest.mark.parametrize("preference", ["score", "cost", "pareto", "accuracy", "other"])
def test_final_selection_does_not_change_pareto_current(tmp_path, preference, monkeypatch):
    engine = make_engine(tmp_path, child_score=0.95, final_selection=preference)
    if preference == "pareto":
        def unexpected(*args, **kwargs):
            raise AssertionError("Archive-only results must not select a single candidate")

        monkeypatch.setattr(engine, "_select_best", unexpected)
    result = engine.run()
    state = engine._state()
    seed_id, child_id = state.archive_candidate_ids
    expected_id = child_id if preference == "cost" else seed_id
    if preference == "pareto":
        expected_id = None
    assert state.current_candidate_id == child_id
    assert result.selected_candidate_id == state.selected_candidate_id == expected_id
    expected_score = None if preference == "pareto" else (0.95 if preference == "cost" else 1.0)
    assert result.development_score == expected_score
    assert len(engine.store.events_of_type("BestCandidateSelected")) == (0 if preference == "pareto" else 1)
    assert engine.store.events_of_type("IterationCompleted")[0].payload["resulting_candidate_id"] == child_id
    output = engine.store.read_json("result.json")
    assert output["selected_candidate_id"] == expected_id
    assert output["development_score"] == expected_score
    assert output["current_candidate_id"] == child_id
    assert result.current_candidate_id == child_id
    assert result.archive_candidate_ids == state.archive_candidate_ids
    archive_event = engine.store.events_of_type("ParetoArchiveUpdated")[-1]
    evaluations = {
        evaluation.candidate_id: evaluation for evaluation in state.evaluations.values()
        if evaluation.evaluation_id in archive_event.payload["evaluation_ids"]
    }
    expected_archive = [
        {
            "candidate_id": candidate_id,
            "development_aggregate": score,
            "development_cost": cost,
            "development_evaluation_id": evaluations[candidate_id].evaluation_id,
        }
        for candidate_id, score, cost in [(seed_id, 1.0, 2.0), (child_id, 0.95, 1.0)]
    ]
    assert list(result.archive) == output["archive"] == expected_archive
    assert engine.store.events_of_type("RunCompleted")[-1].payload["archive"] == expected_archive
    dag = engine.store.read_json("evolution_dag.json")
    assert dag["current_candidate_id"] == child_id
    assert dag["selected_candidate_id"] == expected_id
    assert dag["archive_candidate_ids"] == list(result.archive_candidate_ids)
    nodes = {node["candidate_id"]: node for node in dag["nodes"]}
    if preference == "pareto":
        assert nodes[child_id]["status"] == "current"
        assert not any(node["status"] == "selected" for node in nodes.values())
    else:
        assert nodes[expected_id]["status"] == "selected"
    bundle = build_history_bundle(engine.store, {})
    history = json.loads(bundle.workspace_files[f"{HISTORY_ROOT}/summary.json"])
    history_nodes = {node["candidate_id"]: node for node in history["candidates"]}
    assert history["current_candidate_id"] == child_id
    assert history["selected_candidate_id"] == expected_id
    assert history["archive_candidate_ids"] == list(result.archive_candidate_ids)
    for member in expected_archive:
        for key, value in member.items():
            assert nodes[member["candidate_id"]][key] == value
            assert history_nodes[member["candidate_id"]][key] == value
    events = engine.store.events()
    assert make_engine(tmp_path, child_score=0.95, final_selection=preference).run() == result
    assert engine.store.events() == events


@pytest.mark.parametrize("preference", ["score", "cost", "pareto", "other"])
def test_cli_prints_final_preference_result(tmp_path, preference, monkeypatch, capsys):
    engine = make_engine(tmp_path, child_score=0.95, final_selection=preference)
    monkeypatch.setattr(cli, "build_runtime", lambda *args, **kwargs: SimpleNamespace(engine=engine))
    monkeypatch.setattr(sys, "argv", ["autosaddler", "--config", "unused.yaml", "--run-id", "pareto"])
    cli.main()
    output = capsys.readouterr().out
    result = engine.store.read_json("result.json")
    if preference == "pareto":
        assert output.startswith("archive=")
        assert json.loads(output.removeprefix("archive=")) == result["archive"]
    else:
        assert output == (
            f"selected_candidate_id={result['selected_candidate_id']}\n"
            f"development_score={result['development_score']:.6f}\n"
        )


def test_legacy_pareto_result_loads_archive_from_recorded_decision(tmp_path):
    engine = make_engine(tmp_path, child_score=0.95)
    expected = engine.run()
    output = engine.store.read_json("result.json")
    del output["archive"]
    engine.store.write_json("result.json", output, kind="optimization-result")
    events = engine.store.events()
    paid = engine.provider.ledger.entries()
    assert make_engine(tmp_path, child_score=0.95).run() == expected
    assert engine.store.events() == events
    assert engine.provider.ledger.entries() == paid
    assert engine.store.read_json("result.json") == output


@pytest.mark.parametrize("preference", ["score", "cost"])
def test_final_selection_breaks_equal_objective_ties_by_creation_order(tmp_path, preference):
    engine = make_engine(tmp_path, development_cost=2.0, final_selection=preference)
    result = engine.run()
    state = engine._state()
    assert len(state.archive_candidate_ids) == 2
    assert result.selected_candidate_id == state.archive_candidate_ids[-1]


def test_final_selection_uses_archive_evaluation_references(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, child_score=0.95)
    engine.run()
    seed_id, child_id = engine._state().archive_candidate_ids
    development = {
        evaluation.candidate_id: evaluation for evaluation in engine._state().evaluations.values()
        if evaluation.purpose == "development"
    }
    seed_evaluation = replace(
        development[seed_id], evaluation_id="seed-updated",
        observations=tuple(
            replace(observation, score=0.9, cost=Cost(currency_amount=0.5))
            for observation in development[seed_id].observations
        ),
    )
    engine.store.append("EvaluationCompleted", "seed-updated", {"evaluation": record(seed_evaluation)})
    archive = engine.store.events_of_type("ParetoArchiveUpdated")[-1]
    engine.store.append("ParetoArchiveUpdated", "archive-updated", {
        **archive.payload,
        "evaluation_ids": [seed_evaluation.evaluation_id, development[child_id].evaluation_id],
    })
    unreferenced = replace(
        development[child_id], evaluation_id="child-unreferenced",
        observations=tuple(
            replace(observation, score=0.1, cost=Cost(currency_amount=0.01))
            for observation in development[child_id].observations
        ),
    )
    engine.store.append("EvaluationCompleted", "child-unreferenced", {"evaluation": record(unreferenced)})

    def unexpected(*args, **kwargs):
        raise AssertionError("Final selection must not recompute the Pareto front")

    monkeypatch.setattr("autosaddler.v2.core.engine.pareto_front", unexpected)
    assert engine._select_best("cost")[1] == seed_evaluation
    assert engine._select_best("score")[1] == development[child_id]
    archive_members = engine._result_archive()
    assert [member["development_evaluation_id"] for member in archive_members] == [
        seed_evaluation.evaluation_id, development[child_id].evaluation_id,
    ]
    assert [member["development_cost"] for member in archive_members] == [0.5, 1.0]

    engine.store.append("ParetoArchiveUpdated", "archive-child-only", {
        **archive.payload,
        "archive_candidate_ids": [child_id],
    })
    assert engine._select_best("score")[0].candidate_id == child_id
    assert [member["candidate_id"] for member in engine._result_archive()] == [child_id]


@pytest.mark.parametrize("child_score,child_cost,development_score,development_cost,promoted,archive_size", [
    (0.95, 1.0, 0.95, 1.0, True, 2),
    (1.0, 3.0, 1.0, 3.0, False, 1),
    (0.8, 1.0, 0.8, 1.0, False, 1),
    (1.0, 1.0, 0.8, 3.0, True, 1),
])
def test_screen_promotion_is_distinct_from_archive_membership(
    tmp_path, child_score, child_cost, development_score, development_cost, promoted, archive_size,
):
    engine = make_engine(tmp_path, child_score=child_score, child_cost=child_cost,
                         development_score=development_score, development_cost=development_cost)
    engine.run()
    state = engine._state()
    decision = engine.store.events_of_type("AcceptanceDecided")[0]
    completed = engine.store.events_of_type("IterationCompleted")[0]
    child_id = decision.payload["candidate_id"]
    assert decision.payload["verdict"]["accepted"] is promoted
    assert completed.payload["promoted_candidate_id"] == (child_id if promoted else None)
    assert len(state.archive_candidate_ids) == archive_size
    assert completed.payload["resulting_candidate_id"] == state.current_candidate_id
    assert (state.current_candidate_id == child_id) is (archive_size == 2)
    assert len(engine.store.events_of_type("EvaluationCompleted")) == (4 if promoted else 3)
    if promoted:
        archive = engine.store.events_of_type("ParetoArchiveUpdated")[-1]
        reflection = engine.store.events_of_type("DeferredWorkScheduled")[0]
        assert archive.sequence < reflection.sequence


@pytest.mark.parametrize("repetitions", [1, 2, 3])
def test_duplicate_content_preserves_age_and_next_iteration_uses_current(tmp_path, repetitions):
    engine = make_engine(
        tmp_path, repetitions=repetitions, max_iterations=3, child_score=0.95, child_cost=1.0,
        patches=("improved", "baseline", "next"), objectives={"next": (1.0, 0.5)},
    )
    engine.run()
    archives = engine.store.events_of_type("ParetoArchiveUpdated")
    seed_id = archives[0].payload["current_candidate_id"]
    first_id = archives[1].payload["current_candidate_id"]
    assert archives[2].payload["archive_candidate_ids"] == archives[1].payload["archive_candidate_ids"]
    assert archives[2].payload["current_candidate_id"] == first_id
    assert archives[2].payload["evaluation_ids"] == archives[1].payload["evaluation_ids"]
    assert [event.payload["working_candidate_id"] for event in engine.store.events_of_type("IterationStarted")] == [seed_id, first_id, first_id]
    assert engine._state().candidates[seed_id].parent_ids == ()
    assert engine._state().attempted_rollouts == 9 * repetitions
    assert len(engine._state().archive_candidate_ids) == 1
    assert engine._state().current_candidate_id not in {seed_id, first_id}


@pytest.mark.parametrize("repetitions", [2, 3, 5])
@pytest.mark.parametrize("start_iteration", [False, True])
def test_budget_only_prevents_starting_new_iterations(tmp_path, repetitions, start_iteration):
    budget = repetitions + 1 if start_iteration else repetitions - 1
    engine = make_engine(tmp_path, repetitions=repetitions, max_rollouts=budget, max_iterations=2)
    result = engine.run()
    state = engine._state()
    assert result.iterations == int(start_iteration)
    assert state.attempted_rollouts == (4 if start_iteration else 1) * repetitions
    assert state.attempted_rollouts > budget
    assert state.archive_candidate_ids == (result.selected_candidate_id,)
    assert result.development_score == 1.0
    assert not engine.store.events_of_type("RunFailed")
    assert not engine.store.events_of_type("EvaluationFailed")
    events = engine.store.events()
    assert make_engine(tmp_path, repetitions=repetitions, max_rollouts=budget, max_iterations=2).run() == result
    assert engine.store.events() == events


class InjectedInterruption(RuntimeError):
    pass


@pytest.mark.parametrize("preference", ["score", "cost"])
@pytest.mark.parametrize("boundary", ["BestCandidateSelected", "OptimizationCompleted"])
def test_final_selection_resumes_with_same_candidate(tmp_path, preference, boundary):
    engine = make_engine(tmp_path, child_score=0.95, final_selection=preference)

    def interrupt(event):
        if event.event_type == boundary:
            raise InjectedInterruption(boundary)

    engine.store.transition_hook = interrupt
    with pytest.raises(InjectedInterruption):
        engine.run()
    selected = engine.store.events_of_type("BestCandidateSelected")[0]
    paid = engine.provider.ledger.entries()
    resumed = make_engine(tmp_path, child_score=0.95, final_selection=preference)
    result = resumed.run()
    assert result.selected_candidate_id == selected.payload["candidate_id"]
    assert resumed.store.events_of_type("BestCandidateSelected") == (selected,)
    assert resumed.provider.ledger.entries() == paid
    assert resumed._state().current_candidate_id == engine._state().current_candidate_id


@pytest.mark.parametrize("preference", ["score", "pareto"])
def test_resume_after_every_transition_preserves_decisions_and_paid_work(tmp_path, preference):
    baseline = make_engine(tmp_path / "baseline", final_selection=preference, child_score=0.95)
    expected = baseline.run()
    expected_paid = baseline.provider.ledger.entries()

    def exercise(sequence):
        root = tmp_path / f"fault-{sequence}"
        engine = make_engine(root, final_selection=preference, child_score=0.95)

        def interrupt(event):
            if event.sequence == sequence:
                raise InjectedInterruption(event.event_type)

        engine.store.transition_hook = interrupt
        with pytest.raises(InjectedInterruption):
            engine.run()
        resumed = make_engine(root, final_selection=preference, child_score=0.95)
        assert resumed.run() == expected
        paid = resumed.provider.ledger.entries()
        assert [entry for entry in paid if entry["kind"] == "rollout"] == [
            entry for entry in expected_paid if entry["kind"] == "rollout"
        ]
        assert sum(entry["kind"] == "session" for entry in paid) == 2
        assert len({(entry["kind"], entry["key"]) for entry in paid}) == len(paid)
        assert resumed._state().archive_candidate_ids == baseline._state().archive_candidate_ids
        for event_type in ("AcceptanceDecided", "ParetoArchiveUpdated"):
            assert [event.payload for event in resumed.store.events_of_type(event_type)] == [
                event.payload for event in baseline.store.events_of_type(event_type)
            ]
        resumed.store.validate_integrity()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(exercise, range(1, len(baseline.store.events()) + 1)))


@pytest.mark.parametrize("boundary", ["EvaluationCompleted", "ParetoArchiveUpdated", "IterationCompleted"])
@pytest.mark.parametrize("budget", [2, 4])
def test_exhausted_budget_still_settles_durable_development_on_resume(tmp_path, boundary, budget):
    engine = make_engine(tmp_path, max_rollouts=budget)

    def interrupt(event):
        if event.event_type == boundary and len(engine.store.events_of_type("EvaluationCompleted")) == 4:
            raise InjectedInterruption(boundary)

    engine.store.transition_hook = interrupt
    with pytest.raises(InjectedInterruption):
        engine.run()
    resumed = make_engine(tmp_path, max_rollouts=budget)
    resumed.run()
    assert resumed._state().attempted_rollouts == 4
    assert len(resumed.store.events_of_type("ParetoArchiveUpdated")) == 2
    assert len(resumed.store.events_of_type("IterationCompleted")) == 1


@pytest.mark.parametrize("boundary", ["AcceptanceDecided", "ParetoArchiveUpdated"])
def test_resume_does_not_recompute_persisted_pareto_decisions(tmp_path, monkeypatch, boundary):
    engine = make_engine(tmp_path)

    def interrupt(event):
        if event.event_type == boundary and engine.store.events_of_type("AcceptanceDecided"):
            raise InjectedInterruption(boundary)

    engine.store.transition_hook = interrupt
    with pytest.raises(InjectedInterruption):
        engine.run()

    def unexpected(*args, **kwargs):
        raise AssertionError("Persisted decision was recomputed")

    monkeypatch.setattr(ParetoAcceptance, "compare", unexpected)
    if boundary == "ParetoArchiveUpdated":
        monkeypatch.setattr("autosaddler.v2.core.engine.pareto_front", unexpected)
    make_engine(tmp_path).run()

