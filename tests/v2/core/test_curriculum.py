from __future__ import annotations

import math

import pytest

from autosaddler.v2.core.curriculum import (
    CURRICULUM_NAMESPACE,
    CURRICULUM_SCHEMA_VERSION,
    ActiveSaddlerTaskSelectionPolicy,
    ArmScore,
    CurriculumState,
    PatternObservation,
    activity_ema,
    derive_pattern_id,
    failing_case_ids,
    pattern_observations,
    resolve_action,
    softmax_floor,
)
from autosaddler.v2.core.domain import Case
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.core.scheduling import AdaptiveTaskSelectionPolicy

BEFORE = "sha256:before"
AFTER = "sha256:after"
PARENT = "sha256:" + "a" * 64
CHILD = "sha256:" + "b" * 64


def event(sequence: int, event_type: str, payload: dict) -> RunEvent:
    return RunEvent.create(
        sequence=sequence,
        run_id="run",
        event_type=event_type,
        operation_id=f"run:{event_type}:{sequence}",
        payload=payload,
    )


def change(sequence: int, kind: str, iteration: int, **payload) -> RunEvent:
    return event(
        sequence,
        "ExtensionStateChanged",
        {
            "namespace": CURRICULUM_NAMESPACE,
            "schema_version": CURRICULUM_SCHEMA_VERSION,
            "change": kind,
            "iteration": iteration,
            **payload,
        },
    )


def tag(pattern_id: str, case_id: str, *, source: str = "pre_patch", evaluation_id: str = BEFORE) -> dict:
    return {
        "pattern_id": pattern_id,
        "case_id": case_id,
        "candidate_id": PARENT if source == "pre_patch" else CHILD,
        "evaluation_id": evaluation_id,
        "source": source,
        "root_cause": "missing capability",
    }


def registry_events() -> list[RunEvent]:
    return [
        event(1, "BatchSampled", {"iteration": 0, "case_ids": ["c1", "c2"], "provenance": {"policy": "activesaddler"}}),
        change(
            2,
            "patterns_extracted",
            0,
            new_patterns=[
                {"pattern_id": "pattern-old", "key": "old", "label": "Old failure"},
                {"pattern_id": "pattern-new", "key": "new", "label": "Post-patch failure"},
            ],
            tags=[
                tag("pattern-old", "c1"),
                tag("pattern-old", "c2"),
                tag("pattern-old", "c2", source="post_patch", evaluation_id=AFTER),
                tag("pattern-new", "c2", source="post_patch", evaluation_id=AFTER),
            ],
        ),
    ]


def test_softmax_floor_matches_activesaddler_sampler() -> None:
    assert softmax_floor({}, temperature=0.15, min_prob=0.0) == {}
    assert softmax_floor({"a": 0.1}, temperature=0.15, min_prob=0.5) == {"a": 1.0}

    soft = softmax_floor({"a": 1.0, "b": 0.0}, temperature=0.5, min_prob=0.0)
    assert math.isclose(sum(soft.values()), 1.0)
    assert math.isclose(soft["a"], 1.0 / (1.0 + math.exp(-2.0)))

    floored = softmax_floor({"a": 1.0, "b": 0.0, "c": 0.0}, temperature=0.01, min_prob=0.1)
    assert math.isclose(sum(floored.values()), 1.0)
    assert min(floored.values()) >= 0.1

    assert softmax_floor({"a": 1.0, "b": 0.0, "c": 0.0}, temperature=0.1, min_prob=0.4) == {
        "a": 1 / 3,
        "b": 1 / 3,
        "c": 1 / 3,
    }
    with pytest.raises(ValueError, match="temperature"):
        softmax_floor({"a": 1.0}, temperature=0.0, min_prob=0.0)
    with pytest.raises(ValueError, match="minimum probability"):
        softmax_floor({"a": 1.0}, temperature=0.1, min_prob=1.0)


@pytest.mark.parametrize(
    ("num_arms", "unseen", "requested", "expected"),
    [
        (0, 3, None, "unseen_draw"),
        (0, 3, "pull", "unseen_draw"),
        (0, 0, None, "empty"),
        (2, 3, "draw", "unseen_draw"),
        (2, 0, "draw", "arm_pull"),
        (2, 3, "pull", "arm_pull"),
    ],
)
def test_resolve_action_applies_cold_start_and_empty_pool_rules(num_arms, unseen, requested, expected) -> None:
    assert resolve_action(num_arms=num_arms, unseen_count=unseen, requested=requested) == expected


def test_resolve_action_requires_a_decision_once_arms_exist() -> None:
    with pytest.raises(ValueError, match="pull"):
        resolve_action(num_arms=1, unseen_count=1, requested=None)


def test_replay_folds_patterns_scores_pulls_and_probe_points() -> None:
    events = [
        *registry_events(),
        change(
            3,
            "arm_scores",
            1,
            scores=[
                {
                    "pattern_id": "pattern-old",
                    "severity": 1.0,
                    "fixability": 0.5,
                    "breadth": 0.5,
                    "side_effect": 0.0,
                    "rationale": "first",
                }
            ],
        ),
        change(
            4,
            "arm_scores",
            1,
            scores=[
                {
                    "pattern_id": "pattern-old",
                    "severity": 0.0,
                    "fixability": 0.0,
                    "breadth": 0.0,
                    "side_effect": 0.0,
                    "rationale": "replacement",
                }
            ],
        ),
        event(
            5,
            "BatchSampled",
            {"iteration": 1, "case_ids": ["c1"], "provenance": {"policy": "activesaddler", "chosen_arm": "pattern-old"}},
        ),
        change(6, "arm_decision", 1, action="arm_pull", requested_action="pull"),
    ]

    state = CurriculumState.replay(events)

    assert list(state.patterns) == ["pattern-old", "pattern-new"]
    assert state.patterns["pattern-old"].case_ids == ("c1", "c2")
    assert len(state.patterns["pattern-old"].tags) == 3
    assert state.executed_case_ids == {"c1", "c2"}
    assert state.arms(("c3", "c2", "c1")) == {"pattern-old": ("c2", "c1"), "pattern-new": ("c2",)}
    assert state.unseen_case_ids(("c3", "c1", "c4")) == ("c3", "c4")
    assert state.arm_score("pattern-old", 1) == 0.25
    assert state.arm_score("pattern-old", 2) == 0.0
    assert state.arm_score("pattern-new", 1) == 0.0
    assert [(pull.iteration, pull.pattern_id, pull.case_ids) for pull in state.pulls] == [(1, "pattern-old", ("c1",))]
    assert state.decisions[1]["action"] == "arm_pull"


def test_replay_rejects_tags_for_unknown_patterns() -> None:
    with pytest.raises(ValueError, match="unknown pattern"):
        CurriculumState.replay([change(1, "patterns_extracted", 0, new_patterns=[], tags=[tag("missing", "c1")])])


def test_observations_skip_arms_created_only_by_post_patch_analysis() -> None:
    state = CurriculumState.replay(registry_events())

    observations = pattern_observations(state, iteration=0, batch_case_ids=("c1", "c2"), after_evaluation_id=AFTER)

    assert observations == [
        {
            "pattern_id": "pattern-old",
            "active": 0.5,
            "evaluated_case_ids": ["c1", "c2"],
            "tagged_case_ids": ["c2"],
        }
    ]


def test_all_pass_observations_mark_every_overlapping_arm_inactive() -> None:
    state = CurriculumState.replay(registry_events())

    observations = pattern_observations(state, iteration=1, batch_case_ids=("c2",), after_evaluation_id=None)

    assert [(item["pattern_id"], item["active"], item["tagged_case_ids"]) for item in observations] == [
        ("pattern-old", 0.0, []),
        ("pattern-new", 0.0, []),
    ]


def test_activity_ema_is_seeded_and_ordered_by_iteration() -> None:
    observations = [
        PatternObservation(iteration=2, active=1.0, evaluated_case_ids=("c1",), tagged_case_ids=("c1",)),
        PatternObservation(iteration=1, active=0.0, evaluated_case_ids=("c1",), tagged_case_ids=()),
    ]

    assert activity_ema((), 0.9) == 1.0
    assert math.isclose(activity_ema(observations, 0.5), 0.75)
    with pytest.raises(ValueError):
        activity_ema(observations, 0.0)


def test_arm_score_combines_four_axes_and_validates_range() -> None:
    score = ArmScore(iteration=0, severity=1.0, fixability=0.5, breadth=0.5, side_effect=1.0, rationale="r")

    assert score.value == 0.5
    with pytest.raises(ValueError, match="severity"):
        ArmScore(iteration=0, severity=1.5, fixability=0.0, breadth=0.0, side_effect=0.0, rationale="r")


def test_pattern_ids_and_failures_are_deterministic() -> None:
    first = derive_pattern_id(iteration=3, key="k", label="Label")

    assert first == derive_pattern_id(iteration=3, key="k", label="Label")
    assert first != derive_pattern_id(iteration=4, key="k", label="Label")
    assert first.startswith("pattern-")
    assert failing_case_ids({"a": 0.0, "b": 0.5, "c": 0.49}) == ("a", "c")


def policy(**overrides) -> ActiveSaddlerTaskSelectionPolicy:
    values = {
        "batch_size": 2,
        "seed": 7,
        "softmax_temperature": 0.15,
        "min_prob": 0.0,
        "ema_eta": 0.9,
        "pattern_extraction_timeout_seconds": 30.0,
        "arm_scoring_timeout_seconds": 20.0,
    }
    values.update(overrides)
    return ActiveSaddlerTaskSelectionPolicy(**values)


CASES = tuple(Case(case_id=f"c{index}", split="train", payload={}) for index in range(1, 6))


def test_activesaddler_draws_unseen_cases_in_fixed_seeded_order() -> None:
    selector = policy()
    order = selector.draw_order(CASES)
    state = CurriculumState.replay(registry_events())

    selection = selector.select_curriculum(CASES, 1, state=state, action="unseen_draw")

    unseen = [case_id for case_id in order if case_id not in {"c1", "c2"}]
    assert selection.case_ids == tuple(unseen[:2])
    assert selection.provenance["action"] == "unseen_draw"
    assert selection.provenance["chosen_arm"] is None
    assert selection.provenance["unseen_case_ids"] == unseen
    assert selector.draw_order(CASES) == order


def test_activesaddler_pull_is_replayable_and_reports_arm_snapshot() -> None:
    selector = policy(batch_size=1)
    state = CurriculumState.replay(registry_events())

    first = selector.select_curriculum(CASES, 4, state=state, action="arm_pull")
    second = selector.select_curriculum(CASES, 4, state=state, action="arm_pull")

    assert first == second
    chosen = first.provenance["chosen_arm"]
    assert chosen in {"pattern-old", "pattern-new"}
    assert set(first.case_ids) <= set(state.patterns[str(chosen)].case_ids)
    arms = {item["pattern_id"]: item for item in first.provenance["arms"]}
    assert set(arms) == {"pattern-old", "pattern-new"}
    assert math.isclose(sum(float(item["prob"]) for item in arms.values()), 1.0)
    assert arms[str(chosen)]["selected"] is True


def test_activesaddler_rejects_infeasible_actions_and_settings() -> None:
    selector = policy()
    empty = CurriculumState()

    with pytest.raises(ValueError, match="instantiated arm"):
        selector.select_curriculum(CASES, 0, state=empty, action="arm_pull")
    with pytest.raises(ValueError, match="empty"):
        selector.select_curriculum(CASES, 0, state=empty, action="empty")
    with pytest.raises(ValueError, match="softmax_temperature"):
        policy(softmax_temperature=0.0)
    with pytest.raises(ValueError, match="min_prob"):
        policy(min_prob=1.0)
    with pytest.raises(ValueError, match="ema_eta"):
        policy(ema_eta=0.0)
    with pytest.raises(ValueError, match="timeouts"):
        policy(arm_scoring_timeout_seconds=0.0)


def test_activesaddler_implements_the_adaptive_interface() -> None:
    selector = policy()

    assert isinstance(selector, AdaptiveTaskSelectionPolicy)
    assert selector.namespace == CURRICULUM_NAMESPACE
    assert selector.session_timeouts() == {"extract_patterns": 30.0, "decide_arm": 20.0, "score_arms": 20.0}


def test_passive_policies_are_not_adaptive() -> None:
    from autosaddler.v2.core.policies import EpochShuffledTaskSelectionPolicy, FixedTaskSelectionPolicy

    assert not isinstance(FixedTaskSelectionPolicy(batch_size=1), AdaptiveTaskSelectionPolicy)
    assert not isinstance(EpochShuffledTaskSelectionPolicy(batch_size=1, seed=0), AdaptiveTaskSelectionPolicy)


class EventLog:
    """Synthetic event log for draw-epoch tests."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def batch(self, iteration: int, case_ids: list[str], *, arm: str | None = None) -> "EventLog":
        provenance = {"policy": "activesaddler", "action": "arm_pull" if arm else "unseen_draw", "chosen_arm": arm}
        self.events.append(
            event(len(self.events) + 1, "BatchSampled", {"iteration": iteration, "case_ids": case_ids, "provenance": provenance})
        )
        return self

    def pattern(self, iteration: int, pattern_id: str, *case_ids: str) -> "EventLog":
        self.events.append(
            change(
                len(self.events) + 1,
                "patterns_extracted",
                iteration,
                new_patterns=[{"pattern_id": pattern_id, "key": pattern_id, "label": pattern_id}],
                tags=[tag(pattern_id, case_id) for case_id in case_ids],
            )
        )
        return self

    def state_change(self, payload: dict) -> "EventLog":
        self.events.append(
            event(len(self.events) + 1, "ExtensionStateChanged", {"namespace": CURRICULUM_NAMESPACE, **payload})
        )
        return self


def _request(iteration: int):
    from autosaddler.v2.core.scheduling import SelectionRequest

    return SelectionRequest(
        iteration=iteration,
        train_cases=CASES,
        working_parent_id=PARENT,
        selected_parent_id=PARENT,
        selection_parent_ids=(PARENT,),
        component_sources={},
        selection_rationale="keep the base",
    )


def _epoch_step(log: EventLog, iteration: int):
    from autosaddler.v2.core.scheduling import StateStep

    step = policy().next_selection_step(log.events, _request(iteration))
    return step if isinstance(step, StateStep) and step.name == "draw-epoch" else None


def test_draw_epoch_stays_closed_while_the_draw_pool_has_cases() -> None:
    log = EventLog().batch(0, ["c1", "c2"]).pattern(0, "pattern-a", "c1").batch(1, ["c1"], arm="pattern-a")

    assert _epoch_step(log, 2) is None
    assert CurriculumState.replay(log.events).draw_epoch == 0


def test_draw_epoch_waits_until_every_arm_is_pulled_after_the_pool_empties() -> None:
    log = (
        EventLog()
        .batch(0, ["c1", "c2"])
        .pattern(0, "pattern-a", "c1")
        .batch(1, ["c1"], arm="pattern-a")
        .batch(2, ["c3", "c4", "c5"])
        .pattern(2, "pattern-b", "c3")
    )
    # pattern-a was pulled only before the pool emptied at iteration 2; pattern-b never.
    assert _epoch_step(log, 3) is None
    log.batch(3, ["c1"], arm="pattern-a")
    assert _epoch_step(log, 4) is None
    log.batch(4, ["c3"], arm="pattern-b")

    step = _epoch_step(log, 5)

    assert step is not None
    payload = step.payload
    assert payload["change"] == "draw_epoch_opened" and payload["epoch"] == 1 and payload["iteration"] == 5
    assert payload["previous_epoch_exhausted_iteration"] == 2
    expected = [case_id for case_id in policy().draw_order(CASES, 1) if case_id in {"c2", "c4", "c5"}]
    assert payload["case_ids"] == expected


def test_new_draw_epoch_takes_arm_free_cases_in_order_and_records_the_epoch() -> None:
    from autosaddler.v2.core.policies import TaskSelection
    from autosaddler.v2.core.scheduling import SessionStep
    from autosaddler.v2.prompting.models import Cost, SessionResult

    log = EventLog().batch(0, ["c1", "c2", "c3", "c4", "c5"]).pattern(0, "pattern-a", "c1").batch(1, ["c1"], arm="pattern-a")
    step = _epoch_step(log, 2)
    assert step is not None
    log.state_change(dict(step.payload))
    pool = step.payload["case_ids"]

    decide = policy().next_selection_step(log.events, _request(2))
    assert isinstance(decide, SessionStep) and decide.kind == "decide_arm"
    assert decide.context["task_selection"]["num_unseen"] == len(pool) == 4
    assert decide.context["task_selection"]["draw_epoch"] == 1
    draw = SessionResult(
        status="completed",
        structured_output={"action": "draw", "rationale": "Re-explore prior successes."},
        raw_response="{}",
        tool_calls=(),
        usage=(),
        cost=Cost(sessions=1),
    )
    assert decide.validate(draw) is None
    log.state_change(dict(decide.record(draw)))

    selection = policy().next_selection_step(log.events, _request(2))
    assert isinstance(selection, TaskSelection)
    assert list(selection.case_ids) == pool[:2]
    assert selection.provenance["draw_epoch"] == 1
    assert selection.provenance["unseen_case_ids"] == pool


def test_draw_epochs_repeat_and_skip_when_every_executed_case_owns_an_arm() -> None:
    log = EventLog().batch(0, ["c1", "c2", "c3", "c4", "c5"]).pattern(0, "pattern-a", "c1").batch(1, ["c1"], arm="pattern-a")
    first = _epoch_step(log, 2)
    assert first is not None
    log.state_change(dict(first.payload)).batch(2, list(first.payload["case_ids"]))
    log.batch(3, ["c1"], arm="pattern-a")

    second = _epoch_step(log, 4)

    assert second is not None and second.payload["epoch"] == 2
    assert second.payload["previous_epoch_exhausted_iteration"] == 2
    assert second.payload["case_ids"] == [case_id for case_id in policy().draw_order(CASES, 2) if case_id != "c1"]

    owned = EventLog().batch(0, ["c1", "c2", "c3", "c4", "c5"]).pattern(0, "pattern-a", "c1", "c2", "c3", "c4", "c5")
    owned.batch(1, ["c1", "c2"], arm="pattern-a")
    assert _epoch_step(owned, 2) is None


def test_draw_epoch_opens_instead_of_ending_when_no_arm_exists() -> None:
    log = EventLog().batch(0, ["c1", "c2", "c3", "c4", "c5"])

    step = _epoch_step(log, 1)

    assert step is not None and sorted(step.payload["case_ids"]) == ["c1", "c2", "c3", "c4", "c5"]


def test_draw_epoch_does_not_open_after_the_iteration_decision() -> None:
    log = EventLog().batch(0, ["c1", "c2", "c3", "c4", "c5"]).pattern(0, "pattern-a", "c1").batch(1, ["c1"], arm="pattern-a")
    log.events.append(change(len(log.events) + 1, "arm_decision", 2, action="arm_pull", requested_action="pull"))

    assert _epoch_step(log, 2) is None


def test_draw_epochs_must_open_in_order() -> None:
    log = EventLog().batch(0, ["c1"]).state_change(
        {"schema_version": CURRICULUM_SCHEMA_VERSION, "change": "draw_epoch_opened", "iteration": 1, "epoch": 2, "case_ids": ["c1"]}
    )

    with pytest.raises(ValueError, match="must open in order"):
        CurriculumState.replay(log.events)
