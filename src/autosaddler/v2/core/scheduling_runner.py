"""Durable execution of adaptive task-selection policies on behalf of the engine.

The engine calls this runner at a few fixed points of an iteration when the configured
task-selection policy is adaptive (see ``autosaddler.v2.core.scheduling``). The runner
executes the steps the policy describes, records every result as an
``ExtensionStateChanged`` event in the policy's namespace, and runs the deferred
sessions the policy requests after reflection. It has no knowledge of any concrete
policy.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Candidate,
    Evaluation,
    JsonValue,
    canonical_json,
    sha256_digest,
)
from autosaddler.v2.core.events import RunEvent, operation_id
from autosaddler.v2.core.policies import TaskSelection
from autosaddler.v2.core.ports import ScenarioComponents
from autosaddler.v2.core.run_state import RunState
from autosaddler.v2.core.scheduling import (
    MAX_SELECTION_STEPS,
    PROMPT_OVERLAY_KEY,
    TASK_SELECTION_CONTEXT_KEY,
    AdaptiveTaskSelectionPolicy,
    DeferredRequest,
    IterationFeedback,
    NoSelection,
    SelectionRequest,
    SessionStep,
    StatePayload,
    StateStep,
    iteration_feedback_from,
)
from autosaddler.v2.core.serde import record
from autosaddler.v2.prompting.models import SessionResult
from autosaddler.v2.storage.local import LocalRunStore

RunSession = Callable[..., Awaitable[SessionResult]]


class AdaptiveTaskSelectionRunner:
    def __init__(
        self,
        *,
        policy: AdaptiveTaskSelectionPolicy,
        store: LocalRunStore,
        scenario: ScenarioComponents,
        run_session: RunSession,
        exhausted_error: type[Exception],
    ) -> None:
        self.policy = policy
        self.store = store
        self.scenario = scenario
        self.run_session = run_session
        self.exhausted_error = exhausted_error

    def handles(self, kind: str) -> bool:
        return kind in self.policy.required_session_kinds

    def session_timeout(self, kind: str) -> float | None:
        return self.policy.session_timeouts().get(kind)

    def prompt_context(self, iteration: int) -> dict[str, JsonValue]:
        return {TASK_SELECTION_CONTEXT_KEY: self._overlaid(self.policy.prompt_context(self.store.events(), iteration))}

    async def select_batch(self, request: SelectionRequest) -> tuple[str, ...] | NoSelection:
        iteration = request.iteration
        batch_operation = operation_id(self.store.run_id, "iteration", iteration, "batch")
        executed: set[str] = set()
        for _ in range(MAX_SELECTION_STEPS):
            batch_event = self.store.find("BatchSampled", batch_operation)
            if batch_event is not None:
                return _case_ids(batch_event.payload.get("case_ids"))
            step = self.policy.next_selection_step(self.store.events(), request)
            if isinstance(step, TaskSelection):
                self.store.append(
                    "BatchSampled",
                    batch_operation,
                    {"iteration": iteration, "case_ids": list(step.case_ids), "provenance": step.provenance},
                )
                continue
            if isinstance(step, NoSelection):
                return step
            state_operation = operation_id(self.store.run_id, "iteration", iteration, "selection", step.name)
            if step.name in executed or self.store.find("ExtensionStateChanged", state_operation) is not None:
                raise RuntimeError(f"Adaptive task selection repeated step {step.name!r} in iteration {iteration}")
            executed.add(step.name)
            if isinstance(step, StateStep):
                state = step.payload
            else:
                state = await self._session_state(step, iteration, request.working_parent_id)
            self.store.append("ExtensionStateChanged", state_operation, self._namespaced(state))
        raise RuntimeError(f"Adaptive task selection exceeded {MAX_SELECTION_STEPS} steps in iteration {iteration}")

    def record_no_training_failures(
        self,
        *,
        iteration: int,
        case_ids: Sequence[str],
        parent: Candidate,
        parent_evaluation: Evaluation,
    ) -> None:
        feedback = IterationFeedback(
            iteration=iteration,
            outcome="no_training_failures",
            case_ids=tuple(case_ids),
            working_parent_id=parent.candidate_id,
            child_id=None,
            train_before_evaluation_id=parent_evaluation.evaluation_id,
            train_after_evaluation_id=None,
            train_before_case_scores=_case_scores(parent_evaluation),
            train_after_case_scores=None,
            train_before_evidence=None,
            train_after_evidence=None,
            diagnosis=None,
        )
        parts = ("iteration", iteration, "feedback")
        self._record_changes(parts, self.policy.iteration_changes(self._events_without_state(parts), feedback))

    def reflection_payload(
        self,
        *,
        iteration: int,
        accepted: bool,
        case_ids: Sequence[str],
        parent: Candidate,
        parent_evaluation: Evaluation,
        child: Candidate,
        child_evaluation: Evaluation,
        evidence: ArtifactRef,
        diagnosis: str,
        patch_intent: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        feedback = IterationFeedback(
            iteration=iteration,
            outcome="accepted" if accepted else "declined",
            case_ids=tuple(case_ids),
            working_parent_id=parent.candidate_id,
            child_id=child.candidate_id,
            train_before_evaluation_id=parent_evaluation.evaluation_id,
            train_after_evaluation_id=child_evaluation.evaluation_id,
            train_before_case_scores=_case_scores(parent_evaluation),
            train_after_case_scores=_case_scores(child_evaluation),
            train_before_evidence=evidence,
            train_after_evidence=self.scenario.evidence_builder.build(child_evaluation),
            diagnosis=diagnosis,
            patch_intent=(
                {key: value for key, value in patch_intent.items() if key != "schema_version"}
                if patch_intent is not None
                else None
            ),
        )
        return {"task_selection_feedback": record(feedback)}

    def after_reflection(
        self,
        reflection_obligation_id: str,
        reflection: Mapping[str, object],
        lessons: Sequence[JsonValue],
    ) -> None:
        feedback = reflection.get("task_selection_feedback")
        if feedback is None:
            return
        request = self.policy.after_reflection(self.store.events(), iteration_feedback_from(feedback), lessons)
        if request is None:
            return
        owning_iteration = cast(int, reflection.get("owning_iteration"))
        candidate_id = str(reflection.get("candidate_id", ""))
        obligation_id = sha256_digest(
            canonical_json({"iteration": owning_iteration, "candidate_id": candidate_id, "kind": request.kind})
        )
        self.store.append(
            "DeferredWorkScheduled",
            operation_id(self.store.run_id, "deferred", reflection_obligation_id, "schedule", request.kind),
            {
                "obligation_id": obligation_id,
                "owning_iteration": owning_iteration,
                "candidate_id": candidate_id,
                "session_kind": request.kind,
                "stage": request.stage,
                "task_selection": cast(JsonValue, request.payload),
            },
        )

    async def drain_deferred(self) -> None:
        while pending := sorted(
            (obligation_id, obligation)
            for obligation_id, obligation in RunState.replay(self.store.events()).pending_obligations.items()
            if self.handles(str(obligation.get("session_kind", "")))
        ):
            await self._run_deferred(*pending[0])

    async def _run_deferred(self, obligation_id: str, obligation: Mapping[str, object]) -> None:
        payload = obligation.get("task_selection")
        if not isinstance(payload, Mapping):
            raise TypeError("Task-selection deferred work requires a payload object")
        request = DeferredRequest(
            kind=str(obligation.get("session_kind", "")),
            stage=str(obligation.get("stage", "")),
            payload=cast(Mapping[str, JsonValue], payload),
        )
        candidate_id = str(obligation.get("candidate_id", ""))
        parts = ("deferred", obligation_id, "state")
        # Exclude this obligation's own state so a partially recorded result replays identically.
        events = self._events_without_state(parts)
        spec = self.scenario.prompt_pack.session(
            request.kind, self._with_overlay(self.policy.deferred_context(events, request))
        )
        try:
            result = await self.run_session(
                logical_operation_id=operation_id(self.store.run_id, "deferred", obligation_id, request.kind),
                spec=spec,
                source_workspace=None,
                result_validator=lambda value: self.policy.deferred_failure_reason(events, request, value),
                metrics_context={
                    "stage": request.stage,
                    "iteration": cast(int, obligation.get("owning_iteration")),
                    "candidate_id": candidate_id,
                },
            )
        except self.exhausted_error as error:
            self._record_changes(parts, self.policy.deferred_exhausted_changes(events, request, str(error)))
            self.store.append(
                "DeferredWorkAbandoned",
                operation_id(self.store.run_id, "deferred", obligation_id, "abandon"),
                {"obligation_id": obligation_id, "candidate_id": candidate_id, "reason": str(error)},
            )
            return
        self._record_changes(parts, self.policy.deferred_changes(events, request, result))
        self.store.append(
            "DeferredWorkCompleted",
            operation_id(self.store.run_id, "deferred", obligation_id, "complete"),
            {"obligation_id": obligation_id, "candidate_id": candidate_id},
        )

    async def _session_state(self, step: SessionStep, iteration: int, candidate_id: str) -> StatePayload:
        try:
            result = await self.run_session(
                logical_operation_id=operation_id(self.store.run_id, "iteration", iteration, step.name),
                spec=self.scenario.prompt_pack.session(step.kind, self._with_overlay(step.context)),
                source_workspace=None,
                result_validator=step.validate,
                metrics_context={"stage": step.stage, "iteration": iteration, "candidate_id": candidate_id},
            )
        except self.exhausted_error as error:
            if step.record_exhausted is None:
                raise
            return step.record_exhausted(str(error))
        return step.record(result)

    def _with_overlay(self, context: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        selection = context.get(TASK_SELECTION_CONTEXT_KEY, {})
        if not isinstance(selection, Mapping):
            raise TypeError(f"Session context {TASK_SELECTION_CONTEXT_KEY!r} must be an object")
        return {**context, TASK_SELECTION_CONTEXT_KEY: self._overlaid(selection)}

    def _overlaid(self, selection: Mapping[str, JsonValue]) -> JsonValue:
        if PROMPT_OVERLAY_KEY in selection:
            raise ValueError(f"Task-selection prompt context must not set its own {PROMPT_OVERLAY_KEY!r}")
        return {**selection, PROMPT_OVERLAY_KEY: self.policy.prompt_overlay}

    def _record_changes(self, parts: tuple[object, ...], changes: Sequence[StatePayload]) -> None:
        for index, change in enumerate(changes):
            state_operation = operation_id(self.store.run_id, *parts, index)
            if self.store.find("ExtensionStateChanged", state_operation) is None:
                self.store.append("ExtensionStateChanged", state_operation, self._namespaced(change))

    def _events_without_state(self, parts: tuple[object, ...]) -> tuple[RunEvent, ...]:
        prefix = operation_id(self.store.run_id, *parts) + ":"
        return tuple(
            event
            for event in self.store.events()
            if not (event.event_type == "ExtensionStateChanged" and event.operation_id.startswith(prefix))
        )

    def _namespaced(self, state: StatePayload) -> dict[str, Any]:
        if "namespace" in state:
            raise ValueError("Task-selection state payloads must not set their own namespace")
        return {"namespace": self.policy.namespace, **state}


def _case_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise TypeError("sampled case_ids must be a non-empty list of strings")
    return tuple(value)


def _case_scores(evaluation: Evaluation) -> dict[str, float]:
    scores: dict[str, float] = {}
    for case_id in evaluation.requested_case_ids:
        values = [
            observation.score
            for observation in evaluation.observations
            if observation.case_id == case_id and observation.is_valid and observation.score is not None
        ]
        if not values:
            raise ValueError(f"Evaluation has no valid scores for case {case_id}")
        scores[case_id] = sum(values) / len(values)
    return scores
