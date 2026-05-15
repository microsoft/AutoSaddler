from __future__ import annotations

import asyncio
import shutil
import time
from uuid import uuid4
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from autosaddler.v2.core.attempts import EventEvaluationAttemptSink
from autosaddler.v2.core.domain import (
    ArtifactRef,
    Candidate,
    Case,
    Evaluation,
    JsonValue,
    PatchVerdict,
    canonical_json,
    sha256_digest,
)
from autosaddler.v2.core.events import ITERATION_COMPLETION_SCHEMA_VERSION, operation_id
from autosaddler.v2.core.metrics import EventModelUsageSink
from autosaddler.v2.core.policies import DevelopmentDecision, PolicyBundle
from autosaddler.v2.core.ports import (
    AgentProvider,
    CompositionPlan,
    EvaluationContext,
    MutationContext,
    MutationOutcome,
    MutationSession,
    ScenarioComponents,
    WorkspaceDelta,
)
from autosaddler.v2.core.run_state import RunState
from autosaddler.v2.core.serde import candidate_from, evaluation_from, record, session_result_from
from autosaddler.v2.prompting.models import (
    SessionRequest,
    SessionResult,
    SessionSpec,
    Usage,
    session_output_validation_error,
)
from autosaddler.v2.storage.local import LocalRunStore


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    run_id: str
    selected_candidate_id: str
    development_score: float
    iterations: int


class SessionRetriesExhausted(RuntimeError):
    pass


class AutoSaddlerEngine:
    def __init__(
        self,
        *,
        store: LocalRunStore,
        scenario: ScenarioComponents,
        provider: AgentProvider,
        policies: PolicyBundle,
        diagnosis_patch_timeout_seconds: float = 30.0,
        selection_timeout_seconds: float | None = None,
        reflection_timeout_seconds: float | None = None,
        session_retries: int = 2,
        session_retry_backoff_seconds: float = 0.0,
    ) -> None:
        if session_retries < 0:
            raise ValueError("Session retries cannot be negative")
        if session_retry_backoff_seconds < 0:
            raise ValueError("Session retry backoff cannot be negative")
        self.store = store
        self.scenario = scenario
        self.provider = provider
        self.policies = policies
        self.diagnosis_patch_timeout_seconds = diagnosis_patch_timeout_seconds
        self.selection_timeout_seconds = selection_timeout_seconds or diagnosis_patch_timeout_seconds
        self.reflection_timeout_seconds = reflection_timeout_seconds or diagnosis_patch_timeout_seconds
        self.session_retries = session_retries
        self.session_retry_backoff_seconds = session_retry_backoff_seconds
        self.run_invocation_id = uuid4().hex
        self._iteration_started_at: dict[int, float] = {}

    def run(self) -> OptimizationResult:
        return asyncio.run(self.optimize())

    async def optimize(self) -> OptimizationResult:
        self._run_invocation_started_at = time.monotonic()
        try:
            return await self._optimize()
        except asyncio.CancelledError as error:
            self.store.append(
                "RunInterrupted",
                operation_id(self.store.run_id, "run", self.run_invocation_id, "interrupted"),
                {
                    "run_invocation_id": self.run_invocation_id,
                    "wall_seconds": self._run_wall_seconds(),
                    "error_type": type(error).__name__,
                },
            )
            raise
        except Exception as error:
            if self.store.transition_hook_error is error:
                raise
            self.store.append(
                "RunFailed",
                operation_id(self.store.run_id, "run", self.run_invocation_id, "failed"),
                {
                    "run_invocation_id": self.run_invocation_id,
                    "wall_seconds": self._run_wall_seconds(),
                    "error_type": type(error).__name__,
                    "error": str(error)[:2_000],
                },
            )
            raise

    async def _optimize(self) -> OptimizationResult:
        state = self._state()
        if state.run_completed:
            return self._load_result()
        if state.candidates:
            self.store.validate_integrity()
            prior_sequence = self.store.events()[-1].sequence
            self.store.append(
                "RunResumed",
                operation_id(self.store.run_id, "resume", prior_sequence + 1),
                {"prior_sequence": prior_sequence, "run_invocation_id": self.run_invocation_id},
            )
            state = self._state()
        else:
            seed = self.scenario.harness_space.seed()
            self.store.append(
                "RunStarted",
                operation_id(self.store.run_id, "run", "start"),
                {
                    "scenario": {"name": self.scenario.name, "version": self.scenario.version},
                    "seed_candidate": record(seed),
                    "run_invocation_id": self.run_invocation_id,
                },
            )
            state = self._state()

        seed_id = state.accepted_candidate_ids[0]
        seed = state.candidates[seed_id]
        await self._ensure_evaluation(
            logical_operation_id=operation_id(self.store.run_id, "seed", "development"),
            candidate=seed,
            cases=self.scenario.development_cases,
            purpose="development",
            iteration=0,
        )

        for iteration in range(self.policies.budget.max_iterations):
            if iteration > 0:
                await self._drain_deferred()
            state = self._state()
            if not self.policies.budget.allows_iteration(
                iteration=iteration,
                attempted_rollouts=state.attempted_rollouts,
            ):
                break
            await self._run_iteration(iteration)

        await self._drain_deferred()
        state = self._state()
        if state.pending_obligations:
            raise RuntimeError("Optimization cannot complete with pending deferred work")
        finalization_started = time.monotonic()
        selected, development = self._select_best()
        best_operation = operation_id(self.store.run_id, "best-candidate")
        self.store.append(
            "BestCandidateSelected",
            best_operation,
            {
                "candidate_id": selected.candidate_id,
                "evaluation_id": development.evaluation_id,
                "development_aggregate": development.aggregate_score,
            },
        )
        result = OptimizationResult(
            run_id=self.store.run_id,
            selected_candidate_id=selected.candidate_id,
            development_score=_required_score(development),
            iterations=len(self.store.events_of_type("IterationStarted")),
        )
        self.store.write_json("result.json", record(result), kind="optimization-result")
        optimization_operation = operation_id(self.store.run_id, "optimization", "complete")
        if self.store.find("OptimizationCompleted", optimization_operation) is None:
            self.store.append(
                "OptimizationCompleted",
                optimization_operation,
                {
                    "candidate_id": selected.candidate_id,
                    "wall_seconds": time.monotonic() - finalization_started,
                },
            )
        self.store.append(
            "RunCompleted",
            operation_id(self.store.run_id, "run", "complete"),
            {
                **record(result),
                "run_invocation_id": self.run_invocation_id,
                "wall_seconds": self._run_wall_seconds(),
            },
        )
        return result

    async def _run_iteration(self, iteration: int) -> None:
        iteration_operation = operation_id(self.store.run_id, "iteration", iteration)
        if self.store.find("IterationCompleted", iteration_operation) is not None:
            return
        self._iteration_started_at[iteration] = time.monotonic()
        self.store.append("IterationStarted", iteration_operation, {"iteration": iteration})

        batch_operation = operation_id(self.store.run_id, "iteration", iteration, "batch")
        batch_event = self.store.find("BatchSampled", batch_operation)
        if batch_event is None:
            selection = self.policies.task_selection.select(self.scenario.train_cases, iteration)
            batch_event = self.store.append(
                "BatchSampled",
                batch_operation,
                {
                    "iteration": iteration,
                    "case_ids": list(selection.case_ids),
                    "provenance": selection.provenance,
                },
            )
        case_ids = _string_list(batch_event.payload.get("case_ids"), "sampled case_ids")
        cases_by_id = {case.case_id: case for case in self.scenario.train_cases}
        cases = tuple(cases_by_id[case_id] for case_id in case_ids)

        state = self._state()
        component_source_options: dict[str, list[str]] = {}
        for candidate_id in state.accepted_candidate_ids:
            change = state.candidates[candidate_id].change
            if change is None:
                continue
            for unit in change.changed_units:
                component_source_options.setdefault(unit, []).append(candidate_id)
        evolve_spec = self.scenario.prompt_pack.session(
            "evolve",
            {
                "iteration": iteration,
                "candidate_ids": list(state.accepted_candidate_ids),
                "component_source_options": component_source_options,
                "train_case_ids": list(case_ids),
            },
        )
        evolve = await self._ensure_session(
            logical_operation_id=operation_id(self.store.run_id, "iteration", iteration, "evolve"),
            spec=evolve_spec,
            source_workspace=None,
            result_validator=lambda result: _selection_failure_reason(result, state.accepted_candidate_ids),
            metrics_context={"stage": "proposal.selection", "iteration": iteration},
        )
        selection_plan = _selection_plan(evolve, state.accepted_candidate_ids)
        selected_parent = state.candidates[selection_plan.parents[0]]
        if selection_plan.selections:
            composition_operation = operation_id(self.store.run_id, "iteration", iteration, "composition")
            composition_event = self.store.find("CandidateFinalized", composition_operation)
            if composition_event is None:
                composed = self.scenario.harness_space.compose(selection_plan)
                parent = state.candidates.get(composed.candidate_id, composed)
                self.store.append(
                    "CandidateFinalized",
                    composition_operation,
                    {"iteration": iteration, "kind": "composition", "candidate": record(parent)},
                )
            else:
                parent = candidate_from(composition_event.payload.get("candidate"))
        else:
            parent = selected_parent
        parent_evaluation = await self._ensure_evaluation(
            logical_operation_id=operation_id(self.store.run_id, "iteration", iteration, "train-before"),
            candidate=parent,
            cases=cases,
            purpose="train_before",
            iteration=iteration,
        )
        if all(observation.disposition == "success" for observation in parent_evaluation.observations):
            self._complete_iteration(
                iteration=iteration,
                resulting_candidate=selected_parent,
                evaluated_candidates=(parent,),
                accepted=False,
                outcome="no_training_failures",
                details={
                    "selection_parent_ids": list(selection_plan.parents),
                    "component_sources": dict(selection_plan.selections),
                },
            )
            return
        evidence = self.scenario.evidence_builder.build(parent_evaluation)

        diagnosis_spec = self.scenario.prompt_pack.session(
            "diagnose_patch",
            {
                "iteration": iteration,
                "candidate_ids": [parent.candidate_id],
                "train_case_ids": list(case_ids),
                "evidence": record(evidence),
            },
        )
        mutation_context = MutationContext(
            iteration=iteration,
            patch_label=diagnosis_spec.mutation_label or "diagnosis-patch",
            evidence=evidence,
            workspace_root=self.store.run_dir / "workspaces" / f"iteration-{iteration:04d}-mutation",
        )
        mutation_session = self.scenario.harness_space.begin_mutation(parent, mutation_context)
        candidate_operation = operation_id(self.store.run_id, "iteration", iteration, "candidate")
        diagnosis_operation = operation_id(self.store.run_id, "iteration", iteration, "diagnose-patch")
        try:
            diagnosis = await self._ensure_session(
                logical_operation_id=diagnosis_operation,
                spec=diagnosis_spec,
                source_workspace=mutation_session.workspace,
                mutation_session=mutation_session,
                metrics_context={
                    "stage": "proposal.patch",
                    "iteration": iteration,
                    "candidate_id": parent.candidate_id,
                },
            )
        except SessionRetriesExhausted as error:
            reason = f"diagnosis session failed after retries: {error}"
            self._reject_mutation(
                candidate_operation=candidate_operation,
                iteration=iteration,
                parent_ids=selection_plan.parents,
                case_ids=case_ids,
                diagnosis_operation=operation_id(self.store.run_id, "iteration", iteration, "diagnose-patch"),
                reason=reason,
            )
            self._complete_rejected_iteration(
                iteration=iteration,
                resulting_parent=selected_parent,
                working_parent=parent,
                case_ids=case_ids,
                reason=reason,
            )
            return

        rejected = self.store.find("MutationRejected", candidate_operation)
        if rejected is not None:
            self._complete_rejected_iteration(
                iteration=iteration,
                resulting_parent=selected_parent,
                working_parent=parent,
                case_ids=case_ids,
                reason=str(rejected.payload.get("verification_failure", "mutation rejected")),
            )
            return
        finalized = self.store.find("CandidateFinalized", candidate_operation)
        if finalized is None:
            try:
                outcome = self._mutation_outcome(
                    logical_operation_id=diagnosis_operation,
                    result=diagnosis,
                )
                self.scenario.harness_space.apply_mutation(mutation_session, outcome)
                child = self.scenario.harness_space.finalize(mutation_session)
            except (FileNotFoundError, TypeError, ValueError) as error:
                reason = f"{type(error).__name__}: {error}"
                self._reject_mutation(
                    candidate_operation=candidate_operation,
                    iteration=iteration,
                    parent_ids=selection_plan.parents,
                    case_ids=case_ids,
                    diagnosis_operation=operation_id(self.store.run_id, "iteration", iteration, "diagnose-patch"),
                    reason=reason,
                )
                self._complete_rejected_iteration(
                    iteration=iteration,
                    resulting_parent=selected_parent,
                    working_parent=parent,
                    case_ids=case_ids,
                    reason=reason,
                )
                return
            self.store.append(
                "CandidateFinalized",
                candidate_operation,
                {"iteration": iteration, "candidate": record(child)},
            )
        else:
            child = candidate_from(finalized.payload.get("candidate"))

        child_evaluation = await self._ensure_evaluation(
            logical_operation_id=operation_id(self.store.run_id, "iteration", iteration, "train-after"),
            candidate=child,
            cases=cases,
            purpose="train_after",
            iteration=iteration,
        )
        acceptance_operation = operation_id(self.store.run_id, "iteration", iteration, "acceptance")
        acceptance_event = self.store.find("AcceptanceDecided", acceptance_operation)
        if acceptance_event is None:
            verdict = self.policies.acceptance.compare(parent_evaluation, child_evaluation)
            acceptance_event = self.store.append(
                "AcceptanceDecided",
                acceptance_operation,
                {
                    "iteration": iteration,
                    "parent_candidate_id": parent.candidate_id,
                    "parent_ids": list(selection_plan.parents),
                    "child_candidate_id": child.candidate_id,
                    "verdict": record(verdict),
                },
            )
        verdict = _verdict(acceptance_event.payload.get("verdict"))

        development_operation = operation_id(self.store.run_id, "iteration", iteration, "development-gate")
        development_event = self.store.find("DevelopmentGateDecided", development_operation)
        if development_event is None:
            decision = self.policies.development.decide(verdict)
            development_event = self.store.append(
                "DevelopmentGateDecided",
                development_operation,
                {
                    "iteration": iteration,
                    "candidate_id": child.candidate_id,
                    "evaluate": decision.evaluate,
                    "reason": decision.reason,
                },
            )
        decision = DevelopmentDecision(
            evaluate=development_event.payload.get("evaluate") is True,
            reason=str(development_event.payload.get("reason", "")),
        )
        development_evaluation = None
        if decision.evaluate:
            development_evaluation = await self._ensure_evaluation(
                logical_operation_id=operation_id(self.store.run_id, "iteration", iteration, "development"),
                candidate=child,
                cases=self.scenario.development_cases,
                purpose="development",
                iteration=iteration,
            )

        evolution_operation = operation_id(self.store.run_id, "iteration", iteration, "evolution")
        self.store.append(
            "EvolutionRecorded",
            evolution_operation,
            {
                "iteration": iteration,
                "candidate_id": child.candidate_id,
                "parent_ids": list(child.parent_ids),
                "accepted": verdict.accepted,
                "train_before_evaluation_id": parent_evaluation.evaluation_id,
                "train_after_evaluation_id": child_evaluation.evaluation_id,
                "development_aggregate": development_evaluation.aggregate_score if development_evaluation else None,
            },
        )
        obligation_id = sha256_digest(
            canonical_json({"iteration": iteration, "candidate_id": child.candidate_id, "kind": "reflect"})
        )
        self.store.append(
            "DeferredWorkScheduled",
            operation_id(self.store.run_id, "iteration", iteration, "deferred-reflect"),
            {
                "obligation_id": obligation_id,
                "owning_iteration": iteration,
                "candidate_id": child.candidate_id,
                "parent_candidate_id": selected_parent.candidate_id,
                "working_parent_candidate_id": parent.candidate_id,
                "selection_parent_ids": list(selection_plan.parents),
                "component_sources": dict(selection_plan.selections),
                "selection_rationale": selection_plan.rationale,
                "session_kind": "reflect",
                "train_case_ids": list(case_ids),
                "train_before_aggregate": parent_evaluation.aggregate_score,
                "train_after_aggregate": child_evaluation.aggregate_score,
                "train_before_case_scores": _case_aggregate_scores(parent_evaluation),
                "train_after_case_scores": _case_aggregate_scores(child_evaluation),
                "accepted_by_minibatch_gate": verdict.accepted,
                "parent_development_aggregate": self._latest_development_aggregate(selected_parent.candidate_id),
                "candidate_development_aggregate": (
                    development_evaluation.aggregate_score if development_evaluation else None
                ),
                "changed_components": list(child.change.changed_units) if child.change is not None else [],
                "updates": dict(_optional_session_mapping(diagnosis, "updates")),
                "diagnosis": _optional_session_string(diagnosis, "diagnosis"),
                "patch_phase": mutation_context.patch_label,
                "acceptance_reason": verdict.reason,
            },
        )
        self._complete_iteration(
            iteration=iteration,
            resulting_candidate=child if verdict.accepted else selected_parent,
            evaluated_candidates=(parent, child),
            accepted=verdict.accepted,
            outcome="accepted" if verdict.accepted else "declined",
        )

    def _reject_mutation(
        self,
        *,
        candidate_operation: str,
        iteration: int,
        parent_ids: tuple[str, ...],
        case_ids: tuple[str, ...],
        diagnosis_operation: str,
        reason: str,
    ) -> None:
        self.store.append(
            "MutationRejected",
            candidate_operation,
            {
                "iteration": iteration,
                "parent_ids": list(parent_ids),
                "batch_case_ids": list(case_ids),
                "diagnosis_operation_id": diagnosis_operation,
                "verification_failure": reason,
            },
        )

    def _complete_rejected_iteration(
        self,
        *,
        iteration: int,
        resulting_parent: Candidate,
        working_parent: Candidate,
        case_ids: tuple[str, ...],
        reason: str,
    ) -> None:
        self._complete_iteration(
            iteration=iteration,
            resulting_candidate=resulting_parent,
            evaluated_candidates=(working_parent,),
            accepted=False,
            outcome="mutation_rejected",
            details={
                "batch_case_ids": list(case_ids),
                "verification_failure": reason,
            },
        )

    def _complete_iteration(
        self,
        *,
        iteration: int,
        resulting_candidate: Candidate,
        evaluated_candidates: Sequence[Candidate],
        accepted: bool,
        outcome: str,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        evaluated_candidate_ids = [candidate.candidate_id for candidate in evaluated_candidates]
        if not evaluated_candidate_ids or len(set(evaluated_candidate_ids)) != len(evaluated_candidate_ids):
            raise ValueError("Iteration completion requires unique evaluated candidate IDs")
        state = self._state()
        if resulting_candidate.candidate_id not in state.accepted_candidate_ids:
            raise ValueError("Iteration resulting candidate must be present in accepted state")
        unknown = sorted(set(evaluated_candidate_ids) - state.candidates.keys())
        if unknown:
            raise ValueError(f"Iteration evaluated candidates are unknown: {unknown}")
        if accepted != (outcome == "accepted"):
            raise ValueError("Only an accepted iteration may use the accepted outcome")
        if accepted and resulting_candidate.candidate_id != evaluated_candidate_ids[-1]:
            raise ValueError("An accepted iteration must carry forward its final evaluated candidate")
        extra = dict(details or {})
        reserved = {
            "schema_version",
            "iteration",
            "resulting_candidate_id",
            "evaluated_candidate_ids",
            "accepted",
            "outcome",
            "wall_seconds",
        }
        conflicts = sorted(reserved & extra.keys())
        if conflicts:
            raise ValueError(f"Iteration completion details override reserved fields: {conflicts}")
        self.store.append(
            "IterationCompleted",
            operation_id(self.store.run_id, "iteration", iteration),
            {
                "schema_version": ITERATION_COMPLETION_SCHEMA_VERSION,
                "iteration": iteration,
                "resulting_candidate_id": resulting_candidate.candidate_id,
                "evaluated_candidate_ids": evaluated_candidate_ids,
                "accepted": accepted,
                "outcome": outcome,
                "wall_seconds": self._iteration_wall_seconds(iteration),
                **extra,
            },
        )

    async def _drain_deferred(self) -> None:
        for obligation_id, obligation in sorted(self._state().pending_obligations.items()):
            kind = str(obligation.get("session_kind", ""))
            if kind != "reflect":
                raise ValueError(f"Unknown deferred session kind: {kind}")
            candidate_id = str(obligation.get("candidate_id", ""))
            train_case_ids = _string_list(obligation.get("train_case_ids"), "deferred train case IDs")
            spec = self.scenario.prompt_pack.session(
                "reflect",
                {
                    "iteration": cast(int, obligation.get("owning_iteration")),
                    "candidate_ids": [candidate_id],
                    "train_case_ids": list(train_case_ids),
                    "parent_candidate_id": cast(JsonValue, obligation.get("parent_candidate_id")),
                    "working_parent_candidate_id": cast(
                        JsonValue,
                        obligation.get("working_parent_candidate_id"),
                    ),
                    "selection_parent_ids": cast(JsonValue, obligation.get("selection_parent_ids")),
                    "component_sources": cast(JsonValue, obligation.get("component_sources")),
                    "selection_rationale": cast(JsonValue, obligation.get("selection_rationale")),
                    "train_before_aggregate": cast(JsonValue, obligation.get("train_before_aggregate")),
                    "train_after_aggregate": cast(JsonValue, obligation.get("train_after_aggregate")),
                    "train_before_case_scores": cast(
                        JsonValue,
                        obligation.get("train_before_case_scores"),
                    ),
                    "train_after_case_scores": cast(
                        JsonValue,
                        obligation.get("train_after_case_scores"),
                    ),
                    "accepted_by_minibatch_gate": cast(
                        JsonValue,
                        obligation.get("accepted_by_minibatch_gate"),
                    ),
                    "parent_development_aggregate": cast(
                        JsonValue,
                        obligation.get("parent_development_aggregate"),
                    ),
                    "candidate_development_aggregate": cast(
                        JsonValue,
                        obligation.get("candidate_development_aggregate"),
                    ),
                    "changed_components": cast(JsonValue, obligation.get("changed_components")),
                    "updates": cast(JsonValue, obligation.get("updates")),
                    "diagnosis": cast(JsonValue, obligation.get("diagnosis")),
                    "acceptance_reason": cast(JsonValue, obligation.get("acceptance_reason")),
                },
            )
            try:
                result = await self._ensure_session(
                    logical_operation_id=operation_id(self.store.run_id, "deferred", obligation_id, "reflect"),
                    spec=spec,
                    source_workspace=None,
                    result_validator=lambda value: _reflection_failure_reason(value, train_case_ids),
                    metrics_context={
                        "stage": "proposal.reflection",
                        "iteration": cast(int, obligation.get("owning_iteration")),
                        "candidate_id": candidate_id,
                    },
                )
            except SessionRetriesExhausted as error:
                self.store.append(
                    "DeferredWorkAbandoned",
                    operation_id(self.store.run_id, "deferred", obligation_id, "abandon"),
                    {
                        "obligation_id": obligation_id,
                        "candidate_id": candidate_id,
                        "reason": str(error),
                    },
                )
                continue
            lessons = _session_list(result, "lessons")
            _validate_lesson_case_ids(lessons, train_case_ids)
            self.store.append(
                "ExtensionStateChanged",
                operation_id(self.store.run_id, "deferred", obligation_id, "lessons"),
                {
                    "namespace": "autosaddler.lessons",
                    "schema_version": "autosaddler-lessons/v1",
                    "candidate_id": candidate_id,
                    "owning_iteration": cast(int, obligation.get("owning_iteration")),
                    "lessons": lessons,
                },
            )
            self.store.append(
                "DeferredWorkCompleted",
                operation_id(self.store.run_id, "deferred", obligation_id, "complete"),
                {"obligation_id": obligation_id, "candidate_id": candidate_id},
            )

    async def _ensure_session(
        self,
        *,
        logical_operation_id: str,
        spec: SessionSpec,
        source_workspace: Path | None,
        metrics_context: Mapping[str, JsonValue] | None = None,
        mutation_session: MutationSession | None = None,
        result_validator: Callable[[SessionResult], str | None] | None = None,
    ) -> SessionResult:
        failed_events = [
            event
            for event in self.store.events_of_type("SessionFailed")
            if event.payload.get("logical_operation_id") == logical_operation_id
            and isinstance(event.payload.get("result"), Mapping)
        ]
        for failed in failed_events:
            self._ensure_session_budget(failed.operation_id, session_result_from(failed.payload["result"]))

        completed_failure: str | None = None
        completed_events = [
            event
            for event in self.store.events_of_type("SessionCompleted")
            if event.payload.get("logical_operation_id") == logical_operation_id
        ]
        for completed in reversed(completed_events):
            result = session_result_from(completed.payload.get("result"))
            completed_failure = _session_failure_reason(result, spec)
            if completed_failure is None and result_validator is not None:
                completed_failure = result_validator(result)
            if completed_failure is not None:
                continue
            self._ensure_session_budget(logical_operation_id, result)
            return result

        starts = [
            event
            for event in self.store.events_of_type("SessionStarted")
            if event.payload.get("logical_operation_id") == logical_operation_id
        ]
        for started in starts:
            if (
                self.store.find("SessionCompleted", started.operation_id) is None
                and self.store.find("SessionFailed", started.operation_id) is None
            ):
                session_id = str(started.payload.get("session_id") or "")
                result_path = self.store.run_dir / "sessions" / session_id.removeprefix("sha256:") / "result.json"
                if result_path.is_file():
                    result = session_result_from(
                        self.store.read_json(result_path.relative_to(self.store.run_dir).as_posix())
                    )
                    result_ref = self.store.write_json(
                        result_path.relative_to(self.store.run_dir).as_posix(),
                        record(result),
                        kind="session-result",
                    )
                    observed_usage_events = [
                        event
                        for event in self.store.events_of_type("ModelUsageObserved")
                        if event.payload.get("attempt_operation_id") == started.operation_id
                    ]
                    observed_usage_sequences = {event.payload.get("usage_sequence") for event in observed_usage_events}
                    recovered_sink: EventModelUsageSink | None = None
                    if len(observed_usage_sequences) < len(result.usage):
                        original_invocation_id = started.payload.get("run_invocation_id")
                        if not isinstance(original_invocation_id, str) and observed_usage_events:
                            original_invocation_id = observed_usage_events[0].payload.get("run_invocation_id")
                        if not isinstance(original_invocation_id, str):
                            original_invocation_id = self.run_invocation_id
                        recovered_sink = EventModelUsageSink(
                            store=self.store,
                            attempt_operation_id=started.operation_id,
                            context={
                                "run_invocation_id": original_invocation_id,
                                "logical_operation_id": logical_operation_id,
                                "session_id": session_id,
                                "session_kind": spec.kind,
                                "provider": type(self.provider).__name__,
                                "attempt": started.payload.get("attempt"),
                                **{
                                    key: value
                                    for key, value in started.payload.items()
                                    if key in {"stage", "iteration", "candidate_id"}
                                },
                            },
                        )
                        for sequence, usage in enumerate(result.usage):
                            if sequence not in observed_usage_sequences:
                                recovered_sink.observe_at(sequence, usage)
                    if recovered_sink is not None and recovered_sink.write_errors:
                        self._record_observability_degradation(
                            attempt_operation=started.operation_id,
                            stage=started.payload.get("stage", "unknown"),
                            errors=recovered_sink.write_errors,
                        )
                    recovery_error = _session_failure_reason(result, spec)
                    if recovery_error is None and result_validator is not None:
                        recovery_error = result_validator(result)
                    terminal_payload: dict[str, JsonValue] = {
                        "logical_operation_id": logical_operation_id,
                        "session_id": session_id,
                        "result": record(result),
                        "result_artifact": record(result_ref),
                        "wall_seconds": 0.0,
                        "recovered_from_result_artifact": True,
                    }
                    if recovery_error is None:
                        request_artifact = started.payload.get("request")
                        if not isinstance(request_artifact, Mapping):
                            raise TypeError("Session request artifact must be an object")
                        request_value = self.store.read_json(str(request_artifact.get("uri", "")))
                        if not isinstance(request_value, Mapping):
                            raise TypeError("Session request must be an object")
                        mutation_delta = self._capture_mutation_delta(
                            mutation_session,
                            self.store.run_dir / str(request_value.get("workspace", "")),
                            started.operation_id,
                        )
                        recovery_error = _workspace_delta_failure_reason(result, mutation_delta)
                    if recovery_error is None:
                        if mutation_delta is not None:
                            terminal_payload["workspace_delta"] = record(mutation_delta)
                        self.store.append("SessionCompleted", started.operation_id, terminal_payload)
                        self._ensure_session_budget(logical_operation_id, result)
                        return result
                    self._ensure_session_budget(started.operation_id, result)
                    self.store.append(
                        "SessionFailed",
                        started.operation_id,
                        {
                            **terminal_payload,
                            "status": result.status,
                            "error": recovery_error,
                            "usage_incomplete": False,
                        },
                    )
                    continue
                self.store.append(
                    "SessionFailed",
                    started.operation_id,
                    {
                        "logical_operation_id": logical_operation_id,
                        "session_id": started.payload.get("session_id"),
                        "status": "interrupted",
                        "error": "Session start had no terminal event during replay",
                        "wall_seconds": 0.0,
                        "usage_incomplete": True,
                    },
                )

        max_attempts = self.session_retries + 1
        attempt = len(starts) + 1
        last_error = completed_failure or "session retry budget exhausted"
        while attempt <= max_attempts:
            session_id = sha256_digest(canonical_json({"operation_id": logical_operation_id, "attempt": attempt}))
            attempt_workspace = self.store.run_dir / "workspaces" / ".attempts" / session_id.removeprefix("sha256:")
            _prepare_session_attempt_workspace(source_workspace, attempt_workspace)
            attempt_operation = f"{logical_operation_id}:attempt:{attempt}"
            usage_context: dict[str, JsonValue] = {
                "run_invocation_id": self.run_invocation_id,
                "logical_operation_id": logical_operation_id,
                "session_id": session_id,
                "session_kind": spec.kind,
                "provider": type(self.provider).__name__,
                "attempt": attempt,
                **dict(metrics_context or {}),
            }
            usage_sink = EventModelUsageSink(
                store=self.store,
                attempt_operation_id=attempt_operation,
                context=usage_context,
            )
            request = SessionRequest(
                session_id=session_id,
                operation_id=attempt_operation,
                spec=spec,
                workspace=attempt_workspace,
                timeout_seconds=self._session_timeout(spec.kind),
                usage_observer=usage_sink.observe,
                trace_dir=(self.store.run_dir / "sessions" / session_id.removeprefix("sha256:")),
            )
            request_ref = self.store.write_json(
                f"sessions/{session_id.removeprefix('sha256:')}/request.json",
                {
                    "session_id": session_id,
                    "operation_id": request.operation_id,
                    "spec": record(spec),
                    "workspace": _relative_workspace(self.store.run_dir, attempt_workspace),
                    "timeout_seconds": request.timeout_seconds,
                },
                kind="session-request",
            )
            self.store.append(
                "SessionStarted",
                attempt_operation,
                {
                    "logical_operation_id": logical_operation_id,
                    "session_id": session_id,
                    "attempt": attempt,
                    "request": record(request_ref),
                    "run_invocation_id": self.run_invocation_id,
                    **dict(metrics_context or {}),
                },
            )
            attempt_started = time.monotonic()
            try:
                result = await self.provider.run(request)
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                if not any(
                    event.payload.get("attempt_operation_id") == attempt_operation
                    for event in self.store.events_of_type("ModelUsageObserved")
                ):
                    usage_sink.observe(
                        Usage(
                            status="failed",
                            error_type=type(error).__name__,
                            usage_incomplete=True,
                        )
                    )
                self.store.append(
                    "SessionFailed",
                    attempt_operation,
                    {
                        "logical_operation_id": logical_operation_id,
                        "session_id": session_id,
                        "status": "failed",
                        "error": last_error,
                        "wall_seconds": time.monotonic() - attempt_started,
                        "usage_incomplete": True,
                        **dict(metrics_context or {}),
                    },
                )
            else:
                result_ref = self.store.write_json(
                    f"sessions/{session_id.removeprefix('sha256:')}/result.json",
                    record(result),
                    kind="session-result",
                )
                last_error = _session_failure_reason(result, spec)
                if last_error is None and result_validator is not None:
                    last_error = result_validator(result)
                observed_usage_sequences = {
                    event.payload.get("usage_sequence")
                    for event in self.store.events_of_type("ModelUsageObserved")
                    if event.payload.get("attempt_operation_id") == attempt_operation
                }
                for sequence, usage in enumerate(result.usage):
                    if sequence not in observed_usage_sequences:
                        usage_sink.observe_at(sequence, usage)
                if last_error is not None and not result.usage and not observed_usage_sequences:
                    usage_sink.observe(
                        Usage(
                            status=result.status if result.status != "completed" else "failed",
                            error_type="SessionResultError",
                            usage_incomplete=True,
                        )
                    )
                if usage_sink.write_errors:
                    self._record_observability_degradation(
                        attempt_operation=attempt_operation,
                        stage=(metrics_context or {}).get("stage", "unknown"),
                        errors=usage_sink.write_errors,
                    )
                if last_error is None:
                    mutation_delta = self._capture_mutation_delta(
                        mutation_session,
                        attempt_workspace,
                        attempt_operation,
                    )
                    last_error = _workspace_delta_failure_reason(result, mutation_delta)
                if last_error is None:
                    terminal_payload: dict[str, JsonValue] = {
                        "logical_operation_id": logical_operation_id,
                        "session_id": session_id,
                        "result": record(result),
                        "result_artifact": record(result_ref),
                        "wall_seconds": time.monotonic() - attempt_started,
                        **dict(metrics_context or {}),
                    }
                    if mutation_delta is not None:
                        terminal_payload["workspace_delta"] = record(mutation_delta)
                    self.store.append(
                        "SessionCompleted",
                        attempt_operation,
                        terminal_payload,
                    )
                    self._ensure_session_budget(logical_operation_id, result)
                    return result
                self._ensure_session_budget(attempt_operation, result)
                self.store.append(
                    "SessionFailed",
                    attempt_operation,
                    {
                        "logical_operation_id": logical_operation_id,
                        "session_id": session_id,
                        "status": result.status,
                        "error": last_error,
                        "result": record(result),
                        "result_artifact": record(result_ref),
                        "wall_seconds": time.monotonic() - attempt_started,
                        "usage_incomplete": usage_sink.observed_count > 0,
                        **dict(metrics_context or {}),
                    },
                )
            if usage_sink.write_errors:
                self._record_observability_degradation(
                    attempt_operation=attempt_operation,
                    stage=(metrics_context or {}).get("stage", "unknown"),
                    errors=usage_sink.write_errors,
                )
            if attempt < max_attempts and self.session_retry_backoff_seconds:
                await asyncio.sleep(self.session_retry_backoff_seconds * (2 ** (attempt - 1)))
            attempt += 1
        raise SessionRetriesExhausted(f"Provider session failed after {max_attempts} attempts: {last_error}")

    def _capture_mutation_delta(
        self,
        session: MutationSession | None,
        attempt_workspace: Path,
        attempt_operation_id: str,
    ) -> WorkspaceDelta | None:
        if session is None:
            return None
        return self.scenario.harness_space.capture_attempt_delta(
            session,
            attempt_workspace,
            attempt_operation_id=attempt_operation_id,
        )

    def _mutation_outcome(
        self,
        *,
        logical_operation_id: str,
        result: SessionResult,
    ) -> MutationOutcome:
        completed = next(
            event
            for event in self.store.events_of_type("SessionCompleted")
            if event.payload.get("logical_operation_id") == logical_operation_id
        )
        raw_delta = completed.payload.get("workspace_delta")
        delta = None
        if isinstance(raw_delta, Mapping):
            raw_artifact = raw_delta.get("artifact")
            if not isinstance(raw_artifact, Mapping):
                raise TypeError("Workspace delta artifact must be an object")
            raw_paths = raw_delta.get("changed_paths")
            if not isinstance(raw_paths, list) or not all(isinstance(path, str) for path in raw_paths):
                raise TypeError("Workspace delta changed paths must be strings")
            delta = WorkspaceDelta(
                attempt_operation_id=str(raw_delta.get("attempt_operation_id", "")),
                changed_paths=tuple(raw_paths),
                artifact=ArtifactRef(
                    uri=str(raw_artifact.get("uri", "")),
                    kind=str(raw_artifact.get("kind", "")),
                    sha256=str(raw_artifact["sha256"]) if raw_artifact.get("sha256") is not None else None,
                    bytes=int(raw_artifact["bytes"]) if raw_artifact.get("bytes") is not None else None,
                ),
            )
        outcome = MutationOutcome(
            result=result,
            workspace_delta=delta,
        )
        delta_error = _workspace_delta_failure_reason(result, delta)
        if delta_error is not None:
            raise ValueError(delta_error)
        return outcome

    async def _ensure_evaluation(
        self,
        *,
        logical_operation_id: str,
        candidate: Candidate,
        cases: Sequence[Case],
        purpose: str,
        iteration: int,
    ) -> Evaluation:
        completed = self.store.find("EvaluationCompleted", logical_operation_id)
        if completed is not None:
            evaluation = evaluation_from(completed.payload.get("evaluation"))
            self._ensure_evaluation_budget(logical_operation_id, evaluation)
            return evaluation
        split = cases[0].split if cases else None
        if split is None or any(case.split != split for case in cases):
            raise ValueError("Evaluation cases must be a non-empty single split")
        self.store.append(
            "EvaluationStarted",
            logical_operation_id,
            {
                "candidate_id": candidate.candidate_id,
                "case_ids": [case.case_id for case in cases],
                "split": split,
                "purpose": purpose,
                "iteration": iteration,
            },
        )
        evaluation_id = sha256_digest(logical_operation_id)
        artifact_prefix = {
            "train": "evaluations",
            "development": "quarantine/dev",
            "test": "post_optimization/test",
        }[split]
        context = EvaluationContext(
            run_id=self.store.run_id,
            operation_id=logical_operation_id,
            iteration=iteration,
            purpose=cast(Any, purpose),
            split=split,
            repetitions=self.scenario.evaluation_repetitions,
            capture_traces=split == "train",
            artifact_dir=self.store.run_dir / artifact_prefix / evaluation_id.removeprefix("sha256:"),
            attempt_sink=EventEvaluationAttemptSink(
                store=self.store,
                evaluation_operation_id=logical_operation_id,
                metrics_context={
                    "run_invocation_id": self.run_invocation_id,
                    "candidate_id": candidate.candidate_id,
                    "iteration": iteration,
                    "split": split,
                    "evaluation_purpose": purpose,
                },
            ),
        )
        evaluation = await self.scenario.evaluator.evaluate(candidate, cases, context)
        evaluation_ref = self.store.write_json(
            f"{artifact_prefix}/{evaluation.evaluation_id.removeprefix('sha256:')}/evaluation.json",
            record(evaluation),
            kind="evaluation",
        )
        self.store.append(
            "EvaluationCompleted",
            logical_operation_id,
            {"evaluation": record(evaluation), "artifact": record(evaluation_ref)},
        )
        self._ensure_evaluation_budget(logical_operation_id, evaluation)
        return evaluation

    def _ensure_session_budget(self, logical_operation_id: str, result: SessionResult) -> None:
        self.store.append(
            "BudgetUpdated",
            f"{logical_operation_id}:budget",
            {"source": "session", "logical_operation_id": logical_operation_id, "delta": record(result.cost)},
        )

    def _session_timeout(self, kind: str) -> float:
        if kind == "evolve":
            return self.selection_timeout_seconds
        if kind == "diagnose_patch":
            return self.diagnosis_patch_timeout_seconds
        if kind == "reflect":
            return self.reflection_timeout_seconds
        raise ValueError(f"Unknown session kind: {kind}")

    def _iteration_wall_seconds(self, iteration: int) -> float:
        started = self._iteration_started_at.get(iteration)
        if started is None:
            raise RuntimeError(f"Iteration {iteration} has no monotonic start time")
        return time.monotonic() - started

    def _run_wall_seconds(self) -> float:
        started = getattr(self, "_run_invocation_started_at", None)
        if not isinstance(started, float):
            raise RuntimeError("Run invocation has no monotonic start time")
        return time.monotonic() - started

    def _record_observability_degradation(
        self,
        *,
        attempt_operation: str,
        stage: JsonValue,
        errors: Sequence[str],
    ) -> None:
        self.store.append(
            "ObservabilityDegraded",
            f"{attempt_operation}:observability-degraded",
            {
                "run_invocation_id": self.run_invocation_id,
                "attempt_operation_id": attempt_operation,
                "stage": stage,
                "errors": list(errors),
            },
        )

    def _ensure_evaluation_budget(self, logical_operation_id: str, evaluation: Evaluation) -> None:
        self.store.append(
            "BudgetUpdated",
            f"{logical_operation_id}:budget",
            {
                "source": "evaluation",
                "logical_operation_id": logical_operation_id,
                "attempted_rollouts": evaluation.attempted_rollouts,
                "valid_rollouts": evaluation.valid_rollouts,
            },
        )

    def _select_best(self) -> tuple[Candidate, Evaluation]:
        state = self._state()
        candidates = []
        for candidate_id in state.accepted_candidate_ids:
            evaluations = [
                evaluation
                for evaluation in state.evaluations.values()
                if evaluation.candidate_id == candidate_id and evaluation.split == "development"
            ]
            if evaluations:
                candidates.append((state.candidates[candidate_id], evaluations[-1]))
        return self.policies.ranking.select(candidates)

    def _latest_development_aggregate(self, candidate_id: str) -> float | None:
        evaluations = [
            evaluation
            for evaluation in self._state().evaluations.values()
            if evaluation.candidate_id == candidate_id and evaluation.split == "development"
        ]
        return evaluations[-1].aggregate_score if evaluations else None

    def _state(self) -> RunState:
        return RunState.replay(self.store.events())

    def _load_result(self) -> OptimizationResult:
        value = self.store.read_json("result.json")
        if not isinstance(value, Mapping):
            raise TypeError("Optimization result must be an object")
        return OptimizationResult(
            run_id=str(value.get("run_id", "")),
            selected_candidate_id=str(value.get("selected_candidate_id", "")),
            development_score=float(value.get("development_score")),
            iterations=int(value.get("iterations")),
        )


def _required_score(evaluation: Evaluation) -> float:
    if evaluation.aggregate_score is None:
        raise ValueError("Selected development evaluation has no valid aggregate")
    return evaluation.aggregate_score


def _prepare_session_attempt_workspace(source: Path | None, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is None:
        destination.mkdir()
        return

    if not source.is_dir():
        raise FileNotFoundError(f"Session source workspace does not exist: {source}")

    source_root = source.resolve()

    def ignore(path: str, names: list[str]) -> set[str]:
        current = Path(path).resolve()
        ignored: set[str] = set()
        if current == source_root:
            ignored.update({".git", ".github", ".claude", ".copilot", "AGENTS.md", "CLAUDE.md"})
        if current.name == ".autosaddler":
            ignored.update({"session_output.json", "session_output_schema.json"})
        return ignored.intersection(names)

    shutil.copytree(source, destination, symlinks=True, ignore=ignore)


def _case_aggregate_scores(evaluation: Evaluation) -> dict[str, float]:
    result: dict[str, float] = {}
    for case_id in evaluation.requested_case_ids:
        scores = [
            observation.score
            for observation in evaluation.observations
            if observation.case_id == case_id and observation.is_valid and observation.score is not None
        ]
        if not scores:
            raise ValueError(f"Evaluation has no valid scores for case {case_id}")
        result[case_id] = sum(scores) / len(scores)
    return result


def _session_failure_reason(result: SessionResult, spec: SessionSpec) -> str | None:
    if result.status != "completed":
        return result.error or f"session ended with status {result.status}"
    if not result.raw_response.strip():
        return "session returned an empty final response"
    if result.structured_output is None:
        return "session completed without structured output"
    return session_output_validation_error(spec.output_schema, result.structured_output)


def _selection_plan(result: SessionResult, allowed: tuple[str, ...]) -> CompositionPlan:
    output = result.structured_output
    if output is None:
        raise ValueError("Evolution session produced no structured output")
    parent_ids = _string_list(output.get("parent_ids"), "evolution parent_ids")
    if len(set(parent_ids)) != len(parent_ids) or any(parent_id not in allowed for parent_id in parent_ids):
        raise ValueError("Evolution session must select unique accepted parents")
    sources_value = output.get("component_sources", {})
    if not isinstance(sources_value, Mapping) or any(
        not isinstance(component, str) or not isinstance(source_id, str)
        for component, source_id in sources_value.items()
    ):
        raise TypeError("Evolution component_sources must map component IDs to candidate IDs")
    raw_selections = cast(Mapping[str, str], sources_value)
    selections = {
        component: source_id for component, source_id in raw_selections.items() if source_id != parent_ids[0]
    }
    source_ids = set(selections.values())
    if any(source_id not in parent_ids for source_id in source_ids):
        raise ValueError("Every composition source must be declared in parent_ids")
    if set(parent_ids[1:]) != source_ids - {parent_ids[0]}:
        raise ValueError("Every additional composition parent must supply a component")
    rationale_value = output.get("rationale", "Selected an accepted parent without composition.")
    if not isinstance(rationale_value, str) or not rationale_value:
        raise TypeError("Evolution rationale must be a non-empty string")
    return CompositionPlan(
        parents=parent_ids,
        selections=dict(selections),
        overrides={},
        rationale=rationale_value,
    )


def _selection_failure_reason(result: SessionResult, allowed: tuple[str, ...]) -> str | None:
    try:
        _selection_plan(result, allowed)
    except (TypeError, ValueError) as error:
        return str(error)
    return None


def _workspace_delta_failure_reason(result: SessionResult, delta: WorkspaceDelta | None) -> str | None:
    output = result.structured_output
    if not isinstance(output, Mapping) or "changed_paths" not in output:
        return None
    declared = output.get("changed_paths")
    if not isinstance(declared, list) or any(not isinstance(path, str) for path in declared):
        return "Session output changed_paths must be a list of strings"
    observed = set(delta.changed_paths) if delta is not None else set()
    if set(declared) != observed:
        return f"Session output changed_paths do not match workspace delta: declared={sorted(declared)!r}, observed={sorted(observed)!r}"
    return None


def _optional_session_mapping(result: SessionResult, key: str) -> Mapping[str, JsonValue]:
    output = result.structured_output
    if not isinstance(output, Mapping):
        raise TypeError("Session output must be an object")
    value = output.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Session output {key!r} must be an object")
    return cast(Mapping[str, JsonValue], value)


def _optional_session_string(result: SessionResult, *keys: str) -> str:
    output = result.structured_output
    if not isinstance(output, Mapping):
        raise TypeError("Session output must be an object")
    for key in keys:
        value = output.get(key)
        if isinstance(value, str):
            return value
    return ""


def _session_list(result: SessionResult, key: str) -> list[JsonValue]:
    output = result.structured_output
    if output is None or not isinstance(output.get(key), list):
        raise TypeError(f"Session output {key!r} must be a list")
    return cast(list[JsonValue], output[key])


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{label} must be a non-empty list of strings")
    return tuple(value)


def _validate_lesson_case_ids(
    lessons: Sequence[JsonValue],
    train_case_ids: Sequence[str],
) -> None:
    allowed = set(train_case_ids)
    for index, lesson in enumerate(lessons):
        if not isinstance(lesson, Mapping):
            raise TypeError(f"Reflection lesson {index} must be an object")
        evidence_case_ids = lesson.get("evidence_case_ids")
        if not isinstance(evidence_case_ids, list) or any(
            not isinstance(case_id, str) for case_id in evidence_case_ids
        ):
            raise TypeError(f"Reflection lesson {index} evidence_case_ids must be strings")
        unknown = sorted(set(evidence_case_ids) - allowed)
        if unknown:
            raise ValueError(f"Reflection lesson {index} references non-training case IDs: {unknown}")


def _reflection_failure_reason(result: SessionResult, train_case_ids: Sequence[str]) -> str | None:
    try:
        lessons = _session_list(result, "lessons")
        _validate_lesson_case_ids(lessons, train_case_ids)
    except (TypeError, ValueError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def _verdict(value: object) -> PatchVerdict:
    if not isinstance(value, Mapping):
        raise TypeError("Acceptance verdict must be an object")
    compared = _string_list(value.get("compared_case_ids"), "compared case IDs")
    return PatchVerdict(
        before_score=float(value.get("before_score")),
        after_score=float(value.get("after_score")),
        compared_case_ids=compared,
        accepted=value.get("accepted") is True,
        reason=str(value.get("reason", "")),
    )


def _relative_workspace(run_dir: Path, workspace: Path) -> str:
    resolved = workspace.resolve()
    if not resolved.is_relative_to(run_dir):
        raise ValueError(f"Session workspace must be inside the run directory: {workspace}")
    return resolved.relative_to(run_dir).as_posix()
