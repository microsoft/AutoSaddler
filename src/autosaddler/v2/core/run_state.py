from __future__ import annotations

from dataclasses import dataclass

from autosaddler.v2.core.domain import Candidate, Evaluation
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.core.serde import candidate_from, evaluation_from


@dataclass(frozen=True, slots=True)
class RunState:
    candidates: dict[str, Candidate]
    accepted_candidate_ids: tuple[str, ...]
    evaluations: dict[str, Evaluation]
    pending_obligations: dict[str, dict[str, object]]
    attempted_rollouts: int
    selected_candidate_id: str | None
    optimization_completed: bool
    run_completed: bool

    @classmethod
    def replay(cls, events: tuple[RunEvent, ...]) -> "RunState":
        candidates: dict[str, Candidate] = {}
        accepted: list[str] = []
        evaluations: dict[str, Evaluation] = {}
        obligations: dict[str, dict[str, object]] = {}
        selected: str | None = None
        optimization_completed = False
        run_completed = False
        for event in events:
            if event.event_type == "RunStarted":
                seed_value = event.payload.get("seed_candidate")
                seed = candidate_from(seed_value)
                candidates[seed.candidate_id] = seed
                accepted.append(seed.candidate_id)
            elif event.event_type == "CandidateFinalized":
                candidate = candidate_from(event.payload.get("candidate"))
                candidates[candidate.candidate_id] = candidate
            elif event.event_type == "EvaluationCompleted":
                evaluations[event.operation_id] = evaluation_from(event.payload.get("evaluation"))
            elif event.event_type == "EvolutionRecorded" and event.payload.get("accepted") is True:
                candidate_id = event.payload.get("candidate_id")
                if isinstance(candidate_id, str) and candidate_id not in accepted:
                    accepted.append(candidate_id)
            elif event.event_type == "DeferredWorkScheduled":
                obligation_id = event.payload.get("obligation_id")
                if not isinstance(obligation_id, str):
                    raise TypeError("DeferredWorkScheduled requires obligation_id")
                obligations[obligation_id] = dict(event.payload)
            elif event.event_type in {"DeferredWorkCompleted", "DeferredWorkAbandoned"}:
                obligation_id = event.payload.get("obligation_id")
                if isinstance(obligation_id, str):
                    obligations.pop(obligation_id, None)
            elif event.event_type == "BestCandidateSelected":
                candidate_id = event.payload.get("candidate_id")
                if not isinstance(candidate_id, str):
                    raise TypeError("BestCandidateSelected requires candidate_id")
                selected = candidate_id
            elif event.event_type == "OptimizationCompleted":
                optimization_completed = True
            elif event.event_type == "RunCompleted":
                run_completed = True
        return cls(
            candidates=candidates,
            accepted_candidate_ids=tuple(accepted),
            evaluations=evaluations,
            pending_obligations=obligations,
            attempted_rollouts=sum(1 for event in events if event.event_type == "EvaluationAttemptStarted"),
            selected_candidate_id=selected,
            optimization_completed=optimization_completed,
            run_completed=run_completed,
        )