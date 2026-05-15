from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Protocol

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Candidate,
    Case,
    ChangeSummary,
    Cost,
    Evaluation,
    EvaluationPurpose,
    JsonValue,
    Observation,
    Split,
)
from autosaddler.v2.prompting.models import Capability, SessionKind, SessionRequest, SessionResult, SessionSpec, Usage


@dataclass(frozen=True, slots=True)
class MutationContext:
    iteration: int
    patch_label: str
    evidence: ArtifactRef | None
    workspace_root: Path


@dataclass(slots=True)
class MutationSession:
    session_id: str
    parent: Candidate
    workspace: Path
    context: MutationContext
    output_contract: ArtifactRef | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceDelta:
    attempt_operation_id: str
    changed_paths: tuple[str, ...]
    artifact: ArtifactRef


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    result: SessionResult
    workspace_delta: WorkspaceDelta | None


@dataclass(frozen=True, slots=True)
class CompositionPlan:
    parents: tuple[str, ...]
    selections: Mapping[str, str]
    overrides: Mapping[str, str]
    rationale: str


@dataclass(frozen=True, slots=True)
class MaterializedHarness:
    root: Path
    candidate_id: str
    release: Callable[[], None]


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    run_id: str
    operation_id: str
    iteration: int | None
    purpose: EvaluationPurpose
    split: Split
    repetitions: int
    capture_traces: bool
    artifact_dir: Path
    attempt_sink: "EvaluationAttemptSink"


class EvaluationAttemptSink(Protocol):
    def pending(
        self,
        *,
        candidate_id: str,
        case_id: str,
        repetition: int,
    ) -> tuple[str, int] | None: ...

    def completed(
        self,
        *,
        candidate_id: str,
        case_id: str,
        repetition: int,
    ) -> Observation | None: ...

    def start(
        self,
        *,
        candidate_id: str,
        case_id: str,
        repetition: int,
    ) -> tuple[str, int]: ...

    def complete(self, attempt_id: str, observation: Observation, cost: Cost) -> None: ...

    def fail(self, attempt_id: str, error_kind: str, cost: Cost) -> None: ...

    def observe_usage(self, attempt_id: str, usage: Usage) -> None: ...


class HarnessSpace(Protocol):
    def seed(self) -> Candidate: ...
    def begin_mutation(self, parent: Candidate, context: MutationContext) -> MutationSession: ...
    def capture_attempt_delta(
        self,
        session: MutationSession,
        attempt_workspace: Path,
        *,
        attempt_operation_id: str,
    ) -> WorkspaceDelta | None: ...
    def apply_mutation(self, session: MutationSession, outcome: MutationOutcome) -> None: ...
    def finalize(self, session: MutationSession) -> Candidate: ...
    def compose(self, plan: CompositionPlan) -> Candidate: ...
    def materialize(self, candidate: Candidate, purpose: str) -> MaterializedHarness: ...
    def diff(self, parent: Candidate, child: Candidate) -> ChangeSummary: ...


class Evaluator(Protocol):
    async def evaluate(
        self,
        candidate: Candidate,
        cases: Sequence[Case],
        context: EvaluationContext,
    ) -> Evaluation: ...


class EvidenceBuilder(Protocol):
    def build(self, evaluation: Evaluation) -> ArtifactRef: ...


class PromptPack(Protocol):
    def session(self, kind: SessionKind, context: Mapping[str, JsonValue]) -> SessionSpec: ...


class AgentProvider(Protocol):
    async def run(self, request: SessionRequest) -> SessionResult: ...


@dataclass(frozen=True, slots=True)
class ScenarioComponents:
    name: str
    version: str
    harness_space: HarnessSpace
    evaluator: Evaluator
    evidence_builder: EvidenceBuilder
    prompt_pack: PromptPack
    train_cases: tuple[Case, ...]
    development_cases: tuple[Case, ...]
    required_capabilities: frozenset[Capability]
    evaluation_repetitions: int = 1
    resolved_entities: Mapping[str, str | Mapping[str, JsonValue]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("Scenario name and version must be non-empty")
        if self.evaluation_repetitions < 1:
            raise ValueError("Scenario evaluation_repetitions must be positive")
        train_ids = {case.case_id for case in self.train_cases}
        development_ids = {case.case_id for case in self.development_cases}
        if not train_ids or not development_ids:
            raise ValueError("Scenarios require non-empty train and development cases")
        if train_ids & development_ids:
            raise ValueError("Train and development case IDs must be disjoint")
        if any(case.split != "train" for case in self.train_cases):
            raise ValueError("train_cases may contain the train split only")
        if any(case.split != "development" for case in self.development_cases):
            raise ValueError("development_cases may contain the development split only")
        object.__setattr__(self, "resolved_entities", MappingProxyType(dict(self.resolved_entities)))