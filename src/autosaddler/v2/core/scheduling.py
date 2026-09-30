"""Adaptive extension of the task-selection interface.

Passive task-selection policies implement ``select(cases, iteration)`` and choose a
training batch before the evolution session. An adaptive policy instead chooses the
batch for the prepared working parent and may need optimizer sessions to do so, so it
describes that work as steps and the engine executes them durably:

- ``next_selection_step`` returns the next session or state change needed before
  selection, and finally a ``TaskSelection`` or ``NoSelection``;
- ``iteration_changes`` records state from an iteration's outcome;
- ``after_reflection`` may request one deferred session after reflection.

Every step result is appended as an ``ExtensionStateChanged`` event in the policy's
namespace before the policy is consulted again, and every policy method is a pure
function of the replayed events it receives. A resumed run therefore replays the
same steps without repeating paid work.

Every session context the engine builds for an adaptive policy carries
``task_selection.prompt_overlay``, the name the policy declares for the prompt assets
it needs. Scenario prompt packs branch on that name, never on the policy's registry
name, so another policy can reuse an overlay without scenario changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from autosaddler.v2.core.domain import ArtifactRef, Case, JsonValue, freeze_json_mapping
from autosaddler.v2.core.events import RunEvent
from autosaddler.v2.core.serde import artifact_from
from autosaddler.v2.prompting.models import SessionKind, SessionResult

if TYPE_CHECKING:
    from autosaddler.v2.core.policies import TaskSelection

StatePayload: TypeAlias = Mapping[str, JsonValue]
MAX_SELECTION_STEPS = 8
TASK_SELECTION_CONTEXT_KEY = "task_selection"
PROMPT_OVERLAY_KEY = "prompt_overlay"


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    iteration: int
    train_cases: tuple[Case, ...]
    working_parent_id: str
    selected_parent_id: str
    selection_parent_ids: tuple[str, ...]
    component_sources: Mapping[str, str]
    selection_rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_sources", MappingProxyType(dict(self.component_sources)))


@dataclass(frozen=True, slots=True)
class SessionStep:
    """Run one optimizer session and record the state derived from its result."""

    name: str
    kind: SessionKind
    context: Mapping[str, JsonValue]
    stage: str
    validate: Callable[[SessionResult], str | None]
    record: Callable[[SessionResult], StatePayload]
    record_exhausted: Callable[[str], StatePayload] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.kind or not self.stage:
            raise ValueError("Session steps require a name, kind, and metrics stage")
        object.__setattr__(self, "context", freeze_json_mapping(self.context))


@dataclass(frozen=True, slots=True)
class StateStep:
    """Record a state change that needs no optimizer session."""

    name: str
    payload: StatePayload

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("State steps require a name")
        object.__setattr__(self, "payload", freeze_json_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class NoSelection:
    reason: str


SelectionStep: TypeAlias = "SessionStep | StateStep | TaskSelection | NoSelection"


@dataclass(frozen=True, slots=True)
class IterationFeedback:
    """Training facts of one iteration, recorded for adaptive policies only."""

    iteration: int
    outcome: str
    case_ids: tuple[str, ...]
    working_parent_id: str
    child_id: str | None
    train_before_evaluation_id: str
    train_after_evaluation_id: str | None
    train_before_case_scores: Mapping[str, float]
    train_after_case_scores: Mapping[str, float] | None
    train_before_evidence: ArtifactRef | None
    train_after_evidence: ArtifactRef | None
    diagnosis: str | None
    # The diagnosis session's structured output without its schema version.
    patch_intent: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_before_case_scores", MappingProxyType(dict(self.train_before_case_scores)))
        if self.train_after_case_scores is not None:
            object.__setattr__(self, "train_after_case_scores", MappingProxyType(dict(self.train_after_case_scores)))
        if self.patch_intent is not None:
            object.__setattr__(self, "patch_intent", freeze_json_mapping(self.patch_intent))


def iteration_feedback_from(value: object) -> IterationFeedback:
    if not isinstance(value, Mapping):
        raise TypeError("Iteration feedback must be an object")
    case_ids = value.get("case_ids")
    if not isinstance(case_ids, list) or any(not isinstance(item, str) for item in case_ids):
        raise TypeError("Iteration feedback case_ids must be strings")
    return IterationFeedback(
        iteration=int(value["iteration"]),
        outcome=str(value["outcome"]),
        case_ids=tuple(case_ids),
        working_parent_id=str(value["working_parent_id"]),
        child_id=_optional_string(value.get("child_id")),
        train_before_evaluation_id=str(value["train_before_evaluation_id"]),
        train_after_evaluation_id=_optional_string(value.get("train_after_evaluation_id")),
        train_before_case_scores=_scores(value.get("train_before_case_scores")),
        train_after_case_scores=(
            _scores(value.get("train_after_case_scores"))
            if value.get("train_after_case_scores") is not None
            else None
        ),
        train_before_evidence=_optional_artifact(value.get("train_before_evidence")),
        train_after_evidence=_optional_artifact(value.get("train_after_evidence")),
        diagnosis=_optional_string(value.get("diagnosis")),
        patch_intent=_optional_mapping(value.get("patch_intent")),
    )


def _optional_mapping(value: object) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Iteration feedback patch_intent must be an object")
    return value


@dataclass(frozen=True, slots=True)
class DeferredRequest:
    """One deferred optimizer session requested after reflection."""

    kind: SessionKind
    stage: str
    payload: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not self.kind or not self.stage:
            raise ValueError("Deferred requests require a session kind and metrics stage")
        object.__setattr__(self, "payload", freeze_json_mapping(self.payload))


@runtime_checkable
class AdaptiveTaskSelectionPolicy(Protocol):
    namespace: str
    prompt_overlay: str
    required_session_kinds: frozenset[str]

    def settings_record(self) -> dict[str, JsonValue]: ...

    def session_timeouts(self) -> Mapping[str, float]: ...

    def prompt_context(self, events: Sequence[RunEvent], iteration: int) -> Mapping[str, JsonValue]: ...

    def next_selection_step(self, events: Sequence[RunEvent], request: SelectionRequest) -> SelectionStep: ...

    def iteration_changes(
        self,
        events: Sequence[RunEvent],
        feedback: IterationFeedback,
    ) -> Sequence[StatePayload]: ...

    def after_reflection(
        self,
        events: Sequence[RunEvent],
        feedback: IterationFeedback,
        lessons: Sequence[JsonValue],
    ) -> DeferredRequest | None: ...

    def deferred_context(self, events: Sequence[RunEvent], request: DeferredRequest) -> Mapping[str, JsonValue]: ...

    def deferred_failure_reason(
        self,
        events: Sequence[RunEvent],
        request: DeferredRequest,
        result: SessionResult,
    ) -> str | None: ...

    def deferred_changes(
        self,
        events: Sequence[RunEvent],
        request: DeferredRequest,
        result: SessionResult,
    ) -> Sequence[StatePayload]: ...

    def deferred_exhausted_changes(
        self,
        events: Sequence[RunEvent],
        request: DeferredRequest,
        error: str,
    ) -> Sequence[StatePayload]: ...


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Iteration feedback string fields must be strings or null")
    return value


def _optional_artifact(value: object) -> ArtifactRef | None:
    return artifact_from(value) if value is not None else None


def _scores(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or isinstance(score, bool) or not isinstance(score, (int, float))
        for key, score in value.items()
    ):
        raise TypeError("Iteration feedback case scores must map case IDs to numbers")
    return {str(key): float(score) for key, score in value.items()}
