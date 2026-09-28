"""Data models for the EvolutionDAG proposer.

All dataclass definitions with JSON serialization/deserialization support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# SDK Session metadata
# ---------------------------------------------------------------------------


@dataclass
class SDKSessionInfo:
    """Metadata extracted from a Claude Agent SDK session JSON file."""

    model: str
    timeout: float

    tool_call_count: int
    turns: int

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int

    session_json_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SDKSessionInfo:
        return cls(**d)


# ---------------------------------------------------------------------------
# SelectionDecision — Agent records (Session 0)
# ---------------------------------------------------------------------------


@dataclass
class SelectionDecision:
    """Which candidate(s) were used to build this iteration's worktree."""

    parent_candidates: list[int]  # candidate indices used (via rsync/cherry-pick)
    reasoning: str  # why these candidates were selected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SelectionDecision:
        return cls(
            parent_candidates=d["parent_candidates"],
            reasoning=d["reasoning"],
        )


# ---------------------------------------------------------------------------
# PatchIntent — Agent records (Session 1)
# ---------------------------------------------------------------------------


@dataclass
class PatchIntent:
    """What the Agent intended to change and why."""

    target_scenarios: list[str]
    approach: str

    files_changed: list[str]
    change_summary: str

    diagnosis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatchIntent:
        return cls(
            target_scenarios=d["target_scenarios"],
            approach=d["approach"],
            files_changed=d["files_changed"],
            change_summary=d["change_summary"],
            diagnosis=d.get("diagnosis"),
        )


# ---------------------------------------------------------------------------
# ReflectionEntry — Agent records (Session 2)
# ---------------------------------------------------------------------------


@dataclass
class ReflectionEntry:
    """Per-scenario reflection after seeing initial/re-evaluation results."""

    scenario_id: str
    status_change: str  # "fixed" | "regressed" | "still_failing" | "still_passing"
    explanation: str
    root_cause: str | None = None
    prevention_or_next: str | None = None
    generalization_note: str | None = None  # development set analysis

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReflectionEntry:
        return cls(
            scenario_id=d["scenario_id"],
            status_change=d["status_change"],
            explanation=d["explanation"],
            root_cause=d.get("root_cause"),
            prevention_or_next=d.get("prevention_or_next"),
            generalization_note=d.get("generalization_note"),
        )


# ---------------------------------------------------------------------------
# ScenarioImpact — Outer loop computes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioImpact:
    """Per-scenario initial/re-evaluation impact of a patch."""

    scenario_id: str
    score_before: float  # 0 or 1
    score_after: float  # 0 or 1
    status_change: str  # "fixed" | "regressed" | "still_failing" | "still_passing"
    rationale_before: str | None = None
    rationale_after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScenarioImpact:
        return cls(**d)


# ---------------------------------------------------------------------------
# PatchVerdict — Outer loop computes
# ---------------------------------------------------------------------------


@dataclass
class PatchVerdict:
    """Overall verdict for a patch: effectiveness + safety."""

    is_good_patch: bool
    effectiveness: bool  # target scenario(s) fixed
    safety: bool  # no regressions

    scenario_impacts: list[ScenarioImpact]
    reflections: list[ReflectionEntry]
    lessons_learned: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scenario_impacts"] = [si.to_dict() for si in self.scenario_impacts]
        d["reflections"] = [r.to_dict() for r in self.reflections]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatchVerdict:
        return cls(
            is_good_patch=d["is_good_patch"],
            effectiveness=d["effectiveness"],
            safety=d["safety"],
            scenario_impacts=[ScenarioImpact.from_dict(si) for si in d.get("scenario_impacts", [])],
            reflections=[ReflectionEntry.from_dict(r) for r in d.get("reflections", [])],
            lessons_learned=d.get("lessons_learned", []),
        )


# ---------------------------------------------------------------------------
# EvolutionEdge
# ---------------------------------------------------------------------------


@dataclass
class EvolutionEdge:
    """Edge in the evolution DAG: parent → child relationship."""

    parent_idx: int
    child_idx: int
    edge_type: str  # "base" | "cherry_pick"

    # Impact fields — filled after evaluation for both base and cherry_pick edges
    code_diff: str | None = None
    code_diff_path: str | None = None
    code_diff_sha256: str | None = None
    code_diff_size_bytes: int | None = None
    files_changed: list[str] | None = None

    mini_batch_ids: list[str] | None = None
    score_before: float | None = None
    score_after: float | None = None
    score_delta: float | None = None
    improved: bool | None = None

    scenarios_fixed: list[str] | None = None
    scenarios_regressed: list[str] | None = None
    scenarios_still_failing: list[str] | None = None
    scenarios_still_passing: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionEdge:
        return cls(**d)


# ---------------------------------------------------------------------------
# EvolutionNode
# ---------------------------------------------------------------------------


@dataclass
class EvolutionNode:
    """A candidate node in the evolution DAG."""

    idx: int
    iteration: int
    created_at: str  # ISO timestamp

    score_train_before: float | None = None
    score_train_after: float | None = None
    score_val: float | None = None
    val_evaluated: bool = False

    mini_batch_ids: list[str] = field(default_factory=list)

    base_parent_idx: int | None = None

    selection_decision: SelectionDecision | None = None
    patch_intent: PatchIntent | None = None
    patch_verdict: PatchVerdict | None = None

    worktree_path: str = ""
    commit_hash: str | None = None
    train_before_cycle_dir: str | None = None
    train_after_cycle_dir: str | None = None

    sdk_session_selection: SDKSessionInfo | None = None
    sdk_session_patch: SDKSessionInfo | None = None
    sdk_session_reflection: SDKSessionInfo | None = None

    abandoned: bool = False  # True if session failed or verification failed
    accepted: bool | None = None  # Engine's acceptance decision (None = not yet decided)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["selection_decision"] = self.selection_decision.to_dict() if self.selection_decision else None
        d["patch_intent"] = self.patch_intent.to_dict() if self.patch_intent else None
        d["patch_verdict"] = self.patch_verdict.to_dict() if self.patch_verdict else None
        d["sdk_session_selection"] = self.sdk_session_selection.to_dict() if self.sdk_session_selection else None
        d["sdk_session_patch"] = self.sdk_session_patch.to_dict() if self.sdk_session_patch else None
        d["sdk_session_reflection"] = self.sdk_session_reflection.to_dict() if self.sdk_session_reflection else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionNode:
        sel_dec = SelectionDecision.from_dict(d["selection_decision"]) if d.get("selection_decision") else None
        intent = PatchIntent.from_dict(d["patch_intent"]) if d.get("patch_intent") else None
        verdict = PatchVerdict.from_dict(d["patch_verdict"]) if d.get("patch_verdict") else None
        sess_sel = SDKSessionInfo.from_dict(d["sdk_session_selection"]) if d.get("sdk_session_selection") else None
        sess_patch = SDKSessionInfo.from_dict(d["sdk_session_patch"]) if d.get("sdk_session_patch") else None
        sess_refl = SDKSessionInfo.from_dict(d["sdk_session_reflection"]) if d.get("sdk_session_reflection") else None
        return cls(
            idx=d["idx"],
            iteration=d["iteration"],
            created_at=d["created_at"],
            score_train_before=d.get("score_train_before"),
            score_train_after=d.get("score_train_after"),
            score_val=d.get("score_val"),
            val_evaluated=d.get("val_evaluated", False),
            mini_batch_ids=d.get("mini_batch_ids", []),
            base_parent_idx=d.get("base_parent_idx"),
            selection_decision=sel_dec,
            patch_intent=intent,
            patch_verdict=verdict,
            worktree_path=d.get("worktree_path", ""),
            commit_hash=d.get("commit_hash"),
            train_before_cycle_dir=d.get("train_before_cycle_dir"),
            train_after_cycle_dir=d.get("train_after_cycle_dir"),
            sdk_session_selection=sess_sel,
            sdk_session_patch=sess_patch,
            sdk_session_reflection=sess_refl,
            abandoned=d.get("abandoned", False),
            accepted=d.get("accepted"),
        )


# ---------------------------------------------------------------------------
# ScenarioRegistry models
# ---------------------------------------------------------------------------


@dataclass
class ScenarioSnapshot:
    """A single evaluation result for a scenario at a particular iteration."""

    iteration: int
    candidate_idx: int
    score: float  # 0 or 1
    status: str  # "pass" | "fail"
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScenarioSnapshot:
        return cls(**d)


@dataclass
class AttemptedFix:
    """Record of a fix attempt for a scenario."""

    iteration: int
    candidate_idx: int
    approach: str
    result: str  # "fixed" | "not_fixed"
    failure_reason: str | None = None
    prevention_or_next: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttemptedFix:
        return cls(
            iteration=d["iteration"],
            candidate_idx=d["candidate_idx"],
            approach=d["approach"],
            result=d["result"],
            failure_reason=d.get("failure_reason"),
            prevention_or_next=d.get("prevention_or_next"),
        )


@dataclass
class ScenarioEntry:
    """Full history and metadata for a single scenario."""

    scenario_id: str
    task_description: str | None = None
    history: list[ScenarioSnapshot] = field(default_factory=list)
    category: str = "consistently_failing"
    sensitive_to_files: list[str] = field(default_factory=list)
    known_root_causes: list[str] = field(default_factory=list)
    attempted_fixes: list[AttemptedFix] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "task_description": self.task_description,
            "history": [s.to_dict() for s in self.history],
            "category": self.category,
            "sensitive_to_files": self.sensitive_to_files,
            "known_root_causes": self.known_root_causes,
            "attempted_fixes": [f.to_dict() for f in self.attempted_fixes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScenarioEntry:
        return cls(
            scenario_id=d["scenario_id"],
            task_description=d.get("task_description"),
            history=[ScenarioSnapshot.from_dict(s) for s in d.get("history", [])],
            category=d.get("category", "consistently_failing"),
            sensitive_to_files=d.get("sensitive_to_files", []),
            known_root_causes=d.get("known_root_causes", []),
            attempted_fixes=[AttemptedFix.from_dict(f) for f in d.get("attempted_fixes", [])],
        )


# ---------------------------------------------------------------------------
# AccumulatedLessons
# ---------------------------------------------------------------------------


@dataclass
class LessonEntry:
    """A single lesson learned from a patch attempt."""

    pattern: str
    evidence: str
    source_iteration: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LessonEntry:
        return cls(**d)


@dataclass
class AccumulatedLessons:
    """Good and bad patterns accumulated across iterations."""

    good_patterns: list[LessonEntry] = field(default_factory=list)
    bad_patterns: list[LessonEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "good_patterns": [p.to_dict() for p in self.good_patterns],
            "bad_patterns": [p.to_dict() for p in self.bad_patterns],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccumulatedLessons:
        return cls(
            good_patterns=[LessonEntry.from_dict(p) for p in d.get("good_patterns", [])],
            bad_patterns=[LessonEntry.from_dict(p) for p in d.get("bad_patterns", [])],
        )
