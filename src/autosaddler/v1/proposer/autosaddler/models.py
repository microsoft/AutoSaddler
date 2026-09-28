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

    # Timing and richer usage metrics. Defaults keep backward compatibility
    # with session JSON produced before these fields were tracked.
    wall_clock_s: float = 0.0
    duration_ms: int = 0
    duration_api_ms: int = 0
    num_turns: int = 0
    total_cost_usd: float | None = None
    cache_creation_input_tokens: int = 0
    model_usage: dict[str, Any] | None = None
    reasoning_tokens: int = 0
    llm_call_count: int = 0
    usage_event_count: int = 0
    duplicate_usage_event_count: int = 0
    copilot_nano_aiu: float | None = None
    reported_cost_usd: float | None = None
    metered_cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    cost_source: str | None = None
    cost_is_estimate: bool = False
    attempt_count: int = 1
    attempt_accounting_complete: bool = True
    final_attempt_cost_usd: float | None = None

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
    pulled_arm_id: str | None = None  # pattern_id of the pulled arm (None for unseen draws)
    sampling_completed: bool = False

    base_parent_idx: int | None = None

    selection_decision: SelectionDecision | None = None
    patch_intent: PatchIntent | None = None
    patch_verdict: PatchVerdict | None = None

    worktree_path: str = ""
    commit_hash: str | None = None  # Post-patch commit (after Session 1)
    pre_patch_commit: str | None = None  # Pre-patch commit (after Session 0, before Session 1)
    train_before_cycle_dir: str | None = None
    train_after_cycle_dir: str | None = None

    sdk_session_selection: SDKSessionInfo | None = None
    sdk_session_patch: SDKSessionInfo | None = None
    sdk_session_reflection: SDKSessionInfo | None = None
    sdk_session_pattern_extraction: SDKSessionInfo | None = None
    sdk_session_arm_scoring: SDKSessionInfo | None = None
    sdk_session_unseen_scenario_exploration: SDKSessionInfo | None = None

    abandoned: bool = False  # True if session failed or verification failed
    abandon_reason: str | None = None  # e.g. all_pass, session1_failed, verify_failed, resume_incomplete
    accepted: bool | None = None  # Engine's acceptance decision (None = not yet decided)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["selection_decision"] = self.selection_decision.to_dict() if self.selection_decision else None
        d["patch_intent"] = self.patch_intent.to_dict() if self.patch_intent else None
        d["patch_verdict"] = self.patch_verdict.to_dict() if self.patch_verdict else None
        d["sdk_session_selection"] = self.sdk_session_selection.to_dict() if self.sdk_session_selection else None
        d["sdk_session_patch"] = self.sdk_session_patch.to_dict() if self.sdk_session_patch else None
        d["sdk_session_reflection"] = self.sdk_session_reflection.to_dict() if self.sdk_session_reflection else None
        d["sdk_session_pattern_extraction"] = self.sdk_session_pattern_extraction.to_dict() if self.sdk_session_pattern_extraction else None
        d["sdk_session_arm_scoring"] = self.sdk_session_arm_scoring.to_dict() if self.sdk_session_arm_scoring else None
        d["sdk_session_unseen_scenario_exploration"] = self.sdk_session_unseen_scenario_exploration.to_dict() if self.sdk_session_unseen_scenario_exploration else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionNode:
        sel_dec = SelectionDecision.from_dict(d["selection_decision"]) if d.get("selection_decision") else None
        intent = PatchIntent.from_dict(d["patch_intent"]) if d.get("patch_intent") else None
        verdict = PatchVerdict.from_dict(d["patch_verdict"]) if d.get("patch_verdict") else None
        sess_sel = SDKSessionInfo.from_dict(d["sdk_session_selection"]) if d.get("sdk_session_selection") else None
        sess_patch = SDKSessionInfo.from_dict(d["sdk_session_patch"]) if d.get("sdk_session_patch") else None
        sess_refl = SDKSessionInfo.from_dict(d["sdk_session_reflection"]) if d.get("sdk_session_reflection") else None
        sess_pattern = SDKSessionInfo.from_dict(d["sdk_session_pattern_extraction"]) if d.get("sdk_session_pattern_extraction") else None
        sess_arm = SDKSessionInfo.from_dict(d["sdk_session_arm_scoring"]) if d.get("sdk_session_arm_scoring") else None
        sess_unseen = SDKSessionInfo.from_dict(d["sdk_session_unseen_scenario_exploration"]) if d.get("sdk_session_unseen_scenario_exploration") else None
        legacy_sampling_completed = (
            int(d.get("iteration", 0)) == 0
            or bool(d.get("mini_batch_ids"))
            or d.get("patch_verdict") is not None
            or d.get("train_before_cycle_dir") is not None
            or bool(d.get("abandoned", False))
            or d.get("accepted") is not None
        )
        return cls(
            idx=d["idx"],
            iteration=d["iteration"],
            created_at=d["created_at"],
            score_train_before=d.get("score_train_before"),
            score_train_after=d.get("score_train_after"),
            score_val=d.get("score_val"),
            val_evaluated=d.get("val_evaluated", False),
            mini_batch_ids=d.get("mini_batch_ids", []),
            pulled_arm_id=d.get("pulled_arm_id"),
            sampling_completed=d.get(
                "sampling_completed",
                legacy_sampling_completed,
            ),
            base_parent_idx=d.get("base_parent_idx"),
            selection_decision=sel_dec,
            patch_intent=intent,
            patch_verdict=verdict,
            worktree_path=d.get("worktree_path", ""),
            commit_hash=d.get("commit_hash"),
            pre_patch_commit=d.get("pre_patch_commit"),
            train_before_cycle_dir=d.get("train_before_cycle_dir"),
            train_after_cycle_dir=d.get("train_after_cycle_dir"),
            sdk_session_selection=sess_sel,
            sdk_session_patch=sess_patch,
            sdk_session_reflection=sess_refl,
            sdk_session_pattern_extraction=sess_pattern,
            sdk_session_arm_scoring=sess_arm,
            sdk_session_unseen_scenario_exploration=sess_unseen,
            abandoned=d.get("abandoned", False),
            abandon_reason=d.get("abandon_reason"),
            accepted=d.get("accepted"),
        )


@dataclass
class ArmPullRecord:
    """One pull of an arm (pattern) with its per-scenario outcome.

    Unifies the kinds of pull a single arm can have across iterations so that
    BOTH Session 1 (diagnose/patch) and Session 4 (arm scoring) render the same
    history:

    * ``"patched"`` — the pull produced a patch; ``scenario_outcomes`` are the
      before→after impacts and ``node.patch_verdict`` carries the reflections.
    * ``"all_pass_skip"`` — every mini-batch scenario already passed in
      train_before, so the iteration was abandoned with no patch;
      ``scenario_outcomes`` are all ``still_passing``.
    * ``"failed_attempt"`` — the diagnose/patch session or verification failed,
      producing no usable patch; ``scenario_outcomes`` may be empty.
    """

    node: EvolutionNode
    kind: str  # "patched" | "all_pass_skip" | "failed_attempt"
    scenario_outcomes: list[ScenarioImpact] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.node.idx,
            "iteration": self.node.iteration,
            "kind": self.kind,
            "scenario_outcomes": [si.to_dict() for si in self.scenario_outcomes],
        }


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
