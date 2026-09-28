"""PatternRegistry: failure pattern tracking with EMA score computation.

Manages failure patterns as an independent data structure persisted to
``pattern_registry.json`` in the session root. Each pattern is a
symptom-level abstraction of root causes, shared across multiple
(harness, trace, scenario) tuples.

Each pattern is an *arm* of the ActiveSaddler infinite-armed bandit. Arms are
selected from the agent's four-axis learning-progress estimate (see
:meth:`PatternRegistry.compute_agent_scores`). The registry additionally
tracks a mechanical failure-activity proxy:

  Xbar_p <- (1 - eta) * Xbar_p + eta * active_t(p)

an exponential moving average (EMA) over the arm's OWN observation sequence
(Graves et al., 2017; Matiisen et al., 2017). The EMA is *rested*: an arm
that is not observed in an iteration keeps its statistics frozen (no
elapsed-time drift). It is exposed to the agent as diagnostic context (the
``pattern score`` CLI and the Session 4 arm table) and does not drive arm
selection directly.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PatternObservation:
    """A single per-iteration outcome for a pattern (auto-derived by outer loop).

    Unified Bandit Sampling records exactly ONE observation per pattern per
    AutoSaddler iteration, valued by the pattern's POST-patch activity (the
    fraction of the scenarios associated with the pattern and evaluated in the
    iteration's mini-batch that were still tagged to it after the patch. The
    evaluated and still-tagged scenario IDs are retained alongside the fraction
    so downstream analysis has the complete observation. The post-patch state
    is the correct, non-stale estimate of "will this pattern be active when next
    sampled?", since the next sample runs on the just-patched harness. The
    pre-patch outcome (the reward of the past sampling decision) is intentionally
    NOT recorded here: it is not needed for the next sampling decision and would
    double-count the correlated pre/post outcomes.

    ``active=1.0`` means the pattern was tagged in all evaluated scenarios;
    ``active=0.0`` means the pattern was not tagged in any. Intermediate values
    represent the proportion of evaluated scenarios where the pattern was still
    tagged (proportional reward).
    """

    iteration: int  # AutoSaddler iteration number (1-indexed)
    active: float  # proportion of evaluated pattern scenarios still tagged [0,1]
    evaluated_scenario_ids: list[str] = field(default_factory=list)
    tagged_scenario_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatternObservation:
        return cls(
            iteration=d["iteration"],
            active=d.get("active", d.get("failed", 0)),  # backward compat
            evaluated_scenario_ids=list(d.get("evaluated_scenario_ids", [])),
            tagged_scenario_ids=list(d.get("tagged_scenario_ids", [])),
        )


@dataclass
class AgentScore:
    """A single agent learning-progress estimate for a pattern.

    Recorded by the arm-scoring session BEFORE the mini-batch is selected. The
    effective score is always derived from the four axis values (severity,
    fixability, breadth, and inverted side_effect). The legacy ``score`` field
    remains readable for registry compatibility but is not used for selection.
    No EMA is applied to the four-axis score.
    """

    iteration: int  # AutoSaddler iteration number (1-indexed)
    score: float | None = None  # legacy raw score; ignored by current selection
    severity: float | None = None  # how badly the pattern currently fails
    fixability: float | None = None  # reachable by a harness patch?
    breadth: float | None = None  # how widely a fix transfers to held-out tasks
    side_effect: float | None = None  # risk of regressing other patterns
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentScore:
        return cls(
            iteration=d["iteration"],
            score=d.get("score"),
            severity=d.get("severity"),
            fixability=d.get("fixability"),
            breadth=d.get("breadth"),
            side_effect=d.get("side_effect"),
            rationale=d.get("rationale", ""),
        )


@dataclass
class PatternTuple:
    """A (harness, trace, scenario) tuple tagged with a failure pattern."""

    harness_idx: int  # DAG node idx (foreign key to EvolutionDAG)
    trace_dir: str  # cycle_dir path containing execution traces
    scenario_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatternTuple:
        return cls(
            harness_idx=d["harness_idx"],
            trace_dir=d["trace_dir"],
            scenario_id=d["scenario_id"],
        )


@dataclass
class PatternEvidence:
    """One root-cause value recorded by a specific harness observation."""

    harness_idx: int
    trace_dir: str
    scenario_id: str
    root_cause: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatternEvidence:
        return cls(
            harness_idx=int(d["harness_idx"]),
            trace_dir=str(d.get("trace_dir", "")),
            scenario_id=str(d["scenario_id"]),
            root_cause=str(d.get("root_cause", "")),
        )


@dataclass
class FailurePattern:
    """A failure pattern: symptom-level label shared across tuples."""

    pattern_id: str
    label: str  # Symptom-level abstract description
    tuples: list[PatternTuple] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)  # scenario_id -> root_cause
    evidence_baseline: dict[str, str] = field(default_factory=dict)
    evidence_history: list[PatternEvidence] = field(default_factory=list)
    observations: list[PatternObservation] = field(default_factory=list)
    agent_scores: list[AgentScore] = field(default_factory=list)
    created_at: str = ""
    created_iteration: int = 0
    last_observed_iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "label": self.label,
            "tuples": [t.to_dict() for t in self.tuples],
            "evidence": self.evidence,
            "evidence_baseline": self.evidence_baseline,
            "evidence_history": [item.to_dict() for item in self.evidence_history],
            "observations": [o.to_dict() for o in self.observations],
            "agent_scores": [a.to_dict() for a in self.agent_scores],
            "created_at": self.created_at,
            "created_iteration": self.created_iteration,
            "last_observed_iteration": self.last_observed_iteration,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FailurePattern:
        evidence = d.get("evidence", {})
        has_evidence_tracking = (
            "evidence_baseline" in d or "evidence_history" in d
        )
        return cls(
            pattern_id=d["pattern_id"],
            label=d["label"],
            tuples=[PatternTuple.from_dict(t) for t in d.get("tuples", [])],
            evidence=evidence,
            evidence_baseline=d.get(
                "evidence_baseline",
                {} if has_evidence_tracking else dict(evidence),
            ),
            evidence_history=[
                PatternEvidence.from_dict(item)
                for item in d.get("evidence_history", [])
            ],
            observations=[PatternObservation.from_dict(o) for o in d.get("observations", [])],
            agent_scores=[AgentScore.from_dict(a) for a in d.get("agent_scores", [])],
            created_at=d.get("created_at", ""),
            created_iteration=int(d.get("created_iteration", 0)),
            last_observed_iteration=d.get("last_observed_iteration", 0),
        )


# ---------------------------------------------------------------------------
# PatternRegistry
# ---------------------------------------------------------------------------


class PatternRegistry:
    """Registry of failure patterns with persistent EMA and agent scores.

    Independent data structure from EvolutionDAG. References DAG node indices
    via PatternTuple.harness_idx (foreign key relationship).
    """

    def __init__(self, session_root: str) -> None:
        self.session_root = session_root
        self._json_path = Path(session_root) / "pattern_registry.json"
        self._gz_path = self._json_path.with_suffix(".json.gz")

        self.metadata: dict[str, Any] = {
            "session_root": session_root,
            "total_patterns": 0,
            "last_updated": "",
        }
        self.patterns: dict[str, FailurePattern] = {}

        # Reverse index: scenario_id -> set of pattern_ids
        self._scenario_to_patterns: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Serialize the registry to gzip-compressed JSON (atomic write)."""
        self.metadata["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.metadata["total_patterns"] = len(self.patterns)

        data = {
            "metadata": self.metadata,
            "patterns": {pid: p.to_dict() for pid, p in self.patterns.items()},
        }

        self._json_path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._gz_path.parent), suffix=".json.gz.tmp",
        )
        try:
            os.close(fd)
            with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=3) as gz:
                json.dump(data, gz, ensure_ascii=False)
            os.replace(tmp_path, str(self._gz_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info("PatternRegistry saved: %d patterns", len(self.patterns))

    def load(self) -> None:
        """Load the registry from disk."""
        path = self._gz_path if self._gz_path.exists() else self._json_path
        if not path.exists():
            logger.info("No existing pattern registry found at %s", self.session_root)
            return

        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
        except Exception as exc:
            logger.exception("Failed to load pattern registry from %s", path)
            raise RuntimeError(
                f"Failed to load pattern registry from {path}"
            ) from exc

        self.metadata = data.get("metadata", self.metadata)
        self.metadata["schema_version"] = 2
        self.patterns = {
            pid: FailurePattern.from_dict(pdata)
            for pid, pdata in data.get("patterns", {}).items()
        }
        for pattern in self.patterns.values():
            scores_by_iteration: dict[int, AgentScore] = {}
            for score in pattern.agent_scores:
                scores_by_iteration[score.iteration] = score
            pattern.agent_scores = [
                scores_by_iteration[iteration]
                for iteration in sorted(scores_by_iteration)
            ]

        # Rebuild reverse index
        self._rebuild_reverse_index()

        logger.info("PatternRegistry loaded: %d patterns", len(self.patterns))

    def _rebuild_reverse_index(self) -> None:
        """Rebuild the scenario_id -> pattern_ids reverse index."""
        self._scenario_to_patterns = {}
        for pid, pattern in self.patterns.items():
            for t in pattern.tuples:
                self._scenario_to_patterns.setdefault(t.scenario_id, set()).add(pid)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def register(self, label: str, created_iteration: int = 0) -> str:
        """Register a new failure pattern. Returns the new pattern_id."""
        pattern_id = str(uuid4())[:8]
        # Ensure uniqueness
        while pattern_id in self.patterns:
            pattern_id = str(uuid4())[:8]

        pattern = FailurePattern(
            pattern_id=pattern_id,
            label=label,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_iteration=created_iteration,
        )
        self.patterns[pattern_id] = pattern
        logger.info("Registered pattern %s: %s", pattern_id, label)
        return pattern_id

    def tag(
        self,
        pattern_id: str,
        harness_idx: int,
        trace_dir: str,
        scenario_id: str,
        root_cause: str = "",
    ) -> None:
        """Tag a (harness, trace, scenario) tuple with a pattern."""
        if pattern_id not in self.patterns:
            raise KeyError(f"Pattern '{pattern_id}' not found in registry")

        pattern = self.patterns[pattern_id]

        def record_evidence() -> None:
            if not root_cause:
                return
            pattern.evidence[scenario_id] = root_cause
            pattern.evidence_history.append(
                PatternEvidence(
                    harness_idx=harness_idx,
                    trace_dir=trace_dir,
                    scenario_id=scenario_id,
                    root_cause=root_cause,
                )
            )

        # Avoid duplicate tuples
        for existing in pattern.tuples:
            if (
                existing.harness_idx == harness_idx
                and existing.trace_dir == trace_dir
                and existing.scenario_id == scenario_id
            ):
                # Already tagged, just update evidence if provided
                record_evidence()
                return

        new_tuple = PatternTuple(
            harness_idx=harness_idx,
            trace_dir=trace_dir,
            scenario_id=scenario_id,
        )
        pattern.tuples.append(new_tuple)

        record_evidence()

        # Update reverse index
        self._scenario_to_patterns.setdefault(scenario_id, set()).add(pattern_id)

    def observe(
        self,
        pattern_id: str,
        iteration: int,
        active: float,
        evaluated_scenario_ids: list[str] | None = None,
        tagged_scenario_ids: list[str] | None = None,
        **_kwargs: Any,
    ) -> None:
        """Record an iteration outcome for a pattern.

        Called by the outer loop after Session 3 completes. Records the
        proportion of the scenarios associated with the pattern and evaluated
        in the iteration's mini-batch that were still tagged (0.0 = fully
        resolved, 1.0 = fully active), together with both scenario-ID sets.

        Legacy keyword arguments (version_weight, failed, scenario_id,
        harness_idx) are accepted but ignored for backward compatibility.
        """
        if pattern_id not in self.patterns:
            raise KeyError(f"Pattern '{pattern_id}' not found in registry")

        pattern = self.patterns[pattern_id]
        obs = PatternObservation(
            iteration=iteration,
            active=active,
            evaluated_scenario_ids=sorted(set(evaluated_scenario_ids or [])),
            tagged_scenario_ids=sorted(set(tagged_scenario_ids or [])),
        )
        pattern.observations.append(obs)
        pattern.last_observed_iteration = max(
            pattern.last_observed_iteration, iteration,
        )

    # ------------------------------------------------------------------
    # Agent scoring
    # ------------------------------------------------------------------

    def record_agent_score(
        self,
        pattern_id: str,
        iteration: int,
        severity: float | None = None,
        fixability: float | None = None,
        breadth: float | None = None,
        side_effect: float | None = None,
        rationale: str = "",
    ) -> None:
        """Record the agent's learning-progress estimate for a pattern.

        Called by the arm-scoring session (Session 4) BEFORE the
        mini-batch is selected. All four axes are required; phi is computed as
        ``(severity + fixability + breadth + (1 - side_effect)) / 4``.
        """
        if pattern_id not in self.patterns:
            raise KeyError(f"Pattern '{pattern_id}' not found in registry")

        axes = (severity, fixability, breadth, side_effect)
        if any(value is None for value in axes):
            raise ValueError(
                "severity, fixability, breadth, and side_effect are required"
            )

        def _clamp(v: float | None) -> float | None:
            return None if v is None else max(0.0, min(1.0, float(v)))

        agent_score = AgentScore(
            iteration=iteration,
            score=None,
            severity=_clamp(severity),
            fixability=_clamp(fixability),
            breadth=_clamp(breadth),
            side_effect=_clamp(side_effect),
            rationale=rationale,
        )
        scores = self.patterns[pattern_id].agent_scores
        scores[:] = [score for score in scores if score.iteration != iteration]
        scores.append(agent_score)
        scores.sort(key=lambda score: score.iteration)

    def get_agent_score(
        self,
        pattern_id: str,
        iteration: int,
    ) -> AgentScore | None:
        """Return the score recorded for one exact pattern iteration."""
        pattern = self.patterns.get(pattern_id)
        if pattern is None:
            return None
        return next(
            (
                score
                for score in reversed(pattern.agent_scores)
                if score.iteration == iteration
            ),
            None,
        )

    @staticmethod
    def _four_axis_score(a: AgentScore) -> float | None:
        """Mean of the four axes with side-effect inverted (risk -> benefit).

        ``phi = (severity + fixability + breadth + (1 - side_effect)) / 4``,
        ``side_effect`` is a RISK (higher = worse), so it enters as
        ``1 - side_effect``. Returns ``None`` unless all four axes are present.
        """
        if any(
            value is None
            for value in (a.severity, a.fixability, a.breadth, a.side_effect)
        ):
            return None
        assert a.severity is not None
        assert a.fixability is not None
        assert a.breadth is not None
        assert a.side_effect is not None
        return (
            a.severity
            + a.fixability
            + a.breadth
            + (1.0 - a.side_effect)
        ) / 4.0

    def compute_agent_scores(
        self,
        breakdown: dict[str, dict] | None = None,
        default: float = 0.0,
        eta: float = 0.3,
        required_iteration: int | None = None,
    ) -> dict[str, float]:
        """Return ``{pattern_id: phi_t(p)}`` from the latest agent estimate.

        Agent scoring: phi is the mean of severity, fixability,
        breadth, and inverted side-effect (no EMA smoothing). Patterns owning
        no scenarios are not pullable arms
        and score 0.0. Patterns with tuples but no agent score yet fall back to
        a conservative ``default=0.0`` — an un-rated arm must not be rewarded
        with high priority; the sampler's min-probability floor keeps it
        eligible for a later re-check.

        When ``breakdown`` is requested, it also includes the mechanical
        failure-activity observations and EMA. Those fields are diagnostic
        context only; ``score`` remains the effective agent estimate used for
        arm selection.

        When ``required_iteration`` is provided, only a score recorded for that
        exact iteration is eligible. This prevents a stale score from a prior
        prepared harness from influencing the current arm selection.
        """
        activity_breakdown: dict[str, dict] = {}
        self.compute_scores(eta=eta, breakdown=activity_breakdown)
        scores: dict[str, float] = {}
        for pid, pattern in self.patterns.items():
            activity = activity_breakdown.get(pid, {})
            if not pattern.tuples:
                scores[pid] = 0.0
                if breakdown is not None:
                    breakdown[pid] = {
                        "label": pattern.label,
                        "scenarios": self.get_scenarios_for_pattern(pid),
                        "observations": activity.get("observations", []),
                        "num_observations": activity.get("num_observations", 0),
                        "ema": activity.get("ema", 0.0),
                        "score": 0.0,
                        "num_agent_scores": len(pattern.agent_scores),
                    }
                continue
            latest = (
                self.get_agent_score(pid, required_iteration)
                if required_iteration is not None
                else (pattern.agent_scores[-1] if pattern.agent_scores else None)
            )
            if latest is None:
                scores[pid] = default
            else:
                four_axis_score = self._four_axis_score(latest)
                scores[pid] = default if four_axis_score is None else four_axis_score
            if breakdown is not None:
                breakdown[pid] = {
                    "label": pattern.label,
                    "scenarios": self.get_scenarios_for_pattern(pid),
                    "observations": activity.get("observations", []),
                    "num_observations": activity.get("num_observations", 0),
                    "ema": activity.get("ema", 0.0),
                    "score": scores[pid],
                    "severity": latest.severity if latest else None,
                    "fixability": latest.fixability if latest else None,
                    "breadth": latest.breadth if latest else None,
                    "side_effect": latest.side_effect if latest else None,
                    "rationale": latest.rationale if latest else "",
                    "num_agent_scores": len(pattern.agent_scores),
                }
        return scores

    # ------------------------------------------------------------------
    # Failure-activity EMA (diagnostic context)
    # ------------------------------------------------------------------

    def compute_scores(
        self,
        current_iter: int = 0,
        eta: float = 0.3,
        breakdown: dict[str, dict] | None = None,
        **_kwargs: Any,
    ) -> dict[str, float]:
        """Compute the failure-activity EMA for all patterns.

        ``phi_t(p) = Xbar_p``, an exponential moving average of the pattern's
        failure activity over its OWN observation sequence::

            Xbar_p <- (1 - eta) * Xbar_p + eta * active_t(p)

        The bandit is *rested*: the EMA folds only the pattern's recorded
        observations, with NO elapsed-time discounting, so an arm that was not
        observed in an iteration keeps its statistics frozen. A brand-new
        arm (created by a failure, with no observations yet) is seeded at 1.0,
        reflecting the failure that created it.

        Args:
            current_iter: Accepted for API compatibility / display context;
                it does NOT affect the EMA (rested bandit, no time discount).
            eta: EMA smoothing factor (0 < eta <= 1). Higher = faster
                adaptation to recent observations.
            breakdown: Optional dict populated with per-pattern score
                components for logging/analysis.

        Returns:
            Dict mapping pattern_id to its score ``phi_t(p)``.
        """
        scores: dict[str, float] = {}

        for pid, pattern in self.patterns.items():
            ordered = sorted(pattern.observations, key=lambda o: o.iteration)

            if not pattern.tuples:
                # An untagged pattern owns no scenarios -> not a pullable arm.
                scores[pid] = 0.0
                if breakdown is not None:
                    breakdown[pid] = {
                        "label": pattern.label,
                        "scenarios": self.get_scenarios_for_pattern(pid),
                        "observations": [o.to_dict() for o in ordered],
                        "ema": 0.0,
                        "num_observations": len(ordered),
                        "score": 0.0,
                    }
                continue

            # EMA over the arm's own observation sequence, seeded at 1.0.
            ema = 1.0
            for obs in ordered:
                ema = (1.0 - eta) * ema + eta * obs.active
            scores[pid] = ema

            if breakdown is not None:
                breakdown[pid] = {
                    "label": pattern.label,
                    "scenarios": self.get_scenarios_for_pattern(pid),
                    "observations": [o.to_dict() for o in ordered],
                    "ema": ema,
                    "num_observations": len(ordered),
                    "score": ema,
                }

        return scores

    def compute_score_for_pattern(
        self,
        pattern_id: str,
        current_iter: int = 0,
        eta: float = 0.3,
        **_kwargs: Any,
    ) -> float:
        """Compute the EMA score ``phi_t(p)`` for a single pattern."""
        scores = self.compute_scores(current_iter, eta)
        return scores.get(pattern_id, 0.0)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_scenarios_for_pattern(self, pattern_id: str) -> list[str]:
        """Return all scenario IDs tagged with a given pattern."""
        if pattern_id not in self.patterns:
            return []
        return list(
            dict.fromkeys(
                item.scenario_id for item in self.patterns[pattern_id].tuples
            )
        )

    def get_patterns_for_scenario(self, scenario_id: str) -> list[str]:
        """Return all pattern IDs tagged to a given scenario."""
        return list(self._scenario_to_patterns.get(scenario_id, set()))

    def get_all_tagged_scenario_ids(self) -> set[str]:
        """Return all scenario IDs that have at least one pattern tag."""
        return set(self._scenario_to_patterns.keys())

    def list_patterns(self) -> list[FailurePattern]:
        """Return all patterns sorted by creation time."""
        return sorted(self.patterns.values(), key=lambda p: p.created_at)

    def get_pattern(self, pattern_id: str) -> FailurePattern | None:
        """Get a specific pattern by ID."""
        return self.patterns.get(pattern_id)

    def get_top_patterns_by_score(
        self,
        current_iter: int = 0,
        top_k: int = 5,
        eta: float = 0.3,
        **_kwargs: Any,
    ) -> list[tuple[str, float]]:
        """Return top-k patterns by EMA score, descending."""
        scores = self.compute_scores(current_iter, eta)
        sorted_patterns = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_patterns[:top_k]
