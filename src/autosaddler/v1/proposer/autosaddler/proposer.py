"""EvolutionDAG LR Scheduler v2 Proposer: refactored prompt architecture.

Implements the ``ProposeNewCandidate`` protocol using a three-session
SDK agent approach per iteration:
- Session 0: Candidate selection (determine initial patch base)
- Session 1: Diagnose failing scenarios + apply patches
- Session 2: Reflect on initial/re-evaluation results

Key differences from v1:
- CLAUDE.md is the central always-on document (env, benchmark, pipeline)
- Session prompts are separate from skills
- Skills contain only methodology (no benchmark-specific content)
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import git

from autosaddler.v1.core.data_loader import DataId
from autosaddler.v1.core.state import GEPAState
from autosaddler.v1.proposer.autosaddler.artifact_paths import iteration_artifact_path
from autosaddler.v1.proposer.autosaddler.dag import EvolutionDAG
from autosaddler.v1.proposer.autosaddler.evaluator import (
    compute_all_scenario_impacts,
    compute_pass_rate,
    parse_evaluation_results,
)
from autosaddler.v1.proposer.autosaddler.lesson_manager import (
    update_lessons,
    update_scenario_registry,
    update_scenario_registry_from_reflections,
)
from autosaddler.v1.proposer.autosaddler.models import (
    EvolutionNode,
    PatchIntent,
    PatchVerdict,
    SDKSessionInfo,
)
from autosaddler.v1.proposer.autosaddler.prompt_builder import (
    build_arm_scoring_prompt,
    build_session0_prompt,
    build_session1_prompt,
    build_session2_prompt,
    build_session3_prompt,
    build_skill_prefix,
    build_unseen_scenario_exploration_prompt,
    install_evo_dag_cli,
    install_pattern_cli,
    install_prompts_and_skills,
    resolve_prompt_bundle,
)
from autosaddler.v1.proposer.autosaddler.strategy import (
    UNSEEN_SCENARIO_EXPLORATION_SESSION,
    StrategySpec,
    canonicalize_sampling_strategy,
    resolve_strategy,
)
from autosaddler.v1.proposer.base import CandidateProposal, ProposeNewCandidate
from autosaddler.v1.sdk_session import SdkConfig, aggregate_model_usage
from autosaddler.v1.strategies.batch_sampler import (
    ActiveSaddlerBanditSampler,
    EpochShuffledBatchSampler,
)

if TYPE_CHECKING:
    from autosaddler.v1.proposer.autosaddler.pattern_registry import PatternRegistry

logger = logging.getLogger(__name__)


class IncompleteIterationError(RuntimeError):
    """Raised when resume is attempted from a non-transactional boundary."""


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------

def _run_async(coro):  # noqa: ANN001, ANN202
    """Run an async coroutine from sync code, handling nested event loops."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    import concurrent.futures

    result = None
    exception = None

    def _thread_target():
        nonlocal result, exception
        try:
            result = asyncio.run(coro)
        except Exception as e:
            exception = e

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_thread_target).result()

    if exception is not None:
        raise exception
    return result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------



@dataclass
class EvolutionDAGConfig:
    """Configuration for the EvolutionDAG proposer."""

    # SDK settings
    claude_agent_sdk_model: str = "Claude Opus 4.6"
    copilot_model: str = "claude-opus-4.6"
    diagnosis_patch_timeout: float = 18000.0  # 5 hours for thorough analysis
    reflection_timeout: float = 3600.0  # 1 hour for reflection
    candidate_selection_timeout: float = 1800.0  # 30 min for candidate selection

    # Mini-batch
    train_minibatch_size: int = 10
    seed: int = 42

    # SDK backend
    sdk_config: SdkConfig = field(default_factory=SdkConfig)

    # Phase transition: capability → steering
    capability_phase_iterations: int = 0
    capability_phase_epochs: int = 1
    # Transition mode:
    #   "iterations"    → use capability_phase_iterations / capability_phase_epochs
    #   "full_coverage" → switch to steering only AFTER every training scenario
    #                     has been observed (sampled into a mini-batch) at least
    #                     once. Robust to the dynamic sampler re-drawing the same
    #                     scenarios; the transition point is recorded in the DAG.
    capability_transition_mode: str = "iterations"
    # full_coverage safety valve: force the transition once meta-iteration
    # reaches this value even if coverage is incomplete (0 = disabled).
    capability_phase_max_iterations: int = 0

    # Session 0 control
    skip_session0: bool = False  # Skip candidate selection for first few iterations

    # ActiveSaddler: infinite-armed bandit curriculum.
    #   sampling_strategy:
    #     "autosaddler" (passive epoch shuffle) | "activesaddler" (agent
    #     scoring + agent-decided arm creation)
    sampling_strategy: str = "autosaddler"
    eta: float = 0.3  # EMA smoothing for the failure-activity context shown to the agent
    softmax_temperature: float = 0.15  # temperature tau for stochastic arm sampling (> 0)
    min_prob: float = 0.02  # per-arm minimum sampling probability epsilon in [0, 1)
    pattern_extraction_timeout: float = 3600.0  # 1 hour for pattern extraction
    arm_scoring_timeout: float = 3600.0  # 1 hour for the agent arm-scoring session

    def __post_init__(self) -> None:
        self.sampling_strategy = canonicalize_sampling_strategy(
            self.sampling_strategy
        ).value
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("eta must be in (0, 1]")
        if self.softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        if not 0.0 <= self.min_prob < 1.0:
            raise ValueError("min_prob must be in [0, 1)")

    @property
    def strategy(self) -> StrategySpec:
        return resolve_strategy(self.sampling_strategy)

    @property
    def active_model(self) -> str:
        """Return the model name for the active SDK backend."""
        if self.sdk_config.backend == "copilot":
            return self.sdk_config.copilot_model or self.copilot_model
        return self.claude_agent_sdk_model

    @property
    def pattern_sampling_enabled(self) -> bool:
        """True when failure-pattern (arm) based sampling is active (``"activesaddler"``)."""
        return self.strategy.pattern_sampling

    @property
    def agent_scoring_enabled(self) -> bool:
        """True when arm scores phi_t(p) are produced by the agent (Session 4)."""
        return self.strategy.agent_scoring

    @property
    def agent_arm_creation_enabled(self) -> bool:
        """True when the agent decides pull-vs-draw (Session 3.5).

        The agent chooses pull-or-draw before the mini-batch is drawn.
        """
        return self.strategy.agent_arm_creation


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


class AutoSaddlerProposer(ProposeNewCandidate[DataId]):
    """EvolutionDAG proposer v2 with refactored prompt architecture.

    Three sessions per iteration:
    0. Candidate Selection: Analyze prior candidates + select base code
    1. Diagnose + Patch: Diagnose failures + apply patches
    2. Reflection: Analyze initial/re-evaluation results + record learnings

    Between sessions, the outer loop handles:
    - Mini-batch sampling and evaluation
    - Initial/re-evaluation comparison
    - DAG updates (verdict, lessons, scenario registry)
    """

    def __init__(
        self,
        *,
        logger: Any,
        trainset: list,
        adapter: Any,  # MetaAREAdapter
        config: EvolutionDAGConfig,
        experiment_tracker: Any | None = None,
    ) -> None:
        self._logger = logger
        self.trainset = trainset
        self._adapter = adapter
        self._config = config
        self._experiment_tracker = experiment_tracker

        self._pending_proposals: deque[CandidateProposal] = deque()
        self._meta_iteration = 0

        # Scenario ID mapping: DataId (int index as str) ↔ full scenario name
        # Pattern registry uses full names; GEPA DataLoader uses integer indices.
        self._idx_to_scenario: dict[str, str] = {}
        self._scenario_to_idx: dict[str, int] = {}
        for i, inst in enumerate(trainset):
            sid = inst.scenario_id
            self._idx_to_scenario[str(i)] = sid
            self._scenario_to_idx[sid] = i

        # DAG instance — initialized on first propose()
        self._dag: EvolutionDAG | None = None

        # PatternRegistry — initialized on first propose() alongside DAG
        self._pattern_registry: PatternRegistry | None = None

        # Mini-batch sampler
        mbs = config.train_minibatch_size
        if mbs > 0:
            if config.pattern_sampling_enabled:
                # ActiveSaddlerBanditSampler requires PatternRegistry (deferred init)
                self._batch_sampler: EpochShuffledBatchSampler | ActiveSaddlerBanditSampler | None = None
                self._deferred_score_sampler = True
            else:
                self._batch_sampler = EpochShuffledBatchSampler(
                    minibatch_size=mbs, rng=random.Random(config.seed),
                )
                self._deferred_score_sampler = False
        else:
            self._batch_sampler = None
            self._deferred_score_sampler = False

        # Track worktrees: candidate idx → path
        self._worktree_map: dict[int, Path] = {}

    # ------------------------------------------------------------------
    # DAG ↔ State index mapping
    # ------------------------------------------------------------------

    def _resolve_state_parent_idx(
        self, dag: EvolutionDAG, dag_idx: int, state: GEPAState,
    ) -> int:
        """Map a DAG node idx to the corresponding state.program_candidates index."""
        visited: set[int] = set()
        cur = dag_idx
        while cur in dag.nodes and cur not in visited:
            visited.add(cur)
            worktree = dag.nodes[cur].worktree_path
            for state_idx, candidate in enumerate(state.program_candidates):
                if candidate.get("__autosaddler_worktree__", "") == worktree:
                    return state_idx
            parent = dag.nodes[cur].base_parent_idx
            if parent is None:
                break
            cur = parent
        return 0

    # ------------------------------------------------------------------
    # Val score sync from engine state
    # ------------------------------------------------------------------

    def _sync_val_scores_from_state(
        self, dag: EvolutionDAG, state: GEPAState,
    ) -> None:
        """Write back val scores and acceptance status from the engine state to DAG nodes."""
        if not state.program_candidates:
            return

        val_scores = state.program_full_scores_val_set

        worktree_to_state_idx: dict[str, int] = {}
        for state_idx, candidate in enumerate(state.program_candidates):
            wt = candidate.get("__autosaddler_worktree__", "")
            if wt:
                worktree_to_state_idx[wt] = state_idx

        accepted_worktrees = set(worktree_to_state_idx.keys())

        updated = False
        for node_idx, node in dag.nodes.items():
            # Sync val scores
            if not node.val_evaluated and node.worktree_path:
                state_idx = worktree_to_state_idx.get(node.worktree_path)
                if state_idx is not None and state_idx < len(val_scores):
                    dag.update_val_score(node_idx, val_scores[state_idx])
                    updated = True
                    logger.info(
                        "Synced val score for DAG node %d (state idx %d): %.4f",
                        node_idx, state_idx, val_scores[state_idx],
                    )

            # Sync acceptance status: if node has a verdict but no acceptance
            # decision yet, determine from engine state
            if (
                node.accepted is None
                and node.patch_verdict is not None
                and node.worktree_path
                and not node.abandoned
            ):
                is_accepted = node.worktree_path in accepted_worktrees
                dag.set_accepted(node_idx, is_accepted)
                updated = True
                logger.info(
                    "Synced acceptance for DAG node %d: %s",
                    node_idx, "accepted" if is_accepted else "rejected",
                )

        if updated:
            dag.save()

    # ------------------------------------------------------------------
    # Phase determination
    # ------------------------------------------------------------------

    def _get_phase_for_iteration(self, iteration: int) -> str:
        """Determine the phase (capability/steering) for a given iteration."""
        if self._config.capability_transition_mode == "full_coverage":
            # Steering begins only after every training scenario has been
            # observed at least once. capability_end is the last capability
            # iteration (the one that completed coverage); None while still
            # covering, so the run stays in capability until then.
            capability_end = self._capability_end_iteration()
            if capability_end is None:
                return "capability"
            return "capability" if iteration <= capability_end else "steering"

        # Legacy iteration/epoch schedule (unchanged).
        if self._config.capability_phase_iterations > 0:
            capability_iterations = self._config.capability_phase_iterations
        else:
            trainset_size = len(self.trainset)
            mbs = self._config.train_minibatch_size
            iterations_per_epoch = (trainset_size + mbs - 1) // mbs
            capability_iterations = self._config.capability_phase_epochs * iterations_per_epoch
        return "capability" if iteration <= capability_iterations else "steering"

    def _capability_end_iteration(self) -> int | None:
        """Last capability iteration (T) under ``full_coverage`` mode.

        Returns the recorded transition iteration once every training scenario
        has been observed (or the safety valve fired), else ``None``. Stored in
        the DAG metadata so it is monotonic and survives process restarts.
        """
        if self._dag is None:
            return None
        rec = self._dag.metadata.get("capability_end_iteration")
        return int(rec) if rec is not None else None

    def _maybe_record_capability_end(self, dag: EvolutionDAG) -> None:
        """Record the capability→steering transition iteration (once).

        Under ``full_coverage`` mode, records ``capability_end_iteration`` in
        the DAG metadata as soon as the union of every node's ``mini_batch_ids``
        covers the whole training set (or the ``capability_phase_max_iterations``
        safety valve fires). The record is sticky: once set it never changes, so
        the phase is monotonic and consistent for historical nodes. The current
        iteration's node is already in the DAG, so its mini-batch counts.
        """
        if self._config.capability_transition_mode != "full_coverage":
            return
        if dag.metadata.get("capability_end_iteration") is not None:
            return  # already recorded — sticky

        observed: set[str] = set()
        for n in dag.nodes.values():
            observed.update(n.mini_batch_ids or [])
        trainset_ids = {str(i) for i in range(len(self.trainset))}
        covered = len(observed & trainset_ids)
        total = len(trainset_ids)

        full = total > 0 and trainset_ids <= observed
        cap = self._config.capability_phase_max_iterations
        forced = cap > 0 and self._meta_iteration >= cap

        if full or forced:
            dag.metadata["capability_end_iteration"] = self._meta_iteration
            dag.save()
            reason = "full coverage" if full else "max-iteration fallback"
            self._logger.log(
                f"[phase] capability_end = iter {self._meta_iteration} "
                f"({reason}; observed {covered}/{total} train scenarios). "
                f"Steering starts next iteration."
            )
        else:
            self._logger.log(
                f"[phase] coverage {covered}/{total} train scenarios observed "
                f"— staying in capability (iter {self._meta_iteration})"
            )

    # ------------------------------------------------------------------
    # Deferred Session 2 (reflection from previous iteration)
    # ------------------------------------------------------------------

    def _run_deferred_session2(
        self, dag: EvolutionDAG, state: GEPAState,
    ) -> None:
        """Run Session 2 for the previous iteration, if pending.

        Session 2 is deferred to the start of the next iteration so that
        the AutoSaddler engine's dev-set evaluation has completed and the current
        candidate's dev score is available for generalization analysis.
        """
        # Find nodes that have a verdict but no reflections yet
        candidates_needing_reflection: list[int] = []
        for idx, node in dag.nodes.items():
            if node.iteration == 0:
                continue  # seed has no patch to reflect on
            if node.patch_verdict is None:
                continue  # not yet evaluated
            if node.patch_verdict.reflections:
                continue  # already reflected
            if node.sdk_session_reflection is not None:
                continue  # reflection session already ran
            candidates_needing_reflection.append(idx)

        if not candidates_needing_reflection:
            return

        session_root = str(self._adapter._session_root)

        for idx in candidates_needing_reflection:
            node = dag.nodes[idx]
            worktree = Path(node.worktree_path) if node.worktree_path else None
            if worktree is None or not worktree.exists():
                logger.warning(
                    "Skipping deferred Session 2 for C%d: worktree not found", idx,
                )
                continue

            self._logger.log(
                f"Running deferred Session 2 (Reflection) for C{idx} "
                f"(iter {node.iteration})..."
            )

            all_worktrees = {
                i: str(p) for i, p in self._worktree_map.items()
            }

            scenario_impacts = (
                node.patch_verdict.scenario_impacts if node.patch_verdict else []
            )

            session2_prompt = build_session2_prompt(
                node, scenario_impacts, all_worktrees, dag=dag,
                phase=self._get_phase_for_iteration(node.iteration),
                sampling_strategy=self._config.sampling_strategy,
            )

            cli_env = install_evo_dag_cli(session_root, str(worktree))

            reflection_output_dir = (
                node.train_after_cycle_dir
                or node.train_before_cycle_dir
            )

            session2_result = self._run_sdk_session(
                worktree_path=worktree,
                prompt=build_skill_prefix(
                    session=2,
                    phase=self._get_phase_for_iteration(node.iteration),
                    sampling_strategy=self._config.sampling_strategy,
                ) + session2_prompt,
                model=self._config.active_model,
                timeout=self._config.reflection_timeout,
                extra_env=cli_env,
                session_type="reflection",
                iteration=node.iteration,
                candidate_idx=idx,
                artifact_dir=reflection_output_dir,
            )
            if session2_result and not reflection_output_dir:
                logger.warning(
                    "C%d has no cycle_dir for reflection JSON — "
                    "skipping session info extraction",
                    idx,
                )
            if session2_result and reflection_output_dir:
                session2_info = self._extract_session_info(
                    session2_result, self._config.active_model,
                    self._config.reflection_timeout,
                    reflection_output_dir, "reflection",
                    node.iteration, idx,
                )
            else:
                session2_info = None

            # Reload DAG to pick up CLI changes (e.g. pending reflections
            # written by evo-dag update-reflection during the session),
            # then set reflection session info AFTER reload so it isn't
            # overwritten.
            dag.load()
            node = dag.nodes[idx]
            if session2_info:
                dag.set_sdk_session_info(idx, reflection=session2_info)
            reflections = dag.get_pending_reflections(idx)

            if reflections and node.patch_verdict:
                lessons_learned = []
                for r in reflections:
                    if r.status_change == "fixed":
                        lessons_learned.append(f"[GOOD] {r.explanation}")
                    elif r.status_change == "regressed":
                        msg = f"[BAD] {r.explanation}"
                        if r.prevention_or_next:
                            msg += f" → {r.prevention_or_next}"
                        lessons_learned.append(msg)
                    elif r.status_change == "still_failing":
                        msg = f"[INEFFECTIVE] {r.explanation}"
                        if r.prevention_or_next:
                            msg += f" | Next: {r.prevention_or_next}"
                        lessons_learned.append(msg)

                node.patch_verdict.reflections = reflections
                node.patch_verdict.lessons_learned = lessons_learned
                update_lessons(dag, node, node.patch_verdict)
                update_scenario_registry_from_reflections(dag, node)

            dag.save()
            self._logger.log(f"Deferred Session 2 for C{idx} completed")

    # ------------------------------------------------------------------
    # Session 3: Pattern Extraction (ActiveSaddler)
    # ------------------------------------------------------------------

    def _run_session3_pattern_extraction(
        self, dag: EvolutionDAG, state: GEPAState,
    ) -> None:
        """Run Session 3 (Pattern Extraction) after reflection completes.

        Extracts failure patterns from diagnosis/reflection results and tags
        (harness, trace, scenario) tuples. Only runs when ActiveSaddler
        pattern-based sampling is enabled ("activesaddler").
        """
        if not self._config.pattern_sampling_enabled:
            return

        registry = self._ensure_pattern_registry()

        # Find nodes that have completed reflection but no pattern extraction yet
        candidates_needing_extraction: list[int] = []
        for idx, node in dag.nodes.items():
            if node.iteration == 0:
                continue  # seed
            if node.abandoned:
                continue
            if node.patch_verdict is None:
                continue  # not yet evaluated
            if not node.patch_verdict.reflections:
                continue  # reflection not yet done
            # Check if patterns were already extracted for this node
            # by checking if Session 3 session info was recorded
            if node.sdk_session_pattern_extraction is not None:
                continue
            candidates_needing_extraction.append(idx)

        if not candidates_needing_extraction:
            return

        session_root = str(self._adapter._session_root)

        for idx in candidates_needing_extraction:
            node = dag.nodes[idx]
            worktree = Path(node.worktree_path) if node.worktree_path else None
            if worktree is None or not worktree.exists():
                logger.warning(
                    "Skipping Session 3 for C%d: worktree not found", idx,
                )
                continue

            self._logger.log(
                f"Running Session 3 (Pattern Extraction) for C{idx} "
                f"(iter {node.iteration})..."
            )

            session1_diagnosis = (
                node.patch_intent.diagnosis
                if node.patch_intent and node.patch_intent.diagnosis
                else ""
            )
            session1_patch_approach = (
                node.patch_intent.approach if node.patch_intent else ""
            )
            proposer_reasoning_path = str(worktree / "proposer_reasoning.md")

            # Gather pre-patch failures (from diagnosis)
            pre_patch_failures = []
            if node.patch_verdict:
                for si in node.patch_verdict.scenario_impacts:
                    if si.score_before < 0.5:  # Failed before patch
                        entry: dict[str, Any] = {
                            "scenario_id": si.scenario_id,
                            "session1_diagnosis": session1_diagnosis,
                            "proposer_reasoning_path": proposer_reasoning_path,
                            "session2_root_cause": "",
                            "session2_explanation": "",
                        }
                        # Session 2 reviews the original failure after seeing
                        # both the before/after traces and patch outcome.
                        for refl in node.patch_verdict.reflections:
                            if refl.scenario_id == si.scenario_id:
                                entry["session2_root_cause"] = refl.root_cause or ""
                                entry["session2_explanation"] = refl.explanation or ""
                                break
                        pre_patch_failures.append(entry)

            # Gather post-patch failures (from reflection)
            post_patch_failures = []
            if node.patch_verdict:
                for refl in node.patch_verdict.reflections:
                    if refl.status_change in ("still_failing", "regressed"):
                        post_patch_failures.append({
                            "scenario_id": refl.scenario_id,
                            "status_change": refl.status_change,
                            "session1_patch_approach": session1_patch_approach,
                            "session2_root_cause": refl.root_cause or "",
                            "session2_explanation": refl.explanation or "",
                        })

            # Skip if no failures to process
            if not pre_patch_failures and not post_patch_failures:
                self._logger.log(f"Session 3 for C{idx}: no failures to process")
                continue

            session3_prompt = build_session3_prompt(
                iteration=node.iteration,
                candidate_idx=idx,
                worktree_path=str(worktree),
                session_root=session_root,
                before_output_dir=node.train_before_cycle_dir or "",
                after_output_dir=node.train_after_cycle_dir or "",
                pre_patch_failures=pre_patch_failures,
                post_patch_failures=post_patch_failures,
                sampling_strategy=self._config.sampling_strategy,
            )

            # Install pattern CLI (iteration-based score display)
            pattern_cli_env = install_pattern_cli(
                session_root, str(worktree),
                current_iteration=node.iteration,
                eta=self._config.eta,
                sampling_strategy=self._config.sampling_strategy,
                session=3,
            )
            # Also need evo-dag CLI for history context
            evo_cli_env = install_evo_dag_cli(session_root, str(worktree))
            combined_env = {**evo_cli_env, **pattern_cli_env}
            output_dir = node.train_after_cycle_dir or node.train_before_cycle_dir

            session3_result = self._run_sdk_session(
                worktree_path=worktree,
                prompt=build_skill_prefix(
                    session=3,
                    phase=self._get_phase_for_iteration(node.iteration),
                    sampling_strategy=self._config.sampling_strategy,
                ) + session3_prompt,
                model=self._config.active_model,
                timeout=self._config.pattern_extraction_timeout,
                extra_env=combined_env,
                session_type="pattern_extraction",
                iteration=node.iteration,
                candidate_idx=idx,
                artifact_dir=output_dir,
            )

            # Reload registry (CLI may have modified it)
            registry.load()

            # ── Auto-derive pattern observations from mini-batch + tagging ──
            # For each pattern whose scenarios were in the mini-batch, record
            # active=1 (tagged) or active=0 (not tagged) for the activity EMA.
            self._record_pattern_observations(registry, node, idx)

            # Save session results JSON (like other sessions)
            if session3_result and output_dir:
                session3_info = self._extract_session_info(
                    session3_result, self._config.active_model,
                    self._config.pattern_extraction_timeout,
                    output_dir, "pattern_extraction",
                    node.iteration, idx,
                )
                if session3_info:
                    dag.set_sdk_session_info(idx, pattern_extraction=session3_info)
                    dag.save()

            if session3_result:
                self._logger.log(f"Session 3 for C{idx} completed")
            else:
                self._logger.log(f"Session 3 for C{idx} failed (non-fatal)")

    def _record_pattern_observations(
        self,
        registry: PatternRegistry,
        node: EvolutionNode,
        candidate_idx: int,
    ) -> None:
        """Auto-derive failure-activity observations from this iteration's batch.

        Side-observation design: EVERY failure pattern whose tagged scenarios
        appear in this iteration's mini-batch receives exactly ONE observation,
        valued by its POST-patch activity — the fraction of the scenarios
        associated with that pattern and evaluated in the mini-batch that were
        still tagged to it after the patch (0.0 = resolved / cause shifted
        away, 1.0 = still active). Both the evaluated and still-tagged scenario
        IDs are retained. Because patterns share scenarios (many-to-many), a
        single batch execution legitimately observes and updates several arms
        at once.

        This is compatible with the *rested* bandit: patterns with NO scenario
        in the batch are left untouched — their EMA stays frozen (no
        elapsed-time drift, and no count-based bonus that would grow purely
        from not being pulled). Observations feed the EMA score phi (see
        ``PatternRegistry.compute_scores``).

        Post-patch (not pre-patch) activity is used because phi must predict
        "will this pattern be active when next sampled?", and the next sample
        runs on the just-patched harness.

        Patterns whose ONLY tuples were just created in this iteration's
        post-patch analysis receive NO observation — arm creation != arm pull;
        they keep their seed score (EMA phi = 1.0) until first pulled.
        """
        if not node.mini_batch_ids:
            return

        iteration = node.iteration

        # Convert mini-batch indices to full scenario names for comparison
        # with pattern tuples (which use full names).
        mini_batch_scenario_names: set[str] = set()
        for idx_str in node.mini_batch_ids:
            name = self._idx_to_scenario.get(idx_str)
            if name:
                mini_batch_scenario_names.add(name)

        if not mini_batch_scenario_names:
            logger.warning(
                "Could not resolve any mini-batch IDs to scenario names for C%d",
                candidate_idx,
            )
            return

        after_dir = node.train_after_cycle_dir or ""

        obs_count = 0

        for pattern in registry.patterns.values():
            pattern_scenario_ids = {t.scenario_id for t in pattern.tuples}
            overlap = mini_batch_scenario_names & pattern_scenario_ids
            if not overlap:
                continue

            # Determine if this pattern existed BEFORE this iteration's
            # Session 3. Patterns just created have tuples only from the
            # current after_dir — they should NOT receive observations.
            # Registration (arm creation) != observation (arm pulled).
            # Newly registered patterns start with optimistic prior.
            has_prior_tuples = any(
                t.harness_idx != candidate_idx or t.trace_dir != after_dir
                for t in pattern.tuples
                if t.scenario_id in mini_batch_scenario_names
            )

            if not has_prior_tuples:
                # Pattern was just created in this iteration's Session 3.
                # Arm creation != arm pull: it records no observation and keeps
                # its seed score (EMA phi = 1.0), so the softmax sampler gives
                # it a high pull probability in a future iteration.
                continue

            # Single observation valued by POST-patch activity: the fraction
            # of this pattern's scenarios evaluated in the mini-batch that were
            # still tagged to it after the patch. A pattern that was sampled
            # but is no longer the active cause (resolved, or the failure
            # shifted to another pattern) records 0.0 and is naturally
            # deprioritized; one that is still the active cause records toward
            # 1.0. No pre-patch observation is recorded (see method docstring).
            tagged_after_scenario_ids: list[str] = []
            if after_dir:
                tagged_after_scenario_ids = sorted(
                    sid for sid in overlap
                    if any(
                        t.harness_idx == candidate_idx
                        and t.trace_dir == after_dir
                        and t.scenario_id == sid
                        for t in pattern.tuples
                    )
                )
            evaluated_scenario_ids = sorted(overlap)
            post_reward = len(tagged_after_scenario_ids) / len(evaluated_scenario_ids)
            registry.observe(
                pattern_id=pattern.pattern_id,
                iteration=iteration,
                active=post_reward,
                evaluated_scenario_ids=evaluated_scenario_ids,
                tagged_scenario_ids=tagged_after_scenario_ids,
            )
            obs_count += 1

        if obs_count > 0:
            registry.save()
            logger.info(
                "Recorded %d pattern observations (post-patch, iter %d) for C%d",
                obs_count, iteration, candidate_idx,
            )

        # ── Record (scenario, harness) probe points (N_t) ──
        # N_t counts DISTINCT (scenario, harness) pairs executed so far. Both
        # the pre-patch pull (on this node's pre-patch harness) and the
        # post-patch re-evaluation (on the patched harness) are distinct probe
        # points, so BOTH are recorded. Re-running the same scenario on the
        # same harness adds no new point (deduplicated in record_probe_points).
        if isinstance(self._batch_sampler, ActiveSaddlerBanditSampler):
            batch_names = list(mini_batch_scenario_names)
            before_tag = node.pre_patch_commit or f"C{candidate_idx}:before"
            added_probe_points = self._batch_sampler.record_probe_points(
                batch_names,
                before_tag,
            )
            if after_dir:
                after_tag = node.commit_hash or f"C{candidate_idx}:after"
                added_probe_points.extend(
                    self._batch_sampler.record_probe_points(batch_names, after_tag)
                )
            self._persist_sampler_probe_points(
                iteration,
                added_probe_points,
                artifact_dir=node.train_before_cycle_dir,
                candidate_idx=candidate_idx,
            )

    # ------------------------------------------------------------------
    # ProposeNewCandidate protocol
    # ------------------------------------------------------------------

    def propose(
        self, state: GEPAState[Any, DataId],
    ) -> CandidateProposal | None:
        """Propose a new candidate. One candidate per iteration."""
        if self._pending_proposals:
            return self._pending_proposals.popleft()

        try:
            if self._dag is None:
                self._ensure_dag(state)
            else:
                self._meta_iteration += 1
            self._logger.log(
                f"\n{'='*60}\n"
                f"EVOLUTION-DAG v2 ITERATION {self._meta_iteration} "
                f"(engine iter {state.i})\n"
                f"{'='*60}"
            )
            proposal = self._run_iteration(state)
        except IncompleteIterationError:
            self._cancel_reserved_eval_cycle()
            raise
        except Exception:
            self._cancel_reserved_eval_cycle()
            logger.exception(
                "Failed in EvolutionDAG v2 iteration %d", self._meta_iteration,
            )
            return None

        if proposal is None:
            self._logger.log("EVOLUTION-DAG v2: no valid candidate generated")
            return None

        return proposal

    def _cancel_reserved_eval_cycle(self) -> None:
        cancel = getattr(self._adapter, "cancel_reserved_eval_cycle", None)
        if not callable(cancel):
            return
        canceled_dir = cancel()
        if canceled_dir is not None:
            self._logger.log(
                f"Released unconsumed evaluation reservation: {canceled_dir}"
            )

    def finalize(self, state: GEPAState) -> None:
        """Run final reflection and pattern extraction for the last iteration.

        Called by the engine after the main loop exits so that the last
        iteration's deferred Sessions 2 and 3 are not skipped.
        At this point the engine has already completed the dev-set
        evaluation for the last candidate, so val scores are available.
        """
        try:
            dag = self._ensure_dag(state)
            self._sync_val_scores_from_state(dag, state)
        except Exception:
            logger.exception("finalize: failed to prepare deferred sessions")
            return

        try:
            self._run_deferred_session2(dag, state)
        except Exception:
            logger.exception("finalize: deferred session 2 failed (non-fatal)")

        try:
            dag.load()
            self._run_session3_pattern_extraction(dag, state)
        except Exception:
            logger.exception("finalize: session 3 failed (non-fatal)")

    # ------------------------------------------------------------------
    # DAG initialization
    # ------------------------------------------------------------------

    def _ensure_dag(self, state: GEPAState) -> EvolutionDAG:
        """Initialize or load the DAG, creating seed node if needed."""
        if self._dag is not None:
            return self._dag

        session_root = str(self._adapter._session_root)
        dag = EvolutionDAG(session_root)
        dag.load()

        if not dag.nodes:
            seed_worktree = self._get_or_create_seed_worktree(state)
            seed_score = 0.0
            if state.program_full_scores_val_set:
                seed_score = state.program_full_scores_val_set[0]
            dag.add_seed_node(str(seed_worktree), seed_score)
            self._worktree_map[0] = seed_worktree
            self._logger.log(f"Created seed node with score {seed_score:.4f}")

        for idx, node in dag.nodes.items():
            if node.worktree_path:
                wt = Path(node.worktree_path)
                if wt.exists():
                    self._worktree_map[idx] = wt

        # Restore the next iteration before propose() logs its header or creates
        # a node, avoiding a misleading iteration-1 header after resume.
        if dag.nodes:
            max_iteration = max(n.iteration for n in dag.nodes.values())
            if max_iteration >= self._meta_iteration:
                logger.info(
                    "Restoring _meta_iteration from DAG: %d → %d",
                    self._meta_iteration, max_iteration + 1,
                )
                self._meta_iteration = max_iteration + 1

        incomplete = [
            node
            for node in dag.nodes.values()
            if (
                node.iteration > 0
                and not node.abandoned
                and node.patch_verdict is None
                and node.accepted is None
            )
        ]
        if incomplete:
            first_iteration = min(node.iteration for node in incomplete)
            raise IncompleteIterationError(
                "Interrupted AutoSaddler iteration detected at "
                f"iteration {first_iteration} in {session_root}. Resuming "
                "requires sampler state and DAG state to share a clean boundary."
            )

        self._dag = dag
        return dag

    def _ensure_pattern_registry(self):
        """Initialize or load the PatternRegistry."""
        if self._pattern_registry is not None:
            return self._pattern_registry

        from autosaddler.v1.proposer.autosaddler.pattern_registry import PatternRegistry

        session_root = str(self._adapter._session_root)
        registry = PatternRegistry(session_root)
        registry.load()
        self._pattern_registry = registry

        # Initialize ActiveSaddlerBanditSampler if deferred
        if self._deferred_score_sampler and self._batch_sampler is None:
            mbs = self._config.train_minibatch_size
            state_path = Path(self._adapter._session_root) / "bandit_state.json"
            self._batch_sampler = ActiveSaddlerBanditSampler(
                minibatch_size=mbs,
                pattern_registry=registry,
                eta=self._config.eta,
                temperature=self._config.softmax_temperature,
                min_prob=self._config.min_prob,
                scenario_to_idx=self._scenario_to_idx,
                state_path=str(state_path),
                rng=random.Random(self._config.seed),
            )
            self._logger.log("Initialized ActiveSaddlerBanditSampler")

        return registry

    @staticmethod
    def _discard_read_only_session_edits(prepared_worktree: Path) -> None:
        """Reset accidental edits while preserving the prepared commit."""
        try:
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=str(prepared_worktree),
                capture_output=True,
                timeout=30,
                check=False,
            )
        except Exception:
            logger.warning("Failed to reset prepared worktree after read-only session")

    def _run_unseen_scenario_exploration_session(
        self,
        dag: EvolutionDAG,
        state: GEPAState,
        prepared_candidate_idx: int,
        prepared_worktree: Path,
        provisional_parent_idx: int,
        provisional_parent_commit: str,
        session0_status: str,
        train_before_cycle_dir: str | Path,
    ) -> str:
        """ActiveSaddler Session 3.5: decide PULL or DRAW before Session 4."""
        from autosaddler.v1.core.data_loader import ListDataLoader

        registry = self._ensure_pattern_registry()
        sampler = self._batch_sampler
        if not isinstance(sampler, ActiveSaddlerBanditSampler):
            return "draw"

        if not any(pattern.tuples for pattern in registry.patterns.values()):
            self._logger.log("Session 3.5: no known arms -> draw")
            return "draw"
        if not prepared_worktree.exists():
            logger.warning("Session 3.5 skipped: prepared worktree not found")
            return "draw"

        prepared_node = dag.nodes[prepared_candidate_idx]
        prepared_commit = prepared_node.pre_patch_commit
        if not prepared_commit:
            raise RuntimeError(
                f"Prepared harness C{prepared_candidate_idx} has no commit"
            )

        session_root = str(self._adapter._session_root)
        decision_path = iteration_artifact_path(
            train_before_cycle_dir,
            self._meta_iteration,
            prepared_candidate_idx,
            "arm_decision",
        )
        try:
            decision_path.unlink()
        except FileNotFoundError:
            pass

        loader = ListDataLoader(self.trainset)
        unseen_pool_size = sampler.unseen_pool_size(loader)
        prompt = build_unseen_scenario_exploration_prompt(
            iteration=self._meta_iteration,
            prepared_candidate_idx=prepared_candidate_idx,
            prepared_worktree_path=str(prepared_worktree),
            prepared_commit=prepared_commit,
            provisional_parent_idx=provisional_parent_idx,
            provisional_parent_commit=provisional_parent_commit,
            session0_status=session0_status,
            session_root=session_root,
            dag=dag,
            registry=registry,
            unseen_pool_size=unseen_pool_size,
            eta=self._config.eta,
        )
        pattern_cli_env = install_pattern_cli(
            session_root,
            str(prepared_worktree),
            current_iteration=self._meta_iteration,
            eta=self._config.eta,
            sampling_strategy=self._config.sampling_strategy,
            session=UNSEEN_SCENARIO_EXPLORATION_SESSION,
            artifact_dir=train_before_cycle_dir,
            candidate_idx=prepared_candidate_idx,
        )
        evo_cli_env = install_evo_dag_cli(session_root, str(prepared_worktree))
        phase = self._get_phase_for_iteration(self._meta_iteration)
        self._logger.log(
            "Running Session 3.5 (Unseen Scenario Exploration) on prepared "
            f"C{prepared_candidate_idx}..."
        )
        result = self._run_sdk_session(
            worktree_path=prepared_worktree,
            prompt=build_skill_prefix(
                session=UNSEEN_SCENARIO_EXPLORATION_SESSION,
                phase=phase,
                sampling_strategy=self._config.sampling_strategy,
            ) + prompt,
            model=self._config.active_model,
            timeout=self._config.arm_scoring_timeout,
            extra_env={**evo_cli_env, **pattern_cli_env},
            session_type="unseen_scenario_exploration",
            iteration=self._meta_iteration,
            candidate_idx=prepared_candidate_idx,
            artifact_dir=train_before_cycle_dir,
        )
        self._discard_read_only_session_edits(prepared_worktree)

        if result:
            session_info = self._extract_session_info(
                result,
                self._config.active_model,
                self._config.arm_scoring_timeout,
                str(train_before_cycle_dir),
                "unseen_scenario_exploration",
                self._meta_iteration,
                prepared_candidate_idx,
            )
            if session_info:
                dag.set_sdk_session_info(
                    prepared_candidate_idx,
                    unseen_scenario_exploration=session_info,
                )
                dag.save()
        else:
            self._logger.log(
                "Session 3.5 failed (non-fatal) — defaulting to PULL"
            )

        action = "pull"
        try:
            if decision_path.exists():
                import json as _json

                data = _json.loads(decision_path.read_text(encoding="utf-8"))
                if data.get("action") in ("pull", "draw"):
                    action = data["action"]
        except Exception:
            logger.warning("Failed to read Session 3.5 decision; defaulting to PULL")
        if action == "draw" and unseen_pool_size == 0:
            self._logger.log(
                "Session 3.5 selected DRAW with an empty unseen pool -> PULL fallback"
            )
            action = "pull"
        self._logger.log(f"Session 3.5 decision: {action}")
        return action

    def _run_arm_scoring_session(
        self,
        dag: EvolutionDAG,
        state: GEPAState,
        prepared_candidate_idx: int,
        prepared_worktree: Path,
        provisional_parent_idx: int,
        provisional_parent_commit: str,
        session0_status: str,
        train_before_cycle_dir: str | Path,
    ) -> None:
        """ActiveSaddler Session 4: score arms after a PULL decision.

        A read-only session run on the Session 0-prepared harness before the
        mini-batch is selected. The agent records scores via ``pattern rate``
        (the sampler then uses each pattern's latest raw score as phi_t(p)).

        Called only after Session 3.5 selects PULL.
        """
        registry = self._ensure_pattern_registry()
        sampler = self._batch_sampler
        if not isinstance(sampler, ActiveSaddlerBanditSampler):
            return

        expected_arm_ids = sorted(
            pattern_id
            for pattern_id, pattern in registry.patterns.items()
            if pattern.tuples
        )
        if not expected_arm_ids:
            return

        if not prepared_worktree.exists():
            logger.warning("Arm scoring skipped: prepared worktree not found")
            return

        session_root = str(self._adapter._session_root)
        self._logger.log(
            f"Running Session 4 (Arm Scoring) on prepared C{prepared_candidate_idx}..."
        )

        prepared_node = dag.nodes[prepared_candidate_idx]
        prepared_commit = prepared_node.pre_patch_commit
        if not prepared_commit:
            raise RuntimeError(
                f"Prepared harness C{prepared_candidate_idx} has no commit"
            )

        scoring_prompt = build_arm_scoring_prompt(
            iteration=self._meta_iteration,
            prepared_candidate_idx=prepared_candidate_idx,
            prepared_worktree_path=str(prepared_worktree),
            prepared_commit=prepared_commit,
            provisional_parent_idx=provisional_parent_idx,
            provisional_parent_commit=provisional_parent_commit,
            session0_status=session0_status,
            session_root=session_root,
            dag=dag,
            registry=registry,
            eta=self._config.eta,
            sampling_strategy=self._config.sampling_strategy,
        )

        pattern_cli_env = install_pattern_cli(
            session_root, str(prepared_worktree),
            current_iteration=self._meta_iteration,
            eta=self._config.eta,
            sampling_strategy=self._config.sampling_strategy,
            session=4,
        )
        evo_cli_env = install_evo_dag_cli(session_root, str(prepared_worktree))
        combined_env = {**evo_cli_env, **pattern_cli_env}

        phase = self._get_phase_for_iteration(self._meta_iteration)
        result = self._run_sdk_session(
            worktree_path=prepared_worktree,
            prompt=build_skill_prefix(
                session=4,
                phase=phase,
                sampling_strategy=self._config.sampling_strategy,
            ) + scoring_prompt,
            model=self._config.active_model,
            timeout=self._config.arm_scoring_timeout,
            extra_env=combined_env,
            session_type="arm_scoring",
            iteration=self._meta_iteration,
            candidate_idx=prepared_candidate_idx,
            artifact_dir=train_before_cycle_dir,
        )

        # Reload registry to pick up the agent's `pattern rate` writes (shared
        # object with the sampler, so the sampler sees the new scores).
        registry.load()

        self._discard_read_only_session_edits(prepared_worktree)

        if result:
            # Export and attach directly: the current node already exists.
            session_info = self._extract_session_info(
                result, self._config.active_model, self._config.arm_scoring_timeout,
                str(train_before_cycle_dir), "arm_scoring", self._meta_iteration,
                prepared_candidate_idx,
            )
            if session_info:
                dag.set_sdk_session_info(
                    prepared_candidate_idx,
                    arm_scoring=session_info,
                )
                dag.save()
        missing_arm_ids = [
            pattern_id
            for pattern_id in expected_arm_ids
            if registry.get_agent_score(pattern_id, self._meta_iteration) is None
        ]
        if result is None or missing_arm_ids:
            detail = (
                f"missing current-iteration scores for {missing_arm_ids}"
                if missing_arm_ids
                else "SDK session failed"
            )
            raise IncompleteIterationError(
                f"Session 4 for C{prepared_candidate_idx} is incomplete: {detail}"
            )
        self._logger.log(
            f"Session 4 (Arm Scoring) for C{prepared_candidate_idx} completed"
        )

    def _get_or_create_seed_worktree(self, state: GEPAState) -> Path:
        """Get the seed candidate's worktree."""
        if state.program_candidates:
            seed = state.program_candidates[0]
            if "__autosaddler_worktree__" in seed:
                return Path(seed["__autosaddler_worktree__"])
            try:
                wt, _ = self._adapter._worktree_pool.get_or_create(
                    seed, lambda _wt, _c: None,
                )
                return wt
            except Exception:
                pass
        return self._create_worktree(0, 0)

    # ------------------------------------------------------------------
    # Main iteration flow
    # ------------------------------------------------------------------

    def _prepare_iteration_harness(
        self,
        *,
        dag: EvolutionDAG,
        session_root: str,
        base_parent_idx: int,
        base_parent_worktree: Path,
        new_worktree: Path,
        current_idx: int,
        phase: str,
        train_before_cycle_dir: str | Path,
    ) -> tuple[dict[str, str], str, str]:
        """Run Session 0 and commit the exact harness used by later sessions."""
        prompt_bundle = resolve_prompt_bundle(
            sampling_strategy=self._config.sampling_strategy,
        )
        install_prompts_and_skills(
            str(new_worktree),
            prompt_bundle.claude_md,
            phase=phase,
            skill_names=prompt_bundle.skill_names,
        )
        cli_env = install_evo_dag_cli(session_root, str(new_worktree))
        self._logger.log("CLAUDE.md, skills, and CLI installed")
        dag.save()

        session0_result = None
        session0_status = "skipped"
        if not self._config.skip_session0 and self._meta_iteration > 1:
            self._logger.log("Starting Session 0 (Candidate Selection)...")
            session0_prompt = build_session0_prompt(
                iteration=self._meta_iteration,
                worktree_path=str(new_worktree),
                parent_worktree=str(base_parent_worktree),
                base_parent_idx=base_parent_idx,
                session_root=session_root,
                dag=dag,
                phase=phase,
                sampling_strategy=self._config.sampling_strategy,
            )
            session0_result = self._run_sdk_session(
                worktree_path=new_worktree,
                prompt=build_skill_prefix(
                    session=0,
                    phase=phase,
                    sampling_strategy=self._config.sampling_strategy,
                ) + session0_prompt,
                model=self._config.active_model,
                timeout=self._config.candidate_selection_timeout,
                extra_env=cli_env,
                session_type="selection",
                iteration=self._meta_iteration,
                candidate_idx=current_idx,
                artifact_dir=train_before_cycle_dir,
            )
            session0_status = "completed" if session0_result else "failed"
            self._logger.log(
                "Session 0 completed"
                if session0_result
                else "Session 0 failed — proceeding with provisional base"
            )

        # Reload CLI updates before attaching SDK metadata.
        dag.load()
        if session0_result:
            session0_info = self._extract_session_info(
                session0_result,
                self._config.active_model,
                self._config.candidate_selection_timeout,
                str(train_before_cycle_dir),
                "selection",
                self._meta_iteration,
                current_idx,
            )
            if session0_info:
                dag.set_sdk_session_info(current_idx, selection=session0_info)
                dag.save()

        if not self._verify_worktree(new_worktree):
            self._logger.log(
                "Post-Session-0 verification FAILED — restoring provisional base"
            )
            parent_commit = git.Repo(base_parent_worktree).head.commit.hexsha
            repo = git.Repo(new_worktree)
            repo.git.reset("--hard", parent_commit)
            repo.git.clean("-fd", "-e", "CLAUDE.md", "-e", ".claude/")
            dag.clear_selection_decision(current_idx)
            session0_status = "verification_failed_fallback"
            install_prompts_and_skills(
                str(new_worktree),
                prompt_bundle.claude_md,
                phase=phase,
                skill_names=prompt_bundle.skill_names,
            )
            dag.save()

        prepared_commit = self._commit_changes(new_worktree)
        if not prepared_commit:
            raise RuntimeError(
                f"Failed to commit Session 0-prepared harness C{current_idx}"
            )
        dag.nodes[current_idx].pre_patch_commit = prepared_commit
        dag.save()
        return cli_env, session0_status, prepared_commit

    def _run_iteration(
        self, state: GEPAState,
    ) -> CandidateProposal | None:
        """Execute the full iteration with 3 sessions.

        Pattern-strategy flow:
        1. Run deferred Session 2 and Session 3
        2. Fork worktree + create a prepared-unsampled DAG node
        3. [Session 0] Prepare and commit the harness
        4. [ActiveSaddler Session 3.5] Decide PULL/DRAW when enabled
        5. [Session 4] Score arms after a PULL decision
        6. Sample mini-batch and mark the node sampled
        7. Initial evaluation, Session 1, and re-evaluation
        8. Complete the edge/verdict and return the proposal

        Epoch preserves the main-compatible sampling-first order, followed by
        Session 0 and evaluation.
        → Engine: acceptance gate → dev-set eval → state update
        → Next iteration step 1: val scores sync → deferred Session 2
        """
        dag = self._ensure_dag(state)
        session_root = str(self._adapter._session_root)

        # Sync val scores from engine state into DAG nodes
        self._sync_val_scores_from_state(dag, state)

        # ── Step 1: Deferred Session 2 from previous iteration ────────

        self._run_deferred_session2(dag, state)

        # ── Step 1.5: Session 3 — Pattern Extraction (ActiveSaddler) ──

        self._run_session3_pattern_extraction(dag, state)

        # Ensure PatternRegistry is loaded (for ActiveSaddlerBanditSampler)
        if self._config.pattern_sampling_enabled:
            self._ensure_pattern_registry()

        # Find the current base parent (exclude abandoned nodes)
        eligible_nodes = [
            n for n in dag.nodes.values() if not n.abandoned
        ]
        if not eligible_nodes:
            eligible_nodes = list(dag.nodes.values())  # fallback
        base_parent = max(eligible_nodes, key=lambda n: n.iteration)
        base_parent_idx = base_parent.idx
        base_parent_worktree = Path(base_parent.worktree_path)

        self._logger.log(
            f"Base parent: C{base_parent_idx} "
            f"(iter {base_parent.iteration}, "
            f"val={base_parent.score_val})"
        )

        base_parent_commit = git.Repo(base_parent_worktree).head.commit.hexsha

        if self._config.pattern_sampling_enabled:
            # Pattern strategies prepare the actual harness first. Session 4
            # and all sampler decisions therefore see Session 0's committed
            # code rather than the provisional parent.
            new_worktree = self._fork_worktree(
                base_parent_worktree,
                self._meta_iteration,
            )
            self._logger.log(f"Forked worktree: {new_worktree}")
            node = dag.add_node(
                iteration=self._meta_iteration,
                worktree_path=str(new_worktree),
                base_parent_idx=base_parent_idx,
                sampling_completed=False,
            )
            current_idx = node.idx
            self._worktree_map[current_idx] = new_worktree
            dag.add_base_edge(base_parent_idx, current_idx)

            phase = self._get_phase_for_iteration(self._meta_iteration)
            self._logger.log(
                f"Phase: {phase} (iteration {self._meta_iteration})"
            )
            phase_before = f"iter{self._meta_iteration:02d}_train_before"
            train_before_cycle_dir = self._adapter.reserve_eval_cycle(
                phase_before,
                iteration=self._meta_iteration,
                capture_traces=True,
            )
            cli_env, session0_status, prepared_commit = (
                self._prepare_iteration_harness(
                    dag=dag,
                    session_root=session_root,
                    base_parent_idx=base_parent_idx,
                    base_parent_worktree=base_parent_worktree,
                    new_worktree=new_worktree,
                    current_idx=current_idx,
                    phase=phase,
                    train_before_cycle_dir=train_before_cycle_dir,
                )
            )

            forced_action: str | None = None
            if self._config.agent_arm_creation_enabled:
                forced_action = self._run_unseen_scenario_exploration_session(
                    dag,
                    state,
                    current_idx,
                    new_worktree,
                    base_parent_idx,
                    base_parent_commit,
                    session0_status,
                    train_before_cycle_dir,
                )
                if forced_action == "pull":
                    self._run_arm_scoring_session(
                        dag,
                        state,
                        current_idx,
                        new_worktree,
                        base_parent_idx,
                        base_parent_commit,
                        session0_status,
                        train_before_cycle_dir,
                    )

            mini_batch_ids, mini_batch = self._sample_mini_batch(
                state,
                forced_action=forced_action,
                artifact_dir=train_before_cycle_dir,
                candidate_idx=current_idx,
            )
            pulled_arm_id = None
            snapshot = getattr(self._batch_sampler, "last_score_snapshot", None)
            if isinstance(snapshot, dict):
                pulled_arm_id = snapshot.get("chosen_arm")
            dag.set_sampling_result(
                current_idx,
                mini_batch_ids,
                pulled_arm_id,
            )
            dag.save()
            self._maybe_record_capability_end(dag)
        else:
            # Preserve main-compatible Epoch ordering: sample first, then let
            # Session 0 prepare the harness that will be evaluated.
            mini_batch_ids, mini_batch = self._sample_mini_batch(state)
            new_worktree = self._fork_worktree(
                base_parent_worktree,
                self._meta_iteration,
            )
            self._logger.log(f"Forked worktree: {new_worktree}")
            node = dag.add_node(
                iteration=self._meta_iteration,
                worktree_path=str(new_worktree),
                base_parent_idx=base_parent_idx,
                mini_batch_ids=mini_batch_ids,
                sampling_completed=True,
            )
            current_idx = node.idx
            self._worktree_map[current_idx] = new_worktree
            dag.add_base_edge(base_parent_idx, current_idx)
            self._maybe_record_capability_end(dag)
            phase = self._get_phase_for_iteration(self._meta_iteration)
            self._logger.log(
                f"Phase: {phase} (iteration {self._meta_iteration})"
            )
            phase_before = f"iter{self._meta_iteration:02d}_train_before"
            train_before_cycle_dir = self._adapter.reserve_eval_cycle(
                phase_before,
                iteration=self._meta_iteration,
                capture_traces=True,
            )
            cli_env, session0_status, prepared_commit = (
                self._prepare_iteration_harness(
                    dag=dag,
                    session_root=session_root,
                    base_parent_idx=base_parent_idx,
                    base_parent_worktree=base_parent_worktree,
                    new_worktree=new_worktree,
                    current_idx=current_idx,
                    phase=phase,
                    train_before_cycle_dir=train_before_cycle_dir,
                )
            )

        self._logger.log(f"Mini-batch: {len(mini_batch_ids)} scenarios")
        node = dag.nodes[current_idx]

        self._logger.log("Initial evaluation on mini-batch...")
        initial_eval = self._evaluate_candidate(
            state, node, mini_batch, mini_batch_ids,
            worktree_override=new_worktree,
        )
        initial_results = parse_evaluation_results(initial_eval["cycle_dir"])
        if initial_results:
            resolved_ids = sorted(initial_results.keys())
            if resolved_ids != sorted(mini_batch_ids):
                self._logger.log(
                    f"Resolved mini_batch_ids: {mini_batch_ids} → {resolved_ids}"
                )
                mini_batch_ids = resolved_ids

        initial_pass_rate = compute_pass_rate(initial_results, mini_batch_ids)
        self._logger.log(f"Initial pass rate: {initial_pass_rate:.4f}")

        dag.set_cycle_dirs(current_idx, train_before_cycle_dir=initial_eval["cycle_dir"])
        dag.save()  # Persist before_cycle_dir before any dag.load() overwrites it

        if initial_pass_rate >= 1.0:
            self._logger.log("All scenarios already passing — skipping iteration")
            dag.load()
            node = dag.nodes[current_idx]
            # All scenarios passed → no patch and no Session 3 run. We MUST
            # still record observations + probe points, otherwise this
            # iteration produces zero learning signal: every sampled
            # pre-existing pattern would keep its (stale, high) EMA score and
            # be re-sampled next iteration, and N_t would not advance.
            # Recording active=0 for the sampled patterns lowers their EMA,
            # and the pre-patch probe points advance N_t.
            if self._config.pattern_sampling_enabled:
                try:
                    registry = self._ensure_pattern_registry()
                    self._record_pattern_observations(registry, node, current_idx)
                except Exception:
                    logger.exception(
                        "Failed to record all-pass observations for C%d",
                        current_idx,
                    )
            # Record this pull as a first-class event: "already passing" is a
            # legitimate arm outcome, not a non-event. Persist the train_before
            # pass rate and write a still_passing snapshot to the scenario
            # registry so every consumer (evo-dag show scenario/history/node,
            # Session 1 prior attempts, Session 4 arm scoring) sees it. No patch
            # was made, so score_train_after and patch_verdict stay None (avoids
            # polluting acceptance / Session 2 / Session 3 logic).
            dag.set_train_scores(current_idx, initial_pass_rate, None)
            node = dag.nodes[current_idx]
            try:
                still_passing = compute_all_scenario_impacts(
                    initial_results, initial_results, mini_batch_ids,
                )
                update_scenario_registry(dag, node, still_passing)
            except Exception:
                logger.exception(
                    "Failed to record all-pass scenario history for C%d",
                    current_idx,
                )
            node.abandoned = True
            node.abandon_reason = "all_pass"
            dag.save()
            return None

        initial_scores = {
            sid: initial_results.get(sid, {}).get("score", 0.0)
            for sid in mini_batch_ids
        }
        initial_rationales = {
            sid: initial_results.get(sid, {}).get("rationale")
            for sid in mini_batch_ids
        }

        # ── Step 7: Session 1 — Diagnose + Patch ─────────────────────

        self._logger.log("Starting Session 1 (Diagnose + Patch)...")

        # Collect cherry-pick parents from DAG edges
        cherry_pick_parents: list[tuple[int, str]] = []
        for edge in dag.edges.values():
            if edge.child_idx == current_idx and edge.edge_type == "cherry_pick":
                cp_wt = self._worktree_map.get(edge.parent_idx)
                cp_wt_str = str(cp_wt) if cp_wt else "(unknown)"
                cherry_pick_parents.append((edge.parent_idx, cp_wt_str))

        # Complete pull history on the SAME arm (pattern), matched by
        # pulled_arm_id (not scenario overlap): patched attempts (approach,
        # dev-set delta, per-scenario reflections), all-pass skips (scenarios
        # already passing), and failed attempts. Empty for a freshly created /
        # never-repeated arm.
        arm_pull_history = dag.get_arm_pull_history(
            arm_id=dag.nodes[current_idx].pulled_arm_id,
            exclude_idx=current_idx,
        )
        dev_by_idx = {n.idx: n.score_val for n in dag.nodes.values()}

        session1_prompt = build_session1_prompt(
            iteration=self._meta_iteration,
            candidate_idx=current_idx,
            worktree_path=str(new_worktree),
            parent_worktree=str(base_parent_worktree),
            base_parent_idx=base_parent_idx,
            mini_batch_ids=mini_batch_ids,
            before_scores=initial_scores,
            before_rationales=initial_rationales,
            before_output_dir=initial_eval["cycle_dir"],
            phase=phase,
            cherry_pick_parents=cherry_pick_parents or None,
            arm_pull_history=arm_pull_history,
            dev_by_idx=dev_by_idx,
            sampling_strategy=self._config.sampling_strategy,
        )
        session1_result = self._run_sdk_session(
            worktree_path=new_worktree,
            prompt=build_skill_prefix(
                session=1,
                phase=phase,
                sampling_strategy=self._config.sampling_strategy,
            ) + session1_prompt,
            model=self._config.active_model,
            timeout=self._config.diagnosis_patch_timeout,
            extra_env=cli_env,
            session_type="patch",
            iteration=self._meta_iteration,
            candidate_idx=current_idx,
            artifact_dir=initial_eval["cycle_dir"],
        )

        if session1_result is None:
            self._logger.log("Session 1 failed — marking node as abandoned")
            dag.load()
            dag.nodes[current_idx].abandoned = True
            dag.nodes[current_idx].abandon_reason = "session1_failed"
            dag.save()
            return None

        # Verify + commit
        if not self._verify_worktree(new_worktree):
            self._logger.log("Verification FAILED — marking node as abandoned")
            dag.load()
            dag.nodes[current_idx].abandoned = True
            dag.nodes[current_idx].abandon_reason = "verify_failed"
            dag.save()
            return None

        dag.load()
        node = dag.nodes[current_idx]

        commit_hash = self._commit_changes(new_worktree)
        dag.set_commit_hash(current_idx, commit_hash or "")

        session1_info = self._extract_session_info(
            session1_result, self._config.active_model,
            self._config.diagnosis_patch_timeout, initial_eval["cycle_dir"], "patch",
            self._meta_iteration, current_idx,
        )
        if session1_info:
            dag.set_sdk_session_info(current_idx, patch=session1_info)

        # Auto-generate patch_intent if SDK didn't call evo-dag update-intent
        if node.patch_intent is None:
            files_changed = self._capture_changed_files(new_worktree)
            if files_changed:
                dag.update_patch_intent(
                    current_idx,
                    PatchIntent(
                        target_scenarios=mini_batch_ids,
                        approach="(auto-generated: SDK did not call evo-dag update-intent)",
                        files_changed=files_changed,
                        change_summary="See git diff for details",
                    ),
                )
                logger.warning(
                    "Auto-generated patch_intent for C%d "
                    "(SDK did not call update-intent)",
                    current_idx,
                )

        # ── Step 8: Re-evaluation on mini-batch ───────────────────────

        self._logger.log("Re-evaluating patched worktree on mini-batch...")
        phase_after = f"iter{self._meta_iteration:02d}_train_after"
        self._adapter.set_eval_phase(phase_after, iteration=self._meta_iteration)
        reeval = self._evaluate_candidate(
            state, node, mini_batch, mini_batch_ids,
            worktree_override=new_worktree,
        )
        reeval_results = parse_evaluation_results(reeval["cycle_dir"])
        reeval_pass_rate = compute_pass_rate(reeval_results, mini_batch_ids)
        dag.set_cycle_dirs(current_idx, train_after_cycle_dir=reeval["cycle_dir"])
        dag.set_train_scores(current_idx, initial_pass_rate, reeval_pass_rate)

        self._logger.log(
            f"Re-evaluation pass rate: {reeval_pass_rate:.4f} "
            f"(delta: {reeval_pass_rate - initial_pass_rate:+.4f})"
        )

        # ── Step 9: Initial/re-evaluation comparison + edge completion ─

        scenario_impacts = compute_all_scenario_impacts(
            initial_results, reeval_results, mini_batch_ids,
        )

        code_diff = self._compute_diff(base_parent_worktree, new_worktree)
        files_changed = self._capture_changed_files(new_worktree)

        dag.fill_base_edge_impact(
            base_parent_idx, current_idx,
            mini_batch_ids=mini_batch_ids,
            score_before=initial_pass_rate,
            score_after=reeval_pass_rate,
            scenario_impacts=scenario_impacts,
            code_diff=code_diff,
            files_changed=files_changed,
        )

        # Fill impact data for cherry-pick edges (same evaluation data, diff against each cp parent)
        cp_edges = [
            e for e in dag.get_edges_for_node(current_idx)
            if e.edge_type == "cherry_pick"
        ]
        for cp_edge in cp_edges:
            cp_parent_worktree = self._worktree_map.get(cp_edge.parent_idx)
            if cp_parent_worktree and cp_parent_worktree.exists():
                cp_diff = self._compute_diff(cp_parent_worktree, new_worktree)
                # Extract files_changed from the diff output (diff against cp parent, not base parent)
                cp_files = self._extract_files_from_diff(cp_diff)
            else:
                cp_diff = code_diff
                cp_files = files_changed
            dag.fill_cherry_pick_edge_impact(
                cp_edge.parent_idx, current_idx,
                mini_batch_ids=mini_batch_ids,
                score_before=initial_pass_rate,
                score_after=reeval_pass_rate,
                scenario_impacts=scenario_impacts,
                code_diff=cp_diff,
                files_changed=cp_files,
            )

        dag.save()

        # ── Step 10: DAG update (verdict without reflections) ─────────
        #
        # Verdict is recorded now with scenario_impacts but without
        # reflections. Session 2 runs at the start of the next iteration
        # (after the engine has completed dev-set eval), so reflections
        # and lessons are added then.

        dag.load()
        node = dag.nodes[current_idx]

        fixed = [si for si in scenario_impacts if si.status_change == "fixed"]
        regressed = [si for si in scenario_impacts if si.status_change == "regressed"]

        intent = node.patch_intent
        target_scenarios = intent.target_scenarios if intent else []
        effectiveness = any(
            si.status_change == "fixed"
            for si in scenario_impacts
            if si.scenario_id in target_scenarios
        ) if target_scenarios else len(fixed) > 0
        safety = len(regressed) == 0

        verdict = PatchVerdict(
            is_good_patch=effectiveness and safety,
            effectiveness=effectiveness,
            safety=safety,
            scenario_impacts=scenario_impacts,
            reflections=[],
            lessons_learned=[],
        )
        dag.update_patch_verdict(current_idx, verdict)

        update_scenario_registry(dag, node, scenario_impacts)

        self._logger.log(
            f"Verdict: good_patch={verdict.is_good_patch} "
            f"(effectiveness={effectiveness}, safety={safety})"
        )

        dag.save()

        # ── Step 11: Return proposal ──────────────────────────────────

        subsample_before = [initial_scores.get(sid, 0.0) for sid in mini_batch_ids]
        subsample_after = [
            reeval_results.get(sid, {}).get("score", 0.0) for sid in mini_batch_ids
        ]

        candidate: dict[str, str] = {
            "__autosaddler_worktree__": str(new_worktree),
        }

        state_parent_idx = self._resolve_state_parent_idx(dag, base_parent_idx, state)

        proposal = CandidateProposal(
            candidate=candidate,
            parent_program_ids=[state_parent_idx],
            subsample_indices=mini_batch_ids,
            subsample_scores_before=subsample_before,
            subsample_scores_after=subsample_after,
            tag="evolution_dag_v2",
            metadata={
                "reasoning": self._read_reasoning(new_worktree),
                "files_modified": files_changed,
                "commit_hash": commit_hash,
                "meta_iteration": self._meta_iteration,
                "candidate_idx": current_idx,
                "initial_pass_rate": initial_pass_rate,
                "reeval_pass_rate": reeval_pass_rate,
                "is_good_patch": verdict.is_good_patch,
                "fixed_count": len(fixed),
                "regressed_count": len(regressed),
            },
        )

        self._logger.log(
            f"Proposal ready: C{current_idx} "
            f"(initial={initial_pass_rate:.4f}, reeval={reeval_pass_rate:.4f})"
        )

        return proposal

    # ------------------------------------------------------------------
    # Mini-batch sampling
    # ------------------------------------------------------------------

    def _sample_mini_batch(
        self,
        state: GEPAState,
        forced_action: str | None = None,
        *,
        artifact_dir: str | Path | None = None,
        candidate_idx: int | None = None,
    ) -> tuple[list[str], list]:
        """Sample a mini-batch from the training set.

        ``forced_action`` is the agent's arm-creation decision for the
        ActiveSaddlerBanditSampler; ignored by other samplers.
        """
        if self._batch_sampler is not None:
            from autosaddler.v1.core.data_loader import ListDataLoader

            loader = ListDataLoader(self.trainset)
            if isinstance(self._batch_sampler, ActiveSaddlerBanditSampler):
                ids = self._batch_sampler.next_minibatch_ids(
                    loader, state, forced_action=forced_action,
                )
            else:
                ids = self._batch_sampler.next_minibatch_ids(loader, state)
            batch = loader.fetch(ids)
            scenario_ids = [str(sid) for sid in ids]

            # Mark as executed for ActiveSaddlerBanditSampler unseen pool tracking
            if isinstance(self._batch_sampler, ActiveSaddlerBanditSampler):
                if artifact_dir is None or candidate_idx is None:
                    raise RuntimeError(
                        "Bandit sampling requires a reserved train-before "
                        "artifact directory and candidate index"
                    )
                self._batch_sampler.mark_executed(ids)
                self._persist_sampler_snapshot(
                    self._batch_sampler.last_score_snapshot,
                    artifact_dir=artifact_dir,
                    candidate_idx=candidate_idx,
                )
        else:
            batch = self.trainset
            scenario_ids = [str(i) for i in range(len(batch))]

        return scenario_ids, batch

    def _sampler_trace_path(
        self,
        iteration: int,
        artifact_dir: str | Path,
        candidate_idx: int,
    ) -> Path:
        canonical_path = iteration_artifact_path(
            artifact_dir,
            iteration,
            candidate_idx,
            "sampler_trace",
        )
        legacy_path = (
            Path(self._adapter._session_root)
            / "sampler_trace"
            / f"iter{iteration:02d}.json"
        )
        if canonical_path.exists() or not legacy_path.exists():
            return canonical_path
        return legacy_path

    def _persist_sampler_snapshot(
        self,
        snapshot: dict | None,
        *,
        artifact_dir: str | Path,
        candidate_idx: int,
    ) -> None:
        """Persist the sampler's per-iteration decision snapshot.

        Captures the agent's decision (arm pull vs. unseen draw), each arm's
        score and sampling probability, and the selected mini-batch.
        Trace persistence is part of the deterministic resume contract; write
        failures therefore stop the run instead of silently losing replay state.
        """
        if not snapshot:
            return
        snapshot = {
            **snapshot,
            "strategy": self._config.strategy.name.value,
            "scoring_mode": self._config.strategy.scoring,
            "arm_creation": self._config.strategy.arm_creation,
        }
        iteration = int(snapshot.get("iteration", 0))
        snapshot["candidate_idx"] = candidate_idx
        out_path = self._sampler_trace_path(
            iteration,
            artifact_dir,
            candidate_idx,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = out_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        os.replace(temporary, out_path)
        self._logger.log(
            f"Saved sampler score snapshot → {out_path} "
            f"(action={snapshot.get('action')}, {snapshot.get('num_arms', 0)} arms)"
        )

    def _persist_sampler_probe_points(
        self,
        iteration: int,
        added_probe_points: list[str],
        *,
        artifact_dir: str | Path,
        candidate_idx: int,
    ) -> None:
        """Merge exact probe-point additions into an existing sampler trace."""
        trace_path = self._sampler_trace_path(
            iteration,
            artifact_dir,
            candidate_idx,
        )
        if not trace_path.exists():
            raise RuntimeError(
                f"Sampler trace is missing while recording probe points: {trace_path}"
            )
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        merged = list(
            dict.fromkeys(
                [
                    *(trace.get("probe_points_added") or []),
                    *added_probe_points,
                ]
            )
        )
        trace["probe_points_added"] = merged
        self._persist_sampler_snapshot(
            trace,
            artifact_dir=artifact_dir,
            candidate_idx=candidate_idx,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        state: GEPAState,
        node: Any,
        batch: list,
        mini_batch_ids: list[str],
        worktree_override: Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate a candidate on a batch and return the cycle dir."""
        worktree = worktree_override or Path(node.worktree_path)
        candidate = {"__autosaddler_worktree__": str(worktree)}

        self._adapter.evaluate(
            batch=batch,
            candidate=candidate,
            capture_traces=True,
        )
        cycle_dir = str(self._adapter.last_cycle_dir) if self._adapter.last_cycle_dir else ""
        if not cycle_dir:
            raise RuntimeError(
                f"Evaluation did not produce a cycle_dir "
                f"(worktree={worktree}, batch_size={len(batch)})"
            )
        return {"cycle_dir": cycle_dir}

    # ------------------------------------------------------------------
    # Worktree management
    # ------------------------------------------------------------------

    def _create_worktree(self, iteration: int, candidate_idx: int) -> Path:
        """Create a new git worktree from the base branch."""
        new_id = f"worktree_{iteration:02d}_{candidate_idx}_{uuid4().hex[:8]}"
        new_path = self._adapter._worktree_dir / new_id
        branch_name = f"evolution-dag-v2/{new_id}"
        base_branch = self._adapter.cfg.base_branch

        base_repo = git.Repo(self._adapter._repo_path)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            base_repo.git.worktree(
                "add", str(new_path), "-b", branch_name, base_branch,
            )
        except git.GitCommandError:
            logger.warning("Stale branch/worktree for %s — cleaning up", new_id)
            try:
                base_repo.git.worktree("remove", str(new_path), "--force")
            except Exception:
                shutil.rmtree(new_path, ignore_errors=True)
            try:
                base_repo.git.worktree("prune")
            except Exception:
                pass
            try:
                base_repo.git.branch("-D", branch_name)
            except Exception:
                pass
            base_repo.git.worktree(
                "add", str(new_path), "-b", branch_name, base_branch,
            )

        return new_path

    def _fork_worktree(self, parent_worktree: Path, iteration: int) -> Path:
        """Fork a new worktree from a parent worktree's HEAD commit."""
        new_id = f"worktree_{iteration:02d}_{uuid4().hex[:8]}"
        new_path = self._adapter._worktree_dir / new_id
        branch_name = f"evolution-dag-v2/{new_id}"

        parent_repo = git.Repo(parent_worktree)
        parent_commit = parent_repo.head.commit.hexsha

        base_repo = git.Repo(self._adapter._repo_path)
        new_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            base_repo.git.worktree(
                "add", str(new_path), "-b", branch_name, parent_commit,
            )
        except git.GitCommandError:
            logger.warning("Stale branch/worktree for %s — cleaning up", new_id)
            try:
                base_repo.git.worktree("remove", str(new_path), "--force")
            except Exception:
                shutil.rmtree(new_path, ignore_errors=True)
            try:
                base_repo.git.worktree("prune")
            except Exception:
                pass
            try:
                base_repo.git.branch("-D", branch_name)
            except Exception:
                pass
            base_repo.git.worktree(
                "add", str(new_path), "-b", branch_name, parent_commit,
            )

        logger.info("Forked worktree %s from %s", new_path.name, parent_commit[:12])
        return new_path

    def _commit_changes(self, worktree: Path) -> str | None:
        """Commit all changes in the worktree."""
        try:
            repo = git.Repo(worktree)
            repo.git.add("-A")
            if repo.is_dirty(index=True):
                repo.git.commit(
                    "-m", f"evolution-dag-v2: iteration {self._meta_iteration}",
                    "--allow-empty",
                )
            return repo.head.commit.hexsha
        except Exception:
            logger.exception("Failed to commit changes in %s", worktree)
            return None

    @staticmethod
    def _capture_changed_files(worktree: Path) -> list[str]:
        """List files modified relative to the parent commit."""
        try:
            repo = git.Repo(worktree)
            changed: list[str] = []
            try:
                parent = repo.head.commit.parents[0] if repo.head.commit.parents else None
                if parent:
                    diffs = parent.diff(repo.head.commit)
                    changed.extend(d.a_path or d.b_path for d in diffs if d.a_path or d.b_path)
            except Exception:
                pass
            staged = [item.a_path for item in repo.index.diff("HEAD")]
            changed.extend(staged)
            unstaged = [item.a_path for item in repo.index.diff(None)]
            changed.extend(unstaged)
            changed.extend(repo.untracked_files)
            filtered = [
                f for f in set(changed)
                if not f.startswith(".claude/")
                and not f.startswith("bin/")
                and f != "CLAUDE.md"
            ]
            return sorted(filtered)
        except Exception:
            return []

    @staticmethod
    def _compute_diff(parent_worktree: Path, child_worktree: Path) -> str:
        """Compute git diff between parent and child worktrees."""
        try:
            result = subprocess.run(
                ["diff", "-ruN", "--exclude=.git", "--exclude=.claude",
                 "--exclude=bin", "--exclude=CLAUDE.md",
                 "--exclude=__pycache__", "--exclude=*.pyc",
                 "--exclude=proposer_reasoning.md",
                 str(parent_worktree), str(child_worktree)],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
            )
            return result.stdout or ""
        except Exception:
            logger.exception("Failed to compute diff")
            return ""

    @staticmethod
    def _extract_files_from_diff(diff_output: str) -> list[str]:
        """Extract changed file paths from a unified diff output."""
        files: set[str] = set()
        for line in diff_output.split("\n"):
            if line.startswith("diff "):
                # diff -ruN produces lines like: diff -ruN a/path/to/file b/path/to/file
                parts = line.split()
                if len(parts) >= 4:
                    # Take the second path (b/...) and strip leading directory
                    path = parts[-1]
                    # Find the first real path component after the worktree prefix
                    for prefix in ("/are/", "/src/", "/config/", "/hook.json"):
                        idx = path.find(prefix)
                        if idx >= 0:
                            path = path[idx + 1:]
                            break
                    if not path.startswith("/"):
                        files.add(path)
        filtered = [
            f for f in files
            if not f.startswith(".claude/")
            and not f.startswith("bin/")
            and f != "CLAUDE.md"
        ]
        return sorted(filtered)

    @staticmethod
    def _read_reasoning(worktree: Path) -> str:
        """Read proposer_reasoning.md if present."""
        path = worktree / "proposer_reasoning.md"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass
        return "(no reasoning provided)"

    # ------------------------------------------------------------------
    # SDK session
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_worktree_retry_state(worktree_path: Path) -> dict[str, Any]:
        """Capture the exact tracked and untracked state at an attempt boundary."""
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        staged = subprocess.run(
            ["git", "diff", "--cached", "--binary"],
            cwd=worktree_path,
            check=True,
            capture_output=True,
        ).stdout
        unstaged = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=worktree_path,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=worktree_path,
            check=True,
            capture_output=True,
        ).stdout
        untracked: dict[str, tuple[bytes, int]] = {}
        for raw_path in untracked_output.split(b"\0"):
            if not raw_path:
                continue
            relative = raw_path.decode(errors="surrogateescape")
            path = worktree_path / relative
            if path.is_file():
                untracked[relative] = (path.read_bytes(), path.stat().st_mode)
        return {
            "head": head,
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
        }

    @staticmethod
    def _restore_worktree_retry_state(
        worktree_path: Path,
        snapshot: dict[str, Any],
    ) -> None:
        """Restore a worktree to a captured attempt boundary."""
        subprocess.run(
            ["git", "reset", "--hard", snapshot["head"]],
            cwd=worktree_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=worktree_path,
            check=True,
            capture_output=True,
        )
        for patch, apply_args in (
            (snapshot["staged"], ["git", "apply", "--index", "--binary", "-"]),
            (snapshot["unstaged"], ["git", "apply", "--binary", "-"]),
        ):
            if patch:
                subprocess.run(
                    apply_args,
                    cwd=worktree_path,
                    input=patch,
                    check=True,
                    capture_output=True,
                )
        for relative, (content, mode) in snapshot["untracked"].items():
            path = worktree_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            os.chmod(path, mode)

    @staticmethod
    def _sdk_retry_state_paths(
        extra_env: dict[str, str] | None,
        *,
        session_type: str | None,
        iteration: int | None,
        candidate_idx: int | None,
        artifact_dir: str | Path | None,
    ) -> list[Path]:
        """Return non-worktree state files an SDK attempt may mutate."""
        del session_type, iteration, candidate_idx, artifact_dir
        paths: set[Path] = set()
        for key in ("EVOLUTION_DAG_PATH", "PATTERN_REGISTRY_PATH"):
            value = (extra_env or {}).get(key)
            if not value:
                continue
            path = Path(value)
            paths.add(path)
            if path.suffix == ".json":
                paths.add(path.with_suffix(".json.gz"))
        return sorted(paths)

    @staticmethod
    def _capture_retry_files(paths: list[Path]) -> dict[Path, bytes | None]:
        return {
            path: path.read_bytes() if path.is_file() else None
            for path in paths
        }

    @staticmethod
    def _restore_retry_files(snapshot: dict[Path, bytes | None]) -> None:
        for path, content in snapshot.items():
            if content is None:
                path.unlink(missing_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, path)

    @staticmethod
    def _sdk_error_artifact_path(
        *,
        session_type: str | None,
        iteration: int | None,
        candidate_idx: int | None,
        artifact_dir: str | Path | None,
    ) -> Path | None:
        if (
            not artifact_dir
            or session_type is None
            or iteration is None
            or candidate_idx is None
        ):
            return None
        return iteration_artifact_path(
            artifact_dir,
            iteration,
            candidate_idx,
            f"{session_type}_error",
        )

    @staticmethod
    def _write_sdk_error_artifact(
        path: Path | None,
        *,
        session_type: str | None,
        iteration: int | None,
        candidate_idx: int | None,
        attempts: list[dict[str, Any]],
    ) -> None:
        if path is None:
            return
        payload = {
            "session_type": session_type,
            "iteration": iteration,
            "candidate_idx": candidate_idx,
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        try:
            from autosaddler.v1.sdk_metrics import (
                session_root_from_artifact_dir,
                write_run_sdk_metrics,
            )

            metrics_root = session_root_from_artifact_dir(path.parent)
            if metrics_root is not None:
                write_run_sdk_metrics(metrics_root)
        except Exception:
            logger.exception("Failed to rebuild SDK metrics after terminal failure")

    @staticmethod
    def _sdk_attempt_metrics(result_or_error: Any) -> dict[str, Any]:
        """Extract auditable metrics from a result or an exception's partial result."""
        if isinstance(result_or_error, dict):
            result = result_or_error
        else:
            result = getattr(result_or_error, "session_result", None)
        if not isinstance(result, dict):
            return {"accounting_complete": False}
        usage = result.get("usage") or []
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens": 0,
        }
        aliases = {
            "input_tokens": ("input_tokens", "promptTokens"),
            "output_tokens": ("output_tokens", "completionTokens"),
            "cache_read_input_tokens": (
                "cache_read_input_tokens",
                "cache_read_tokens",
            ),
            "cache_creation_input_tokens": (
                "cache_creation_input_tokens",
                "cache_write_tokens",
            ),
            "reasoning_tokens": ("reasoning_tokens",),
        }
        for item in usage:
            if not isinstance(item, dict):
                continue
            for output_name, source_names in aliases.items():
                for source_name in source_names:
                    value = item.get(source_name)
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        totals[output_name] += int(value)
                        break
        meta = result.get("result_meta") or {}
        return {
            "accounting_complete": meta.get("total_cost_usd") is not None,
            "outcome": meta.get("outcome"),
            "session_id": meta.get("session_id"),
            "wall_clock_s": result.get("wall_clock_s", 0.0) or 0.0,
            "llm_call_count": meta.get("llm_call_count", len(usage)) or 0,
            **totals,
            "copilot_nano_aiu": meta.get("copilot_nano_aiu"),
            "reported_cost_usd": meta.get("reported_cost_usd"),
            "metered_cost_usd": meta.get("metered_cost_usd"),
            "estimated_cost_usd": meta.get("estimated_cost_usd"),
            "total_cost_usd": meta.get("total_cost_usd"),
            "cost_source": meta.get("cost_source"),
            "cost_is_estimate": meta.get("cost_is_estimate", False),
            "model_usage": meta.get("model_usage"),
            "usage": usage,
        }

    @staticmethod
    def _logical_attempt_totals(attempts: list[dict[str, Any]]) -> dict[str, Any]:
        token_fields = (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "reasoning_tokens",
            "llm_call_count",
        )
        complete = bool(attempts) and all(
            attempt.get("accounting_complete", False) for attempt in attempts
        )
        cost_fields = (
            "reported_cost_usd",
            "metered_cost_usd",
            "estimated_cost_usd",
            "total_cost_usd",
        )
        nano_aiu = [
            float(attempt["copilot_nano_aiu"])
            for attempt in attempts
            if attempt.get("copilot_nano_aiu") is not None
        ]
        return {
            "attempt_accounting_complete": complete,
            "logical_copilot_nano_aiu": sum(nano_aiu) if nano_aiu else None,
            **{
                f"logical_{field}": (
                    sum(
                        float(attempt[field])
                        for attempt in attempts
                        if attempt.get(field) is not None
                    )
                    if complete
                    and any(attempt.get(field) is not None for attempt in attempts)
                    else None
                )
                for field in cost_fields
            },
            **{
                f"logical_{field}": sum(
                    int(attempt.get(field, 0) or 0) for attempt in attempts
                )
                for field in token_fields
            },
            "logical_wall_clock_s": sum(
                float(attempt.get("wall_clock_s", 0.0) or 0.0)
                for attempt in attempts
            ),
        }

    def _run_sdk_session(
        self,
        worktree_path: Path,
        prompt: str,
        model: str,
        timeout: float,
        extra_env: dict[str, str] | None = None,
        session_type: str | None = None,
        iteration: int | None = None,
        candidate_idx: int | None = None,
        artifact_dir: str | Path | None = None,
        max_retries: int = 5,
        initial_backoff: float = 60.0,
    ) -> dict[str, Any] | None:
        """Run fresh SDK attempts with transactional retry and an audit ledger."""
        import time

        from autosaddler.v1.sdk_session import ContentFilterError, RateLimitError

        retry_policy = self._config.sdk_config.retry
        backoff = initial_backoff
        rate_limit_failures = 0
        content_filter_retries = 0
        attempt_count = 0
        failed_attempts: list[dict[str, Any]] = []
        worktree_snapshot = self._capture_worktree_retry_state(worktree_path)
        retry_file_snapshot = self._capture_retry_files(
            self._sdk_retry_state_paths(
                extra_env,
                session_type=session_type,
                iteration=iteration,
                candidate_idx=candidate_idx,
                artifact_dir=artifact_dir,
            )
        )
        error_artifact_path = self._sdk_error_artifact_path(
            session_type=session_type,
            iteration=iteration,
            candidate_idx=candidate_idx,
            artifact_dir=artifact_dir,
        )
        if error_artifact_path is not None:
            error_artifact_path.unlink(missing_ok=True)

        while True:
            attempt_count += 1
            try:
                session_result = _run_async(
                    self._async_sdk_session(
                        worktree_path=worktree_path,
                        prompt=prompt,
                        model=model,
                        timeout=timeout,
                        extra_env=extra_env,
                    )
                )
                session_result["retry"] = {
                    "attempt_count": attempt_count,
                    "content_filter_retries": content_filter_retries,
                    "rate_limit_retries": rate_limit_failures,
                }
                final_attempt = {
                    "attempt": attempt_count,
                    "classification": "success",
                    "will_retry": False,
                    **self._sdk_attempt_metrics(session_result),
                }
                session_result["attempts"] = [
                    *failed_attempts,
                    final_attempt,
                ]
                result_meta = session_result.setdefault("result_meta", {})
                result_meta["final_attempt_cost_usd"] = result_meta.get(
                    "total_cost_usd"
                )
                result_meta.update(
                    self._logical_attempt_totals(session_result["attempts"])
                )
                return session_result
            except ContentFilterError as exc:
                can_retry = (
                    content_filter_retries
                    < retry_policy.content_filter_max_retries
                )
                retry_number = content_filter_retries + 1
                delay = (
                    retry_policy.delay_for_content_filter_retry(retry_number)
                    if can_retry
                    else 0.0
                )
                failed_attempts.append(
                    {
                        "attempt": attempt_count,
                        "classification": "content_filter",
                        **exc.to_dict(),
                        "will_retry": can_retry,
                        "retry_delay_s": delay,
                        **self._sdk_attempt_metrics(exc),
                    }
                )
                self._restore_worktree_retry_state(worktree_path, worktree_snapshot)
                self._restore_retry_files(retry_file_snapshot)
                if can_retry:
                    content_filter_retries += 1
                    logger.warning(
                        "Content filter blocked SDK session (retry %d/%d). "
                        "Starting a fresh session in %.0fs...",
                        content_filter_retries,
                        retry_policy.content_filter_max_retries,
                        delay,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue
                self._write_sdk_error_artifact(
                    error_artifact_path,
                    session_type=session_type,
                    iteration=iteration,
                    candidate_idx=candidate_idx,
                    attempts=failed_attempts,
                )
                return None
            except RateLimitError as exc:
                rate_limit_failures += 1
                can_retry = rate_limit_failures < max_retries
                failed_attempts.append(
                    {
                        "attempt": attempt_count,
                        "classification": "rate_limit",
                        "message": str(exc),
                        "will_retry": can_retry,
                        "retry_delay_s": backoff if can_retry else 0.0,
                        **self._sdk_attempt_metrics(exc),
                    }
                )
                self._restore_worktree_retry_state(worktree_path, worktree_snapshot)
                self._restore_retry_files(retry_file_snapshot)
                if can_retry:
                    logger.warning(
                        "Rate-limited (attempt %d/%d). Retrying in %.0fs...",
                        rate_limit_failures,
                        max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 600.0)
                    continue
                self._write_sdk_error_artifact(
                    error_artifact_path,
                    session_type=session_type,
                    iteration=iteration,
                    candidate_idx=candidate_idx,
                    attempts=failed_attempts,
                )
                return None
            except Exception as exc:
                failed_attempts.append(
                    {
                        "attempt": attempt_count,
                        "classification": "non_retryable",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "will_retry": False,
                        "retry_delay_s": 0.0,
                        **self._sdk_attempt_metrics(exc),
                    }
                )
                self._restore_worktree_retry_state(worktree_path, worktree_snapshot)
                self._restore_retry_files(retry_file_snapshot)
                logger.exception("SDK session failed")
                self._write_sdk_error_artifact(
                    error_artifact_path,
                    session_type=session_type,
                    iteration=iteration,
                    candidate_idx=candidate_idx,
                    attempts=failed_attempts,
                )
                return None

    async def _async_sdk_session(
        self,
        worktree_path: Path,
        prompt: str,
        model: str,
        timeout: float,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Async SDK session with extra env vars for evo-dag CLI."""
        from autosaddler.v1.sdk_session import run_sdk_session

        old_env: dict[str, str | None] = {}
        if extra_env:
            for key, value in extra_env.items():
                old_env[key] = os.environ.get(key)
                os.environ[key] = value

        try:
            result = await run_sdk_session(
                cwd=worktree_path,
                prompt=prompt,
                model=model,
                timeout=timeout,
                sdk_config=self._config.sdk_config,
                track_events=True,
            )
            return result
        finally:
            if extra_env:
                for key in extra_env:
                    if old_env.get(key) is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_env[key]

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_worktree(self, worktree: Path) -> bool:
        """Verify the modified worktree (syntax + import check).

        Syntax check uses the current Python since py_compile
        doesn't need third-party packages.  Import check uses the target
        worktree's Python environment (via the adapter's activate_command)
        because the worktree code depends on packages only installed there.
        """
        self._logger.log("Verifying modified worktree...")

        modified_py = [
            f for f in self._capture_changed_files(worktree)
            if f.endswith(".py")
        ]
        for py_file in modified_py:
            py_path = worktree / py_file
            if not py_path.exists():
                continue
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(py_path)],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                self._logger.log(
                    f"Syntax error in {py_file}:\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                return False

        # Import check: run in the worktree's own Python environment
        # so that third-party dependencies (inputimeout, mammoth, etc.)
        # are available.
        activate_cmd = getattr(
            getattr(self._adapter, "cfg", None), "activate_command", ""
        )
        import_check = getattr(
            getattr(self._adapter, "cfg", None), "import_check_statement",
            "pass"  # generic default; adapter config should provide the real statement
        )
        if activate_cmd:
            shell_cmd = f"{activate_cmd} && PYTHONPATH={worktree} python -c \"{import_check}\""
            result = subprocess.run(
                ["bash", "-c", shell_cmd],
                cwd=str(worktree),
                capture_output=True,
                timeout=30,
            )
        else:
            # Fallback: use current python (may fail if deps are missing)
            result = subprocess.run(
                ["python3", "-c", import_check],
                cwd=str(worktree),
                capture_output=True,
                timeout=30,
                env={**dict(os.environ), "PYTHONPATH": str(worktree)},
            )
        if result.returncode != 0:
            self._logger.log(
                f"Import check failed:\n"
                f"{result.stderr.decode(errors='replace')}"
            )
            return False

        self._logger.log("Verification PASSED")
        return True

    # ------------------------------------------------------------------
    # Session info extraction
    # ------------------------------------------------------------------

    def _extract_session_info(
        self,
        session_result: dict[str, Any],
        model: str,
        timeout: float,
        output_dir: str,
        session_type: str,
        iteration: int,
        candidate_idx: int,
    ) -> SDKSessionInfo | None:
        """Extract SDKSessionInfo from a session result and dump JSON."""
        try:
            tool_calls = session_result.get("tool_calls", [])
            turns = session_result.get("turns", 0)
            usage = session_result.get("usage") or []

            input_tokens = 0
            output_tokens = 0
            cache_read = 0
            cache_creation = 0
            reasoning_tokens = 0
            for u in usage:
                if isinstance(u, dict):
                    input_tokens += u.get("input_tokens", 0) or u.get("promptTokens", 0) or 0
                    output_tokens += u.get("output_tokens", 0) or u.get("completionTokens", 0) or 0
                    cache_read += (
                        u.get("cache_read_input_tokens", 0)  # Claude SDK
                        or u.get("cache_read_tokens", 0)     # Copilot SDK
                        or 0
                    )
                    cache_creation += u.get("cache_creation_input_tokens", 0) or 0
                    reasoning_tokens += u.get("reasoning_tokens", 0) or 0

            wall_clock_s = session_result.get("wall_clock_s", 0.0) or 0.0
            meta = session_result.get("result_meta") or {}
            attempts = session_result.get("attempts") or []
            attempt_accounting_complete = meta.get(
                "attempt_accounting_complete",
                not attempts or len(attempts) == 1,
            )
            wall_clock_s = meta.get("logical_wall_clock_s", wall_clock_s) or 0.0
            model_usage = meta.get("model_usage")
            inclusive_usage = aggregate_model_usage(model_usage)
            if inclusive_usage is not None:
                input_tokens = int(inclusive_usage["input_tokens"])
                output_tokens = int(inclusive_usage["output_tokens"])
                cache_read = int(inclusive_usage["cache_read_input_tokens"])
                cache_creation = int(inclusive_usage["cache_creation_input_tokens"])
            total_cost_usd = meta.get("total_cost_usd")
            if attempts:
                total_cost_usd = (
                    meta.get("logical_total_cost_usd")
                    if attempt_accounting_complete
                    else None
                )
            if (
                total_cost_usd is None
                and inclusive_usage is not None
                and (not attempts or attempt_accounting_complete)
            ):
                total_cost_usd = float(inclusive_usage["total_cost_usd"])
            if meta.get("logical_input_tokens") is not None:
                input_tokens = int(meta["logical_input_tokens"])
                output_tokens = int(meta["logical_output_tokens"])
                cache_read = int(meta["logical_cache_read_input_tokens"])
                cache_creation = int(meta["logical_cache_creation_input_tokens"])
                reasoning_tokens = int(meta["logical_reasoning_tokens"])

            if not output_dir:
                logger.warning(
                    "Empty output_dir for %s session (iter=%d, C%d) — "
                    "skipping JSON dump to avoid writing to CWD",
                    session_type, iteration, candidate_idx,
                )
                return None

            out_path = Path(output_dir)
            if not out_path.is_absolute():
                logger.warning(
                    "Relative output_dir %r for %s session — "
                    "skipping JSON dump to avoid writing to CWD",
                    output_dir, session_type,
                )
                return None

            out_path.mkdir(parents=True, exist_ok=True)
            json_path = out_path / f"iter{iteration:02d}_c{candidate_idx}_{session_type}.json"

            session_data = {
                "model": model,
                "timeout": timeout,
                "session_type": session_type,
                "iteration": iteration,
                "candidate_idx": candidate_idx,
                "tool_call_count": len(tool_calls),
                "turns": turns,
                "wall_clock_s": wall_clock_s,
                "duration_ms": meta.get("duration_ms"),
                "duration_api_ms": meta.get("duration_api_ms"),
                "num_turns": meta.get("num_turns"),
                "llm_call_count": meta.get(
                    "logical_llm_call_count",
                    meta.get("llm_call_count", len(usage)),
                ),
                "usage_event_count": meta.get("usage_event_count", len(usage)),
                "duplicate_usage_event_count": meta.get(
                    "duplicate_usage_event_count", 0
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
                "reasoning_tokens": reasoning_tokens,
                "copilot_nano_aiu": meta.get(
                    "logical_copilot_nano_aiu", meta.get("copilot_nano_aiu")
                ),
                "reported_cost_usd": meta.get(
                    "logical_reported_cost_usd", meta.get("reported_cost_usd")
                ),
                "metered_cost_usd": meta.get(
                    "logical_metered_cost_usd", meta.get("metered_cost_usd")
                ),
                "estimated_cost_usd": meta.get(
                    "logical_estimated_cost_usd", meta.get("estimated_cost_usd")
                ),
                "total_cost_usd": total_cost_usd,
                "final_attempt_cost_usd": meta.get("final_attempt_cost_usd"),
                "attempt_count": len(attempts) or 1,
                "attempt_accounting_complete": attempt_accounting_complete,
                "cost_source": meta.get("cost_source"),
                "cost_is_estimate": meta.get("cost_is_estimate", False),
                "session_id": meta.get("session_id"),
                "model_usage": model_usage,
                "retry": session_result.get("retry"),
                "attempts": attempts,
                "tool_calls": tool_calls,
                "usage": usage,
                "raw_response": session_result.get("raw_response", ""),
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            try:
                from autosaddler.v1.sdk_metrics import (
                    session_root_from_artifact_dir,
                    write_run_sdk_metrics,
                )

                metrics_root = session_root_from_artifact_dir(out_path)
                if metrics_root is not None:
                    write_run_sdk_metrics(metrics_root)
            except Exception:
                logger.exception("Failed to rebuild run-level SDK metrics")

            return SDKSessionInfo(
                model=model,
                timeout=timeout,
                tool_call_count=len(tool_calls),
                turns=turns,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read,
                session_json_path=str(json_path),
                wall_clock_s=wall_clock_s,
                duration_ms=meta.get("duration_ms", 0) or 0,
                duration_api_ms=meta.get("duration_api_ms", 0) or 0,
                num_turns=meta.get("num_turns", 0) or 0,
                total_cost_usd=total_cost_usd,
                cache_creation_input_tokens=cache_creation,
                model_usage=model_usage,
                reasoning_tokens=reasoning_tokens,
                llm_call_count=meta.get(
                    "logical_llm_call_count",
                    meta.get("llm_call_count", len(usage)),
                ) or 0,
                usage_event_count=meta.get("usage_event_count", len(usage)) or 0,
                duplicate_usage_event_count=meta.get(
                    "duplicate_usage_event_count", 0
                ) or 0,
                copilot_nano_aiu=meta.get(
                    "logical_copilot_nano_aiu", meta.get("copilot_nano_aiu")
                ),
                reported_cost_usd=meta.get(
                    "logical_reported_cost_usd", meta.get("reported_cost_usd")
                ),
                metered_cost_usd=meta.get(
                    "logical_metered_cost_usd", meta.get("metered_cost_usd")
                ),
                estimated_cost_usd=meta.get(
                    "logical_estimated_cost_usd", meta.get("estimated_cost_usd")
                ),
                cost_source=meta.get("cost_source"),
                cost_is_estimate=meta.get("cost_is_estimate", False),
                attempt_count=len(attempts) or 1,
                attempt_accounting_complete=attempt_accounting_complete,
                final_attempt_cost_usd=meta.get("final_attempt_cost_usd"),
            )
        except Exception:
            logger.exception("Failed to extract session info")
            return None
