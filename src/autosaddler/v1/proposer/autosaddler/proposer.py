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
from typing import Any
from uuid import uuid4

import git

from autosaddler.v1.core.data_loader import DataId
from autosaddler.v1.core.state import GEPAState
from autosaddler.v1.proposer.base import CandidateProposal, ProposeNewCandidate
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
    PatchIntent,
    PatchVerdict,
    SDKSessionInfo,
)
from autosaddler.v1.proposer.autosaddler.prompt_builder import (
    build_claude_md,
    build_session0_prompt,
    build_session1_prompt,
    build_session2_prompt,
    build_skill_prefix,
    install_evo_dag_cli,
    install_prompts_and_skills,
)
from autosaddler.v1.sdk_session import SdkConfig
from autosaddler.v1.strategies.batch_sampler import EpochShuffledBatchSampler

logger = logging.getLogger(__name__)


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

    # Session 0 control
    skip_session0: bool = False  # Skip candidate selection for first few iterations

    @property
    def active_model(self) -> str:
        """Return the model name for the active SDK backend."""
        if self.sdk_config.backend == "copilot":
            return self.sdk_config.copilot_model or self.copilot_model
        return self.claude_agent_sdk_model


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

        # DAG instance — initialized on first propose()
        self._dag: EvolutionDAG | None = None

        # Mini-batch sampler
        mbs = config.train_minibatch_size
        if mbs > 0:
            self._batch_sampler: EpochShuffledBatchSampler | None = (
                EpochShuffledBatchSampler(
                    minibatch_size=mbs, rng=random.Random(config.seed),
                )
            )
        else:
            self._batch_sampler = None

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
        if self._config.capability_phase_iterations > 0:
            capability_iterations = self._config.capability_phase_iterations
        else:
            trainset_size = len(self.trainset)
            mbs = self._config.train_minibatch_size
            iterations_per_epoch = (trainset_size + mbs - 1) // mbs
            capability_iterations = self._config.capability_phase_epochs * iterations_per_epoch
        return "capability" if iteration <= capability_iterations else "steering"

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
            )

            cli_env = install_evo_dag_cli(session_root, str(worktree))

            session2_result = self._run_sdk_session(
                worktree_path=worktree,
                prompt=build_skill_prefix(session=2, phase=self._get_phase_for_iteration(node.iteration)) + session2_prompt,
                model=self._config.active_model,
                timeout=self._config.reflection_timeout,
                extra_env=cli_env,
            )

            reflection_output_dir = (
                node.train_before_cycle_dir
                or node.train_after_cycle_dir
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
    # ProposeNewCandidate protocol
    # ------------------------------------------------------------------

    def propose(
        self, state: GEPAState[Any, DataId],
    ) -> CandidateProposal | None:
        """Propose a new candidate. One candidate per iteration."""
        if self._pending_proposals:
            return self._pending_proposals.popleft()

        self._meta_iteration += 1
        self._logger.log(
            f"\n{'='*60}\n"
            f"EVOLUTION-DAG v2 ITERATION {self._meta_iteration} "
            f"(engine iter {state.i})\n"
            f"{'='*60}"
        )

        try:
            proposal = self._run_iteration(state)
        except Exception:
            logger.exception(
                "Failed in EvolutionDAG v2 iteration %d", self._meta_iteration,
            )
            return None

        if proposal is None:
            self._logger.log("EVOLUTION-DAG v2: no valid candidate generated")
            return None

        return proposal

    def finalize(self, state: GEPAState) -> None:
        """Run final reflection for the last iteration.

        Called by the engine after the main loop exits so that the last
        iteration's deferred Session 2 (reflection) is not skipped.
        At this point the engine has already completed the dev-set
        evaluation for the last candidate, so val scores are available.
        """
        try:
            dag = self._ensure_dag(state)
            self._sync_val_scores_from_state(dag, state)
            self._run_deferred_session2(dag, state)
        except Exception:
            logger.exception("finalize: deferred session 2 failed (non-fatal)")

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

        # Restore _meta_iteration from DAG to avoid duplicate iteration
        # numbers after process restart
        if dag.nodes:
            max_iteration = max(n.iteration for n in dag.nodes.values())
            if max_iteration > self._meta_iteration:
                logger.info(
                    "Restoring _meta_iteration from DAG: %d → %d",
                    self._meta_iteration, max_iteration,
                )
                self._meta_iteration = max_iteration

        self._dag = dag
        return dag

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

    def _run_iteration(
        self, state: GEPAState,
    ) -> CandidateProposal | None:
        """Execute the full iteration with 3 sessions.

        Flow:
        1. Run deferred Session 2 from previous iteration (if pending)
        2. Sample mini-batch
        3. Fork worktree + create DAG node
        4. Determine phase + build CLAUDE.md + install skills
        5. [Session 0] Candidate selection (optional, iter > 1)
        6. Initial evaluation on mini-batch (evaluates worktree after Session 0)
        7. [Session 1] Diagnose + Patch
        8. Re-evaluation on mini-batch
        9. Initial/re-evaluation comparison + edge completion
        10. DAG update (verdict without reflections)
        11. Return proposal
        → Engine: acceptance gate → dev-set eval → state update
        → Next iteration step 1: val scores sync → deferred Session 2
        """
        dag = self._ensure_dag(state)
        session_root = str(self._adapter._session_root)

        # Sync val scores from engine state into DAG nodes
        self._sync_val_scores_from_state(dag, state)

        # ── Step 1: Deferred Session 2 from previous iteration ────────

        self._run_deferred_session2(dag, state)

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

        # ── Step 2: Mini-batch sampling ───────────────────────────────

        mini_batch_ids, mini_batch = self._sample_mini_batch(state)
        self._logger.log(f"Mini-batch: {len(mini_batch_ids)} scenarios")

        # ── Step 3: Fork worktree + create DAG node ───────────────────

        new_worktree = self._fork_worktree(base_parent_worktree, self._meta_iteration)
        self._logger.log(f"Forked worktree: {new_worktree}")

        node = dag.add_node(
            iteration=self._meta_iteration,
            worktree_path=str(new_worktree),
            base_parent_idx=base_parent_idx,
            mini_batch_ids=mini_batch_ids,
        )
        current_idx = node.idx
        self._worktree_map[current_idx] = new_worktree

        dag.add_base_edge(base_parent_idx, current_idx)

        # ── Step 4: Phase determination + CLAUDE.md + install ─────────

        phase = self._get_phase_for_iteration(self._meta_iteration)
        self._logger.log(
            f"Phase: {phase} "
            f"(iteration {self._meta_iteration})"
        )

        claude_md_content = build_claude_md()

        install_prompts_and_skills(str(new_worktree), claude_md_content, phase=phase)
        cli_env = install_evo_dag_cli(session_root, str(new_worktree))
        self._logger.log("CLAUDE.md, skills, and CLI installed")

        dag.save()

        # ── Step 5: Session 0 — Candidate Selection (optional) ───────

        session0_result = None
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
            )
            session0_result = self._run_sdk_session(
                worktree_path=new_worktree,
                prompt=build_skill_prefix(session=0, phase=phase) + session0_prompt,
                model=self._config.active_model,
                timeout=self._config.candidate_selection_timeout,
                extra_env=cli_env,
            )
            if session0_result:
                self._logger.log("Session 0 completed")
                # Session 0 JSON will be saved after cycle_dir is created (Step 6)
            else:
                self._logger.log("Session 0 failed — proceeding with default base")

        # Verify worktree after Session 0 (may have modified code via rsync)
        if not self._verify_worktree(new_worktree):
            self._logger.log(
                "Post-Session-0 verification FAILED — resetting to parent"
            )
            subprocess.run(
                ["git", "checkout", "."],
                cwd=str(new_worktree),
                capture_output=True,
                timeout=30,
            )

        # ── Step 6: Initial evaluation on mini-batch ──────────────────

        self._logger.log("Initial evaluation on mini-batch...")
        phase_before = f"iter{self._meta_iteration:02d}_train_before"
        self._adapter.set_eval_phase(phase_before, iteration=self._meta_iteration)
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

        # Save Session 0 JSON now that cycle_dir exists
        if session0_result:
            session0_info = self._extract_session_info(
                session0_result, self._config.active_model,
                self._config.candidate_selection_timeout, initial_eval["cycle_dir"], "selection",
                self._meta_iteration, current_idx,
            )
            if session0_info:
                dag.load()
                dag.set_sdk_session_info(current_idx, selection=session0_info)
                dag.save()

        if initial_pass_rate >= 1.0:
            self._logger.log("All scenarios already passing — skipping iteration")
            dag.load()
            dag.nodes[current_idx].abandoned = True
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
        )
        session1_result = self._run_sdk_session(
            worktree_path=new_worktree,
            prompt=build_skill_prefix(session=1, phase=phase) + session1_prompt,
            model=self._config.active_model,
            timeout=self._config.diagnosis_patch_timeout,
            extra_env=cli_env,
        )

        if session1_result is None:
            self._logger.log("Session 1 failed — marking node as abandoned")
            dag.load()
            dag.nodes[current_idx].abandoned = True
            dag.save()
            return None

        # Verify + commit
        if not self._verify_worktree(new_worktree):
            self._logger.log("Verification FAILED — marking node as abandoned")
            dag.load()
            dag.nodes[current_idx].abandoned = True
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
        self, state: GEPAState,
    ) -> tuple[list[str], list]:
        """Sample a mini-batch from the training set."""
        if self._batch_sampler is not None:
            from autosaddler.v1.core.data_loader import ListDataLoader

            loader = ListDataLoader(self.trainset)
            ids = self._batch_sampler.next_minibatch_ids(loader, state)
            batch = loader.fetch(ids)
            scenario_ids = [str(sid) for sid in ids]
        else:
            batch = self.trainset
            scenario_ids = [str(i) for i in range(len(batch))]

        return scenario_ids, batch

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

    def _run_sdk_session(
        self,
        worktree_path: Path,
        prompt: str,
        model: str,
        timeout: float,
        extra_env: dict[str, str] | None = None,
        max_retries: int = 5,
        initial_backoff: float = 60.0,
    ) -> dict[str, Any] | None:
        """Run an SDK session with the evo-dag CLI on PATH.

        Retries with exponential backoff on rate-limit (429) errors.
        """
        import time

        from autosaddler.v1.sdk_session import RateLimitError

        backoff = initial_backoff
        for attempt in range(1, max_retries + 1):
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
                return session_result
            except RateLimitError:
                if attempt < max_retries:
                    logger.warning(
                        "Rate-limited (attempt %d/%d). "
                        "Retrying in %.0fs...",
                        attempt, max_retries, backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 600.0)  # cap at 10 minutes
                else:
                    logger.error(
                        "Rate-limited after %d retries — giving up",
                        max_retries,
                    )
                    return None
            except Exception:
                logger.exception("SDK session failed")
                return None
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
            for u in usage:
                if isinstance(u, dict):
                    input_tokens += u.get("input_tokens", 0) or u.get("promptTokens", 0) or 0
                    output_tokens += u.get("output_tokens", 0) or u.get("completionTokens", 0) or 0
                    cache_read += (
                        u.get("cache_read_input_tokens", 0)  # Claude SDK
                        or u.get("cache_read_tokens", 0)     # Copilot SDK
                        or 0
                    )

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
                "tool_calls": tool_calls,
                "usage": usage,
                "raw_response": session_result.get("raw_response", ""),
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            return SDKSessionInfo(
                model=model,
                timeout=timeout,
                tool_call_count=len(tool_calls),
                turns=turns,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read,
                session_json_path=str(json_path),
            )
        except Exception:
            logger.exception("Failed to extract session info")
            return None
