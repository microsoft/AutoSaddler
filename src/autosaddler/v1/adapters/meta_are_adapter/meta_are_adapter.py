"""AutoSaddler adapter for Meta-ARE default agent on GAIA2 benchmark.

Executes Meta-ARE's default agent on GAIA2 scenarios in isolated git
worktrees, evaluates via the autosaddler pipeline, and returns
evaluation scores and traces in AutoSaddler's standard format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import git

from autosaddler.v1.core.adapter import EvaluationBatch, GEPAAdapter
from autosaddler.v1.sdk_session import SdkConfig

logger = logging.getLogger(__name__)


class SeedEvalReuseError(RuntimeError):
    """Raised when reusing a prior seed evaluation fails in a way that must
    abort the run (e.g. missing source, or the source does not cover the
    current validation set).

    This is intentionally NOT swallowed by ``evaluate``'s generic error
    handling so a misconfigured reuse cannot silently corrupt the baseline
    with 0.0 scores.
    """


class HookConfigEvaluationError(RuntimeError):
    """Raised when a candidate hook configuration cannot be evaluated."""


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

@dataclass
class MetaAREDataInst:
    """A single GAIA2 scenario identifier."""
    scenario_id: str


@dataclass
class MetaARETrajectory:
    """Captured execution trace for a single scenario."""
    scenario_id: str
    lite_trace: dict[str, Any] = field(default_factory=dict)
    judge_result: dict[str, Any] = field(default_factory=dict)
    output_entry: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaAREOutput:
    """Raw output for a single scenario."""
    scenario_id: str
    score: float
    passed: bool
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetaAREAdapterConfig:
    """Configuration for the Meta-ARE AutoSaddler adapter."""

    # Paths (must be set via config YAML)
    meta_are_repo: str = ""
    session_root_base: str = "outputs"
    dataset_path: str = ""
    activate_command: str = ""

    # Git
    base_branch: str = "main"

    # Benchmark execution
    agent: str = "default"
    model: str = ""
    model_provider: str = "openai"
    model_endpoint: str | None = None
    reasoning_effort: str | None = None
    model_azure_config_dir: str | None = None
    judge_model: str = "gpt-4.1-mini"
    judge_provider: str = "openai"
    judge_endpoint: str | None = None
    judge_azure_config_dir: str | None = None
    scenario_timeout: int = 3600
    num_runs: int = 1
    max_concurrent: int = 4
    # Concurrency for dev-set (valset) evaluations. When None, dev-set
    # evaluations fall back to ``max_concurrent`` (mini-batch concurrency).
    max_concurrent_dev: int | None = None
    benchmark_config: str = "search"
    split: str = "validation"

    # SDK backend configuration
    sdk_config: SdkConfig = field(default_factory=SdkConfig)

    # Worktree verification (used by proposer for import checks)
    import_check_statement: str = ""

    # Seed-eval reuse (unified evaluation starting point for fair comparison
    # across methods). When set, the seed evaluation reuses a prior
    # initial-harness seed_val result instead of re-running the benchmark.
    # Accepts a ``seed_val_XXXX_hhhhhh`` cycle dir, a run timestamp dir (with
    # ``cycles/seed_val_*``), or a ``run/`` dir directly. Empty = run from scratch.
    seed_eval_source: str = ""
    # "copy" (default) duplicates outputs into the new run; "symlink" saves disk.
    seed_eval_reuse_mode: str = "copy"


def _load_adapter_config(cfg: dict[str, Any]) -> MetaAREAdapterConfig:
    """Build adapter config from a raw dict (e.g. from YAML)."""
    ac = MetaAREAdapterConfig()
    for key in (
        "meta_are_repo", "session_root_base", "dataset_path", "activate_command",
        "base_branch", "agent", "model", "model_provider", "model_endpoint",
        "reasoning_effort", "model_azure_config_dir",
        "judge_model", "judge_provider", "judge_endpoint",
        "judge_azure_config_dir", "scenario_timeout", "num_runs", "max_concurrent",
        "max_concurrent_dev", "benchmark_config", "split", "import_check_statement",
        "seed_eval_source", "seed_eval_reuse_mode",
    ):
        if key in cfg:
            setattr(ac, key, cfg[key])
    if "sdk_config" in cfg:
        sc_cfg = cfg["sdk_config"]
        if isinstance(sc_cfg, dict):
            ac.sdk_config = SdkConfig(**sc_cfg)
        elif isinstance(sc_cfg, SdkConfig):
            ac.sdk_config = sc_cfg
    return ac


# ---------------------------------------------------------------------------
# Worktree pool
# ---------------------------------------------------------------------------


class WorktreePool:
    """Persistent patched worktrees keyed by candidate hash."""

    def __init__(
        self,
        repo_path: Path,
        worktree_dir: Path,
        base_branch: str,
        session_id: str,
    ) -> None:
        self._repo_path = repo_path
        self._worktree_dir = worktree_dir
        self._base_branch = base_branch
        self._session_id = session_id
        self._pool: dict[str, Path] = {}  # hash → worktree_path
        self._lock = threading.Lock()

        # Prune stale worktree refs from previous crashed sessions
        try:
            repo = git.Repo(self._repo_path)
            repo.git.worktree("prune")
        except Exception:
            pass

    def get_or_create(
        self,
        candidate: dict[str, str],
        patch_fn: Any,
        *,
        parent_worktree: Path | None = None,
    ) -> tuple[Path, bool]:
        """Return *(worktree_path, cache_hit)*.

        *cache_hit=True* means the worktree was reused (no patching).
        *cache_hit=False* means a new worktree was created and *patch_fn*
        was called to patch it.

        Parameters
        ----------
        parent_worktree:
            If provided, the new worktree is forked from this worktree's
            HEAD commit instead of ``base_branch``.  This allows patches
            to accumulate across iterations.
        """
        key = self._hash(candidate, self._session_id)
        with self._lock:
            if key in self._pool:
                return self._pool[key], True

            existing_path = self._worktree_dir / f"seed_{key}"
            if existing_path.is_dir():
                try:
                    existing_repo = git.Repo(existing_path)
                    existing_repo.head.commit.hexsha
                    expected_branch = f"autosaddler/seed_{key}"
                    if (
                        existing_repo.active_branch.name != expected_branch
                        or existing_repo.is_dirty(untracked_files=True)
                    ):
                        raise ValueError("branch mismatch or uncommitted changes")
                except Exception:
                    logger.warning(
                        "Existing pooled path is not safely reusable: %s",
                        existing_path,
                    )
                else:
                    self._pool[key] = existing_path
                    logger.info(
                        "Reusing existing pooled worktree %s",
                        existing_path,
                    )
                    return existing_path, True

        # Determine the git ref to fork from
        base_ref: str | None = None
        if parent_worktree is not None:
            try:
                parent_repo = git.Repo(parent_worktree)
                base_ref = parent_repo.head.commit.hexsha
                logger.info(
                    "Forking worktree from parent %s (commit %s)",
                    parent_worktree.name, base_ref[:8],
                )
            except Exception:
                logger.warning(
                    "Could not read parent worktree %s — "
                    "falling back to base_branch",
                    parent_worktree,
                    exc_info=True,
                )

        wt = self._create_worktree(key, base_ref=base_ref)
        patch_fn(wt, candidate)
        # Commit the fully-patched state as a clean restore point.
        # Future child worktrees will fork from this commit.
        self._commit_baseline(wt)
        with self._lock:
            self._pool[key] = wt
        return wt, False

    def cleanup_all(self) -> None:
        """Remove all pooled worktrees."""
        with self._lock:
            paths = list(self._pool.values())
            self._pool.clear()
        for p in paths:
            self._remove_worktree(p)

    # -- internal helpers --------------------------------------------------

    def _create_worktree(
        self, key: str, *, base_ref: str | None = None,
    ) -> Path:
        """Create a git worktree for the given cache key.

        Parameters
        ----------
        base_ref:
            Git ref (commit SHA, branch) to use as the starting point.
            If ``None``, falls back to ``self._base_branch``.
        """
        repo = git.Repo(self._repo_path)
        wt_path = self._worktree_dir / f"seed_{key}"
        branch_name = f"autosaddler/seed_{key}"
        start_point = base_ref or self._base_branch
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Creating pooled worktree %s  branch=%s  base=%s",
            wt_path, branch_name, start_point[:12],
        )
        try:
            repo.git.worktree(
                "add", str(wt_path), "-b", branch_name, start_point,
            )
        except git.GitCommandError:
            # Branch or worktree may be stale from a previous crashed session.
            logger.warning(
                "Stale worktree/branch detected for %s — cleaning up", key,
            )
            try:
                repo.git.worktree("remove", str(wt_path), "--force")
            except Exception:
                shutil.rmtree(wt_path, ignore_errors=True)
            try:
                repo.git.worktree("prune")
            except Exception:
                pass
            try:
                repo.git.branch("-D", branch_name)
            except Exception:
                pass
            # Retry after cleanup
            repo.git.worktree(
                "add", str(wt_path), "-b", branch_name, start_point,
            )
        return wt_path

    def _remove_worktree(self, wt_path: Path) -> None:
        try:
            repo = git.Repo(self._repo_path)
            repo.git.worktree("remove", str(wt_path), "--force")
        except Exception:
            logger.warning(
                "git worktree remove failed; removing directory manually",
            )
            shutil.rmtree(wt_path, ignore_errors=True)
        try:
            repo = git.Repo(self._repo_path)
            repo.git.worktree("prune")
        except Exception:
            pass

    @staticmethod
    def _commit_baseline(wt_path: Path) -> None:
        """Commit the fully-patched working tree as a clean restore point.

        Child worktrees created via ``parent_worktree`` will fork from
        this commit, enabling patch accumulation across iterations.
        """
        repo = git.Repo(wt_path)
        repo.git.add("-A")
        # --allow-empty handles the (rare) case where patching made no
        # changes — we still want a consistent restore point.
        repo.git.commit(
            "--allow-empty", "-m", "autosaddler: baseline",
        )

    @staticmethod
    def _hash(candidate: dict[str, str], session_id: str) -> str:
        blob = (session_id + json.dumps(candidate, sort_keys=True)).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MetaAREAdapter(
    GEPAAdapter[MetaAREDataInst, MetaARETrajectory, MetaAREOutput]
):
    """AutoSaddler adapter that evaluates Meta-ARE agents on GAIA2.

    The adapter receives a pre-patched worktree path via the
    ``__autosaddler_worktree__`` candidate key and runs the
    ``are-benchmark`` harness on the requested scenario batch.

    For each ``evaluate`` call the adapter:
    1. Writes ``candidate.json`` for provenance.
    2. Uses the pre-patched worktree provided by the proposer.
    3. Runs ``are-benchmark run`` on the requested scenario batch.
    4. Parses ``output.jsonl`` + lite traces and returns scores.
    """

    def __init__(self, config: dict[str, Any] | MetaAREAdapterConfig) -> None:
        if isinstance(config, dict):
            self.cfg = _load_adapter_config(config)
        else:
            self.cfg = config

        self._repo_path = Path(self.cfg.meta_are_repo).resolve()
        # session_root is set by set_session_root(); until then use base as fallback
        self._session_root: Path | None = None
        self._eval_counter = 0
        self._last_cycle_dir: Path | None = None
        self._last_worktree_path: Path | None = None

        # Phase tracking for structured directory naming
        self._iteration = 0
        self._phase = "init"
        self._forced_phase: str | None = None
        self._reserved_eval: tuple[str, bool] | None = None

        # Worktree pool (created in set_session_root)
        self._worktree_pool: WorktreePool | None = None

    # ------------------------------------------------------------------
    # Session root management
    # ------------------------------------------------------------------

    def _reconstruct_eval_state_from_cycles(self) -> None:
        """Reconstruct ``_eval_counter``/``_iteration``/``_phase`` from existing
        cycle directories so a resumed run continues the same eval numbering as
        an uninterrupted run. No-op on a fresh run (empty cycles/).
        """
        import re as _re

        cycles_dir = self._cycles_dir
        if not cycles_dir.exists():
            return
        # Matches <optional iter{N}_><phase>_<4-digit counter>_<hex uid>.
        # arm_scoring dirs (no counter suffix) are ignored, which is correct:
        # they are not evaluations and do not advance _eval_counter.
        pattern = _re.compile(r"^(?:iter(\d+)_)?(.+?)_(\d{4})_[a-f0-9]+$")
        max_counter = 0
        max_iteration = 0
        for entry in cycles_dir.iterdir():
            if not entry.is_dir():
                continue
            m = pattern.match(entry.name)
            if not m:
                continue
            iter_n = int(m.group(1)) if m.group(1) else 0
            counter = int(m.group(3))
            if counter > max_counter:
                max_counter = counter
            if iter_n > max_iteration:
                max_iteration = iter_n
        if max_counter > 0:
            self._eval_counter = max_counter
            self._iteration = max_iteration
            self._phase = "seed_val"
            logger.info(
                "Reconstructed eval state from cycles: _eval_counter=%d, "
                "_iteration=%d", self._eval_counter, self._iteration,
            )

    def set_session_root(self, session_root: Path | str) -> None:
        """Set the session root and create the standard directory layout.

        Creates::

            <session_root>/
            ├── worktrees/   ← persistent pooled git worktrees
            └── cycles/      ← per-evaluation benchmark outputs

        On resume (when cycles/ already contains directories from a prior
        run), the adapter reconstructs ``_eval_counter``/``_iteration``/
        ``_phase`` so subsequent evaluations continue with the correct
        numbering — identical to an uninterrupted run.
        """
        self._session_root = Path(session_root).resolve()
        self._worktree_dir.mkdir(parents=True, exist_ok=True)
        self._cycles_dir.mkdir(parents=True, exist_ok=True)

        # On resume (cycles/ already populated from a prior run), continue the
        # eval-directory numbering instead of restarting the counter at 0.
        self._reconstruct_eval_state_from_cycles()

        # Initialize worktree pool for this session
        session_id = self._session_root.name
        self._worktree_pool = WorktreePool(
            repo_path=self._repo_path,
            worktree_dir=self._worktree_dir,
            base_branch=self.cfg.base_branch,
            session_id=session_id,
        )
        logger.info("Session root: %s", self._session_root)

    @property
    def _worktree_dir(self) -> Path:
        if self._session_root:
            return self._session_root / "worktrees"
        return Path(self.cfg.session_root_base).resolve() / "worktrees"

    @property
    def _cycles_dir(self) -> Path:
        if self._session_root:
            return self._session_root / "cycles"
        return Path(self.cfg.session_root_base).resolve() / "cycles"

    @property
    def last_cycle_dir(self) -> Path | None:
        """Path to the most recently created cycle directory."""
        return self._last_cycle_dir

    # ------------------------------------------------------------------
    # Evaluation phase tracking
    # ------------------------------------------------------------------

    def set_eval_phase(self, phase: str, iteration: int | None = None) -> None:
        """Override the next evaluation's phase label.

        Useful for integration with engine callbacks or custom loops.
        The override is consumed by the next ``_resolve_eval_id`` call
        and then cleared.
        """
        self._forced_phase = phase
        if iteration is not None:
            self._iteration = iteration

    def reserve_eval_cycle(
        self,
        phase: str,
        *,
        iteration: int,
        capture_traces: bool,
    ) -> Path:
        """Reserve the directory consumed by the next evaluation."""
        if self._reserved_eval is not None:
            raise RuntimeError("An evaluation cycle is already reserved")
        self.set_eval_phase(phase, iteration=iteration)
        eval_id = self._resolve_eval_id(capture_traces)
        self._reserved_eval = (eval_id, capture_traces)
        cycle_dir = self._cycles_dir / eval_id
        cycle_dir.mkdir(parents=True, exist_ok=True)
        return cycle_dir

    def cancel_reserved_eval_cycle(self) -> Path | None:
        """Release an unconsumed reservation while preserving its artifacts."""
        if self._reserved_eval is None:
            return None
        eval_id, _capture_traces = self._reserved_eval
        self._reserved_eval = None
        return self._cycles_dir / eval_id

    def _consume_eval_id(self, capture_traces: bool) -> str:
        if self._reserved_eval is None:
            return self._resolve_eval_id(capture_traces)
        eval_id, reserved_capture_traces = self._reserved_eval
        if capture_traces != reserved_capture_traces:
            raise RuntimeError(
                "Reserved evaluation capture_traces mismatch: "
                f"expected {reserved_capture_traces}, got {capture_traces}"
            )
        self._reserved_eval = None
        return eval_id

    def _resolve_eval_id(self, capture_traces: bool) -> str:
        """Generate a structured eval directory name from phase state.

        Uses a state machine that infers the evaluation phase from the
        ``capture_traces`` flag and the sequence of prior calls:

        =============  ===================  ==========================
        Prior phase    capture_traces       Resulting phase
        =============  ===================  ==========================
        init           any                  ``seed_val``
        seed_val       True                 ``iter{N}_train_before``
        train_before   False                ``iter{N}_train_after``
        train_after    False                ``iter{N}_val``
        train_after    True                 ``iter{N+1}_train_before``
        val            True                 ``iter{N+1}_train_before``
        =============  ===================  ==========================
        """
        self._eval_counter += 1
        uid = uuid4().hex[:6]
        counter = f"{self._eval_counter:04d}"

        # Explicit override takes priority
        if self._forced_phase is not None:
            forced_phase = self._forced_phase
            label = f"{forced_phase}_{counter}_{uid}"
            self._forced_phase = None
            if forced_phase == "seed_val":
                self._phase = "seed_val"
            else:
                for logical_phase in ("train_before", "train_after", "val"):
                    if (
                        forced_phase == logical_phase
                        or forced_phase.endswith(f"_{logical_phase}")
                    ):
                        self._phase = logical_phase
                        break
            return label

        # State-machine inference
        if self._phase == "init":
            self._phase = "seed_val"
            return f"seed_val_{counter}_{uid}"

        if capture_traces:
            self._iteration += 1
            self._phase = "train_before"
            return f"iter{self._iteration:02d}_train_before_{counter}_{uid}"

        # capture_traces=False
        if self._phase == "train_before":
            self._phase = "train_after"
            return f"iter{self._iteration:02d}_train_after_{counter}_{uid}"

        if self._phase == "train_after":
            self._phase = "val"
            return f"iter{self._iteration:02d}_val_{counter}_{uid}"

        # Fallback for unexpected transitions
        self._phase = "val"
        prefix = f"iter{self._iteration:02d}_" if self._iteration > 0 else ""
        return f"{prefix}val_{counter}_{uid}"

    # ------------------------------------------------------------------
    # Candidate format detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_candidate_format(candidate: dict[str, str]) -> str:
        """Detect the candidate format for routing in evaluate().

        Returns:
            ``"autosaddler"``  — pre-patched worktree (``__autosaddler_worktree__``)
            ``"unknown"``       — unrecognised format (fallback error)
        """
        if "__autosaddler_worktree__" in candidate:
            return "autosaddler"
        return "unknown"

    # ------------------------------------------------------------------
    # Adapter.evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[MetaAREDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[MetaARETrajectory, MetaAREOutput]:
        eval_id = self._consume_eval_id(capture_traces)

        # Always write candidate.json for provenance
        cycle_dir = self._cycles_dir / eval_id
        cycle_dir.mkdir(parents=True, exist_ok=True)
        self._last_cycle_dir = cycle_dir
        candidate_json_path = cycle_dir / "candidate.json"
        self._write_candidate_json(candidate, candidate_json_path)

        try:
            assert self._worktree_pool is not None, (
                "WorktreePool not initialised — call set_session_root() first"
            )

            candidate_format = self._detect_candidate_format(candidate)

            if candidate_format == "autosaddler":
                # Worktree already created and patched by the proposer.
                worktree_path = Path(candidate["__autosaddler_worktree__"])
                if not worktree_path.exists():
                    raise FileNotFoundError(
                        f"AutoSaddler worktree not found: {worktree_path}"
                    )
                logger.info(
                    "Worktree for %s [autosaddler]: %s",
                    eval_id, worktree_path,
                )
            else:
                raise ValueError(
                    f"Unsupported candidate format: {candidate_format}. "
                    f"Only 'autosaddler' is supported."
                )

            # Track the worktree used for this evaluation so the
            # proposer can set __parent_worktree__ on child candidates.
            self._last_worktree_path = worktree_path

            # Run benchmark
            scenario_ids = [inst.scenario_id for inst in batch]
            output_dir = cycle_dir / "run"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Unified evaluation starting point: for the seed evaluation, reuse
            # a prior initial-harness seed_val result (if configured) instead of
            # re-running the benchmark. Everything downstream (GEPAState, DAG
            # seed node, RNG-driven sampling) reproduces identically because it
            # derives solely from these per-scenario scores.
            seed_eval_source = (
                self._resolve_seed_eval_source()
                if eval_id.startswith("seed_val_")
                else None
            )
            if seed_eval_source is not None:
                self._reuse_seed_eval(
                    seed_eval_source, output_dir, scenario_ids, cycle_dir,
                )
            else:
                hook_config_path = self._find_hook_config(worktree_path)
                # Dev-set (valset) evaluations run with capture_traces=False and
                # may use a higher concurrency than mini-batch evaluations
                # (capture_traces=True), which are bounded to keep resource usage
                # in check during the reflective loop.
                max_concurrent = self.cfg.max_concurrent
                if not capture_traces and self.cfg.max_concurrent_dev is not None:
                    max_concurrent = self.cfg.max_concurrent_dev
                self._run_benchmark(
                    worktree_path, scenario_ids, output_dir,
                    hook_config_path=hook_config_path,
                    max_concurrent=max_concurrent,
                )

            # Parse results
            outputs, scores, trajectories = self._parse_results(
                output_dir, scenario_ids, capture_traces,
            )
        except (SeedEvalReuseError, HookConfigEvaluationError):
            # Harness-integrity failures must abort rather than silently
            # producing 0.0 scores for a different or incomplete candidate.
            logger.error("Evaluation integrity failure for %s - aborting", eval_id)
            raise
        except Exception:
            logger.exception("evaluate failed for %s", eval_id)
            outputs = [
                MetaAREOutput(
                    scenario_id=inst.scenario_id,
                    score=0.0,
                    passed=False,
                    status="error",
                )
                for inst in batch
            ]
            scores = [0.0] * len(batch)
            trajectories = (
                [
                    MetaARETrajectory(scenario_id=inst.scenario_id)
                    for inst in batch
                ]
                if capture_traces
                else None
            )
        # No finally cleanup — worktree stays in pool for reuse

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    # ------------------------------------------------------------------
    # Candidate JSON export
    # ------------------------------------------------------------------

    def _write_candidate_json(
        self, candidate: dict[str, str], path: Path,
    ) -> None:
        """Write candidate.json for provenance."""
        candidate_format = self._detect_candidate_format(candidate)

        if candidate_format == "autosaddler":
            data: dict[str, Any] = {
                "format": "autosaddler",
                "worktree": candidate.get("__autosaddler_worktree__"),
            }
        else:
            # Fallback: store the raw candidate (without meta keys)
            data = {
                "format": candidate_format,
                "candidate": {
                    k: v for k, v in candidate.items()
                    if not k.startswith("__")
                },
            }

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up adapter resources (no-op; worktrees are preserved)."""
        pass

    # ------------------------------------------------------------------
    # Benchmark execution
    # ------------------------------------------------------------------

    def _find_hook_config(self, worktree_path: Path) -> Path | None:
        """Find hook.json generated by hook patch plan execution."""
        hook_path = worktree_path / "hook.json"
        if hook_path.exists():
            return hook_path
        return None

    # ------------------------------------------------------------------
    # Seed-eval reuse (unified evaluation starting point)
    # ------------------------------------------------------------------

    def _resolve_seed_eval_source(self) -> Path | None:
        """Resolve the configured seed-eval reuse source to a ``run/`` dir.

        Returns the source ``run/`` directory whose benchmark outputs should
        be reused for the seed evaluation, or ``None`` when reuse is disabled.

        Accepts (via ``cfg.seed_eval_source``):
          - a ``seed_val_XXXX_hhhhhh`` cycle dir (contains ``run/``)
          - a run timestamp dir (contains ``cycles/seed_val_*``)
          - a ``run/`` dir directly

        Raises ``SeedEvalReuseError`` if a source is configured but cannot be
        resolved, so a misconfiguration aborts the run instead of silently
        running the benchmark from scratch.
        """
        raw = (self.cfg.seed_eval_source or "").strip()
        if not raw:
            return None

        src = Path(raw).expanduser()
        if not src.exists():
            raise SeedEvalReuseError(f"seed_eval_source does not exist: {src}")
        src = src.resolve()

        # Case 1: already a run/ dir (has output.jsonl or lite/ inside)
        if src.name == "run" or (src / "output.jsonl").exists() or (src / "lite").is_dir():
            return src

        # Case 2: a cycle dir containing run/
        if (src / "run").is_dir():
            return src / "run"

        # Case 3: a run timestamp dir containing cycles/seed_val_*
        cycles_dir = src / "cycles"
        if cycles_dir.is_dir():
            seed_dirs = sorted(p for p in cycles_dir.glob("seed_val_*") if p.is_dir())
            if not seed_dirs:
                raise SeedEvalReuseError(
                    f"No seed_val_* cycle found under {cycles_dir}"
                )
            chosen = seed_dirs[0]
            if len(seed_dirs) > 1:
                logger.warning(
                    "Multiple seed_val_* dirs under %s; using earliest: %s",
                    cycles_dir, chosen.name,
                )
            run_dir = chosen / "run"
            if not run_dir.is_dir():
                raise SeedEvalReuseError(
                    f"Seed cycle {chosen} has no run/ directory"
                )
            return run_dir

        raise SeedEvalReuseError(
            f"Could not locate seed-eval run/ outputs from source: {src}"
        )

    def _reuse_seed_eval(
        self,
        source_run: Path,
        output_dir: Path,
        scenario_ids: list[str],
        cycle_dir: Path,
    ) -> None:
        """Reuse a prior seed evaluation's benchmark outputs.

        Copies (or symlinks) the source ``run/`` subtree into ``output_dir``
        so that ``_parse_results`` yields scores identical to a from-scratch
        seed evaluation, validates that every requested scenario is covered,
        and records a provenance manifest.

        Raises ``SeedEvalReuseError`` on any coverage/copy failure so the run
        aborts rather than silently scoring missing scenarios as 0.0.
        """
        mode = (self.cfg.seed_eval_reuse_mode or "copy").strip().lower()
        logger.info(
            "Reusing seed evaluation from %s (mode=%s) - skipping benchmark run",
            source_run, mode,
        )

        try:
            for item in sorted(source_run.iterdir()):
                dest = output_dir / item.name
                if dest.is_symlink() or dest.is_file():
                    dest.unlink()
                elif dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                if mode == "symlink":
                    dest.symlink_to(item.resolve())
                elif item.is_dir():
                    shutil.copytree(item, dest, symlinks=True)
                else:
                    shutil.copy2(item, dest)
        except Exception as e:
            raise SeedEvalReuseError(
                f"Failed to {mode} seed-eval outputs from {source_run}: {e}"
            ) from e

        # Validate coverage against output.jsonl (the score source used by
        # _parse_results). lite/ may legitimately have fewer files, so it is
        # NOT authoritative for coverage.
        covered = self._collect_scored_scenario_ids(output_dir)
        missing = [sid for sid in scenario_ids if sid not in covered]
        if missing:
            raise SeedEvalReuseError(
                f"Reused seed evaluation covers {len(covered)} scenarios but is "
                f"missing {len(missing)}/{len(scenario_ids)} requested val "
                f"scenarios (source={source_run}). The seed_eval_source val set "
                f"must match this run's val set. First missing: {missing[:5]}"
            )
        logger.info(
            "Seed-eval reuse coverage OK: %d/%d scenarios present",
            len(scenario_ids), len(scenario_ids),
        )

        self._write_reuse_manifest(source_run, cycle_dir, scenario_ids, covered)

    @staticmethod
    def _collect_scored_scenario_ids(output_dir: Path) -> set[str]:
        """Scenario ids that ``_parse_results`` can score from output.jsonl.

        ``_parse_results`` derives each score from output.jsonl entries
        (falling back to 0.0 when absent), so coverage must be measured
    against output.jsonl, not lite traces.
        """
        covered: set[str] = set()
        for jsonl_file in output_dir.rglob("output.jsonl"):
            try:
                for entry in MetaAREAdapter._read_jsonl(jsonl_file):
                    meta = entry.get("metadata", {})
                    sid = entry.get("task_id") or meta.get("scenario_id", "")
                    if sid:
                        covered.add(sid)
            except Exception as e:
                logger.warning("Failed reading %s for coverage: %s", jsonl_file, e)
        return covered

    def _write_reuse_manifest(
        self,
        source_run: Path,
        cycle_dir: Path,
        scenario_ids: list[str],
        covered: set[str],
    ) -> None:
        """Write a provenance manifest recording the seed-eval reuse."""
        cfg = self.cfg
        source_run_id = ""
        source_cycle = ""
        try:
            source_cycle = source_run.parent.name
            source_run_id = source_run.parent.parent.parent.name
        except Exception:
            pass

        manifest = {
            "reused": True,
            "source_run": str(source_run),
            "source_cycle": source_cycle,
            "source_run_id": source_run_id,
            "reuse_mode": (cfg.seed_eval_reuse_mode or "copy"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requested_scenarios": len(scenario_ids),
            "covered_scenarios": len(covered),
            "harness_fingerprint": {
                "base_branch": cfg.base_branch,
                "agent": cfg.agent,
                "model": cfg.model,
                "model_provider": cfg.model_provider,
                "reasoning_effort": cfg.reasoning_effort,
                "judge_model": cfg.judge_model,
                "judge_provider": cfg.judge_provider,
                "dataset_path": cfg.dataset_path,
                "split": cfg.split,
            },
        }
        payload = json.dumps(manifest, indent=2, ensure_ascii=False)
        try:
            (cycle_dir / "seed_eval_reuse.json").write_text(payload)
            if self._session_root is not None:
                (self._session_root / "seed_eval_reuse.json").write_text(payload)
        except Exception as e:  # best-effort provenance; never abort the run
            logger.warning("Failed to write seed-eval reuse manifest: %s", e)

        self._maybe_warn_harness_mismatch(source_run)

    def _maybe_warn_harness_mismatch(self, source_run: Path) -> None:
        """Warn (never abort) if a source reuse manifest's harness differs.

        Only effective when the source itself was produced via reuse (i.e. it
        carries a ``seed_eval_reuse.json``). From-scratch sources carry no
        fingerprint, so the check is silently skipped.
        """
        candidates = [
            source_run.parent / "seed_eval_reuse.json",
            source_run.parent.parent.parent / "seed_eval_reuse.json",
        ]
        for man_path in candidates:
            try:
                if not man_path.exists():
                    continue
                fp = json.loads(man_path.read_text()).get("harness_fingerprint", {})
            except Exception:
                continue
            current = {
                "base_branch": self.cfg.base_branch,
                "model": self.cfg.model,
                "model_provider": self.cfg.model_provider,
                "judge_model": self.cfg.judge_model,
                "dataset_path": self.cfg.dataset_path,
                "split": self.cfg.split,
            }
            for key, cur_val in current.items():
                prior_val = fp.get(key)
                if prior_val is not None and prior_val != cur_val:
                    logger.warning(
                        "Seed-eval reuse harness mismatch on '%s': source=%r "
                        "current=%r. Reused baseline may not match this run's "
                        "initial harness.",
                        key, prior_val, cur_val,
                    )
            break

    def _run_benchmark(
        self,
        worktree_path: Path,
        scenario_ids: list[str],
        output_dir: Path,
        *,
        hook_config_path: Path | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        """Run are-benchmark in the worktree for the given scenarios.

        ``max_concurrent`` overrides the number of scenarios run in parallel;
        when None it falls back to ``cfg.max_concurrent``.
        """
        # Stage scenario files into a temp directory
        staging_dir = self._stage_scenarios(scenario_ids)

        cfg = self.cfg
        parts = [
            "are-benchmark", "run",
            f"--agent {cfg.agent}",
            f"--dataset {staging_dir}" if staging_dir else f"--dataset {cfg.dataset_path}",
            "--config ." if staging_dir else f"--config {cfg.benchmark_config}",
        ]
        if cfg.model:
            parts.append(f"--model {cfg.model}")
        if cfg.model_provider:
            parts.append(f"--provider {cfg.model_provider}")
        if cfg.model_endpoint:
            parts.append(f"--endpoint {cfg.model_endpoint}")
        if cfg.reasoning_effort:
            parts.append(f"--reasoning_effort {cfg.reasoning_effort}")
        if cfg.model_azure_config_dir:
            parts.append(
                f"--azure_config_dir {shlex.quote(cfg.model_azure_config_dir)}"
            )
        if cfg.judge_model:
            parts.append(f"--judge_model {cfg.judge_model}")
        if cfg.judge_provider:
            parts.append(f"--judge_provider {cfg.judge_provider}")
        if cfg.judge_endpoint:
            parts.append(f"--judge_endpoint {cfg.judge_endpoint}")
        if cfg.judge_azure_config_dir:
            parts.append(
                "--judge_azure_config_dir "
                f"{shlex.quote(cfg.judge_azure_config_dir)}"
            )

        parts.append(f"--output_dir {output_dir}")
        parts.append(f"--scenario_timeout {cfg.scenario_timeout}")
        parts.append(f"--num_runs {cfg.num_runs}")
        concurrency = max_concurrent if max_concurrent is not None else cfg.max_concurrent
        parts.append(f"--max_concurrent_scenarios {concurrency}")
        parts.append("--trace_dump_format both")

        # A hook file is part of the candidate harness and must be evaluated.
        if hook_config_path and hook_config_path.exists():
            parts.append(f"--hook-config {shlex.quote(str(hook_config_path))}")

        cmd_str = " ".join(parts)

        # Build a shell command that activates the venv and runs the benchmark
        # inside the worktree (so ARE picks up the patched code).
        shell_cmd = f"{cfg.activate_command} && {cmd_str}"

        timeout = cfg.scenario_timeout * len(scenario_ids) * cfg.num_runs + 120
        logger.info("Running benchmark: %s", cmd_str)

        # Prepend worktree to PYTHONPATH so that ``import are`` resolves from
        # the patched worktree, overriding the .pth editable-install path
        # that points at the original repo.
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(worktree_path) + (":" + existing_pp if existing_pp else "")

        result = subprocess.run(
            shell_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(worktree_path),
            env=env,
        )

        # Clean up the staging directory now that the benchmark has finished
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)

        if result.returncode != 0:
            logger.error(
                "Benchmark command failed (exit %d):\nstdout: %s\nstderr: %s",
                result.returncode,
                result.stdout[-2000:] if result.stdout else "",
                result.stderr[-2000:] if result.stderr else "",
            )
            if hook_config_path and hook_config_path.exists():
                raise HookConfigEvaluationError(
                    "Benchmark failed while evaluating hook config "
                    f"{hook_config_path} (exit {result.returncode})"
                )
            # Don't raise — we'll parse whatever partial results exist

    def _stage_scenarios(self, scenario_ids: list[str]) -> Path | None:
        """Create a temp directory with symlinks to requested scenario JSON files."""
        dataset_path = Path(self.cfg.dataset_path)
        if not dataset_path.exists():
            logger.warning("Dataset path %s does not exist", dataset_path)
            return None

        staging_dir = Path(tempfile.mkdtemp(prefix="autosaddler_scenario_staging_"))
        staged = 0

        for sid in scenario_ids:
            # Search for scenario JSON file — GAIA2 files are named
            # <index>_<scenario_id>.json (e.g. 0072_scenario_universe_30_k0yt0a.json)
            matches = list(dataset_path.rglob(f"*_{sid}.json"))
            if matches:
                (staging_dir / matches[0].name).symlink_to(matches[0])
                staged += 1
            else:
                logger.warning("Could not find scenario file for %s", sid)

        if staged == 0:
            shutil.rmtree(staging_dir, ignore_errors=True)
            return None

        logger.debug("Staged %d scenario(s) in %s", staged, staging_dir)
        return staging_dir

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_results(
        self,
        output_dir: Path,
        scenario_ids: list[str],
        capture_traces: bool,
    ) -> tuple[
        list[MetaAREOutput],
        list[float],
        list[MetaARETrajectory] | None,
    ]:
        """Parse output.jsonl and lite traces into AutoSaddler evaluation results."""
        # Parse output.jsonl entries
        results_by_id: dict[str, dict[str, Any]] = {}
        for jsonl_file in output_dir.rglob("output.jsonl"):
            for entry in self._read_jsonl(jsonl_file):
                meta = entry.get("metadata", {})
                task_id = entry.get("task_id") or meta.get("scenario_id", "")
                results_by_id[task_id] = entry

        # Parse lite traces
        traces_by_id: dict[str, dict[str, Any]] = {}
        for lite_file in output_dir.rglob("lite/*.json"):
            try:
                data = json.loads(lite_file.read_text())
                sid = data.get("scenario_id", lite_file.stem)
                traces_by_id[sid] = data
            except Exception as e:
                logger.warning("Failed to parse lite trace %s: %s", lite_file, e)

        # Also try benchmark_stats.json for supplementary info
        stats_file = output_dir / "benchmark_stats.json"
        if stats_file.exists():
            try:
                json.loads(stats_file.read_text())
            except Exception:
                pass

        # Build per-scenario outputs
        outputs: list[MetaAREOutput] = []
        scores: list[float] = []
        trajectories: list[MetaARETrajectory] = [] if capture_traces else []

        for sid in scenario_ids:
            entry = results_by_id.get(sid, {})
            meta = entry.get("metadata", {})

            raw_score = entry.get("score")
            if raw_score is not None:
                score = float(raw_score)
                passed = score > 0
            else:
                status = meta.get("status", "unknown")
                passed = status == "success"
                score = 1.0 if passed else 0.0

            output = MetaAREOutput(
                scenario_id=sid,
                score=score,
                passed=passed,
                status=meta.get("status", "unknown"),
                metadata=meta,
            )
            outputs.append(output)
            scores.append(score)

            if capture_traces:
                lite_data = traces_by_id.get(sid, {})
                judge_result = {
                    "validation_decision": lite_data.get("validation_decision", "Unknown"),
                    "validation_rationale": lite_data.get("validation_rationale", ""),
                }
                trajectories.append(MetaARETrajectory(
                    scenario_id=sid,
                    lite_trace=lite_data,
                    judge_result=judge_result,
                    output_entry=entry,
                ))

        return outputs, scores, trajectories if capture_traces else None

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file and return list of parsed JSON objects."""
        entries: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
