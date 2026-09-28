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
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import git

from autosaddler.v1.core.adapter import EvaluationBatch, GEPAAdapter
from autosaddler.v1.sdk_session import SdkConfig

logger = logging.getLogger(__name__)


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
    judge_model: str = "gpt-4.1-mini"
    judge_provider: str = "openai"
    scenario_timeout: int = 3600
    num_runs: int = 1
    max_concurrent: int = 4
    benchmark_config: str = "search"
    split: str = "validation"

    # SDK backend configuration
    sdk_config: SdkConfig = field(default_factory=SdkConfig)

    # Worktree verification (used by proposer for import checks)
    import_check_statement: str = ""


def _load_adapter_config(cfg: dict[str, Any]) -> MetaAREAdapterConfig:
    """Build adapter config from a raw dict (e.g. from YAML)."""
    ac = MetaAREAdapterConfig()
    for key in (
        "meta_are_repo", "session_root_base", "dataset_path", "activate_command",
        "base_branch", "agent", "model", "model_provider", "model_endpoint",
        "reasoning_effort", "judge_model", "judge_provider", "scenario_timeout",
        "num_runs", "max_concurrent", "benchmark_config", "split",
        "import_check_statement",
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

        # Worktree pool (created in set_session_root)
        self._worktree_pool: WorktreePool | None = None

    # ------------------------------------------------------------------
    # Session root management
    # ------------------------------------------------------------------

    def set_session_root(self, session_root: Path | str) -> None:
        """Set the session root and create the standard directory layout.

        Creates::

            <session_root>/
            ├── worktrees/   ← persistent pooled git worktrees
            └── cycles/      ← per-evaluation benchmark outputs
        """
        self._session_root = Path(session_root).resolve()
        self._worktree_dir.mkdir(parents=True, exist_ok=True)
        self._cycles_dir.mkdir(parents=True, exist_ok=True)

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
            label = f"{self._forced_phase}_{counter}_{uid}"
            self._forced_phase = None
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
        prefix = f"iter{self._iteration:02d}_" if self._iteration > 0 else ""
        return f"{prefix}eval_{counter}_{uid}"

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
        eval_id = self._resolve_eval_id(capture_traces)

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

            hook_config_path = self._find_hook_config(worktree_path)
            self._run_benchmark(
                worktree_path, scenario_ids, output_dir,
                hook_config_path=hook_config_path,
            )

            # Parse results
            outputs, scores, trajectories = self._parse_results(
                output_dir, scenario_ids, capture_traces,
            )
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

    def _run_benchmark(
        self,
        worktree_path: Path,
        scenario_ids: list[str],
        output_dir: Path,
        *,
        hook_config_path: Path | None = None,
    ) -> None:
        """Run are-benchmark in the worktree for the given scenarios."""
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
        if cfg.judge_model:
            parts.append(f"--judge_model {cfg.judge_model}")
        if cfg.judge_provider:
            parts.append(f"--judge_provider {cfg.judge_provider}")

        parts.append(f"--output_dir {output_dir}")
        parts.append(f"--scenario_timeout {cfg.scenario_timeout}")
        parts.append(f"--num_runs {cfg.num_runs}")
        parts.append(f"--max_concurrent_scenarios {cfg.max_concurrent}")
        parts.append("--trace_dump_format both")

        # Pass hook config if available (only if the installed ARE supports it)
        if hook_config_path and hook_config_path.exists():
            probe = subprocess.run(
                f"{cfg.activate_command} && are-benchmark run --help",
                shell=True, capture_output=True, text=True, timeout=30,
            )
            if "--hook-config" in (probe.stdout or ""):
                parts.append(f"--hook-config {hook_config_path}")
            else:
                logger.warning(
                    "Installed are-benchmark does not support --hook-config; skipping hook config"
                )

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
