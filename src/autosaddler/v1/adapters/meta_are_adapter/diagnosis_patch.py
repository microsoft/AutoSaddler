#!/usr/bin/env python3
"""Diagnosis-patch runner for individual failed scenarios.

For each failed scenario from an initial harness evaluation, creates an
isolated git worktree from the base branch, installs CLAUDE.md + skills,
builds a Session 1 prompt using the existing harness traces, and runs an
SDK agent session to diagnose the root cause and apply patches.

No re-evaluation is performed.  The purpose is to collect:
  1. ``proposer_reasoning.md`` — written by the SDK agent in the worktree
  2. ``*_patch.json``          — SDK session metadata (tool calls, tokens)

Usage:
    python -m autosaddler.v1.adapters.meta_are_adapter.diagnosis_patch \\
        --initial-harness /path/to/initial_harness/train_YYYYMMDD-HHMMSS \\
        --config configs/v1/meta_are.yaml \
        [--phase capability] \\
        [--scenario-filter id1,id2] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import git

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading (reused from optimize.py)
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config with env-var expansion."""
    import re

    import yaml

    with open(config_path) as f:
        raw = f.read()

    def _expand_var(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(3)
        val = os.environ.get(var_name, "")
        if val:
            return val
        if default is not None:
            return default
        return ""

    expanded = re.sub(r'\$\{([^}:]+)(:-([^}]*))?\}', _expand_var, raw)
    expanded = os.path.expandvars(expanded)
    return yaml.safe_load(expanded)


def _build_sdk_config(cfg: dict[str, Any]) -> Any:
    """Build SdkConfig from the full config dict."""
    from autosaddler.v1.sdk_session import SdkConfig

    sdk_cfg = cfg.get("sdk", {})
    claude_cfg = sdk_cfg.get("claude", {})
    copilot_cfg = sdk_cfg.get("copilot", {})

    return SdkConfig(
        backend=sdk_cfg.get("backend", "claude"),
        claude_base_url=os.environ.get(
            "ANTHROPIC_BASE_URL",
            claude_cfg.get("base_url", "https://api.anthropic.com"),
        ),
        claude_api_key=os.environ.get(
            "ANTHROPIC_API_KEY", claude_cfg.get("api_key", ""),
        ) or "EMPTY",
        claude_permission_mode=claude_cfg.get("permission_mode", "bypassPermissions"),
        claude_model=claude_cfg.get("model"),
        claude_effort=claude_cfg.get("effort", "max"),
        claude_allowed_tools=claude_cfg.get("allowed_tools"),
        claude_setting_sources=claude_cfg.get("setting_sources"),
        claude_mcp_servers=claude_cfg.get("mcp_servers"),
        claude_plugins=claude_cfg.get("plugins"),
        copilot_model=copilot_cfg.get("model"),
        copilot_effort=copilot_cfg.get("effort", "max"),
        copilot_allowed_tools=copilot_cfg.get("allowed_tools"),
    )


# ---------------------------------------------------------------------------
# Failed scenario extraction
# ---------------------------------------------------------------------------

def _extract_failed_scenarios(
    initial_harness_dir: Path,
) -> list[dict[str, Any]]:
    """Parse output.jsonl and return failed scenario entries."""
    output_jsonl = initial_harness_dir / "results" / "output.jsonl"
    if not output_jsonl.exists():
        raise FileNotFoundError(f"output.jsonl not found: {output_jsonl}")

    failed: list[dict[str, Any]] = []
    with open(output_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            meta = entry.get("metadata", {})
            if meta.get("status") == "failed":
                failed.append({
                    "scenario_id": entry.get("task_id")
                    or meta.get("scenario_id", ""),
                    "score": float(entry.get("score", 0.0)),
                    "rationale": meta.get("rationale", ""),
                    "has_exception": meta.get("has_exception", False),
                    "exception_message": meta.get("exception_message", ""),
                })
    return failed


# ---------------------------------------------------------------------------
# Worktree helpers
# ---------------------------------------------------------------------------

def _create_worktree(
    repo_path: Path,
    worktree_dir: Path,
    scenario_id: str,
    base_branch: str,
) -> Path:
    """Create a git worktree for a single scenario."""
    short_id = scenario_id.replace("scenario_universe_29_", "")
    wt_name = f"diag_{short_id}"
    wt_path = worktree_dir / wt_name
    branch_name = f"diagnosis/{short_id}"

    repo = git.Repo(repo_path)
    wt_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        repo.git.worktree("add", str(wt_path), "-b", branch_name, base_branch)
    except git.GitCommandError:
        logger.warning("Stale worktree/branch for %s — cleaning up", wt_name)
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
        repo.git.worktree("add", str(wt_path), "-b", branch_name, base_branch)

    logger.info("Created worktree %s from %s", wt_path, base_branch)
    return wt_path


def _create_cycle_dir_with_scenario_results(
    cycles_dir: Path,
    scenario_id: str,
    initial_results_dir: Path,
) -> Path:
    """Create a cycle directory with only this scenario's results.

    Instead of symlinking the entire results dir (which contains all 75
    scenarios), creates a ``run/`` subdirectory with:
    - ``output.jsonl`` containing only this scenario's entry
    - ``lite/`` with symlinks to only this scenario's lite trace(s)
    - ``hf/`` with symlinks to only this scenario's hf trace(s)
    """
    short_id = scenario_id.replace("scenario_universe_29_", "")
    cycle_dir = cycles_dir / f"diag_{short_id}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    run_dir = cycle_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Filter output.jsonl to this scenario only
    src_jsonl = initial_results_dir / "output.jsonl"
    if src_jsonl.exists():
        dst_jsonl = run_dir / "output.jsonl"
        with open(src_jsonl) as f:
            lines = f.readlines()
        with open(dst_jsonl, "w") as f:
            for line in lines:
                line_s = line.strip()
                if not line_s:
                    continue
                entry = json.loads(line_s)
                task_id = entry.get("task_id") or entry.get("metadata", {}).get("scenario_id", "")
                if task_id == scenario_id:
                    f.write(line_s + "\n")

    # 2. Symlink matching lite trace files
    src_lite = initial_results_dir / "lite"
    if src_lite.exists():
        dst_lite = run_dir / "lite"
        dst_lite.mkdir(parents=True, exist_ok=True)
        for f in src_lite.iterdir():
            if scenario_id in f.name:
                dst = dst_lite / f.name
                if not dst.exists():
                    dst.symlink_to(f)

    # 3. Symlink matching hf trace files
    src_hf = initial_results_dir / "hf"
    if src_hf.exists():
        dst_hf = run_dir / "hf"
        dst_hf.mkdir(parents=True, exist_ok=True)
        for f in src_hf.iterdir():
            if scenario_id in f.name:
                dst = dst_hf / f.name
                if not dst.exists():
                    dst.symlink_to(f)

    # 4. Symlink benchmark_stats.json if exists (for reference)
    src_stats = initial_results_dir / "benchmark_stats.json"
    if src_stats.exists():
        dst_stats = run_dir / "benchmark_stats.json"
        if not dst_stats.exists():
            dst_stats.symlink_to(src_stats)

    return cycle_dir


# ---------------------------------------------------------------------------
# SDK session helpers
# ---------------------------------------------------------------------------

def _run_async(coro):  # noqa: ANN001, ANN202
    """Run an async coroutine from sync code."""
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


def _run_sdk_session_with_retry(
    *,
    worktree_path: Path,
    prompt: str,
    model: str,
    timeout: float,
    sdk_config: Any,
    extra_env: dict[str, str] | None = None,
    max_retries: int = 5,
    initial_backoff: float = 60.0,
) -> dict[str, Any] | None:
    """Run an SDK session with exponential backoff on rate-limit errors."""
    from autosaddler.v1.sdk_session import RateLimitError, run_sdk_session

    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        old_env: dict[str, str | None] = {}
        if extra_env:
            for key, value in extra_env.items():
                old_env[key] = os.environ.get(key)
                os.environ[key] = value
        try:
            result = _run_async(
                run_sdk_session(
                    cwd=worktree_path,
                    prompt=prompt,
                    model=model,
                    timeout=timeout,
                    sdk_config=sdk_config,
                    track_events=True,
                )
            )
            return result
        except RateLimitError:
            if attempt < max_retries:
                logger.warning(
                    "Rate-limited (attempt %d/%d). Retrying in %.0fs...",
                    attempt, max_retries, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 600.0)
            else:
                logger.error("Rate-limited after %d retries — giving up", max_retries)
                return None
        except Exception:
            logger.exception("SDK session failed")
            return None
        finally:
            if extra_env:
                for key in extra_env:
                    if old_env.get(key) is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_env[key]  # type: ignore[assignment]
    return None


def _extract_session_info(
    session_result: dict[str, Any],
    model: str,
    timeout: float,
    output_dir: str,
    iteration: int,
    candidate_idx: int,
) -> Path | None:
    """Extract session info and write patch.json. Returns the JSON path."""
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
                    u.get("cache_read_input_tokens", 0)
                    or u.get("cache_read_tokens", 0)
                    or 0
                )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / f"iter{iteration:02d}_c{candidate_idx}_patch.json"

        session_data = {
            "model": model,
            "timeout": timeout,
            "session_type": "patch",
            "iteration": iteration,
            "candidate_idx": candidate_idx,
            "tool_call_count": len(tool_calls),
            "turns": turns,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "tool_calls": tool_calls,
            "usage": usage,
            "raw_response": session_result.get("raw_response", ""),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return json_path
    except Exception:
        logger.exception("Failed to extract session info")
        return None


# ---------------------------------------------------------------------------
# Verification (lightweight)
# ---------------------------------------------------------------------------

def _verify_worktree(
    worktree: Path,
    activate_command: str,
    import_check_statement: str,
) -> bool:
    """Verify the modified worktree (syntax + import check)."""
    import subprocess

    # Find modified .py files
    try:
        repo = git.Repo(worktree)
        changed: list[str] = []
        if repo.head.commit.parents:
            parent = repo.head.commit.parents[0]
            diffs = parent.diff(repo.head.commit)
            changed.extend(d.a_path or d.b_path for d in diffs if d.a_path or d.b_path)
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        changed.extend(staged)
        unstaged = [item.a_path for item in repo.index.diff(None)]
        changed.extend(unstaged)
        changed.extend(repo.untracked_files)
        modified_py = [
            f for f in set(changed)
            if f.endswith(".py")
            and not f.startswith(".claude/")
            and not f.startswith("bin/")
        ]
    except Exception:
        modified_py = []

    # Syntax check
    for py_file in modified_py:
        py_path = worktree / py_file
        if not py_path.exists():
            continue
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(py_path)],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            logger.error(
                "Syntax error in %s:\n%s",
                py_file, result.stderr.decode(errors="replace"),
            )
            return False

    # Import check
    if activate_command and import_check_statement:
        shell_cmd = (
            f"{activate_command} && "
            f"PYTHONPATH={worktree} python -c \"{import_check_statement}\""
        )
        result = subprocess.run(
            ["bash", "-c", shell_cmd],
            cwd=str(worktree),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(
                "Import check failed:\n%s",
                result.stderr.decode(errors="replace"),
            )
            return False

    return True


def _get_changed_files(worktree: Path) -> list[str]:
    """List files changed relative to parent commit."""
    try:
        repo = git.Repo(worktree)
        changed: list[str] = []
        if repo.head.commit.parents:
            parent = repo.head.commit.parents[0]
            diffs = parent.diff(repo.head.commit)
            changed.extend(d.a_path or d.b_path for d in diffs if d.a_path or d.b_path)
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        changed.extend(staged)
        unstaged = [item.a_path for item in repo.index.diff(None)]
        changed.extend(unstaged)
        changed.extend(repo.untracked_files)
        return sorted(
            f for f in set(changed)
            if not f.startswith(".claude/")
            and not f.startswith("bin/")
            and f != "CLAUDE.md"
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_one_scenario(
    *,
    scenario: dict[str, Any],
    repo_path: Path,
    worktree_dir: Path,
    cycles_dir: Path,
    session_root: str,
    initial_results_dir: Path,
    base_branch: str,
    phase: str,
    model: str,
    sdk_config: Any,
    diagnosis_patch_timeout: float,
    activate_command: str,
    import_check_statement: str,
    candidate_idx: int,
) -> dict[str, Any]:
    """Run diagnosis-patch for a single scenario. Returns a result dict."""
    from autosaddler.v1.proposer.autosaddler.prompt_builder import (
        build_claude_md,
        build_session1_prompt,
        build_skill_prefix,
        install_evo_dag_cli,
        install_prompts_and_skills,
    )

    scenario_id = scenario["scenario_id"]
    logger.info("=" * 60)
    logger.info("Processing %s (C%d)", scenario_id, candidate_idx)
    logger.info("=" * 60)

    result: dict[str, Any] = {
        "scenario_id": scenario_id,
        "status": "error",
        "worktree": None,
        "patch_json": None,
        "reasoning_md": None,
        "verification_passed": None,
        "files_changed": [],
        "initial_rationale": scenario.get("rationale", ""),
    }

    try:
        # 1. Create worktree
        worktree = _create_worktree(repo_path, worktree_dir, scenario_id, base_branch)
        result["worktree"] = str(worktree)

        # 2. Install CLAUDE.md + skills
        claude_md = build_claude_md()
        install_prompts_and_skills(str(worktree), claude_md, phase=phase)

        # 3. Install evo-dag CLI
        cli_env = install_evo_dag_cli(session_root, str(worktree))

        # 4. Create cycle dir with only this scenario's results
        cycle_dir = _create_cycle_dir_with_scenario_results(
            cycles_dir, scenario_id, initial_results_dir,
        )

        # 5. Build Session 1 prompt
        # Use the initial harness results dir as before_output_dir
        # so the agent can read traces from there
        before_output_dir = str(initial_results_dir)

        session1_prompt = build_session1_prompt(
            iteration=1,
            candidate_idx=candidate_idx,
            worktree_path=str(worktree),
            parent_worktree=str(worktree),  # same as base (no parent chain)
            base_parent_idx=0,
            mini_batch_ids=[scenario_id],
            before_scores={scenario_id: 0.0},
            before_rationales={scenario_id: scenario.get("rationale", "")},
            before_output_dir=before_output_dir,
            phase=phase,
            cherry_pick_parents=None,
        )

        full_prompt = (
            build_skill_prefix(session=1, phase=phase) + session1_prompt
        )

        # 6. Run SDK session
        logger.info("Starting SDK session for %s (timeout=%.0fs)...", scenario_id, diagnosis_patch_timeout)
        session_result = _run_sdk_session_with_retry(
            worktree_path=worktree,
            prompt=full_prompt,
            model=model,
            timeout=diagnosis_patch_timeout,
            sdk_config=sdk_config,
            extra_env=cli_env,
        )

        if session_result is None:
            result["status"] = "sdk_failed"
            logger.error("SDK session failed for %s", scenario_id)
            return result

        # 7. Extract session info → patch.json
        patch_json_path = _extract_session_info(
            session_result,
            model=model,
            timeout=diagnosis_patch_timeout,
            output_dir=str(cycle_dir),
            iteration=1,
            candidate_idx=candidate_idx,
        )
        if patch_json_path:
            result["patch_json"] = str(patch_json_path)

        # 8. Check for proposer_reasoning.md
        reasoning_path = worktree / "proposer_reasoning.md"
        if reasoning_path.exists():
            result["reasoning_md"] = str(reasoning_path)
            logger.info("proposer_reasoning.md found for %s", scenario_id)
        else:
            logger.warning("proposer_reasoning.md NOT found for %s", scenario_id)

        # 9. Verification (non-blocking)
        verified = _verify_worktree(worktree, activate_command, import_check_statement)
        result["verification_passed"] = verified
        if not verified:
            logger.warning("Verification FAILED for %s", scenario_id)

        # 10. Collect changed files
        result["files_changed"] = _get_changed_files(worktree)

        result["status"] = "completed"
        logger.info("Completed %s: %d files changed", scenario_id, len(result["files_changed"]))

    except Exception:
        logger.exception("Error processing %s", scenario_id)
        result["status"] = "error"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diagnosis-patch sessions on failed scenarios from initial harness",
    )
    parser.add_argument(
        "--initial-harness",
        type=str,
        required=True,
        help="Path to the initial harness directory (e.g. .../train_20260608-161836)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="capability",
        choices=["capability", "steering"],
        help="Optimization phase (default: capability)",
    )
    parser.add_argument(
        "--scenario-filter",
        type=str,
        default=None,
        help="Comma-separated scenario short IDs to process (e.g. 71j6lf,7v5wh0). "
             "If not set, all failed scenarios are processed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and exit without running SDK sessions",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    cfg = _load_config(args.config)
    adapter_cfg = cfg.get("adapter", {})
    as_cfg = cfg.get("autosaddler", {})

    meta_are_repo = Path(adapter_cfg.get("meta_are_repo", os.environ.get("META_ARE_REPO", ""))).resolve()
    if not meta_are_repo.exists():
        raise FileNotFoundError(f"META_ARE_REPO not found: {meta_are_repo}")

    base_branch = adapter_cfg.get("base_branch", os.environ.get("META_ARE_BASE_BRANCH", "main"))
    activate_command = adapter_cfg.get("activate_command", "")
    import_check_statement = adapter_cfg.get("import_check_statement", "")

    sdk_config = _build_sdk_config(cfg)
    model = as_cfg.get("claude_agent_sdk_model", "Claude Opus 4.6")
    if sdk_config.backend == "copilot":
        model = sdk_config.copilot_model or as_cfg.get("copilot_model", "claude-opus-4.6")
    diagnosis_patch_timeout = as_cfg.get("diagnosis_patch_timeout", 18000.0)

    # ── Parse initial harness ─────────────────────────────────────────
    initial_harness_dir = Path(args.initial_harness).resolve()
    if not initial_harness_dir.exists():
        raise FileNotFoundError(f"Initial harness dir not found: {initial_harness_dir}")

    initial_results_dir = initial_harness_dir / "results"
    if not initial_results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {initial_results_dir}")

    failed_scenarios = _extract_failed_scenarios(initial_harness_dir)
    logger.info("Found %d failed scenarios", len(failed_scenarios))

    # ── Apply scenario filter ─────────────────────────────────────────
    if args.scenario_filter:
        filter_ids = {s.strip() for s in args.scenario_filter.split(",")}
        # Match against both full ID and short ID
        filtered = []
        for s in failed_scenarios:
            sid = s["scenario_id"]
            short = sid.replace("scenario_universe_29_", "")
            if sid in filter_ids or short in filter_ids:
                filtered.append(s)
        logger.info("Filtered to %d scenarios (from %d)", len(filtered), len(failed_scenarios))
        failed_scenarios = filtered

    if not failed_scenarios:
        logger.info("No failed scenarios to process")
        return

    # ── Session root ──────────────────────────────────────────────────
    session_root_base = Path(adapter_cfg.get("session_root_base", str(meta_are_repo / "autosaddler")))
    session_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_root = session_root_base / f"diagnosis_{session_ts}"
    session_root.mkdir(parents=True, exist_ok=True)

    worktree_dir = session_root / "worktrees"
    worktree_dir.mkdir(parents=True, exist_ok=True)
    cycles_dir = session_root / "cycles"
    cycles_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Session root: %s", session_root)
    logger.info("Phase: %s", args.phase)
    logger.info("Model: %s (backend: %s)", model, sdk_config.backend)
    logger.info("Timeout per scenario: %.0fs", diagnosis_patch_timeout)

    # ── Initialize minimal DAG ────────────────────────────────────────
    from autosaddler.v1.proposer.autosaddler.dag import EvolutionDAG

    dag = EvolutionDAG(str(session_root))
    dag.add_seed_node(
        worktree_path=str(initial_harness_dir / "worktree"),
        score_val=0.0,  # placeholder
    )

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("Would process %d failed scenarios:", len(failed_scenarios))
        for i, s in enumerate(failed_scenarios, 1):
            sid = s["scenario_id"]
            rationale = s.get("rationale", "")
            if len(rationale) > 100:
                rationale = rationale[:100] + "..."
            logger.info("  %d. %s — %s", i, sid, rationale)
        logger.info("Session root would be: %s", session_root)
        return

    # ── Process each scenario ─────────────────────────────────────────
    results: list[dict[str, Any]] = []
    for i, scenario in enumerate(failed_scenarios, 1):
        logger.info(
            "\n[%d/%d] %s",
            i, len(failed_scenarios), scenario["scenario_id"],
        )

        # Each scenario gets a unique candidate index in the DAG
        candidate_idx = i

        # Add DAG node for this scenario
        scenario_id = scenario["scenario_id"]
        short_id = scenario_id.replace("scenario_universe_29_", "")
        placeholder_wt = str(worktree_dir / f"diag_{short_id}")
        dag.add_node(
            iteration=1,
            worktree_path=placeholder_wt,
            base_parent_idx=0,
            mini_batch_ids=[scenario_id],
        )
        dag.add_base_edge(0, candidate_idx)
        dag.save()

        r = _run_one_scenario(
            scenario=scenario,
            repo_path=meta_are_repo,
            worktree_dir=worktree_dir,
            cycles_dir=cycles_dir,
            session_root=str(session_root),
            initial_results_dir=initial_results_dir,
            base_branch=base_branch,
            phase=args.phase,
            model=model,
            sdk_config=sdk_config,
            diagnosis_patch_timeout=diagnosis_patch_timeout,
            activate_command=activate_command,
            import_check_statement=import_check_statement,
            candidate_idx=candidate_idx,
        )
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────
    completed = sum(1 for r in results if r["status"] == "completed")
    sdk_failed = sum(1 for r in results if r["status"] == "sdk_failed")
    errors = sum(1 for r in results if r["status"] == "error")

    summary = {
        "timestamp": session_ts,
        "initial_harness": str(initial_harness_dir),
        "session_root": str(session_root),
        "phase": args.phase,
        "model": model,
        "sdk_backend": sdk_config.backend,
        "total_failed": len(failed_scenarios),
        "completed": completed,
        "sdk_failed": sdk_failed,
        "errors": errors,
        "scenarios": {
            r["scenario_id"]: {
                k: v for k, v in r.items() if k != "scenario_id"
            }
            for r in results
        },
    }

    summary_path = session_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DIAGNOSIS-PATCH COMPLETE")
    logger.info("  Total: %d | Completed: %d | SDK failed: %d | Errors: %d",
                len(failed_scenarios), completed, sdk_failed, errors)
    logger.info("  Session root: %s", session_root)
    logger.info("  Summary: %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
