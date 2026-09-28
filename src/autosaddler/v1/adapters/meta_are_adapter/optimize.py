#!/usr/bin/env python3
"""AutoSaddler training script for Meta-ARE default agent.

Loads configuration, reads train/val scenario IDs, instantiates the
MetaAREAdapter, and runs the optimization loop to evolve system prompts.

Usage:
    python -m autosaddler.v1.adapters.meta_are_adapter.optimize \\
    --config configs/v1/meta_are.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)



def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file with environment variable expansion.

    Supports ``${VAR}``, ``$VAR``, and ``${VAR:-default}`` placeholders.
    """
    import re
    with open(config_path) as f:
        raw = f.read()

    def _expand_var(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(3)  # group 3 = default after :-
        val = os.environ.get(var_name, "")
        if val:
            return val
        if default is not None:
            return default
        return ""

    # Handle ${VAR:-default} and ${VAR} patterns
    expanded = re.sub(r'\$\{([^}:]+)(:-([^}]*))?\}', _expand_var, raw)
    # Handle $VAR patterns (simple, no braces)
    expanded = os.path.expandvars(expanded)
    return yaml.safe_load(expanded)


def load_scenario_ids(file_path: str) -> list[str]:
    """Load scenario IDs from a JSON config or newline-separated text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario ID file not found: {file_path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        ids = [sid for group in data.values() for sid in group]
    else:
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    logger.info("Loaded %d scenario IDs from %s", len(ids), file_path)
    return ids


def _build_autosaddler_proposer(
    *,
    cfg: dict[str, Any],
    opt_cfg: dict[str, Any],
    trainset: list,
    adapter: Any,
    run_dir: str,
    sdk_config: Any = None,
) -> Any:
    """Build the AutoSaddler proposer from config."""
    from autosaddler.v1.logging.logger import Logger
    from autosaddler.v1.proposer.autosaddler import AutoSaddlerProposer
    from autosaddler.v1.proposer.autosaddler.proposer import EvolutionDAGConfig
    from autosaddler.v1.sdk_session import SdkConfig

    as_cfg = cfg.get("autosaddler", {})
    seed = opt_cfg.get("seed", 42)

    evo_dag_config = EvolutionDAGConfig(
        claude_agent_sdk_model=as_cfg.get("claude_agent_sdk_model", "Claude Opus 4.6"),
        copilot_model=as_cfg.get("copilot_model", "claude-opus-4.6"),
        diagnosis_patch_timeout=as_cfg.get("diagnosis_patch_timeout", 18000.0),
        reflection_timeout=as_cfg.get("reflection_timeout", 3600.0),
        candidate_selection_timeout=as_cfg.get("candidate_selection_timeout", 1800.0),
        train_minibatch_size=as_cfg.get("train_minibatch_size", 10),
        seed=seed,
        sdk_config=sdk_config or SdkConfig(),
        capability_phase_iterations=as_cfg.get("capability_phase_iterations", 0),
        capability_phase_epochs=as_cfg.get("capability_phase_epochs", 1),
        skip_session0=as_cfg.get("skip_session0", False),
    )

    return AutoSaddlerProposer(
        logger=Logger(str(Path(run_dir) / "run_log.txt")),
        trainset=trainset,
        adapter=adapter,
        config=evo_dag_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AutoSaddler optimization on Meta-ARE default agent",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running optimization",
    )
    parser.add_argument(
        "--mutation-strategy",
        type=str,
        choices=["autosaddler"],
        default="autosaddler",
        help="Mutation strategy (default: autosaddler)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    # Resolve relative paths against config file directory
    config_dir = Path(args.config).resolve().parent

    # --------------- Dataset ---------------
    dataset_cfg = cfg.get("dataset", {})
    train_file = Path(dataset_cfg["train_file"])
    val_file = Path(dataset_cfg["val_file"])
    if not train_file.is_absolute():
        train_file = config_dir / train_file
    if not val_file.is_absolute():
        val_file = config_dir / val_file
    train_ids = load_scenario_ids(str(train_file))
    val_ids = load_scenario_ids(str(val_file))

    adapter_cfg = cfg.get("adapter", {})
    meta_are_repo = adapter_cfg.get("meta_are_repo")
    if not isinstance(meta_are_repo, str) or not meta_are_repo.strip():
        raise ValueError(
            "adapter.meta_are_repo must be configured; set the required META_ARE_REPO environment variable"
        )

    if args.dry_run:
        session_root_base = Path(adapter_cfg.get("session_root_base", "outputs"))
        session_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        logger.info("=== DRY RUN ===")
        logger.info("Session root: %s", session_root_base / session_ts)
        logger.info("Train scenarios: %s", train_ids)
        logger.info("Val scenarios: %s", val_ids)
        return

    from autosaddler.v1.adapters.meta_are_adapter.meta_are_adapter import MetaAREDataInst

    trainset = [MetaAREDataInst(scenario_id=sid) for sid in train_ids]
    valset = [MetaAREDataInst(scenario_id=sid) for sid in val_ids]

    # --------------- Seed candidate ---------------
    # The seed harness is the unmodified base branch. We create a worktree
    # from the base branch and pass it directly — no patching needed.
    seed_candidate: dict[str, str] = {}
    if "seed_candidate" in cfg:
        seed_candidate.update(cfg["seed_candidate"])

    # --------------- Adapter ---------------
    from autosaddler.v1.adapters.meta_are_adapter.meta_are_adapter import MetaAREAdapter

    adapter = MetaAREAdapter(config=cfg.get("adapter", {}))

    # --------------- SDK backend config ---------------
    from autosaddler.v1.sdk_session import SdkConfig

    sdk_cfg = cfg.get("sdk", {})
    claude_cfg = sdk_cfg.get("claude", {})
    copilot_cfg = sdk_cfg.get("copilot", {})

    sdk_config = SdkConfig(
        backend=sdk_cfg.get("backend", "claude"),
        # Claude Agent SDK settings (from sdk.claude section)
        claude_base_url=os.environ.get("ANTHROPIC_BASE_URL", claude_cfg.get("base_url", "https://api.anthropic.com")),
        claude_api_key=os.environ.get("ANTHROPIC_API_KEY", claude_cfg.get("api_key", "")) or "EMPTY",
        claude_permission_mode=claude_cfg.get("permission_mode", "bypassPermissions"),
        claude_model=claude_cfg.get("model"),
        claude_effort=claude_cfg.get("effort", "max"),
        claude_allowed_tools=claude_cfg.get("allowed_tools"),
        claude_setting_sources=claude_cfg.get("setting_sources"),
        claude_mcp_servers=claude_cfg.get("mcp_servers"),
        claude_plugins=claude_cfg.get("plugins"),
        # Copilot SDK settings (from sdk.copilot section)
        copilot_model=copilot_cfg.get("model"),
        copilot_effort=copilot_cfg.get("effort", "max"),
        copilot_allowed_tools=copilot_cfg.get("allowed_tools"),
    )
    adapter.cfg.sdk_config = sdk_config

    # --------------- Session root ---------------
    opt_cfg = cfg.get("optimization", {})
    session_root_base = Path(
        adapter_cfg.get("session_root_base", "outputs")
    )
    session_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_root = session_root_base / session_ts
    adapter.set_session_root(session_root)
    run_dir = str(session_root)

    # --------------- Seed candidate ---------------
    # Create a clean base-branch worktree for seed eval.
    # The seed prompts are identical to what's already in base_branch,
    # so we skip the unnecessary SDK patching by using autosaddler format.
    seed_worktree, _ = adapter._worktree_pool.get_or_create(
        seed_candidate,
        lambda wt, cand: None,  # no-op — base branch already has seed prompts
    )
    seed_candidate["__autosaddler_worktree__"] = str(seed_worktree)
    logger.info("Seed worktree (no patching needed): %s", seed_worktree)

    # --------------- Run optimization ---------------
    from autosaddler.v1 import optimize

    Path(run_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting AutoSaddler optimization...")
    logger.info("  Components: %s", list(seed_candidate.keys()))
    logger.info("  Train examples: %d", len(trainset))
    logger.info("  Val examples: %d", len(valset))
    logger.info("  Max metric calls: %s", opt_cfg.get("max_metric_calls"))
    logger.info("  Max candidate proposals: %s", opt_cfg.get("max_candidate_proposals"))

    max_metric_calls = opt_cfg.get("max_metric_calls")
    max_candidate_proposals = opt_cfg.get("max_candidate_proposals")

    if max_metric_calls is None and max_candidate_proposals is None:
        raise ValueError(
            "At least one of 'max_metric_calls' or 'max_candidate_proposals' must be set."
        )

    stop_callbacks: list | None = None
    if max_candidate_proposals is not None:
        from autosaddler.v1.utils.stop_condition import MaxCandidateProposalsStopper
        stop_callbacks = [MaxCandidateProposalsStopper(max_candidate_proposals)]

    reflective_proposer_override = _build_autosaddler_proposer(
        cfg=cfg,
        opt_cfg=opt_cfg,
        trainset=trainset,
        adapter=adapter,
        run_dir=run_dir,
        sdk_config=sdk_config,
    )
    logger.info("Using AutoSaddler proposer (DAG-based evolution with phase scheduling)")

    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflective_proposer_override=reflective_proposer_override,
        max_metric_calls=max_metric_calls,
        stop_callbacks=stop_callbacks,
        frontier_type=opt_cfg.get("frontier_type", "instance"),
        perfect_score=opt_cfg.get("perfect_score", 1.0),
        seed=opt_cfg.get("seed", 42),
        run_dir=run_dir,
        display_progress_bar=opt_cfg.get("display_progress_bar", True),
    )

    # --------------- Output results ---------------
    logger.info("Optimization complete!")
    logger.info("  Best candidate index: %d", result.best_idx)
    logger.info("  Best validation score: %.4f", result.val_aggregate_scores[result.best_idx])
    logger.info("  Total candidates explored: %d", len(result.candidates))
    logger.info("  All val scores: %s", result.val_aggregate_scores)

    best = result.best_candidate
    if isinstance(best, str):
        best = {"prompt": best}

    if run_dir:
        output_file = Path(run_dir) / "best_candidate.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "best_candidate": best,
                    "best_score": result.val_aggregate_scores[result.best_idx],
                    "total_candidates": len(result.candidates),
                    "all_scores": result.val_aggregate_scores,
                },
                f,
                indent=2,
            )
        logger.info("Best candidate saved to %s", output_file)


if __name__ == "__main__":
    main()
