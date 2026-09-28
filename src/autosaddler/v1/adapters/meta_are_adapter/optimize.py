#!/usr/bin/env python3
"""AutoSaddler training script for Meta-ARE default agent.

Loads configuration, reads train/val scenario IDs, instantiates the
MetaAREAdapter, and runs the optimization loop to evolve system prompts.

Usage:
    python -m autosaddler.v1.adapters.meta_are_adapter.optimize \\
        --config configs/v1/meta_are_activesaddler.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosaddler.v1.utils.config import load_yaml_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _ensure_resume_transaction_complete(run_dir: Path) -> None:
    marker = run_dir / ".resume_transaction.json"
    if marker.exists():
        raise RuntimeError(
            f"Interrupted resume transaction found: {marker}. Restore the "
            "run checkpoint before resuming."
        )



def load_config(config_path: str) -> dict[str, Any]:
    """Load an environment-expanded YAML config with optional overlays."""
    return load_yaml_config(config_path)


def _is_sensitive_config_key(key: str) -> bool:
    normalized = "_".join(
        part for part in "".join(
            character if character.isalnum() else "_"
            for character in key.lower()
        ).split("_") if part
    )
    return (
        normalized in {"token", "password", "passphrase", "authorization", "cookie"}
        or normalized.endswith(("_api_key", "_token", "_secret", "_credential", "_private_key"))
        or normalized in {"api_key", "apikey", "secret", "credential", "private_key"}
    )


def _redact_resume_config(value: Any, *, key: str = "") -> Any:
    """Return a JSON-safe config tree with credential values removed."""
    if key and _is_sensitive_config_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(child_key): _redact_resume_config(
                child_value,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_resume_config(item) for item in value]
    return value


def _effective_config_sha256(cfg: dict[str, Any]) -> str:
    """Hash the fully resolved execution config without credential material."""
    canonical = json.dumps(
        _redact_resume_config(cfg),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _resume_config_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return resume-critical config values without persisting credentials."""
    from autosaddler.v1.proposer.autosaddler.prompt_builder import resolve_prompt_bundle
    from autosaddler.v1.proposer.autosaddler.strategy_settings import (
        resolve_strategy_settings,
    )
    from autosaddler.v1.sdk_session import build_sdk_retry_config

    adapter = cfg.get("adapter", {})
    optimization = cfg.get("optimization", {})
    autosaddler = cfg.get("autosaddler", {})
    sdk = cfg.get("sdk", {})
    claude = sdk.get("claude") if isinstance(sdk.get("claude"), dict) else sdk
    copilot = sdk.get("copilot", {})
    provider = copilot.get("provider", {}) if isinstance(copilot, dict) else {}
    retry = build_sdk_retry_config(sdk)
    strategy_settings = resolve_strategy_settings(autosaddler)
    prompt_bundle = resolve_prompt_bundle(
        sampling_strategy=strategy_settings.strategy.name.value,
    )

    return {
        "effective_config_sha256": _effective_config_sha256(cfg),
        "dataset": cfg.get("dataset", {}),
        "adapter": {
            key: adapter.get(key)
            for key in (
                "base_branch",
                "agent",
                "model",
                "model_provider",
                "model_endpoint",
                "model_azure_config_dir",
                "reasoning_effort",
                "judge_model",
                "judge_provider",
                "judge_endpoint",
                "judge_azure_config_dir",
                "dataset_path",
                "max_concurrent",
                "max_concurrent_dev",
            )
        },
        "optimization": {"seed": optimization.get("seed", 42)},
        "autosaddler": {
            key: autosaddler.get(key)
            for key in (
                "claude_agent_sdk_model",
                "train_minibatch_size",
                "capability_phase_iterations",
                "capability_phase_epochs",
                "capability_transition_mode",
                "capability_phase_max_iterations",
                "skip_session0",
            )
        } | {
            "sampling_strategy": strategy_settings.strategy.name.value,
            "sampler_family": strategy_settings.strategy.sampler_family,
            "strategy_settings": strategy_settings.to_fingerprint_dict(),
            "prompt_bundle": {
                "sha256": prompt_bundle.sha256,
                "renderer_sha256": prompt_bundle.renderer_sha256,
                "skills": list(prompt_bundle.skill_names),
                "sessions": list(prompt_bundle.session_numbers),
                "pattern_cli_capabilities": sorted(
                    prompt_bundle.pattern_cli_capabilities
                ),
            },
        },
        "sdk": {
            "backend": sdk.get("backend", "claude"),
            "retry": {
                "content_filter_max_retries": retry.content_filter_max_retries,
                "content_filter_retry_delays_s": list(
                    retry.content_filter_retry_delays_s
                ),
            },
            "claude": {
                key: claude.get(key)
                for key in ("auth_mode", "base_url", "model", "effort", "azure_resource")
            },
            "copilot": {
                "model": copilot.get("model") if isinstance(copilot, dict) else None,
                "effort": copilot.get("effort") if isinstance(copilot, dict) else None,
                "provider": {
                    key: provider.get(key)
                    for key in ("type", "base_url", "wire_api", "model_id", "wire_model")
                },
            },
        },
    }


def _write_or_validate_run_config(
    run_dir: Path,
    config_path: str,
    cfg: dict[str, Any],
    *,
    resume: bool,
    train_ids: list[str] | None = None,
    val_ids: list[str] | None = None,
    strict: bool = False,
) -> None:
    """Persist config provenance and optionally enforce it on resume."""
    metadata_path = run_dir / "run_config.json"
    fingerprint = _resume_config_fingerprint(cfg)
    source_path = Path(config_path).resolve()
    runtime = {
        "train_size": len(train_ids) if train_ids is not None else None,
        "val_size": len(val_ids) if val_ids is not None else None,
        "train_ids_sha256": (
            hashlib.sha256(json.dumps(train_ids).encode()).hexdigest()
            if train_ids is not None
            else None
        ),
        "val_ids_sha256": (
            hashlib.sha256(json.dumps(val_ids).encode()).hexdigest()
            if val_ids is not None
            else None
        ),
    }

    if resume:
        if not metadata_path.exists():
            if strict:
                raise ValueError(
                    "Strict resume config verification requires run_config.json; "
                    f"none was found in {run_dir}"
                )
            logger.warning(
                "No run_config.json found in legacy run %s; "
                "resume config cannot be verified",
                run_dir,
            )
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("fingerprint")
        expected_runtime = metadata.get("runtime") or {}
        runtime_mismatch = any(
            expected_runtime.get(key) is not None
            and runtime.get(key) is not None
            and expected_runtime[key] != runtime[key]
            for key in runtime
        )
        if expected != fingerprint or runtime_mismatch:
            changed_sections = [
                key
                for key in sorted(set(expected or {}) | set(fingerprint))
                if (expected or {}).get(key) != fingerprint.get(key)
            ]
            if runtime_mismatch:
                changed_sections.append("runtime_dataset")
            message = (
                "Resume config does not match the original run "
                f"(changed sections: {', '.join(changed_sections) or 'unknown'}). "
                f"Original config: {metadata.get('config_path', '<unknown>')}"
            )
            if strict:
                raise ValueError(message)
            logger.warning("%s; continuing for source-compatible resume", message)
            return
        logger.info("Resume config fingerprint matches %s", metadata_path)
        return

    raw_config = source_path.read_bytes()
    metadata = {
        "version": 3,
        "config_path": str(source_path),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "fingerprint": fingerprint,
        "runtime": runtime,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(metadata_path)


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
    from autosaddler.v1.proposer.autosaddler.strategy_settings import (
        resolve_strategy_settings,
    )
    from autosaddler.v1.sdk_session import SdkConfig

    as_cfg = cfg.get("autosaddler", {})
    seed = opt_cfg.get("seed", 42)
    strategy_settings = resolve_strategy_settings(as_cfg)

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
        capability_transition_mode=as_cfg.get("capability_transition_mode", "iterations"),
        capability_phase_max_iterations=as_cfg.get("capability_phase_max_iterations", 0),
        skip_session0=as_cfg.get("skip_session0", False),
        # ActiveSaddler: infinite-armed bandit curriculum
        sampling_strategy=strategy_settings.strategy.name.value,
        eta=strategy_settings.eta,
        softmax_temperature=strategy_settings.softmax_temperature,
        min_prob=strategy_settings.min_prob,
        pattern_extraction_timeout=strategy_settings.pattern_extraction_timeout,
        arm_scoring_timeout=strategy_settings.arm_scoring_timeout,
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
    parser.add_argument(
        "--seed-eval-source",
        type=str,
        default=None,
        help=(
            "Reuse a prior initial-harness seed_val result instead of "
            "re-running the seed evaluation. Accepts a seed_val_* cycle dir, "
            "a run timestamp dir (with cycles/seed_val_*), or a run/ dir. "
            "Overrides adapter.seed_eval_source from the config."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Resume a previous run from an existing run directory that "
            "contains state.bin. The engine reloads the checkpoint (GEPA "
            "state, DAG, epoch-shuffle RNG or bandit state) and continues from the next "
            "iteration. A new timestamp dir is NOT created. The run must have "
            "stopped at a clean iteration boundary."
        ),
    )
    parser.add_argument(
        "--strict-resume-config",
        action="store_true",
        help="Reject resume when config or dataset identity differs from run_config.json.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    # CLI override for the seed-eval reuse source (takes priority over config).
    if args.seed_eval_source:
        cfg.setdefault("adapter", {})["seed_eval_source"] = args.seed_eval_source
        logger.info("Seed-eval reuse source (from CLI): %s", args.seed_eval_source)

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
        session_root = (
            Path(args.resume).resolve()
            if args.resume
            else session_root_base / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        )
        logger.info("=== DRY RUN ===")
        logger.info("Session root: %s", session_root)
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
    from autosaddler.v1.sdk_session import (
        SdkConfig,
        build_copilot_provider,
        build_sdk_retry_config,
    )

    sdk_cfg = cfg.get("sdk", {})
    # Nested sections are the canonical schema. Falling back to the sdk block
    # keeps Azure configs written before backend selection was introduced.
    claude_cfg = sdk_cfg.get("claude")
    if not isinstance(claude_cfg, dict):
        claude_cfg = sdk_cfg
    copilot_cfg = sdk_cfg.get("copilot", {})

    sdk_config = SdkConfig(
        backend=sdk_cfg.get("backend", "claude"),
        # Claude Agent SDK settings (from sdk.claude section)
        claude_base_url=(
            os.environ.get("ANTHROPIC_BASE_URL")
            or claude_cfg.get("base_url", "https://api.anthropic.com")
        ),
        claude_api_key=(
            os.environ.get("ANTHROPIC_API_KEY")
            or claude_cfg.get("api_key", "")
            or "EMPTY"
        ),
        claude_permission_mode=claude_cfg.get("permission_mode", "bypassPermissions"),
        claude_model=claude_cfg.get("model"),
        claude_auth_mode=claude_cfg.get("auth_mode", "api_key"),
        claude_azure_config_dir=claude_cfg.get("azure_config_dir"),
        claude_azure_resource=claude_cfg.get("azure_resource"),
        claude_custom_headers=claude_cfg.get("custom_headers", {}),
        claude_token_helper_ttl_ms=claude_cfg.get("token_helper_ttl_ms", 2_700_000),
        claude_effort=claude_cfg.get("effort", "max"),
        claude_allowed_tools=claude_cfg.get("allowed_tools"),
        claude_setting_sources=claude_cfg.get("setting_sources"),
        claude_mcp_servers=claude_cfg.get("mcp_servers"),
        claude_plugins=claude_cfg.get("plugins"),
        # Copilot SDK settings (from sdk.copilot section)
        copilot_model=copilot_cfg.get("model"),
        copilot_effort=copilot_cfg.get("effort", "max"),
        copilot_allowed_tools=copilot_cfg.get("allowed_tools"),
        copilot_provider=build_copilot_provider(copilot_cfg),
        retry=build_sdk_retry_config(sdk_cfg),
    )
    adapter.cfg.sdk_config = sdk_config

    # --------------- Session root ---------------
    opt_cfg = cfg.get("optimization", {})
    adapter_cfg = cfg.get("adapter", {})
    session_root_base = Path(
        adapter_cfg.get("session_root_base", "outputs")
    )
    if args.resume:
        # Resume mode: reuse the existing run directory in place.
        session_root = Path(args.resume).resolve()
        _ensure_resume_transaction_complete(session_root)
        if not (session_root / "state.bin").exists():
            raise FileNotFoundError(
                f"Cannot resume: {session_root}/state.bin not found. "
                f"Provide a run directory that contains state.bin."
            )
        logger.info("RESUME mode: continuing from existing run %s", session_root)
    else:
        session_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        session_root = session_root_base / session_ts
    adapter.set_session_root(session_root)
    run_dir = str(session_root)

    if args.resume:
        _write_or_validate_run_config(
            session_root,
            args.config,
            cfg,
            resume=True,
            train_ids=train_ids,
            val_ids=val_ids,
            strict=args.strict_resume_config,
        )
    else:
        _write_or_validate_run_config(
            session_root,
            args.config,
            cfg,
            resume=False,
            train_ids=train_ids,
            val_ids=val_ids,
        )

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
    if max_candidate_proposals is not None and not args.resume:
        from autosaddler.v1.utils.stop_condition import MaxCandidateProposalsStopper
        stop_callbacks = [MaxCandidateProposalsStopper(max_candidate_proposals)]
    elif max_candidate_proposals is not None and args.resume:
        # state.i counts engine iterations (incl. failed proposals) and may
        # already exceed max_candidate_proposals; rely on max_metric_calls.
        logger.info(
            "RESUME mode: ignoring max_candidate_proposals=%s",
            max_candidate_proposals,
        )
    # Graceful stop on SIGINT/SIGTERM: stops at the next iteration boundary,
    # leaving a clean checkpoint (state.bin + DAG + sampler state).
    from autosaddler.v1.utils.stop_condition import SignalStopper
    stop_callbacks = (stop_callbacks or []) + [SignalStopper()]

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
