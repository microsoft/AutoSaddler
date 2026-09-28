#!/usr/bin/env python3
"""pattern CLI tool: query and update the PatternRegistry.

Standalone script deployed into the session root for Agent use.
Reads ``PATTERN_REGISTRY_PATH`` env var to locate ``pattern_registry.json``.

Commands:
    pattern list                     — List all patterns in the canonical table
  pattern show <id>                — Show pattern details
  pattern register --label "..."   — Register new pattern, prints pattern_id
  pattern tag --pattern-id <id> --harness <idx> --trace <dir> --scenario <sid> [--root-cause "..."]
  pattern score [--top-k <n>]      — Show top-k patterns by score
  pattern scenarios --pattern-id <id>  — List scenarios for a pattern
    pattern rate --pattern-id <id> ...   — Record an agent progress score
    pattern decide --action pull|draw    — Record an agent arm decision

The outer loop sets ``PATTERN_CLI_CAPABILITIES`` so each strategy/session can
execute only the commands in its capability manifest.
"""

from __future__ import annotations

import argparse
import os
import sys

from autosaddler.v1.proposer.autosaddler.artifact_paths import iteration_artifact_path
from autosaddler.v1.proposer.autosaddler.pattern_reporting import (
    format_pattern_observation,
    render_pattern_table,
)
from autosaddler.v1.proposer.autosaddler.prompt_builder import render_arm_pull_record


def _load_registry():
    """Load the PatternRegistry from the configured path."""
    registry_path = os.environ.get("PATTERN_REGISTRY_PATH")
    if not registry_path:
        print("ERROR: PATTERN_REGISTRY_PATH environment variable not set.", file=sys.stderr)
        sys.exit(1)

    from autosaddler.v1.proposer.autosaddler.pattern_registry import PatternRegistry

    session_root = os.path.dirname(registry_path)
    registry = PatternRegistry(session_root)
    registry.load()
    return registry


def _get_current_iter() -> int:
    """Get the current iteration from environment or default to 1."""
    return int(os.environ.get("CURRENT_ITERATION", "1"))


def _get_score_params() -> dict:
    """Get score computation parameters from environment."""
    return {
        "eta": float(os.environ.get("ETA", "0.3")),
    }


def _enforce_command_capability(command: str) -> None:
    """Reject commands outside the manifest supplied by the outer loop."""
    manifest = os.environ.get("PATTERN_CLI_CAPABILITIES")
    if manifest is None:
        print(
            "ERROR: PATTERN_CLI_CAPABILITIES is not set; refusing to run "
            "the pattern CLI without an explicit session capability manifest.",
            file=sys.stderr,
        )
        sys.exit(2)
    allowed = {item.strip() for item in manifest.split(",") if item.strip()}
    if command not in allowed:
        strategy = os.environ.get("SAMPLING_STRATEGY", "unknown")
        print(
            f"ERROR: pattern {command} is not available for strategy "
            f"'{strategy}' in this session.",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    """List all patterns with activity, observations, and scenario IDs."""
    registry = _load_registry()
    current_iter = _get_current_iter()
    params = _get_score_params()

    patterns = registry.list_patterns()
    if not patterns:
        print("No patterns registered yet.")
        return

    scores = registry.compute_scores(current_iter, **params)

    print(f"Patterns ({len(patterns)}):\n")
    print(render_pattern_table(registry, scores, include_untagged=True))


def cmd_show(args: argparse.Namespace) -> None:
    """Show details for a specific pattern."""
    registry = _load_registry()
    current_iter = _get_current_iter()
    params = _get_score_params()

    pattern = registry.get_pattern(args.pattern_id)
    if pattern is None:
        print(f"ERROR: Pattern '{args.pattern_id}' not found.", file=sys.stderr)
        available = [p.pattern_id for p in registry.list_patterns()]
        if available:
            print(f"Available: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)

    score = registry.compute_score_for_pattern(pattern.pattern_id, current_iter, **params)

    print(f"Pattern: {pattern.pattern_id}")
    print(f"Label: {pattern.label}")
    print(f"Created: {pattern.created_at}")
    print(f"Last observed: iteration {pattern.last_observed_iteration}")
    print(f"Score: {score:.4f}")

    if pattern.tuples:
        print(f"\nTagged tuples ({len(pattern.tuples)}):")
        for t in pattern.tuples:
            print(f"  - harness=C{t.harness_idx}, scenario={t.scenario_id}")
            print(f"    trace: {t.trace_dir}")

    if pattern.evidence:
        print("\nEvidence (root causes):")
        for sid, rc in pattern.evidence.items():
            print(f"  [{sid}]: {rc}")

    if pattern.observations:
        print(f"\nObservations ({len(pattern.observations)}):")
        for obs in pattern.observations:
            print(f"  {format_pattern_observation(obs)}")


def cmd_register(args: argparse.Namespace) -> None:
    """Register a new failure pattern."""
    registry = _load_registry()

    pattern_id = registry.register(
        args.label,
        created_iteration=_get_current_iter(),
    )
    registry.save()

    # Print the ID so the agent can capture it
    print(f"REGISTERED: {pattern_id}")
    print(f"Label: {args.label}")


def cmd_tag(args: argparse.Namespace) -> None:
    """Tag a (harness, trace, scenario) tuple with a pattern."""
    registry = _load_registry()

    for pid in args.pattern_id:
        try:
            registry.tag(
                pattern_id=pid,
                harness_idx=args.harness,
                trace_dir=args.trace,
                scenario_id=args.scenario,
                root_cause=args.root_cause or "",
            )
            print(f"Tagged: pattern={pid}, harness=C{args.harness}, scenario={args.scenario}")
        except KeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    registry.save()


def cmd_score(args: argparse.Namespace) -> None:
    """Show top-k patterns by score."""
    registry = _load_registry()
    current_iter = _get_current_iter()
    params = _get_score_params()

    top_k = args.top_k or 10
    top_patterns = registry.get_top_patterns_by_score(current_iter, top_k=top_k, **params)

    if not top_patterns:
        print("No patterns with scores.")
        return

    print(f"Top {len(top_patterns)} patterns by score (iter={current_iter}):\n")
    print(f"{'Rank':<5} {'ID':<10} {'Score':<10} {'Label'}")
    print("-" * 60)

    for rank, (pid, score) in enumerate(top_patterns, 1):
        pattern = registry.get_pattern(pid)
        label = pattern.label if pattern else "(unknown)"
        print(f"{rank:<5} {pid:<10} {score:<10.4f} {label}")


def cmd_scenarios(args: argparse.Namespace) -> None:
    """List scenarios tagged with a pattern."""
    registry = _load_registry()

    scenarios = registry.get_scenarios_for_pattern(args.pattern_id)
    if not scenarios:
        print(f"No scenarios tagged with pattern '{args.pattern_id}'.")
        return

    print(f"Scenarios for pattern '{args.pattern_id}' ({len(scenarios)}):")
    for sid in sorted(scenarios):
        patterns_for_sid = registry.get_patterns_for_scenario(sid)
        other_patterns = [p for p in patterns_for_sid if p != args.pattern_id]
        suffix = f"  (also: {', '.join(other_patterns)})" if other_patterns else ""
        print(f"  - {sid}{suffix}")


def cmd_history(args: argparse.Namespace) -> None:
    """Show complete or recent exact-arm pull histories."""
    from autosaddler.v1.proposer.autosaddler.dag import EvolutionDAG

    registry = _load_registry()
    registry_path = os.environ["PATTERN_REGISTRY_PATH"]
    dag = EvolutionDAG(os.path.dirname(registry_path))
    dag.load()

    requested_ids = [
        pattern_id
        for group in (args.pattern_id or [])
        for pattern_id in (group if isinstance(group, list) else [group])
    ]
    if not requested_ids:
        requested_ids = [pattern.pattern_id for pattern in registry.list_patterns()]

    dev_by_idx = {
        node.idx: node.score_val
        for node in dag.nodes.values()
        if node.score_val is not None
    }
    rendered_any = False
    for pattern_id in requested_ids:
        pattern = registry.get_pattern(pattern_id)
        if pattern is None:
            print(f"ERROR: Pattern '{pattern_id}' not found.", file=sys.stderr)
            continue

        complete_history = dag.get_arm_pull_history(pattern_id)
        if not complete_history:
            continue
        shown_history = complete_history
        history_label = "complete"
        if args.last_k is not None:
            shown_history = complete_history[-args.last_k:]
            history_label = f"latest {len(shown_history)} of {len(complete_history)}"

        rendered_any = True
        activity = registry.compute_score_for_pattern(
            pattern_id,
            _get_current_iter(),
            **_get_score_params(),
        )
        scenarios = registry.get_scenarios_for_pattern(pattern_id)
        print(f"Pattern: {pattern_id}")
        print(f"Label: {pattern.label}")
        print(f"Current activity: {activity:.4f}")
        print(f"Current scenarios: {', '.join(sorted(scenarios)) or '(none)'}")
        print(f"Pull history ({history_label}):")
        for record in shown_history:
            print()
            print("\n".join(render_arm_pull_record(record, dev_by_idx)))
        print()

    if not rendered_any:
        print("No arm pull history recorded yet.")


def cmd_rate(args: argparse.Namespace) -> None:
    """Record the agent's learning-progress score for a pattern.

    All four axes are required. The effective phi is
    ``(severity + fixability + breadth + (1 - side_effect)) / 4``.
    """
    axes = (args.severity, args.fixability, args.breadth, args.side_effect)
    if any(value is None for value in axes):
        print(
            "ERROR: pattern rate requires --severity, --fixability, "
            "--breadth, and --side-effect.",
            file=sys.stderr,
        )
        sys.exit(2)
    if any(not 0.0 <= value <= 1.0 for value in axes):
        print(
            "ERROR: severity, fixability, breadth, and side-effect must all "
            "be in [0,1].",
            file=sys.stderr,
        )
        sys.exit(2)

    registry = _load_registry()
    current_iter = _get_current_iter()
    try:
        registry.record_agent_score(
            pattern_id=args.pattern_id,
            iteration=current_iter,
            severity=args.severity,
            fixability=args.fixability,
            breadth=args.breadth,
            side_effect=args.side_effect,
            rationale=args.rationale or "",
        )
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    registry.save()
    phi = (
        args.severity
        + args.fixability
        + args.breadth
        + (1.0 - args.side_effect)
    ) / 4.0
    print(f"Rated: pattern={args.pattern_id}, phi={phi:.3f} (iter={current_iter})")


def cmd_decide(args: argparse.Namespace) -> None:
    """Record the agent's pull-vs-draw arm-creation decision.

    ``--action pull`` means work on an existing failure pattern (arm) this
    iteration; ``--action draw`` means explore the unseen scenario pool for
    new failure types. The proposer reads this file to override the sampler's
    arm-creation step.
    """
    import json

    registry_path = os.environ.get("PATTERN_REGISTRY_PATH")
    if not registry_path:
        print("ERROR: PATTERN_REGISTRY_PATH environment variable not set.", file=sys.stderr)
        sys.exit(1)
    current_iter = _get_current_iter()
    artifact_dir = os.environ.get("AUTOSADDLER_ITERATION_ARTIFACT_DIR")
    candidate_idx_value = os.environ.get("CURRENT_CANDIDATE_IDX")
    if not artifact_dir or not candidate_idx_value:
        print(
            "ERROR: pattern decide requires "
            "AUTOSADDLER_ITERATION_ARTIFACT_DIR and CURRENT_CANDIDATE_IDX.",
            file=sys.stderr,
        )
        sys.exit(1)
    candidate_idx = int(candidate_idx_value)
    out_path = iteration_artifact_path(
        artifact_dir,
        current_iter,
        candidate_idx,
        "arm_decision",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        payload = {
            "iteration": current_iter,
            "candidate_idx": candidate_idx,
            "action": args.action,
            "rationale": args.rationale or "",
        }
        json.dump(payload, f, indent=2)
    print(f"Decision recorded: action={args.action} (iter={current_iter})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pattern",
        description="Manage the PatternRegistry for failure pattern tracking.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    subparsers.add_parser(
        "list",
        help="List all patterns with activity, observations, and scenarios",
    )

    # show
    show_parser = subparsers.add_parser("show", help="Show pattern details")
    show_parser.add_argument("pattern_id", help="Pattern ID to show")

    # register
    reg_parser = subparsers.add_parser("register", help="Register a new pattern")
    reg_parser.add_argument("--label", required=True, help="Symptom-level label for the pattern")

    # tag
    tag_parser = subparsers.add_parser("tag", help="Tag a tuple with pattern(s)")
    tag_parser.add_argument("--pattern-id", required=True, action="append", help="Pattern ID(s) to tag (repeatable)")
    tag_parser.add_argument("--harness", required=True, type=int, help="Harness (DAG node) index")
    tag_parser.add_argument("--trace", required=True, help="Trace directory path")
    tag_parser.add_argument("--scenario", required=True, help="Scenario ID")
    tag_parser.add_argument("--root-cause", default="", help="Root cause description (evidence)")

    # score
    score_parser = subparsers.add_parser("score", help="Show top patterns by score")
    score_parser.add_argument("--top-k", type=int, default=10, help="Number of top patterns to show")

    # scenarios
    scn_parser = subparsers.add_parser("scenarios", help="List scenarios for a pattern")
    scn_parser.add_argument("--pattern-id", required=True, help="Pattern ID")

    # history
    history_parser = subparsers.add_parser(
        "history",
        help="Show exact-arm pull histories",
    )
    history_parser.add_argument(
        "--pattern-id",
        action="append",
        nargs="+",
        help="Pattern ID(s) to show; repeatable (default: all pulled arms)",
    )
    history_parser.add_argument(
        "--last-k",
        type=int,
        default=None,
        help="Show only the K most recent pulls for each selected arm",
    )

    # rate (agent learning-progress score)
    rate_parser = subparsers.add_parser("rate", help="Record the agent's learning-progress score for a pattern")
    rate_parser.add_argument("--pattern-id", required=True, help="Pattern ID to rate")
    rate_parser.add_argument("--severity", type=float, required=True, help="How badly the pattern currently fails [0,1]")
    rate_parser.add_argument("--fixability", type=float, required=True, help="Reachable by a harness patch [0,1]")
    rate_parser.add_argument("--breadth", type=float, required=True, help="How widely a fix transfers to held-out tasks [0,1]")
    rate_parser.add_argument("--side-effect", type=float, required=True, dest="side_effect", help="Risk of regressing other patterns [0,1]")
    rate_parser.add_argument("--rationale", default="", help="Short justification (logged for calibration)")

    # decide (agent pull-vs-draw arm-creation decision)
    dec_parser = subparsers.add_parser("decide", help="Record the pull-vs-draw arm-creation decision")
    dec_parser.add_argument("--action", required=True, choices=["pull", "draw"], help="'pull' an existing failure pattern, or 'draw' from the unseen scenario pool")
    dec_parser.add_argument("--rationale", default="", help="Short justification (logged)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    _enforce_command_capability(args.command)

    cmd_map = {
        "list": cmd_list,
        "show": cmd_show,
        "register": cmd_register,
        "tag": cmd_tag,
        "score": cmd_score,
        "scenarios": cmd_scenarios,
        "history": cmd_history,
        "rate": cmd_rate,
        "decide": cmd_decide,
    }

    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
