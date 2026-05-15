#!/usr/bin/env python3
"""evo-dag CLI tool: query and update the EvolutionDAG.

Standalone script deployed into the session root for Agent use.
Reads ``EVOLUTION_DAG_PATH`` env var to locate ``evolution_dag.json``.
"""

from __future__ import annotations

import os
import sys
import textwrap


def _load_dag():
    """Load the EvolutionDAG from the JSON path."""
    dag_path = os.environ.get("EVOLUTION_DAG_PATH")
    if not dag_path:
        print("ERROR: EVOLUTION_DAG_PATH environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Import from the installed package
    from autosaddler.v1.proposer.autosaddler.dag import EvolutionDAG

    session_root = os.path.dirname(dag_path)
    dag = EvolutionDAG(session_root)
    dag.load()
    return dag


def _format_score(score, precision=4):
    """Format a score value for display."""
    if score is None:
        return "n/a"
    return f"{score:.{precision}f}"


def _node_label(idx):
    """Get display label for a node index."""
    return "seed" if idx == 0 else f"C{idx}"


# ---------------------------------------------------------------------------
# Show commands
# ---------------------------------------------------------------------------


def cmd_summary(_args):
    """Show DAG-wide summary."""
    dag = _load_dag()
    summary = dag.get_summary()

    print(f"Iteration: {summary['total_iterations']} | Nodes: {summary['num_nodes']}")
    if summary["best_val_idx"] is not None:
        print(f"Best dev: {_node_label(summary['best_val_idx'])} ({_format_score(summary['best_val_score'])})")
    if summary["current_base_idx"] is not None:
        print(f"Current base parent: {_node_label(summary['current_base_idx'])}")

    if summary["edges"]:
        print("\nDAG edges:")
        for e in summary["edges"]:
            line = f"  {e['parent']} → {e['child']} ({e['type']}"
            if "delta" in e and e["delta"] is not None:
                line += f", {e['delta']:+.2f}"
            line += ")"
            if e.get("regression"):
                line += " ← regression"
            print(line)


def cmd_show_node(args):
    """Show details for a specific node."""
    if not args:
        print("Usage: evo-dag show node <idx>", file=sys.stderr)
        sys.exit(1)
    try:
        idx = int(args[0])
    except ValueError:
        print(f"ERROR: '{args[0]}' is not a valid node index (expected integer).", file=sys.stderr)
        sys.exit(1)
    dag = _load_dag()
    if idx not in dag.nodes:
        print(f"ERROR: Node {idx} not found. Available: {sorted(dag.nodes.keys())}", file=sys.stderr)
        sys.exit(1)
    node = dag.get_node(idx)
    edges = dag.get_edges_for_node(idx)

    print(f"{_node_label(idx)} (iteration {node.iteration})")
    print(f"  Created: {node.created_at}")
    print(f"  Worktree: {node.worktree_path}")
    if node.commit_hash:
        print(f"  Commit: {node.commit_hash[:12]}")

    print("\n  Scores:")
    print(f"    Train before: {_format_score(node.score_train_before)}")
    print(f"    Train after:  {_format_score(node.score_train_after)}")
    print(f"    Dev:          {_format_score(node.score_val)} (evaluated: {node.val_evaluated})")

    if node.accepted is not None:
        print(f"  Accepted by engine: {node.accepted}")

    if edges:
        print("\n  Parents:")
        for e in edges:
            parent_label = _node_label(e.parent_idx)
            line = f"    {parent_label} ({e.edge_type})"
            if e.edge_type == "base" and e.score_before is not None:
                line += (
                    f" — before: {_format_score(e.score_before)}, "
                    f"after: {_format_score(e.score_after)}, "
                    f"delta: {_format_score(e.score_delta)}"
                )
            print(line)

    if node.mini_batch_ids:
        print(f"\n  Mini-batch ({len(node.mini_batch_ids)} scenarios):")
        for sid in node.mini_batch_ids:
            print(f"    - {sid}")

    if node.selection_decision:
        sd = node.selection_decision
        parents = ', '.join(_node_label(p) for p in sd.parent_candidates)
        print("\n  Selection Decision:")
        print(f"    Parents: {parents}")
        print(f"    Reasoning: {sd.reasoning}")

    if node.patch_intent:
        pi = node.patch_intent
        print("\n  Patch Intent:")
        print(f"    Targets: {', '.join(pi.target_scenarios)}")
        print(f"    Approach: {pi.approach}")
        print(f"    Files: {', '.join(pi.files_changed)}")
        print(f"    Summary: {pi.change_summary}")

    if node.patch_verdict:
        pv = node.patch_verdict
        print("\n  Patch Verdict:")
        print(f"    Good patch: {pv.is_good_patch}")
        print(f"    Effectiveness: {pv.effectiveness} | Safety: {pv.safety}")
        if pv.scenario_impacts:
            for si in pv.scenario_impacts:
                print(f"    {si.scenario_id}: {si.status_change} ({_format_score(si.score_before)}→{_format_score(si.score_after)})")

    if node.train_before_cycle_dir:
        print("\n  Output dirs:")
        print(f"    Before: {node.train_before_cycle_dir}")
        if node.train_after_cycle_dir:
            print(f"    After:  {node.train_after_cycle_dir}")

    if node.sdk_session_selection:
        ss = node.sdk_session_selection
        print("\n  SDK Session (Selection):")
        print(f"    Model: {ss.model} | Tool calls: {ss.tool_call_count} | Turns: {ss.turns}")
        print(f"    JSON: {ss.session_json_path}")

    if node.sdk_session_patch:
        sp = node.sdk_session_patch
        print("\n  SDK Session (Patch):")
        print(f"    Model: {sp.model} | Tool calls: {sp.tool_call_count} | Turns: {sp.turns}")
        print(f"    Tokens: in={sp.input_tokens}, out={sp.output_tokens}, cache={sp.cache_read_input_tokens}")
        print(f"    JSON: {sp.session_json_path}")

    if node.sdk_session_reflection:
        sr = node.sdk_session_reflection
        print("\n  SDK Session (Reflection):")
        print(f"    Model: {sr.model} | Tool calls: {sr.tool_call_count} | Turns: {sr.turns}")
        print(f"    JSON: {sr.session_json_path}")


def cmd_show_edge(args):
    """Show details for a specific edge."""
    if len(args) < 2:
        print("Usage: evo-dag show edge <parent_idx> <child_idx>", file=sys.stderr)
        sys.exit(1)
    try:
        parent_idx = int(args[0])
        child_idx = int(args[1])
    except ValueError:
        print(f"ERROR: Edge indices must be integers, got '{args[0]}' and '{args[1]}'.", file=sys.stderr)
        sys.exit(1)
    dag = _load_dag()
    key = f"{parent_idx}->{child_idx}"
    if key not in dag.edges:
        print(f"ERROR: Edge {parent_idx}→{child_idx} not found.", file=sys.stderr)
        available = sorted(dag.edges.keys())
        if available:
            print(f"Available edges: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)
    edge = dag.get_edge(parent_idx, child_idx)

    print(f"Edge: {_node_label(parent_idx)} → {_node_label(child_idx)} ({edge.edge_type})")

    if edge.score_before is not None:
        print(f"\n  Before: {_format_score(edge.score_before)}")
        print(f"  After:  {_format_score(edge.score_after)}")
        print(f"  Delta:  {_format_score(edge.score_delta)}")
        print(f"  Improved: {edge.improved}")

    if edge.scenarios_fixed:
        print(f"\n  Fixed ({len(edge.scenarios_fixed)}):")
        for s in edge.scenarios_fixed:
            print(f"    + {s}")
    if edge.scenarios_regressed:
        print(f"\n  Regressed ({len(edge.scenarios_regressed)}):")
        for s in edge.scenarios_regressed:
            print(f"    - {s}")
    if edge.scenarios_still_failing:
        print(f"\n  Still failing ({len(edge.scenarios_still_failing)}):")
        for s in edge.scenarios_still_failing:
            print(f"    = {s}")
    if edge.scenarios_still_passing:
        print(f"\n  Still passing ({len(edge.scenarios_still_passing)}):")
        for s in edge.scenarios_still_passing:
            print(f"    = {s}")

    if edge.files_changed:
        print(f"\n  Files changed ({len(edge.files_changed)}):")
        for f in edge.files_changed:
            print(f"    {f}")

    edge_diff = dag.get_edge_diff_text(edge)
    if edge_diff:
        print("\n  Code diff:")
        # Truncate very long diffs
        diff_lines = edge_diff.split("\n")
        if len(diff_lines) > 100:
            for line in diff_lines[:100]:
                print(f"    {line}")
            print(f"    ... ({len(diff_lines) - 100} more lines)")
        else:
            for line in diff_lines:
                print(f"    {line}")


def cmd_show_scenario(args):
    """Show scenario history and metadata."""
    dag = _load_dag()
    if not args:
        print("Usage: evo-dag show scenario <scenario_id>", file=sys.stderr)
        print("       scenario_id is a string identifier, NOT a numeric index.", file=sys.stderr)
        sys.exit(1)

    scenario_id = args[0]
    entry = dag.get_scenario(scenario_id)

    if entry is None:
        print(f"Scenario '{scenario_id}' not found in registry.")
        # List available scenarios to help the agent
        available = list(dag.scenario_registry.keys()) if hasattr(dag, "scenario_registry") else []
        if available:
            print(f"\nAvailable scenarios ({len(available)}):")
            for sid in sorted(available)[:20]:
                print(f"  - {sid}")
            if len(available) > 20:
                print(f"  ... and {len(available) - 20} more")
        else:
            print("No scenarios in registry yet. Run an evaluation first.")
        return

    print(f"Scenario: {entry.scenario_id}")
    print(f"Category: {entry.category}")

    if getattr(entry, "task_description", None):
        wrapped = textwrap.fill(
            entry.task_description, width=80,
            initial_indent="Task: ", subsequent_indent="      ",
        )
        print(f"\n{wrapped}")

    if entry.history:
        print(f"\nHistory ({len(entry.history)} evaluations):")
        for snap in entry.history:
            print(
                f"  Iter {snap.iteration} (C{snap.candidate_idx}): "
                f"{snap.status} ({snap.score})"
            )
            if snap.rationale:
                # Wrap long rationale
                wrapped = textwrap.fill(snap.rationale, width=80, initial_indent="    ", subsequent_indent="    ")
                print(wrapped)

    if entry.known_root_causes:
        print("\nKnown root causes:")
        for rc in entry.known_root_causes:
            print(f"  - {rc}")

    if entry.attempted_fixes:
        print("\nAttempted fixes:")
        for fix in entry.attempted_fixes:
            print(f"  Iter {fix.iteration} (C{fix.candidate_idx}): {fix.result}")
            print(f"    Approach: {fix.approach}")
            if getattr(fix, "failure_reason", None):
                wrapped = textwrap.fill(
                    fix.failure_reason, width=76,
                    initial_indent="    Why failed: ", subsequent_indent="                ",
                )
                print(wrapped)
            if getattr(fix, "prevention_or_next", None):
                wrapped = textwrap.fill(
                    fix.prevention_or_next, width=76,
                    initial_indent="    Next suggestion: ", subsequent_indent="                     ",
                )
                print(wrapped)

    if entry.sensitive_to_files:
        print("\nSensitive to files:")
        for f in entry.sensitive_to_files:
            print(f"  - {f}")


def cmd_show_lessons(_args):
    """Show accumulated lessons."""
    dag = _load_dag()
    lessons = dag.get_lessons()

    if lessons.good_patterns:
        print(f"Good patterns ({len(lessons.good_patterns)}):")
        for lp in lessons.good_patterns:
            print(f"  [iter {lp.source_iteration}] {lp.pattern}")
            print(f"    Evidence: {lp.evidence}")
    else:
        print("Good patterns: (none yet)")

    print()

    if lessons.bad_patterns:
        print(f"Bad patterns ({len(lessons.bad_patterns)}):")
        for lp in lessons.bad_patterns:
            print(f"  [iter {lp.source_iteration}] {lp.pattern}")
            print(f"    Evidence: {lp.evidence}")
    else:
        print("Bad patterns: (none yet)")


def cmd_show_current_batch(_args):
    """Show current mini-batch status."""
    dag = _load_dag()
    batch = dag.get_current_batch()

    if batch.get("status") == "no nodes":
        print("No nodes in DAG yet.")
        return

    print(f"Iteration: {batch['iteration']}")
    print(f"Candidate: C{batch['candidate_idx']}")
    print(f"Base parent: {_node_label(batch['base_parent_idx']) if batch['base_parent_idx'] is not None else 'seed'}")
    print(f"Worktree: {batch['worktree_path']}")

    if batch.get("score_train_before") is not None:
        print(f"\nInitial score: {_format_score(batch['score_train_before'])}")
    if batch.get("score_train_after") is not None:
        print(f"Re-evaluation score:  {_format_score(batch['score_train_after'])}")

    if batch.get("train_before_cycle_dir"):
        print(f"\nInitial eval output dir: {batch['train_before_cycle_dir']}")
    if batch.get("train_after_cycle_dir"):
        print(f"Re-evaluation output dir:  {batch['train_after_cycle_dir']}")

    if batch.get("mini_batch_ids"):
        print(f"\nMini-batch scenarios ({len(batch['mini_batch_ids'])}):")
        for sid in batch["mini_batch_ids"]:
            task_desc = ""
            entry = dag.scenario_registry.get(sid)
            if entry and getattr(entry, "task_description", None):
                desc = entry.task_description
                if len(desc) > 120:
                    desc = desc[:117] + "..."
                task_desc = f" — {desc}"
            print(f"  - {sid}{task_desc}")


def cmd_show_lineage(_args):
    """Show DAG lineage visualization."""
    dag = _load_dag()
    lineage = dag.get_lineage()

    if not lineage["nodes"]:
        print("No nodes in DAG yet.")
        return

    # Build adjacency from actual edges
    children: dict[str, list[tuple[str, str]]] = {}  # parent_label → [(child_label, type)]
    node_labels = {n["idx"]: n["label"] for n in lineage["nodes"]}
    roots: set[str] = set(node_labels.values())

    for e in lineage["edges"]:
        parent = e["parent"]
        child = e["child"]
        children.setdefault(parent, []).append((child, e["type"]))
        roots.discard(child)

    # Follow chains: walk base edges greedily, print branches on separate lines
    visited: set[str] = set()
    branch_lines: list[str] = []
    cp_lines: list[str] = []

    def _follow_chain(start: str) -> str:
        """Follow single-child base edges into one line; branch when >1 child."""
        parts = [start]
        cur = start
        visited.add(cur)
        while True:
            kids = children.get(cur, [])
            base_kids = [(c, t) for c, t in kids if t == "base" and c not in visited]
            cp_kids = [(c, t) for c, t in kids if t == "cherry_pick"]
            # Record cherry_pick edges
            for c, _t in cp_kids:
                cp_lines.append(f"  {cur} ──cp──→ {c}")
            if len(base_kids) == 1:
                child_label = base_kids[0][0]
                parts.append(f" ──b──→ {child_label}")
                visited.add(child_label)
                cur = child_label
            else:
                # 0 or 2+ children → stop this chain, recurse for branches
                for child_label, _t in sorted(base_kids, key=lambda x: x[0]):
                    if child_label not in visited:
                        chain = _follow_chain(child_label)
                        branch_lines.append(f"  {cur} ──b──→ {chain}")
                break
        return "".join(parts)

    for root in sorted(roots):
        if root not in visited:
            print(_follow_chain(root))

    for line in branch_lines:
        print(line)
    for line in cp_lines:
        print(line)

    print()
    print("b = base, cp = cherry_pick")


# ---------------------------------------------------------------------------
# Show history (unified view)
# ---------------------------------------------------------------------------


def _print_filtered_diff(code_diff):
    """Print code diff, filtering binary noise and showing clean file headers."""
    lines = code_diff.split("\n")
    in_skip_section = False
    current_file = None
    has_output = False

    for line in lines:
        if line.startswith("Binary files"):
            continue

        if line.startswith("diff "):
            if "proposer_reasoning.md" in line:
                in_skip_section = True
                continue
            in_skip_section = False
            # Extract relative path from the second file path
            parts = line.split()
            if parts:
                path = parts[-1]
                for prefix in ["/are/", "/src/", "/config/"]:
                    idx = path.find(prefix)
                    if idx >= 0:
                        path = path[idx + 1:]
                        break
                if not has_output:
                    print("\nDiff:")
                    has_output = True
                elif current_file:
                    print()
                current_file = path
                print(f"  --- {path} ---")
            continue

        if in_skip_section:
            continue

        if line.startswith("---") or line.startswith("+++"):
            continue

        if line.startswith("@@") or line.startswith("+") or line.startswith("-") or line.startswith(" "):
            if has_output:
                print(f"  {line}")


def _print_scenario_history_entry(dag, node, si):
    """Print per-scenario detail in show history output."""
    before_str = "PASS" if si.score_before >= 0.5 else "FAIL"
    after_str = "PASS" if si.score_after >= 0.5 else "FAIL"

    targeted = ""
    if node.patch_intent and si.scenario_id in node.patch_intent.target_scenarios:
        targeted = " (targeted)"

    marker = ""
    if si.status_change == "fixed":
        marker = " ★ FIXED"
    elif si.status_change == "regressed":
        marker = " ✗ REGRESSED"

    print(f"\n  ── {si.scenario_id}: {before_str}→{after_str}{targeted}{marker} ──")

    # Task description from scenario registry
    entry = dag.scenario_registry.get(si.scenario_id)
    if entry and getattr(entry, "task_description", None):
        wrapped = textwrap.fill(
            entry.task_description, width=76,
            initial_indent="  Task: ", subsequent_indent="        ",
        )
        print(wrapped)

    # Find matching reflection
    refl = None
    if node.patch_verdict:
        for r in node.patch_verdict.reflections:
            if r.scenario_id == si.scenario_id:
                refl = r
                break

    if refl:
        # Show root_cause if available (separate from explanation)
        root_cause = getattr(refl, "root_cause", None) or None
        if root_cause:
            wrapped = textwrap.fill(
                root_cause, width=76,
                initial_indent="  Root cause: ", subsequent_indent="              ",
            )
            print(wrapped)

        if si.status_change == "fixed":
            wrapped = textwrap.fill(
                refl.explanation, width=76,
                initial_indent="  How fixed: ", subsequent_indent="             ",
            )
            print(wrapped)
            if refl.prevention_or_next:
                wrapped = textwrap.fill(
                    refl.prevention_or_next, width=76,
                    initial_indent="  Lesson: ", subsequent_indent="          ",
                )
                print(wrapped)
        elif si.status_change == "regressed":
            wrapped = textwrap.fill(
                refl.explanation, width=76,
                initial_indent="  What broke: ", subsequent_indent="              ",
            )
            print(wrapped)
            if refl.prevention_or_next:
                wrapped = textwrap.fill(
                    refl.prevention_or_next, width=76,
                    initial_indent="  Prevention: ", subsequent_indent="              ",
                )
                print(wrapped)
        elif si.status_change == "still_failing":
            wrapped = textwrap.fill(
                refl.explanation, width=76,
                initial_indent="  Why patch failed: ", subsequent_indent="                    ",
            )
            print(wrapped)
            if refl.prevention_or_next:
                wrapped = textwrap.fill(
                    refl.prevention_or_next, width=76,
                    initial_indent="  Next suggestion: ", subsequent_indent="                   ",
                )
                print(wrapped)
    elif si.status_change not in ("still_passing",):
        # No reflection — show rationale as fallback
        if si.rationale_after:
            rat = si.rationale_after
            if len(rat) > 300:
                rat = rat[:300] + "..."
            wrapped = textwrap.fill(
                rat, width=76,
                initial_indent="  Rationale: ", subsequent_indent="             ",
            )
            print(wrapped)


def cmd_show_history(_args):
    """Show comprehensive patch history with diffs, reflections, and lessons."""
    dag = _load_dag()

    if not dag.nodes:
        print("No history yet.")
        return

    nodes_sorted = sorted(
        [n for n in dag.nodes.values() if n.iteration > 0],
        key=lambda n: n.iteration,
    )

    if not nodes_sorted:
        print("No iterations yet (only seed node).")
        return

    for node in nodes_sorted:
        parent_label = _node_label(node.base_parent_idx) if node.base_parent_idx is not None else "seed"

        verdict_str = ""
        if node.patch_verdict:
            if node.patch_verdict.is_good_patch:
                verdict_str = " ✓ good patch"
            elif not node.patch_verdict.safety:
                verdict_str = " ✗ regression"
            else:
                verdict_str = " ✗ ineffective"

        accepted_str = ""
        if node.accepted is True:
            accepted_str = " [accepted]"
        elif node.accepted is False:
            accepted_str = " [rejected]"

        print(f"{'=' * 60}")
        print(f"Iteration {node.iteration} (C{node.idx}, parent={parent_label}){verdict_str}{accepted_str}")
        print(f"{'=' * 60}")

        if node.score_train_before is not None and node.score_train_after is not None:
            delta = node.score_train_after - node.score_train_before
            print(f"Score: {_format_score(node.score_train_before)} → {_format_score(node.score_train_after)} ({delta:+.4f})")

        if node.patch_intent:
            pi = node.patch_intent
            if getattr(pi, "diagnosis", None):
                wrapped = textwrap.fill(
                    pi.diagnosis, width=76,
                    initial_indent="Diagnosis: ", subsequent_indent="           ",
                )
                print(f"\n{wrapped}")
            print(f"\nPatch approach: {pi.approach}")
            if pi.files_changed:
                print(f"Files changed: {', '.join(pi.files_changed)}")

        # Filtered code diff
        edge_key = f"{node.base_parent_idx}->{node.idx}" if node.base_parent_idx is not None else None
        edge = dag.edges.get(edge_key) if edge_key else None
        if edge:
            edge_diff = dag.get_edge_diff_text(edge)
            if edge_diff:
                _print_filtered_diff(edge_diff)

        # Per-scenario details
        if node.patch_verdict and node.patch_verdict.scenario_impacts:
            interesting = [
                si for si in node.patch_verdict.scenario_impacts
                if si.status_change != "still_passing"
            ]
            still_passing = [
                si for si in node.patch_verdict.scenario_impacts
                if si.status_change == "still_passing"
            ]

            if interesting:
                print("\n--- Scenario Results ---")
                for si in interesting:
                    _print_scenario_history_entry(dag, node, si)

            if still_passing:
                # Separate still_passing with meaningful analysis from trivial ones
                analyzed_sp = []
                trivial_sp = []
                for si in still_passing:
                    refl = None
                    if node.patch_verdict:
                        for r in node.patch_verdict.reflections:
                            if r.scenario_id == si.scenario_id:
                                refl = r
                                break
                    trivial_msg = "No interference observed. Scenario continues to pass."
                    if refl and refl.explanation and refl.explanation != trivial_msg:
                        analyzed_sp.append((si, refl))
                    else:
                        trivial_sp.append(si)

                for si, refl in analyzed_sp:
                    print(f"\n  ── {si.scenario_id}: PASS→PASS (no interference) ──")
                    wrapped = textwrap.fill(
                        refl.explanation, width=76,
                        initial_indent="  Why safe: ", subsequent_indent="            ",
                    )
                    print(wrapped)

                if trivial_sp:
                    sp_ids = ", ".join(si.scenario_id for si in trivial_sp)
                    print(f"\n  PASS→PASS (no interference): {sp_ids}")

        print()


# ---------------------------------------------------------------------------
# Update commands
# ---------------------------------------------------------------------------


def cmd_update_selection(args):
    """Record candidate selection decision (Session 0)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="evo-dag update-selection",
        description="Record which candidate(s) were selected to build on.",
    )
    parser.add_argument(
        "--parent-candidates", required=True,
        help="Comma-separated candidate indices used (e.g. '2' or '2,3')",
    )
    parser.add_argument(
        "--reasoning", required=True,
        help="Why these candidates were selected",
    )
    parser.add_argument(
        "--node", type=int, default=None,
        help="Target node index (default: latest node)",
    )
    parsed = parser.parse_args(args)

    dag = _load_dag()
    from autosaddler.v1.proposer.autosaddler.models import SelectionDecision

    parent_candidates = [
        int(x.strip()) for x in parsed.parent_candidates.split(",") if x.strip()
    ]

    decision = SelectionDecision(
        parent_candidates=parent_candidates,
        reasoning=parsed.reasoning,
    )

    target_idx = parsed.node
    if target_idx is None:
        target_idx = max(dag.nodes.values(), key=lambda n: n.iteration).idx
    elif target_idx not in dag.nodes:
        print(f"ERROR: Node {target_idx} not found. Available: {sorted(dag.nodes.keys())}", file=sys.stderr)
        sys.exit(1)
    dag.update_selection_decision(target_idx, decision)
    dag.save()
    parents_str = ", ".join(_node_label(p) for p in parent_candidates)
    print(f"Selection recorded for {_node_label(target_idx)}: parents={parents_str}.")


def cmd_update_intent(args):
    """Record patch intent (Session 1)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="evo-dag update-intent",
        description="Record patch intent for the current iteration.",
        epilog=textwrap.dedent("""\
            Example:
              evo-dag update-intent \\
                --target-scenarios "scenario_001,scenario_002" \\
                --diagnosis "Agent misinterprets 'let me know' as 'stop and wait'. Root cause is conditional instruction parsing." \\
                --approach "Fix email forwarding by correcting tool docstring" \\
                --files-changed "are/simulation/tools/email.py" \\
                --change-summary "Updated forward_email docstring to clarify usage"
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target-scenarios", required=True, help="Comma-separated scenario IDs targeted by this patch")
    parser.add_argument("--diagnosis", default=None, help="Root cause diagnosis: why the target scenarios are failing")
    parser.add_argument("--approach", required=True, help="Brief description of the patch approach")
    parser.add_argument("--files-changed", required=True, help="Comma-separated file paths that were modified")
    parser.add_argument("--change-summary", required=True, help="What was changed and why")
    parser.add_argument("--node", type=int, default=None, help="Target node index (default: latest node)")
    parsed = parser.parse_args(args)

    dag = _load_dag()
    from autosaddler.v1.proposer.autosaddler.models import PatchIntent

    intent = PatchIntent(
        target_scenarios=[s.strip() for s in parsed.target_scenarios.split(",")],
        approach=parsed.approach,
        files_changed=[f.strip() for f in parsed.files_changed.split(",")],
        change_summary=parsed.change_summary,
        diagnosis=parsed.diagnosis,
    )

    target_idx = parsed.node
    if target_idx is None:
        target_idx = max(dag.nodes.values(), key=lambda n: n.iteration).idx
    elif target_idx not in dag.nodes:
        print(f"ERROR: Node {target_idx} not found. Available: {sorted(dag.nodes.keys())}", file=sys.stderr)
        sys.exit(1)
    dag.update_patch_intent(target_idx, intent)
    dag.save()
    print(f"Patch intent recorded for {_node_label(target_idx)}.")


def cmd_update_reflection(args):
    """Record per-scenario reflection (Session 2)."""
    import argparse

    parser = argparse.ArgumentParser(prog="evo-dag update-reflection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", help="Scenario ID (for individual reflection)")
    group.add_argument("--batch-still-passing", help="Comma-separated scenario IDs to batch-record as still_passing")
    parser.add_argument("--status", choices=["fixed", "regressed", "still_failing", "still_passing"])
    parser.add_argument("--explanation", help="Why this result occurred (for still_failing: why the patch did not work)")
    parser.add_argument("--root-cause", default=None, help="Root cause of the original failure (separate from explanation)")
    parser.add_argument("--prevention-or-next", default=None, help="Prevention or next approach")
    parser.add_argument("--generalization-note", default=None, help="Dev set accuracy analysis: why accuracy changed compared to prior candidates, based on this patch")
    parser.add_argument("--node", type=int, default=None, help="Target node index (default: latest node)")
    parsed = parser.parse_args(args)

    dag = _load_dag()
    from autosaddler.v1.proposer.autosaddler.models import ReflectionEntry

    target_idx = parsed.node
    if target_idx is None:
        target_idx = max(dag.nodes.values(), key=lambda n: n.iteration).idx
    elif target_idx not in dag.nodes:
        print(f"ERROR: Node {target_idx} not found. Available: {sorted(dag.nodes.keys())}", file=sys.stderr)
        sys.exit(1)

    if parsed.batch_still_passing:
        # Batch mode for still_passing scenarios
        ids = [s.strip() for s in parsed.batch_still_passing.split(",") if s.strip()]
        for sid in ids:
            reflection = ReflectionEntry(
                scenario_id=sid,
                status_change="still_passing",
                explanation="No interference observed. Scenario continues to pass.",
                prevention_or_next=None,
            )
            dag.add_reflection(target_idx, reflection)
        dag.save()
        print(f"Batch reflection recorded for {len(ids)} still_passing scenarios on {_node_label(target_idx)}.")
    else:
        # Individual mode
        if not parsed.status:
            print("ERROR: --status is required for individual reflection.", file=sys.stderr)
            sys.exit(1)
        if not parsed.explanation:
            print("ERROR: --explanation is required for individual reflection.", file=sys.stderr)
            sys.exit(1)
        reflection = ReflectionEntry(
            scenario_id=parsed.scenario,
            status_change=parsed.status,
            explanation=parsed.explanation,
            root_cause=parsed.root_cause,
            prevention_or_next=parsed.prevention_or_next,
            generalization_note=parsed.generalization_note,
        )
        dag.add_reflection(target_idx, reflection)
        dag.save()
        print(f"Reflection recorded for {parsed.scenario} ({parsed.status}) on {_node_label(target_idx)}.")


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

COMMANDS = {
    "summary": (cmd_summary, "Show DAG-wide summary"),
    "show": None,  # sub-dispatcher
    "update-selection": (cmd_update_selection, "Record candidate selection"),
    "update-intent": (cmd_update_intent, "Record patch intent"),
    "update-reflection": (cmd_update_reflection, "Record scenario reflection"),
}

SHOW_SUBCOMMANDS = {
    "node": (cmd_show_node, "Show node details", "<idx>"),
    "edge": (cmd_show_edge, "Show edge details", "<parent_idx> <child_idx>"),
    "scenario": (cmd_show_scenario, "Show scenario history", "<scenario_id>"),
    "current-batch": (cmd_show_current_batch, "Show current mini-batch", ""),
    "history": (cmd_show_history, "Full patch history with diffs and reflections", ""),
    "lineage": (cmd_show_lineage, "Show DAG lineage visualization", ""),
    "lessons": (cmd_show_lessons, "Show accumulated lessons", ""),
}


def print_usage():
    """Print usage help."""
    print("Usage: evo-dag <command> [args...]")
    print()
    print("Query commands:")
    print("  summary                          Show DAG-wide summary")
    print("  show node <idx>                  Show node details")
    print("  show edge <parent> <child>       Show edge initial/re-evaluation details")
    print("  show scenario <id>               Show scenario history")
    print("  show current-batch               Show current mini-batch status")
    print("  show history                     Full patch history with diffs and reflections")
    print("  show lineage                     Show DAG lineage visualization")
    print("  show lessons                     Show accumulated lessons")
    print()
    print("Update commands:")
    print("  update-selection                 Record candidate selection (--help for args)")
    print("  update-intent                    Record patch intent (--help for args)")
    print("  update-reflection                Record scenario reflection (--help for args)")
    print()
    print("All update commands accept --node <idx> to target a specific node (default: latest).")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd in ("--help", "-h", "help"):
        print_usage()
        sys.exit(0)

    if cmd == "show":
        if len(sys.argv) < 3:
            print("Usage: evo-dag show <subcommand> [args...]")
            print("Subcommands:", ", ".join(SHOW_SUBCOMMANDS.keys()))
            sys.exit(1)
        subcmd = sys.argv[2]
        if subcmd not in SHOW_SUBCOMMANDS:
            print(f"Unknown show subcommand: {subcmd}")
            print("Available:", ", ".join(SHOW_SUBCOMMANDS.keys()))
            sys.exit(1)
        handler, _desc, _usage = SHOW_SUBCOMMANDS[subcmd]
        handler(sys.argv[3:])
    elif cmd == "update-selection":
        cmd_update_selection(sys.argv[2:])
    elif cmd == "update-intent":
        cmd_update_intent(sys.argv[2:])
    elif cmd == "update-reflection":
        cmd_update_reflection(sys.argv[2:])
    elif cmd == "summary":
        cmd_summary(sys.argv[2:])
    else:
        print(f"Unknown command: {cmd}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
