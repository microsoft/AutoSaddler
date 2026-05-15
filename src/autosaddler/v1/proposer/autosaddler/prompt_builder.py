"""CLAUDE.md, session prompt, and skill installation into worktrees.

Replaces the v1 ``skill_builder.py`` with a clear separation between:
- CLAUDE.md: Always-on context (env, benchmark, pipeline, CLI, skills intro)
- Session prompts: Per-session task descriptions with dynamic context
- Skills: Static methodology files (no benchmark-specific content)

Functions:
- ``build_claude_md()``: Renders the CLAUDE.md template with iteration context.
- ``build_session0_prompt()``: Renders Session 0 (candidate selection) prompt.
- ``build_session1_prompt()``: Renders Session 1 (diagnose + patch) prompt.
- ``build_session2_prompt()``: Renders Session 2 (reflection) prompt.
- ``install_prompts_and_skills()``: Installs CLAUDE.md, session prompts, and skills
  into the worktree for Claude Code discovery.
- ``install_evo_dag_cli()``: Deploys the evo-dag CLI wrapper.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autosaddler.v1.proposer.autosaddler.dag import EvolutionDAG
    from autosaddler.v1.proposer.autosaddler.models import EvolutionNode, ScenarioImpact

logger = logging.getLogger(__name__)

# Directory containing the templates and static skills
_MODULE_DIR = Path(__file__).parent
_CLAUDE_MD_TEMPLATE = _MODULE_DIR / "CLAUDE.md"
_SESSION_PROMPTS_DIR = _MODULE_DIR / "session_prompts"
_SKILLS_DIR = _MODULE_DIR / "skills"

# All skills are installed regardless of phase; the session prompt
# tells the agent which skill to use based on the current phase.
_ALL_SKILLS = ["diagnose", "capability-patch", "steering-patch", "patch-verification", "history-analysis"]


def build_claude_md() -> str:
    """Read the CLAUDE.md template.

    CLAUDE.md is session-independent and contains no template variables.
    It is always loaded by Claude Code at startup and provides:
    - Optimization pipeline overview
    - evo-dag CLI reference
    - Skills introduction
    - Benchmark and agent framework info
    - Constraints
    """
    return _CLAUDE_MD_TEMPLATE.read_text(encoding="utf-8")


def build_session0_prompt(
    *,
    iteration: int,
    worktree_path: str,
    parent_worktree: str,
    base_parent_idx: int,
    session_root: str,
    dag: EvolutionDAG,
    phase: str = "capability",
) -> str:
    """Render Session 0 (candidate selection) prompt.

    Pre-computes candidate performance table, DAG topology, and accumulated
    lessons from the DAG so the agent can make an informed selection without
    needing to run multiple CLI queries.
    """
    template = (_SESSION_PROMPTS_DIR / "session0_candidate_selection.md").read_text(encoding="utf-8")

    # ── Candidate performance table (sorted by dev score descending) ──
    table_lines = [
        "| Candidate | Parent | Dev Score | Train \u0394 | Fixed | Regressed | Approach |",
        "|-----------|--------|-----------|---------|-------|-----------|----------|",
    ]
    prior_nodes = sorted(
        [n for n in dag.nodes.values() if n.iteration < iteration],
        key=lambda n: (n.score_val if n.score_val is not None else -1),
        reverse=True,
    )
    for node in prior_nodes:
        label = "C0 (seed)" if node.idx == 0 else f"C{node.idx}"
        # Parent: base parent + cherry-pick parents
        parent_parts = []
        if node.base_parent_idx is not None:
            parent_parts.append(f"C{node.base_parent_idx}")
        # Add cherry-pick parents from edges
        for edge in dag.get_edges_for_node(node.idx):
            if edge.edge_type == "cherry_pick" and edge.parent_idx != node.base_parent_idx:
                parent_parts.append(f"C{edge.parent_idx}(cp)")
        parent = ", ".join(parent_parts) if parent_parts else "-"
        val = f"{node.score_val:.4f}" if node.score_val is not None else "pending"

        if node.score_train_before is not None and node.score_train_after is not None:
            delta = node.score_train_after - node.score_train_before
            train_d = f"{node.score_train_before:.2f}\u2192{node.score_train_after:.2f} ({delta:+.2f})"
        else:
            train_d = "-"

        fixed_count = 0
        regression_count = 0
        if node.patch_verdict:
            fixed_count = sum(
                1 for si in node.patch_verdict.scenario_impacts
                if si.status_change == "fixed"
            )
            regression_count = sum(
                1 for si in node.patch_verdict.scenario_impacts
                if si.status_change == "regressed"
            )

        approach = "-"
        if node.patch_intent and node.patch_intent.approach:
            approach = node.patch_intent.approach
            if len(approach) > 60:
                approach = approach[:57] + "..."

        table_lines.append(
            f"| {label} | {parent} | {val} | {train_d} "
            f"| {fixed_count} | {regression_count} | {approach} |"
        )
    candidate_table = "\n".join(table_lines)

    # ── DAG topology ──
    summary = dag.get_summary()
    topo_lines: list[str] = []
    if summary.get("best_val_idx") is not None:
        best_label = "seed" if summary["best_val_idx"] == 0 else f"C{summary['best_val_idx']}"
        topo_lines.append(
            f"Best dev: {best_label} "
            f"({summary['best_val_score']:.4f})"
        )
        topo_lines.append("")
    if summary.get("edges"):
        topo_lines.append("Edges:")
        for e in summary["edges"]:
            line = f"  {e['parent']} \u2192 {e['child']} ({e['type']}"
            if e.get("delta") is not None:
                line += f", \u0394={e['delta']:+.2f}"
            line += ")"
            if e.get("regression"):
                line += " \u2190 regression"
            topo_lines.append(line)
    dag_topology = "\n".join(topo_lines) if topo_lines else "(seed only)"

    return template.format(
        iteration=iteration,
        phase=phase,
        worktree_path=worktree_path,
        parent_worktree=parent_worktree,
        base_parent_idx=base_parent_idx,
        session_root=session_root,
        candidate_table=candidate_table,
        dag_topology=dag_topology,
    )


def build_session1_prompt(
    *,
    iteration: int,
    candidate_idx: int,
    worktree_path: str,
    parent_worktree: str,
    base_parent_idx: int,
    mini_batch_ids: list[str],
    before_scores: dict[str, float],
    before_rationales: dict[str, str | None],
    before_output_dir: str,
    phase: str = "capability",
    cherry_pick_parents: list[tuple[int, str]] | None = None,
) -> str:
    """Render Session 1 (diagnose + patch) prompt."""
    template = (_SESSION_PROMPTS_DIR / "session1_diagnose_patch.md").read_text(encoding="utf-8")

    # Mini-batch listing
    mini_batch_lines = []
    for sid in mini_batch_ids:
        score = before_scores.get(sid, 0.0)
        status = "PASS" if score >= 0.5 else "FAIL"
        mini_batch_lines.append(f"  - `{sid}`: {status} ({score:.0f})")
    mini_batch_listing = "\n".join(mini_batch_lines) if mini_batch_lines else "  (none)"

    # Initial scores listing
    before_lines = []
    for sid in mini_batch_ids:
        score = before_scores.get(sid, 0.0)
        status = "PASS" if score >= 0.5 else "FAIL"
        line = f"  - `{sid}`: {status}"
        if score < 0.5:
            rationale = before_rationales.get(sid)
            if rationale:
                if len(rationale) > 200:
                    rationale = rationale[:200] + "..."
                line += f"\n    Rationale: {rationale}"
        before_lines.append(line)
    before_scores_listing = "\n".join(before_lines) if before_lines else "  (all passing)"

    # Pass rate
    scores_list = [before_scores.get(sid, 0.0) for sid in mini_batch_ids]
    before_pass_rate = f"{sum(scores_list) / len(scores_list):.4f}" if scores_list else "n/a"

    # Phase-specific patch types
    if phase == "capability":
        patch_types_section = (
            "**This is the CAPABILITY phase.** Strongly prefer patches that "
            "change executable code — add new tool methods, expose new "
            "parameters, fix tool implementations, or modify agent loop "
            "logic. When you add or modify tools, parameters, or code, "
            "also align the system prompt, tool docstrings, and hooks so "
            "the agent is aware of the changes. Prompt-only changes are discouraged "
            "in this phase; save those for the steering phase unless they "
            "are needed to accompany a code change.\n\n"
            "| Patch Type | Skill |\n"
            "|-----------|-------|\n"
            "| New tool / argument | `capability-patch` |\n"
            "| Implementation fix | `capability-patch` |\n"
            "| Infrastructure | `capability-patch` |\n"
        )
    else:
        patch_types_section = (
            "**This is the STEERING phase.** The agent's capabilities are "
            "already in place — now refine HOW it uses them. Focus on "
            "text-level changes: prompt rules, tool description "
            "corrections, and PreToolUse hook reminders. Adding new code "
            "or modifying tool implementations is discouraged unless "
            "necessary to support a steering fix.\n\n"
            "| Patch Type | Skill |\n"
            "|-----------|-------|\n"
            "| Tool description | `steering-patch` |\n"
            "| Prompt rule | `steering-patch` |\n"
            "| Hook | `steering-patch` |\n"
        )

    # Cherry-pick parents section
    if cherry_pick_parents:
        cp_lines = [f"- **Cherry-pick parent**: C{idx} (worktree: `{wt}`)" for idx, wt in cherry_pick_parents]
        cherry_pick_parents_section = "\n".join(cp_lines) + "\n"
    else:
        cherry_pick_parents_section = ""

    return template.format(
        iteration=iteration,
        candidate_idx=candidate_idx,
        worktree_path=worktree_path,
        parent_worktree=parent_worktree,
        base_parent_idx=base_parent_idx,
        num_scenarios=len(mini_batch_ids),
        mini_batch_listing=mini_batch_listing,
        before_pass_rate=before_pass_rate,
        before_scores_listing=before_scores_listing,
        before_output_dir=before_output_dir,
        phase=phase,
        patch_types_section=patch_types_section,
        cherry_pick_parents_section=cherry_pick_parents_section,
    )


def build_session2_prompt(
    node: EvolutionNode,
    scenario_impacts: list[ScenarioImpact],
    all_worktrees: dict[int, str] | None = None,
    dag: EvolutionDAG | None = None,
    phase: str = "unknown",
) -> str:
    """Render Session 2 (reflection) prompt with initial/re-evaluation results."""
    template = (_SESSION_PROMPTS_DIR / "session2_reflection.md").read_text(encoding="utf-8")

    # Results summary
    fixed = [si for si in scenario_impacts if si.status_change == "fixed"]
    regressed = [si for si in scenario_impacts if si.status_change == "regressed"]
    still_failing = [si for si in scenario_impacts if si.status_change == "still_failing"]
    still_passing = [si for si in scenario_impacts if si.status_change == "still_passing"]

    total = len(scenario_impacts)
    results_summary = (
        f"Total scenarios: {total}\n"
        f"  Fixed (FAIL→PASS): {len(fixed)}\n"
        f"  Regressed (PASS→FAIL): {len(regressed)}\n"
        f"  Still failing (FAIL→FAIL): {len(still_failing)}\n"
        f"  Still passing (PASS→PASS): {len(still_passing)}"
    )

    # Per-scenario details
    detail_lines = []
    for si in scenario_impacts:
        before_str = "PASS" if si.score_before >= 0.5 else "FAIL"
        after_str = "PASS" if si.score_after >= 0.5 else "FAIL"
        detail_lines.append(f"### {si.scenario_id}: {si.status_change} ({before_str} → {after_str})")
        detail_lines.append("")
        if si.rationale_before:
            detail_lines.append(f"**Before rationale:** {si.rationale_before}")
            detail_lines.append("")
        if si.rationale_after:
            detail_lines.append(f"**After rationale:** {si.rationale_after}")
            detail_lines.append("")
    per_scenario_details = "\n".join(detail_lines) if detail_lines else "(no scenarios)"

    # Dev score history and generalization section (conditional)
    dev_lines: list[str] = []
    has_dev_scores = False
    if dag is not None:
        scored_nodes = sorted(
            [n for n in dag.nodes.values() if n.score_val is not None],
            key=lambda n: n.iteration,
        )
        if scored_nodes:
            has_dev_scores = True
            for n in scored_nodes:
                label = "C0 (seed)" if n.idx == 0 else f"C{n.idx}"
                marker = " ← current" if n.idx == node.idx else ""
                dev_lines.append(f"  {label}: {n.score_val:.4f}{marker}")
        if node.score_val is None:
            dev_lines.append(
                f"\n  C{node.idx} (current): not evaluated"
                f" (patch was not accepted on the mini-batch)."
            )

    # Build generalization section only if dev scores exist
    if has_dev_scores:
        dev_table = "\n".join(dev_lines)
        generalization_section = (
            "## Development Set Accuracy History\n\n"
            f"{dev_table}"
        )
        generalization_workflow_step = (
            "### 6. Generalization analysis\n\n"
            f"This candidate (C{node.idx}) was evaluated on the development set.\n"
            "Compare dev scores across candidates in the history above and analyze:\n\n"
            f"1. **Compare dev scores**: Did this candidate's dev accuracy improve, stay flat, or\n"
            f"   drop compared to the best previous candidate? Compared to the immediate\n"
            f"   parent (C{node.base_parent_idx if node.base_parent_idx is not None else 'seed'})?\n\n"
            "2. **Attribute the change**: Reason about **why** dev accuracy changed\n"
            "   based on what the patch modified — e.g., a tool fix that addresses a\n"
            "   common pattern should help unseen scenarios; a narrow prompt rule may not.\n\n"
            "3. **Extract generalization lessons**: Record what you learn about which\n"
            "   types of patches generalize well and which don't. This guides future\n"
            "   iterations' patch strategy.\n\n"
            "4. **Record via `--generalization-note`**:\n"
            "```bash\n"
            "evo-dag update-reflection \\\n"
            f'  --node {node.idx} \\\n'
            '  --scenario "<id>" --status "fixed" \\\n'
            '  --root-cause "..." --explanation "..." \\\n'
            '  --generalization-note "Dev accuracy changed X→Y. Reason: ..."\n'
            "```\n\n"
            "If the dev score **dropped** despite mini-batch improvement, the patch\n"
            "likely overfits to the mini-batch. Record this explicitly as a bad\n"
            "pattern in `--prevention-or-next`."
        )
    else:
        generalization_section = ""
        generalization_workflow_step = ""

    # Read proposer_reasoning.md from worktree (Session 1 output)
    proposer_reasoning = "(no proposer reasoning available)"
    if node.worktree_path:
        reasoning_path = Path(node.worktree_path) / "proposer_reasoning.md"
        if reasoning_path.exists():
            try:
                proposer_reasoning = reasoning_path.read_text(encoding="utf-8")
            except Exception:
                pass

    return template.format(
        iteration=node.iteration,
        candidate_idx=node.idx,
        base_parent_idx=node.base_parent_idx if node.base_parent_idx is not None else "seed",
        worktree_path=node.worktree_path or "(not available)",
        parent_worktree=(
            all_worktrees.get(node.base_parent_idx, "(not available)")
            if all_worktrees and node.base_parent_idx is not None
            else "(not available)"
        ),
        phase=phase,
        before_output_dir=node.train_before_cycle_dir or "(not available)",
        results_summary=results_summary,
        per_scenario_details=per_scenario_details,
        generalization_section=generalization_section,
        generalization_workflow_step=generalization_workflow_step,
        train_after_cycle_dir=node.train_after_cycle_dir or "(not available)",
        proposer_reasoning=proposer_reasoning,
    )


def install_prompts_and_skills(
    worktree_path: str,
    built_claude_md: str,
    phase: str = "capability",
) -> None:
    """Install CLAUDE.md and skill files into the worktree.

    - ``CLAUDE.md``: Always-on context at worktree root
    - All skills: installed under ``.claude/skills/`` regardless of phase.
      The session prompt tells the agent which skill to use.
    """
    wt = Path(worktree_path)

    # Install CLAUDE.md at worktree root
    (wt / "CLAUDE.md").write_text(built_claude_md, encoding="utf-8")
    logger.info("Installed CLAUDE.md at %s (phase=%s)", wt / "CLAUDE.md", phase)

    # Install all skills
    skills_target = wt / ".claude" / "skills"
    if skills_target.exists():
        shutil.rmtree(skills_target)

    for skill_name in _ALL_SKILLS:
        src = _SKILLS_DIR / skill_name / "SKILL.md"
        if not src.exists():
            logger.warning("SKILL.md not found: %s", src)
            continue
        dst_dir = skills_target / skill_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "SKILL.md")
        logger.info("Installed SKILL.md: %s", dst_dir / "SKILL.md")


# ---------------------------------------------------------------------------
# Skill prefix for user prompt (explicit inline injection)
# ---------------------------------------------------------------------------

def _read_skill(name: str) -> str:
    """Read a SKILL.md file by skill name."""
    path = _SKILLS_DIR / name / "SKILL.md"
    if not path.exists():
        logger.warning("SKILL.md not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8")


def build_skill_prefix(session: int, phase: str = "capability") -> str:
    """Build a skill prefix to prepend to the user prompt.

    Reads CLAUDE.md and session-appropriate SKILL.md files, then combines
    them into a string that is prepended to the user prompt so Claude Code
    receives the instructions directly in context.

    This approach works reliably with copilot proxy (unlike
    --append-system-prompt which may be ignored).

    Parameters
    ----------
    session:
        Session number (0, 1, or 2).
    phase:
        Current optimization phase (``"capability"`` or ``"steering"``).
        Only affects Session 1 skill selection.

    Returns
    -------
    str: Skill prefix string to prepend to user prompt.
    """
    parts: list[str] = []

    # CLAUDE.md is loaded from the worktree root by Claude Code (system prompt).
    # patch-verification is loaded from .claude/skills/ by Claude Code.
    # Only inject session-specific methodology skills inline.

    # Session-specific skills
    # history-analysis is injected in all sessions as the first step
    parts.append(f"## Skill: history-analysis\n{_read_skill('history-analysis')}")

    if session == 0:
        pass  # No additional inline skills; patch-verification is in .claude/skills/

    elif session == 1:
        parts.append(f"## Skill: diagnose\n{_read_skill('diagnose')}")
        if phase == "capability":
            parts.append(f"## Skill: capability-patch\n{_read_skill('capability-patch')}")
        else:
            parts.append(f"## Skill: steering-patch\n{_read_skill('steering-patch')}")

    elif session == 2:
        parts.append(f"## Skill: diagnose\n{_read_skill('diagnose')}")

    if not parts:
        return ""

    return "Follow these skill instructions:\n\n" + "\n\n".join(parts) + "\n\n---\n\n"


def install_evo_dag_cli(session_root: str, worktree_path: str) -> dict[str, str]:
    """Deploy the evo-dag CLI and return env vars for the SDK session.

    Creates a wrapper script in the session root that invokes the CLI module,
    and returns environment variables needed for the CLI to function.
    """
    import os
    import stat

    session_root_path = Path(session_root)
    dag_json_path = session_root_path / "evolution_dag.json"

    bin_dir = session_root_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script_path = bin_dir / "evo-dag"

    package_source_root = Path(__file__).parents[4]
    script_content = f"""#!/usr/bin/env bash
export PYTHONPATH="{package_source_root}:${{PYTHONPATH:-}}"
export EVOLUTION_DAG_PATH="{dag_json_path}"
exec python3 -m autosaddler.v1.proposer.autosaddler.cli "$@"
"""
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("Installed evo-dag CLI at %s", script_path)

    return {
        "EVOLUTION_DAG_PATH": str(dag_json_path),
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
    }
