"""Static strategy prompt loading and skill installation into worktrees.

Each strategy has a complete, inspectable bundle under ``strategy_prompts/``.
The selected CLAUDE document is installed as ``CLAUDE.md``; session templates
only receive runtime data such as iteration IDs, worktree paths, and result
tables. Shared skills remain under ``skills/`` and are filtered by StrategySpec.

Functions:
- ``build_claude_md()``: Loads the selected static CLAUDE document.
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

from autosaddler.v1.proposer.autosaddler.prompt_bundle import (
    COMMON_SKILLS,
    PromptBundle,
    get_session_prompt_path,
    load_prompt_bundle,
)
from autosaddler.v1.proposer.autosaddler.pattern_reporting import render_pattern_table
from autosaddler.v1.proposer.autosaddler.strategy import (
    UNSEEN_SCENARIO_EXPLORATION_SESSION,
    SessionNumber,
    resolve_strategy,
)

if TYPE_CHECKING:
    from autosaddler.v1.proposer.autosaddler.dag import EvolutionDAG
    from autosaddler.v1.proposer.autosaddler.models import ArmPullRecord, EvolutionNode, ScenarioImpact

logger = logging.getLogger(__name__)

# Directory containing the templates and static skills
_MODULE_DIR = Path(__file__).parent
_STRATEGY_PROMPTS_DIR = _MODULE_DIR / "strategy_prompts"
_SKILLS_DIR = _MODULE_DIR / "skills"


def resolve_prompt_bundle(
    *,
    sampling_strategy: str = "autosaddler",
    session_scope: str = "full",
) -> PromptBundle:
    """Load the complete inspectable bundle selected by configuration."""
    return load_prompt_bundle(
        asset_root=_STRATEGY_PROMPTS_DIR,
        canonical_root=_MODULE_DIR,
        skill_root=_SKILLS_DIR,
        sampling_strategy=sampling_strategy,
        session_scope=session_scope,
        renderer_source=Path(__file__).read_text(encoding="utf-8"),
    )


def _read_session_template(
    session: SessionNumber,
    sampling_strategy: str,
) -> str:
    spec = resolve_strategy(sampling_strategy)
    if session not in spec.enabled_sessions:
        raise ValueError(
            f"Session {session} is not enabled for strategy {spec.name.value}"
        )
    path = get_session_prompt_path(
        _STRATEGY_PROMPTS_DIR,
        _MODULE_DIR,
        spec,
        session,
    )
    return path.read_text(encoding="utf-8")


def build_claude_md(
    *,
    sampling_strategy: str = "autosaddler",
    session_scope: str = "full",
) -> str:
    """Build strategy-correct CLAUDE.md content.

    CLAUDE.md is session-independent and contains no template variables.
    It is always loaded by Claude Code at startup and provides:
    - Optimization pipeline overview
    - evo-dag CLI reference
    - Skills introduction
    - Benchmark and agent framework info
    - Constraints
    """
    return resolve_prompt_bundle(
        sampling_strategy=sampling_strategy,
        session_scope=session_scope,
    ).claude_md


def build_session0_prompt(
    *,
    iteration: int,
    worktree_path: str,
    parent_worktree: str,
    base_parent_idx: int,
    session_root: str,
    dag: EvolutionDAG,
    phase: str = "capability",
    sampling_strategy: str = "autosaddler",
) -> str:
    """Render Session 0 (candidate selection) prompt.

    Pre-computes candidate performance table, DAG topology, and accumulated
    lessons from the DAG so the agent can make an informed selection without
    needing to run multiple CLI queries.
    """
    template = _read_session_template(0, sampling_strategy)

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


def render_arm_pull_record(rec: "ArmPullRecord", dev_by_idx: dict | None) -> list[str]:
    """Render one arm-pull record to markdown lines.

    Shared by Session 1 (diagnose/patch) and Session 4 (arm scoring) so both see
    IDENTICAL per-pull detail. Covers the three pull kinds: ``patched`` (approach
    + dev-set delta + per-scenario reflections), ``all_pass_skip`` (scenarios
    already passing, no patch), and ``failed_attempt`` (no usable patch).
    """
    n = rec.node
    _dev = dev_by_idx or {}
    base_idx = n.base_parent_idx
    base_label = f"C{base_idx}" if base_idx is not None else "seed"

    intent_lines: list[str] = []
    if n.patch_intent:
        intent = n.patch_intent
        if intent.diagnosis:
            intent_lines.append(f"- **Diagnosis**: {intent.diagnosis}")
        if intent.target_scenarios:
            intent_lines.append(
                f"- **Target scenarios**: {', '.join(intent.target_scenarios)}"
            )
        if intent.approach:
            intent_lines.append(f"- **Approach**: {intent.approach}")
        if intent.files_changed:
            intent_lines.append(
                f"- **Files changed**: {', '.join(intent.files_changed)}"
            )
        if intent.change_summary:
            intent_lines.append(f"- **Change summary**: {intent.change_summary}")

    if rec.kind == "all_pass_skip":
        sids = [si.scenario_id for si in rec.scenario_outcomes] or list(n.mini_batch_ids)
        return [
            f"### C{n.idx} (iteration {n.iteration}, base {base_label}) "
            "\u2014 all scenarios already PASSING \u2192 no patch (skipped)",
            f"- **Outcome**: {', '.join(sids) or '(scenarios)'} passed on "
            "train_before; no fix was needed this pull.",
        ]

    if rec.kind == "failed_attempt":
        reason = n.abandon_reason or "failed"
        return [
            f"### C{n.idx} (iteration {n.iteration}, base {base_label}) "
            f"\u2014 attempt produced NO usable patch ({reason})",
            *intent_lines,
            "- **Outcome**: the diagnose/patch or verification step failed; "
            "no scenario result recorded.",
        ]

    # kind == "patched"
    nd = _dev.get(n.idx)
    pd = _dev.get(base_idx) if base_idx is not None else None
    if nd is None:
        dev_str = "dev: not evaluated (patch not accepted on mini-batch)"
    elif pd is None:
        dev_str = f"dev: {nd:.4f}"
    else:
        dev_str = f"dev: {pd:.4f} \u2192 {nd:.4f} ({nd - pd:+.4f})"
    lines = [
        f"### C{n.idx} (iteration {n.iteration}, base {base_label}) \u2014 {dev_str}",
        *intent_lines,
    ]
    if not intent_lines:
        lines.append("- **Approach**: (no approach recorded)")
    reflections = n.patch_verdict.reflections if n.patch_verdict else []
    for r in reflections:
        lines.append(f"- **`{r.scenario_id}`** ({r.status_change}):")
        if r.root_cause:
            lines.append(f"  - Root cause: {r.root_cause}")
        if r.explanation:
            lines.append(f"  - What happened: {r.explanation}")
        if r.prevention_or_next:
            lines.append(f"  - Next / prevention: {r.prevention_or_next}")
        if r.generalization_note:
            lines.append(f"  - Generalization (dev-set): {r.generalization_note}")
    return lines


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
    arm_pull_history: list[ArmPullRecord] | None = None,
    dev_by_idx: dict[int, float | None] | None = None,
    sampling_strategy: str = "autosaddler",
) -> str:
    """Render Session 1 (diagnose + patch) prompt."""
    template = _read_session_template(1, sampling_strategy)

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

    # Complete pull history on the SAME arm (pattern) across all prior
    # iterations it was pulled: patched attempts (approach, dev-set delta, every
    # per-scenario reflection), all-pass skips (scenarios already passing, no
    # patch), and failed attempts. Rendered via the shared per-record helper so
    # Session 1 and Session 4 show identical detail. No truncation, no cap.
    arm_history_section = ""
    if arm_pull_history:
        hist_lines = [
            "## Prior Attempts on This Arm",
            "",
            "This exact arm (same failure pattern) was pulled in earlier "
            "iterations. Below is the COMPLETE history of those pulls \u2014 each "
            "patched attempt's approach, its **dev-set** accuracy impact, and "
            "every per-scenario reflection; plus pulls where the scenarios "
            "already passed (no patch) or the attempt failed.",
            "",
            "**Key constraint**: Do NOT repeat approaches that already failed to "
            "fix the target, or that fixed the mini-batch but dropped dev-set "
            "accuracy. The reflections below explain what didn't work (see each "
            "'Next / prevention'). Build on approaches that fixed the target "
            "while keeping dev-set accuracy flat or higher, focus on any "
            "still-failing scenario, and fix any regressions they introduced. If "
            "prior pulls show the scenarios already passing, treat the current "
            "failure as possibly intermittent \u2014 avoid over-fitting a patch to "
            "a transient failure.",
            "",
        ]
        for rec in arm_pull_history:
            hist_lines.extend(render_arm_pull_record(rec, dev_by_idx))
            hist_lines.append("")
        arm_history_section = "\n".join(hist_lines)

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
        arm_history_section=arm_history_section,
    )


def build_session2_prompt(
    node: EvolutionNode,
    scenario_impacts: list[ScenarioImpact],
    all_worktrees: dict[int, str] | None = None,
    dag: EvolutionDAG | None = None,
    phase: str = "unknown",
    sampling_strategy: str = "autosaddler",
) -> str:
    """Render Session 2 (reflection) prompt with initial/re-evaluation results."""
    template = _read_session_template(2, sampling_strategy)

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
    skill_names: tuple[str, ...] | None = None,
) -> None:
    """Install CLAUDE.md and skill files into the worktree.

    - ``CLAUDE.md``: Always-on context at worktree root
        - Strategy capabilities: installed under ``.claude/skills/``.
    """
    wt = Path(worktree_path)

    # Install CLAUDE.md at worktree root
    (wt / "CLAUDE.md").write_text(built_claude_md, encoding="utf-8")
    logger.info("Installed CLAUDE.md at %s (phase=%s)", wt / "CLAUDE.md", phase)

    # Install only capabilities exposed by the resolved strategy bundle.
    skills_target = wt / ".claude" / "skills"
    if skills_target.exists():
        shutil.rmtree(skills_target)

    for skill_name in skill_names or COMMON_SKILLS:
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


def build_skill_prefix(
    session: SessionNumber,
    phase: str = "capability",
    sampling_strategy: str = "autosaddler",
) -> str:
    """Build a skill prefix to prepend to the user prompt.

    Reads CLAUDE.md and session-appropriate SKILL.md files, then combines
    them into a string that is prepended to the user prompt so Claude Code
    receives the instructions directly in context.

    This approach works reliably with copilot proxy (unlike
    --append-system-prompt which may be ignored).

    Parameters
    ----------
    session:
        Session number enabled by the selected strategy.
    phase:
        Current optimization phase (``"capability"`` or ``"steering"``).
        Only affects Session 1 skill selection.
    Returns
    -------
    str: Skill prefix string to prepend to user prompt.
    """
    spec = resolve_strategy(sampling_strategy)
    if session not in spec.enabled_sessions:
        raise ValueError(
            f"Session {session} is not enabled for strategy {spec.name.value}"
        )
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

    elif session == 3:
        parts.append(f"## Skill: symptom-extract\n{_read_skill('symptom-extract')}")
        parts.append(f"## Skill: symptom-normalize\n{_read_skill('symptom-normalize')}")

    elif session == 4:
        parts.append(f"## Skill: progress-scoring\n{_read_skill('progress-scoring')}")

    elif session == UNSEEN_SCENARIO_EXPLORATION_SESSION:
        pass  # Decision methodology is fully contained in the Session 3.5 prompt.

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

    gepa_src = Path(__file__).parents[4]  # .../src
    venv_python = Path(gepa_src).parent / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    script_content = f"""#!/usr/bin/env bash
export PYTHONPATH="{gepa_src}:${{PYTHONPATH:-}}"
export EVOLUTION_DAG_PATH="{dag_json_path}"
exec "{python_exec}" -m autosaddler.v1.proposer.autosaddler.cli "$@"
"""
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("Installed evo-dag CLI at %s", script_path)

    return {
        "EVOLUTION_DAG_PATH": str(dag_json_path),
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
    }


def install_pattern_cli(
    session_root: str,
    worktree_path: str,
    current_iteration: int = 1,
    eta: float = 0.3,
    sampling_strategy: str = "activesaddler",
    session: SessionNumber = 3,
    artifact_dir: str | Path | None = None,
    candidate_idx: int | None = None,
) -> dict[str, str]:
    """Deploy the pattern CLI and return env vars for the SDK session.

    Creates a wrapper script in the session root that invokes the pattern CLI
    module, and returns environment variables needed for the CLI to function.
    """
    import os
    import stat

    spec = resolve_strategy(sampling_strategy)
    capabilities = spec.pattern_cli_capabilities_for_session(session)
    if not capabilities:
        raise ValueError(
            f"pattern CLI is not available in Session {session} for "
            f"strategy {spec.name.value}"
        )

    session_root_path = Path(session_root)
    registry_path = session_root_path / "pattern_registry.json"

    bin_dir = session_root_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script_path = bin_dir / "pattern"

    gepa_src = Path(__file__).parents[4]  # .../src
    venv_python = Path(gepa_src).parent / ".venv" / "bin" / "python"
    python_exec = str(venv_python) if venv_python.exists() else "python3"
    capability_list = ",".join(sorted(capabilities))
    if (artifact_dir is None) != (candidate_idx is None):
        raise ValueError("artifact_dir and candidate_idx must be set together")
    if (
        session == UNSEEN_SCENARIO_EXPLORATION_SESSION
        and artifact_dir is None
    ):
        raise ValueError(
            "Session 3.5 requires artifact_dir and candidate_idx"
        )
    artifact_exports = ""
    if artifact_dir is not None and candidate_idx is not None:
        artifact_exports = (
            f'export AUTOSADDLER_ITERATION_ARTIFACT_DIR="{artifact_dir}"\n'
            f'export CURRENT_CANDIDATE_IDX="{candidate_idx}"\n'
        )
    script_content = f"""#!/usr/bin/env bash
export PYTHONPATH="{gepa_src}:${{PYTHONPATH:-}}"
export PATTERN_REGISTRY_PATH="{registry_path}"
export CURRENT_ITERATION="{current_iteration}"
{artifact_exports}export ETA="{eta}"
export SAMPLING_STRATEGY="{spec.name.value}"
export PATTERN_CLI_CAPABILITIES="{capability_list}"
export SCORING_MODE="{spec.scoring}"
exec "{python_exec}" -m autosaddler.v1.proposer.autosaddler.pattern_cli "$@"
"""
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("Installed pattern CLI at %s", script_path)

    env = {
        "PATTERN_REGISTRY_PATH": str(registry_path),
        "CURRENT_ITERATION": str(current_iteration),
        "ETA": str(eta),
        "SAMPLING_STRATEGY": spec.name.value,
        "PATTERN_CLI_CAPABILITIES": capability_list,
        "SCORING_MODE": spec.scoring,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
    }
    if artifact_dir is not None and candidate_idx is not None:
        env["AUTOSADDLER_ITERATION_ARTIFACT_DIR"] = str(artifact_dir)
        env["CURRENT_CANDIDATE_IDX"] = str(candidate_idx)
    return env


def build_session3_prompt(
    *,
    iteration: int,
    candidate_idx: int,
    worktree_path: str,
    session_root: str,
    before_output_dir: str,
    after_output_dir: str,
    pre_patch_failures: list[dict],
    post_patch_failures: list[dict],
    sampling_strategy: str = "activesaddler",
) -> str:
    """Render Session 3 (pattern extraction) prompt.

    Parameters
    ----------
    iteration: Current iteration number.
    candidate_idx: Current candidate DAG node index.
    worktree_path: Path to the current worktree.
    session_root: Path to the session root directory.
    before_output_dir: Path to pre-patch evaluation output.
    after_output_dir: Path to post-patch evaluation output.
    pre_patch_failures: Dicts containing the batch-level Session 1 diagnosis,
        proposer reasoning path, and Session 2's reviewed root cause and
        post-hoc patch-effect explanation.
    post_patch_failures: Dicts containing status change, Session 1's batch-level
        patch approach, and Session 2's failure cause and post-hoc explanation.
    """
    spec = resolve_strategy(sampling_strategy)
    if not spec.pattern_sampling:
        raise ValueError(
            f"Session 3 is not enabled for strategy {spec.name.value}"
        )
    template = _read_session_template(3, sampling_strategy)

    # Format pre-patch failures
    if pre_patch_failures:
        pre_lines = []
        for f in pre_patch_failures:
            pre_lines.append(f"- **{f['scenario_id']}**")
            pre_lines.append(
                "  - Session 1 diagnosis before patching (batch-level): "
                f"{f.get('session1_diagnosis') or '(not recorded)'}"
            )
            pre_lines.extend(
                [
                    "  - Session 1 detailed reasoning file: "
                    f"`{f['proposer_reasoning_path']}`",
                    "    - Contains task descriptions, expected vs. actual behavior, "
                    "trace failure points, diagnosed root causes, and proposed "
                    "resolution strategies from before patching.",
                ]
            )
            pre_lines.append(
                "  - Session 2 reviewed underlying root cause (written after "
                "observing the patch outcome; what originally caused the failure): "
                f"{f.get('session2_root_cause') or '(not recorded)'}"
            )
            pre_lines.append(
                "  - Session 2 post-hoc patch-effect explanation (how the patch "
                "affected that cause): "
                f"{f.get('session2_explanation') or '(not recorded)'}"
            )
        pre_patch_section = "\n".join(pre_lines)
    else:
        pre_patch_section = "(none — all scenarios passed before patching)"

    # Format post-patch failures
    if post_patch_failures:
        post_lines = []
        for f in post_patch_failures:
            status = f.get("status_change", "still_failing")
            post_lines.append(f"- **{f['scenario_id']}** ({status})")
            post_lines.append(f"  - Status change: `{status}`")
            post_lines.append(
                "  - Session 1 patch approach (batch-level): "
                f"{f.get('session1_patch_approach') or '(not recorded)'}"
            )
            cause_label = (
                "remaining or revised failure cause"
                if status == "still_failing"
                else "new regression cause or failure mechanism"
            )
            post_lines.append(
                f"  - Session 2 root cause ({cause_label}; what caused the "
                "post-patch failure): "
                f"{f.get('session2_root_cause') or '(not recorded)'}"
            )
            post_lines.append(
                "  - Session 2 post-hoc patch-effect explanation (how the patch "
                "affected or introduced that cause): "
                f"{f.get('session2_explanation') or '(not recorded)'}"
            )
        post_patch_section = "\n".join(post_lines)
    else:
        post_patch_section = "(none — all scenarios passed after patching)"

    return template.format(
        iteration=iteration,
        candidate_idx=candidate_idx,
        worktree_path=worktree_path,
        session_root=session_root,
        before_output_dir=before_output_dir,
        after_output_dir=after_output_dir,
        pre_patch_failures=pre_patch_section,
        post_patch_failures=post_patch_section,
    )


def _build_prepared_arm_context(
    *,
    iteration: int,
    prepared_candidate_idx: int,
    prepared_worktree_path: str,
    prepared_commit: str,
    provisional_parent_idx: int,
    provisional_parent_commit: str,
    session0_status: str,
    session_root: str,
    dag: EvolutionDAG,
    registry,
    eta: float = 0.3,
) -> dict[str, object]:
    """Build shared prepared-harness and candidate-arm prompt context."""
    prepared_node = dag.nodes[prepared_candidate_idx]
    provisional_parent = dag.nodes[provisional_parent_idx]
    selection = prepared_node.selection_decision
    selection_json_path = (
        prepared_node.sdk_session_selection.session_json_path
        if prepared_node.sdk_session_selection
        else "(not available)"
    )
    summary_lines = [
        "### Session 0 Preparation",
        "",
        f"- **Status**: {session0_status}",
        "- **Referenced candidates**: " + (
            ", ".join(f"C{idx}" for idx in selection.parent_candidates)
            if selection and selection.parent_candidates
            else "(not recorded)"
        ),
        "- **Selection reasoning**: " + (
            selection.reasoning if selection else "(not recorded)"
        ),
        f"- **Session 0 JSON**: `{selection_json_path}`",
        f"- **Prepared commit**: `{prepared_commit}`",
        "- **Authoritative prepared diff**: "
        f"`git diff {provisional_parent_commit} {prepared_commit}`",
        "",
        f"### Provisional Parent C{provisional_parent_idx}",
        "",
    ]
    if provisional_parent.iteration == 0:
        summary_lines.append("(seed harness — no prior Session 1/2 records)")
    else:
        intent = provisional_parent.patch_intent
        summary_lines.extend(
            [
                "#### Session 1 Patch Intent",
                "",
                "- **Targets**: " + (
                    ", ".join(intent.target_scenarios)
                    if intent and intent.target_scenarios
                    else "(not recorded)"
                ),
                "- **Diagnosis**: " + (
                    intent.diagnosis if intent and intent.diagnosis else "(not recorded)"
                ),
                "- **Approach**: " + (
                    intent.approach if intent else "(not recorded)"
                ),
                "- **Files changed**: " + (
                    ", ".join(intent.files_changed)
                    if intent and intent.files_changed
                    else "(not recorded)"
                ),
                "- **Change summary**: " + (
                    intent.change_summary if intent else "(not recorded)"
                ),
                "",
                "#### Session 2 Reflections",
                "",
            ]
        )
        reflections = (
            provisional_parent.patch_verdict.reflections
            if provisional_parent.patch_verdict
            else []
        )
        if not reflections:
            summary_lines.append("(not recorded)")
        for reflection in reflections:
            summary_lines.extend(
                [
                    f"##### {reflection.scenario_id}",
                    "",
                    f"- **Status**: {reflection.status_change}",
                    f"- **Root cause**: {reflection.root_cause or '(not recorded)'}",
                    f"- **Explanation**: {reflection.explanation or '(not recorded)'}",
                    "- **Prevention or next**: "
                    f"{reflection.prevention_or_next or '(not recorded)'}",
                    "- **Generalization note**: "
                    f"{reflection.generalization_note or '(not recorded)'}",
                    "",
                ]
            )
    prepared_harness_summary = "\n".join(summary_lines)

    # ── Candidate patterns table (arms owning >= 1 scenario) ──
    scores = registry.compute_scores(iteration, eta)
    arms = [p for p in registry.list_patterns() if p.tuples]
    candidate_patterns = (
        render_pattern_table(registry, scores, include_untagged=False)
        if arms
        else "(no candidate patterns yet)"
    )

    return {
        "iteration": iteration,
        "prepared_candidate_idx": prepared_candidate_idx,
        "prepared_worktree_path": prepared_worktree_path,
        "prepared_commit": prepared_commit,
        "provisional_parent_idx": provisional_parent_idx,
        "provisional_parent_commit": provisional_parent_commit,
        "selection_session_json_path": selection_json_path,
        "session_root": session_root,
        "prepared_harness_summary": prepared_harness_summary,
        "candidate_patterns": candidate_patterns,
        "num_arms": len(arms),
    }


def build_arm_scoring_prompt(
    *,
    iteration: int,
    prepared_candidate_idx: int,
    prepared_worktree_path: str,
    prepared_commit: str,
    provisional_parent_idx: int,
    provisional_parent_commit: str,
    session0_status: str,
    session_root: str,
    dag: EvolutionDAG,
    registry,
    eta: float = 0.3,
    sampling_strategy: str = "activesaddler",
) -> str:
    """Render scoring-only Session 4 for ActiveSaddler."""
    template = _read_session_template(4, sampling_strategy)
    context = _build_prepared_arm_context(
        iteration=iteration,
        prepared_candidate_idx=prepared_candidate_idx,
        prepared_worktree_path=prepared_worktree_path,
        prepared_commit=prepared_commit,
        provisional_parent_idx=provisional_parent_idx,
        provisional_parent_commit=provisional_parent_commit,
        session0_status=session0_status,
        session_root=session_root,
        dag=dag,
        registry=registry,
        eta=eta,
    )
    return template.format(**context)


def build_unseen_scenario_exploration_prompt(
    *,
    iteration: int,
    prepared_candidate_idx: int,
    prepared_worktree_path: str,
    prepared_commit: str,
    provisional_parent_idx: int,
    provisional_parent_commit: str,
    session0_status: str,
    session_root: str,
    dag: EvolutionDAG,
    registry,
    unseen_pool_size: int,
    eta: float = 0.3,
) -> str:
    """Render ActiveSaddler Session 3.5 before any Session 4 scoring."""
    template = _read_session_template(
        UNSEEN_SCENARIO_EXPLORATION_SESSION,
        "activesaddler",
    )
    context = _build_prepared_arm_context(
        iteration=iteration,
        prepared_candidate_idx=prepared_candidate_idx,
        prepared_worktree_path=prepared_worktree_path,
        prepared_commit=prepared_commit,
        provisional_parent_idx=provisional_parent_idx,
        provisional_parent_commit=provisional_parent_commit,
        session0_status=session0_status,
        session_root=session_root,
        dag=dag,
        registry=registry,
        eta=eta,
    )
    return template.format(**context, unseen_pool_size=unseen_pool_size)
