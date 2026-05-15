# Session 1: Diagnose Failures + Apply Patches

## Mandatory Skills

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize any skill — execute the full procedure described
in each one. These skills are installed at `.claude/skills/<name>/SKILL.md`
in the current worktree.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `history-analysis` | **Before any other work** (Step 1) | Structured analysis of the full evolution history — identifies relevant lessons, prior attempts, proven strategies, and regression-prone areas |
| 2 | `diagnose` | **Before applying any patch** (Step 2) | Root-cause analysis from agent traces and codebase — pinpoints exactly why each failing scenario fails |
| 3 | `capability-patch` or `steering-patch` | **When applying patches** (Step 3) | Phase-appropriate patch methodology — `capability-patch` during capability phase, `steering-patch` during steering phase. Follow the skill matching the current phase |
| 4 | `patch-verification` | **After applying all patches** (Step 4) | Crash-safety verification — syntax, imports, docstrings, hooks, logic. A crashing patch is worse than no patch |

> **Enforcement**: Every step above is mandatory and sequential. Do NOT
> apply patches without first completing diagnosis. Do NOT finish the
> session without running the full `patch-verification` procedure.
> Do NOT skip `history-analysis` — it prevents repeating known failures.

## Goal

Fix failing scenarios in this mini-batch by diagnosing root causes from
execution traces and the agent codebase, then applying targeted code
patches — without regressing scenarios that already pass.

Each scenario is a task the agent must solve. The initial evaluation on the
mini-batch has already run the agent on each scenario and recorded its
execution trace:
every tool call, every response, and every reasoning step. Your job is to
read these traces alongside the agent codebase, identify **why** the agent
failed, determine what code change would fix the root cause, and apply it.

The optimization objective is to **maximize performance across the full
task distribution** — not just this mini-batch. Patches should address
general behavioral patterns, not hardcode scenario-specific answers.
A patch that fixes one scenario by adding a general rule will also help
similar unseen scenarios; a patch that hardcodes a specific answer helps
only that one scenario and may harm others.

**Acceptance**: A patch is accepted if the sum of re-evaluation scores
exceeds the sum of initial scores. Even fixing a single scenario is enough. However, regressions (passing → failing) are
recorded as bad patterns — avoid them.

## Context

- **Iteration**: {iteration}
- **Candidate**: C{candidate_idx}
- **Current worktree**: `{worktree_path}`
- **Base parent**: C{base_parent_idx} (worktree: `{parent_worktree}`)
{cherry_pick_parents_section}- **Phase**: {phase}
- **Initial evaluation output**: `{before_output_dir}`

## Current Mini-Batch ({num_scenarios} scenarios)
{mini_batch_listing}

### Initial Scores (pass rate: {before_pass_rate})
{before_scores_listing}

## Patch Types for This Phase

{patch_types_section}

## Workflow

### 1. Analyze history

**MANDATORY first step.** Run the `history-analysis` skill to build a
structured understanding of the full evolution history. Do NOT skip this
step or truncate `evo-dag show history` with `head`/`tail`. The skill
provides the methodology for reading the complete history without
truncation. Focus on:
- Relevant lessons (good/bad patterns) for the current mini-batch
- Prior attempts on the failing scenarios — what worked and what failed
- Proven patch strategies and regression-prone areas

### 2. Diagnose failing scenarios

For each failing scenario, use the `diagnose` skill to identify the root
cause. The key is to find **why** the agent made the wrong decision, not
just **what** went wrong. Diagnosis requires reading both the execution
trace and the agent codebase.

**Trace files** (output dir: `{before_output_dir}`):
See CLAUDE.md's **Iteration Output Structure** section for the exact file
layout. Key files to read:
- **Evaluation rationale**: Per-scenario scores and judge rationale — the
  evaluator's summary of what was expected vs what the agent produced.
- **Agent execution traces**: Per-scenario full tool call, response, and
  reasoning traces — where you find the exact point where the agent
  diverged from correct behavior.

**Agent codebase**: current worktree `{worktree_path}`

The `diagnose` skill provides the full methodology — follow it for each
failing scenario.

### 3. Apply patches

Based on your diagnosis, use the phase-appropriate patch skill:
- **Capability phase** (`capability-patch`): Expand what the agent CAN DO
  — new tool methods, parameter additions, implementation fixes,
  infrastructure changes. These unlock scenarios that are unreachable
  through prompt tuning alone.
- **Steering phase** (`steering-patch`): Refine HOW the agent behaves
  — prompt rules, tool description corrections, PreToolUse hooks. These
  fine-tune the agent's use of existing capabilities.

**Patch guidelines**:
- **Target the root cause, not the symptom.** If the agent sends a wrong
  email subject, the fix is a general rule about deriving subjects from
  user wording — not hardcoding the correct subject for one scenario.
- **Keep patches generalizable.** Rules should be abstract: no scenario
  IDs, no specific names, no hardcoded answers. A good patch helps all
  scenarios that share the same root cause pattern.
- **Align agent-facing text with capability changes.** When you add or
  modify tools, parameters, or infrastructure code, also update the
  system prompt, tool docstrings, and/or hooks so the agent is aware of
  the changes. Check for conflicting rules.
- **Learn from patch history.** Check `evo-dag show history` — it contains
  diffs, per-scenario results, reflections, and accumulated lessons. Use
  it to understand: which patches generalized well to the dev set, which
  caused regressions, which root-cause patterns were effectively resolved,
  and which approaches repeatedly failed. Build on proven strategies and
  avoid repeating known bad patterns.
- **Prefer replacement over addition** when modifying prompts. The system
  prompt has a finite attention budget — adding rules without removing or
  merging existing ones dilutes their impact.

### 4. Verify patches

After applying all patches, you MUST run the `patch-verification` skill to
ensure the patched codebase has no runtime errors. A patch that crashes at
runtime is worse than no patch — verification is mandatory before
committing. Do NOT skip this step under any circumstances.

### 5. Write reasoning and record intent

After applying all patches, write your reasoning to `proposer_reasoning.md`
in the working directory root. For each targeted scenario, include: task
description, expected vs actual behavior, trace analysis (which step went
wrong), root cause, and resolution strategy (generalizable rule). This
sharpens the `--diagnosis` that propagates to future iterations via
`evo-dag show history`. Then record intent:

```bash
evo-dag update-intent \
  --target-scenarios "id1,id2" \
  --diagnosis "Root cause analysis of why target scenarios fail" \
  --approach "Brief description of the patch approach" \
  --files-changed "f1.py,f2.py" \
  --change-summary "What was changed and why"
```

**CRITICAL**: You MUST run `evo-dag update-intent` before finishing this
session. Failure to do so will result in incomplete iteration records and
degrade the quality of future iterations' patch history analysis.

## What Happens Next

After this session, the outer loop re-evaluates the patched worktree on the
same mini-batch, computes initial/re-evaluation per-scenario impacts
(fixed, regressed, still_failing, still_passing), and records the patch
verdict. If the patch is accepted (re-evaluation score > initial score),
the candidate is also evaluated on the full dev-set to measure
generalizability. Then
**Session 2 (Reflection)** runs, where you analyze the results and record
structured reflections for each scenario.
