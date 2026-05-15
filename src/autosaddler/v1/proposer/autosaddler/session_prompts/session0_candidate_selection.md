# Session 0: Candidate Selection

## Mandatory Skills

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize any skill — execute the full procedure described
in each one. These skills are installed at `.claude/skills/<name>/SKILL.md`
in the current worktree.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `history-analysis` | **Before any other work** (Step 1) | Structured analysis of the full evolution history — classifies every patch as revert vs. preserve to inform candidate selection |
| 2 | `patch-verification` | **After any codebase change** (Step 5) | Verifies the prepared base codebase has no runtime errors — a broken base causes every subsequent session to fail |

> **Enforcement**: If you make any change to the worktree (rsync, cherry-pick,
> revert, or code edit), you MUST run the `patch-verification` skill's full
> procedure before finishing. Skipping verification is a critical failure.

## Goal

Prepare the best possible base codebase for this iteration's patch.
The optimization objective is to **maximize performance across the task
distribution**, with development set (dev) accuracy as the proxy.

Your selection must be **data-driven**, based on dev scores, regression
analysis, and cross-candidate code comparison — not on attachment to
accumulated patches.

**Anti–Sunk-Cost Principle**: The number of iterations invested in a
lineage is NOT a reason to continue it. A long lineage with declining
dev scores is a signal to switch, not to persist. Evaluate each
candidate by its **measured performance** and the **quality of its code**,
not by how many patches led to it.

## Context

- **Iteration**: {iteration}
- **Phase**: {phase}
- **Current worktree**: `{worktree_path}` (fork of C{base_parent_idx})
- **Base parent**: C{base_parent_idx} (worktree: `{parent_worktree}`)

## Candidate Performance (sorted by dev score)

{candidate_table}

**Note**: Dev scores are only available for candidates that passed mini-batch
acceptance. "pending" means the candidate was not evaluated on the dev set.
Use patch history, raw traces, and codebase inspection to assess these.

## DAG Topology

{dag_topology}

## Selection Strategy

Choose the base candidate by weighing dev scores, regression history,
score trajectory, and code quality together. Dev scores are noisy — small
differences may be evaluation variance, so corroborate with regression
counts and code inspection. Do not stay on a lineage just because many
patches were accumulated; patches only have value if performance improved.

**Options**:
- **Continue from latest**: When the lineage is near its peak and stable.
  Revert any regressions and cherry-pick fixes from other candidates.
- **Switch parent**: When dev score has dropped well below the best
  candidate and the trajectory is not recovering. Cherry-pick verified
  good patches from the old lineage.
- **Combine candidates**: When different candidates have complementary
  strengths in non-overlapping areas.

**Key rules**:
- Regressions identified here must be reverted in THIS session. Do NOT
  defer to Session 1 — Session 1 only addresses the current mini-batch
  and will not fix prior regressions.
- When cherry-picking, prefer diff-level precision over copying entire
  files. Use `evo-dag show edge` to see exact diffs and apply only the
  relevant changes. Whole-file copies via rsync bring unintended changes.
- Compare top candidates' codebases directly (`diff -rq`) to find
  cherry-pick opportunities that aren't visible from scores alone.

## Workflow

### 1. Analyze history

**MANDATORY first step.** Run the `history-analysis` skill to build a
structured understanding of the full evolution history. Do NOT skip this
step or truncate `evo-dag show history` with `head`/`tail`. The skill
provides the methodology for reading the complete history without
truncation.

The purpose of this analysis is to **classify every patch** in the history
into two categories that directly inform candidate selection:

#### Patches to revert (caused severe regressions)
- Which patches introduced regressions? Identify the exact iteration,
  candidate, and code diff that caused each regression.
- How severe is each regression? (How many scenarios regressed? Did dev
  score drop?)
- Is the regression still present in the current lineage, or was it
  already fixed by a later patch?
- What files were modified — so you know exactly what to revert or undo.

#### Patches to preserve (drove performance gains)
- Which patches produced the largest improvements in dev score?
- Which patches fixed scenarios that stayed fixed in subsequent iterations
  (durable fixes vs. fragile ones)?
- Which patches generalized well beyond their target mini-batch?
- What files and patterns were involved — so you know what to protect
  when combining candidates.

This classification directly drives your selection decision:
- **Continue from latest**: Revert the harmful diffs while keeping the
  beneficial ones.
- **Switch parent**: If too many regressions have accumulated, switch to
  the best candidate and cherry-pick the preserved patches.
- **Combine candidates**: If different candidates have complementary
  preserved patches, combine them.

### 2. Investigate candidates

Use the following sources to decide which candidate(s) to build on:

1. **History analysis output**: The structured summary from Step 1 —
   lineage trajectory, lessons, and cross-candidate strengths.
2. **Specific candidate**: `evo-dag show node <idx>` — scores, patch intent,
   verdict, output directories.
3. **Raw traces**: Read evaluation output and trace files in the iteration
   output directories listed by `evo-dag show node` or
   `evo-dag show current-batch`. See CLAUDE.md's **Iteration Output
   Structure** section for the file layout.
4. **Candidate codebases**: Read source files directly in other candidates'
   worktrees to understand what changed and whether a fix is worth porting.
   Worktree paths are shown in the candidate table and `evo-dag show node`.
5. **Cross-candidate code comparison (MANDATORY for top candidates)**:
   For the top 2-3 candidates by dev score, directly compare their
   codebases to identify divergent files and cherry-pick opportunities:
   ```bash
   diff -rq {session_root}/worktrees/<candidate_A_dir>/ {session_root}/worktrees/<candidate_B_dir>/ | grep -v '.git'
   ```
   Then read the divergent files in both worktrees to understand which
   candidate's version is better for each file. This comparison reveals
   cherry-pick opportunities that are invisible from score data alone.

### 3. Revert regressions (MANDATORY if continuing from latest)

If you chose to continue from the latest candidate, you MUST revert all
identified regressions in THIS session. Do NOT defer regression fixes to
Session 1 — Session 1 only addresses the current mini-batch and will not
fix prior regressions.

For each regression identified in Step 1:
1. **Locate the exact diff** that caused the regression using
   `evo-dag show edge <parent> <child>` for the iteration that introduced
   it.
2. **Compare with pre-regression code**: Read the same file in the
   pre-regression candidate's worktree to see what the code looked like
   before the harmful change.
3. **Revert only the harmful lines**: Do not revert the entire patch if it
   also contains beneficial changes. Surgically revert only the lines that
   caused the regression.
4. **Preserve beneficial changes**: If a patch contains both good and bad
   changes, keep the good parts. Check whether specific scenarios were
   fixed by specific lines in the diff to determine what to preserve.

After reverting, run the `patch-verification` skill to ensure no runtime
errors were introduced.

### 4. Prepare the base codebase

Each candidate has a **base parent** (the candidate it was forked from) and
optionally **cherry-pick parents** (other candidates whose code was
referenced, copied, or combined). The base parent is set automatically by
the outer loop. Cherry-pick parents are what you record here.

Your worktree starts as a fork of C{base_parent_idx} (base parent).

#### Switching base entirely

If you decided to switch base, copy the new base's entire codebase:
```bash
rsync -a --exclude='.git' {session_root}/worktrees/<candidate_dir>/ ./
```
Then cherry-pick good patches from the old lineage using the diff-level
method below.

#### Diff-level cherry-pick (preferred over file-level copy)

When porting specific fixes from another candidate, use **diff-level
precision** rather than copying entire files. Copying entire files brings
unintended changes from the source candidate.

1. **Identify the exact diff**: Use `evo-dag show edge <parent> <child>`
   to see the exact code changes that constituted the fix you want to port.
2. **Compare files**: Read the specific file in both the source candidate's
   worktree and your current worktree. Use `diff` to see differences:
   ```bash
   diff {session_root}/worktrees/<source_candidate_dir>/<file> ./<file>
   ```
3. **Apply only the relevant changes**: Manually apply only the lines from
   the diff that correspond to the fix. Do not copy unrelated changes that
   the source candidate may have accumulated.
4. **Verify**: After each cherry-pick, run the `patch-verification` skill.

### 5. Verify changes

If you made any changes to the worktree (rsync, cherry-pick, revert, etc.),
you MUST run the `patch-verification` skill to ensure the codebase has no
runtime errors. A broken base codebase will cause every subsequent session
to fail — verification is mandatory.

### 6. Record selection

After preparing the base code, record your selection with
`evo-dag update-selection`. List **all** candidates you referenced — whether
you switched base entirely, cherry-picked specific files, or just used their
code as reference for a re-implementation:
```bash
evo-dag update-selection \
  --parent-candidates "<idx1>,<idx2>,..." \
  --reasoning "<what you took from each and why>"
```

**CRITICAL**: You MUST run `evo-dag update-selection` before finishing this
session. Failure to do so will result in lost selection history that future
iterations need to understand the DAG lineage.

## What Happens Next

After this session, the outer loop runs **Session 1 (Diagnose + Patch)**:
the agent receives the current mini-batch's failing scenarios with execution
traces, diagnoses root causes, and applies targeted code patches to the
worktree you prepared here.
