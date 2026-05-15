---
name: history-analysis
description: "Use to systematically analyze the full evolution history without output truncation — produces a structured summary of what matters for the current iteration"
---

# History Analysis

## Overview

As iterations accumulate, the output of `evo-dag show history` grows too
large to read in a single terminal command. Naively piping through
`head -200` or `tail -200` discards critical information — early
iterations' lessons are lost with `tail`, and recent results are lost
with `head`. This skill provides a structured methodology to analyze
the **complete** history using targeted CLI commands, producing a focused
summary of what matters for the current iteration.

## When to Use

At the **start** of every session (Session 0, 1, and 2) before any other
work. The history analysis provides the context needed for informed
decisions — skipping it leads to repeated mistakes, redundant patches,
and missed lessons.

## The Problem with Truncation

**Do NOT** pipe `evo-dag show history` through `head`, `tail`, or any
truncation command. This loses information:

- `head -N`: Loses all recent iterations' results and reflections
- `tail -N`: Loses early iterations' foundational lessons and patterns
- Increasing the number (`head -300`, `tail -500`) is a losing battle —
  the history grows every iteration

## History Analysis Procedure

### Step 1: Quick Orientation

Start with small, complete outputs to establish context:

```bash
# DAG topology and best candidate (always small output)
evo-dag summary

# DAG lineage visualization (always small output)
evo-dag show lineage
```

From this, note:
- How many iterations have been run
- Which candidate has the best dev score
- The current lineage path

### Step 2: Full History via File Redirect

Redirect the full history to a temporary file and read it with file
tools. This avoids terminal output truncation entirely:

```bash
evo-dag show history > /tmp/evo_history.txt
```

Then read the file in sections using file reading tools (e.g., `cat` with
line ranges, or IDE file reading). This lets you see the **complete**
history regardless of length.

Read the file in manageable sections:
- Start from the beginning to understand early foundational changes
- Read the end to see the most recent iterations
- Search for specific patterns or scenario IDs as needed

As you read, classify each iteration's patch into one of:

1. **Good patch** (✓ good patch): Note the iteration, approach, files
   changed, and which scenarios were fixed. These are patches to preserve.
2. **Regression patch** (✗ regression): Note the iteration, exact diff
   that caused harm, which scenarios regressed, and whether the regression
   persists. These are patches to revert.
3. **Ineffective patch** (✗ ineffective): Note what was tried and why it
   failed — to avoid repeating the same approach.

For each regression or good patch, also check the actual code diff in
the history output. The diff shows exactly what lines were added/removed
in which files — this is essential for knowing what to revert or protect
when combining candidates.

### Step 3: Targeted Deep Dives

For specific information, use targeted commands that produce focused
output:

```bash
# Specific scenario's full history (root causes, attempted fixes)
evo-dag show scenario <scenario_id>

# Specific candidate's details (scores, intent, verdict)
evo-dag show node <idx>

# Specific edge's diff and impacts
evo-dag show edge <parent_idx> <child_idx>

# Current mini-batch context
evo-dag show current-batch
```

Use `evo-dag show edge <parent> <child>` to inspect the full diff of a
specific patch when the diff in `show history` is truncated. This gives
the complete code diff, files changed, and per-scenario impacts (fixed,
regressed, still_failing, still_passing).

### Step 4: Produce a Structured Summary

After reading the full history, produce a summary organized by these
categories. The specific focus depends on the current session:

#### For Session 0 (Candidate Selection):

The goal is to classify every patch as **revert** vs. **preserve** so you
can select the best candidate or combine candidates optimally:

- **Patches to revert**: Which patches caused regressions? For each, note
  the iteration, candidate index, exact files/diffs that caused harm, the
  scenarios that regressed, and whether the regression persists in the
  current lineage.
- **Patches to preserve**: Which patches drove the largest dev score gains?
  Which fixes were durable (stayed fixed in subsequent iterations)? Which
  generalized well beyond the target mini-batch? Note the files and
  patterns involved so you know what to protect.
- **Cross-candidate complementarity**: Do different candidates have
  non-overlapping strengths (e.g., one fixed email tools, another fixed
  calendar tools)? If so, their preserved patches can be combined.
- **Lineage health**: Is the current lineage accumulating regressions
  faster than fixes? If so, switching parent may be better than surgical
  revert.

This classification directly drives the selection decision:
- **Continue from latest**: Revert the harmful diffs while keeping the
  beneficial ones.
- **Switch parent**: If too many regressions have accumulated, switch to
  the best candidate and cherry-pick the preserved patches.
- **Combine candidates**: If different candidates have complementary
  preserved patches, combine them.

#### For Session 1 (Diagnose + Patch):

The goal is to inform diagnosis and patching — avoid repeating failed
approaches and build on proven strategies:

- **Relevant good/bad patterns**: Which patterns from the history apply to
  the current mini-batch's failing scenarios?
- **Prior attempts on target scenarios**: For each failing scenario, use
  `evo-dag show scenario <id>` to check what was tried before, what
  worked, and what failed. Do not re-attempt approaches that already
  failed.
- **Proven patch strategies**: Which types of patches (tool fixes,
  docstring changes, hooks, prompt rules) have historically generalized
  well to the dev set?
- **Regression-prone areas**: Which files or components have historically
  caused regressions when modified? Take extra care with these.

#### For Session 2 (Reflection):

The goal is to write meaningful reflections that add new information
beyond what prior iterations already captured:

- **Historical context for each scenario**: For each scenario in the
  mini-batch, what is its full history? How many times has it been
  attempted? Use `evo-dag show scenario <id>` for each.
- **Pattern evolution**: Are there emerging patterns (recurring root
  causes, files that keep causing problems) not yet captured in the
  history's good/bad patterns?
- **Dev score attribution**: For prior patches, which ones improved/hurt
  dev accuracy and why? Use this to write informed
  `--generalization-note` on reflections.

## Key Principles

1. **Never truncate**: Use file redirect + file reading instead of
   pipe truncation
2. **Start broad, then narrow**: Summary → lessons → full history →
   targeted queries
3. **Focus on relevance**: Not all history matters equally for the current
   iteration — prioritize recent iterations, target scenarios' history,
   and accumulated lessons
4. **Produce actionable output**: The summary should directly inform the
   next step — candidate selection, diagnosis, or reflection writing
