# Session 2: Reflection

## Mandatory Skills

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize any skill — execute the full procedure described
in each one. These skills are installed at `.claude/skills/<name>/SKILL.md`
in the current worktree.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `history-analysis` | **Before any other work** (Step 1) | Structured analysis of the full evolution history — provides historical context for each scenario, pattern evolution, and dev score attribution needed for meaningful reflections |
| 2 | `diagnose` | **For every regressed AND fixed scenario** (Step 3) | Causal attribution — determines whether each state change (PASS↔FAIL) was truly caused by the patch or is a stochastic artifact from LLM non-determinism. Accurate classification is critical to avoid polluting lessons with false signal |

> **Enforcement**: Do NOT attribute any state change to "stochastic noise"
> or "LLM non-determinism" without running the full `diagnose` skill
> procedure. Every fixed/regressed classification must cite specific
> evidence from the diagnosis.

## Goal

Analyze the initial/re-evaluation results for every scenario in the
mini-batch and record structured reflections that future iterations can
learn from.

Your reflections feed directly into `evo-dag show history` — the primary
knowledge store that future iterations consult when diagnosing failures,
choosing patch strategies, and avoiding past mistakes. The quality of your
reflections determines how effectively the pipeline learns. Be specific:
reference exact tool calls, code paths, and behavioral patterns. Vague
reflections waste learning potential.

## Context

- **Iteration**: {iteration}
- **Candidate**: C{candidate_idx}
- **Current worktree**: `{worktree_path}`
- **Base parent**: C{base_parent_idx} (worktree: `{parent_worktree}`)
- **Phase**: {phase}
- **Initial evaluation output**: `{before_output_dir}`
- **Re-evaluation output**: `{train_after_cycle_dir}`

## Session 1 Reasoning

The following is the proposer's diagnosis and patch rationale from
Session 1. Use this to understand the **intent** behind the patch —
what root causes were identified, what the patch was designed to fix,
and what behavioral changes were expected.

{proposer_reasoning}

## Results Summary

{results_summary}

## Per-Scenario Details

{per_scenario_details}

{generalization_section}

## The Four Outcomes

| Before | After | Status | Key Question |
|--------|-------|--------|-------------|
| FAIL | PASS | `fixed` | What root cause did the patch resolve? How? |
| PASS | FAIL | `regressed` | What did the patch break? Why? |
| FAIL | FAIL | `still_failing` | Why was the patch insufficient? What to try next? |
| PASS | PASS | `still_passing` | Did the patch interact with this scenario at all? |

## Workflow

### 1. Analyze history

**MANDATORY first step.** Run the `history-analysis` skill to build a
structured understanding of the full evolution history. Do NOT skip this
step or truncate `evo-dag show history` with `head`/`tail`. The skill
provides the methodology for reading the complete history without
truncation. Focus on:
- Historical context for each scenario in the mini-batch
- Pattern evolution — emerging patterns not yet captured in lessons
- Dev score attribution — which patches improved/hurt dev accuracy and why

This context is essential for writing meaningful reflections — you need to
know what was tried before to explain why this iteration's results differ,
and to write `--prevention-or-next` guidance that adds new information
rather than repeating existing lessons.

### 2. Read traces (by priority)

Before traces are in `{before_output_dir}`.
After traces are in `{train_after_cycle_dir}`.
See CLAUDE.md's **Iteration Output Structure** section for the exact file
layout within each output directory.

Compare before and after traces to understand what the patch changed in
the agent's behavior.

**Priority order**:
1. **Regressed** — always read both before and after traces. You must
   understand what behavior was correct before and what the patch broke.
2. **Still-failing (targeted)** — read to check for partial progress. Did
   the failure point shift? Is the agent closer to correct behavior?
3. **Fixed** — skim the after trace to confirm the patch resolved the root
   cause as intended, not through a lucky side effect.
4. **Still-passing** — skip unless the scenario uses the same tools or
   components you modified.

### 3. Diagnose state-changed scenarios (MANDATORY)

**For every regressed AND fixed scenario**, run the full `diagnose` skill
to determine whether the state change was **truly caused by the patch** or
is a **stochastic artifact** (caused by LLM non-determinism).

LLM non-determinism means the same agent code can produce different
tool-call sequences across runs. This affects both directions:
- A **regressed** scenario (PASS→FAIL) may have nothing to do with the
  patch — the agent simply took a different reasoning path.
- A **fixed** scenario (FAIL→PASS) may not be a real fix — the agent may
  have succeeded by luck, not because of the patch. Recording this as a
  true fix pollutes `good_patterns` with false signal.

Accurate causal attribution is critical: false regressions accumulate
noise in `bad_patterns`, and false fixes accumulate noise in
`good_patterns`. Both degrade the quality of lessons for future iterations.

**How to distinguish true vs. stochastic state changes:**

1. **Read the before trace**: Identify the exact tool-call sequence and
   reasoning steps that led to the before result.
2. **Read the after trace**: Identify where the agent's behavior diverged.
3. **Check the code diff** (`evo-dag show edge {base_parent_idx} {candidate_idx}`):
   Does the diff touch any code path, prompt text, hook, or tool that the
   scenario exercises? If the diff is completely unrelated to the
   scenario's tool usage and divergence point, it is likely stochastic.
4. **Check the divergence point**: Is the behavioral change at a step that
   the patch modified (true causation) or at an unrelated step where the
   agent simply made a different LLM-driven choice (stochastic)?

**Classification criteria:**
- **True (regression or fix)**: The diff modifies code/prompt/hook that
  the scenario directly exercises, AND the after trace shows the agent
  behaving differently at the modified point in a way that explains the
  outcome change.
- **Stochastic**: The diff does NOT touch anything the scenario exercises,
  OR the agent's divergence point is unrelated to the patch (e.g.,
  different search query phrasing, different email wording).
- **Uncertain**: The diff touches a shared component but the causal link
  is unclear. Record as uncertain with specific evidence.

**Do NOT** attribute any state change to "LLM non-determinism" or
"stochastic noise" without performing the above analysis. Every
classification must cite:
- The specific divergence point in the traces
- Whether the diff touches the relevant code path
- The evidence for or against a causal link

Record the classification in `--explanation` and `--prevention-or-next`.
For true regressions, explain what the patch broke. For stochastic
regressions, note the evidence so future iterations don't over-correct.
For true fixes, explain the causal chain from patch to fix. For stochastic
fixes, note the evidence so future iterations don't over-rely on the
patch strategy.

### 4. Inspect code changes

Run `evo-dag show edge {base_parent_idx} {candidate_idx}` to see the code
diff, files changed, and per-scenario impacts. Cross-reference the diff
with the trace behavior to understand causality.

For deeper inspection, read source files directly in the current worktree
`{worktree_path}`.

### 5. Record reflections

Use `evo-dag update-reflection` for **every** scenario in the mini-batch.

**Fields**:
- `--root-cause`: The underlying reason the scenario fails (independent of
  the patch). What is the agent doing wrong and why?
- `--explanation`: For `fixed` — how the patch resolved it. For
  `still_failing` — why the patch did NOT work (what was insufficient).
  For `regressed` — what the patch broke and how. For `still_passing` —
  why the patch did not interfere (only for scenarios using modified
  components).
- `--prevention-or-next`: What to try next, or what to avoid. This is
  critical for all non-passing outcomes — it creates the lessons that
  future iterations rely on.
- `--generalization-note`: How this scenario's outcome relates to
  development set accuracy. Did the patch generalize beyond the
  mini-batch? Record when dev scores are available.

**Per outcome**:

**`fixed`** — You MUST have completed the diagnosis step (step 3) before
recording this reflection. Classify as true fix or stochastic artifact.
```bash
# True fix (patch caused the success):
evo-dag update-reflection \
  --node {candidate_idx} \
  --scenario "<id>" --status "fixed" \
  --root-cause "Agent called X with wrong param Y because docstring said Z." \
  --explanation "TRUE FIX: Patch corrected docstring to clarify Y. Agent now calls correctly at step 4 — directly caused by diff in tools/email.py L42." \
  --prevention-or-next "For similar tool-misuse scenarios, check docstring accuracy first."

# Stochastic artifact (not caused by the patch):
evo-dag update-reflection \
  --node {candidate_idx} \
  --scenario "<id>" --status "fixed" \
  --root-cause "Agent happened to choose correct search query in after trace." \
  --explanation "STOCHASTIC: Diff only touches calendar tool. This scenario uses search tool exclusively. Agent succeeded because it picked a better search query by chance at step 2 — unrelated to patch." \
  --prevention-or-next "Not a reliable fix — scenario may fail again. Root cause (weak search strategy) still needs addressing."
```

**`regressed`** (highest priority) — You MUST have completed the
diagnosis step (step 3) before recording this reflection. Classify as
true regression or stochastic artifact with evidence.
```bash
# True regression (patch caused the failure):
evo-dag update-reflection \
  --node {candidate_idx} \
  --scenario "<id>" --status "regressed" \
  --root-cause "This scenario relied on optional param Y in tool X." \
  --explanation "TRUE REGRESSION: Docstring change removed note about Y being optional. Agent stopped passing Y. Divergence at step 5 where agent no longer passes Y — directly caused by diff in tools/email.py L42." \
  --prevention-or-next "When modifying docstrings, preserve param optionality annotations."

# Stochastic artifact (not caused by the patch):
evo-dag update-reflection \
  --node {candidate_idx} \
  --scenario "<id>" --status "regressed" \
  --root-cause "Agent used different search query phrasing in after trace." \
  --explanation "STOCHASTIC: Diff only touches calendar tool docstring. This scenario uses email tool exclusively. Agent diverged at step 3 with different query wording — unrelated to patch." \
  --prevention-or-next "No action needed — stochastic noise. Do not over-correct for this scenario."
```

**`still_failing`** — Explain why the patch was insufficient. What should
the next iteration try instead?
```bash
evo-dag update-reflection \
  --node {candidate_idx} \
  --scenario "<id>" --status "still_failing" \
  --root-cause "Agent uses absolute dates instead of relative dates in messages." \
  --explanation "Prompt rule 'use relative dates' was too weak — agent still used 'Oct 22'. Partial progress: bullet formatting was fixed." \
  --prevention-or-next "Stronger rule needed. Consider PreToolUse hook on send_message for just-in-time reminder."
```

**`still_passing`** — For scenarios that use the same tools or components
you modified, explain which specific patch change could have caused
interference and why it didn't. This is valuable signal for understanding
patch safety.
```bash
evo-dag update-reflection \
  --node {candidate_idx} \
  --scenario "<id>" --status "still_passing" \
  --explanation "Uses send_message but passes because it sends user-provided verbatim content. Our rule only affects agent-composed messages."
```
For scenarios unrelated to the patch, use the batch command:
```bash
evo-dag update-reflection --node {candidate_idx} --batch-still-passing "id1,id2,..."
```

{generalization_workflow_step}

## Causal Attribution Checklist

Before finishing, verify each regressed AND fixed scenario's reflection:

1. **Classified**: Explicitly labeled as TRUE (REGRESSION/FIX) or STOCHASTIC
2. **Evidence-based**: Cites the specific divergence point in traces
3. **Diff-linked**: States whether the diff touches the relevant code path
4. **Actionable**: True regressions have prevention guidance; true fixes
   explain the causal chain; stochastic cases explicitly state the
   evidence and implications

## Reflection Quality Checklist

Before finishing, verify each reflection meets these criteria:

1. **Specific**: References exact tool calls, steps, or code paths
2. **Causal**: Explains WHY the outcome occurred, not just WHAT happened
3. **Actionable**: `--prevention-or-next` gives clear guidance for future
   iterations — not generic advice like "try harder"
4. **Distinct**: Each reflection adds unique information (no copy-paste
   templates across scenarios)

## Constraints

- Record a reflection for **every** scenario in the mini-batch.
- Use `evo-dag update-reflection` — do NOT write reflections to files.
- Do NOT modify source code during reflection — Session 2 is analysis only.
- Do NOT read raw traces from development set output directories — only
  aggregate dev accuracy is available for generalization analysis.
- Do NOT read raw traces from development set dirs (`_val_` or `seed_val_`
  output dirs) — only aggregate accuracy is available.

**CRITICAL**: You MUST run `evo-dag update-reflection` for **every**
scenario in the mini-batch before finishing this session. Use individual
calls for fixed/regressed/still_failing scenarios and
`--batch-still-passing` for unaffected passing scenarios. Missing
reflections degrade the quality of accumulated lessons for future
iterations.
