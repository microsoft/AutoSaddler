# Session 4: Arm Scoring (Curriculum) (Core)

## Mandatory Skills (Core)

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize any skill — execute the full procedure described
in each one. These skills are supplied with this session.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `history-analysis` | Before any scoring (Step 1) | Understand the Session 0-prepared harness and evolution history |
| 2 | `progress-scoring` | For every candidate pattern (Steps 3–5) | Estimate each arm's cardinal learning-progress score |

## Goal (Core)

For every **candidate failure-pattern arm**, estimate a cardinal
learning-progress score φ ∈ [0, 1] representing the expected improvement in
the prepared harness’s population performance from allocating the next
optimization step to that arm. The estimate should reflect the likely outcome
of executing the arm’s supporting scenarios and attempting to address the
harness weakness represented by the pattern, including both the potential
performance gain and the probability of obtaining a broadly effective,
dev-preserving update.

Base each score on the prepared harness, the arm’s complete execution and
optimization history, prior dev-set outcomes, and the available evidence about
the pattern’s severity, fixability, breadth, and risk of side effects. Return
every cardinal score in `scores`.

This is a **read-only analysis** session: do NOT modify the codebase. Only
inspect the harness, failure patterns, supporting scenarios, and optimization
history, then return the scores.

## Context (Core)

Read `.autosaddler/session_context.json`:

- **Iteration**: `iteration`
- **Session 0-prepared harness**: `candidate_ids[0]` (the working parent)
- **Provisional parent**: `selected_parent_candidate_id`, with the Session 0
  plan in `selection_parent_ids`, `component_sources`, and `selection_rationale`

## Candidate Failure Patterns (arms to score) (Core)

Score **every** pattern in `task_selection.arm_ids`. Their registry entries are
in `.autosaddler/curriculum/patterns.json`.

### Candidate Registry Fields (Core)

- `pattern_id`: Stable failure-pattern ID used by the registry files.
- `activity`: Rested EMA of the raw observations, seeded at `1.00`. This is a
  mechanical severity reference, not the Agent learning-progress score φ.
- `observations`: Complete per-iteration
  observation history. Each entry has `iteration`, `evaluated_case_ids`,
  `tagged_case_ids`, and `active`.
  `evaluated_case_ids` lists the scenarios associated with this pattern that were
  evaluated in that iteration's mini-batch, and `tagged_case_ids` lists the subset that
  was still tagged with the pattern after patching. `active` is the corresponding
  `tagged/evaluated` fraction. An empty list
  means the arm has not been selected again since the pattern was most
  recently observed in a scenario, so no subsequent activity observation has
  been recorded.
- `num_cases`: Number of unique scenarios currently tagged with the pattern.
- `last_observed_iteration`: Most recent iteration in which the arm was selected and an activity observation was recorded, or `null` if the arm has not been selected again since the pattern was most recently observed in a scenario.
- `label`: Symptom-level failure-pattern description.
- `case_ids`: Complete list of scenario IDs currently owned by the arm.

## Workflow (Core)
### 1. Understand the current harness (history-analysis) (Core)

Run the `history-analysis` skill. Inspect the current prepared candidate and
its provisional parent in `.autosaddler/history/candidates/`. Read the actual
Session 0 change in the history edges and diffs.
You need this context to assess all four scoring axes: *severity* (whether the failure pattern is likely still active under the prepared harness), *fixability* (whether and where a patch can address it), *breadth* (how broadly a successful repair may generalize), and *side-effect risk* (which components or behaviors a patch may affect or regress).


### 2. Review prior pulls for each candidate arm (Core)

Read the histories of all arms that have previously been selected:
`.autosaddler/curriculum/pull_history/<pattern_id>.json`.

Each file holds the complete pull history of one arm. If reading every file is too large to review efficiently, you may start from the most recent pulls of each arm during initial triage. This is a convenience, not a substitute for complete review: inspect an arm's full history before rating it whenever older attempts could change your judgment. The files distinguish patched attempts, all-pass skips, and failed attempts and include diagnoses, patch intents, dev-set impact, per-scenario results, and lessons.

### 3. Inspect each candidate pattern (Core)

For every pattern in the registry:
- `.autosaddler/curriculum/patterns/<pattern_id>.json` — label, tagged scenarios, evidence (root causes).
- `.autosaddler/curriculum/cases/<file>.json` (paths in the pattern's
  `case_history_paths`) — for each of the pattern's scenarios, read its
  failure history and relate it to the current harness change.

### 4. Estimate learning progress (progress-scoring) (Core)

Follow the `progress-scoring` skill. For each pattern, judge the four
considerations (severity, fixability, breadth, side-effect), then rate
**all four axes carefully** in [0, 1]. The final score φ is the mean of those
four axes with side-effect inverted —
`(severity + fixability + breadth + (1 - side_effect)) / 4`. Your per-axis
ratings (not a single holistic judgment) determine the result.

**Use the per-arm pull history as primary evidence, and weigh the dev-set, not
just the mini-batch.** A scenario merely passing the mini-batch in a recent pull
does NOT by itself mean the pattern is resolved — learning progress is measured
on the **dev/held-out** distribution:
- If a prior patch fixed the scenario but **dropped dev-set accuracy** (so it was
  rejected and reverted), a better dev-preserving fix is still needed → progress
  may remain **high**.
- If the scenario is genuinely resolved (passing with no dev headroom left), or
  its failure is intrinsically unfixable without regressing dev → progress is
  **low**.
- If recent pulls show the scenarios repeatedly passing, lower the score, **but
  do not force it to 0** — the pass may be intermittent, and the sampler's
  minimum-probability floor still re-checks low-scored arms periodically to catch
  silent regressions.

### 5. Record every score (Core)

> **Reminder**: φ is computed from your four axes (see step 4). `side_effect` is a RISK — higher means worse.

For each candidate pattern, return one `scores` entry with its `pattern_id`,
`severity`, `fixability`, `breadth`, `side_effect`, and a `rationale`.

**CRITICAL**: rate every pattern in `task_selection.arm_ids` exactly once before
finishing. A missing or duplicate pattern invalidates the scores and the
session is retried, so do not leave any un-rated.

## What Happens Next (Core)

The outer loop samples one pattern with probability proportional to `softmax(φ / τ)` and sends its scenarios to Sessions 1 (Diagnose Failures + Apply Patches).
