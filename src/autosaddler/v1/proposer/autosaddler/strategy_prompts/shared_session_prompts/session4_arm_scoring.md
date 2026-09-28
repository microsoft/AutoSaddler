# Session 4: Arm Scoring (Curriculum)

## Mandatory Skills

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize any skill — execute the full procedure described
in each one. These skills are installed at `.claude/skills/<name>/SKILL.md`
in the current worktree.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `history-analysis` | Before any scoring (Step 1) | Understand the Session 0-prepared harness and evolution history |
| 2 | `progress-scoring` | For every candidate pattern (Steps 3–5) | Estimate each arm's cardinal learning-progress score |

## Goal

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
the pattern’s severity, fixability, breadth, and risk of side effects. Record
every cardinal score with `pattern rate`.

This is a **read-only analysis** session: do NOT modify the codebase. Only
inspect the harness, failure patterns, supporting scenarios, and optimization
history, then record the scores.

## Context

- **Iteration**: {iteration}
- **Session 0-prepared harness**: C{prepared_candidate_idx}
- **Prepared worktree**: `{prepared_worktree_path}`
- **Prepared commit**: `{prepared_commit}`
- **Provisional parent**: C{provisional_parent_idx} (`{provisional_parent_commit}`)
- **Session 0 JSON**: `{selection_session_json_path}`
- **Session root**: `{session_root}`

### Prepared Harness Summary

{prepared_harness_summary}

## Candidate Failure Patterns (arms to score)

Score **every** pattern below.

{candidate_patterns}

### Candidate Table Columns

- `Pattern`: Stable failure-pattern ID used by the `pattern` and `evo-dag` CLIs.
- `Activity`: Rested EMA of the raw observations, seeded at `1.00`. This is a
  mechanical severity reference, not the Agent learning-progress score φ.
- `Observations`: Complete per-iteration observation history. Each entry uses
  `iteration:tagged/evaluated (active=...), tagged=[...], evaluated=[...]`.
  `evaluated` lists the scenarios associated with this pattern that were
  evaluated in that iteration's mini-batch, and `tagged` lists the subset that
  was still tagged with the pattern after patching. `active` is the corresponding
  `tagged/evaluated` fraction. For legacy observations whose scenario IDs were
  not recorded, the counts appear as `?/?` and both lists as `(not recorded)`;
  `(none)` means the arm has not been selected again since the pattern was most
  recently observed in a scenario, so no subsequent activity observation has
  been recorded.
- `#Scen`: Number of unique scenarios currently tagged with the pattern.
- `LastObs`: Most recent iteration in which the arm was selected and an activity observation was recorded, or `0` if the arm has not been selected again since the pattern was most recently observed in a scenario.
- `Label`: Symptom-level failure-pattern description.
- `Scenarios`: Complete list of scenario IDs currently owned by the arm.

## Workflow
### 1. Understand the current harness (history-analysis)

Run the `history-analysis` skill. Inspect the current prepared node with
`evo-dag show node {prepared_candidate_idx}` and its provisional parent with
`evo-dag show node {provisional_parent_idx}`. Read the actual Session 0 change
with `git diff {provisional_parent_commit} {prepared_commit}`.
You need this context to assess all four scoring axes: *severity* (whether the failure pattern is likely still active under the prepared harness), *fixability* (whether and where a patch can address it), *breadth* (how broadly a successful repair may generalize), and *side-effect risk* (which components or behaviors a patch may affect or regress).


### 2. Review prior pulls for each candidate arm

Read the histories of all arms that have previously been selected:

```bash
pattern history
```

With no `--pattern-id`, the command shows every historically pulled arm. If the full output is too large to review efficiently, you may optionally narrow it to a subset in one call with `pattern history --pattern-id <id1> <id2>`, or use `--last-k <K>` to show only the K most recent entries for each selected arm during initial triage. These options are convenience filters, not substitutes for complete review: inspect an arm's full history before rating it whenever older attempts could change your judgment. The output distinguishes patched attempts, all-pass skips, and failed attempts and includes diagnoses, patch intents, dev-set impact, and per-scenario reflections.

### 3. Inspect each candidate pattern

For every pattern in the table:
- `pattern show <pattern_id>` — label, tagged scenarios, evidence (root causes).
- `evo-dag show scenario <sid>` — for each of the pattern's scenarios, read its
  failure history and relate it to the current harness change.

### 4. Estimate learning progress (progress-scoring)

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

### 5. Record every score

> **Reminder**: φ is computed from your four axes (see step 4). `--side-effect` is a RISK — higher means worse.

For each candidate pattern:
```bash
pattern rate --pattern-id <id> \
  --severity <s> --fixability <f> --breadth <b> --side-effect <e> \
  --rationale "..."
```
**CRITICAL**: rate every pattern in the table before finishing. An un-rated
pattern is treated as **low priority (score 0.0)** — if you skip one it gets
deprioritized rather than reflecting your judgment, so do not leave any
un-rated.

## What Happens Next

The outer loop samples one pattern with probability proportional to `softmax(φ / τ)` and sends its scenarios to Sessions 1 (Diagnose Failures + Apply Patches).
