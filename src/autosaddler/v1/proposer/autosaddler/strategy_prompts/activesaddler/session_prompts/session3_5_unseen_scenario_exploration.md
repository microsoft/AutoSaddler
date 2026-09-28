# Session 3.5: Unseen Scenario Exploration Decision

## Mandatory Skills

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize the skill — execute the full procedure described in
it. This skill is installed at `.claude/skills/<name>/SKILL.md` in the current
worktree.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `history-analysis` | Before the decision (Step 1) | Understand the Session 0-prepared harness and evolution history |

## Goal

Decide whether this iteration should exploit a known failure-pattern arm
(PULL) or explore unseen scenarios that may reveal a new failure type (DRAW).
Record exactly one decision with `pattern decide`.

This is an exploration-exploitation trade-off in **arm space**:

- **PULL** spends this iteration on an already-discovered pattern. Session 4
  then scores every known pattern so the sampler can choose an arm to repair.
- **DRAW** probes scenarios that have never been executed. This may reveal a
  failure pattern not yet represented by the known arms; Session 4 is skipped.

There is **no fixed formula or threshold** for this decision. Do not decide
mechanically from `|P_t|`, `|U_t|`, Activity, or any single statistic. Weigh all
available evidence and make one holistic decision.
Here, `P_t` is the set of failure patterns discovered by iteration `t`, and `U_t` is the set of scenarios still unseen at that iteration; `|...|` denotes the size of a set.

This is a **read-only analysis** session: do NOT modify the codebase and do NOT
rate arms.

## Context

- **Iteration**: {iteration}
- **Session 0-prepared harness**: C{prepared_candidate_idx}
- **Prepared worktree**: `{prepared_worktree_path}`
- **Prepared commit**: `{prepared_commit}`
- **Provisional parent**: C{provisional_parent_idx} (`{provisional_parent_commit}`)
- **Session 0 JSON**: `{selection_session_json_path}`
- **Session root**: `{session_root}`
- **Discovered failure patterns**: |P_t| = {num_arms}
- **Unseen scenarios remaining**: |U_t| = {unseen_pool_size}

### Prepared Harness Summary

{prepared_harness_summary}

## Candidate Failure Patterns (known arms)

Use the patterns below to judge whether a worthwhile known target exists.

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

Use this context to judge whether known-arm evidence still applies to the
prepared harness and whether its weaknesses remain reachable by another patch.

### 2. Review prior pulls for each candidate arm

Read the histories of all arms that have previously been selected:

```bash
pattern history
```

With no `--pattern-id`, the command shows every historically pulled arm. If the
full output is too large for initial triage, narrow it with
`pattern history --pattern-id <id1> <id2>` or apply `--last-k <K>`
separately to each selected arm. These are convenience filters, not substitutes
for complete review when older attempts could change the PULL/DRAW decision.

The output distinguishes patched attempts, all-pass skips, and failed attempts
and includes patch approaches, dev-set impact, and per-scenario reflections.

### 3. Inspect promising known patterns

Use `pattern show <pattern_id>` for any pattern that appears impactful. Read its
label, tagged scenarios, root-cause evidence, and observations. You do not need
to assign a learning-progress score in this session; make a coarse judgment of
whether at least one strong known target exists.

### 4. Weigh the PULL/DRAW considerations

Use all two lenses below. They guide attention but do not form a formula.

1. **Is there a clearly worthwhile known pattern to fix?**
   A severe, plausibly fixable, broad pattern argues for PULL. If known patterns
   look near-resolved, unreachable, or exhausted by repeated low-yield attempts,
   diminishing returns argue for DRAW.
2. **How complete is coverage of the failure surface?**
   Few discovered arms and a large unseen pool suggest that known patterns do
   not yet represent the harness's weaknesses. A small or exhausted unseen pool
   leaves little discovery value and favors PULL.

### 5. Make and record one decision

The discovery value of DRAW is a prediction: unseen scenario contents are not
known. Do not over-explore late when a strong fix is available, and do not
over-exploit early when the failure surface is poorly mapped.

Record exactly one action:

```bash
pattern decide --action pull \
  --rationale "why a known pattern is the best use of this iteration"
# OR
pattern decide --action draw \
  --rationale "why discovery is more valuable than fixing known patterns now"
```

Do NOT call `pattern rate` in this session.

## What Happens Next

- After PULL, Session 4 (arm scoring) runs on the same prepared harness and scores every arm.
- After DRAW, Session 4 (arm scoring) is skipped and the outer loop samples unseen scenarios.