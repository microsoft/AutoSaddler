# Session 3.5: Unseen Scenario Exploration Decision (Core)

## Mandatory Skills (Core)

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize the skill — execute the full procedure described in
it. This skill is supplied with this session.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `history-analysis` | Before the decision (Step 1) | Understand the Session 0-prepared harness and evolution history |

## Goal (Core)

Decide whether this iteration should exploit a known failure-pattern arm
(PULL) or explore unseen scenarios that may reveal a new failure type (DRAW).
Return exactly one decision as `action` (`pull` or `draw`) with its `rationale`.

This is an exploration-exploitation trade-off in **arm space**:

- **PULL** spends this iteration on an already-discovered pattern. Session 4
  then scores every known pattern so the sampler can choose an arm to repair.
- **DRAW** probes scenarios that have never been executed. This may reveal a
  failure pattern not yet represented by the known arms; Session 4 is skipped.
  In a later draw epoch (`task_selection.draw_epoch` > 0), DRAW instead
  re-explores prior successes that did not instantiate an arm.

There is **no fixed formula or threshold** for this decision. Do not decide
mechanically from `|P_t|`, `|U_t|`, Activity, or any single statistic. Weigh all
available evidence and make one holistic decision.
Here, `P_t` is the set of failure patterns discovered by iteration `t`, and `U_t` is the set of scenarios still unseen at that iteration; `|...|` denotes the size of a set.

This is a **read-only analysis** session: do NOT modify the codebase and do NOT
rate arms.

## Context (Core)

Read `.autosaddler/session_context.json`:

- **Iteration**: `iteration`
- **Session 0-prepared harness**: `candidate_ids[0]` (the working parent)
- **Provisional parent**: `selected_parent_candidate_id`, with the Session 0
  plan in `selection_parent_ids`, `component_sources`, and `selection_rationale`
- **Discovered failure patterns**: |P_t| = `task_selection.num_arms`
- **Unseen scenarios remaining**: |U_t| = `task_selection.num_unseen`
- **Draw epoch**: `task_selection.draw_epoch` (0 while never-executed scenarios remain)

## Candidate Failure Patterns (known arms) (Core)

Use the patterns in `.autosaddler/curriculum/patterns.json` to judge whether a
worthwhile known target exists. `task_selection.arm_ids` lists the known arms.

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

Use this context to judge whether known-arm evidence still applies to the
prepared harness and whether its weaknesses remain reachable by another patch.

### 2. Review prior pulls for each candidate arm (Core)

Read the histories of all arms that have previously been selected:
`.autosaddler/curriculum/pull_history/<pattern_id>.json`.

Each file holds the complete pull history of one arm. If reading every file is
too large for initial triage, start from the most recent pulls of each arm.
This is a convenience, not a substitute
for complete review when older attempts could change the PULL/DRAW decision.

The files distinguish patched attempts, all-pass skips, and failed attempts
and include patch approaches, dev-set impact, per-scenario results, and lessons.

### 3. Inspect promising known patterns (Core)

Read `.autosaddler/curriculum/patterns/<pattern_id>.json` for any pattern that appears impactful. Read its
label, tagged scenarios, root-cause evidence, and observations. You do not need
to assign a learning-progress score in this session; make a coarse judgment of
whether at least one strong known target exists.

### 4. Weigh the PULL/DRAW considerations (Core)

Use all two lenses below. They guide attention but do not form a formula.

1. **Is there a clearly worthwhile known pattern to fix?**
   A severe, plausibly fixable, broad pattern argues for PULL. If known patterns
   look near-resolved, unreachable, or exhausted by repeated low-yield attempts,
   diminishing returns argue for DRAW.
2. **How complete is coverage of the failure surface?**
   Few discovered arms and a large unseen pool suggest that known patterns do
   not yet represent the harness's weaknesses. A small or exhausted unseen pool
   leaves little discovery value and favors PULL. In a later draw epoch the pool
   holds scenarios that passed before, so its discovery value comes from
   regressions or new failures the harness changes since then may have caused.

### 5. Make and record one decision (Core)

The discovery value of DRAW is a prediction: unseen scenario contents are not
known. Do not over-explore late when a strong fix is available, and do not
over-exploit early when the failure surface is poorly mapped.

Return exactly one action:

- `action: "pull"` with a `rationale` explaining why a known pattern is the
  best use of this iteration, OR
- `action: "draw"` with a `rationale` explaining why discovery is more
  valuable than fixing known patterns now.

Do NOT return arm scores in this session.

## What Happens Next (Core)

- After PULL, Session 4 (arm scoring) runs on the same prepared harness and scores every arm.
- After DRAW, Session 4 (arm scoring) is skipped and the outer loop samples unseen scenarios.
