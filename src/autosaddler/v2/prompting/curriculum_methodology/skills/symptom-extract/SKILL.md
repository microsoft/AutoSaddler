---
name: symptom-extract
description: "Use to generate one candidate symptom label per failed scenario from its root cause or reflection analysis, without seeing existing failure patterns."
---

# Skill: Symptom Extract (Core)

## Purpose (Core)

Generate a **candidate symptom label** from a single failure's root cause
or reflection analysis — WITHOUT seeing existing patterns.

This skill deliberately avoids showing existing patterns to prevent
**anchoring bias**: when shown a list of existing categories, LLMs tend to
force-fit new observations into existing ones (over-merge), or lock in a
premature ontology that prevents discovering better categorizations.

The normalization step (separate skill) handles matching against existing
patterns after extraction is complete.

## Procedure (Core)

For each failed scenario, perform these steps:

### Step 1: Gather inputs (Core)

Read the relevant analysis for this failure using the sources below.

#### For pre-patch failures (diagnosed in Session 1) (Core)

The root cause was identified during Session 1's diagnosis step. Access it
via one of these methods (in order of convenience):

1. **Session context** (preferred): `.autosaddler/session_context.json`
   - **Diagnosis** (`diagnosis`): The root cause analysis written by Session 1
   - **Patch intent** (`patch_intent`): The patch strategy Session 1 applied

   For the full history including diagnosis text, read the iteration files
   under `.autosaddler/history/iterations/`.

2. **Agent execution trace** (for deeper behavioral analysis):
   the scenario's record in the pre-patch training evidence.
   The full tool-call trace showing where the agent diverged from correct
   behavior. Use this to understand the *behavioral mechanism* of failure.

3. **Evaluation rationale**:
   the scenario's scores and judge rationale in the same evidence record.
   Shows what was expected vs what the agent produced.

#### For post-patch failures (analyzed in Session 2) (Core)

The reflection analysis explains why the scenario still fails or regressed
after the patch. Access it via:

1. **Session context** (preferred): `post_patch_failures` in
   `.autosaddler/session_context.json` shows per-scenario status changes
   (still_failing, regressed), and `lessons` holds Session 2's lessons.

   For the scenario's history across iterations, read its file under
   `.autosaddler/curriculum/cases/` (paths in `cases.json`), including earlier
   root causes and the lessons that cite it.

2. **Post-patch agent trace** (for behavioral comparison):
   the scenario's record in the post-patch training evidence.
   Compare with the pre-patch record to see what changed.

> **Tip**: Start with the session context for a structured summary,
> then drill into trace records only if you need more behavioral detail for
> your symptom extraction.

### Step 2: Identify the behavioral pattern (Core)

Ask yourself:
- What is the **general behavioral issue** that caused this failure?
- Would a different scenario with a different surface question hit the
  same underlying problem?
- Is this a problem with tool usage, reasoning, prompt interpretation,
  or something else?

The symptom should describe the **type of failure**, not the specific
scenario details.

### Step 3: Generate the candidate symptom (Core)

Write a concise symptom label that satisfies these conditions:

1. **More abstract than the root cause**: The root cause describes why
   *this specific scenario* failed. The symptom describes the *class of
   failures* this belongs to.

2. **Actionable via a single patch**: A well-designed harness patch should
   be able to address this symptom. If the symptom is too broad (e.g.,
   "agent makes mistakes"), it's not actionable.

3. **Scenario-independent language**: Do not reference specific dates,
   names, or task details from the scenario. Use general terms.

4. **Distinguishable**: If two failures have the symptom, they should
   share a root cause pattern. If they don't, the symptom is too broad.

### Step 4: Format the output (Core)

Write all extracted candidate symptoms to `symptom_candidates.md` in the
working directory root. For each failure, use this format:
```
Scenario: <case_id>
Source: <"pre_patch" or "post_patch">
Root cause (preserved): <original root cause text>
Candidate symptom: <YOUR SYMPTOM LABEL>
Rationale: <one sentence explaining why this abstraction level is right>
```

Complete `symptom_candidates.md` for every failed scenario before starting the `symptom-normalize` skill.

## Examples (Core)

### Example 1: Good extraction (Core)

```
Root cause: "The agent searched for 'November cancellations' but the
query matched both Nov 2 and Nov 6 events. The harness processes 'the
cancellation' as singular, creating only one entry instead of two."

Candidate symptom: "Singular/plural ambiguity in natural language
references causes incorrect count of matched entities"

Rationale: This generalizes beyond November cancellations to any case
where grammatical number creates ambiguity in entity matching.
```

### Example 2: Too specific (bad) (Core)

```
Candidate symptom: "November cancellation query returns wrong count"
Problem: References specific scenario details; won't match other
scenarios with the same underlying issue.
```

### Example 3: Too abstract (bad) (Core)

```
Candidate symptom: "Agent produces wrong answer"
Problem: Every failure could match this; not actionable for patching.
```

## Key Principle (Core)

**Trust your judgment on abstraction level.** The symptom should be at the
level where a single code change (prompt edit, tool fix, logic change)
could plausibly address all failures sharing this symptom.
