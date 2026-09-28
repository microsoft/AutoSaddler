# Skill: Symptom Extract

## Purpose

Generate a **candidate symptom label** from a single failure's root cause
or reflection analysis — WITHOUT seeing existing patterns.

This skill deliberately avoids showing existing patterns to prevent
**anchoring bias**: when shown a list of existing categories, LLMs tend to
force-fit new observations into existing ones (over-merge), or lock in a
premature ontology that prevents discovering better categorizations.

The normalization step (separate skill) handles matching against existing
patterns after extraction is complete.

## Procedure

For each failed scenario, perform these steps:

### Step 1: Gather inputs

Read the relevant analysis for this failure using the sources below.

#### For pre-patch failures (diagnosed in Session 1)

The root cause was identified during Session 1's diagnosis step. Access it
via one of these methods (in order of convenience):

1. **evo-dag CLI** (preferred):
   ```bash
   evo-dag show node <candidate_idx>
   ```
   Look at the "Patch Intent" section which includes:
   - **Diagnosis**: The root cause analysis written by Session 1 (stored
     via `evo-dag update-intent --diagnosis "..."`)
   - **Targets**: Which scenarios the patch aimed to fix
   - **Approach**: The patch strategy applied

   For the full history including diagnosis text:
   ```bash
   evo-dag show history
   ```
   This shows the `Diagnosis:` field for each iteration's patch intent.

2. **proposer_reasoning.md** (Session 1 detailed reasoning):
   ```
   <worktree_path>/proposer_reasoning.md
   ```
   Session 1 writes its full diagnosis and patching rationale here.
   Contains the detailed root cause analysis, reasoning about why each
   scenario fails, and the justification for the chosen patch approach.
   This is the most comprehensive single-file source for understanding
   the diagnosis.

3. **Session 1 JSON log**:
   ```
   <before_output_dir>/iter<N>_c<X>_patch.json
   ```
   Contains the full Session 1 transcript including diagnosis reasoning,
   tool calls, and identified root causes in `raw_response`.

4. **Agent execution trace** (for deeper behavioral analysis):
   ```
   <before_output_dir>/run/lite/<scenario_id>.json
   ```
   The full tool-call trace showing where the agent diverged from correct
   behavior. Use this to understand the *behavioral mechanism* of failure.

5. **Evaluation rationale**:
   ```
   <before_output_dir>/run/output.jsonl
   ```
   Per-scenario scores and judge rationale (JSONL, one entry per scenario).
   Shows what was expected vs what the agent produced.

#### For post-patch failures (analyzed in Session 2)

The reflection analysis explains why the scenario still fails or regressed
after the patch. Access it via:

1. **evo-dag CLI** (preferred):
   ```bash
   evo-dag show node <candidate_idx>
   ```
   The "Patch Verdict" section shows per-scenario status changes
   (still_failing, regressed) with explanations.

   For detailed reflection text:
   ```bash
   evo-dag show scenario <scenario_id>
   ```
   Shows the full history of this scenario across iterations, including
   the Session 2 reflection with root cause, explanation, and next steps.

2. **Session 2 JSON log**:
   ```
   <after_output_dir>/iter<N>_c<X>_reflection.json
   ```
   Contains the full Session 2 transcript with reflection reasoning
   in `raw_response`.

3. **Post-patch agent trace** (for behavioral comparison):
   ```
   <after_output_dir>/run/lite/<scenario_id>.json
   ```
   Compare with the pre-patch trace at
   `<before_output_dir>/run/lite/<scenario_id>.json` to see what changed.

> **Tip**: Start with `evo-dag show node <idx>` for a structured summary,
> then drill into trace files only if you need more behavioral detail for
> your symptom extraction.

### Step 2: Identify the behavioral pattern

Ask yourself:
- What is the **general behavioral issue** that caused this failure?
- Would a different scenario with a different surface question hit the
  same underlying problem?
- Is this a problem with tool usage, reasoning, prompt interpretation,
  or something else?

The symptom should describe the **type of failure**, not the specific
scenario details.

### Step 3: Generate the candidate symptom

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

### Step 4: Format the output

Write all extracted candidate symptoms to `symptom_candidates.md` in the
working directory root. For each failure, use this format:
```
Scenario: <scenario_id>
Source: <"diagnosis" or "reflection">
Root cause (preserved): <original root cause text>
Candidate symptom: <YOUR SYMPTOM LABEL>
Rationale: <one sentence explaining why this abstraction level is right>
```

Complete `symptom_candidates.md` for every failed scenario before starting the `symptom-normalize` skill.

## Examples

### Example 1: Good extraction

```
Root cause: "The agent searched for 'November cancellations' but the
query matched both Nov 2 and Nov 6 events. The harness processes 'the
cancellation' as singular, creating only one entry instead of two."

Candidate symptom: "Singular/plural ambiguity in natural language
references causes incorrect count of matched entities"

Rationale: This generalizes beyond November cancellations to any case
where grammatical number creates ambiguity in entity matching.
```

### Example 2: Too specific (bad)

```
Candidate symptom: "November cancellation query returns wrong count"
Problem: References specific scenario details; won't match other
scenarios with the same underlying issue.
```

### Example 3: Too abstract (bad)

```
Candidate symptom: "Agent produces wrong answer"
Problem: Every failure could match this; not actionable for patching.
```

## Key Principle

**Trust your judgment on abstraction level.** The symptom should be at the
level where a single code change (prompt edit, tool fix, logic change)
could plausibly address all failures sharing this symptom.
