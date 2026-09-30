---
name: symptom-normalize
description: "Use after symptom extraction to judge every candidate symptom against all existing failure patterns as same, composition, or new."
---

# Skill: Symptom Normalize (Core)

## Purpose (Core)

Determine whether a candidate symptom (extracted bias-free in the previous
step) corresponds to an **existing pattern**, is a **composition** of
existing patterns, or is a **genuinely new** pattern.

This is the only step where you see the existing pattern list. The
candidate symptom was already generated independently to avoid anchoring.
Now you perform a comparison-only judgment.

## Procedure (Core)

### Step 1: Read extracted candidates (Core)

Read `symptom_candidates.md` from the working directory root. This file is the
complete output of the preceding `symptom-extract` step. Normalize every
candidate symptom recorded in it; do not omit or rewrite candidates before
comparing them with existing patterns.

### Step 2: Retrieve existing patterns (Core)

Read `.autosaddler/curriculum/patterns.json` to view **all** currently
registered patterns. You must review every one, not just top-scored.

Each entry has these fields:

- `pattern_id`: Stable failure-pattern ID used by the registry files.
- `activity`: Rested EMA of the raw observations, seeded at `1.00`. This is a
  mechanical severity reference, not the Agent learning-progress score φ.
- `observations`: Complete per-iteration observation history. Each entry has
  `iteration`, `evaluated_case_ids`, `tagged_case_ids`, and `active`.
  `evaluated_case_ids` lists the scenarios associated with this pattern that were
  evaluated in that iteration's mini-batch, and `tagged_case_ids` lists the subset that
  was still tagged with the pattern after patching. `active` is the corresponding
  `tagged/evaluated` fraction. An empty list
  means the arm has not been selected again since the pattern was most
  recently observed in a scenario, so no subsequent activity observation has
  been recorded.
- `num_cases`: Number of unique scenarios currently tagged with the pattern.
- `last_observed_iteration`: Most recent iteration in which the arm was selected and an activity
  observation was recorded, or `null` if the arm has not been selected again since
  the pattern was most recently observed in a scenario.
- `label`: Symptom-level failure-pattern description.
- `case_ids`: Complete list of scenario IDs currently owned by the arm.

For more detail on a specific pattern (tagged scenarios, evidence), read
`.autosaddler/curriculum/patterns/<pattern_id>.json`.

To see which scenarios are already tagged to a pattern, read its `case_ids`.

> **Important**: Review ALL patterns regardless of activity. A low-activity
> pattern (recently fixed or rarely observed) may still describe the
> same failure mechanism as your candidate symptom. Activity is irrelevant
> for normalization — only the label and evidence matter.

### Step 3: For each candidate symptom, perform 3-way judgment (Core)

Compare the candidate symptom against the existing pattern list and
classify it into exactly one of three categories:

#### (a) Same — candidate matches an existing pattern (Core)

The candidate symptom describes the same underlying behavioral issue as an
existing pattern, even if the wording differs.

**Criteria for "same":**
- Both would be fixed by the same code change
- Both describe the same failure mechanism at the same abstraction level
- Surface wording differences are immaterial (e.g., "plural ambiguity" vs
  "singular/multiple confusion" — same thing)

**Action**: Link to the existing pattern ID.

#### (b) Composition — candidate is a combination of existing patterns (Core)

The failure involves multiple independent issues that already have their
own patterns. The scenario fails because of pattern A AND pattern B
together.

**Criteria for "composition":**
- You can identify 2+ existing patterns that independently contribute
- Fixing either one alone might not fix the scenario, but each is a
  recognized independent issue
- The combination does NOT represent a fundamentally new failure mode

**Action**: Tag with multiple existing pattern IDs. Do NOT create a new
composite pattern.

#### (c) New — genuinely novel failure type (Core)

The candidate describes a failure that does NOT match any existing pattern
and is NOT a composition of existing ones.

**Criteria for "new":**
- No existing pattern describes the same failure mechanism
- The candidate cannot be decomposed into existing patterns
- The failure is specific enough to be actionable (not too abstract)

**Action**: Register as a new pattern with a `new_patterns` entry.

### Step 4: Execute the decision (Core)

Based on your judgment, add the appropriate entries to the session output.

> **Note**: Every tag's `source` selects the evaluation (`pre_patch` or
> `post_patch`) of the current iteration, as listed in the session context.

**For (a) Same:** tag the (candidate, evaluation, scenario) tuple with the existing pattern.
```json
{"case_id": "<case_id>", "source": "<pre_patch or post_patch>",
 "pattern_refs": ["<existing_id>"],
 "root_cause": "Brief root cause text preserved as evidence"}
```

**For (b) Composition:** tag with multiple pattern IDs.
```json
{"case_id": "<case_id>", "source": "<pre_patch or post_patch>",
 "pattern_refs": ["<id1>", "<id2>"],
 "root_cause": "Composite: issue A interacts with issue B"}
```

**For (c) New:** register a new pattern, then tag with its key.
```json
{"new_patterns": [{"key": "<new_key>", "label": "DESCRIPTIVE SYMPTOM LABEL"}],
 "tags": [{"case_id": "<case_id>", "source": "<pre_patch or post_patch>",
           "pattern_refs": ["<new_key>"], "root_cause": "Root cause text..."}]}
```
The outer loop assigns the pattern ID of each new key.

> **You do NOT need to return observations** — observations are
> automatically derived by the outer loop from your tagging results.

**Verification** — after all tagging is complete, confirm that every
candidate in `symptom_candidates.md` has a tag and that every new pattern key
is referenced by at least one tag.

## Decision Guidelines (Core)

### Prefer "same" when in doubt between "same" and "new" (Core)

Over-splitting (creating too many patterns) fragments the signal and
prevents the sampler from building reliable per-pattern estimates.
If you're 60%+ confident it's the same issue, link to existing.

### Prefer "new" when in doubt between "same" and a forced match (Core)

Do NOT force a candidate into an existing pattern just because the list
is short. If the failure mechanism is clearly different, create a new
pattern. The cost of over-merge (hiding distinct issues under one label)
is worse than over-split in the long run.

### Use "composition" sparingly (Core)

Most failures have a single root cause. Composition is for cases where
two genuinely independent issues interact. Do not use composition just
because the scenario has multiple symptoms — those symptoms may share a
single root cause.

## Example Normalization (Core)

```
Existing patterns:
  abc123 — "Singular/plural ambiguity in entity count"
  def456 — "Missing pagination in search results"
  ghi789 — "Premature tool termination before validating output"

Candidate: "Agent counts only first-page results when multiple pages exist"
Judgment: (a) Same as def456 — both describe incomplete search due to
          missing pagination handling.
Action: tag with pattern_refs ["def456"]

Candidate: "Agent sees 'the event' but matches multiple events without
           disambiguation, then only retrieves first page of matches"
Judgment: (b) Composition of abc123 + def456 — ambiguity issue AND
          pagination issue both contribute.
Action: tag with pattern_refs ["abc123", "def456"]

Candidate: "Agent creates duplicate tool calls for the same action
           within a single turn"
Judgment: (c) New — no existing pattern describes redundant tool invocation.
Action: new_patterns entry with label "Redundant duplicate tool calls within
        single reasoning turn", then tag with its key
```
