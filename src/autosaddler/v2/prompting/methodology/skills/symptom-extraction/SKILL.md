---
name: symptom-extraction
description: "Use to write one bias-free candidate symptom per failure before looking at existing failure patterns."
---

# Symptom Extraction (Core)

## Purpose (Core)

Generate a candidate symptom label for each failure from its root cause and
behavioral evidence without seeing existing patterns. Showing an existing
category list first invites force-fitting new failures into old categories or
locking in a premature ontology; normalization handles matching afterwards.

## Gather Inputs (Core)

For each failure in the session context:

1. Read the diagnosis and the reflection lessons that cite the case.
2. For a pre-patch failure, read the case's pre-patch training evidence to
	 find where behavior first diverged from the task.
3. For a post-patch failure, compare its pre-patch and post-patch evidence
	 and decide whether the original cause remains or the patch introduced a
	 new regression mechanism.

## Write The Candidate Symptom (Core)

Ask what general behavioral issue caused the failure and whether a different
case with a different surface request would hit the same problem. The label
must be:

1. **more abstract than the root cause**: it names the class of failures, not
	 this case's details;
2. **actionable by one change**: one coherent harness change could plausibly
	 repair every failure with this symptom;
3. **case-independent**: no dates, names, or task-specific values;
4. **distinguishable**: failures that share the symptom share a root-cause
	 pattern.

"Agent produces a wrong answer" is too abstract; "a specific request returns
the wrong count" is too specific. "Singular/plural ambiguity in references
causes an incorrect count of matched entities" is at the right level.

## Record Every Candidate (Core)

Append each candidate to `symptom_candidates.md` in the workspace root before
starting normalization:

```text
Case: <case_id>
Source: <pre_patch or post_patch>
Root cause (preserved): <root cause>
Candidate symptom: <label>
Rationale: <why this abstraction level is right>
```
