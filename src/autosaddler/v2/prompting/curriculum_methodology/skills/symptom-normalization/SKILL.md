---
name: symptom-normalization
description: "Use after symptom extraction to match every candidate against all existing failure patterns as same, composition, or new."
---

# Symptom Normalization (Core)

## Purpose (Core)

Decide for every extracted candidate whether it corresponds to an existing
pattern, is a composition of existing patterns, or is genuinely new. This is
the only step that reads the existing pattern registry.

## Procedure (Core)

1. Read `symptom_candidates.md` completely. Normalize every candidate without
	 rewriting or dropping any.
2. Read `.autosaddler/curriculum/patterns.json` and review every pattern
	 regardless of activity; a low-activity pattern may still describe the same
	 mechanism. Open a pattern detail file when its label alone is ambiguous.
3. Classify each candidate:
	 - **same**: the same failure mechanism at the same abstraction level; one
		 change would fix both. Tag the existing pattern ID.
	 - **composition**: two or more existing patterns independently contribute
		 and the combination is not a new failure mode. Tag every contributing
		 existing pattern ID; do not create a composite pattern.
	 - **new**: no existing pattern describes the mechanism and it does not
		 decompose into existing patterns. Declare a new pattern and tag it.

## Decision Guidelines (Core)

- Between same and new, prefer same when you are at least moderately
	confident the mechanism matches: over-splitting fragments the curriculum
	signal.
- Do not force a clearly different mechanism into an existing pattern because
	the registry is small: hidden distinct issues are worse in the long run.
- Use composition sparingly. Multiple symptoms usually share one root cause.
