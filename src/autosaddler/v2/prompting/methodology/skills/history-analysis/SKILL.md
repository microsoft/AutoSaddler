---
name: history-analysis
description: "Use to systematically analyze the full evolution history without output truncation and produce a focused summary for selection, diagnosis, or reflection."
---

# History Analysis (Core)

## Purpose (Core)

Use this skill first in selection, diagnosis, and reflection. The canonical
history is a complete event-derived navigation surface, not a terminal dump to
truncate. It prevents repeated failed approaches, accidental loss of durable
improvements, and decisions based only on the latest candidate.

## Read The Canonical Bundle (Core)

1. Read `.autosaddler/history/manifest.json` and verify its event boundary,
	 entry points, limits, and explicit omissions.
2. Read `summary.json` for accepted lineage, candidate statuses, measured
	 development aggregates, changed-unit summaries, and detail paths.
3. Read `current_batch.json`, then follow relevant case and iteration paths for
	 the current training batch.
4. Read `lessons.json` for recorded good and bad patterns from prior
	 iterations.
5. Open candidate details and diffs only when the current decision depends on
	 them. Read each selected artifact completely within its declared limits.

Do not search arbitrary run files for development or test records beyond the
aggregate fields supplied by the bundle.

## Classify Prior Changes (Core)

Classify relevant changes as:

- **durable improvement**: a measured gain with a plausible mechanism that
	survives later evaluation or descendants;
- **fragile improvement**: a gain observed once, with weak attribution or
	later instability;
- **regression**: a change linked to previously working behavior becoming
	worse;
- **ineffective attempt**: a measured change that did not repair its target;
- **rejected mutation**: useful verifier or contract evidence, but never a
	candidate outcome;
- **uncertain**: insufficient, conflicting, or stochastic evidence.

Track repeated failed approaches, regression-prone units, durable changes that
must be preserved, and complementary strengths across accepted candidates.
Distinguish lineage health from a single score: a high-scoring branch may still
carry unresolved regressions or coupled changes.

## Session Focus (Core)

- **Evolution**: compare selectable accepted parents, durable gains,
	regressions, compatible units, and whether composition is attributable.
- **Diagnosis**: find prior attempts on the current failure pattern, proven or
	failed interventions, and units whose modification has broad impact.
- **Reflection**: determine what the current result adds beyond existing
	lessons and whether similar outcomes have been durable or stochastic.

Produce a focused assessment that directly supports the current session. Do
not restate the entire history when a smaller evidence-backed comparison is
sufficient.