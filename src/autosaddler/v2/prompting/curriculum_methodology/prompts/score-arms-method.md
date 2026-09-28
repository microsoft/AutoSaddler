# Arm Scoring Method (Core)

For every failure-pattern arm listed in the output schema, estimate the
learning progress expected from allocating the next experiment to that arm:
the improvement in the working parent's population performance if its cases
were evaluated and the underlying weakness were repaired. This is read-only
analysis: do not modify the workspace. Begin with the `history-analysis`
skill, then follow the `progress-scoring` skill for every arm.

## Gather Evidence (Core)

1. Read `.autosaddler/session_context.json` for the working parent, its
	 selected parent, composition sources, and selection rationale, and use the
	 history bundle to read what the working parent changed.
2. Read `.autosaddler/curriculum/patterns.json` and every arm's pattern detail
	 for its label, cases, root-cause evidence, and observations.
3. Read every arm's complete pull history before rating it when older attempts
	 could change the judgment. Pull history, including development impact, is
	 the primary evidence of realized progress.

## Rate Four Axes (Core)

Rate `severity`, `fixability`, `breadth`, and `side_effect` in [0, 1] for each
arm. The optimizer computes the arm score as
`(severity + fixability + breadth + (1 - side_effect)) / 4`; your per-axis
ratings, not a separate holistic judgment, determine it. `side_effect` is a
risk: higher is worse.

Judge progress on the held-out distribution, not a single training batch. A
fix that repaired training cases but lowered development accuracy leaves
progress available. A genuinely resolved or intrinsically unfixable pattern
offers little progress. Repeated passes lower the score but should not force
it to zero, because passes can be intermittent.

## Return Every Score (Core)

Return exactly one entry per arm in `scores` with all four axes and a short
`rationale`. A missing or duplicate arm invalidates the session.
