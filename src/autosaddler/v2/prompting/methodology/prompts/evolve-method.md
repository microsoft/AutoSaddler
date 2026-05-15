# Candidate Selection And Composition Method (Core)

The evolution session chooses where the next measured experiment starts. It
does not mutate a candidate. Begin with the `history-analysis` skill and read
`.autosaddler/history/manifest.json`; use its entry points to inspect accepted
candidates, iteration outcomes, changed units, diffs, and lessons.

## Compare Candidates (Core)

For every plausible accepted parent, assess:

- measured training and development outcomes, including whether gains endured
	in descendants;
- accepted lineage and candidate status;
- the exact changed units and the failure patterns those changes targeted;
- fixed, regressed, unchanged, rejected, or uncertain outcomes;
- prior recorded good and bad patterns;
- compatibility with the current training batch and with other proposed
	source candidates.

Classify useful changes as durable, fragile, regression-prone, ineffective, or
uncertain. Candidate recency, lineage length, and patch volume are not quality
signals. Declined and rejected attempts may explain what to avoid, but they
are not selectable parents.

## Choose The Base (Core)

Default to one measured base parent. Put that accepted candidate first in the
schema-defined parent list. Prefer the base with the strongest relevant,
durable evidence and the fewest unresolved regressions, not merely the highest
single noisy measurement.

Switch away from the latest lineage when accumulated regressions or conflicting
assumptions make another accepted candidate a cleaner base. Explain the choice
using supplied candidate IDs and recorded evidence.

## Compose Only When Attributable (Core)

Use additional parents only when they contain complementary improvements that
can be attributed to selectable units in the supplied composition schema.
Before composing, check that:

1. each source candidate is accepted and explicitly available;
2. each requested unit exists in the schema for that source candidate;
3. the units have independent or compatible assumptions;
4. the composition preserves required companion changes;
5. no known regression is imported with the selected unit.

Do not invent candidate IDs, unit names, or line-level merge capabilities. If
the schema exposes only whole files or components, composition operates at
that granularity. When evidence is weak or changes are coupled, select one
base and leave composition empty.