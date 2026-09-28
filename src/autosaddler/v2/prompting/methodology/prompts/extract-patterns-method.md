# Failure-Pattern Extraction Method (Core)

Pattern extraction turns the failures of a completed experiment into
symptom-level failure patterns. Patterns are the arms of the curriculum: each
arm is later scored by the learning progress expected from repairing it and
sampled to choose the next training batch. Begin with the `history-analysis`
skill, then read `.autosaddler/session_context.json` and
`.autosaddler/curriculum/manifest.json`.

## Process Every Failure (Core)

The session context lists `pre_patch_failures`, which failed before the patch,
and `post_patch_failures`, which still failed or regressed after it. Process
every listed failure. Use the diagnosis, the reflection lessons, matched
training scores, and the supplied training evidence to understand each
failure's root cause and behavioral mechanism.

## Extract Before Normalizing (Core)

1. Follow the `symptom-extract` skill for every failure without reading the
	 existing pattern registry. Write each candidate to
	 `symptom_candidates.md` in the workspace root before normalizing any of
	 them. Separating extraction from normalization avoids anchoring new
	 failures to old categories.
2. Follow the `symptom-normalize` skill: compare every candidate with every
	 pattern in `.autosaddler/curriculum/patterns.json` and decide whether it
	 is the same as an existing pattern, a composition of existing patterns, or
	 a new pattern.

## Pattern Quality (Core)

- Preserve each root cause as evidence; the pattern label is an abstraction
	that generalizes across cases.
- Keep patterns atomic: one coherent harness change should be able to repair
	one pattern. Express complex failures as compositions of atomic patterns.
- Do not over-merge failures with different root causes, and do not
	over-split surface variants of one behavioral issue.

## Return The Registry Update (Core)

Return every candidate in `symptoms`. Declare each new pattern once in
`new_patterns` with a short unique `key` and a descriptive symptom `label`.
Return one `tags` entry per processed failure: its `case_id`, its `source`
(`pre_patch` or `post_patch`), the `pattern_refs` it belongs to (existing
pattern IDs or keys of new patterns), and the preserved `root_cause`. Every
new pattern must be referenced by at least one tag. Do not compute
observations; the optimizer derives them from the returned tags.
