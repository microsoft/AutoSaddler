# Arm Decision Method (Core)

This session decides whether the next experiment exploits a known
failure-pattern arm (`pull`) or explores never-executed training cases that
may reveal a new failure pattern (`draw`). It is read-only analysis: do not
modify the workspace and do not score arms. Begin with the `history-analysis`
skill, then read `.autosaddler/session_context.json` and
`.autosaddler/curriculum/manifest.json`.

## Understand The Working Parent (Core)

The session context names the working parent candidate prepared by the
evolution session, its selected parent, any composition sources, and the
selection rationale. Use the history bundle to judge whether evidence about
known arms still applies to this harness and whether its weaknesses remain
reachable by another patch.

## Weigh Pull Against Draw (Core)

There is no fixed formula or threshold. Do not decide mechanically from the
number of known arms, the number of unseen cases, activity, or any single
statistic. Consider both lenses together:

1. **Is there a clearly worthwhile known pattern?** A severe, plausibly
	 fixable, broad pattern argues for `pull`. Patterns that look resolved,
	 unreachable, or exhausted by repeated low-yield attempts argue for `draw`.
2. **How complete is coverage of the failure surface?** Few known arms and a
	 large unseen pool suggest the registry does not yet represent the harness's
	 weaknesses. A small or exhausted unseen pool leaves little discovery value.

Read the complete pull history of any arm that could change the decision:
patched attempts, all-pass skips, failed attempts, and development impact.
The value of a draw is a prediction because unseen case contents are unknown.
Do not over-explore late when a strong fix is available, and do not
over-exploit early when the failure surface is poorly mapped.

## Return One Decision (Core)

Return exactly one `action` and a `rationale` that cites the evidence. After a
`pull`, a scoring session rates every arm and the optimizer samples one arm by
its score. After a `draw`, scoring is skipped and never-executed cases are
sampled. A `draw` with no unseen cases left becomes a `pull`.
