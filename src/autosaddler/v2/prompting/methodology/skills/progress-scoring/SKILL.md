---
name: progress-scoring
description: "Use to rate four [0,1] learning-progress axes for each failure-pattern arm; the arm score is their fixed mean with side-effect risk inverted."
---

# Progress Scoring (Core)

## Target (Core)

For each arm, rate `severity`, `fixability`, `breadth`, and `side_effect` in
[0, 1]. The optimizer computes the learning-progress score
`(severity + fixability + breadth + (1 - side_effect)) / 4`, which estimates how
much the working parent's true population performance would rise if this
pattern were diagnosed and repaired now. It is not a probability, a rank, or a
failure frequency. Put your judgment into the four axes.

## Axes (Core)

1. **Severity**: how badly the pattern fails on the current harness. Separate
	 a superficial slip from a core capability breakdown. Judge "resolved" on
	 the held-out distribution using pull history and development deltas, not a
	 single training pass.
2. **Fixability**: whether a harness change can reach the cause. A severe
	 pattern caused by an intrinsic model limitation or by something outside the
	 harness yields little progress.
3. **Breadth**: how widely a fix would transfer. A systematic cause seen across
	 many cases and contexts transfers widely; an idiosyncratic one does not.
4. **Side-effect risk**: whether a fix would regress working behavior or other
	 patterns. Progress is net, not gross.

## Evidence (Core)

Ground fixability and side-effect risk in what the working parent changed.
Use each arm's pull history as primary evidence of realized progress:
patched attempts, all-pass skips, failed attempts, diagnoses, and development
impact. Read the pattern's cases, root-cause evidence, and observations, and
relate them to the harness. Rate every arm in the output schema.
