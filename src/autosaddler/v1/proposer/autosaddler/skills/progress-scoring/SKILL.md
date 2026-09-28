---
name: progress-scoring
description: "Use to rate four [0,1] learning-progress axes for each candidate failure pattern; phi is their fixed mean with side-effect risk inverted."
---

# Progress Scoring

## What you are estimating (the target — fixed)

For each candidate failure pattern (arm), rate four axes in [0, 1]: severity,
fixability, breadth, and side-effect risk. The sampler computes the cardinal
learning-progress score **φ ∈ [0, 1]** with the fixed formula

`φ = (severity + fixability + breadth + (1 - side_effect)) / 4`.

φ estimates how much the CURRENT harness's true population performance would
rise if this pattern were diagnosed and patched now.

This target is fixed. φ is NOT a probability, NOT a rank, and NOT "how often it
fails". It is the **magnitude of expected improvement**. Higher φ = repairing
this pattern now is expected to improve the harness more.

**Your judgment belongs in the four axis values.** Use the evidence and context
below to rate each axis carefully; `side_effect` is a risk, so higher is worse
and the formula uses `1 - side_effect`.

## Four scoring axes

These axes determine φ directly. Rate all four; omitting one makes the rating
invalid.

1. **Severity — how badly is this pattern failing on the CURRENT harness?**
   Read the traces and scenarios. Distinguish a superficial slip from a core
   capability breakdown. An already-resolved pattern offers ≈0 progress —
   but judge "resolved" on the dev/held-out distribution using the arm's pull
   history and dev-set deltas, not a single mini-batch pass (a fix that flipped
   the scenario but dropped dev accuracy was reverted, so progress may remain).
   (Failure frequency alone captures only part of severity — and the other
   three considerations below require actually reading the traces and harness,
   not just counting failures.)

2. **Fixability — can a harness patch actually repair it?**
   Trace the root cause: does it live in a reachable part of the harness
   (prompt / tool / middleware / loop logic), or beyond it (e.g. an intrinsic
   LLM limitation, or a cause outside the harness)? A severe but unpatchable
   pattern yields little progress.

3. **Breadth — how widely would a fix transfer?**
   Progress is improvement over the whole task distribution, not the repair of
   one scenario. Is this a systematic, recurring cause seen across many
   scenarios/contexts (wide transfer, high progress), or a one-off idiosyncratic
   failure (narrow, low progress)?

4. **Side-effect risk — would a patch here regress other patterns?**
   You can read the harness, so reason about what a fix would touch and whether
   it risks breaking working behavior. Progress is NET, not gross: a fix that
   regresses elsewhere yields little net gain.

## How to judge, using the CLI

1. Read what the confirmed harness just changed: `evo-dag show node <idx>` and
   `evo-dag show edge <parent> <child>`, plus the worktree diff. This grounds
   *fixability* (what is reachable) and *side-effect risk* (what a patch touches).
2. Review prior optimization attempts with `pattern history`. With no
   `--pattern-id`, it shows the complete pull history for every historically
   pulled arm. To inspect a subset together, run
   `pattern history --pattern-id <id1> <id2>`. For initial triage of a
   long history, `--last-k <K>` limits each selected arm separately; inspect the
   complete history before rating when older attempts could change the judgment.
   Use its diagnoses, patch intents, patched attempts, all-pass skips, failed
   attempts, dev-set impacts, and per-scenario reflections as primary evidence
   of realized progress.
3. For each candidate pattern: `pattern show <pattern_id>` for its label, tagged
   scenarios, and evidence; then `evo-dag show scenario <sid>` for each scenario
   to read its failure history and relate it to the harness change.
4. Rate all four axes for each pattern. The sampler computes φ from the fixed
   mean formula; do not supply a separate holistic score.

## Recording

Record each pattern's estimate with the `pattern` CLI (one call per pattern):

```bash
pattern rate --pattern-id <id> \
   --severity <s> --fixability <f> --breadth <b> --side-effect <e> \
  --rationale "one or two sentences justifying the score"
```

The sampler derives φ from the four axis values; the rationale is retained for
calibration analysis against realized progress. Rate EVERY candidate pattern
shown in the session prompt.
