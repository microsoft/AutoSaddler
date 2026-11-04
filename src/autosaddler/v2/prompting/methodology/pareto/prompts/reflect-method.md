# Reflection Method (Core)

Reflection converts a completed experiment into durable, appropriately
qualified learning. Begin with the `history-analysis` skill, then compare the
current session context with matched before/after training score/cost measurements,
promotion, changed units, diagnosis, and archive-referenced aggregate development information
when present.

## Classify The Evidence (Core)

For each relevant training outcome, use these analytical categories:

- **fixed**: a previously failing behavior now succeeds;
- **regressed**: a previously successful behavior now fails;
- **still failing**: the attempted change did not resolve the observed cause;
- **still passing**: existing behavior survived the change;
- **uncertain or stochastic**: repetitions or evidence do not support stable
	attribution.
- **cost reduction or increase**: a measured change in task-agent cost;
- **accuracy/cost trade-off**: movement in one objective at the other's expense;
- **uncertain cost effect**: missing or incomplete usage prevents attribution.

These categories guide analysis; return only the lesson structure required by
the supplied output schema.

Separate cost reduction with preserved behavior from trade-offs and regressions.
For accuracy gains with higher cost, ask whether the additional work bought the
gain. For cost savings with accuracy loss, ask whether removed work was redundant
or required. Also distinguish unchanged objectives from uncompensated one-axis
or joint losses. These movement descriptions do not decide promotion.

## Test Attribution (Core)

Link the exact changed unit to the first before/after behavioral or cost divergence.
Ask whether the candidate change altered information, capability, decision, or
runtime behavior in the way the diagnosis predicted. A score or cost change without
that link is correlation, not a demonstrated mechanism.

Treat the promotion decision as a policy outcome, not proof that every part of
the diagnosis was correct. Aggregate development movement can support or
contradict broad generalization, but it cannot establish a hidden per-case
mechanism. One run is insufficient for a universal rule.

Check whether the changed path actually ran and whether it produced the intended
behavior. Distinguish supported, partially supported, unsupported, contradicted,
and unmeasured mechanisms from objective movement and policy decisions. Attribute
movement as patch-caused, stochastic, mixed, or uncertain only after inspecting
the available training traces, diff, and outcomes. Express this analysis in the
existing lessons.

Distinguish training promotion, development archive membership, and carried-forward
current; a promoted child can be development-dominated. For deferred reflection,
use the owning iteration's recorded current and decisions, not a later summary's
current. Read the recorded Bayesian bootstrap joint support and screening decision.

When development aggregates are supplied, compare both objectives with the
seed, immediate parent, owning iteration's current, and other archive members
whose measurements are available. Check whether the training movement held on
development data and whether the result was nondominated. Attribute aggregate
movement only to supported mechanisms; when this candidate has no development
measurement, state that generalization is unmeasured.

## Record Durable Lessons (Core)

Add only lessons that are reusable, evidence-supported, and not already
captured by history. A good lesson states its scope, observed evidence, causal
confidence, and practical implication. Preserve uncertainty when repetitions
conflict or when multiple changed units prevent clean attribution.

For regressions, identify the unsafe assumption or overly broad effect to
avoid. For still-failing cases, distinguish an incorrect diagnosis from an
insufficient implementation. For fixes, distinguish a durable mechanism from
a narrow or fragile gain. Do not infer development or test case details from
aggregate scores.