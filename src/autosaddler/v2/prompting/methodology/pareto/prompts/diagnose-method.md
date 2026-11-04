# Diagnosis And Change Method (Core)

The diagnosis session must turn measured training evidence into a verifiable
intervention. Follow the supplied skills in this order:
`history-analysis`, `diagnose`, the plugin-specific mutation skill, and the
plugin-specific verification skill. Read `.autosaddler/history/manifest.json`
before interpreting the current evidence.

## Establish The Failure Or Cost Mechanism (Core)

For each target failure or cost mechanism, compare the evaluator outcome with the complete
available trace and repetitions. Write down:

- what behavior was required by the supplied evidence;
- what the candidate actually did;
- the first point where those paths diverged or avoidable work began;
- what information and capabilities were available at that point;
- whether repetitions show consistent or conflicting behavior.

An evaluator rationale is a lead, not a root cause. The earliest divergence or cost mechanism
must be traced to the candidate surface that controlled the observed decision
or behavior before proposing a patch.

Consider both low-accuracy cases with improvement headroom and expensive runs
with savings headroom, including an all-success batch. Inspect redundant calls,
repeated exploration, oversized context, and unnecessary output. Use repeated
successful runs as evidence for making cheap-and-correct behavior more reliable.

## Build The Causal Chain (Core)

Trace the path from agent-visible input through decision, action or output,
runtime behavior, evaluator result, and measured task-agent cost. Inspect the relevant prompt,
description, schema, configuration, implementation, hook, or loop boundary as
allowed by the plugin. Explain why that surface produced the divergence or avoidable cost.

Cross-check prior attempts on the same cases and changed units. A useful
diagnosis is specific, causal, evidence-cited, actionable, and distinct from a
previously failed explanation. Label uncertainty instead of filling gaps with
assumptions.

## Choose An Intervention (Core)

Consider the available intervention surfaces before editing. Choose the
approach that most robustly repairs the causal boundary and generalizes across
similar unseen cases.

Preserve passing behavior and compatible durable improvements. Never encode a
task-specific name, ID, answer, or lookup table.

Apply the intervention to the fixed current candidate, not a reselected parent.
State expected effects on both accuracy and cost, trade-offs, and regression risks.
The configured Bayesian bootstrap screen advances a candidate when persisted
joint support for promising movement within the accuracy-loss bound reaches
its threshold. The Pareto archive decision happens during development evaluation.

## Verify And Report (Core)

Review the complete mutation, its dependencies, rendered or agent-visible
form, and likely effect on existing callers or passing cases. Run every check
required by the plugin contract. The final structured result must describe
the causal diagnosis, intended effect, and exact changed units; those claims
must agree with the actual verified mutation.
