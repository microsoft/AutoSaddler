---
name: causal-diagnosis
description: "Use before proposing a Pareto mutation to diagnose measured failures or avoidable task-agent cost from evaluator outcomes, traces, history, and the controlling candidate surfaces."
---

# Causal Diagnosis (Core)

## Purpose (Core)

Use this skill before proposing any mutation. Diagnosis identifies why the
current candidate produced a measured failure or avoidable cost and which candidate surface most
directly controls it. A symptom summary or evaluator paraphrase is not a
diagnosis.

Determine what added work bought for accuracy and what removed work saved or
harmed. An expensive run may contain necessary work; a cheaper run may have
skipped required behavior.

## Procedure (Core)

For every target failure or cost mechanism:

1. Read the evaluator outcome, recorded task-agent cost, and complete available training trace across all
	repetitions.
2. State required behavior and observed behavior separately.
3. Locate the first decision, action, output, or runtime boundary where they
	diverge or avoidable work begins.
4. Identify exactly what the agent or system could observe and do at that
	point.
5. Follow the relevant prompt, description, schema, configuration,
	implementation, hook, or loop path end to end.
6. Explain the causal chain from visible input to decision, runtime effect,
	evaluator result, and measured task-agent cost.
7. Compare repetitions and note consistent or conflicting observations.
8. Cross-check prior attempts and lessons for the same case, failure pattern or cost mechanism,
	and changed unit.
9. Classify the cause as behavior or description, missing capability,
	implementation, configuration, or infrastructure.

## Objective Attribution (Core)

- Inspect execution shape: model/tool calls, retries, stalls, recovery,
	validation, and behavioral divergence between runs.
- Inspect context shape: prompt/schema size, tool-output volume, cache behavior,
	compaction, and reasoning tokens when the staged evidence exposes them.
- Connect these observations to accuracy and recorded cost. Neither token counts
	nor call counts establish value alone; do not treat all input tokens as equally
	priced.

Compare cheap versus expensive and high- versus low-score runs of the same case
across all available repetitions. Identify what the cheap, correct run did
differently and whether a reusable change can make that behavior reliable.
Name the controlling surface and the expected joint gain, accuracy gain,
cost saving, or trade-off.

## Intervention Test (Core)

List the plausible mutation surfaces allowed by the plugin. Choose the
approach that most robustly repairs the causal boundary and generalizes beyond
the sampled case. The diagnosis must explain:

- why this intervention can change the observed behavior;
- which passing behavior, callers, or companion surfaces could be affected;
- how the proposed effect can be checked before evaluation.

Assess expected effects on both accuracy and cost, including trade-offs and
regression risks.

## Quality Checklist (Core)

A useful diagnosis is:

- **specific**: identifies the exact divergence and controlling surface;
- **causal**: explains why the candidate produced the outcome;
- **evidence-cited**: relies only on staged training evidence and code;
- **actionable**: implies a permitted, testable intervention;
- **distinct**: does not repeat a failed prior explanation without new evidence;
- **generalizable**: targets a reusable mechanism rather than a case answer.

If any item is unsupported, state the uncertainty.