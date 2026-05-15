---
name: causal-diagnosis
description: "Use before proposing a mutation to diagnose measured failures from evaluator outcomes, traces, history, and the controlling candidate surfaces."
---

# Causal Diagnosis (Core)

## Purpose (Core)

Use this skill before proposing any mutation. Diagnosis identifies why the
current candidate produced a measured failure and which candidate surface most
directly controls it. A symptom summary or evaluator paraphrase is not a
diagnosis.

## Procedure (Core)

For every target failure:

1. Read the evaluator outcome and complete available training trace across all
	repetitions.
2. State required behavior and observed behavior separately.
3. Locate the first decision, action, output, or runtime boundary where they
	diverge.
4. Identify exactly what the agent or system could observe and do at that
	point.
5. Follow the relevant prompt, description, schema, configuration,
	implementation, hook, or loop path end to end.
6. Explain the causal chain from visible input to decision, runtime effect,
	and evaluator result.
7. Compare repetitions and note consistent or conflicting observations.
8. Cross-check prior attempts and lessons for the same case, failure pattern,
	and changed unit.
9. Classify the cause as behavior or description, missing capability,
	implementation, configuration, or infrastructure.

## Intervention Test (Core)

List the plausible mutation surfaces allowed by the plugin. Choose the
approach that most robustly repairs the causal boundary and generalizes beyond
the sampled case. The diagnosis must explain:

- why this intervention can change the observed behavior;
- which passing behavior, callers, or companion surfaces could be affected;
- how the proposed effect can be checked before evaluation.

## Quality Checklist (Core)

A useful diagnosis is:

- **specific**: identifies the exact divergence and controlling surface;
- **causal**: explains why the candidate produced the outcome;
- **evidence-cited**: relies only on staged training evidence and code;
- **actionable**: implies a permitted, testable intervention;
- **distinct**: does not repeat a failed prior explanation without new evidence;
- **generalizable**: targets a reusable mechanism rather than a case answer.

If any item is unsupported, state the uncertainty.