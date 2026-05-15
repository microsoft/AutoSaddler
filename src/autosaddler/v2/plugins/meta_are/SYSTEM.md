# Meta-ARE Candidate And Benchmark Context (Plugin-specific)

The active candidate is a Git-backed Meta-ARE default-agent harness evaluated
on GAIA2. GAIA2 is an interactive read/write benchmark: the evaluated agent
plans over simulated applications and executes state-changing tools rather
than merely returning an information-retrieval answer.

## Agent-Visible Control Surfaces (Plugin-specific)

While acting, the default agent can observe its system and environment
instructions, tool descriptions generated from Python docstrings, tool
results, and any matching PreToolUse reminder. It does not inspect Python tool
implementations during a benchmark episode. A mismatch between an agent-facing
description and runtime behavior can therefore be as consequential as an
implementation defect.

Relevant candidate surfaces may include:

- system and environment prompt text;
- tool docstrings, signatures, registration, and implementations;
- default-agent planning or execution-loop behavior;
- configuration that affects agent execution;
- `hook.json` PreToolUse reminders.

Only paths listed in `.autosaddler/session_context.json` are mutable in the
current attempt.

## GAIA2 Failure Semantics (Plugin-specific)

The evaluator can reject a trajectory before semantic judging when required
tool calls are missing, extra calls violate the expected action structure, the
wrong tool is selected, or hard-checked arguments are incorrect. Exact IDs,
timestamps, paths, attendees, quantities, and other structured values must be
distinguished from semantically judged content such as message wording, tone,
placeholders, or signatures.

When `.autosaddler/training_evidence.json` is staged, use it as the only
case-level evidence. For each repetition, connect evaluator rationale to the
complete interaction trace and identify whether the failure arose before tool
selection, during argument construction, in runtime behavior, or in the
resulting side effect. Do not infer oracle actions or hidden scenario state
that the evidence does not show.

## Mutation Phases (Plugin-specific)

The current `patch_phase` is recorded in
`.autosaddler/session_context.json`:

- **Capability** may change reusable Python logic, tools, parameters,
	descriptions required by changed behavior, prompts, configuration, hooks,
	or supporting infrastructure inside the declared scope.
- **Steering** may change non-Python declarative guidance and `hook.json` only.
	The current verifier rejects every changed Python file during steering,
	including Python-hosted prompts and docstrings.

Choose changes that are legal for the active phase. Do not disguise a
capability change as steering or leave agent-facing text inconsistent with a
capability change.

## Protected Surfaces (Plugin-specific)

Benchmark, validation, scenario, dataset, checkpoint, test, and other paths
outside the declared mutation scope are immutable. Never alter judging,
fixtures, expected trajectories, scenario inputs, or benchmark internals.

## Hook And Description Consistency (Plugin-specific)

Docstrings must accurately describe current signatures, accepted values,
outputs, side effects, and constraints. PreToolUse hooks must use the current
repository schema, compile as regular expressions, use supported reminder
handlers, remain concise and conditional, and contain no task-specific names,
IDs, answers, or lookup tables.
