---
name: verification-baseline
description: "Use after applying a candidate change to verify its diff, scope, syntax, integration behavior, and consistency before returning it."
---

# Verification Baseline (Core)

## Purpose (Core)

Verification asks whether the complete candidate is internally consistent,
loadable, within scope, and capable of producing the intended effect without
obvious collateral damage. It is not permission to weaken the intended change
until a check passes.

## Procedure (Core)

Before returning a change:

1. Review the complete diff or update set. Remove unintended edits, debug text,
	stale artifacts, no-op churn, and unrelated formatting.
2. Confirm every changed unit is inside the declared mutation scope, every
	declared unit actually changed, and every protected input is untouched.
3. Trace dependencies and callers of shared surfaces. Preserve registration,
	imports, defaults, signatures, schemas, side effects, and companion files
	unless the diagnosis requires a coordinated change.
4. Validate syntax and structure with the plugin's available checks. Confirm
	the candidate loads through its real integration path, not only an isolated
	parser.
5. Inspect the final rendered or agent-visible form of prompts, descriptions,
	templates, hooks, and schemas. Check placeholders, escaping, formatting,
	examples, parameter names, and runtime behavior for consistency.
6. Exercise empty inputs, boundary values, error paths, and representative
	existing callers when they can reach the changed surface.
7. Recheck behavior that already passed and could be affected. A targeted fix
	should not silently broaden, suppress, or reorder unrelated behavior.
8. Confirm the mutation is substantive: the effective candidate differs in
	the intended way after normalization, rendering, or serialization.
9. Return only when the proposed effect, actual changed units, verification
	evidence, and structured result agree exactly.

## Failure Handling (Core)

If verification fails, repair or remove the invalid change and rerun the same
check. Do not hide the failure, modify protected evaluation logic, fabricate a
successful result, or change intended semantics merely to satisfy a superficial
check. The plugin-specific verification skill may impose stricter rules; those
rules are part of the candidate contract.