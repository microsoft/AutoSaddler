---
name: capability-patch
description: "Selected during the capability phase to add tools, expose parameters, fix implementations, or remove infrastructure limitations that restrict the agent."
---

# Meta-ARE Capability Patch (Plugin-specific)

## When Capability Is Required (Plugin-specific)

Use this skill when GAIA2 evidence shows that correct behavior is impossible or
structurally unreliable with the current default-agent action space. Eligible
causes include a missing tool, insufficient parameter, implementation defect,
incorrect return shape, missing registration, agent-loop limitation,
configuration mismatch, or absent runtime support.

Use the core diagnosis to identify which executable capability boundary should
change.

## Capability Surfaces (Plugin-specific)

- Add a tool only when existing tools cannot express the required reusable
	operation. Follow current registration and module conventions.
- Add or change a parameter only when the agent needs information or control
	that the current signature cannot represent. Inspect all callers and retain
	compatible defaults where the repository contract permits them.
- Fix an implementation only after confirming the runtime behavior differs
	from its intended contract or fails a relevant boundary case.
- Change loop or configuration behavior only when traces show a structural
	execution limitation rather than an isolated model decision.

## Agent Discoverability (Plugin-specific)

When capability, signature, output, side effect, or constraint changes, update
every affected agent-facing docstring or prompt in the same mutation. Confirm
the rendered tool description teaches the model how and when to use the new
behavior. Remove stale workarounds or conflicting descriptions rather than
stacking contradictory guidance.

## Capability Safety (Plugin-specific)

Keep the implementation narrow, preserve unrelated callers and passing
behavior, and test registration plus the real import path. Do not access
protected benchmark internals, cross candidate boundaries, or encode
task-specific names, IDs, answers, or lookup tables.
