---
name: patch-verification
description: "Use after applying a Meta-ARE patch to check syntax, imports, integration behavior, hooks, mutation scope, and result consistency."
---

# Meta-ARE Patch Verification (Plugin-specific)

Apply the core verification baseline and run available local checks. Then
review the candidate against the Meta-ARE constraints below. AutoSaddler
performs the configured import, scope, training-case-ID, and exact-diff checks
automatically after the session; do not claim those automatic checks ran from
inside the optimizer session.

## Python And Integration Checks (Plugin-specific)

- Compile every modified Python file.
- Run available targeted import or integration checks for changed code.
- Confirm changed signatures, docstrings, implementations, and callers remain
	mutually consistent.

## Hook Checks (Plugin-specific)

- Parse `hook.json` as JSON and require the supported PreToolUse structure.
- Compile every matcher regular expression.
- Require supported reminder handlers and nonblank reminder text.
- Reject task-specific names, IDs, answers, or lookup tables.

## Scope And Result Checks (Plugin-specific)

- Confirm no benchmark, validation, scenario, data, checkpoint, test, or other
	protected path changed.
- During steering, apply the exact boundary defined by the `steering-patch`
	skill.
- Confirm every diff path is allowlisted and `changed_paths` exactly matches
	the final repository-relative diff.
