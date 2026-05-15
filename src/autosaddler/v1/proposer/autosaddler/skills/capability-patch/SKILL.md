---
name: capability-patch
description: "Selected during the capability phase — adds new tools, exposes new parameters, fixes tool implementations, or removes infrastructure limitations to expand what the agent can do"
---

# Capability Patch

## Overview

A capability patch expands the agent's capabilities — by adding new tool
methods, exposing new parameters, fixing tool implementations, or removing
infrastructure limitations. These changes let the agent do things it could
not do before.

**Key distinction from steering patches:** A capability patch changes
executable code — tool methods, parameters, implementation logic, or
infrastructure. A steering patch only changes text the agent reads
(docstrings, prompts, hooks) without modifying any code. The outer loop
selects which skill to use based on the current **phase**: capability
phase uses this skill, steering phase uses `steering-patch`.

## When to Use

Based on the diagnosis results, generate a patch that resolves the
identified root cause. Typical root causes addressable by capability
patches:

| Root Cause Category | Signal | Example |
|---------------------|--------|---------|
| **Missing capability** | No tool exists for the required action | Agent needs to search emails by date range but no such tool exists |
| **Tool bug** | Tool exists but produces wrong results | `search_contacts` returns empty for valid queries due to case-sensitive matching |
| **Insufficient parameters** | Tool exists but lacks needed arguments | Agent needs to filter by sender but the search tool only accepts keyword |
| **Missing agent loop logic** | Agent loop lacks a preprocessing step, reminder, or context-injection logic | No budget reminder causes agent to exhaust iterations without wrapping up; no environment summary injection at start leaves agent unaware of available data |
| **Infrastructure issue** | Infrastructure limits prevent success | Agent hits the iteration budget on multi-step tasks |

## Workflow

### Step 1: Consult History

Before writing any code, check what has been tried before:

```bash
# Full patch history: diffs, reflections, lessons (good/bad patterns)
evo-dag show history

# Per-scenario history: prior root causes and attempted fixes
evo-dag show scenario <scenario_id>
```

- **Build on proven strategies**: Which capability patches generalized
  well to the dev set? What root-cause patterns did they resolve?
- **Avoid known bad patterns**: Which patches caused regressions or
  failed to generalize? Understand why and do not repeat.
- **Do not re-attempt failed approaches**: If a specific fix was already
  tried on this scenario and failed, try a different approach.

### Step 2: Apply the Patch

Implement the change based on the diagnosis. The common patterns below
are reference examples — patches are not limited to these categories and
may combine multiple approaches.

**Align agent-facing text with capability changes.** When you add or
modify tools, parameters, or infrastructure code, also update the
system prompt, tool docstrings, and/or hooks so the agent is aware of
the changes. Without this, the agent may ignore new tools or continue
using outdated workarounds. Check that no conflicting rules exist in
the system prompt.

### Step 3: Verify

Run the `patch-verification` skill to confirm the patch won't crash at
runtime (syntax, imports, docstring format, JSON validity).

### Step 4: Write Reasoning and Record Intent

After applying and verifying the patch, write your reasoning to
`proposer_reasoning.md` in the working directory root. Then record intent:

```bash
evo-dag update-intent \
  --target-scenarios "id1,id2" \
  --diagnosis "Root cause analysis from diagnosis" \
  --approach "What you changed and why" \
  --files-changed "file1.py,file2.py" \
  --change-summary "Summary of code changes"
```

## Common Patterns

The following are frequently observed patch patterns. Use them as
reference — patches may combine multiple patterns or take entirely
different approaches depending on the diagnosis.

### New Tool Addition

Add a new tool method when existing tools can't solve the scenario or
force the agent into inefficient workarounds.

**Trace signals:**
- Agent cannot perform the required action at all — no tool supports it
- Agent tries workarounds, wastes steps, or gives up
- Agent calls the same tool many times in a loop with different parameters
- Agent exhausts iteration budget on repetitive operations

**Checklist:**
- Follow the codebase's existing tool registration conventions
- Write a comprehensive docstring — the agent only sees the docstring
- Verify the new tool doesn't shadow existing methods
- Respect module boundaries — do not access internals of other modules

### Argument Modification

Add or fix tool parameters to expose filtering, sorting, or selection
capabilities that the current API doesn't provide.

**Trace signals:**
- Agent needs to filter/search by a field that isn't exposed as a parameter
- Agent fetches all results and manually filters in the reasoning loop
- Agent passes a parameter value that the tool doesn't accept

**Checklist:**
- Add the new parameter with a sensible default (backward-compatible)
- Update the docstring to document the new parameter — the agent only
  learns about it through the docstring
- Verify existing callers are unaffected by the default value

### Implementation Fix

Fix bugs or extend functionality in tool internals so the tool
produces correct results or handles edge cases.

**Trace signals:**
- Tool returns wrong results, empty results, or crashes
- Tool works for common cases but fails on edge cases
- Tool's return format doesn't match what the docstring promises

**Checklist:**
- Confirm the bug by reading the implementation code, not just the trace
- Fix the root cause, not a symptom
- If the docstring was accurate but the implementation was wrong, only
  fix the implementation — do not weaken the docstring
- If the implementation was correct but the docstring was misleading,
  consider whether this is actually a `steering-patch` (docstring-only fix)

### Infrastructure Change

Modify agent configuration, execution loop, or environment settings to
remove structural limitations.

**Trace signals:**
- Agent hits the iteration limit with task still incomplete
- Runtime errors, state corruption, or configuration mismatches
- Agent configuration doesn't match scenario requirements

### Agent Loop Logic Change

Add preprocessing steps, reminders, or context-injection logic to the
agent loop to address structural gaps in how the agent operates.

**Trace signals:**
- Agent exhausts iterations without attempting to wrap up or finalize
- Agent starts tasks without awareness of the environment state
- Agent repeatedly forgets constraints that should be reinforced at runtime

## Constraints

- **No hard-coded answers**: Do not embed task-specific answers, lookup
  tables, or scenario-specific logic. Patches must be general-purpose.
- **No protected-path modifications**: Never modify evaluation, judging,
  ground truth, or benchmark infrastructure.
- **No cross-worktree edits**: Only modify files in your current worktree.
  Iteration output dirs and other worktrees are read-only records.

## Safety Assessment

Before committing a capability patch, assess its risk:

1. **History check**: Consult `evo-dag show history` for the full patch
   history with diffs, per-scenario results, reflections, and accumulated
   lessons (good/bad patterns). Learn from what worked, avoid what failed.
2. **Scenario check**: Consult `evo-dag show scenario <id>` for each
   target scenario — prior root causes and previously attempted fixes.
3. **Dependencies**: What other modules call or import this code? Use
   `grep` to find all references before modifying a shared function.
4. **PASS scenario impact**: Will this change affect behavior for scenarios
   that are currently passing? A net-positive patch that regresses passing
   scenarios will record a bad pattern in lessons.
5. **Reversibility**: Can this be cleanly reverted if it causes regressions?
