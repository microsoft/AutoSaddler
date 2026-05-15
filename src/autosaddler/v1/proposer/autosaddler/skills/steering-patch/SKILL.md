---
name: steering-patch
description: "Selected during the steering phase — refines agent behavior by modifying prompts, tool descriptions, or hook reminders without changing executable code"
---

# Steering Patch

## Overview

A steering patch refines the agent's behavior within the existing
capabilities. It modifies how the agent uses existing tools — through
prompt rules, tool description corrections, or hook reminders.

**Key distinction from capability patches:** A steering patch does not add
new tool methods, new parameters, or modify tool implementation code.
It only changes the *text* the agent reads (docstrings, prompts, hooks).
The outer loop selects which skill to use based on the current **phase**:
steering phase uses this skill, capability phase uses `capability-patch`.

## When to Use

Based on the diagnosis results, generate a patch that resolves the
identified root cause. Typical root causes addressable by steering
patches:

| Root Cause Category | Signal | Example |
|---------------------|--------|---------|
| **Misleading description** | Docstring doesn't match implementation | Agent uses correct tool but passes wrong parameter format |
| **Missing description detail** | Docstring lacks constraints or semantics | Agent has correct intent but doesn't know about a required field |
| **Behavioral pattern failure** | Multiple scenarios share the same mistake | Agent consistently forgets to check preconditions before acting |
| **Missing runtime guidance** | Agent needs just-in-time hints for specific tools | Agent misuses a tool because it doesn't recall a constraint at call time |

## Workflow

### Step 1: Consult History

Before writing any changes, check what has been tried before:

```bash
# Full patch history: diffs, reflections, lessons (good/bad patterns)
evo-dag show history

# Per-scenario history: prior root causes and attempted fixes
evo-dag show scenario <scenario_id>
```

- **Build on proven strategies**: Which steering patches generalized
  well to the dev set? What root-cause patterns did they resolve?
- **Avoid known bad patterns**: Which patches caused regressions or
  failed to generalize? Which rule phrasings were too weak or too
  aggressive?
- **Do not re-attempt failed approaches**: If a specific fix was already
  tried on this scenario and failed, try a different approach.

### Step 2: Apply the Patch

Implement the change based on the diagnosis. The common patterns below
are reference examples — patches are not limited to these categories and
may combine multiple approaches.

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

### Tool Description Fix

Fix or supplement tool docstrings when they are inaccurate, incomplete,
or misleading. Only the docstring text is modified — the tool's code
and parameters remain unchanged.

**Trace signals:**
- Agent uses correct tool but wrong parameters due to misleading docstring
- Agent has correct intent but calls the wrong tool due to unclear description
- Agent misinterprets a tool's return value or side effects
- Agent doesn't know about a constraint that the implementation enforces

### Prompt Rule Addition/Modification

Add, modify, or replace behavioral rules in the system prompt or
agent configuration. Prompt rules are always in context and affect every
scenario — use PreToolUse hooks instead when the guidance applies only
to a specific tool.

**Guidelines:**
- Keep rules abstract — no scenario IDs, specific names, or hardcoded answers
- Prefer replacement over addition (the prompt has a finite attention budget):
  can you update an existing rule? Can two related rules be merged?
- Use conditional phrasing like "When X, prefer Y" instead of "NEVER do X"
  — absolute directives can suppress valid behaviors
- Verify no conflicts: re-read all existing rules, check for contradictions.
  If a conflict exists, consider a PreToolUse hook instead (more targeted,
  less interference)

### PreToolUse Hook

Create or modify hooks that inject just-in-time information right before
a specific tool is called. Hooks activate only for their matched tool,
making them targeted and low-interference. For the hook configuration
format and schema, refer to the **Hook Configuration** section in
CLAUDE.md.

**Design guidelines:**
- Match the exact tool that needs the hint
- Keep messages concise (under 2 sentences)
- No hardcoded answers — guide behavior, don't provide scenario-specific
  answers
- Use conditional language: "If the task requires X" rather than absolute
  directives
- State facts about constraints rather than prescribing alternatives
  (e.g., "this tool cannot do X" rather than "use tool Y instead")

## Constraints

- **No hard-coded answers**: Do not embed task-specific answers, lookup
  tables, or scenario-specific logic. Patches must be general-purpose.
- **No code changes**: Do not modify tool implementation code, add new
  tool methods, or change parameters. Only text changes are allowed.
- **No cross-worktree edits**: Only modify files in your current worktree.
  Iteration output dirs and other worktrees are read-only records.

## Safety Assessment

Before committing a steering patch, assess its risk:

1. **History check**: Consult `evo-dag show history` for the full patch
   history with diffs, per-scenario results, reflections, and accumulated
   lessons (good/bad patterns). Learn from what worked, avoid what failed.
2. **Scenario check**: Consult `evo-dag show scenario <id>` for each
   target scenario — prior root causes and previously attempted fixes.
3. **Non-conflicting**: No contradiction with existing rules or hooks.
4. **PASS scenario impact**: Would this change affect correct behavior
   in passing scenarios?
5. **Conciseness**: Is the change short enough to hold the agent's attention?
   Verbose rules dilute the prompt's signal.
6. **Hook validity**: If adding hooks — valid JSON, compilable matcher
   regex, handler type is `"reminder"`.
