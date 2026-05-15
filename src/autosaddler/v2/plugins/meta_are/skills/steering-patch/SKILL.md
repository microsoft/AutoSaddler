---
name: steering-patch
description: "Selected during the steering phase to refine agent behavior through prompts, tool descriptions, or hook reminders without changing executable code."
---

# Meta-ARE Steering Patch (Plugin-specific)

## Steering Boundary (Plugin-specific)

Use steering iterations to refine how the default agent chooses among existing
capabilities. Modify non-Python declarative guidance or `hook.json` PreToolUse
configuration only. The verifier rejects all Python-file changes, including
Python-hosted prompts and docstrings, during this phase.

## Declarative Guidance (Plugin-specific)

Prefer correcting, replacing, or consolidating an existing instruction over
appending a competing rule. Keep guidance short, operational, conditional, and
general across GAIA2 scenarios. Check the complete surrounding prompt for
contradictions and avoid absolute rules that suppress valid behavior.

Use always-on guidance only for behavior that applies broadly. If guidance is
specific to one tool boundary, prefer a targeted PreToolUse reminder.

## PreToolUse Hooks (Plugin-specific)

Preserve unrelated hook entries and use the repository's supported nested
PreToolUse reminder schema. Match only the intended tool with a valid regular
expression. Keep reminders concise, state the relevant constraint rather than
a case-specific action, and verify the final JSON structure.

Never include task-specific names, IDs, answers, or lookup tables. A hook may
guide a decision but must not solve a benchmark case for the agent.
