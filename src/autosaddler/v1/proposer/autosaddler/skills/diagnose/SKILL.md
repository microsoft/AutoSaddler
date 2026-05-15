---
name: diagnose
description: "Use to systematically diagnose why scenarios fail — identifies root causes by analyzing agent traces, codebase, and behavioral patterns"
---

# Diagnosis Methodology

## Overview

Diagnosis identifies the root cause of scenario failures by tracing the
agent's execution step-by-step and reading the agent codebase to understand
why the code produces the observed behavior. A correct diagnosis is the
foundation for an effective patch — without understanding WHY the agent
fails, patches are shots in the dark.

## When to Use

Before applying any patch. Every patch should be preceded by diagnosis
of at least the target failing scenarios.

## Diagnosis Workflow

### 1. Read Evaluation Output
Start with the evaluation output for the scenario to get the score and
rationale. The rationale provides a high-level summary of why the scenario
failed. Refer to the session prompt or CLAUDE.md for the exact file paths.

### 2. Read Agent Execution Trace
Read the full agent trace file for the scenario to see:
- Every tool call the agent made
- Every tool response
- The agent's reasoning at each step
- Where the agent deviated from expected behavior

### 3. Read the Agent Codebase
Read the relevant source files to understand why the code caused the
observed behavior:
- **System prompt**: What instructions did the agent receive?
- **Tool docstrings**: What did the agent see about the tool's API?
- **Tool implementation**: What does the tool actually do? Is there a
  mismatch between the docstring and the implementation?
- **Hook configurations**: Are there PreToolUse hooks that should have
  guided the agent but didn't?
- **Agent loop logic**: Are there infrastructure constraints (budget,
  timeouts) that prevented completion?

Trace the code path: what the agent saw (docstring/prompt) → what it
decided (trace reasoning) → what happened (tool implementation).

### 4. Identify the Failure Point

Pinpoint the exact step where the agent's behavior diverged from what was
needed. Compare what the agent did (from the trace) against what it should
have done (from the evaluation rationale and oracle expectations).

Key questions:
- At which step did the agent first make a wrong decision?
- What information was available to the agent at that point?
- What did the agent see (prompt, docstring, tool response) that led to
  the wrong decision?
- Was the root cause in the code (what the tool does), in the text (what
  the agent reads about the tool), or in the agent's reasoning?

### 5. Build a Comprehensive Root Cause Analysis

Go beyond surface-level symptoms. The same failure can often be addressed
through multiple approaches — a missing action might be fixable by adding
a new tool, by improving an existing tool's output, by adding a prompt
rule, or by inserting a hook reminder. Your job is to understand the
**full causal chain** deeply enough that the patch strategy can be chosen
based on which approach is most robust and generalizable.

For each failing scenario, document:
- **What happened**: The exact sequence of agent actions that led to failure
- **Why it happened**: The code path — what the agent saw (docstring/prompt),
  what it decided (trace reasoning), what actually happened (implementation)
- **What should have happened**: The expected behavior based on the oracle
- **Where the gap is**: The precise mismatch between the agent's world model
  and reality — is it in the tool's behavior, the agent's understanding of
  the tool, the agent's planning, or the environment's constraints?

### 6. Cross-Reference with History
- Check `evo-dag show history` for the full patch history — look for
  similar failure patterns that were diagnosed before, what fix was
  applied, and whether it worked. Use this to avoid known pitfalls and
  build on proven strategies.
- Check `evo-dag show scenario <id>` for prior attempts on this specific
  scenario — known root causes and previously attempted fixes. Avoid
  repeating approaches that already failed.

## Diagnosis Quality Checklist

1. **Specific**: Points to the exact line/step where the agent fails
2. **Causal**: Explains WHY the agent makes that mistake (not just what it does wrong)
3. **Actionable**: Clearly implies what kind of patch would fix it
4. **Unique**: Distinguished from prior failed attempts on the same scenario
