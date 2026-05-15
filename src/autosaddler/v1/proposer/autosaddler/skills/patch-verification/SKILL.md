---
name: patch-verification
description: "Use after applying any patch to thoroughly verify the agent codebase has no bugs — checks syntax, imports, runtime behavior, docstrings, templates, hooks, and logical correctness"
---

# Patch Verification

## Overview

A patch that crashes or introduces bugs at runtime is worse than no patch
at all. Every patch must pass thorough verification before being committed.

**Only fix what would actually crash or produce incorrect behavior — do
NOT change the patch's intent.**

## When to Use

After applying ANY patch type (capability or steering).

## Verification Procedure

### Step 1: Review the Diff

```bash
git diff
```

- Does the diff match your intent? No unintended changes?
- Are there leftover debug statements, commented-out code, or TODOs?
- Are there any files modified that should not have been changed?

### Step 2: Syntax and Import Check

For every modified Python file:

```bash
python3 -m py_compile <modified_file>
python3 -c "import <modified_module>"
```

- No `SyntaxError`, `IndentationError`, or `TabError`
- No `ImportError` or `ModuleNotFoundError`
- No `NameError` from undefined variables or functions at module level

### Step 3: Docstring Integrity

Docstrings are the agent's only interface to tools — errors here directly
cause agent failures.

- No placeholder values (`[...]`, `{...}`, `<placeholder>`) — these crash
  downstream parsers
- All examples use concrete, valid values
- Parameter names in the docstring match the actual method signature
- Return value description matches what the implementation actually returns
- No stale documentation from copy-paste (e.g., wrong tool name, wrong
  parameter list from another method)
- No literal escape sequences (`\n`, `\t`, etc.) used to represent
  multi-line examples or formatting inside docstrings — these produce
  actual control characters in the parsed string and can cause
  `SyntaxError` or silently corrupt the docstring text. Use triple-quoted
  strings with real newlines instead

### Step 4: Template String Safety

If the patch modifies any string that uses `.format()` or f-string
interpolation:

- Every `{placeholder}` has a corresponding `.format()` argument
- No extra or missing placeholders from copy-paste
- No unescaped curly braces in literal text (use `{{` and `}}` to escape)
- No cross-file contamination between template variables

### Step 5: Hook JSON Validation

If the patch modifies hook configuration:

```bash
python3 -c "import json; json.load(open('<hook_file>'))"
```

- Valid JSON syntax (no trailing commas, unquoted keys, etc.)
- Correct schema structure (hooks array with matcher + handler entries)
- Each matcher regex compiles without error:
  ```bash
  python3 -c "import re; re.compile('<matcher_value>')"
  ```
- Handler type matches the expected type for the hook category
  (e.g., `"reminder"` for PreToolUse hooks)
- Each handler has non-blank text content

### Step 6: Logic and Correctness

Read each modified function/method end-to-end and verify:

- **Control flow**: No unreachable code, no missing return statements,
  no infinite loops
- **Edge cases**: Does the code handle empty inputs, None values, missing
  keys, and boundary conditions?
- **Type consistency**: Are function arguments and return values used
  consistently with their expected types?
- **Variable scope**: No shadowed variables, no use-before-assignment,
  no stale references to renamed variables
- **Side effects**: Does the change unintentionally modify shared state,
  global variables, or mutable default arguments?
- **Dependencies**: If the modified function is called by other modules,
  verify the signature change is backward-compatible (e.g., new parameters
  have defaults)

### Step 7: Integration Check

- If a new tool was added, verify it is discoverable by the agent
  (registered, decorated, or configured correctly per the codebase's
  conventions)
- If a tool signature changed, verify all existing callers pass the
  correct arguments
- If prompt text was modified, verify the complete prompt still renders
  correctly (no broken concatenation, no missing sections)

### Step 8: Preserve Intent

After all verification, re-read `git diff` one final time:

- Only fix runtime crash or correctness issues found during verification
- Do NOT remove functionality or change patch behavior
- Do NOT alter logic beyond what prevents crashes or bugs
