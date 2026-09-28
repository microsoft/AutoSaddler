# Session 3: Pattern Extraction

## Mandatory Skills

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize any skill — execute the full procedure described
in each one. These skills are installed at `.claude/skills/<name>/SKILL.md`
in the current worktree.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `symptom-extract` | For each failed scenario | Symptom extraction from diagnosis/reflection results — generates candidate symptoms |
| 2 | `symptom-normalize` | After extracting all candidates | 3-way normalization against existing patterns — links, composes, or creates patterns to maintain atomic pattern integrity |

> **Enforcement**: You MUST process every failed scenario. Do NOT skip the extraction step for any failed scenario. During extraction (Skill 1: `symptom-extract`), do NOT examine existing patterns; consult them only during normalization (Skill 2: `symptom-normalize`). This separation is required to avoid anchoring bias.

## Goal

Extract failure patterns from diagnosis and reflection results, then tag
each failed (harness, trace, scenario) tuple with the identified pattern(s).

Failure patterns are **symptom-level abstractions** — more abstract than
individual root causes, but specific enough to distinguish different failure
types. They are the **arms** of the curriculum sampler.

At each outer-loop iteration, each arm (i.e., failure pattern) is assigned a learning-progress score estimating the expected improvement in the current harness’s population performance that would result from selecting the arm and successfully addressing its corresponding failure pattern in the next optimization step.

## Context

- **Iteration**: {iteration}
- **Candidate**: C{candidate_idx}
- **Current worktree**: `{worktree_path}`
- **Session root**: `{session_root}`
- **Pre-patch trace dir**: `{before_output_dir}`
- **Post-patch trace dir**: `{after_output_dir}`

## Failed Scenarios to Process

### Pre-patch failures (from diagnosis)

These scenarios failed BEFORE any patch was applied. Each entry provides the
batch-level Session 1 diagnosis and detailed reasoning-file location from before
patching, followed by Session 2's reviewed underlying root cause and post-hoc
analysis of how the patch affected that cause.

{pre_patch_failures}

### Post-patch failures (from reflection)

These scenarios STILL FAILED or REGRESSED after the patch was applied.
Each entry provides the status change, the batch-level Session 1 patch approach,
and Session 2's root cause and post-hoc patch-effect explanation. For
`still_failing`, the root cause is the remaining or revised failure cause; for
`regressed`, it is the newly observed regression cause or failure mechanism.

{post_patch_failures}

## Workflow

### 1. Extract candidate symptoms

For each failed scenario listed above:

1. For pre-patch failures, review the Session 1 diagnosis/reasoning and Session
   2's reviewed cause and patch-effect analysis. For post-patch failures, also
   use the status change and Session 1 patch approach to distinguish a remaining
   cause from a newly introduced regression mechanism.
2. Follow the `symptom-extract` skill to generate a **candidate symptom**
   label. Do NOT look at existing patterns during this step.
3. Record every candidate symptom and its associated root cause in
   `symptom_candidates.md` at the working directory root.

Complete `symptom_candidates.md` for all failures before moving to normalization.

### 2. Normalize against existing patterns

After all candidate symptoms are extracted:

1. Read `symptom_candidates.md` from the working directory root.
2. Run `pattern list` to retrieve all existing patterns.
3. For each candidate symptom in the file, follow the `symptom-normalize` skill to
   perform a **3-way judgment**:
   - **(a) Same**: the candidate matches an existing pattern → link to it
   - **(b) Composition**: the candidate is a combination of existing atomic
     patterns → tag with multiple existing patterns
   - **(c) New**: genuinely novel failure type → register as a new pattern

### 3. Tag tuples

For each resolved pattern assignment:

1. Register new patterns if needed:
   ```
   pattern register --label "DESCRIPTIVE SYMPTOM LABEL"
   ```

2. Tag the (harness, trace, scenario) tuple:
   ```
   pattern tag --pattern-id <id> --harness {candidate_idx} --trace <dir> --scenario <sid> --root-cause "..."
   ```
   For compositions, repeat `--pattern-id` for each atomic pattern.

   > **IMPORTANT**: Always use `--harness {candidate_idx}` (the current
   > candidate) for ALL tagging commands — both pre-patch and post-patch
   > failures. The `--trace` directory distinguishes before/after. The
   > harness identifies which iteration this observation belongs to.

> **Note**: In Session 3, the agent only needs to execute `pattern register` and `pattern tag`. The agent does not need to calculate or record observation values separately. For each previously known pattern, the outer loop automatically calls `PatternRegistry.observe()` with the scenarios associated with that pattern and evaluated in the mini-batch, the subset still tagged after patching, and the resulting activity fraction.

## Important Notes

- **Root cause is preserved as evidence**, not used directly as the pattern.
  The pattern label should be an abstraction that generalizes across
  multiple scenarios.
- **Patterns should be atomic** — a single patch should be able to fix one
  pattern. Complex failures are expressed as compositions (multi-tagging).
- **Do NOT over-merge**: if two failures have different root causes, they
  should be different patterns even if they look superficially similar.
- **Do NOT over-split**: if the same behavioral issue manifests with minor
  surface variations, it's the same pattern.