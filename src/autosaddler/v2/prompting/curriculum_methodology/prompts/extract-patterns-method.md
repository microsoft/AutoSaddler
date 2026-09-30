# Session 3: Pattern Extraction (Core)

## Mandatory Skills (Core)

You MUST read and follow the SKILL.md for each skill listed below.
Do NOT skip or summarize any skill — execute the full procedure described
in each one. These skills are supplied with this session.

| Order | Skill | When | Why |
|-------|-------|------|-----|
| 1 | `symptom-extract` | For each failed scenario | Symptom extraction from diagnosis/reflection results — generates candidate symptoms |
| 2 | `symptom-normalize` | After extracting all candidates | 3-way normalization against existing patterns — links, composes, or creates patterns to maintain atomic pattern integrity |

> **Enforcement**: You MUST process every failed scenario. Do NOT skip the extraction step for any failed scenario. During extraction (Skill 1: `symptom-extract`), do NOT examine existing patterns; consult them only during normalization (Skill 2: `symptom-normalize`). This separation is required to avoid anchoring bias.

## Goal (Core)

Extract failure patterns from diagnosis and reflection results, then tag
each failed (candidate, evaluation, scenario) tuple with the identified pattern(s).

Failure patterns are **symptom-level abstractions** — more abstract than
individual root causes, but specific enough to distinguish different failure
types. They are the **arms** of the curriculum sampler.

At each outer-loop iteration, each arm (i.e., failure pattern) is assigned a learning-progress score estimating the expected improvement in the current harness’s population performance that would result from selecting the arm and successfully addressing its corresponding failure pattern in the next optimization step.

## Context (Core)

Read `.autosaddler/session_context.json`:

- **Iteration**: `iteration`
- **Candidate**: `candidate_ids[0]` (the patched candidate); the working parent
  is `task_selection.working_parent_candidate_id`
- **Pre-patch training evidence**: `task_selection.train_before_evidence`
- **Post-patch training evidence**: `task_selection.train_after_evidence`

## Failed Scenarios to Process (Core)

### Pre-patch failures (from diagnosis) (Core)

These scenarios failed BEFORE any patch was applied. `pre_patch_failures`
lists each with its scores and status change; `diagnosis` and `patch_intent`
provide the batch-level Session 1 diagnosis and patch intent from before
patching, and `lessons` provides Session 2's reviewed lessons on how the patch
affected the failures.

### Post-patch failures (from reflection) (Core)

These scenarios STILL FAILED or REGRESSED after the patch was applied.
`post_patch_failures` lists each with its status change; `patch_intent`
provides the batch-level Session 1 patch approach, and `lessons` provides
Session 2's reviewed lessons on the patch effect. For
`still_failing`, the root cause is the remaining or revised failure cause; for
`regressed`, it is the newly observed regression cause or failure mechanism.

## Workflow (Core)

### 1. Extract candidate symptoms (Core)

For each failed scenario listed above:

1. For pre-patch failures, review the Session 1 diagnosis and intent and
   Session 2's lessons. For post-patch failures, also
   use the status change and Session 1 patch approach to distinguish a remaining
   cause from a newly introduced regression mechanism.
2. Follow the `symptom-extract` skill to generate a **candidate symptom**
   label. Do NOT look at existing patterns during this step.
3. Record every candidate symptom and its associated root cause in
   `symptom_candidates.md` at the working directory root.

Complete `symptom_candidates.md` for all failures before moving to normalization.

### 2. Normalize against existing patterns (Core)

After all candidate symptoms are extracted:

1. Read `symptom_candidates.md` from the working directory root.
2. Read `.autosaddler/curriculum/patterns.json` to retrieve all existing patterns.
3. For each candidate symptom in the file, follow the `symptom-normalize` skill to
   perform a **3-way judgment**:
   - **(a) Same**: the candidate matches an existing pattern → link to it
   - **(b) Composition**: the candidate is a combination of existing atomic
     patterns → tag with multiple existing patterns
   - **(c) New**: genuinely novel failure type → register as a new pattern

### 3. Tag tuples (Core)

For each resolved pattern assignment:

1. Register new patterns if needed: add one `new_patterns` entry with a short
   unique `key` and a `label` ("DESCRIPTIVE SYMPTOM LABEL").

2. Tag the (candidate, evaluation, scenario) tuple: add one `tags` entry with
   the scenario's `case_id`, its `source` (`pre_patch` or `post_patch`), its
   `pattern_refs` (existing pattern IDs or keys of new patterns), and its
   `root_cause`. For compositions, list every atomic pattern in `pattern_refs`.

   > **IMPORTANT**: `source` selects the evaluation: `pre_patch` tags the
   > working parent's initial evaluation and `post_patch` tags the patched
   > candidate's re-evaluation. The outer loop records both under this
   > iteration.

> **Note**: In Session 3, the agent only needs to return new patterns and tags. The agent does not need to calculate or record observation values separately. For each previously known pattern, the outer loop automatically records an observation with the scenarios associated with that pattern and evaluated in the mini-batch, the subset still tagged after patching, and the resulting activity fraction.

## Important Notes (Core)

- **Root cause is preserved as evidence**, not used directly as the pattern.
  The pattern label should be an abstraction that generalizes across
  multiple scenarios.
- **Patterns should be atomic** — a single patch should be able to fix one
  pattern. Complex failures are expressed as compositions (multi-tagging).
- **Do NOT over-merge**: if two failures have different root causes, they
  should be different patterns even if they look superficially similar.
- **Do NOT over-split**: if the same behavioral issue manifests with minor
  surface variations, it's the same pattern.
