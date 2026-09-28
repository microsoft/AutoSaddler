# Skill: Symptom Normalize

## Purpose

Determine whether a candidate symptom (extracted bias-free in the previous
step) corresponds to an **existing pattern**, is a **composition** of
existing patterns, or is a **genuinely new** pattern.

This is the only step where you see the existing pattern list. The
candidate symptom was already generated independently to avoid anchoring.
Now you perform a comparison-only judgment.

## Procedure

### Step 1: Read extracted candidates

Read `symptom_candidates.md` from the working directory root. This file is the
complete output of the preceding `symptom-extract` step. Normalize every
candidate symptom recorded in it; do not omit or rewrite candidates before
comparing them with existing patterns.

### Step 2: Retrieve existing patterns

Run the `pattern` CLI to view **all** currently registered patterns:

```bash
# List ALL patterns (you must review every one, not just top-scored)
pattern list
```

Output format:
```text
| Pattern | Activity | Observations | #Scen | LastObs | Label | Scenarios |
|---------|----------|--------------|-------|---------|-------|-----------|
| abc12345 | 0.84 | 3:2/3 (active=0.67), tagged=[scenario-id-1, scenario-id-2], evaluated=[scenario-id-1, scenario-id-2, scenario-id-3] | 3 | 3 | Singular/plural ambiguity in entity count | scenario-id-1, scenario-id-2, scenario-id-3 |
| def67890 | 1.00 | (none) | 1 | 0 | Missing pagination in search results | scenario-id-4 |
```

Table columns:

- `Pattern`: Stable failure-pattern ID used by the `pattern` and `evo-dag` CLIs.
- `Activity`: Rested EMA of the raw observations, seeded at `1.00`. This is a
  mechanical severity reference, not the Agent learning-progress score φ.
- `Observations`: Complete per-iteration observation history. Each entry uses
  `iteration:tagged/evaluated (active=...), tagged=[...], evaluated=[...]`.
  `evaluated` lists the scenarios associated with this pattern that were
  evaluated in that iteration's mini-batch, and `tagged` lists the subset that
  was still tagged with the pattern after patching. `active` is the corresponding
  `tagged/evaluated` fraction. For legacy observations whose scenario IDs were
  not recorded, the counts appear as `?/?` and both lists as `(not recorded)`;
  `(none)` means the arm has not been selected again since the pattern was most
  recently observed in a scenario, so no subsequent activity observation has
  been recorded.
- `#Scen`: Number of unique scenarios currently tagged with the pattern.
- `LastObs`: Most recent iteration in which the arm was selected and an activity
  observation was recorded, or `0` if the arm has not been selected again since
  the pattern was most recently observed in a scenario.
- `Label`: Symptom-level failure-pattern description.
- `Scenarios`: Complete list of scenario IDs currently owned by the arm.

For more detail on a specific pattern (tagged scenarios, evidence):
```bash
pattern show <pattern_id>
```

To see which scenarios are already tagged to a pattern:
```bash
pattern scenarios --pattern-id <pattern_id>
```

> **Important**: Review ALL patterns regardless of activity. A low-activity
> pattern (recently fixed or rarely observed) may still describe the
> same failure mechanism as your candidate symptom. Activity is irrelevant
> for normalization — only the label and evidence matter.

### Step 3: For each candidate symptom, perform 3-way judgment

Compare the candidate symptom against the existing pattern list and
classify it into exactly one of three categories:

#### (a) Same — candidate matches an existing pattern

The candidate symptom describes the same underlying behavioral issue as an
existing pattern, even if the wording differs.

**Criteria for "same":**
- Both would be fixed by the same code change
- Both describe the same failure mechanism at the same abstraction level
- Surface wording differences are immaterial (e.g., "plural ambiguity" vs
  "singular/multiple confusion" — same thing)

**Action**: Link to the existing pattern ID.

#### (b) Composition — candidate is a combination of existing patterns

The failure involves multiple independent issues that already have their
own patterns. The scenario fails because of pattern A AND pattern B
together.

**Criteria for "composition":**
- You can identify 2+ existing patterns that independently contribute
- Fixing either one alone might not fix the scenario, but each is a
  recognized independent issue
- The combination does NOT represent a fundamentally new failure mode

**Action**: Tag with multiple existing pattern IDs. Do NOT create a new
composite pattern.

#### (c) New — genuinely novel failure type

The candidate describes a failure that does NOT match any existing pattern
and is NOT a composition of existing ones.

**Criteria for "new":**
- No existing pattern describes the same failure mechanism
- The candidate cannot be decomposed into existing patterns
- The failure is specific enough to be actionable (not too abstract)

**Action**: Register as a new pattern using `pattern register --label "..."`.

### Step 4: Execute the decision

Based on your judgment, execute the appropriate `pattern` CLI commands.

> **Note**: The `pattern` command is pre-installed on PATH. Always use
> `--harness {candidate_idx}` for all tagging commands (the current
> candidate, as shown in the session prompt context).

**For (a) Same:**
```bash
# Tag the (harness, trace, scenario) tuple with the existing pattern
pattern tag --pattern-id <existing_id> \
  --harness <candidate_idx> \
  --trace <cycle_dir> \
  --scenario <scenario_id> \
  --root-cause "Brief root cause text preserved as evidence"
```

**For (b) Composition:**
```bash
# Tag with multiple pattern IDs (repeat --pattern-id for each)
pattern tag --pattern-id <id1> --pattern-id <id2> \
  --harness <candidate_idx> \
  --trace <cycle_dir> \
  --scenario <scenario_id> \
  --root-cause "Composite: issue A interacts with issue B"
```

**For (c) New:**
```bash
# Register a new pattern — prints the assigned pattern_id
pattern register --label "DESCRIPTIVE SYMPTOM LABEL"
# Example output: REGISTERED: a1b2c3d4

# Tag with the newly created pattern ID
pattern tag --pattern-id <new_id> \
  --harness <candidate_idx> \
  --trace <cycle_dir> \
  --scenario <scenario_id> \
  --root-cause "Root cause text..."
```

> **You do NOT need to call `pattern observe`** — observations are
> automatically derived by the outer loop from your tagging results.

**Verification** — after all tagging is complete, confirm the registry
state is consistent:
```bash
pattern list
pattern scenarios --pattern-id <id>
```

## Decision Guidelines

### Prefer "same" when in doubt between "same" and "new"

Over-splitting (creating too many patterns) fragments the signal and
prevents the sampler from building reliable per-pattern estimates.
If you're 60%+ confident it's the same issue, link to existing.

### Prefer "new" when in doubt between "same" and a forced match

Do NOT force a candidate into an existing pattern just because the list
is short. If the failure mechanism is clearly different, create a new
pattern. The cost of over-merge (hiding distinct issues under one label)
is worse than over-split in the long run.

### Use "composition" sparingly

Most failures have a single root cause. Composition is for cases where
two genuinely independent issues interact. Do not use composition just
because the scenario has multiple symptoms — those symptoms may share a
single root cause.

## Example Normalization

```
Existing patterns:
  abc123 — "Singular/plural ambiguity in entity count"
  def456 — "Missing pagination in search results"
  ghi789 — "Premature tool termination before validating output"

Candidate: "Agent counts only first-page results when multiple pages exist"
Judgment: (a) Same as def456 — both describe incomplete search due to
          missing pagination handling.
Action: pattern tag --pattern-id def456 ...

Candidate: "Agent sees 'the event' but matches multiple events without
           disambiguation, then only retrieves first page of matches"
Judgment: (b) Composition of abc123 + def456 — ambiguity issue AND
          pagination issue both contribute.
Action: pattern tag --pattern-id abc123 --pattern-id def456 ...

Candidate: "Agent creates duplicate tool calls for the same action
           within a single turn"
Judgment: (c) New — no existing pattern describes redundant tool invocation.
Action: pattern register --label "Redundant duplicate tool calls within
        single reasoning turn"
```
