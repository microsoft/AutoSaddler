### ActiveSaddler (Core)

ActiveSaddler is an automated curriculum-learning layer for budgeted harness
optimization. Rather than selecting training scenarios according to a fixed
schedule, it continuously adapts which scenarios should drive the next harness
update based on the current prepared harness, previously observed failures, and
the outcomes of earlier optimization attempts. Its objective is to allocate a
limited rollout budget to the harness weaknesses whose continued diagnosis and
repair are most likely to improve population-level performance.

ActiveSaddler represents recurring harness weaknesses as **failure-pattern
arms**. After each optimization iteration, failed execution traces are analyzed
and grouped into reusable symptom-level patterns informed by their underlying
harness-level root causes. Each arm maintains the scenarios that have exhibited
the corresponding pattern, together with its execution history, attempted
repairs, development-set outcomes, and subsequent observations. This allows
evidence to be pooled across scenarios that fail for the same reason and makes
the shared weakness a reusable optimization target.

At each iteration, ActiveSaddler makes two curriculum decisions:

1. **PULL or DRAW**: It decides whether to revisit a known failure-pattern arm or execute previously unseen scenarios that may reveal a new failure type.
2. **Arm selection after PULL**: If PULL is chosen, ActiveSaddler assigns a cardinal learning-progress score to every known arm. Each arm is scored with respect to the state of the harness scheduled for evaluation in the current iteration, considering the current severity and fixability of its associated failure pattern, the potential breadth of improvement if that pattern is addressed, and the risk of introducing regressions. The resulting score estimates the expected population-level improvement from targeting that arm under the current harness state.

The selected scenarios are then passed to the existing diagnosis-and-patching
pipeline. The resulting mini-batch behavior, development-set impact, patch
verdict, and reflection lessons are written back into the pattern and
evolution histories. As the harness changes, these outcomes update which
patterns remain relevant and how valuable another attempt on each pattern is.
Through this feedback loop, ActiveSaddler co-evolves the scenario curriculum
with the harness instead of treating scenario selection as a fixed
pre-optimization choice.

### Terminology (Core)

- **Failure-pattern arm**: A reusable symptom-level representation of a recurring
  harness weakness. An arm owns the scenarios tagged with that pattern and
  accumulates root-cause evidence, observations, pull attempts, patch outcomes,
  and development-set results.
- **Known-arm set (`P_t`)**: Failure-pattern arms discovered before iteration
  `t`. These are the candidates that can be scored and selected by a PULL.
- **Unseen pool (`U_t`)**: Training scenarios that have never been executed.
  A DRAW samples from this pool to search for failure types not represented by
  the known arms. Once all offline training scenarios have been explored and every arm has since
  been visited, prior successes that did not instantiate an arm become eligible
  for exploration again, analogous to additional epochs over seen examples in
  conventional offline learning: when the pool is empty and every
  current arm has been pulled since it emptied, executed scenarios that own no
  arm form the pool of the next draw epoch.
- **PULL / DRAW**: The curriculum action for one iteration. PULL exploits one
  known arm; DRAW explores unseen scenarios. Agent Session 3.5 records this
  decision before any optional arm scoring.
- **Learning-progress score (`phi`)**: The cardinal value used to sample a known
  arm after PULL. For Agent scoring it is
  `(severity + fixability + breadth + (1 - side_effect)) / 4`.
- **Outer loop**: The AutoSaddler engine, which owns sampling, evaluation,
  acceptance, and the recorded run history.
- **Scenario**: A training case. Files and session outputs identify it by its
  `case_id`.
- **Sessions**: Session 0 is the `evolve` session, Session 1 is the
  `diagnose_patch` session, Session 2 is the `reflect` session, Session 3 is
  the `extract_patterns` session, Session 3.5 is the `decide_arm` session, and
  Session 4 is the `score_arms` session.

### Iteration Flow (Core)

With ActiveSaddler, each iteration proceeds in the following order. It replaces
the order in which the core pipeline samples training cases before parent
selection.

1. **Session 0 — Candidate Selection**: Analyze prior candidates via
  `.autosaddler/history/`. Decide which candidate's
  code to build on. You may combine units from multiple candidates. The
  resulting Session 0-prepared harness (the working parent) is committed before
  curriculum decisions and sampling, so all later sessions operate on the
  exact same prepared code.
2. **Session 3.5 — Unseen Scenario Exploration**: If no known arm owns any
  scenario, the outer loop chooses DRAW without opening this session. Otherwise,
  inspect the Session 0-prepared harness, the number of discovered arms and
  unseen scenarios, the complete candidate-pattern registry, prior pull histories,
  and evidence for promising known patterns. Make one holistic PULL/DRAW
  decision with no fixed threshold and return exactly one `action`
  in the session output. This session is read-only with respect to the harness and must not
  rate arms. An
  impossible DRAW with an empty unseen pool falls back to PULL.
3. **Session 4 — Arm Scoring** (PULL only): DRAW skips this session. After PULL, Analyze the prepared harness, every
  known failure-pattern arm, complete prior pull histories, scenario histories,
  and development-set outcomes. Return the four consideration axes for every
  candidate arm in the session output; the outer loop computes the cardinal
  learning-progress score from them.
  Session 4 is read-only with respect to the harness; only registry score
  records may change.
4. **Mini-batch sampling**: The outer loop executes the Session 3.5 decision. A
  DRAW takes up to the configured batch size from the unseen pool of the
  current draw epoch. A PULL uses the newly recorded Session 4 scores, samples one arm with
  floored softmax probability, and selects up to the configured batch size of
  that arm's scenarios. The chosen action, arm, scores, probabilities, and
  scenario IDs are persisted in the sampler trace.
5. **Initial evaluation on mini-batch**: The outer loop evaluates the
  Session 0-prepared candidate on the selected mini-batch, producing
  per-scenario scores and agent traces.
6. **Session 1 — Diagnose + Patch**: Read the initial evaluation traces
  and the agent codebase to diagnose failing scenarios, apply targeted
  code patches, and return the intent in the session output. The `diagnose` skill guides root-cause
  analysis; the phase-appropriate patch skill (`capability-patch` or
  `steering-patch`) guides implementation. When this iteration pulled a known
  arm, the prompt also points to the complete prior pull history of that exact
  arm.
7. **Re-evaluation on mini-batch**: The outer loop re-evaluates the patched
  candidate on the same mini-batch, producing re-evaluation scores.
8. **Initial/re-evaluation comparison**: The outer loop computes
  per-scenario impacts (fixed, regressed, still_failing, still_passing)
  and records the patch verdict.
9. **Evaluation on full dev-set** (conditional): If the patch is accepted
  (re-evaluation score > initial score), the outer loop evaluates the
  candidate on the full development set to measure generalizability.
10. **Learning update**: The outer loop records the iteration in the run history,
   from which the scenario histories and the evolution history are derived.
11. **Session 2 — Reflection**: Analyze the initial/re-evaluation results
   and dev-set scores to extract lessons — what worked, what didn't, and
   why. Return the lessons in the session output. *Implementation note*: This
   step runs as deferred work after the iteration's evaluations.
12. **Session 3 — Pattern Extraction**: After reflection is complete, inspect
   every failed pre-patch and post-patch scenario. First extract candidate
   symptoms without viewing existing patterns; then normalize each candidate
   as the same as existing arms, a composition of existing arms, or a new arm.
   Register new patterns and tag each failed `(candidate, evaluation, scenario)` tuple.
  For each previously known pattern represented in the mini-batch, the outer
  loop records the associated scenarios that were evaluated, the subset still
  tagged after patching, and their post-patch activity fraction. It also records
  the distinct probe points used by the persisted curriculum state.
   *Implementation note*: like Session 2, Session 3 runs as deferred work.

### Strategy Configuration: agent (Core)

- **Scoring**: After choosing PULL, Session 4 scores every arm using the mean `(severity + fixability + breadth + (1 - side_effect)) / 4`.
- **Arm creation**: Session 3.5 decides PULL versus DRAW in its session output. DRAW skips Session 4; PULL proceeds to scoring.
- **Arm selection**: Existing arms are sampled with `softmax(phi / tau)` and a
  per-arm probability floor `min_prob`.

Session 3 creates atomic failure-pattern arms after reflection. The pattern
registry files are available only in Sessions 1, 3, 3.5, and 4; Session 1
reads only the pulled arm's pull history, and only the Sessions 3, 3.5, and 4
outputs change the registry.

### Mini-batch Evaluation (Core)

Each iteration evaluates a sampled mini-batch of scenarios (a subset of the
full training set). This keeps each iteration fast while allocating the limited
rollout budget to scenarios that are informative for the current prepared
harness. The initial evaluation and re-evaluation always use the exact same
selected scenario IDs, so score changes measure the effect of the Session 1
patch rather than a change in batch composition.

Unlike epoch shuffle, ActiveSaddler does not partition the full training set
into fixed batches. A DRAW selects up to the configured mini-batch size from
the persisted, deterministically ordered unseen pool of the current draw
epoch: never-executed scenarios first, then, in each later epoch, the executed
scenarios that own no arm. A PULL selects exactly one known failure-pattern arm and executes all
of its scenarios when it owns no more than the configured batch size, or a
uniform random subset when it owns more. A mini-batch may therefore be smaller
than the configured size when the unseen pool or selected arm contains fewer
scenarios.

## Pattern Registry (Core)

The pattern registry holds failure-pattern arms and their curriculum state.
Each pull-history file joins exact-arm pull records with Session 1 diagnoses and patch intents,
development-set impact, per-scenario results, and Session 2 lessons. The registry is read-only
workspace data under `.autosaddler/curriculum/`; `manifest.json` lists every file.

### Query files (Core)

| File | Purpose | When to use |
|---------|---------|-------------|
| `patterns.json` | List all patterns, observations, and scenarios | Review the complete known-arm set |
| `patterns/<pattern_id>.json` | Show one pattern's evidence and activity | Inspect a candidate arm |
| `pull_history/<pattern_id>.json` | Show the complete exact-arm pull history | Before deciding or rating arms |
| `cases.json` and `cases/<file>.json` | Show one scenario's history across iterations | Inspect a scenario's scores, root causes, and attempted fixes |
| `decisions.json` | Show every recorded PULL/DRAW decision | Review prior curriculum decisions |

### Session outputs (Core)

| Output | Session | Replaces |
|--------|---------|----------|
| `new_patterns`, `tags` | Session 3 | Pattern registration and tagging |
| `action`, `rationale` | Session 3.5 | Recording the PULL/DRAW decision |
| `scores` | Session 4 | Recording each arm's four axes |

## Skills (Core)

Skills are methodology guides supplied with each session. They provide
structured procedures for specific tasks — general techniques without
benchmark-specific content. The outer loop installs the appropriate skill set
based on the current phase.

### Available skills (Core)

| Skill | Used in | Purpose |
|-------|---------|---------|
| `history-analysis` | Sessions 0/1/2/3/3.5/4 | Analyze evolution history |
| `diagnose` | Session 1 | Root-cause analysis |
| `capability-patch` | Session 1 capability phase | Expand capabilities |
| `steering-patch` | Session 1 steering phase | Refine behavior |
| `patch-verification` | Session 1 | Validate patch safety |
| `symptom-extract` | Session 3 | Bias-free symptom extraction |
| `symptom-normalize` | Session 3 | Normalize atomic patterns |
| `progress-scoring` | Session 4 | Estimate learning progress |

### How to use skills (Core)

Read the skill's `SKILL.md` for its detailed methodology when performing
the corresponding task. Skills tell you WHAT to do and HOW; the session
prompt (provided separately at session start) tells you WHAT context to
work with.
