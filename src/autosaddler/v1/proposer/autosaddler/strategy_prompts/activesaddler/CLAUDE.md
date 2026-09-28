# Harness Optimization

A **harness** is the agent codebase that wraps an LLM to solve benchmark
tasks — it includes the system prompt, tool definitions (docstrings),
tool implementations, agent loop logic, and hook configurations. The
harness determines how the LLM perceives tools, plans actions, and
handles edge cases. A **candidate** is a specific version of this harness
codebase, stored as an isolated git worktree.

This pipeline iteratively optimizes a harness to improve its performance
on a target benchmark. Each iteration evaluates a candidate on a
mini-batch of scenarios, diagnoses failures from execution traces,
applies targeted code patches to produce a new candidate, re-evaluates to
verify fixes, and reflects on the evaluation results to extract reusable
lessons — all accumulated across iterations via an evolution DAG.

## Optimization Pipeline

Similar to how machine learning improves a model through iterative
gradient-based updates on mini-batches, this pipeline improves an agent's
codebase through iterative feedback-driven patches. Each iteration
evaluates the agent on a mini-batch of scenarios, uses the execution
traces as feedback to diagnose failures and apply targeted code patches,
re-evaluates to measure impact, and reflects on the results to extract
reusable lessons. These lessons persist across iterations, guiding future
patches away from known pitfalls and toward proven strategies.

### ActiveSaddler

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
verdict, and per-scenario reflections are written back into the pattern and
evolution histories. As the harness changes, these outcomes update which
patterns remain relevant and how valuable another attempt on each pattern is.
Through this feedback loop, ActiveSaddler co-evolves the scenario curriculum
with the harness instead of treating scenario selection as a fixed
pre-optimization choice.

### Terminology

- **Evolution DAG**: A directed acyclic graph tracking all candidates and
  their relationships. Each **node** is a candidate codebase (C0 = seed,
  C1, C2, ...), each **edge** is a patch from parent to child.
- **Lessons**: Good/bad patterns extracted from reflections, persisted in
  the DAG. Future iterations consult these to avoid repeating mistakes.
- **Scenario registry**: Per-scenario history of scores, root causes, and
  attempted fixes across all iterations — prevents re-trying failed
  approaches on the same or similar scenarios.
- **Patch verdict**: Per-iteration assessment with two dimensions:
  **effectiveness** (did targeted scenarios improve?) and **safety** (zero
  regressions?). Candidates that pass mini-batch acceptance (re-evaluation
  score > initial score) are further evaluated on the development set to
  measure **generalizability** — whether the patch also helps scenarios with
  similar root causes beyond the mini-batch.
- **Phase**: Determines which patch type is encouraged — **capability**
  (expand what the agent can do) or **steering** (refine how it behaves).
  See Phase System below.
- **Skills**: Methodology guides (`.claude/skills/`) providing step-by-step
  procedures for diagnosis, patching, verification, and reflection.
- **Worktree**: Each candidate gets its own isolated git worktree, forked
  from its parent candidate.
- **Failure-pattern arm**: A reusable symptom-level representation of a recurring
  harness weakness. An arm owns the scenarios tagged with that pattern and
  accumulates root-cause evidence, observations, pull attempts, patch outcomes,
  and development-set results.
- **Known-arm set (`P_t`)**: Failure-pattern arms discovered before iteration
  `t`. These are the candidates that can be scored and selected by a PULL.
- **Unseen pool (`U_t`)**: Training scenarios that have never been executed.
  A DRAW samples from this pool to search for failure types not represented by
  the known arms.
- **PULL / DRAW**: The curriculum action for one iteration. PULL exploits one
  known arm; DRAW explores unseen scenarios. Agent Session 3.5 records this
  decision before any optional arm scoring.
- **Learning-progress score (`phi`)**: The cardinal value used to sample a known
  arm after PULL. For Agent scoring it is
  `(severity + fixability + breadth + (1 - side_effect)) / 4`.

### Iteration Flow

Each iteration proceeds in the following order:

1. **Session 0 — Candidate Selection**: Analyze prior candidates via
  `evo-dag summary` and `evo-dag show history`. Decide which candidate's
  code to build on. You may combine ideas from multiple candidates. The
  resulting Session 0-prepared harness is committed before curriculum
  decisions and sampling, so all later sessions operate on the exact same
  prepared code.
2. **Session 3.5 — Unseen Scenario Exploration**: If no known arm owns any
  scenario, the outer loop chooses DRAW without opening this session. Otherwise,
  inspect the Session 0-prepared harness, the number of discovered arms and
  unseen scenarios, the complete candidate-pattern table, prior pull histories,
  and evidence for promising known patterns. Make one holistic PULL/DRAW
  decision with no fixed threshold and record exactly one `pattern decide`
  command. This session is read-only with respect to the harness and must not
  rate arms. An
  impossible DRAW with an empty unseen pool falls back to PULL.
3. **Session 4 — Arm Scoring** (PULL only): DRAW skips this session. After PULL, Analyze the prepared harness, every
  known failure-pattern arm, complete prior pull histories, scenario histories,
  and development-set outcomes. Record a cardinal learning-progress score and
  the four consideration axes for every candidate arm with `pattern rate`.
  Session 4 is read-only with respect to the harness; only registry score
  records may change.
4. **Mini-batch sampling**: The outer loop executes the Session 3.5 decision. A
  DRAW takes up to the configured batch size from the never-executed scenario
  pool. A PULL uses the newly recorded Session 4 scores, samples one arm with
  floored softmax probability, and selects up to the configured batch size of
  that arm's scenarios. The chosen action, arm, scores, probabilities, and
  scenario IDs are persisted in the sampler trace.
5. **Initial evaluation on mini-batch**: The outer loop evaluates the
  Session 0-prepared candidate on the selected mini-batch, producing
  per-scenario scores and agent traces.
6. **Session 1 — Diagnose + Patch**: Read the initial evaluation traces
  and the agent codebase to diagnose failing scenarios, apply targeted
  code patches, write reasoning to `proposer_reasoning.md`, and record
  intent via `evo-dag update-intent`. The `diagnose` skill guides root-cause
  analysis; the phase-appropriate patch skill (`capability-patch` or
  `steering-patch`) guides implementation. When this iteration pulled a known
  arm, the prompt also includes the complete prior pull history of that exact
  arm.
7. **Re-evaluation on mini-batch**: The outer loop re-evaluates the patched
  worktree on the same mini-batch, producing re-evaluation scores.
8. **Initial/re-evaluation comparison**: The outer loop computes
  per-scenario impacts (fixed, regressed, still_failing, still_passing)
  and records the patch verdict.
9. **Evaluation on full dev-set** (conditional): If the patch is accepted
  (re-evaluation score > initial score), the outer loop evaluates the
  candidate on the full development set to measure generalizability.
10. **Learning update**: The outer loop updates the scenario registry and
   saves the evolution DAG.
11. **Session 2 — Reflection**: Analyze the initial/re-evaluation results
   and dev-set scores to extract lessons — what worked, what didn't, and
   why. Record reflections via `evo-dag update-reflection` for every
   scenario. *Implementation note*: This step is deferred to the start of
   the next iteration because the dev-set evaluation runs asynchronously
   in the outer loop.
12. **Session 3 — Pattern Extraction**: After reflection is complete, inspect
   every failed pre-patch and post-patch scenario. First extract candidate
   symptoms without viewing existing patterns; then normalize each candidate
   as the same as existing arms, a composition of existing arms, or a new arm.
   Register new patterns and tag each failed `(harness, trace, scenario)` tuple.
  For each previously known pattern represented in the mini-batch, the outer
  loop records the associated scenarios that were evaluated, the subset still
  tagged after patching, and their post-patch activity fraction. It also records
  the distinct probe points used by the persisted curriculum state.
   *Implementation note*: like Session 2, Session 3 runs at the start of the
   next iteration.

### Strategy Configuration: agent

- **Scoring**: After choosing PULL, Session 4 scores every arm using the mean `(severity + fixability + breadth + (1 - side_effect)) / 4`.
- **Arm creation**: Session 3.5 decides PULL versus DRAW with `pattern decide`. DRAW skips Session 4; PULL proceeds to scoring.
- **Arm selection**: Existing arms are sampled with `softmax(phi / tau)` and a
  per-arm probability floor `min_prob`.

Session 3 creates atomic failure-pattern arms after reflection. Pattern tools are
available only in the Sessions 3, 3.5, and 4 processes that receive them on `PATH`.

### Evolution DAG

The optimization history is stored as a directed acyclic graph rather than
a linear sequence. This is because a single patch may not be universally
better — one candidate might excel at email scenarios while another handles
calendar tasks better. The DAG structure enables:

- **Branching**: Multiple candidates can be derived from the same parent,
  exploring different patch strategies in parallel.
- **Cherry-picking**: Session 0 can combine ideas from multiple candidates
  (e.g., copy a tool fix from C2 into a fork of C3), creating
  cherry-pick edges in the DAG via `evo-dag update-selection`.
- **Informed selection**: Each node records its scores, patch intent,
  verdict, and reflections. Session 0 reads this history to decide which
  candidate to build on — choosing the best base for the current
  mini-batch rather than always following a single lineage.

The DAG also serves as a persistent knowledge store: lessons and the
scenario registry are accumulated DAG-wide, not per-branch. This means
when diagnosing a failure, prior root causes and attempted fixes for the
same or similar scenarios are already available; when patching, accumulated
good/bad patterns help avoid known pitfalls and build on proven strategies
— regardless of which lineage the current iteration belongs to.

### Mini-batch Evaluation

Each iteration evaluates a sampled mini-batch of scenarios (a subset of the
full training set). This keeps each iteration fast while allocating the limited
rollout budget to scenarios that are informative for the current prepared
harness. The initial evaluation and re-evaluation always use the exact same
selected scenario IDs, so score changes measure the effect of the Session 1
patch rather than a change in batch composition.

Unlike epoch shuffle, ActiveSaddler does not partition the full training set
into fixed batches. A DRAW selects up to the configured mini-batch size from
the persisted, deterministically ordered pool of scenarios that have never been
executed. A PULL selects exactly one known failure-pattern arm and executes all
of its scenarios when it owns no more than the configured batch size, or a
uniform random subset when it owns more. A mini-batch may therefore be smaller
than the configured size when the unseen pool or selected arm contains fewer
scenarios.

### Phase System

The pipeline uses a learning-rate-scheduler analogy with two phases:
- **Capability phase** (early iterations): Expand what the agent CAN DO — new
  tools, parameters, implementation fixes, infrastructure changes.
  Uses the `capability-patch` skill.
- **Steering phase** (later iterations): Refine HOW the agent behaves — prompt
  rules, tool descriptions, PreToolUse hooks.
  Uses the `steering-patch` skill.

**Why phases are necessary**: Without explicit phase guidance, LLM-based
proposers gravitate toward prompt patches — they are the lowest-effort
change with an immediate (but shallow) reward signal. When a prompt patch
succeeds, the evolutionary DAG exploits that signal and keeps selecting
prompt-patch candidates as parents, creating a self-reinforcing loop of
prompt-only optimization. This leads to a local optimum where the agent's
fundamental capabilities never expand. The phase system forces capability
patches early — adding new tools, parameters, and implementation fixes —
before steering patches refine behavior within the expanded action space.
Capability patches have higher variance but unlock scenarios that are
unreachable through prompt tuning alone. Once the action space is
sufficiently expanded, the steering phase fine-tunes how the LLM uses
those capabilities — correcting tool descriptions, adding prompt rules,
and inserting hook reminders to eliminate recurring behavioral mistakes.

### Acceptance Criteria

**Mini-batch acceptance**: A patch is accepted if the sum of re-evaluation
scores exceeds the sum of initial scores on the mini-batch. Even a single
scenario improvement is enough.

A patch that fixes one scenario but regresses another may still pass
mini-batch acceptance (net positive), but the regression is recorded as a
**bad pattern** in lessons, penalizing similar approaches in future
iterations.

### How Learning Accumulates

- **Lessons**: Good/bad patterns extracted from reflections. Visible in
  `evo-dag show history`. Future iterations use these to avoid repeating
  mistakes and to build on successes.
- **Scenario Registry**: Query with `evo-dag show scenario <id>`. Prevents
  re-attempting approaches that already failed on a specific scenario.
- **Patch History**: Query with `evo-dag show history`. Full diff +
  reflection history per iteration — provides context on what was tried
  and what worked.

## evo-dag CLI

The `evo-dag` command is pre-installed on PATH. As candidates and traces
accumulate over iterations, navigating them directly becomes impractical.
The CLI provides quick access to the DAG's structured summaries — patch
history, lessons, scenario registry, and diffs — so you can identify
what's relevant before diving into specific raw traces or source files
for detailed analysis. Run `evo-dag help` for the full command list.

### Query commands

| Command | Purpose | When to use |
|---------|---------|-------------|
| `evo-dag summary` | DAG topology, best candidate, all edges | Start of any session — quick orientation |
| `evo-dag show history` | Full patch history: diffs, reflections, lessons | Deep dive into what was tried and learned |
| `evo-dag show node <idx>` | Node details: scores, intent, verdict, output dirs | Inspecting a specific candidate |
| `evo-dag show edge <parent> <child>` | Code diff, per-scenario impacts, files changed | Understanding what a specific patch changed |
| `evo-dag show scenario <id>` | Per-scenario history, root causes, attempted fixes | Before diagnosing a failing scenario |
| `evo-dag show current-batch` | Current mini-batch scenario IDs, output dirs | Orienting to the current iteration |
| `evo-dag show lineage` | DAG lineage visualization with edge types | Understanding branching structure |
| `evo-dag show lessons` | Accumulated good/bad patterns | Checking known patterns before patching |

### Update commands

```bash
# Record candidate selection (Session 0)
evo-dag update-selection \
  --parent-candidates "2" \
  --reasoning "C2 has best score and fewest regressions"

# Record patch intent (Session 1, after applying patches)
evo-dag update-intent \
  --target-scenarios "id1,id2" \
  --diagnosis "Root cause analysis" \
  --approach "What you changed and why" \
  --files-changed "f1.py,f2.py" \
  --change-summary "Summary of code changes"

# Record per-scenario reflection (Session 2)
evo-dag update-reflection \
  --node <candidate_idx> \
  --scenario "<id>" \
  --status "fixed|regressed|still_failing|still_passing" \
  --root-cause "Why it was failing" \
  --explanation "What happened after the patch" \
  --prevention-or-next "Lesson or next step" \
  --generalization-note "Dev set accuracy changed from X to Y, likely because this patch affects ..."

# Batch-record unaffected passing scenarios (Session 2)
evo-dag update-reflection --node <candidate_idx> --batch-still-passing "id1,id2,..."
```

## pattern CLI

The `pattern` command manages failure-pattern arms and their curriculum state.
`pattern history` joins exact-arm pull records with Session 1 diagnoses and patch intents,
development-set impact, and Session 2 reflections. It is available only in
Sessions 3, 3.5, and 4.

### Query commands

| Command | Purpose | When to use |
|---------|---------|-------------|
| `pattern list` | List all patterns, observations, and scenarios | Review the complete known-arm set |
| `pattern show <id>` | Show one pattern's evidence and activity | Inspect a candidate arm |
| `pattern history [--pattern-id <id> ...] [--last-k <K>]` | Show complete or recent exact-arm pull histories | Before deciding or rating arms |
| `pattern score [--top-k <N>]` | Rank current mechanical activity scores | Initial prioritization only |
| `pattern scenarios --pattern-id <id>` | List scenarios owned by one arm | Inspect arm membership |

### Update commands

```bash
pattern register --label "Descriptive symptom-level label"
pattern tag --pattern-id <pattern_id> --harness <candidate_idx> \
  --trace <cycle_dir> --scenario <scenario_id> --root-cause "Root cause evidence"
pattern decide --action pull --rationale "Why a known arm is preferable"
pattern rate --pattern-id <pattern_id> --severity <0-to-1> \
  --fixability <0-to-1> --breadth <0-to-1> --side-effect <0-to-1> \
  --rationale "Score justification"
```

## Skills

Skills are methodology guides installed into `.claude/skills/` in the
worktree. They provide structured procedures for specific tasks — general
techniques without benchmark-specific content. The outer loop installs the
appropriate skill set based on the current phase.

### Available skills

| Skill | Used in | Purpose |
|-------|---------|---------|
| `history-analysis` | Sessions 0/1/2/3/3.5/4 | Analyze evolution history |
| `diagnose` | Sessions 1/2 | Root-cause analysis |
| `capability-patch` | Session 1 capability phase | Expand capabilities |
| `steering-patch` | Session 1 steering phase | Refine behavior |
| `patch-verification` | Session 1 | Validate patch safety |
| `symptom-extract` | Session 3 | Bias-free symptom extraction |
| `symptom-normalize` | Session 3 | Normalize atomic patterns |
| `progress-scoring` | Session 4 | Estimate learning progress |

### How to use skills

Skills are discovered automatically via the `.claude/skills/` directory.
Read the skill's `SKILL.md` for its detailed methodology when performing
the corresponding task. Skills tell you WHAT to do and HOW; the session
prompt (provided separately at session start) tells you WHAT context to
work with.

# Benchmark: GAIA2

## Overview

GAIA2 is an agentic benchmark that evaluates AI agent capabilities in
simulated real-life assistant environments. Unlike the original GAIA (2023),
which was read-only information retrieval, GAIA2 is a **read-and-write**
benchmark focused on interactive behavior and complexity management — agents
must execute multi-step state-changing operations in environments where time
flows continuously and events occur dynamically.

## Dataset Structure

**Universes**: 10 distinct simulated user environments with pre-populated
data. Each universe is a **smartphone mock-up environment** simulating a
persona's daily life — with pre-populated conversation history, emails,
calendar events, contacts, and app interactions.

**Applications** (11): AgentUserInterface, MessagingApp, ChatsApp,
EmailClient, Calendar, Contacts, RentAFlat, Shopping, Cab, City,
FileSystem.

**Scenarios** (800 total): Tasks that a human user would ask their
assistant to perform within this environment.

**Evaluated capabilities** (5 categories, equal weight in final score):

| Capability | What it tests | Example |
|------------|---------------|---------|
| Execution | Multi-step planning and state changes | "Update all contacts aged 24 or younger to be one year older" |
| Search | Cross-source information gathering and synthesis | "Which city do most of my friends live in?" |
| Adaptability | Dynamic response to environmental changes | "Meet my friend to view a property. If she suggests another, replace it" |
| Time | Temporal reasoning and scheduling constraints | "Send messages to colleagues. If no response after 3 minutes, order a cab" |
| Ambiguity | Handling unclear, contradictory, or impossible tasks | "Schedule Yoga each day 6 PM Oct 16-21. Ask me if there are conflicts" |

**Our setup**: 1 universe as training set (~75 scenarios), a different
universe as development set (~65 scenarios) for generalization evaluation.

## Scoring

Each scenario compares the agent's tool-call trace against an oracle
(reference) trace using a per-event graph judge with hard + soft checkers.

### Judge System (GraphPerEventJudge)

The judge follows a strict pipeline:

1. **Preliminary check**: Compare per-app tool call counts (agent vs oracle).
   If counts differ → `ToolCallCountsFailure` (immediate fail).
2. **Topological sort**: Oracle events are ordered by their dependency DAG.
3. **Event matching**: For each oracle event, find the best-matching agent
   event by tool name and time proximity.
4. **Argument validation**: Run the tool-specific judge on each argument
   (see checker types below).
5. **Dependency verification**: Confirm all parent oracle events were already
   matched before this event.
6. **Timing check**: If events are >1 s apart, allow 10 s early / 25 s late
   tolerance.

Extra agent actions are mostly tolerated (1 extra `send_message_to_user`
allowed), but **missing required actions** or **wrong arguments** cause
failure.

### Checker Types

Each tool argument is validated by a **MildToolJudge**: run the hard checker
first; if it fails, return failure immediately. If it passes, run the soft
checker for semantic validation.

**Hard checkers** (deterministic):

| Checker | Behavior | Typical args |
|---------|----------|--------------|
| `eq_checker` | Exact equality | IDs, order_id, item_id |
| `datetime_checker` | Exact match in `YYYY-MM-DD HH:MM:SS` format | start/end times |
| `phone_number_checker` | Extracts digits only (formatting ignored) | phone numbers |
| `path_checker` | `os.normpath()` normalization | file paths |
| `unordered_list_checker` | Set equality (order-independent) | recipients, attendees |
| `list_attendees_checker` | Unordered + user's own name optional | calendar attendees |
| `unordered_path_list_checker` | Set equality on normalized paths | attachment lists |
| `eq_str_strip_checker` | Strip whitespace then equality | names, discount codes |
| `contain_any_checker` | Any target substring found (case-insensitive) | scripted content |
| `contain_all_checker` | All targets found (case-insensitive) | scripted content |

**Soft checkers** (LLM-based, semantic):

| Checker | What it validates |
|---------|-------------------|
| `content_checker` | Semantic equivalence (ignores style/tone differences) |
| `placeholder_checker` | Rejects unfilled placeholders like "[Your Name]" → **hard fail** |
| `signature_checker` | Rejects fake signatures ("Your assistant", "User") |
| `tone_checker` | Checks message tone appropriateness |
| `sanity_checker` | Detects reward hacking (garbled text, technical artifacts) |
| `cab_checker` | Address flexibility ("Home" → user's address) |

### Per-App Argument Validation Rules

```
send_email:
  recipients       → unordered_list_checker
  cc               → unordered_list_checker
  attachment_paths  → unordered_path_list_checker
  subject/content   → content_checker + placeholder_checker + signature_checker + tone_checker

add_calendar_event:
  start_datetime    → datetime_checker  (YYYY-MM-DD HH:MM:SS exactly)
  end_datetime      → datetime_checker
  attendees         → list_attendees_checker  (user name omittable)
  title/description → event_checker  (semantic)

send_message:
  user_id           → eq_checker  (exact match)
  content           → placeholder_checker + tone_checker + message_checker

add_to_cart / checkout:
  item_id           → eq_checker
  quantity          → eq_checker  (exact integer)
  discount_code     → eq_str_strip_checker

add_new_contact:
  first/last_name   → eq_str_strip_checker
  phone             → phone_number_checker  (digits only)
  email             → eq_str_strip_checker
```

Knowing which checker validates each argument lets you diagnose whether a
failure is a **format issue** (hard checker) or a **semantic issue** (soft
checker), which determines the right patch approach.

## Iteration Output Structure

Each iteration produces output directories with the following layout:

**Session root** (shared across iterations):
```
<session_root>/
├── evolution_dag.json.gz         ← Compressed DAG state (all nodes, edges, lessons)
├── pattern_registry.json.gz      ← Patterns, observations, and scores
├── bandit_state.json             ← Deterministic modern-bandit state
└── bin/
  ├── evo-dag                   ← Evolution DAG CLI tool
  └── pattern                   ← PatternRegistry CLI tool
```

**Per-evaluation cycle directories** (named `iter*_train_before_*` and
`iter*_train_after_*`):
```
<cycle_dir>/
├── iter<N>_c<X>_selection.json                    ← SDK session log (Session 0)
├── iterNN_cX_unseen_scenario_exploration.json     ← SDK session log (Session 3.5)
├── iterNN_cX_arm_scoring.json                     ← SDK session log (Session 4)
├── iterNN_cX_arm_decision.json                    ← Agent pull/draw decision
├── iterNN_cX_sampler_trace.json                   ← Per-iteration sampler decision
├── iter<N>_c<X>_patch.json                        ← SDK session log (Session 1)
├── iter<N>_c<X>_reflection.json                   ← SDK session log (Session 2)
├── iterNN_cX_pattern_extraction.json              ← SDK session log (Session 3)
└── run/
    ├── output.jsonl             ← Per-scenario scores + rationale (JSONL)
    ├── benchmark_stats.json     ← Aggregate statistics
    ├── lite/                    ← Agent execution traces (tool calls, reasoning)
    │   └── <scenario_id>.json
    └── hf/                      ← Environment state (app data, events)
        └── <scenario_id>.json
```

## Agent Framework (Meta-ARE)

The working directory is a git worktree of the Meta-ARE agent codebase.

### Meta-ARE Architecture

A ReAct JSON agent runs in a simulated environment with 11 apps (Email,
Calendar, Contacts, Messaging, Shopping, etc.). Each app exposes methods
decorated with `@app_tool()`, which become tools named
`AppName__method_name` (e.g., `EmailClient__send_email`). The agent's LLM
sees only tool descriptions derived from docstrings — never the
implementation code. The agent loop is: pull notifications → call LLM →
parse JSON action → execute tool → observe result → repeat, up to 80
iterations.

**Docstring = Tool Description**: The agent **never sees Python source
code**. It only sees tool descriptions derived from Python docstrings.
This has critical implications:

- Modifying a docstring directly changes the agent's understanding of a tool.
- Adding constraints, format requirements, or return value descriptions to a
  docstring is one of the most effective ways to fix behavioral errors.
- **Crash hazard**: Placeholders like `{{...}}`, `[...]`, or `{{{{...}}}}` in
  docstrings will crash downstream parsers that call `.format()`. Always use
  concrete examples instead.

### Hook Configuration

The agent framework supports **PreToolUse hooks** that inject just-in-time
reminders before specific tool calls. Hooks are configured via `hook.json`
in the worktree root — the evaluation harness detects this file
automatically.

**Format:**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "<regex matching tool name>",
        "description": "<purpose of this hook>",
        "hooks": [
          {
            "type": "reminder",
            "reminder": "<concise hint text>"
          }
        ]
      }
    ]
  }
}
```

**Key details:**
- **Handler type**: Must be `"reminder"` (not `"message"`). Injects a
  static hint once per tool name — the first call receives the hint,
  subsequent calls pass through.
- **Matcher**: Regex matched against tool names. Use exact names for
  single tools, or patterns like `AppName__.*` for all tools of an app.
- If `hook.json` already exists from a prior candidate, **read it first**
  and merge new entries — do not overwrite existing hooks unless
  intentionally removing them.
- Keep hint text concise (under 2 sentences), no hardcoded answers.

### Editable paths

Agent behavior, capabilities, and tool definitions:
  - `are/simulation/agents/default_agent/prompts/additional_prompts.py`
  - `are/simulation/agents/default_agent/base_agent.py`
  - `are/simulation/agents/default_agent/agent_factory.py`
  - `are/simulation/agents/default_agent/are_simulation_main.py`
  - `are/simulation/agents/default_agent/tools/json_action_executor.py`
  - `are/simulation/agents/default_agent/termination_methods/are_simulation.py`
  - `are/simulation/agents/default_agent/steps/are_simulation.py`
  - `are/simulation/agents/are_simulation_agent_config.py`
  - `are/simulation/agents/agent_config_builder.py`
  - `are/simulation/apps/agent_user_interface.py`
  - `are/simulation/apps/apartment_listing.py`
  - `are/simulation/apps/app.py`
  - `are/simulation/apps/cab.py`
  - `are/simulation/apps/calendar.py`
  - `are/simulation/apps/calendar_v2.py`
  - `are/simulation/apps/city.py`
  - `are/simulation/apps/contacts.py`
  - `are/simulation/apps/email_client.py`
  - `are/simulation/apps/messaging.py`
  - `are/simulation/apps/messaging_v2.py`
  - `are/simulation/apps/reminder.py`
  - `are/simulation/apps/sandbox_file_system.py`
  - `are/simulation/apps/shopping.py`
  - `are/simulation/apps/system.py`
  - `are/simulation/apps/virtual_file_system.py`
  - `hook.json`

### Protected paths (do NOT modify)

Simulation engine, evaluation, and ground truth:
  - `are/simulation/validation/`
  - `are/simulation/scenarios/`
  - `are/simulation/benchmark/`
  - `are/simulation/benchmark.py`
  - `are/simulation/tests/`
  - `are/simulation/data/`
  - `are/simulation/data_handler/`
  - `are/simulation/checkpoint/`
  - `are/simulation/tutorials/`
  - `are/simulation/gui/`

# Constraints (global hard rules)
- Do NOT modify protected paths
- Do NOT hard-code task-specific answers or lookup tables
- **Training set**: Mini-batch scores and raw execution traces are available.
  Use traces for diagnosis and patching. Output dirs named
  `iter*_train_before_*` and `iter*_train_after_*`.
- **Development set**: Only aggregate accuracy is available (no traces).
  Use for generalizability analysis in reflections. Output dirs named
  `iter*_val_*` or `seed_val_*`. Do NOT read raw traces from these dirs.
- Do NOT modify files in iteration output dirs or other worktrees (they are read-only records)
- Do NOT modify files in the base repository — only modify files in your
  current working directory (this worktree).
- Record selection via `evo-dag update-selection` before finishing (Session 0)
- Write reasoning to `proposer_reasoning.md` and record intent via
  `evo-dag update-intent` before finishing (Session 1)
- Record reflections via `evo-dag update-reflection` for every scenario (Session 2)
- Tag every failed scenario with `pattern tag` in Session 3.
- Do not inspect existing patterns during symptom extraction; only during normalization.
- In Session 3.5, call `pattern decide` exactly once and do not rate arms.
- After DRAW, skip Session 4. After PULL, rate every candidate arm in Session 4.
