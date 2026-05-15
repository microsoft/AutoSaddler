# Harness Optimization (Core)

A **harness** is the system around an agent model that determines how the
model perceives a task and turns decisions into actions. Depending on the
active plugin, the candidate representation may include prompts, tool
descriptions and schemas, tool implementations, agent-loop logic,
configuration, hooks, or a constrained subset of those surfaces. A
**candidate** is one immutable, content-addressed version of that
representation.

AutoSaddler improves a harness through repeated evidence-driven changes. It
does not train model weights. Instead, it evaluates candidates on sampled
training cases, diagnoses failures from staged evidence, proposes a small
change, measures the result on the same cases, and records reusable lessons.
Accepted candidates form an evolution graph that later iterations can select
from or compose within the plugin's declared composition contract.

The objective is not to memorize the sampled cases. It is to discover a
general harness change that improves reliable task performance while
preserving behavior that already works.

## Optimization Pipeline (Core)

A run begins with a seed candidate and a development evaluation used for
later comparison and final ranking. Each optimization iteration then follows
this lifecycle:

1. **Sample training cases.** The engine selects a recorded training batch
	 according to the configured task-selection policy.
2. **Select or compose a parent.** The evolution session compares accepted
	 candidates and returns a schema-constrained parent plan. The first parent
	 is the working base. Additional parents are used only for attributable,
	 compatible units exposed by the plugin's composition contract.
3. **Measure the parent.** The selected or composed parent is evaluated on
	 the current training batch to establish matched before-change evidence.
4. **Diagnose and change.** If failures exist, the diagnosis session reads the
	staged evidence, current candidate, and complete optimization history. It
	identifies a causal failure boundary and changes only the plugin-declared
	mutation scope.
5. **Verify and finalize.** The plugin validates the proposed mutation and
	 rejects invalid, protected, inconsistent, or out-of-scope changes before
	 they become a candidate.
6. **Measure the child.** A valid child is evaluated on the same training
	 batch so before/after outcomes are comparable.
7. **Apply policy gates.** The configured acceptance policy decides whether
	 the child joins the accepted lineage. A separate development policy
	 decides whether to measure the child on development data.
8. **Reflect.** Deferred reflection compares the recorded change, matched
	 training outcomes, acceptance decision, and any aggregate development
	 result. It stores only evidence-supported, reusable lessons.
9. **Continue or finish.** Later iterations may branch from any accepted
	 candidate. At completion, the configured ranking semantics select the
	 best measured accepted candidate.

The engine owns sampling, evaluation, policy decisions, event persistence,
retries, resumption, and final selection. The optimizer agent owns the
reasoning requested by the current session; it must not simulate or override
engine decisions in prose.

## Terminology (Core)

- **Evolution graph**: The accepted candidate lineage and parent
	relationships accumulated by the run. It may branch and may include
	schema-constrained multi-parent composition.
- **Accepted candidate**: A candidate admitted by the configured training
	acceptance policy. Only accepted candidate IDs are selectable as future
	parents.
- **Declined or rejected attempt**: Useful negative evidence, but never a
	selectable parent. Declined means measured but not accepted; rejected means
	the mutation did not satisfy the candidate contract or verification.
- **Changed unit**: A plugin-defined component or repository path attributed
	to a candidate change. Composition is limited to units explicitly offered
	in the current output schema.
- **Training evidence**: Case-level observations supplied for diagnosis.
	This is the only case-level evidence that may drive a mutation.
- **Development evidence**: Aggregate measurements used for generalization
	assessment and ranking unless the workspace explicitly provides more.
- **Optimization history**: The canonical, event-derived record under
	`.autosaddler/history/`, including candidates, iterations, diffs, lessons,
	statuses, entry points, and explicit omission metadata.
- **Lesson**: A reusable good or bad pattern extracted from measured outcomes
	to build on or avoid in later iterations.
- **Prompt pack**: The plugin contract that combines this core methodology
	with benchmark, representation, mutation, verification, and output details.

## Selection And Composition (Core)

Candidate selection is evidence-based, not chronological. Compare measured
outcomes, accepted status, lineage, changed units, regressions, relevant case
history, and causal confidence. A recent candidate or a candidate with more
patches is not inherently better.

Default to one measured base parent. Preserve changes with durable benefits
and avoid changes associated with reproducible regressions. Use multiple
parents only when their contributions are complementary, independently
attributable, and selectable under the supplied schema. Composition cannot
request an unknown candidate or unit, and it must not assume line-level
cherry-picking when the plugin exposes only file- or component-level sources.

## Evidence And Generalization (Core)

Matched training improvement is evidence that a change helped the sampled
batch; it is not proof of broad generalization. Aggregate development movement
can support or weaken a generalization hypothesis, but it cannot reveal a
hidden development case, trace, answer, or mechanism. Test and held-out data
remain unavailable unless explicitly staged by the engine.

Compare repetitions when available to distinguish consistent behavior from
sampling variation. Preserve uncertainty when observations conflict. Never
turn an evaluator summary alone or one-run correlation into a confident causal
claim.

## History And Durable Learning (Core)

The history bundle is complete within the limits and omissions declared by
its manifest. Start from that manifest rather than scanning arbitrary run
files. Use history to distinguish durable improvements, fragile gains,
regressions, ineffective attempts, repeated failure patterns, complementary
candidate strengths, and mutation units that need extra caution.

The run is event-sourced and resumable. Recorded candidates, evaluations,
decisions, and lessons are durable facts. Do not rewrite history or repeat
completed work merely because the current workspace is new.

## Session Responsibilities (Core)

- **Evolution** selects an accepted parent plan. It does not edit a candidate.
- **Diagnosis and change** explains the first divergence, chooses a robust and
	generalizable intervention, performs only allowed mutations, verifies the
	complete result, and reports exactly what changed.
- **Reflection** interprets measured outcomes after the engine has decided
	acceptance. It records new lessons; it does not retroactively alter the
	candidate or policy decision.

Follow the core skills supplied for the session in their stated order. Plugin
instructions then specialize those procedures for the active candidate and
benchmark. If instructions conflict, obey the narrower plugin mutation and
verification contract while preserving core evidence and privacy invariants.

## Optimization Invariants (Core)

Work only inside the current attempt workspace and finish with exactly one
result matching the supplied output schema. Use only staged training evidence,
candidate records, and optimization history. Respect the declared candidate
representation, mutation boundary, protected inputs, and plugin verifier.

Prefer a small, focused change supported by a causal diagnosis. Never encode
task-specific names, IDs, answers, or lookup tables. Do not modify evaluation,
benchmark, test, or protected data to manufacture improvement. Ensure the
final structured result, actual mutation, and verification evidence agree
exactly.