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
training cases, diagnoses failures or avoidable cost from staged evidence, proposes a small change, measures the result on the same cases, and records reusable lessons. Development-evaluated candidates form a Pareto archive whose newest nondominated member is the working base for later iterations.

The objective is not to memorize the sampled cases. It is to grow a useful
accuracy-cost Pareto frontier across the full task distribution, not just the
current training batch, through general harness changes.

This run uses a non-scalarized accuracy-cost objective: maximize accuracy and
minimize mean task-agent USD cost. Joint improvements, accuracy improvements,
cost improvements, and either direction of accuracy-cost trade-off are valid
directions. Do not create cheap failure by removing required work or validation.

## Optimization Pipeline (Core)

A run begins with a seed candidate and a development evaluation used for
initializing the archive/current, later comparison, and final selection. Each optimization iteration then follows
this lifecycle:

1. **Sample training cases.** The engine selects a recorded training batch
	 according to the configured task-selection policy.
2. **Use the fixed current.** The engine supplies the newest nondominated
	 archive member as the working base. From the second iteration, evolve may
	 prepare this fixed fork locally using eligible archive mechanisms.
3. **Measure the parent.** The prepared candidate (or unchanged current) is 	evaluated on
	 the current training batch to establish matched before-change evidence.
4. **Diagnose and change.** Even if all cases pass, the diagnosis session reads the
	staged evidence, current candidate, and complete optimization history. It
	identifies a causal failure boundary or cost mechanism and changes only the plugin-declared
	mutation scope.
5. **Verify and finalize.** The plugin validates the proposed mutation and
	 rejects invalid, protected, inconsistent, or out-of-scope changes before
	 they become a candidate.
6. **Measure the child.** A valid child is evaluated on the same training
	 batch so before/after outcomes are comparable.
7. **Apply policy gates.** The configured Bayesian bootstrap screen permits
	 bounded accuracy loss and decides promotion to development evaluation,
	 not archive membership. The engine updates the nondominated archive using
	 development score and mean task-agent cost.
8. **Reflect.** Deferred reflection compares the recorded change, matched
	 training outcomes, promotion decision, and any aggregate development score/cost
	 result. It stores only evidence-supported, reusable lessons.
9. **Continue or finish.** Later iterations use the newest nondominated
	 archive member as current. At completion, final selection independently
	 chooses minimum cost for `cost`, returns the full archive for `pareto`,
	 or chooses maximum score for other preferences.

The engine owns sampling, evaluation, policy decisions, event persistence,
retries, resumption, and final selection. The optimizer agent owns the
reasoning requested by the current session; it must not simulate or override
engine decisions in prose.

## Terminology (Core)

- **Evolution graph**: The candidate lineage and parent
	relationships accumulated by the run. Preparation descends from fixed current;
	diagnosis descends from prepared content. Donors are provenance, not extra parents.
- **Accepted candidate**: A candidate admitted by the configured training
	acceptance policy for development evaluation (promoted). Promotion does not
	guarantee archive membership or becoming current.
- **Archive member**: Nondominated under recorded development score/cost.
	A candidate dominates another when it is no worse in either objective and
	strictly better in at least one.
- **Current candidate**: The engine-supplied working base for this iteration.
- **Selected candidate**: The final candidate chosen independently of search current,
	absent when the final preference returns the full archive.
- **Declined or rejected attempt**: Useful negative evidence, but never a
	replacement for the fixed current. Declined means measured but not promoted; rejected means
	the mutation did not satisfy the candidate contract or verification.
- **Changed unit**: A plugin-defined component or repository path attributed
	to a candidate change.
- **Training evidence**: Case-level observations supplied for diagnosis.
	This is the only case-level evidence that may drive a mutation.
- **Development evidence**: Aggregate measurements used for generalization
	assessment and archive membership, limited to supplied score/cost aggregates.
- **Optimization history**: The canonical, event-derived record under
	`.autosaddler/history/`, including candidates, iterations, diffs, lessons,
	statuses, entry points, and explicit omission metadata.
- **Lesson**: A reusable good or bad pattern extracted from measured outcomes
	to build on or avoid in later iterations.
- **Prompt pack**: The plugin contract that combines this core methodology
	with benchmark, representation, mutation, verification, and output details.

## Fixed Current (Core)

The engine fixes current to the newest nondominated archive member. Do not
select or compose another parent, replace the base, or copy another candidate's
whole workspace. Historical candidates and diffs are evidence for localized
changes, not permission to transplant a donor workspace.

Preserve changes with durable benefits and avoid changes associated with
reproducible regressions. Compare measured outcomes, promotion, archive membership,
lineage, changed units, relevant case history, and causal confidence without
changing the engine's current selection.

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

For deferred reflection, use the owning iteration's recorded working and
carried-forward current, not a later history summary's current.

## Session Responsibilities (Core)

- **Diagnosis and change** explains the first divergence or cost mechanism, chooses a robust and
	generalizable intervention, performs only allowed mutations, verifies the
	complete result, and reports exactly what changed.
- **Reflection** interprets measured outcomes after the engine has decided
	promotion and any archive update. It records new lessons; it does not retroactively alter the
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