# Failure-Pattern Curriculum Context (Core)

This run selects training batches with an agent-driven failure-pattern
curriculum. Each failure pattern that owns at least one training case is an
arm. An iteration either draws never-executed training cases to discover new
failure patterns or pulls one known arm and evaluates that arm's cases.

`.autosaddler/session_context.json` contains a `task_selection` object with the
curriculum settings and, once the batch is sampled, the `sampling_action` and
`pulled_arm_id`. `.autosaddler/curriculum/manifest.json` indexes the
read-only pattern registry: `patterns.json` lists every pattern with its
activity, observations, cases, and latest scores; each pattern detail file
keeps its root-cause evidence; each pull-history file lists prior pulls of
that arm with their outcomes and the matching history iteration file.

During selection, the batch is not sampled yet, so `train_case_ids` is empty.
Choose the base from measured history and the known failure patterns rather
than from the next batch. During diagnosis of a pulled arm, read that arm's
pattern detail and complete pull history first: do not repeat approaches that
already failed on the same pattern, and build on attempts that fixed training
cases but lost development accuracy.
