## Prior Attempts on This Arm (Core)

When `task_selection.pulled_arm_id` in `.autosaddler/session_context.json`
names an arm, read `.autosaddler/curriculum/pull_history/<pulled_arm_id>.json`.
If it lists pulls from earlier iterations, this exact arm (same failure
pattern) was pulled in earlier iterations. That file is the COMPLETE history of
those pulls — each patched attempt's approach, its **dev-set** accuracy impact,
its per-scenario results, and the lessons recorded for it; plus pulls where the
scenarios already passed (no patch) or the attempt failed.

**Key constraint**: Do NOT repeat approaches that already failed to fix the
target, or that fixed the mini-batch but dropped dev-set accuracy. The
lessons in that file explain what didn't work. Build on approaches that fixed the target while keeping
dev-set accuracy flat or higher, focus on any still-failing scenario, and fix
any regressions they introduced. If prior pulls show the scenarios already
passing, treat the current failure as possibly intermittent — avoid
over-fitting a patch to a transient failure.
