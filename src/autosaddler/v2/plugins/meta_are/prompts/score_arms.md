# Meta-ARE Arm Scoring Contract (Plugin-specific)

Read `.autosaddler/session_context.json` for the working parent and the arm
IDs to score. Relate each arm's GAIA2 failures to the default-agent prompts,
tool descriptions, tool implementations, loop logic, configuration, or hooks
that a patch would touch. A pattern whose root cause lies in the simulation
engine, the judge, or the dataset is outside the writable harness and has low
fixability.

Return one score per arm using the supplied Meta-ARE arm scoring schema.
