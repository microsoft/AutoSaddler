# Meta-ARE Pattern Extraction Contract (Plugin-specific)

Read `.autosaddler/session_context.json` for the pre-patch and post-patch
GAIA2 failures, matched training scores, diagnosis, and reflection lessons.
`.autosaddler/training_evidence_before.json` holds the working parent's
training evidence and `.autosaddler/training_evidence_after.json` holds the
patched candidate's evidence; each case record includes per-repetition judge
rationale and agent interactions.

Distinguish GAIA2 judge outcomes when abstracting a symptom: a tool-call count
mismatch, a missing required action, a wrong hard-checked argument (IDs,
datetimes, phone numbers, paths, exact strings), a semantic soft-check
failure, a timing or dependency violation, and a crash or invalid output are
different mechanisms. Name the agent behavior that produced the judge outcome,
such as misreading a tool description, skipping a required confirmation, or
formatting an argument incorrectly, rather than the judge message itself.

Return the registry update using the supplied Meta-ARE pattern extraction
schema. `case_id` values must come from the listed failures.
