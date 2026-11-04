# Meta-ARE Pareto Preparation Contract (Plugin-specific)

Read `.autosaddler/session_context.json`. `fixed_current_candidate_id` is the
only permitted base; `donor_candidate_ids` lists eligible archive references.
Edit the supplied current source tree locally within `mutation_scope` and the
configured `patch_phase`.

Return `schema_version: autosaddler-meta-are-evolution/v1`,
`parent_ids: [fixed_current_candidate_id]`, `component_sources: {}`,
`changed_paths`, `donor_candidate_ids`, and `rationale`. Changed paths must match
actual workspace edits; donors must be eligible and actually used as evidence.
For no-op, return empty changed paths and donor IDs.
