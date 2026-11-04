## Meta-ARE Pareto Reflection Evidence (Plugin-specific)

Read the owning iteration's recorded `verdict` for before/after score and cost,
Bayesian bootstrap joint support, and screening parameters when available.
Use the supplied `development_aggregate`, `development_cost`, and
`development_evaluation_id` fields for archive-referenced aggregate provenance.
Read `promoted_candidate_id`, `archive_candidate_ids`, and `current_candidate_id`
from that iteration when present. Return only the existing Meta-ARE lesson schema;
`evidence_case_ids` remains limited to supplied training cases.