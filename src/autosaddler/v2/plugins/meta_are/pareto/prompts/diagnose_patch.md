## Meta-ARE Pareto Evidence (Plugin-specific)

Use each repetition's `score` and `cost.currency_amount` as measured objectives.
Compare `requested_repetitions` with each case's `observed_repetitions` and actual
repetition IDs.
Inspect task-agent `model_usage` and its `usage_incomplete` flags. Missing calls,
missing completeness information, or unknown USD are not evidence of savings.
Cached and uncached input are already priced in the recorded cost; do not add
judge, optimizer, or retry accounting to the objective or reprice tokens yourself.

Use the existing `diagnosis` and `expected_effect` fields to state the cost
mechanism, expected accuracy/cost effects, accuracy risks, and uncertainty.