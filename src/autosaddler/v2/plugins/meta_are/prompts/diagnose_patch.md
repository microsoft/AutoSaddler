# Meta-ARE Mutation Task (Plugin-specific)

Read `.autosaddler/session_context.json` for `patch_phase` and mutation scope,
`.autosaddler/training_evidence.json` for GAIA2 case evidence, and the core
history manifest. Inspect the default-agent prompt, tool docstrings and
signatures, implementations, loop, configuration, or hooks implicated by the
core diagnosis procedure.

Edit only allowlisted files in the current workspace and make one coherent
change legal for the declared phase. In capability phase, implementation and
interface changes must keep descriptions and callers synchronized. In
steering phase, follow the exact mutation boundary in the `steering-patch`
skill.

Run the `patch-verification` procedure after mutation. Return `intent`, causal
`diagnosis`, `expected_effect`, and exact repository-relative `changed_paths`.
All fields must agree with the final diff and use the supplied Meta-ARE output
schema.
