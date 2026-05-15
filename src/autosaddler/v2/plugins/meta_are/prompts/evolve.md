# Meta-ARE Git Composition Contract (Plugin-specific)

Read `.autosaddler/session_context.json` for the candidate IDs and current
training case IDs. The first `parent_id` is materialized as the working Git
base. Each `component_sources` entry replaces one schema-listed repository file
with that file from the named non-base parent.

Whole-file replacement is the only supported Meta-ARE composition unit. List
every referenced source in `parent_ids`, use only file/source combinations
offered by the output schema, and leave `component_sources` empty when a
correct composition would require line-level merging or coupled files that are
not jointly selectable.
