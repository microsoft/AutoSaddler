# Terminal-Bench 2.0 Split Manifests

These files contain task IDs only. Task payloads come from the Harbor `terminal-bench@2.0` dataset.
Each manifest groups its task IDs by the `[metadata].category` value in the task's `task.toml`;
category keys and task IDs are sorted.

| File | Tasks | Purpose |
|---|---:|---|
| `train.json` | 30 | Training split used for harness optimization, across 12 categories |
| `val.json` | 19 | Development/ranking split (`dev` in the original split file), across 9 categories |
| `test.json` | 40 | Held-out test split, across 14 categories; optimization never evaluates these tasks |

The three splits are disjoint and together cover all 89 Terminal-Bench 2.0 tasks. They were drawn
with seed 42 and stratified by the `[metadata].difficulty` value in `task.toml`:

| Split | Easy | Medium | Hard |
|---|---:|---:|---:|
| train | 1 | 19 | 10 |
| val | 1 | 12 | 6 |
| test | 2 | 24 | 14 |

The split was originally published as a single flat file with `train`, `dev`, and `test` task-ID
lists (`tbench2_split_seed42_stratified.json`, SHA-256
`c103a868b7845db27999c75d6f6cd51cfce0fbcfbec2efa736fd56f7fb5cf7d0`). These manifests hold the same
task IDs, regrouped by category to match the GAIA2 manifest format.
