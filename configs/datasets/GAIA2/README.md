# GAIA2 Split Manifests

These files contain case IDs only. Scenario payloads remain in the read-only local GAIA2 tree selected
by `scenario.settings.dataset_root`. AutoSaddler resolves each train and validation ID to exactly one
scenario file and hashes its bytes before evaluation.

| File | Cases | Purpose |
|---|---:|---|
| `train.json` | 75 | Canonical full-run training split from Universe 29 across adaptability, ambiguity, execution, search, and time |
| `val.json` | 65 | Canonical full-run development/ranking split from Universe 30 across all five capabilities |
| `test_universe_21.json` | 107 | Available holdout alternative; not selected by the Meta-ARE configs |
| `test_universe_22.json` | 112 | Selected full-run holdout; its digest and count are recorded, but optimization does not open its scenario payloads |
| `test_universe_27.json` | 81 | Available holdout alternative; not selected by the Meta-ARE configs |
| `train_smoke.json` | 6 | Bounded smoke subset drawn without replacement from `train.json`, covering all five capabilities |
| `val_smoke.json` | 1 | Bounded smoke ranking case drawn from `val.json` |

The smoke training subset preserves the historical six-case selection; the development subset uses
the first canonical adaptability case to keep two optimization iterations affordable. Every smoke ID
is a member of its canonical full-run split. The smoke
configuration points at the full `test_universe_22.json` only to freeze the intended holdout; the V2
optimization engine exposes train evidence and development aggregates, never holdout IDs or payloads.

V1 full and smoke configs use the corresponding train and validation manifests. The V2 smoke uses
`train_smoke.json`, `val_smoke.json`, and the opaque Universe 22 holdout descriptor. A V2 full-run
config can select `train.json`, `val.json`, and `test_universe_22.json` with an appropriate budget.

Provision the V2 smoke payloads from the immutable source revision documented in the repository
README:

```bash
uv run --extra meta-are-setup python scripts/meta_are/provision_gaia2_scenarios.py \
	--destination-root "$PWD/../Meta-ARE/datasets_local/gaia2" \
	--revision 78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99
```

The utility defaults to `train_smoke.json` and `val_smoke.json`. Pass `--manifest` repeatedly to
select other manifests. It writes a source descriptor beside the payload tree and rejects missing,
duplicate, or locally modified scenarios.