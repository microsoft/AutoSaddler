# Scripts

- `meta_are/provision_demo_filesystem.py`: provisions the immutable, provenance-checked demo
  filesystem required by the V2 Meta-ARE plugin.
- `meta_are/provision_gaia2_scenarios.py`: downloads only the GAIA2 scenarios selected by one or
  more split manifests from an immutable dataset revision and verifies idempotent reuse.
- `legacy/train.sh`: runs a V1 Meta-ARE optimization config.
- `legacy/eval.sh`: evaluates the best worktree from a completed V1 run.

Run Python utilities through `uv`, for example:

```bash
uv run --extra meta-are-setup python scripts/meta_are/provision_demo_filesystem.py --help
```
