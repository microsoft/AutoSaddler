# AutoSaddler V1 (Legacy)

V1 remains available for reproducing legacy Meta-ARE workflows. Its implementation lives in this
directory and its Python namespace is `autosaddler.v1`. New integrations should use the current
implementation documented in the [project README](../../../README.md).

Unless stated otherwise, run the commands below from the repository root.

## Source Layout

```text
src/autosaddler/v1/
├── api.py
├── sdk_session.py
├── adapters/
├── core/
├── logging/
├── proposer/
├── strategies/
└── utils/
```

Use explicit V1 imports:

```python
from autosaddler.v1 import optimize
from autosaddler.v1.core.adapter import EvaluationBatch, GEPAAdapter
from autosaddler.v1.proposer.autosaddler import AutoSaddlerProposer
```

## Prerequisites

- Python 3.12-3.14
- `uv` and Git
- A compatible Meta-ARE checkout with its dependencies installed
- Provider credentials required by `configs/v1/meta_are.yaml` or `configs/v1/meta_are_smoke.yaml`

## Meta-ARE Setup

Set up Meta-ARE in a separate checkout:

```bash
git clone https://github.com/facebookresearch/meta-agents-research-environments.git
cd meta-agents-research-environments
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Set up this repository separately:

```bash
git clone <this-repo-url>
cd AutoSaddler
uv sync --extra dev
```

## Environment

| Variable | Required | Description |
|---|---|---|
| `META_ARE_REPO` | Yes | Absolute path to the Meta-ARE checkout |
| `META_ARE_BASE_BRANCH` | No | Base harness branch; defaults to `main` in the legacy launcher |
| `OPENAI_API_KEY` | Yes | Agent or judge credential for the default legacy profile |
| `ANTHROPIC_API_KEY` | Backend-dependent | Claude Agent SDK credential |
| `ANTHROPIC_BASE_URL` | No | Claude-compatible endpoint override |
| `AGENT_MODEL` | No | Task-agent model override |
| `AGENT_MODEL_PROVIDER` | No | Task-agent provider override |
| `AGENT_MODEL_ENDPOINT` | No | Task-agent endpoint override |
| `JUDGE_MODEL` | No | Judge model override |
| `JUDGE_MODEL_PROVIDER` | No | Judge provider override |

```bash
export META_ARE_REPO=/path/to/meta-agents-research-environments
export OPENAI_API_KEY=...
```

## Training

The legacy launchers and configs remain under explicitly named paths:

```bash
bash scripts/legacy/train.sh --config configs/v1/meta_are_smoke.yaml --dry-run
bash scripts/legacy/train.sh --config configs/v1/meta_are_smoke.yaml
bash scripts/legacy/train.sh --config configs/v1/meta_are.yaml
```

The equivalent module command is:

```bash
uv run python -m autosaddler.v1.adapters.meta_are_adapter.optimize \
    --config configs/v1/meta_are_smoke.yaml \
    --dry-run
```

V1 configuration sections are `dataset`, `adapter`, `optimization`, `autosaddler`, and `sdk`.
Environment variables support `${VAR}` and `${VAR:-default}` expansion.

Legacy training outputs are written below `${META_ARE_REPO}/autosaddler/<timestamp>/` and include
`evolution_dag.json`, candidate summaries, logs, `state.bin`, worktrees, cycle outputs, and generated
development outputs. These artifacts are not compatible with the current event store.

## Evaluation

```bash
bash scripts/legacy/eval.sh \
    --train-output /path/to/autosaddler/<timestamp> \
    --dataset /path/to/test_scenarios \
    --model gpt-4.1-mini
```

Use `--worktree` to override automatic best-worktree selection and `--bench-config` to select a
capability subdirectory. The dataset is expected to follow Meta-ARE's
`<config>/<split>/*.json` layout.

V1 outputs should remain read-only when used as historical evidence. They cannot be resumed, forked,
or imported by the current implementation.

## Acknowledgments

The legacy V1 implementation was adapted from [GEPA](https://github.com/gepa-ai/gepa) by Lakshya A Agrawal.