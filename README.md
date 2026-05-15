# 🛠️ AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12-3.14](https://img.shields.io/badge/Python-3.12--3.14-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![arXiv: 2608.23041](https://img.shields.io/badge/arXiv-2608.23041-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.23041)

**AutoSaddler automatically improves LLM-agent harnesses by diagnosing execution traces, applying structured updates to prompts, tools, and middleware, and selecting changes that generalize.**

📄 **[Paper](https://arxiv.org/abs/2608.23041)** · 🌐 **[Project website](https://autosaddler-projectpage.github.io/)** · 🎥 **[Short video](https://autosaddler-projectpage.github.io/#video)**

<p align="center">
  <img src="figures/AutoSaddler_Overall_Framework.jpg" width="100%" alt="AutoSaddler Overall Framework"/>
</p>

Preliminary results report the following test Pass@1 scores across benchmarks and agent harnesses:

| Benchmark | Base agent harness | Base Pass@1 | AutoSaddler Pass@1 | Improvement |
|---|---|---:|---:|---:|
| GAIA2 | Default ReAct agent | 53.0 | **62.0** | **+9.0 pp** |
| SWE-Bench Pro | SWE-agent | 37.3 | **46.9** | **+9.6 pp** |
| Terminal-Bench 2.0 | Terminus 2 | 40.0 | **50.0** | **+10.0 pp** |

See the [paper](https://arxiv.org/abs/2608.23041) and
[interactive project website](https://autosaddler-projectpage.github.io/) for per-model results,
ablations, compute-efficiency plots, and optimization trajectories.

## ✨ Highlights

- **Full-harness optimization:** searches over prompts, tool definitions and implementations, middleware hooks, and agent-loop logic.
- **In-depth diagnosis:** deeply debugs execution traces and the harness codebase to identify root causes rather than relying on shallow reflection.
- **Structured intervention:** targets prompts, tools, and middleware through an explicit patch taxonomy and phased Capability-to-Steering schedule instead of unconstrained editing.
- **Generalization-aware selection:** validates updates beyond the motivating trajectories and uses reflection with an evolution DAG (EvoDAG) to retain broadly useful lessons.
- **Durable execution:** records append-only events, immutable provenance, resumable state, and content-addressed candidates.

## 📣 News

- **2026-08-25:** Added V2 support for optimizing the [Meta-ARE](https://github.com/pshlego/Meta-ARE) harness on GAIA2.
- **2026-08-24:** The AutoSaddler paper was released as [arXiv v1](https://arxiv.org/abs/2608.23041), together with the [project website and short video](https://autosaddler-projectpage.github.io/).

## 🛠️ Installation

AutoSaddler requires Python 3.12-3.14, [uv](https://docs.astral.sh/uv/), and Git.

```bash
git clone https://github.com/microsoft/AutoSaddler.git
cd AutoSaddler
uv sync --extra dev
```

Run Python commands in this repository through `uv run`.

## 🚀 Quick Start

Run the deterministic, credential-free V2 template to exercise the optimization engine, event store,
candidate evolution, and output projections:

```bash
uv run python -m autosaddler.v2.cli \
  --config configs/v2/local_template.yaml \
  --run-id local-template
```

Repeating the command resumes the same run after validating its resolved inputs.

## 🧭 Versions

- **V2 (current):** the durable, plugin-based implementation documented in this README. New users and integrations should start here.
- **V1 (legacy):** the research-quality implementation used for the experiments in the arXiv paper, retained for paper reproduction and reference. See the [V1 README](src/autosaddler/v1/README.md).

## 🗂️ Repository Structure

```text
AutoSaddler/
├── configs/                 # V1/V2 configs and benchmark split manifests
├── docs/                    # Architecture and scenario-integration guides
├── figures/                 # README and paper figures
├── scripts/                 # Data provisioning and legacy launch scripts
├── src/autosaddler/v1/      # Legacy implementation
├── src/autosaddler/v2/      # Current engine, plugins, providers, and storage
└── tests/                   # Characterization and focused V2 tests
```

Start with the [V2 architecture guide](docs/v2-architecture.md) for the current implementation.

## 🧩 How It Works

AutoSaddler formulates harness optimization as offline mini-batch learning. It uses three session
types across the optimization lifecycle:

1. **Diagnosis-Patch:** inspects failed traces and the harness codebase, then proposes structured
   Capability patches (code or infrastructure) and Steering patches (textual behavior changes).
2. **Reflection:** compares pre- and post-patch traces, classifies fixed, regressed, still-failing,
   and still-passing cases, and records reusable lessons.
3. **Evolution:** consults the full EvoDAG to synthesize candidates from successful components and
   lessons across lineages.

Candidate updates are verified on sampled training cases and gated on the development split. When
its rollout budget is exhausted, AutoSaddler returns the highest-ranked development candidate. See
the [V2 architecture guide](docs/v2-architecture.md) for the event lifecycle and invariants.

## 🎯 Supported Harnesses and Benchmarks

The current repository includes:

| Harness | Harness space | Benchmark | Purpose |
|---|---|---|---|
| Deterministic fake harness (`fake`) | Structured component map | Synthetic cases | Local development and tests |
| Meta-ARE Default ReAct Agent (`meta_are`) | Git repository | GAIA2 | End-to-end smoke experiments |

V2 supports immutable, content-addressed component-map and Git candidate spaces. Optimizer sessions
can use the built-in fake provider, Anthropic Claude Agent SDK, or GitHub Copilot SDK transport.

Integrations for additional harnesses (e.g., **OpenClaw** and **Codex**) and benchmarks (e.g., **Terminal-Bench**) are coming. Stay tuned!

## ⚙️ Configuration

Every V2 config starts with `schema_version: autosaddler/v2`.
A **scenario plugin** is the adapter between AutoSaddler's generic optimization engine and a
specific harness/benchmark pair. It supplies the harness space, cases, evaluator, evidence,
prompts, capabilities, and reproducibility metadata. A config selects that plugin and declares four
explicit ownership areas:

| Section | Responsibility |
|---|---|
| `scenario` | Plugin type, immutable sources, datasets, evaluator, and mutable harness surface |
| `optimization` | Task selection, acceptance, development gate, ranking, budget, retries, and timeouts |
| `provider` | Optimizer provider, capabilities, model, endpoint, and provider-specific settings |
| `storage` | Durable run root |

Included configurations:

| Path | Purpose |
|---|---|
| `configs/v2/local_template.yaml` | Credential-free deterministic V2 template |
| `configs/v2/meta_are_smoke.yaml` | Current Meta-ARE/GAIA2 smoke integration |
| `configs/v1/meta_are.yaml` | Legacy full Meta-ARE/GAIA2 run |
| `configs/v1/meta_are_smoke.yaml` | Legacy bounded smoke run |
| `configs/datasets/GAIA2/` | Shared train, development, and test split manifests |

Configuration is strict and fail-closed. A run ID can be reused only when all resolved inputs are
byte-identical; changed source revisions, manifests, settings, or provenance are rejected.

## 🔬 Reproducing the Included GAIA2 Smoke Run

The checked-in V2 smoke config exercises the real optimization pipeline on seven GAIA2 scenarios:
six training cases and one development case, for two optimization iterations. It is a bounded
integration run rather than the full paper experiment, may take several hours, and incurs provider
charges.

### 1. Prepare the repositories

Use this sibling layout:

```text
<parent>/
|-- AutoSaddler/
|-- Meta-ARE/
|-- meta_are_data/
`-- working_dir/
```

Clone the adapted Meta-ARE repository at the revision pinned by the config:

```bash
cd ..
git clone https://github.com/pshlego/Meta-ARE.git Meta-ARE
git -C Meta-ARE checkout --detach 2419824a94fb8211fc8227ada7bff1b29f86e563
mkdir -p working_dir
cd AutoSaddler
uv sync --extra meta-are-setup
```

### 2. Provision the benchmark inputs

Provision the seven manifest-selected GAIA2 payloads from the pinned Hugging Face revision.
`HF_TOKEN` is optional for this public dataset but avoids anonymous rate limits:

```bash
uv run --extra meta-are-setup python scripts/meta_are/provision_gaia2_scenarios.py \
  --destination-root "$PWD/../Meta-ARE/datasets_local/gaia2" \
  --revision 78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99
```

The command must report `"file_count": 7`. Then provision the approximately 260 MB demo
filesystem:

```bash
uv run --extra meta-are-setup python scripts/meta_are/provision_demo_filesystem.py \
  --destination-root "$PWD/../meta_are_data/gaia2_filesystem" \
  --revision 132e26376f5e963bb59f64bcccdd02188cb08dee \
  --meta-are-project ../Meta-ARE
```

Both commands are idempotent, record source revisions and content digests, and reject mismatched
local files. Evaluation runs with Hugging Face clients forced offline.

### 3. Configure providers and run

The smoke config uses OpenAI `gpt-4.1-mini` for the task agent and judge, and Anthropic
`claude-opus-4-6` for optimization:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Run from the external working directory so generated workspaces cannot inherit repository-level
agent instructions through Git ancestry. Use a new run ID for every independent run:

```bash
cd ../working_dir
RUN_ID="meta-are-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
printf 'run_id=%s\n' "$RUN_ID"
uv run --project ../AutoSaddler \
  python -m autosaddler.v2.cli \
  --config ../AutoSaddler/configs/v2/meta_are_smoke.yaml \
  --run-id "$RUN_ID"
```

Runs are written under `working_dir/outputs/v2_meta_are/runs/<run-id>/`. Success writes
`result.json` with `"iterations": 2` and prints the selected candidate and development score.

### Run artifacts and resumption

A run is self-contained and can include:

```text
<run-root>/<run-id>/
├── events.jsonl
├── manifest.json
├── snapshot.json
├── evolution_dag.json
├── metrics.jsonl
├── metrics-summary.json
├── result.json
├── resolved/
├── candidates/
├── evaluations/
├── sessions/
├── mutation-deltas/  # Git harnesses only
└── workspaces/
```

`events.jsonl` is authoritative. To resume after an interruption, confirm no process is using the
run ID and repeat the same command with the same inputs. Never run two processes against one run ID.
Review `sessions/` and `evaluations/` before sharing a run because traces may contain prompts,
responses, tool arguments, working directories, repository metadata, or other sensitive data.

To branch a validated nonterminal checkpoint into a new run:

```bash
uv run python -m autosaddler.v2.cli \
  --config CONFIG.yaml \
  --run-id NEW_RUN_ID \
  --fork-from-run-id SOURCE_RUN_ID \
  --fork-through-sequence LAST_EVENT_SEQUENCE
```

Only `optimization.budget.max_iterations` may differ when initializing a fork. Legacy checkpoints
cannot be imported.

## 🔌 Adding a Harness or Benchmark

A V2 scenario plugin owns the integration boundary: harness space, evaluator, evidence builder,
prompt pack, disjoint train and development cases, provider capabilities, and resolved provenance.

Built-in integrations live under `src/autosaddler/v2/plugins/`. An external package can register a
plugin through the `autosaddler.scenarios` entry-point group without adding scenario-specific code
to this repository. AutoSaddler rejects duplicate names, API-version mismatches, malformed
descriptors, and plugin load failures.

Follow the [scenario integration guide](docs/scenario-integration.md) for the ownership checklist,
package layout, registration contract, tests, and smoke-config requirements. Use
`src/autosaddler/v2/plugins/fake.py` as the smallest deterministic example and
`src/autosaddler/v2/plugins/meta_are/` as a production Git-harness example.
You are encouraged to use a coding agent to conduct the scenario integration by following the
integration guide.

## 🧰 Development and Project Policies

```bash
uv sync --extra dev
uv run ruff check src/autosaddler tests/
uv run python -m pytest tests/ -v --tb=short
uv build
```

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). This project follows the
[Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md), publishes its
[security reporting policy](SECURITY.md), and is available under the [MIT License](LICENSE).

## 📝 Citation

```bibtex
@misc{park2026autosaddlerautomaticharnessoptimization,
  title={AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces},
  author={Sungho Park and Wonjoong Kim and Rongyuan Tan and Jue Zhang and Wook-Shin Han and Pengfei Gao and Chanyoung Park and Yongqiang Yao and Rao Fu and Elsie Nallipogu and Qingwei Lin and Saravan Rajmohan and Dongmei Zhang},
  year={2026},
  eprint={2608.23041},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2608.23041},
}
```
