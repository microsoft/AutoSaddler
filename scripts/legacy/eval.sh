#!/usr/bin/env bash
# ============================================================================
# AutoSaddler — Eval Script
#
# Evaluates the best worktree from a completed AutoSaddler training run.
# Automatically finds the best candidate (highest dev score, latest iteration)
# from evolution_dag.json and runs are-benchmark on specified test scenarios.
#
# Required environment variables:
#   META_ARE_REPO    — path to Meta-ARE repository
#   OPENAI_API_KEY   — OpenAI API key (for judge model)
#
# Usage:
#   bash scripts/legacy/eval.sh \
#       --train-output /path/to/autosaddler/20260511-083111 \
#       --dataset /path/to/gaia2_test_scenarios \
#       --model claude-haiku-4.5
#
# Optional:
#   --provider copilot             (default: copilot)
#   --endpoint http://host:port/v1 (default: none)
#   --judge-model gpt-4.1-mini     (default: gpt-4.1-mini)
#   --judge-provider openai        (default: openai)
#   --num-trials 3                 (default: 1)
#   --max-concurrent 20            (default: 3)
#   --scenario-timeout 7200        (default: 7200)
#   --output-dir /path/to/output   (default: <train-output>/eval_<timestamp>)
#   --worktree /path/to/worktree   (override auto-selection)
#   --split validation             (default: validation)
#   --bench-config mini             (default: . = all capabilities)
# ============================================================================
set -euo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────
TRAIN_OUTPUT=""
DATASET=""
MODEL=""
PROVIDER="copilot"
ENDPOINT=""
JUDGE_MODEL="gpt-4.1-mini"
JUDGE_PROVIDER="openai"
NUM_TRIALS=1
MAX_CONCURRENT=3
SCENARIO_TIMEOUT=7200
OUTPUT_DIR=""
WORKTREE_OVERRIDE=""
SPLIT="validation"
BENCH_CONFIG="."
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# ─── Parse arguments ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-output)    TRAIN_OUTPUT="$2"; shift 2;;
        --dataset)         DATASET="$2"; shift 2;;
        --model)           MODEL="$2"; shift 2;;
        --provider)        PROVIDER="$2"; shift 2;;
        --endpoint)        ENDPOINT="$2"; shift 2;;
        --judge-model)     JUDGE_MODEL="$2"; shift 2;;
        --judge-provider)  JUDGE_PROVIDER="$2"; shift 2;;
        --num-trials)      NUM_TRIALS="$2"; shift 2;;
        --max-concurrent)  MAX_CONCURRENT="$2"; shift 2;;
        --scenario-timeout) SCENARIO_TIMEOUT="$2"; shift 2;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2;;
        --worktree)        WORKTREE_OVERRIDE="$2"; shift 2;;
        --split)           SPLIT="$2"; shift 2;;
        --bench-config)    BENCH_CONFIG="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [[ -z "$TRAIN_OUTPUT" ]]; then
    echo "Usage: bash scripts/legacy/eval.sh --train-output <path> --dataset <path> --model <model> [options]"
    exit 1
fi

if [[ -z "$DATASET" ]]; then
    echo "ERROR: --dataset is required (path to test scenario directory)"
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required (e.g. claude-haiku-4.5, claude-opus-4.6)"
    exit 1
fi

# ─── Verify environment variables ─────────────────────────────────────
: "${META_ARE_REPO:?ERROR: META_ARE_REPO not set. Export it to point to your Meta-ARE repo.}"
: "${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY not set. Export it for judge model access.}"

# ─── Find best worktree from evolution_dag.json ───────────────────────
if [[ -n "$WORKTREE_OVERRIDE" ]]; then
    BEST_WORKTREE="$WORKTREE_OVERRIDE"
    echo "Using manually specified worktree: $BEST_WORKTREE"
else
    DAG_JSON="${TRAIN_OUTPUT}/evolution_dag.json"
    if [[ ! -f "$DAG_JSON" ]]; then
        echo "ERROR: evolution_dag.json not found at $DAG_JSON"
        exit 1
    fi

    BEST_WORKTREE=$(python3 -c "
import json, sys

dag = json.load(open('$DAG_JSON'))
nodes = dag['nodes']

# Find nodes with val scores
scored = []
for nid, n in nodes.items():
    val = n.get('score_val')
    if val is not None and n.get('worktree_path'):
        scored.append((int(nid), val, n['iteration'], n['worktree_path']))

if not scored:
    print('ERROR: No candidates with val scores found', file=sys.stderr)
    sys.exit(1)

# Sort by val score (desc), then iteration (desc) to get best + latest
scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
best = scored[0]
print(best[3])
print(f'Selected C{best[0]} (iter={best[2]}, val={best[1]:.4f})', file=sys.stderr)
")

    if [[ -z "$BEST_WORKTREE" ]]; then
        echo "ERROR: Could not determine best worktree"
        exit 1
    fi
    echo "Auto-selected best worktree: $BEST_WORKTREE"
fi

# ─── Verify worktree ─────────────────────────────────────────────────
if [[ ! -d "$BEST_WORKTREE/are" ]]; then
    echo "ERROR: Worktree not found or invalid at $BEST_WORKTREE"
    exit 1
fi

# ─── Check for hook.json ─────────────────────────────────────────────
HOOK_FLAG=""
if [[ -f "$BEST_WORKTREE/hook.json" ]]; then
    echo "Hook config found: $BEST_WORKTREE/hook.json"
    HOOK_FLAG="--hook-config $BEST_WORKTREE/hook.json"
else
    echo "No hook.json found (proceeding without hooks)"
fi

# ─── Resolve output directory ─────────────────────────────────────────
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${TRAIN_OUTPUT}/eval_${TIMESTAMP}"
fi

# ─── Activate venv from worktree ──────────────────────────────────────
cd "$BEST_WORKTREE"
source "${META_ARE_REPO}/.venv/bin/activate"
export PYTHONPATH="${BEST_WORKTREE}:${PYTHONPATH:-}"

# ─── Build endpoint flag ──────────────────────────────────────────────
ENDPOINT_FLAG=""
if [[ -n "$ENDPOINT" ]]; then
    ENDPOINT_FLAG="--endpoint $ENDPOINT"
fi

# ─── Run evaluation ──────────────────────────────────────────────────
echo "============================================"
echo "AutoSaddler Evaluation"
echo "  Train output:    $TRAIN_OUTPUT"
echo "  Worktree:        $BEST_WORKTREE"
echo "  Dataset:         $DATASET"
echo "  Model:           $MODEL ($PROVIDER)"
echo "  Endpoint:        ${ENDPOINT:-default}"
echo "  Judge:           $JUDGE_MODEL ($JUDGE_PROVIDER)"
echo "  Max concurrent:  $MAX_CONCURRENT"
echo "  Scenario timeout:${SCENARIO_TIMEOUT}s"
echo "  Num trials:      $NUM_TRIALS"
echo "  Split:           $SPLIT"
echo "  Bench config:    $BENCH_CONFIG"
echo "  Output:          $OUTPUT_DIR"
echo "  Started:         $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================"
echo ""

for trial in $(seq 1 "$NUM_TRIALS"); do
    TRIAL_OUTPUT="${OUTPUT_DIR}/trial_${trial}"
    TRIAL_LOG="${OUTPUT_DIR}/trial_${trial}.log"
    mkdir -p "$TRIAL_OUTPUT"

    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> Trial ${trial} / ${NUM_TRIALS}"
    echo ">>> Output: ${TRIAL_OUTPUT}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    are-benchmark run \
        --agent default \
        --dataset "$DATASET" \
        --config "$BENCH_CONFIG" \
        --split "$SPLIT" \
        --model "$MODEL" \
        --provider "$PROVIDER" \
        $ENDPOINT_FLAG \
        --judge_model "$JUDGE_MODEL" \
        --judge_provider "$JUDGE_PROVIDER" \
        --output_dir "$TRIAL_OUTPUT" \
        --scenario_timeout "$SCENARIO_TIMEOUT" \
        --num_runs 1 \
        --max_concurrent_scenarios "$MAX_CONCURRENT" \
        --trace_dump_format both \
        $HOOK_FLAG \
        2>&1 | tee "$TRIAL_LOG"

    echo ""
    echo "Trial ${trial} complete."
    echo ""
done

echo "============================================"
echo "Evaluation Done at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Results: $OUTPUT_DIR"
echo "============================================"
