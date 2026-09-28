#!/usr/bin/env bash
# ============================================================================
# AutoSaddler — Train Script
#
# Runs AutoSaddler optimization on Meta-ARE default agent.
# Uses Claude Agent SDK for agent sessions and OpenAI API for judge.
#
# Required environment variables:
#   META_ARE_REPO    — path to Meta-ARE repository
#   OPENAI_API_KEY   — OpenAI API key (for judge model)
#   ANTHROPIC_API_KEY — Anthropic API key (for Claude Agent SDK, if not set in config)
#
# Usage:
#   bash scripts/legacy/train.sh --config configs/v1/meta_are.yaml
#   bash scripts/legacy/train.sh --config configs/v1/meta_are_smoke.yaml --dry-run
# ============================================================================

# NOTE: Do NOT use 'set -e' here. The main training loop (python optimize)
# may encounter transient errors (API timeouts, agent session failures, etc.)
# that cause a non-zero exit code. With set -e, the entire script terminates
# immediately, killing the tmux session without any error message.

AUTOSADDLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTOSADDLER_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# ─── Parse arguments ─────────────────────────────────────────────────
CONFIG=""
EXTRA_ARGS=()
NEXT_IS_CONFIG=0
for arg in "$@"; do
    if [[ "$arg" == "--config" || "$arg" == "-c" ]]; then
        NEXT_IS_CONFIG=1
        continue
    fi
    if [[ "${NEXT_IS_CONFIG}" == "1" ]]; then
        CONFIG="$arg"
        NEXT_IS_CONFIG=0
        continue
    fi
    EXTRA_ARGS+=("$arg")
done

if [[ -z "$CONFIG" ]]; then
    echo "Usage: bash scripts/legacy/train.sh --config <config.yaml> [--dry-run]"
    exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ─── Activate venv ────────────────────────────────────────────────────
cd "$AUTOSADDLER_DIR"
source "${AUTOSADDLER_DIR}/.venv/bin/activate"
export PYTHONPATH="${AUTOSADDLER_DIR}/src:${PYTHONPATH:-}"

# ─── Verify environment variables ────────────────────────────────────
: "${META_ARE_REPO:?ERROR: META_ARE_REPO not set. Export it to point to your Meta-ARE repo.}"
: "${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY not set. Export it for judge model access.}"
export META_ARE_BASE_BRANCH="${META_ARE_BASE_BRANCH:-main}"

# ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL must be set for Claude Agent SDK
if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${ANTHROPIC_BASE_URL:-}" ]]; then
    echo "WARNING: Neither ANTHROPIC_API_KEY nor ANTHROPIC_BASE_URL is set."
    echo "         Claude Agent SDK sessions will fail unless configured in the YAML."
fi

# ─── Pre-flight checks ───────────────────────────────────────────────
python -c "import claude_agent_sdk; print(f'claude-agent-sdk {claude_agent_sdk.__version__}')" 2>/dev/null || {
    echo "ERROR: claude_agent_sdk not installed. Run: uv pip install -e '.'"
    exit 1
}

python -c "import git" 2>/dev/null || {
    echo "ERROR: gitpython not installed"
    exit 1
}

python -c "from autosaddler.v1.proposer.autosaddler import AutoSaddlerProposer" 2>/dev/null || {
    echo "ERROR: AutoSaddler proposer not importable. Check PYTHONPATH."
    exit 1
}

# ─── Run ──────────────────────────────────────────────────────────────
exec > >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line"; done | tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "AutoSaddler Train"
echo "  Dir:          $AUTOSADDLER_DIR"
echo "  Config:       $CONFIG"
echo "  META_ARE_REPO:$META_ARE_REPO"
echo "  Log:          $LOG_FILE"
echo "  Args:         ${EXTRA_ARGS[*]:-}"
echo "  Started:      $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================"

python -u -m autosaddler.v1.adapters.meta_are_adapter.optimize \
  --config "$CONFIG" \
  --mutation-strategy autosaddler \
  "${EXTRA_ARGS[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Train Done at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
else
    echo "Train FAILED (exit code $EXIT_CODE) at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
fi
echo "Log: $LOG_FILE"
echo "============================================"
exit $EXIT_CODE
