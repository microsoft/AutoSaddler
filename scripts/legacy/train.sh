#!/usr/bin/env bash
# ============================================================================
# AutoSaddler — Train Script
#
# Runs legacy V1 AutoSaddler or ActiveSaddler optimization on the Meta-ARE default agent.
# Supports the Claude Agent SDK and the GitHub Copilot SDK.
#
# Required environment variables:
#   META_ARE_REPO    — path to Meta-ARE repository
# Optional environment variables:
#   META_ARE_BASE_BRANCH — base harness branch in META_ARE_REPO (default: main)
#   Azure CLI profile directories referenced by the selected config
#
# Usage:
#   bash scripts/legacy/train.sh --config configs/v1/meta_are.yaml
#   bash scripts/legacy/train.sh --config configs/v1/meta_are_activesaddler.yaml
#   bash scripts/legacy/train.sh --config configs/v1/meta_are_activesaddler_smoke.yaml --dry-run
# ============================================================================

# NOTE: Do NOT use 'set -e' here. The main training loop (python optimize)
# may encounter transient errors (API timeouts, agent session failures, etc.)
# that cause a non-zero exit code. With set -e, the entire script terminates
# immediately, killing the tmux session without any error message.

AUTOSADDLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${AUTOSADDLER_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

print_usage() {
    echo "Usage: bash scripts/legacy/train.sh --config <config.yaml> [--dry-run]"
}

for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        print_usage
        exit 0
    fi
done

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
    print_usage
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

export META_ARE_BASE_BRANCH="${META_ARE_BASE_BRANCH:-main}"

CONFIG_FACTS="$(python - "$CONFIG" <<'PY'
import sys
from autosaddler.v1.adapters.meta_are_adapter.optimize import load_config

config = load_config(sys.argv[1])
sdk = config.get("sdk", {})
claude = sdk.get("claude") if isinstance(sdk.get("claude"), dict) else sdk
adapter = config.get("adapter", {})
print("|".join([
    str(sdk.get("backend", "claude")),
    str(claude.get("auth_mode", "api_key")),
    str(claude.get("azure_config_dir") or ""),
    str(adapter.get("model_azure_config_dir") or ""),
    str(adapter.get("judge_azure_config_dir") or ""),
    str(adapter.get("model_provider") or ""),
    str(adapter.get("judge_provider") or ""),
]))
PY
)"
if [[ $? -ne 0 || -z "$CONFIG_FACTS" ]]; then
    echo "ERROR: failed to load config: $CONFIG" >&2
    exit 1
fi
IFS='|' read -r CONFIG_SDK_BACKEND CLAUDE_AUTH_MODE SDK_AZURE_DIR MODEL_AZURE_DIR JUDGE_AZURE_DIR MODEL_PROVIDER JUDGE_PROVIDER <<< "$CONFIG_FACTS"

if [[ "$MODEL_PROVIDER" == "openai" || "$JUDGE_PROVIDER" == "openai" ]]; then
    : "${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY not set for the selected OpenAI model or judge provider.}"
fi
if [[ "$CONFIG_SDK_BACKEND" == "claude" && "$CLAUDE_AUTH_MODE" == "api_key" \
      && -z "${ANTHROPIC_API_KEY:-}" && -z "${ANTHROPIC_BASE_URL:-}" ]]; then
    echo "WARNING: Neither ANTHROPIC_API_KEY nor ANTHROPIC_BASE_URL is set."
    echo "         Claude Agent SDK sessions will fail unless configured in the YAML."
fi

if [[ -n "$SDK_AZURE_DIR" || -n "$MODEL_AZURE_DIR" || -n "$JUDGE_AZURE_DIR" ]]; then
    command -v az >/dev/null 2>&1 || {
        echo "ERROR: Azure CLI (az) is not installed or not on PATH."
        exit 1
    }
fi

verify_azure_profile() {
    local label="$1"
    local profile_dir="$2"

    if [[ ! -d "$profile_dir" ]]; then
        echo "ERROR: ${label} Azure profile directory not found: ${profile_dir}"
        exit 1
    fi
    if ! AZURE_CONFIG_DIR="$profile_dir" az account show --query state -o tsv 2>/dev/null | grep -qx "Enabled"; then
        echo "ERROR: ${label} Azure profile is not logged in: ${profile_dir}"
        exit 1
    fi
}

if [[ "$CLAUDE_AUTH_MODE" == "azure_cli_helper" ]]; then
    verify_azure_profile "Claude SDK" "$SDK_AZURE_DIR"
fi
if [[ -n "$MODEL_AZURE_DIR" ]]; then
    verify_azure_profile "Agent model" "$MODEL_AZURE_DIR"
fi
if [[ -n "$JUDGE_AZURE_DIR" ]]; then
    verify_azure_profile "Judge" "$JUDGE_AZURE_DIR"
fi

# ─── Pre-flight checks ───────────────────────────────────────────────
if [[ "$CONFIG_SDK_BACKEND" == "copilot" ]]; then
    python -c "import copilot" 2>/dev/null || {
        echo "ERROR: github-copilot-sdk not installed. Run: uv sync"
        exit 1
    }
else
    python -c "import claude_agent_sdk" 2>/dev/null || {
        echo "ERROR: claude-agent-sdk not installed. Run: uv sync"
        exit 1
    }
fi

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
