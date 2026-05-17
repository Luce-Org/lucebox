#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=49152}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${CLAUDE_TOOLS:=default}"
: "${CLAUDE_TIMEOUT:=300}"
if [[ "${MODEL_SERVER:-}" == "llamacpp" ]]; then
  : "${LLAMA_COMPAT_PROXY:=anthropic}"
fi
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/claude-code.out"
CLAUDE_BIN="${CLAUDE_BIN:-$CLIENT_WORK_DIR/clients/claude_code/npm/bin/claude}"
HOME_DIR="$LOG_DIR/claude-home"
mkdir -p "$HOME_DIR"

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

set +e
HOME="$HOME_DIR" \
ANTHROPIC_API_KEY="$API_KEY" \
ANTHROPIC_BASE_URL="$BASE_URL" \
CLAUDE_CODE_API_BASE_URL="$BASE_URL" \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
CLAUDE_CODE_DISABLE_TELEMETRY=1 \
CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
timeout "${CLAUDE_TIMEOUT}s" "$CLAUDE_BIN" \
  --print \
  --output-format json \
  --model "$MODEL_ID" \
  --tools "$CLAUDE_TOOLS" \
  --permission-mode dontAsk \
  --no-session-persistence \
  "$PROMPT" \
  < /dev/null > "$CLIENT_OUT" 2>&1
RC=$?
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
