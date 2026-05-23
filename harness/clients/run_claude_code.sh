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
export LUCEBOX_SERVER_BACKEND=cpp
source "$SCRIPT_DIR/common.sh"
preflight_require_bin claude

CLIENT_OUT="$LOG_DIR/claude-code.out"
CLAUDE_BIN="${CLAUDE_BIN:-$CLIENT_WORK_DIR/clients/claude_code/npm/bin/claude}"
HOME_DIR="$LOG_DIR/claude-home"
mkdir -p "$HOME_DIR"

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

# When PFLASH_SESSION_ID is set, start a thin proxy that injects
# extra_body.session_id into every /v1/messages request.  The claude CLI
# cannot inject extra_body natively, so the proxy does it transparently.
PROXY_PID=""
CLIENT_BASE_URL="$BASE_URL"
if [[ -n "${PFLASH_SESSION_ID:-}" ]]; then
  PROXY_PORT="${PFLASH_PROXY_PORT:-18082}"
  python3 "$SCRIPT_DIR/session_inject_proxy.py" \
    --host "$HOST" \
    --port "$PROXY_PORT" \
    --upstream "$BASE_URL" \
    --session-id "$PFLASH_SESSION_ID" \
    >> "$LOG_DIR/proxy.log" 2>&1 &
  PROXY_PID=$!
  for _i in $(seq 1 10); do
    if curl -fsS "http://$HOST:$PROXY_PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
      echo "session-inject proxy exited early; log: $LOG_DIR/proxy.log" >&2
      cat "$LOG_DIR/proxy.log" >&2 || true
      exit 1
    fi
  done
  CLIENT_BASE_URL="http://$HOST:$PROXY_PORT"
  echo "[run_claude_code] session-inject proxy up on $CLIENT_BASE_URL (session=$PFLASH_SESSION_ID)"
fi

set +e
HOME="$HOME_DIR" \
ANTHROPIC_API_KEY="$API_KEY" \
ANTHROPIC_BASE_URL="$CLIENT_BASE_URL" \
CLAUDE_CODE_API_BASE_URL="$CLIENT_BASE_URL" \
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

if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
  kill "$PROXY_PID" 2>/dev/null || true
  wait "$PROXY_PID" 2>/dev/null || true
fi

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
