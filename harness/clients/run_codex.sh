#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=32768}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${CODEX_TIMEOUT:=3600}"
if [[ "${MODEL_SERVER:-}" == "llamacpp" ]]; then
  : "${LLAMA_COMPAT_PROXY:=responses}"
fi
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/codex.out"
LAST_MSG="$LOG_DIR/codex-last-message.txt"
CODEX_BIN="${CODEX_BIN:-$CLIENT_WORK_DIR/clients/codex/npm/bin/codex}"
require_client_binary "Codex" "$CODEX_BIN" "codex" "CODEX_BIN"
CODEX_HOME_DIR="$(client_home codex)"
CODEX_SANDBOX="${CODEX_SANDBOX:-danger-full-access}"
CODEX_WIRE_API="${CODEX_WIRE_API:-responses}"
mkdir -p "$CODEX_HOME_DIR"

cat > "$CODEX_HOME_DIR/config.toml" <<TOML
model = "$MODEL_ID"
model_provider = "luce"
approval_policy = "never"
sandbox_mode = "$CODEX_SANDBOX"

[model_providers.luce]
name = "Lucebox"
base_url = "$BASE_URL/v1"
env_key = "OPENAI_API_KEY"
wire_api = "$CODEX_WIRE_API"
TOML

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

codex_env=("HOME=$CODEX_HOME_DIR" "CODEX_HOME=$CODEX_HOME_DIR" "OPENAI_API_KEY=$API_KEY")
set +e
if [[ "$HARNESS_INTERACTIVE" == "1" ]]; then
  codex_cmd=("$CODEX_BIN" --cd "$REPO_DIR" --sandbox "$CODEX_SANDBOX" --model "$MODEL_ID")
  if [[ -n "$INTERACTIVE_PROMPT" ]]; then codex_cmd+=("$INTERACTIVE_PROMPT"); fi
  run_interactive_client "Codex" "$CLIENT_OUT" env "${codex_env[@]}" "${codex_cmd[@]}"
  RC=$?
else
  run_with_timeout "$CODEX_TIMEOUT" env "${codex_env[@]}" \
    "$CODEX_BIN" exec \
    --skip-git-repo-check \
    --sandbox "$CODEX_SANDBOX" \
    --model "$MODEL_ID" \
    --json \
    --output-last-message "$LAST_MSG" \
    "$PROMPT" \
    < /dev/null > "$CLIENT_OUT" 2>&1
  RC=$?
fi
set -e

if [[ "$HARNESS_INTERACTIVE" == "0" ]]; then
  cat "$LAST_MSG" >> "$CLIENT_OUT" 2>/dev/null || true
fi
finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
