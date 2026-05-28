#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=65536}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${PI_TOOLS:=read,grep,find,ls}"
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/pi.out"
PI_BIN="${PI_BIN:-$CLIENT_WORK_DIR/clients/pi/npm/bin/pi}"
require_client_binary "Pi" "$PI_BIN" "pi" "PI_BIN"
HOME_DIR="$LOG_DIR/pi-home"
AGENT_DIR="$HOME_DIR/agent"
PROVIDER_API="${PROVIDER_API:-openai-responses}"
mkdir -p "$AGENT_DIR" "$HOME_DIR/sessions"

cat > "$AGENT_DIR/settings.json" <<JSON
{
  "compaction": {
    "enabled": false
  }
}
JSON

cat > "$AGENT_DIR/models.json" <<JSON
{
  "providers": {
    "lucebox": {
      "baseUrl": "$BASE_URL/v1",
      "api": "$PROVIDER_API",
      "apiKey": "$API_KEY",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "$MODEL_ID",
          "name": "Lucebox DFlash",
          "api": "$PROVIDER_API",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": $MAX_CTX,
          "maxTokens": $MAX_TOKENS,
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        }
      ]
    }
  }
}
JSON

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

set +e
HOME="$HOME_DIR" \
PI_CODING_AGENT_DIR="$AGENT_DIR" \
PI_CODING_AGENT_SESSION_DIR="$HOME_DIR/sessions" \
PI_OFFLINE=1 \
timeout 300s "$PI_BIN" \
  --provider lucebox \
  --model "$MODEL_ID" \
  --print \
  --mode json \
  --tools "$PI_TOOLS" \
  --no-session \
  --offline \
  "$PROMPT" \
  < /dev/null > "$CLIENT_OUT" 2>&1
RC=$?
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
