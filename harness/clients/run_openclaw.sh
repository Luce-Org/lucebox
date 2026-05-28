#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=204800}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${FA_WINDOW:=2048}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/openclaw.out"
OPENCLAW_BIN="${OPENCLAW_BIN:-$CLIENT_WORK_DIR/clients/openclaw/npm/bin/openclaw}"
require_client_binary "OpenClaw" "$OPENCLAW_BIN" "openclaw" "OPENCLAW_BIN"
HOME_DIR="$LOG_DIR/openclaw-home"
CONFIG_PATCH="$LOG_DIR/openclaw.patch.json"
PROVIDER_API="${PROVIDER_API:-openai-completions}"
OPENCLAW_AGENT_ARGS="${OPENCLAW_AGENT_ARGS:-}"
OPENCLAW_SUPPORTS_TOOLS="${OPENCLAW_SUPPORTS_TOOLS:-true}"
mkdir -p "$HOME_DIR"

cat > "$CONFIG_PATCH" <<JSON
{
  "models": {
    "mode": "merge",
    "providers": {
      "lucebox": {
        "baseUrl": "$BASE_URL/v1",
        "apiKey": "$API_KEY",
        "auth": "api-key",
        "api": "$PROVIDER_API",
        "contextWindow": $MAX_CTX,
        "maxTokens": $MAX_TOKENS,
        "models": [
          {
            "id": "$MODEL_ID",
            "name": "Lucebox DFlash",
            "api": "$PROVIDER_API",
            "contextWindow": $MAX_CTX,
            "maxTokens": $MAX_TOKENS,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "compat": {
              "supportsDeveloperRole": false,
              "supportsReasoningEffort": false,
              "supportsTools": $OPENCLAW_SUPPORTS_TOOLS,
              "maxTokensField": "max_tokens"
            }
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": "lucebox/$MODEL_ID",
      "workspace": "$REPO_DIR",
      "skipBootstrap": true,
      "contextInjection": "never",
      "bootstrapMaxChars": 1,
      "bootstrapTotalMaxChars": 1,
      "bootstrapPromptTruncationWarning": "off",
      "experimental": {
        "localModelLean": true
      },
      "compaction": {
        "mode": "default",
        "reserveTokens": 2048,
        "keepRecentTokens": 6000,
        "reserveTokensFloor": 0,
        "maxHistoryShare": 0.85,
        "recentTurnsPreserve": 2,
        "qualityGuard": {
          "enabled": false
        },
        "postIndexSync": "off",
        "postCompactionSections": []
      }
    }
  }
}
JSON

HOME="$HOME_DIR" "$OPENCLAW_BIN" config patch --file "$CONFIG_PATCH" > "$LOG_DIR/openclaw-config.out" 2>&1

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

agent_args=()
if [[ -n "$OPENCLAW_AGENT_ARGS" ]]; then
  read -r -a agent_args <<< "$OPENCLAW_AGENT_ARGS"
fi

set +e
HOME="$HOME_DIR" \
OPENAI_API_KEY="$API_KEY" \
timeout 420s "$OPENCLAW_BIN" agent \
  --local \
  --json \
  --model "lucebox/$MODEL_ID" \
  --session-id "lucebox-client-harness" \
  --timeout 300 \
  "${agent_args[@]}" \
  --message "$PROMPT" \
  < /dev/null > "$CLIENT_OUT" 2>&1
RC=$?
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
