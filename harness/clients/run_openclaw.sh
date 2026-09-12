#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=204800}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
# FA_WINDOW stays 0 (common.sh): finite windows break tool calls.
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${OPENCLAW_TIMEOUT:=3600}"
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/openclaw.out"
OPENCLAW_BIN="${OPENCLAW_BIN:-$CLIENT_WORK_DIR/clients/openclaw/npm/bin/openclaw}"
require_client_binary "OpenClaw" "$OPENCLAW_BIN" "openclaw" "OPENCLAW_BIN"
HOME_DIR="$(client_home openclaw)"
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

run_with_timeout "$OPENCLAW_TIMEOUT" env \
  HOME="$HOME_DIR" \
  "$OPENCLAW_BIN" config patch --file "$CONFIG_PATCH" \
  > "$LOG_DIR/openclaw-config.out" 2>&1

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

set +e
if [[ "$HARNESS_INTERACTIVE" == "1" ]]; then
  openclaw_cmd=("$OPENCLAW_BIN" tui --local --session lucebox-client-harness)
  if [[ -n "$INTERACTIVE_PROMPT" ]]; then openclaw_cmd+=(--message "$INTERACTIVE_PROMPT"); fi
  run_interactive_client "OpenClaw" "$CLIENT_OUT" env \
    HOME="$HOME_DIR" OPENAI_API_KEY="$API_KEY" "${openclaw_cmd[@]}"
  RC=$?
else
  openclaw_cmd=(
    "$OPENCLAW_BIN" agent
    --local
    --json
    --model "lucebox/$MODEL_ID"
    --session-id "lucebox-client-harness"
  )
  if [[ "$OPENCLAW_TIMEOUT" != "0" ]]; then
    openclaw_cmd+=(--timeout "$OPENCLAW_TIMEOUT")
  fi
  if [[ -n "$OPENCLAW_AGENT_ARGS" ]]; then
    read -r -a agent_args <<< "$OPENCLAW_AGENT_ARGS"
    openclaw_cmd+=("${agent_args[@]}")
  fi
  openclaw_cmd+=(--message "$PROMPT")
  run_with_timeout "$OPENCLAW_TIMEOUT" env \
    HOME="$HOME_DIR" \
    OPENAI_API_KEY="$API_KEY" \
    "${openclaw_cmd[@]}" \
    < /dev/null > "$CLIENT_OUT" 2>&1
  RC=$?
fi
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
