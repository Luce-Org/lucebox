#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=86016}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/opencode.out"
EXPORT_OUT="$LOG_DIR/opencode-export.json"
OPENCODE_BIN="${OPENCODE_BIN:-$CLIENT_WORK_DIR/clients/opencode/npm/bin/opencode}"
HOME_DIR="$LOG_DIR/opencode-home"
PROJECT_DIR="$LOG_DIR/opencode-project"
mkdir -p "$HOME_DIR/.config" "$HOME_DIR/.local/share" "$PROJECT_DIR"

for path in "$REPO_DIR"/* "$REPO_DIR"/.[!.]*; do
  [[ -e "$path" ]] || continue
  name="$(basename "$path")"
  [[ "$name" == ".git" ]] && continue
  [[ "$name" == "opencode.json" ]] && continue
  [[ -e "$PROJECT_DIR/$name" ]] || ln -s "$path" "$PROJECT_DIR/$name"
done

cat > "$PROJECT_DIR/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "lucebox/$MODEL_ID",
  "small_model": "lucebox/$MODEL_ID",
  "provider": {
    "lucebox": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Lucebox",
      "options": {
        "baseURL": "$BASE_URL/v1",
        "apiKey": "$API_KEY",
        "timeout": 600000,
        "chunkTimeout": 60000
      },
      "models": {
        "$MODEL_ID": {
          "name": "Lucebox DFlash",
          "limit": {
            "context": $MAX_CTX,
            "output": $MAX_TOKENS
          }
        }
      }
    }
  },
  "tools": {
    "write": false,
    "bash": false
  }
}
JSON

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

set +e
cd "$PROJECT_DIR"
HOME="$HOME_DIR" \
XDG_CONFIG_HOME="$HOME_DIR/.config" \
XDG_DATA_HOME="$HOME_DIR/.local/share" \
OPENAI_API_KEY="$API_KEY" \
timeout 300s "$OPENCODE_BIN" run \
  --pure \
  --model "lucebox/$MODEL_ID" \
  --format json \
  "$PROMPT" \
  < /dev/null > "$CLIENT_OUT" 2>&1
RC=$?
SESSION_ID="$(grep -m1 -o 'ses_[A-Za-z0-9]*' "$CLIENT_OUT" || true)"
if [[ -n "$SESSION_ID" ]]; then
  HOME="$HOME_DIR" \
  XDG_CONFIG_HOME="$HOME_DIR/.config" \
  XDG_DATA_HOME="$HOME_DIR/.local/share" \
  "$OPENCODE_BIN" export "$SESSION_ID" > "$EXPORT_OUT" 2>&1 || true
  cat "$EXPORT_OUT" >> "$CLIENT_OUT"
fi
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
