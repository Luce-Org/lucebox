#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=262144}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
source "$SCRIPT_DIR/common.sh"

WEBUI_PORT="${WEBUI_PORT:-18081}"
WEBUI_BASE="http://127.0.0.1:$WEBUI_PORT"
CLIENT_OUT="$LOG_DIR/openwebui.out"
WEBUI_LOG="$LOG_DIR/openwebui.log"
OPENWEBUI_BIN="${OPENWEBUI_BIN:-$CLIENT_WORK_DIR/clients/openwebui/venv/bin/open-webui}"
DATA_DIR="$LOG_DIR/openwebui-data"
REQUEST_JSON="$LOG_DIR/openwebui-request.json"
AUTH_JSON="$LOG_DIR/openwebui-auth.json"
AUTH_OUT="$LOG_DIR/openwebui-auth.out"
WEBUI_ADMIN_EMAIL_VALUE="${WEBUI_ADMIN_EMAIL_VALUE:-admin@localhost}"
WEBUI_ADMIN_PASSWORD_VALUE="${WEBUI_ADMIN_PASSWORD_VALUE:-admin}"
CURL_MAX_TIME="${CURL_MAX_TIME:-300}"
mkdir -p "$DATA_DIR"

start_lucebox_server
trap 'stop_lucebox_server; [[ -n "${WEBUI_PID:-}" ]] && kill "$WEBUI_PID" 2>/dev/null || true' EXIT
wait_lucebox_server

WEBUI_SECRET_KEY="lucebox-local-secret" \
WEBUI_AUTH=False \
ENABLE_SIGNUP=False \
WEBUI_ADMIN_EMAIL="$WEBUI_ADMIN_EMAIL_VALUE" \
WEBUI_ADMIN_PASSWORD="$WEBUI_ADMIN_PASSWORD_VALUE" \
WEBUI_ADMIN_NAME="Lucebox Harness" \
DEFAULT_MODELS="$MODEL_ID" \
OPENAI_API_BASE_URL="$BASE_URL/v1" \
OPENAI_API_KEY="$API_KEY" \
DATA_DIR="$DATA_DIR" \
"$OPENWEBUI_BIN" serve --host 127.0.0.1 --port "$WEBUI_PORT" > "$WEBUI_LOG" 2>&1 &
WEBUI_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS "$WEBUI_BASE/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if ! kill -0 "$WEBUI_PID" 2>/dev/null; then
    echo "Open WebUI exited early; log: $WEBUI_LOG" >&2
    tail -n 120 "$WEBUI_LOG" >&2 || true
    exit 1
  fi
done

WEBUI_ADMIN_EMAIL_VALUE="$WEBUI_ADMIN_EMAIL_VALUE" \
WEBUI_ADMIN_PASSWORD_VALUE="$WEBUI_ADMIN_PASSWORD_VALUE" \
python3 - "$AUTH_JSON" <<'PY'
import json
import os
import sys

body = {
    "email": os.environ["WEBUI_ADMIN_EMAIL_VALUE"],
    "password": os.environ["WEBUI_ADMIN_PASSWORD_VALUE"],
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(body, f)
PY

AUTH_STATUS=$(curl -sS -o "$AUTH_OUT" -w "%{http_code}" \
  "$WEBUI_BASE/api/v1/auths/signin" \
  -H "Content-Type: application/json" \
  -d @"$AUTH_JSON")

if [[ "$AUTH_STATUS" != 2* ]]; then
  echo "Open WebUI signin failed with HTTP $AUTH_STATUS" > "$CLIENT_OUT"
  cat "$AUTH_OUT" >> "$CLIENT_OUT"
  finish_report "$CLIENT_OUT" 1
  exit 1
fi

WEBUI_TOKEN=$(python3 - "$AUTH_OUT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(data.get("token", ""))
PY
)

if [[ -z "$WEBUI_TOKEN" ]]; then
  echo "Open WebUI signin returned no token" > "$CLIENT_OUT"
  cat "$AUTH_OUT" >> "$CLIENT_OUT"
  finish_report "$CLIENT_OUT" 1
  exit 1
fi

PROMPT="$PROMPT" MODEL_ID="$MODEL_ID" python3 - "$REQUEST_JSON" <<'PY'
import json
import os
import sys

body = {
    "model": os.environ["MODEL_ID"],
    "messages": [
        {"role": "user", "content": os.environ["PROMPT"]},
    ],
    "stream": False,
    "max_tokens": 256,
    "temperature": 0,
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(body, f)
PY

set +e
CHAT_STATUS=$(curl -sS -o "$CLIENT_OUT" -w "%{http_code}" \
  --max-time "$CURL_MAX_TIME" \
  "$WEBUI_BASE/openai/chat/completions" \
  -H "Authorization: Bearer $WEBUI_TOKEN" \
  -H "Content-Type: application/json" \
  -d @"$REQUEST_JSON")
RC=$?
set -e

if [[ "$RC" -eq 0 && "$CHAT_STATUS" != 2* ]]; then
  RC=1
fi
if [[ "$RC" -eq 0 ]] && ! grep -q "$MARKER" "$CLIENT_OUT"; then
  RC=1
fi

{
  echo
  echo "--- openwebui tail ---"
  tail -n 120 "$WEBUI_LOG" || true
} >> "$CLIENT_OUT"
finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
