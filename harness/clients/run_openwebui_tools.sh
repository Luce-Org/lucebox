#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=65536}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${MARKER:=OPENWEBUI_TOOL_OK}"
: "${OPENWEBUI_FUNCTION_CALLING:=native}"
: "${OPENWEBUI_NATIVE_TOOL_CHOICE:=1}"
: "${OPENWEBUI_DISABLE_BUILTIN_TOOLS:=1}"
if [[ -z "${OPENWEBUI_REQUIRE_TOOL_EXECUTION+x}" ]]; then
  if [[ "$OPENWEBUI_FUNCTION_CALLING" == "native" ]]; then
    OPENWEBUI_REQUIRE_TOOL_EXECUTION=0
  else
    OPENWEBUI_REQUIRE_TOOL_EXECUTION=1
  fi
fi
source "$SCRIPT_DIR/common.sh"

WEBUI_PORT="${WEBUI_PORT:-18081}"
WEBUI_BASE="http://127.0.0.1:$WEBUI_PORT"
CLIENT_OUT="$LOG_DIR/openwebui-tools.out"
WEBUI_LOG="$LOG_DIR/openwebui.log"
OPENWEBUI_BIN="${OPENWEBUI_BIN:-$CLIENT_WORK_DIR/clients/openwebui/venv/bin/open-webui}"
require_client_binary "Open WebUI" "$OPENWEBUI_BIN" "openwebui" "OPENWEBUI_BIN"
DATA_DIR="$LOG_DIR/openwebui-data"
TOOL_JSON="$LOG_DIR/openwebui-tool-create.json"
TOOL_CREATE_OUT="$LOG_DIR/openwebui-tool-create.out"
TOOL_EXEC_LOG="$LOG_DIR/openwebui-tool-exec.log"
REQUEST_JSON="$LOG_DIR/openwebui-tool-request.json"
MODEL_JSON="$LOG_DIR/openwebui-model-create.json"
MODEL_CREATE_OUT="$LOG_DIR/openwebui-model-create.out"
AUTH_JSON="$LOG_DIR/openwebui-auth.json"
AUTH_OUT="$LOG_DIR/openwebui-auth.out"
WEBUI_ADMIN_EMAIL_VALUE="${WEBUI_ADMIN_EMAIL_VALUE:-admin@localhost}"
WEBUI_ADMIN_PASSWORD_VALUE="${WEBUI_ADMIN_PASSWORD_VALUE:-admin}"
CURL_MAX_TIME="${CURL_MAX_TIME:-300}"
TOOL_ID="${TOOL_ID:-lucebox_harness_probe}"
if [[ "$OPENWEBUI_DISABLE_BUILTIN_TOOLS" == "1" ]]; then
  WEBUI_MODEL_ID="${OPENWEBUI_MODEL_ID:-${MODEL_ID}-tools}"
else
  WEBUI_MODEL_ID="${OPENWEBUI_MODEL_ID:-$MODEL_ID}"
fi
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
LUCEBOX_OPENWEBUI_TOOL_LOG="$TOOL_EXEC_LOG" \
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

if [[ "$OPENWEBUI_DISABLE_BUILTIN_TOOLS" == "1" ]]; then
  MODEL_ID="$MODEL_ID" WEBUI_MODEL_ID="$WEBUI_MODEL_ID" python3 - "$MODEL_JSON" <<'PY'
import json
import os
import sys

model_id = os.environ["MODEL_ID"]
webui_model_id = os.environ["WEBUI_MODEL_ID"]
disabled_builtin_tools = {
    key: False
    for key in (
        "time",
        "knowledge",
        "chats",
        "memory",
        "web_search",
        "image_generation",
        "code_interpreter",
        "notes",
        "channels",
        "tasks",
        "automations",
        "calendar",
    )
}
body = {
    "id": webui_model_id,
    "base_model_id": model_id,
    "name": webui_model_id,
    "meta": {
        "description": "Local Lucebox DFlash model for harness tool testing.",
        "capabilities": {
            "builtin_tools": False,
        },
        "builtinTools": disabled_builtin_tools,
    },
    "params": {},
    "access_grants": [],
    "is_active": True,
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(body, f)
PY

  MODEL_STATUS=$(curl -sS -o "$MODEL_CREATE_OUT" -w "%{http_code}" \
    "$WEBUI_BASE/api/v1/models/create" \
    -H "Authorization: Bearer $WEBUI_TOKEN" \
    -H "Content-Type: application/json" \
    -d @"$MODEL_JSON")

  if [[ "$MODEL_STATUS" != 2* ]]; then
    echo "Open WebUI model capability setup failed with HTTP $MODEL_STATUS" > "$CLIENT_OUT"
    cat "$MODEL_CREATE_OUT" >> "$CLIENT_OUT"
    finish_report "$CLIENT_OUT" 1
    exit 1
  fi
fi

TOOL_ID="$TOOL_ID" python3 - "$TOOL_JSON" <<'PY'
import json
import os
import sys

tool_id = os.environ["TOOL_ID"]
content = r'''
from pathlib import Path
from pydantic import Field
import json
import os
import time


class Tools:
    def get_lucebox_harness_marker(
        self,
        question: str = Field(..., description="The harness question that must be answered by this tool."),
    ) -> str:
        """
        Return the Lucebox Open WebUI harness marker and record that this Open WebUI tool executed.
        """
        result = {
            "marker": "OPENWEBUI_TOOL_OK",
            "readme_heading": "Projects",
            "dflash_heading": "Luce DFlash",
            "qwen36_supported": True,
            "question": question,
            "ts": int(time.time()),
        }
        log_path = os.environ.get("LUCEBOX_OPENWEBUI_TOOL_LOG")
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")
        return "OPENWEBUI_TOOL_RESULT marker=OPENWEBUI_TOOL_OK readme_heading=Projects dflash_heading=Luce_DFlash qwen36_supported=true"
'''

body = {
    "id": tool_id,
    "name": "Lucebox Harness Probe",
    "content": content,
    "meta": {
        "description": "Tool used by the Lucebox harness to verify Open WebUI tool execution."
    },
    "access_grants": [],
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(body, f)
PY

CREATE_STATUS=$(curl -sS -o "$TOOL_CREATE_OUT" -w "%{http_code}" \
  "$WEBUI_BASE/api/v1/tools/create" \
  -H "Authorization: Bearer $WEBUI_TOKEN" \
  -H "Content-Type: application/json" \
  -d @"$TOOL_JSON")

if [[ "$CREATE_STATUS" != 2* ]]; then
  echo "Open WebUI tool creation failed with HTTP $CREATE_STATUS" > "$CLIENT_OUT"
  cat "$TOOL_CREATE_OUT" >> "$CLIENT_OUT"
  finish_report "$CLIENT_OUT" 1
  exit 1
fi

OPENWEBUI_TOOL_PROMPT="${OPENWEBUI_TOOL_PROMPT:-Use the Open WebUI tool get_lucebox_harness_marker with question 'verify Lucebox Open WebUI tool execution'. Do not answer from memory. After the tool returns, reply with the marker OPENWEBUI_TOOL_OK and then OK_DONE.}"

OPENWEBUI_TOOL_PROMPT="$OPENWEBUI_TOOL_PROMPT" \
WEBUI_MODEL_ID="$WEBUI_MODEL_ID" \
TOOL_ID="$TOOL_ID" \
MAX_TOKENS="$MAX_TOKENS" \
OPENWEBUI_FUNCTION_CALLING="$OPENWEBUI_FUNCTION_CALLING" \
OPENWEBUI_NATIVE_TOOL_CHOICE="$OPENWEBUI_NATIVE_TOOL_CHOICE" \
python3 - "$REQUEST_JSON" <<'PY'
import json
import os
import sys

function_calling = os.environ.get("OPENWEBUI_FUNCTION_CALLING", "native")
body = {
    "model": os.environ["WEBUI_MODEL_ID"],
    "chat_id": "local:lucebox-openwebui-tools",
    "id": "assistant-openwebui-tools-1",
    "user_message": {
        "id": "user-openwebui-tools-1",
    },
    "messages": [
        {"role": "user", "content": os.environ["OPENWEBUI_TOOL_PROMPT"]},
    ],
    "stream": False,
    "max_tokens": int(os.environ.get("MAX_TOKENS", "512")),
    "temperature": 0,
    "tool_ids": [os.environ["TOOL_ID"]],
    "params": {
        "function_calling": function_calling,
    },
}
if function_calling == "native" and os.environ.get("OPENWEBUI_NATIVE_TOOL_CHOICE", "1") != "0":
    body["tool_choice"] = {
        "type": "function",
        "function": {
            "name": "get_lucebox_harness_marker",
        },
    }
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(body, f)
PY

set +e
CHAT_STATUS=$(curl -sS -o "$CLIENT_OUT" -w "%{http_code}" \
  --max-time "$CURL_MAX_TIME" \
  "$WEBUI_BASE/api/chat/completions" \
  -H "Authorization: Bearer $WEBUI_TOKEN" \
  -H "Content-Type: application/json" \
  -d @"$REQUEST_JSON")
RC=$?
set -e

if [[ "$RC" -eq 0 && "$CHAT_STATUS" != 2* ]]; then
  RC=1
fi
if [[ "$RC" -eq 0 && "$OPENWEBUI_FUNCTION_CALLING" == "native" && "$OPENWEBUI_REQUIRE_TOOL_EXECUTION" == "0" ]]; then
  if ! grep -q '"tool_calls"' "$CLIENT_OUT" || ! grep -q 'get_lucebox_harness_marker' "$CLIENT_OUT"; then
    RC=1
  fi
else
  if [[ "$RC" -eq 0 ]] && ! grep -q "$MARKER" "$CLIENT_OUT"; then
    RC=1
  fi
  if [[ "$RC" -eq 0 ]] && { [[ ! -f "$TOOL_EXEC_LOG" ]] || ! grep -q "$MARKER" "$TOOL_EXEC_LOG"; }; then
    RC=1
  fi
fi

{
  echo "--- request ---"
  cat "$REQUEST_JSON" || true
  echo
  echo
  echo "--- validation mode ---"
  echo "function_calling=$OPENWEBUI_FUNCTION_CALLING"
  echo "require_tool_execution=$OPENWEBUI_REQUIRE_TOOL_EXECUTION"
  echo
  if [[ -f "$MODEL_CREATE_OUT" ]]; then
    echo
    echo "--- model create response ---"
    cat "$MODEL_CREATE_OUT" || true
    echo
  fi
  echo
  echo "--- tool create response ---"
  cat "$TOOL_CREATE_OUT" || true
  echo
  echo "--- tool execution log ---"
  if [[ -f "$TOOL_EXEC_LOG" ]]; then
    cat "$TOOL_EXEC_LOG" || true
  else
    echo "missing: $TOOL_EXEC_LOG"
  fi
  echo
  echo "--- openwebui tail ---"
  tail -n 160 "$WEBUI_LOG" || true
} >> "$CLIENT_OUT"

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
