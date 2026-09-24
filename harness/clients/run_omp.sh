#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=65536}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${OMP_TOOLS:=read,grep,glob}"
: "${OMP_TIMEOUT:=3600}"
: "${OMP_STREAM_IDLE_TIMEOUT_MS:=3600000}"
if [[ ! "$OMP_TIMEOUT" =~ ^[0-9]+$ ]]; then
  echo "OMP_TIMEOUT must be a non-negative integer (seconds; 0 disables it)" >&2
  exit 2
fi
if [[ ! "$OMP_STREAM_IDLE_TIMEOUT_MS" =~ ^[0-9]+$ ]]; then
  echo "OMP_STREAM_IDLE_TIMEOUT_MS must be a non-negative integer (0 disables OMP's stream watchdog)" >&2
  exit 2
fi
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/omp.out"
OMP_BIN="${OMP_BIN:-$CLIENT_WORK_DIR/clients/omp/bin/omp}"
require_client_binary "OMP" "$OMP_BIN" "omp" "OMP_BIN"
HOME_DIR="$LOG_DIR/omp-home"
AGENT_DIR="$HOME_DIR/.omp/agent"
mkdir -p "$AGENT_DIR" "$HOME_DIR/sessions"

cat > "$AGENT_DIR/models.yml" <<YAML
providers:
  lucebox:
    baseUrl: "$BASE_URL/v1"
    auth: none
    api: openai-responses
    compat:
      supportsDeveloperRole: false
      supportsReasoningEffort: false
      supportsUsageInStreaming: true
      maxTokensField: max_tokens
      streamIdleTimeoutMs: $OMP_STREAM_IDLE_TIMEOUT_MS
    models:
      - id: "$MODEL_ID"
        name: "Lucebox DFlash"
        reasoning: false
        input: [text]
        contextWindow: $MAX_CTX
        maxTokens: $MAX_TOKENS
        cost:
          input: 0
          output: 0
          cacheRead: 0
          cacheWrite: 0
YAML

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

omp_env=(
  "HOME=$HOME_DIR"
  "PI_CODING_AGENT_DIR=$AGENT_DIR"
  "PI_CODING_AGENT_SESSION_DIR=$HOME_DIR/sessions"
  # An inherited OMP_PROFILE/PI_PROFILE would redirect OMP to a profile
  # agent dir and ignore the generated models.yml above.
  "OMP_PROFILE="
  "PI_PROFILE="
)
omp_cmd=(
  "$OMP_BIN"
  --model "lucebox/$MODEL_ID"
  --print
  --mode json
  --tools "$OMP_TOOLS"
  --no-session
  --no-extensions
  --no-skills
  --no-rules
  --no-title
  "$PROMPT"
)

set +e
cd "$REPO_DIR"
run_with_timeout "$OMP_TIMEOUT" env "${omp_env[@]}" "${omp_cmd[@]}" \
  < /dev/null > "$CLIENT_OUT" 2>&1
RC=$?
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
