#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENTS="$REPO_ROOT/harness/clients"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

FAKE_BIN="$TMP_DIR/bin"
TIMEOUT_CAPTURE="$TMP_DIR/timeout-duration"
FAKE_REPO="$TMP_DIR/repo"
FAKE_TARGET="$TMP_DIR/model.gguf"
FAKE_DRAFT="$TMP_DIR/draft.gguf"
FAKE_SERVER="$TMP_DIR/dflash_server"
FAKE_CLIENT="$TMP_DIR/client"
mkdir -p "$FAKE_BIN"
mkdir -p "$FAKE_REPO"
touch "$FAKE_TARGET" "$FAKE_DRAFT"

cat >"$FAKE_SERVER" <<'EOF'
#!/usr/bin/env bash
exec sleep 600
EOF

cat >"$FAKE_CLIENT" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*"
EOF

cat >"$FAKE_BIN/curl" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$FAKE_BIN/timeout" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$1" >>"${TIMEOUT_CAPTURE:?}"
shift
exec "$@"
EOF
chmod +x "$FAKE_SERVER" "$FAKE_CLIENT" "$FAKE_BIN"/*
export PATH="$FAKE_BIN:$PATH"
export TIMEOUT_CAPTURE

SCRIPT_DIR="$CLIENTS"
CLIENT_WORK_DIR="$TMP_DIR/work"
RUN_DIR="$TMP_DIR/runs"
STAMP="timeout-policy"
source "$CLIENTS/common.sh"

if [[ "$(run_with_timeout 5 printf '%s' bounded)" != "bounded" ]]; then
  echo "bounded client timeout did not execute the command" >&2
  exit 1
fi
if ! grep -Fxq '5s' "$TIMEOUT_CAPTURE"; then
  echo "bounded client timeout used the wrong duration" >&2
  exit 1
fi
rm -f "$TIMEOUT_CAPTURE"
if [[ "$(run_with_timeout 0 printf '%s' unlimited)" != "unlimited" ]]; then
  echo "zero client timeout must execute without a deadline" >&2
  exit 1
fi
if [[ -e "$TIMEOUT_CAPTURE" ]]; then
  echo "zero client timeout must bypass the timeout command" >&2
  exit 1
fi

set +e
invalid_output="$(run_with_timeout invalid true 2>&1)"
invalid_rc=$?
set -e
if [[ "$invalid_rc" -ne 2 ]] ||
   ! grep -Fq 'client timeout must be a non-negative integer' <<<"$invalid_output"; then
  echo "invalid client timeout must fail with a useful error" >&2
  exit 1
fi

run_launcher() {
  local script="$1"
  local client_var="$2"
  local timeout_var="$3"
  local timeout_value="$4"
  shift 4
  env \
    PATH="$FAKE_BIN:$PATH" \
    REPO_DIR="$FAKE_REPO" \
    RUN_DIR="$TMP_DIR/runs" \
    STAMP="${script%.sh}-$timeout_value" \
    TARGET="$FAKE_TARGET" \
    DRAFT="$FAKE_DRAFT" \
    DFLASH_SERVER_BIN="$FAKE_SERVER" \
    AUTO_INSTALL_CLIENTS=0 \
    FA_WINDOW=1 \
    TIMEOUT_CAPTURE="$TIMEOUT_CAPTURE" \
    MARKER=timeout-policy-ok \
    PROMPT=timeout-policy-ok \
    "$client_var=$FAKE_CLIENT" \
    "$timeout_var=$timeout_value" \
    "$@" \
    bash "$CLIENTS/$script"
}

# Execute every CLI launcher with fake client/server binaries. This proves the
# configured deadline reaches the actual client process, not merely a helper
# or a matching source line. OpenClaw has two bounded client invocations: its
# config preflight and the agent itself.
launcher_cases=(
  'run_claude_code.sh|CLAUDE_BIN|CLAUDE_TIMEOUT|1'
  'run_codex.sh|CODEX_BIN|CODEX_TIMEOUT|1'
  'run_hermes.sh|HERMES_BIN|HERMES_TIMEOUT|1'
  'run_openclaw.sh|OPENCLAW_BIN|OPENCLAW_TIMEOUT|2'
  'run_opencode.sh|OPENCODE_BIN|OPENCODE_TIMEOUT|1'
)
for launcher_case in "${launcher_cases[@]}"; do
  IFS='|' read -r script client_var timeout_var expected_calls <<<"$launcher_case"

  rm -f "$TIMEOUT_CAPTURE"
  run_launcher "$script" "$client_var" "$timeout_var" 17 >/dev/null
  if [[ "$(grep -Fxc '17s' "$TIMEOUT_CAPTURE")" -ne "$expected_calls" ]]; then
    echo "$script did not apply its configured timeout to every client command" >&2
    exit 1
  fi

  rm -f "$TIMEOUT_CAPTURE"
  run_launcher "$script" "$client_var" "$timeout_var" 0 >/dev/null
  if [[ -e "$TIMEOUT_CAPTURE" ]]; then
    echo "$script timeout=0 must bypass the timeout command" >&2
    exit 1
  fi
done

set +e
opencode_invalid="$({
  run_launcher run_opencode.sh OPENCODE_BIN OPENCODE_TIMEOUT 17 \
    OPENCODE_REQUEST_TIMEOUT_MS=060000
} 2>&1)"
opencode_invalid_rc=$?
set -e
if [[ "$opencode_invalid_rc" -ne 2 ]] ||
   ! grep -Fq 'canonical non-negative integers' <<<"$opencode_invalid"; then
  echo "OpenCode must reject timeout values that would produce invalid JSON" >&2
  exit 1
fi

grep -Fq ': "${OPENCODE_TIMEOUT:=3600}"' "$CLIENTS/run_opencode.sh"
grep -Fq ': "${OPENCODE_REQUEST_TIMEOUT_MS:=3600000}"' "$CLIENTS/run_opencode.sh"
grep -Fq ': "${OPENCODE_CHUNK_TIMEOUT_MS:=3600000}"' "$CLIENTS/run_opencode.sh"
grep -Fq 'run_with_timeout "$OPENCODE_TIMEOUT"' "$CLIENTS/run_opencode.sh"
grep -Fq ': "${CODEX_TIMEOUT:=3600}"' "$CLIENTS/run_codex.sh"
grep -Fq 'run_with_timeout "$CODEX_TIMEOUT"' "$CLIENTS/run_codex.sh"
grep -Fq ': "${HERMES_TIMEOUT:=3600}"' "$CLIENTS/run_hermes.sh"
grep -Fq 'run_with_timeout "$HERMES_TIMEOUT"' "$CLIENTS/run_hermes.sh"
grep -Fq ': "${CLAUDE_TIMEOUT:=3600}"' "$CLIENTS/run_claude_code.sh"
grep -Fq 'run_with_timeout "$CLAUDE_TIMEOUT"' "$CLIENTS/run_claude_code.sh"
grep -Fq ': "${OPENCLAW_TIMEOUT:=3600}"' "$CLIENTS/run_openclaw.sh"
grep -Fq 'run_with_timeout "$OPENCLAW_TIMEOUT"' "$CLIENTS/run_openclaw.sh"
grep -Fq 'openclaw_cmd+=(--timeout "$OPENCLAW_TIMEOUT")' "$CLIENTS/run_openclaw.sh"
grep -Fq 'CURL_MAX_TIME="${CURL_MAX_TIME:-3600}"' "$CLIENTS/run_openwebui.sh"
grep -Fq 'CURL_MAX_TIME="${CURL_MAX_TIME:-3600}"' "$CLIENTS/run_openwebui_tools.sh"

if grep -REq 'timeout (300|420)s|--timeout 300|CURL_MAX_TIME:-300' "$CLIENTS"/*.sh; then
  echo "a short hard-coded real-client timeout remains" >&2
  exit 1
fi

echo "Client launcher timeout policy: PASS"
