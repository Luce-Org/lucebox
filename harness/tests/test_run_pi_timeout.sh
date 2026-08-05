#!/usr/bin/env bash

set -euo pipefail

RUN_PI="${1:?usage: test_run_pi_timeout.sh <run_pi.sh>}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

FAKE_BIN="$TMP_DIR/bin"
FAKE_TARGET="$TMP_DIR/model.gguf"
FAKE_DRAFT="$TMP_DIR/draft.gguf"
FAKE_SERVER="$TMP_DIR/dflash_server"
FAKE_PI="$TMP_DIR/pi"
mkdir -p "$FAKE_BIN"
touch "$FAKE_TARGET" "$FAKE_DRAFT"

cat >"$FAKE_SERVER" <<'EOF'
#!/usr/bin/env bash
exec sleep 600
EOF

cat >"$FAKE_PI" <<'EOF'
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

# Record the requested duration, then run the command immediately. This keeps
# the regression model-free and proves which timeout policy the harness uses.
cat >"$FAKE_BIN/timeout" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$1" >"${TIMEOUT_CAPTURE:?}"
shift
exec "$@"
EOF

chmod +x "$FAKE_SERVER" "$FAKE_PI" "$FAKE_BIN"/*

run_harness() {
    local stamp="$1"
    local capture="$2"
    shift 2
    env \
        PATH="$FAKE_BIN:$PATH" \
        RUN_DIR="$TMP_DIR/runs" \
        STAMP="$stamp" \
        TARGET="$FAKE_TARGET" \
        DRAFT="$FAKE_DRAFT" \
        FA_WINDOW=1 \
        DFLASH_SERVER_BIN="$FAKE_SERVER" \
        PI_BIN="$FAKE_PI" \
        AUTO_INSTALL_CLIENTS=0 \
        TIMEOUT_CAPTURE="$capture" \
        "$@" \
        bash "$RUN_PI"
}

default_capture="$TMP_DIR/default-timeout"
run_harness default "$default_capture" >/dev/null
if ! grep -Fxq '3600s' "$default_capture"; then
    echo "Pi launcher must allow one hour by default" >&2
    exit 1
fi
if ! grep -Eq '"httpIdleTimeoutMs"[[:space:]]*:[[:space:]]*0' \
        "$TMP_DIR/runs/default/pi-home/agent/settings.json"; then
    echo "Pi launcher must disable Pi's five-minute HTTP idle timeout" >&2
    exit 1
fi

custom_capture="$TMP_DIR/custom-timeout"
run_harness custom "$custom_capture" PI_TIMEOUT=900 >/dev/null
if ! grep -Fxq '900s' "$custom_capture"; then
    echo "PI_TIMEOUT must override the default duration" >&2
    exit 1
fi

unlimited_capture="$TMP_DIR/unlimited-timeout"
run_harness unlimited "$unlimited_capture" PI_TIMEOUT=0 >/dev/null
if [[ -e "$unlimited_capture" ]]; then
    echo "PI_TIMEOUT=0 must bypass the timeout command" >&2
    exit 1
fi

set +e
invalid_output="$(
    run_harness invalid "$TMP_DIR/invalid-timeout" PI_TIMEOUT=invalid 2>&1
)"
invalid_rc=$?
set -e
if [[ "$invalid_rc" -ne 2 ]]; then
    echo "invalid PI_TIMEOUT should exit 2, got $invalid_rc" >&2
    exit 1
fi
if ! grep -Fq 'PI_TIMEOUT must be a non-negative integer' <<<"$invalid_output"; then
    echo "invalid PI_TIMEOUT should explain the accepted format" >&2
    exit 1
fi

echo "Pi timeout harness: PASS"
