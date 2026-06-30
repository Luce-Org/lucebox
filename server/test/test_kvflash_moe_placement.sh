#!/usr/bin/env bash
# Integration test (GPU + model): KVFlash pool-sized KV reservation keeps MoE
# experts hot at high max_ctx, where the full-KV reservation would force them
# cold. Validates the qwen35moe placement-reservation fix end to end.
#
# Hardware-gated (needs a ~24GB GPU + the 35B-A3B MoE GGUF). Not wired into
# ctest; run manually or on a numbered-run box per CONTRIBUTING.md.
#
#   TARGET=/path/Qwen3.6-35B-A3B-...Q3_K_M.gguf bash test_kvflash_moe_placement.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_BIN="${DFLASH_SERVER_BIN:-$REPO/server/build/dflash_server}"
TARGET="${TARGET:-/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/home/peppi/models/qwen3-coder-chat-template.jinja}"
HOST=127.0.0.1; PORT="${PORT:-18080}"
MAX_CTX="${MAX_CTX:-131072}"   # large enough that full-KV forces experts cold
POOL="${POOL:-49152}"
LOCK="${DG_GPU_LOCK:-/tmp/dg_gpu.lock}"

[ -x "$SERVER_BIN" ] || { echo "SKIP: server not built: $SERVER_BIN"; exit 0; }
[ -f "$TARGET" ]     || { echo "SKIP: target GGUF not found: $TARGET"; exit 0; }

fail() { echo "FAIL: $*" >&2; exit 1; }

# Launch the server, wait for /v1/models, capture the placement line, kill it.
# Echoes the dynamic-placement result line.
placement_line() {
    local extra=("$@") log; log="$(mktemp)"
    ( flock "$LOCK" "$SERVER_BIN" "$TARGET" \
        --host "$HOST" --port "$PORT" --max-ctx "$MAX_CTX" --model-name luce \
        --chat-template-file "$CHAT_TEMPLATE" "${extra[@]}" >"$log" 2>&1 ) &
    local pid=$!
    local ready=0
    for _ in $(seq 1 90); do
        curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    # Server-launcher runs under flock in a subshell; find + kill the real server.
    pkill -9 -f "$SERVER_BIN .*--port $PORT" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    [ "$ready" = 1 ] || { cat "$log" >&2; rm -f "$log"; fail "server did not become ready"; }
    grep -E 'dynamic placement result|kvflash.*disabled|placement reserves pool' "$log"
    rm -f "$log"
}

echo "== Arm A: no KVFlash @max_ctx $MAX_CTX (expect the cliff: cold experts > 0) =="
A="$(placement_line)"
echo "$A"
cold_a="$(sed -nE 's/.*result: [0-9]+ hot experts, ([0-9]+) cold experts.*/\1/p' <<<"$A")"
[ -n "$cold_a" ] || fail "could not parse Arm A cold-expert count"
[ "$cold_a" -gt 0 ] || fail "expected cold experts without KVFlash at max_ctx $MAX_CTX, got $cold_a"
echo "  -> $cold_a cold experts (cliff confirmed)"

echo "== Arm B: --kvflash $POOL @max_ctx $MAX_CTX (expect pool reservation, 0 cold) =="
B="$(placement_line --kvflash "$POOL")"
echo "$B"
grep -q 'placement reserves pool KV' <<<"$B" \
    || fail "KVFlash did not reduce the KV reservation to the pool"
cold_b="$(sed -nE 's/.*result: [0-9]+ hot experts, ([0-9]+) cold experts.*/\1/p' <<<"$B")"
[ -n "$cold_b" ] || fail "could not parse Arm B cold-expert count"
[ "$cold_b" -eq 0 ] || fail "expected 0 cold experts with KVFlash, got $cold_b"
echo "  -> 0 cold experts (experts stay hot via pool reservation)"

echo "PASS: kvflash MoE placement — pool reservation keeps experts hot (A: $cold_a cold -> B: 0 cold)"
