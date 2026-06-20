#!/usr/bin/env bash
# Integration test (GPU + model): the qwen35moe KVFlash pooled chunked-prefill
# path (generate_impl chunk loop) is correct under real eviction —
#   (1) it preserves protected (sink) context: a fact in the first chunk is
#       recalled even though the middle is evicted, and
#   (2) it is stable across pool sizes: two different pools produce the same
#       greedy (temp 0) answer (no pool-size-dependent corruption).
#
# Both arms use --kvflash with prompt >> pool so the chunk loop + eviction run
# (cold experts via DFLASH_EXPERT_BUDGET_MB keep moe_hybrid active so the MoE
# chunk loop is the live path). max_ctx 131072 keeps the gate from disabling
# KVFlash. This is the silent-corruption gate for PR A.
#
# Hardware-gated. TARGET=/path/...Q3_K_M.gguf bash test_kvflash_moe_paged.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_BIN="${DFLASH_SERVER_BIN:-$REPO/server/build/dflash_server}"
TARGET="${TARGET:-/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/home/peppi/models/qwen3-coder-chat-template.jinja}"
HOST=127.0.0.1; PORT="${PORT:-18080}"
LOCK="${DG_GPU_LOCK:-/tmp/dg_gpu.lock}"
EXPERT_CAP="${DFLASH_EXPERT_BUDGET_MB:-4000}"
NEEDLE="ZEBRA-9"

[ -x "$SERVER_BIN" ] || { echo "SKIP: server not built: $SERVER_BIN"; exit 0; }
[ -f "$TARGET" ]     || { echo "SKIP: target GGUF not found: $TARGET"; exit 0; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# Needle in the FIRST line (sink chunk, never evicted) + long filler so the
# prompt far exceeds the pool and the chunk loop must evict the middle.
PROMPT_FILE="$(mktemp)"
{
  echo "IMPORTANT: the secret deployment code is $NEEDLE. Remember it."
  for i in $(seq 1 400); do
    echo "Line $i: lorem ipsum dolor sit amet $i, consectetur $((i*7%97)) adipiscing $((i%13)) elit."
  done
  echo "What is the secret deployment code? Answer with just the code."
} > "$PROMPT_FILE"
REQ="$(mktemp)"
python3 - "$PROMPT_FILE" "$REQ" <<'PY'
import json,sys
t=open(sys.argv[1]).read()
json.dump({"model":"luce","messages":[{"role":"user","content":t}],
          "max_tokens":12,"temperature":0,"stream":False}, open(sys.argv[2],"w"))
PY

# Start server with the given pool, POST the fixed request, echo the answer.
# Asserts the chunk loop actually ran (pooled prefill + cold experts).
generate() {
    local pool="$1" log out; log="$(mktemp)"; out="$(mktemp)"
    ( flock "$LOCK" env DFLASH_EXPERT_BUDGET_MB="$EXPERT_CAP" "$SERVER_BIN" "$TARGET" \
        --host "$HOST" --port "$PORT" --max-ctx 131072 --kvflash "$pool" --model-name luce \
        --chat-template-file "$CHAT_TEMPLATE" >"$log" 2>&1 ) &
    local pid=$!
    for _ in $(seq 1 120); do
        curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1 && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    curl -fsS "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        --data @"$REQ" 2>/dev/null \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
          >"$out" 2>/dev/null || true
    pkill -9 -f "$SERVER_BIN .*--port $PORT" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    grep -qE 'result: [0-9]+ hot experts, [1-9][0-9]* cold experts' "$log" \
        || { cat "$log" >&2; fail "cold experts absent (pool=$pool): chunk loop not exercised"; }
    grep -qE 'pooled prefill' "$log" \
        || { cat "$log" >&2; fail "pooled prefill not engaged (pool=$pool): prompt did not exceed pool"; }
    cat "$out"; rm -f "$log" "$out"
}

echo "== Arm 1: --kvflash 2048 (chunk loop, evicting middle) =="
A="$(generate 2048)"; echo "  answer: $A"
echo "== Arm 2: --kvflash 4096 (chunk loop, different pool) =="
B="$(generate 4096)"; echo "  answer: $B"
rm -f "$PROMPT_FILE" "$REQ"

grep -q "$NEEDLE" <<<"$A" || fail "pool 2048 lost the sink needle ($NEEDLE): paging dropped protected context"
grep -q "$NEEDLE" <<<"$B" || fail "pool 4096 lost the sink needle ($NEEDLE)"
[ "$A" = "$B" ] || fail "pool-size-dependent divergence: '$A' != '$B'"
echo "PASS: pooled prefill preserves sink context ($NEEDLE) and is stable across pool sizes"
