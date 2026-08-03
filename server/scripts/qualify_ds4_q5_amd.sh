#!/usr/bin/env bash
set -euo pipefail

# Reproducible model-backed qualification for the AMD q=5 DS4 path.
# One process serves every context so the final 2K leg exercises eviction
# after 16K. Optional A/B switches are deliberately explicit.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="${CHECKOUT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$CHECKOUT/server/build-hip-dual}"
SERVER_BIN="${SERVER_BIN:-$BUILD_DIR/dflash_server}"
TOKENIZER_HARNESS="${TOKENIZER_HARNESS:-$BUILD_DIR/test_tokenizer_harness}"
TARGET_MODEL="${TARGET_MODEL:?set TARGET_MODEL to the target GGUF path}"
DRAFT_MODEL="${DRAFT_MODEL:?set DRAFT_MODEL to the DSpark draft GGUF path}"
HOTNESS_CSV="${HOTNESS_CSV:?set HOTNESS_CSV to the expert hotness CSV path}"
DECODE_HOTNESS_CSV="${DECODE_HOTNESS_CSV:-}"
CONTEXT_CLIENT="${CONTEXT_CLIENT:-$SCRIPT_DIR/ds4_context_sweep.py}"
EXPECTED_SHA256="${EXPECTED_SHA256:-0f785a7ffa406498aafb14553966eaed0f52220fed0f7cc016b66921d104d194}"
PORT="${PORT:-18109}"
MAX_CTX="${MAX_CTX:-18432}"
CACHE_SLOTS="${CACHE_SLOTS:-auto}"
MMVQ_MAX_NCOLS="${MMVQ_MAX_NCOLS:-auto}"
FORCE_GRAPH_REPLAY="${FORCE_GRAPH_REPLAY:-0}"
SERIAL_INDEX_SCAN="${SERIAL_INDEX_SCAN:-0}"
DIRECT_INDEXER_TOPK="${DIRECT_INDEXER_TOPK:-1}"
BLOCK_RADIX_TOPK="${BLOCK_RADIX_TOPK:-1}"
PACK_Q4_INDEXER="${PACK_Q4_INDEXER:-0}"
Q5_VERIFY="${Q5_VERIFY:-1}"
Q6_VERIFY="${Q6_VERIFY:-0}"
FP4_Q5_X4_PLUS1="${FP4_Q5_X4_PLUS1:-auto}"
CRITICAL_PATH_PLACEMENT="${CRITICAL_PATH_PLACEMENT:-0}"
MAIN_TO_PEER_RATE="${MAIN_TO_PEER_RATE:-3.4}"
BALANCE_MIN_HOT="${BALANCE_MIN_HOT:-0}"
EXPERT_BUDGET_MB="${EXPERT_BUDGET_MB:-13200}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TARGETS="${TARGETS:-2048 4096 8192 16384 2048}"
VRAM_MONITOR_SECONDS="${VRAM_MONITOR_SECONDS:-2}"
SET_PERF_LEVEL="${SET_PERF_LEVEL:-1}"
HASH_MODELS="${HASH_MODELS:-0}"
CUDA_GRAPH_STATS_EVERY="${CUDA_GRAPH_STATS_EVERY:-200}"
CUDA_DISABLE_GRAPHS_DEVICES="${CUDA_DISABLE_GRAPHS_DEVICES:-}"
DYNAMIC_ROUTE_BALANCE="${DYNAMIC_ROUTE_BALANCE:-0}"
DYNAMIC_MAIN_SLOTS="${DYNAMIC_MAIN_SLOTS:-3}"
DYNAMIC_MAIN_SLOTS_X2="${DYNAMIC_MAIN_SLOTS_X2:-}"
DYNAMIC_MAIN_SLOTS_X4="${DYNAMIC_MAIN_SLOTS_X4:-}"
SHARED_FFN_PEER_FRACTION="${SHARED_FFN_PEER_FRACTION:-0}"
FUSED_OWNER_RESIDUAL="${FUSED_OWNER_RESIDUAL:-0}"
ALIGN_SHARED_IDS="${ALIGN_SHARED_IDS:-0}"
EXPERT_TOP_K="${EXPERT_TOP_K:-4}"
VERIFY_WIDTH=$((4 + Q5_VERIFY + 2 * Q6_VERIFY))
RUN_ID="${RUN_ID:-ds4-q${VERIFY_WIDTH}-fr${FORCE_GRAPH_REPLAY}-direct${DIRECT_INDEXER_TOPK}-radix${BLOCK_RADIX_TOPK}-x4p1${FP4_Q5_X4_PLUS1}-cp${CRITICAL_PATH_PLACEMENT}-r${MAIN_TO_PEER_RATE}-sf${SHARED_FFN_PEER_FRACTION}-or${FUSED_OWNER_RESIDUAL}-ai${ALIGN_SHARED_IDS}-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$CHECKOUT/results/ds4_q5_context_qualification}"
OUT_DIR="$OUT_ROOT/$RUN_ID"
SERVER_LOG="$OUT_DIR/server.log"

for required in "$SERVER_BIN" "$TOKENIZER_HARNESS" "$TARGET_MODEL" \
    "$DRAFT_MODEL" "$HOTNESS_CSV" "$CONTEXT_CLIENT"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required path: $required" >&2
        exit 2
    fi
done
if [[ -n "$DECODE_HOTNESS_CSV" && ! -e "$DECODE_HOTNESS_CSV" ]]; then
    echo "missing decode hotness path: $DECODE_HOTNESS_CSV" >&2
    exit 2
fi

case "$FORCE_GRAPH_REPLAY:$SERIAL_INDEX_SCAN" in
    0:0|0:1|1:0|1:1) ;;
    *) echo "FORCE_GRAPH_REPLAY and SERIAL_INDEX_SCAN must be 0 or 1" >&2; exit 2 ;;
esac
case "$DIRECT_INDEXER_TOPK" in
    0|1) ;;
    *) echo "DIRECT_INDEXER_TOPK must be 0 or 1" >&2; exit 2 ;;
esac
case "$BLOCK_RADIX_TOPK" in
    0|1) ;;
    *) echo "BLOCK_RADIX_TOPK must be 0 or 1" >&2; exit 2 ;;
esac
case "$PACK_Q4_INDEXER" in
    0|1) ;;
    *) echo "PACK_Q4_INDEXER must be 0 or 1" >&2; exit 2 ;;
esac
case "$Q5_VERIFY" in
    0|1) ;;
    *) echo "Q5_VERIFY must be 0 or 1" >&2; exit 2 ;;
esac
case "$Q6_VERIFY" in
    0|1) ;;
    *) echo "Q6_VERIFY must be 0 or 1" >&2; exit 2 ;;
esac
if ((Q5_VERIFY + Q6_VERIFY > 1)); then
    echo "Q5_VERIFY and Q6_VERIFY are mutually exclusive" >&2
    exit 2
fi
case "$DYNAMIC_ROUTE_BALANCE" in
    0|1) ;;
    *) echo "DYNAMIC_ROUTE_BALANCE must be 0 or 1" >&2; exit 2 ;;
esac
case "$FUSED_OWNER_RESIDUAL" in
    0|1) ;;
    *) echo "FUSED_OWNER_RESIDUAL must be 0 or 1" >&2; exit 2 ;;
esac
case "$ALIGN_SHARED_IDS" in
    0|1) ;;
    *) echo "ALIGN_SHARED_IDS must be 0 or 1" >&2; exit 2 ;;
esac
if [[ ! "$EXPERT_TOP_K" =~ ^[1-9][0-9]*$ ]] || ((EXPERT_TOP_K > 16)); then
    echo "EXPERT_TOP_K must be an integer from 1 through 16" >&2
    exit 2
fi
if [[ ! "$DYNAMIC_MAIN_SLOTS" =~ ^[1-9][0-9]*$ ]] ||
   ((DYNAMIC_MAIN_SLOTS > EXPERT_TOP_K)); then
    echo "DYNAMIC_MAIN_SLOTS must be an integer from 1 through EXPERT_TOP_K ($EXPERT_TOP_K)" >&2
    exit 2
fi
if [[ -n "$DYNAMIC_MAIN_SLOTS_X2" ]] &&
   { [[ ! "$DYNAMIC_MAIN_SLOTS_X2" =~ ^[1-9][0-9]*$ ]] ||
     ((DYNAMIC_MAIN_SLOTS_X2 < 2 || DYNAMIC_MAIN_SLOTS_X2 > 2 * EXPERT_TOP_K)); }; then
    echo "DYNAMIC_MAIN_SLOTS_X2 must be empty or an integer from 2 through $((2 * EXPERT_TOP_K))" >&2
    exit 2
fi
if [[ -n "$DYNAMIC_MAIN_SLOTS_X4" ]] &&
   { [[ ! "$DYNAMIC_MAIN_SLOTS_X4" =~ ^[1-9][0-9]*$ ]] ||
     ((DYNAMIC_MAIN_SLOTS_X4 < 4 || DYNAMIC_MAIN_SLOTS_X4 > 4 * EXPERT_TOP_K)); }; then
    echo "DYNAMIC_MAIN_SLOTS_X4 must be empty or an integer from 4 through $((4 * EXPERT_TOP_K))" >&2
    exit 2
fi
if [[ "$SHARED_FFN_PEER_FRACTION" != 0 &&
      "$SHARED_FFN_PEER_FRACTION" != auto ]] &&
   { [[ ! "$SHARED_FFN_PEER_FRACTION" =~ ^(0[.][0-9]+|1([.]0+)?)$ ]] ||
     ! awk -v value="$SHARED_FFN_PEER_FRACTION" \
         'BEGIN { exit !(value > 0 && value <= 1) }'; }; then
    echo "SHARED_FFN_PEER_FRACTION must be 0, auto, or a value in (0,1]" >&2
    exit 2
fi
case "$FP4_Q5_X4_PLUS1" in
    auto|0|1) ;;
    *) echo "FP4_Q5_X4_PLUS1 must be auto, 0, or 1" >&2; exit 2 ;;
esac
case "$CRITICAL_PATH_PLACEMENT" in
    0|1) ;;
    *) echo "CRITICAL_PATH_PLACEMENT must be 0 or 1" >&2; exit 2 ;;
esac
if [[ ! "$MAIN_TO_PEER_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v value="$MAIN_TO_PEER_RATE" 'BEGIN { exit !(value > 0) }'; then
    echo "MAIN_TO_PEER_RATE must be greater than zero" >&2
    exit 2
fi
if [[ ! "$BALANCE_MIN_HOT" =~ ^[0-9]+$ ]]; then
    echo "BALANCE_MIN_HOT must be a non-negative integer" >&2
    exit 2
fi
if [[ "$MMVQ_MAX_NCOLS" != auto && ! "$MMVQ_MAX_NCOLS" =~ ^[1-8]$ ]]; then
    echo "MMVQ_MAX_NCOLS must be auto or an integer from 1 through 8" >&2
    exit 2
fi
if [[ "$CACHE_SLOTS" != auto && ! "$CACHE_SLOTS" =~ ^([1-9]|1[0-2])$ ]]; then
    echo "CACHE_SLOTS must be auto or an integer from 1 through 12" >&2
    exit 2
fi
case "$HASH_MODELS" in
    0|1) ;;
    *) echo "HASH_MODELS must be 0 or 1" >&2; exit 2 ;;
esac
case "$SET_PERF_LEVEL" in
    0|1) ;;
    *) echo "SET_PERF_LEVEL must be 0 or 1" >&2; exit 2 ;;
esac

if pgrep -f "dflash_server .*--port ${PORT}([[:space:]]|$)" >/dev/null; then
    echo "benchmark port $PORT is already owned by another dflash_server" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"

server_pid=""
monitor_pid=""
cleanup() {
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$SET_PERF_LEVEL" == 1 ]]; then
    rocm-smi -d 0 --setperflevel auto >/dev/null 2>&1 || true
    rocm-smi -d 1 --setperflevel high >/dev/null 2>&1 || true
fi
printf '0\n' >/tmp/ds4_awidth
rm -f /tmp/ds4_spec_q

server_env=(
    env -i
    "HOME=$HOME"
    "USER=${USER:-unknown}"
    "PATH=$PATH"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "GGML_CUDA_GRAPH_STATS=1"
    "GGML_CUDA_GRAPH_STATS_EVERY=$CUDA_GRAPH_STATS_EVERY"
    "LUCE_CUDA_I32_REPEAT=1"
    "DFLASH_DS4_TOPK=4"
    "DFLASH_DS4_FUSED_VERIFY=1"
    "DFLASH_DS4_FUSED_HYBRID_DECODE=1"
    "DFLASH_DS4_TIMING=1"
    "DFLASH_CUDA_MMVQ_MOE_ROWS_PER_BLOCK=2"
    "DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24=1"
    "DFLASH_CUDA_MMVQ_MOE_FP2_PACKED32=0"
    "DFLASH_CUDA_MMVQ_FP4_X4=1"
    "DFLASH_ROCMFP2_FIXED_K=1"
    "DFLASH_ROCMFP3_FIXED_K=1"
    "DFLASH_ROCMFP4_UNROLL2=1"
    "DFLASH_MMID_GROUPED=1"
    "DFLASH_MMID_GROUPED_TYPES=8"
    "DFLASH_MMID_GROUPED_DEVICE=1"
    "DFLASH_DS4_MOE_TP=1"
    "DFLASH_DS4_MOE_TP_INPROC=1"
    "DFLASH_DS4_MOE_TP_GPU=1"
    "DFLASH_EXPERT_BUDGET_MB=$EXPERT_BUDGET_MB"
    "DFLASH_DS4_HOTNESS_CSV=$HOTNESS_CSV"
    "DFLASH_DS4_TP_CAPTURE_CACHE_SLOTS=4"
    "DFLASH_DS4_TP_MASKED_ROUTES=1"
    "DFLASH_DS4_TP_GROUPED_MMVQ=1"
    "DFLASH_DS4_TP_SPLIT_COUNT=1"
    "DFLASH_DS4_TP_ROUTE_PREFORK=1"
    "DFLASH_DS4_TP_DEVICE_JOIN=1"
    "DFLASH_DS4_TP_DEVICE_JOIN_SPLIT=1"
    "DFLASH_DS4_TP_FUSED_HC_JOIN=1"
    "DFLASH_DS4_TP_MAIN_ROUTE_WEIGHTS=1"
    "DFLASH_DS4_TP_COARSE_OWNER=1"
    "DFLASH_DS4_TP_COARSE_OWNER_SPLIT=0"
    "DFLASH_DS4_TP_NATIVE_ROUTE_WIDTH=1"
    "GGML_CUDA_BATCH_PEER_COPIES=1"
    "DFLASH_MOE_DUPLICATE_HOT_ON_COLD=1"
    "DFLASH_DS4_HYBRID_PREFILL_GPU_HC=1"
    "DFLASH_DS4_HYBRID_PREFILL_EAGER=1"
    "DFLASH_MOE_FULL_COLD_PARALLEL=1"
    "DFLASH_DS4_PREFILL_TRACE=0"
    "DFLASH_MOE_PREFILL_PERSISTENT_OWNER_ALLOC=1"
    "DFLASH_DS4_PINNED_ROLLBACK=1"
    "DFLASH_DS4_GPU_ARGMAX_VERIFY=1"
    "DFLASH_DS4_SPEC=1"
    "DFLASH_DS4_SPEC_Q=$VERIFY_WIDTH"
    "DFLASH_DS4_ADAPTIVE_WIDTH=0"
    "DFLASH_DS4_DRAFT=$DRAFT_MODEL"
    "DFLASH_DS4_DRAFT_GPU=0"
    "DFLASH_DS4_DRAFT_CONTEXT_KV_CACHE=1"
    "DFLASH_MOE_FUSED_COMBINE=0"
)

if [[ -n "$DECODE_HOTNESS_CSV" ]]; then
    server_env+=(
        "DFLASH_DS4_DECODE_HOTNESS_CSV=$DECODE_HOTNESS_CSV"
    )
fi
if [[ "$DYNAMIC_ROUTE_BALANCE" == 1 ]]; then
    server_env+=(
        "DFLASH_MOE_TP_DYNAMIC_ROUTE_BALANCE=1"
        "DFLASH_MOE_TP_DYNAMIC_MAIN_SLOTS=$DYNAMIC_MAIN_SLOTS"
    )
    if [[ -n "$DYNAMIC_MAIN_SLOTS_X2" ]]; then
        server_env+=(
            "DFLASH_MOE_TP_DYNAMIC_MAIN_SLOTS_X2=$DYNAMIC_MAIN_SLOTS_X2"
        )
    fi
    if [[ -n "$DYNAMIC_MAIN_SLOTS_X4" ]]; then
        server_env+=(
            "DFLASH_MOE_TP_DYNAMIC_MAIN_SLOTS_X4=$DYNAMIC_MAIN_SLOTS_X4"
        )
    fi
fi
if [[ "$SHARED_FFN_PEER_FRACTION" != 0 ]]; then
    server_env+=(
        "DFLASH_MOE_TP_SHARED_FFN_PEER_FRACTION=$SHARED_FFN_PEER_FRACTION"
        "DFLASH_MOE_TP_MAIN_TO_PEER_RATE=$MAIN_TO_PEER_RATE"
    )
fi
if [[ "$FUSED_OWNER_RESIDUAL" == 1 ]]; then
    server_env+=("DFLASH_MOE_TP_FUSED_OWNER_RESIDUAL=1")
fi
if [[ "$ALIGN_SHARED_IDS" == 1 ]]; then
    server_env+=("DFLASH_CUDA_MMVQ_MOE_ALIGN_SHARED_IDS=1")
fi
if [[ -n "$CUDA_DISABLE_GRAPHS_DEVICES" ]]; then
    server_env+=(
        "GGML_CUDA_DISABLE_GRAPHS_DEVICES=$CUDA_DISABLE_GRAPHS_DEVICES"
    )
fi

# Preserve only the explicit profiler-wrapper controls across env -i. Ordinary
# qualification runs leave these unset and retain the exact established env.
for profiler_var in PROFILED_SERVER_BIN ROCPROF_OUTPUT_DIR \
    ROCPROF_START_SECONDS ROCPROF_DURATION_SECONDS; do
    if [[ -n "${!profiler_var:-}" ]]; then
        server_env+=("$profiler_var=${!profiler_var}")
    fi
done

if [[ "$MMVQ_MAX_NCOLS" != auto ]]; then
    server_env+=("LUCE_MMVQ_MAX_NCOLS=$MMVQ_MAX_NCOLS")
fi
if [[ "$CACHE_SLOTS" != auto ]]; then
    server_env+=("DFLASH_DS4_TP_FUSED_CACHE_SLOTS=$CACHE_SLOTS")
fi

if [[ "$FORCE_GRAPH_REPLAY" == 1 ]]; then
    server_env+=("DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY=1")
fi
if [[ "$SERIAL_INDEX_SCAN" == 1 ]]; then
    server_env+=("GGML_DS4_FA_SERIAL_INDEX_SCAN=1")
fi
if [[ "$DIRECT_INDEXER_TOPK" == 1 ]]; then
    server_env+=("DFLASH_DS4_DIRECT_INDEXER_TOPK=1")
fi
if [[ "$BLOCK_RADIX_TOPK" == 1 ]]; then
    server_env+=("GGML_DS4_TOPK_BLOCK_RADIX=1")
fi
if [[ "$PACK_Q4_INDEXER" == 1 ]]; then
    server_env+=("GGML_DS4_INDEXER_PACK_Q4=1")
fi
if [[ "$Q5_VERIFY" == 1 ]]; then
    server_env+=("DFLASH_DS4_Q5_VERIFY=1")
fi
if [[ "$Q6_VERIFY" == 1 ]]; then
    server_env+=("DFLASH_DS4_Q6_VERIFY=1")
fi
if [[ "$FP4_Q5_X4_PLUS1" != auto ]]; then
    server_env+=("DFLASH_CUDA_MMVQ_FP4_Q5_X4_PLUS1=$FP4_Q5_X4_PLUS1")
fi
if [[ "$CRITICAL_PATH_PLACEMENT" == 1 ]]; then
    server_env+=(
        "DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT=1"
        "DFLASH_DS4_TP_MAIN_TO_PEER_RATE=$MAIN_TO_PEER_RATE"
        "DFLASH_DS4_TP_BALANCE_MIN_HOT=$BALANCE_MIN_HOT"
    )
fi
server_args=(
    "$SERVER_BIN" "$TARGET_MODEL"
    --host 127.0.0.1 --port "$PORT"
    --max-ctx "$MAX_CTX"
    --target-device hip:0
    --prefix-cache-slots 0
    --prefill-cache-slots 0
    --hard-limit-reply-budget 0
    --chunk 2048
    --ds4-fused-decode
    --ds4-expert-top-k "$EXPERT_TOP_K"
    --ds4-prefill sparse
    --peer-access
)

{
    echo "schema_version=1"
    echo "run_id=$RUN_ID"
    echo "source_commit=$(git -C "$CHECKOUT" rev-parse HEAD)"
    echo "force_graph_replay=$FORCE_GRAPH_REPLAY"
    echo "serial_index_scan=$SERIAL_INDEX_SCAN"
    echo "direct_indexer_topk=$DIRECT_INDEXER_TOPK"
    echo "block_radix_topk=$BLOCK_RADIX_TOPK"
    echo "pack_q4_indexer=$PACK_Q4_INDEXER"
    echo "q5_verify=$Q5_VERIFY"
    echo "q6_verify=$Q6_VERIFY"
    echo "verify_width=$VERIFY_WIDTH"
    echo "fp4_q5_x4_plus1=$FP4_Q5_X4_PLUS1"
    echo "critical_path_placement=$CRITICAL_PATH_PLACEMENT"
    echo "main_to_peer_rate=$MAIN_TO_PEER_RATE"
    echo "balance_min_hot=$BALANCE_MIN_HOT"
    echo "decode_hotness_csv=$DECODE_HOTNESS_CSV"
    echo "dynamic_route_balance=$DYNAMIC_ROUTE_BALANCE"
    echo "dynamic_main_slots=$DYNAMIC_MAIN_SLOTS"
    echo "dynamic_main_slots_x2=$DYNAMIC_MAIN_SLOTS_X2"
    echo "dynamic_main_slots_x4=$DYNAMIC_MAIN_SLOTS_X4"
    echo "shared_ffn_peer_fraction=$SHARED_FFN_PEER_FRACTION"
    echo "fused_owner_residual=$FUSED_OWNER_RESIDUAL"
    echo "align_shared_ids=$ALIGN_SHARED_IDS"
    echo "expert_top_k=$EXPERT_TOP_K"
    echo "cache_slots=$CACHE_SLOTS"
    echo "mmvq_max_ncols=$MMVQ_MAX_NCOLS"
    echo "targets=$TARGETS"
    echo "warmup=$WARMUP"
    echo "runs=$RUNS"
    echo "max_tokens=$MAX_TOKENS"
    echo "max_ctx=$MAX_CTX"
    echo "cuda_graph_stats_every=$CUDA_GRAPH_STATS_EVERY"
    echo "cuda_disable_graphs_devices=$CUDA_DISABLE_GRAPHS_DEVICES"
    sha256sum "$SERVER_BIN"
    stat -c 'target_model=%n bytes=%s mtime=%y' "$TARGET_MODEL"
    stat -c 'draft_model=%n bytes=%s mtime=%y' "$DRAFT_MODEL"
    if [[ "$HASH_MODELS" == 1 ]]; then
        sha256sum "$TARGET_MODEL" "$DRAFT_MODEL"
    fi
    printf 'server_env='; printf '%q ' "${server_env[@]}"; echo
    printf 'server_args='; printf '%q ' "${server_args[@]}"; echo
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
} >"$OUT_DIR/manifest.txt"

rocm-smi --showproductname --showdriverversion --showperflevel --showclocks \
    --showmeminfo vram >"$OUT_DIR/rocm-smi-before.txt" 2>&1 || true

"${server_env[@]}" "${server_args[@]}" >"$SERVER_LOG" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 900); do
    if grep -q "listening on" "$SERVER_LOG"; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -160 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done
if [[ "$ready" != 1 ]]; then
    echo "server did not become ready" >&2
    exit 1
fi

if [[ "$VRAM_MONITOR_SECONDS" -gt 0 ]]; then
    (
        while kill -0 "$server_pid" 2>/dev/null; do
            date -u '+sample_utc=%Y-%m-%dT%H:%M:%SZ'
            rocm-smi --showuse --showmeminfo vram 2>&1 || true
            sleep "$VRAM_MONITOR_SECONDS"
        done
    ) >"$OUT_DIR/vram-monitor.log" 2>&1 &
    monitor_pid=$!
fi

# shellcheck disable=SC2206
target_args=($TARGETS)
python3 "$CONTEXT_CLIENT" \
    --url "http://127.0.0.1:$PORT" \
    --model dflash \
    --model-gguf "$TARGET_MODEL" \
    --tokenizer-harness "$TOKENIZER_HARNESS" \
    --targets "${target_args[@]}" \
    --warmup "$WARMUP" --runs "$RUNS" --max-tokens "$MAX_TOKENS" \
    --expected-sha256 "$EXPECTED_SHA256" \
    --json-out "$OUT_DIR/decode-client.json" \
    2>&1 | tee "$OUT_DIR/decode-client.log"

rocm-smi --showperflevel --showclocks --showmeminfo vram \
    >"$OUT_DIR/rocm-smi-after.txt" 2>&1 || true
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUT_DIR/manifest.txt"

echo "OUT_DIR=$OUT_DIR"
grep -E 'DSpark decode|chat DONE|graph.*(warm|replay|invalid)' "$SERVER_LOG" | tail -120 || true
