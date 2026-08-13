#!/usr/bin/env bash
set -euo pipefail

# Reproducible model-backed qualification for the AMD q=5 DS4 path.
# One process serves every context so the final 2K leg exercises eviction
# after 16K. Optional A/B switches are deliberately explicit.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="${CHECKOUT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$CHECKOUT/server/build-hip-dual}"
SERVER_BIN="${SERVER_BIN:-$BUILD_DIR/dflash_server}"
TOKENIZER_HARNESS="${TOKENIZER_HARNESS:-$BUILD_DIR/test_tokenizer_harness}"
TARGET_MODEL="${TARGET_MODEL:?set TARGET_MODEL to the target GGUF path}"
DRAFT_MODEL="${DRAFT_MODEL:?set DRAFT_MODEL to the DSpark draft GGUF path}"
HOTNESS_CSV="${HOTNESS_CSV:?set HOTNESS_CSV to the expert hotness CSV path}"
DECODE_HOTNESS_CSV="${DECODE_HOTNESS_CSV:-}"
CONTEXT_CLIENT="${CONTEXT_CLIENT:-$CHECKOUT/harness/benchmarks/deepseek4/ds4_context_sweep.py}"
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
case "$Q5_VERIFY" in
    0|1) ;;
    *) echo "Q5_VERIFY must be 0 or 1" >&2; exit 2 ;;
esac
FP4_Q5_X4_PLUS1="${FP4_Q5_X4_PLUS1:-auto}"
CRITICAL_PATH_PLACEMENT="${CRITICAL_PATH_PLACEMENT:-0}"
MAIN_TO_PEER_RATE_EXPLICIT=0
if [[ -n "${MAIN_TO_PEER_RATE:-}" ]]; then
    MAIN_TO_PEER_RATE_EXPLICIT=1
fi
MAIN_TO_PEER_RATE="${MAIN_TO_PEER_RATE:-3.4}"
BALANCE_MIN_HOT="${BALANCE_MIN_HOT:-0}"
EXPERT_BUDGET_MB="${EXPERT_BUDGET_MB:-13200}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TARGETS="${TARGETS:-2048 4096 8192 16384 2048}"
VRAM_MONITOR_SECONDS="${VRAM_MONITOR_SECONDS:-2}"
HASH_MODELS="${HASH_MODELS:-0}"
CUDA_GRAPH_STATS_EVERY="${CUDA_GRAPH_STATS_EVERY:-200}"
CUDA_DISABLE_GRAPHS_DEVICES="${CUDA_DISABLE_GRAPHS_DEVICES:-}"
DYNAMIC_ROUTE_BALANCE="${DYNAMIC_ROUTE_BALANCE:-0}"
DYNAMIC_MAIN_SLOTS="${DYNAMIC_MAIN_SLOTS:-auto}"
DYNAMIC_MAIN_SLOTS_X2="${DYNAMIC_MAIN_SLOTS_X2:-}"
DYNAMIC_MAIN_SLOTS_X4="${DYNAMIC_MAIN_SLOTS_X4:-}"
EXPERT_TOP_K="${EXPERT_TOP_K:-4}"
RUN_ID="${RUN_ID:-}"
OUT_ROOT="${OUT_ROOT:-$CHECKOUT/results/ds4_q5_context_qualification}"

for executable in "$SERVER_BIN" "$TOKENIZER_HARNESS"; do
    if [[ ! -f "$executable" || ! -x "$executable" ]]; then
        echo "required executable is missing or not executable: $executable" >&2
        exit 2
    fi
done
for input in "$TARGET_MODEL" "$DRAFT_MODEL" "$HOTNESS_CSV" "$CONTEXT_CLIENT"; do
    if [[ ! -f "$input" || ! -r "$input" ]]; then
        echo "required input is missing or unreadable: $input" >&2
        exit 2
    fi
done
if [[ -n "$DECODE_HOTNESS_CSV" &&
      ( ! -f "$DECODE_HOTNESS_CSV" || ! -r "$DECODE_HOTNESS_CSV" ) ]]; then
    echo "decode hotness profile is missing or unreadable: $DECODE_HOTNESS_CSV" >&2
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
if [[ "${Q6_VERIFY:-0}" != 0 ]]; then
    echo "Q6_VERIFY is unsupported; use Q5_VERIFY=1" >&2
    exit 2
fi
case "$DYNAMIC_ROUTE_BALANCE" in
    0|1) ;;
    *) echo "DYNAMIC_ROUTE_BALANCE must be 0 or 1" >&2; exit 2 ;;
esac
if [[ ! "$EXPERT_TOP_K" =~ ^[1-9][0-9]*$ ]] || ((EXPERT_TOP_K > 6)); then
    echo "EXPERT_TOP_K must be an integer from 1 through 6" >&2
    exit 2
fi
if [[ "$DYNAMIC_MAIN_SLOTS" != auto ]] &&
   { [[ ! "$DYNAMIC_MAIN_SLOTS" =~ ^[1-9][0-9]*$ ]] ||
     ((DYNAMIC_MAIN_SLOTS > EXPERT_TOP_K)); }; then
    echo "DYNAMIC_MAIN_SLOTS must be auto or an integer from 1 through EXPERT_TOP_K ($EXPERT_TOP_K)" >&2
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
explicit_route_quotas=0
if [[ "$DYNAMIC_MAIN_SLOTS" != auto ]]; then
    ((explicit_route_quotas += 1))
fi
if [[ -n "$DYNAMIC_MAIN_SLOTS_X2" ]]; then
    ((explicit_route_quotas += 1))
fi
if [[ -n "$DYNAMIC_MAIN_SLOTS_X4" ]]; then
    ((explicit_route_quotas += 1))
fi
if ((explicit_route_quotas > 1)); then
    echo "set at most one dynamic main-slot quota" >&2
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
   ! awk -v value="$MAIN_TO_PEER_RATE" \
       'BEGIN { number = value + 0; max = 1.7976931348623157e308; exit !(number > 0 && number <= max) }'; then
    echo "MAIN_TO_PEER_RATE must be a finite number greater than zero" >&2
    exit 2
fi
if [[ ! "$BALANCE_MIN_HOT" =~ ^[0-9]+$ ]] ||
   (( ${#BALANCE_MIN_HOT} > 10 )) ||
   { (( ${#BALANCE_MIN_HOT} == 10 )) && [[ "$BALANCE_MIN_HOT" > "2147483647" ]]; }; then
    echo "BALANCE_MIN_HOT must be an integer from 0 through 2147483647" >&2
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
for numeric_setting in PORT MAX_CTX EXPERT_BUDGET_MB WARMUP RUNS MAX_TOKENS \
    VRAM_MONITOR_SECONDS CUDA_GRAPH_STATS_EVERY; do
    numeric_value="${!numeric_setting}"
    if [[ ! "$numeric_value" =~ ^(0|[1-9][0-9]{0,8})$ ]]; then
        echo "$numeric_setting must be a non-negative decimal integer with at most 9 digits" >&2
        exit 2
    fi
done
if (( PORT < 1 || PORT > 65535 || MAX_CTX < 1 || EXPERT_BUDGET_MB < 1 ||
      RUNS < 1 || MAX_TOKENS < 1 || CUDA_GRAPH_STATS_EVERY < 1 )); then
    echo "PORT, MAX_CTX, EXPERT_BUDGET_MB, RUNS, MAX_TOKENS, and CUDA_GRAPH_STATS_EVERY must be positive (PORT <= 65535)" >&2
    exit 2
fi
if [[ ! "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "EXPECTED_SHA256 must contain exactly 64 lowercase hexadecimal characters" >&2
    exit 2
fi
read -r -a target_args <<<"$TARGETS"
if (( ${#target_args[@]} == 0 )); then
    echo "TARGETS must contain at least one context length" >&2
    exit 2
fi
for target in "${target_args[@]}"; do
    if [[ ! "$target" =~ ^[1-9][0-9]{0,8}$ ]]; then
        echo "TARGETS must contain only positive integers" >&2
        exit 2
    fi
    if (( target + MAX_TOKENS > MAX_CTX )); then
        echo "target context $target plus MAX_TOKENS exceeds MAX_CTX=$MAX_CTX" >&2
        exit 2
    fi
done

DYNAMIC_BALANCE_ENV_NAME=""
DYNAMIC_BALANCE_ENV_VALUE=""
DYNAMIC_BALANCE_LABEL="off"
if [[ "$DYNAMIC_ROUTE_BALANCE" == 1 ]]; then
    if [[ "$DYNAMIC_MAIN_SLOTS" != auto ]]; then
        DYNAMIC_BALANCE_ENV_NAME="DFLASH_MOE_TP_DYNAMIC_MAIN_SLOTS"
        DYNAMIC_BALANCE_ENV_VALUE="$DYNAMIC_MAIN_SLOTS"
        DYNAMIC_BALANCE_LABEL="s${DYNAMIC_MAIN_SLOTS}"
    elif [[ -n "$DYNAMIC_MAIN_SLOTS_X2" ]]; then
        DYNAMIC_BALANCE_ENV_NAME="DFLASH_MOE_TP_DYNAMIC_MAIN_SLOTS_X2"
        DYNAMIC_BALANCE_ENV_VALUE="$DYNAMIC_MAIN_SLOTS_X2"
        DYNAMIC_BALANCE_LABEL="s2x${DYNAMIC_MAIN_SLOTS_X2}"
    elif [[ -n "$DYNAMIC_MAIN_SLOTS_X4" ]]; then
        DYNAMIC_BALANCE_ENV_NAME="DFLASH_MOE_TP_DYNAMIC_MAIN_SLOTS_X4"
        DYNAMIC_BALANCE_ENV_VALUE="$DYNAMIC_MAIN_SLOTS_X4"
        DYNAMIC_BALANCE_LABEL="s4x${DYNAMIC_MAIN_SLOTS_X4}"
    elif [[ "$EXPERT_TOP_K" == 4 && "$MAIN_TO_PEER_RATE_EXPLICIT" == 0 ]]; then
        # Preserve the qualified top-4 default. Automatic rate-based scaling is
        # for widened top-k or an explicit operator override.
        DYNAMIC_BALANCE_ENV_NAME="DFLASH_MOE_TP_DYNAMIC_MAIN_SLOTS"
        DYNAMIC_BALANCE_ENV_VALUE=3
        DYNAMIC_BALANCE_LABEL="s3"
    else
        DYNAMIC_BALANCE_ENV_NAME="DFLASH_MOE_TP_MAIN_TO_PEER_RATE"
        DYNAMIC_BALANCE_ENV_VALUE="$MAIN_TO_PEER_RATE"
        DYNAMIC_BALANCE_LABEL="r${MAIN_TO_PEER_RATE}"
    fi
fi

VERIFY_WIDTH=$((4 + Q5_VERIFY))
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="ds4-q${VERIFY_WIDTH}-k${EXPERT_TOP_K}-fr${FORCE_GRAPH_REPLAY}-direct${DIRECT_INDEXER_TOPK}-radix${BLOCK_RADIX_TOPK}-x4p1${FP4_Q5_X4_PLUS1}-cp${CRITICAL_PATH_PLACEMENT}-bal${DYNAMIC_BALANCE_LABEL}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
case "$RUN_ID" in
    .|..|*[!A-Za-z0-9._-]*)
        echo "RUN_ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
        exit 2
        ;;
esac
OUT_DIR="$OUT_ROOT/$RUN_ID"
SERVER_LOG="$OUT_DIR/server.log"

# The script changes physical cards 0 and 1 with rocm-smi. A visibility mask
# that reorders those cards would apply the performance levels to the wrong
# logical devices, so this qualification only accepts the canonical order.
for visibility_var in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
    if declare -p "$visibility_var" >/dev/null 2>&1 &&
       [[ "${!visibility_var}" != "0,1" ]]; then
        echo "$visibility_var must be unset or exactly 0,1" >&2
        exit 2
    fi
done

for required_command in flock pgrep python3 rocm-smi sha256sum; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "required command is unavailable: $required_command" >&2
        exit 2
    fi
done

# Serialize the host-wide performance-level changes across qualification runs.
# /dev/kfd is a stable, kernel-owned inode shared by every ROCm user. flock is
# advisory, so holding it does not interfere with ordinary ROCm device access.
readonly lock_device="/dev/kfd"
if [[ -L "$lock_device" || ! -c "$lock_device" ]]; then
    echo "ROCm lock device is missing or invalid: $lock_device" >&2
    exit 2
fi
if ! exec 9<>"$lock_device"; then
    echo "cannot open ROCm lock device: $lock_device" >&2
    exit 2
fi
if ! lock_owner="$(stat -Lc '%u' "/proc/$$/fd/9")" ||
   [[ "$lock_owner" != 0 || ! -c "/proc/$$/fd/9" ]]; then
    echo "ROCm lock device must be a root-owned character device: $lock_device" >&2
    exit 2
fi
if ! flock -n 9; then
    echo "another DS4 qualification run owns $lock_device" >&2
    exit 2
fi

if pgrep -f '(^|/)dflash_server([[:space:]]|$)' >/dev/null; then
    echo "another dflash_server is running; stop it before changing global GPU performance levels" >&2
    exit 2
fi

mkdir -p "$OUT_ROOT"
if ! mkdir "$OUT_DIR"; then
    echo "refusing to reuse existing output directory: $OUT_DIR" >&2
    exit 2
fi

perf_level() {
    rocm-smi -d "$1" --showperflevel 2>/dev/null |
        awk -F: '/Performance Level:/ { value=$NF; gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); print value; exit }'
}

restore_perf_level() {
    local gpu="$1"
    local level="$2"
    if [[ -n "$level" && "$level" =~ ^[A-Za-z0-9_-]+$ ]]; then
        rocm-smi -d "$gpu" --setperflevel "$level" >/dev/null 2>&1 || true
    fi
}

rocm-smi --showproductname --showdriverversion --showperflevel --showclocks \
    --showmeminfo vram >"$OUT_DIR/rocm-smi-before.txt" 2>&1 || true
gpu0_perf_before="$(perf_level 0 || true)"
gpu1_perf_before="$(perf_level 1 || true)"
if [[ -z "$gpu0_perf_before" || -z "$gpu1_perf_before" ]]; then
    echo "could not read both GPU performance levels; no settings were changed" >&2
    exit 2
fi

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
    restore_perf_level 0 "$gpu0_perf_before"
    restore_perf_level 1 "$gpu1_perf_before"
}
trap cleanup EXIT

rocm-smi -d 0 --setperflevel auto >/dev/null
rocm-smi -d 1 --setperflevel high >/dev/null

server_env=(
    env -i
    "HOME=$HOME"
    "USER=${USER:-unknown}"
    "PATH=$PATH"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "GGML_CUDA_GRAPH_STATS=1"
    "GGML_CUDA_GRAPH_STATS_EVERY=$CUDA_GRAPH_STATS_EVERY"
    "LUCE_CUDA_I32_REPEAT=1"
    "DFLASH_DS4_TOPK=$EXPERT_TOP_K"
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

# env -i deliberately removes ambient tuning knobs. Canonical visibility masks
# validated above remain part of the recorded launch environment.
for visibility_var in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; do
    if declare -p "$visibility_var" >/dev/null 2>&1; then
        server_env+=("$visibility_var=${!visibility_var}")
    fi
done

if [[ -n "$DECODE_HOTNESS_CSV" ]]; then
    server_env+=(
        "DFLASH_DS4_DECODE_HOTNESS_CSV=$DECODE_HOTNESS_CSV"
    )
fi
if [[ "$DYNAMIC_ROUTE_BALANCE" == 1 ]]; then
    server_env+=(
        "DFLASH_MOE_TP_DYNAMIC_ROUTE_BALANCE=1"
        "$DYNAMIC_BALANCE_ENV_NAME=$DYNAMIC_BALANCE_ENV_VALUE"
    )
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
    sha256sum -- "$SERVER_BIN"
    stat -c 'target_model=%n bytes=%s mtime=%y' -- "$TARGET_MODEL"
    stat -c 'draft_model=%n bytes=%s mtime=%y' -- "$DRAFT_MODEL"
    if [[ "$HASH_MODELS" == 1 ]]; then
        sha256sum -- "$TARGET_MODEL" "$DRAFT_MODEL"
    fi
    printf 'server_env='; printf '%q ' "${server_env[@]}"; echo
    printf 'server_args='; printf '%q ' "${server_args[@]}"; echo
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
} >"$OUT_DIR/manifest.txt"

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
