#!/usr/bin/env bash
# Qualified DeepSeek4 profile for a 20 GiB discrete AMD GPU plus 128 GiB
# Strix Halo. The discrete GPU must be hip:0 and Strix Halo hip:1 unless the
# device variables below are overridden.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 TARGET_GGUF DSPARK_GGUF [dflash_server arguments...]" >&2
    exit 2
fi

target=$1
draft=$2
shift 2

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
server_dir=$(cd "$script_dir/.." && pwd)
build_dir=${DFLASH_BUILD_DIR:-$server_dir/build-hip-dual}
binary=$build_dir/dflash_server

for path in "$target" "$draft"; do
    if [[ ! -f $path ]]; then
        echo "missing required file: $path" >&2
        exit 1
    fi
done
if [[ ! -f $binary || ! -x $binary ]]; then
    echo "missing executable: $binary" >&2
    exit 1
fi

main_device=${DS4_MAIN_DEVICE:-hip:0}
peer_device=${DS4_PEER_DEVICE:-1}
draft_device=${DS4_DRAFT_DEVICE:-1}
context=${DS4_CONTEXT:-135168}
expert_budget_mb=${DS4_EXPERT_BUDGET_MB:-10200}
port=${DS4_PORT:-8016}

export DFLASH_DS4_MOE_TP=1
export DFLASH_DS4_MOE_TP_INPROC=1
export DFLASH_DS4_MOE_TP_GPU=$peer_device
export DFLASH_EXPERT_BUDGET_MB=$expert_budget_mb
export DFLASH_DS4_TP_MAIN_TO_PEER_RATE=${DFLASH_DS4_TP_MAIN_TO_PEER_RATE:-100}
export DFLASH_DS4_TP_BALANCE_MIN_HOT=${DFLASH_DS4_TP_BALANCE_MIN_HOT:-31}
export DFLASH_DS4_TP_BALANCE_MAX_HOT=${DFLASH_DS4_TP_BALANCE_MAX_HOT:-50}

# A calibrated routing profile enables the critical-path placement policy.
# Without one, the server remains usable but placement and throughput are not
# the qualified profile documented in docs/DS4.md.
if [[ -n ${DFLASH_DS4_HOTNESS_CSV:-} ]]; then
    if [[ ! -f $DFLASH_DS4_HOTNESS_CSV ]]; then
        echo "missing routing profile: $DFLASH_DS4_HOTNESS_CSV" >&2
        exit 1
    fi
    export DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT=${DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT:-1}
else
    echo "warning: DFLASH_DS4_HOTNESS_CSV is unset; using uncalibrated placement" >&2
fi

export DFLASH_DS4_LONG_CONTEXT_CHUNK=${DFLASH_DS4_LONG_CONTEXT_CHUNK:-2048}
export DFLASH_DS4_DISABLE_LONG_CONTEXT_ARENA_HANDOFF=${DFLASH_DS4_DISABLE_LONG_CONTEXT_ARENA_HANDOFF:-1}
export DFLASH_CUDA_MMQ_FP2_AFFINE_PREFILL_ONLY=${DFLASH_CUDA_MMQ_FP2_AFFINE_PREFILL_ONLY:-1}
export DFLASH_CUDA_MMQ_FP2_AFFINE_CAPTURE=${DFLASH_CUDA_MMQ_FP2_AFFINE_CAPTURE:-1}
export DFLASH_MOE_PREFILL_MASKED_COLD=${DFLASH_MOE_PREFILL_MASKED_COLD:-0}
export DFLASH_DS4_HYBRID_PREFILL_GPU_HC=${DFLASH_DS4_HYBRID_PREFILL_GPU_HC:-1}
export DFLASH_DS4_HYBRID_PREFILL_EAGER=${DFLASH_DS4_HYBRID_PREFILL_EAGER:-1}
export DFLASH_MOE_EXPERT_MAJOR_PINNED_OUTPUT=${DFLASH_MOE_EXPERT_MAJOR_PINNED_OUTPUT:-1}

export LUCE_MMVQ_MAX_NCOLS=${LUCE_MMVQ_MAX_NCOLS:-4}
export DFLASH_HIP_NO_AUTO_UMA=${DFLASH_HIP_NO_AUTO_UMA:-1}
export DFLASH_DS4_TP_GROUPED_MMVQ=${DFLASH_DS4_TP_GROUPED_MMVQ:-1}
export DFLASH_MMID_GROUPED=${DFLASH_MMID_GROUPED:-1}
export DFLASH_MMID_GROUPED_TYPES=${DFLASH_MMID_GROUPED_TYPES:-15}
export DFLASH_CUDA_MMVQ_MOE_FP2_PACKED32=${DFLASH_CUDA_MMVQ_MOE_FP2_PACKED32:-1}
export DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24=${DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24:-1}
export DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24_DECODE_ONLY=${DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24_DECODE_ONLY:-1}
export DFLASH_CUDA_MMVQ_FP4_X4=${DFLASH_CUDA_MMVQ_FP4_X4:-1}
export DFLASH_DS4_TP_MASKED_ROUTES=${DFLASH_DS4_TP_MASKED_ROUTES:-1}
export DFLASH_DS4_TP_DEVICE_JOIN=${DFLASH_DS4_TP_DEVICE_JOIN:-1}
export DFLASH_DS4_TP_NATIVE_ROUTE_WIDTH=${DFLASH_DS4_TP_NATIVE_ROUTE_WIDTH:-1}
export DFLASH_DS4_TP_SPLIT_COUNT=${DFLASH_DS4_TP_SPLIT_COUNT:-1}
export DFLASH_DS4_TP_ROUTE_PREFORK=${DFLASH_DS4_TP_ROUTE_PREFORK:-1}
export DFLASH_DS4_TP_DEVICE_JOIN_SPLIT=${DFLASH_DS4_TP_DEVICE_JOIN_SPLIT:-1}
export DFLASH_DS4_TP_FUSED_HC_JOIN=${DFLASH_DS4_TP_FUSED_HC_JOIN:-1}
export DFLASH_DS4_TP_MAIN_ROUTE_WEIGHTS=${DFLASH_DS4_TP_MAIN_ROUTE_WEIGHTS:-1}
export DFLASH_DS4_TP_COARSE_OWNER=${DFLASH_DS4_TP_COARSE_OWNER:-1}
export DFLASH_DS4_TP_COARSE_OWNER_SPLIT=${DFLASH_DS4_TP_COARSE_OWNER_SPLIT:-0}
export GGML_BATCH_PEER_COPIES=${GGML_BATCH_PEER_COPIES:-1}
export DFLASH_CUDA_MMVQ_MOE_ROWS_PER_BLOCK=${DFLASH_CUDA_MMVQ_MOE_ROWS_PER_BLOCK:-2}
export DFLASH_DS4_TP_CAPTURE_CACHE_SLOTS=${DFLASH_DS4_TP_CAPTURE_CACHE_SLOTS:-4}
export DFLASH_DS4_TP_FUSED_CACHE_SLOTS=${DFLASH_DS4_TP_FUSED_CACHE_SLOTS:-9}
export DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY=${DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY:-1}
export DFLASH_DS4_GPU_ARGMAX_VERIFY=${DFLASH_DS4_GPU_ARGMAX_VERIFY:-1}

export DFLASH_DS4_SPEC=1
export DFLASH_DS4_DRAFT=$draft
export DFLASH_DS4_DRAFT_GPU=$draft_device
export DFLASH_DS4_SPEC_Q=${DFLASH_DS4_SPEC_Q:-4}
export DFLASH_DS4_PINNED_ROLLBACK=${DFLASH_DS4_PINNED_ROLLBACK:-1}
export DFLASH_DS4_FUSED_VERIFY=${DFLASH_DS4_FUSED_VERIFY:-1}
export DFLASH_DS4_TOPK=${DFLASH_DS4_TOPK:-6}

exec "$binary" "$target" \
    --target-device "$main_device" \
    --peer-access \
    --ds4-expert-top-k "$DFLASH_DS4_TOPK" \
    --ds4-prefill sparse \
    --chunk 2048 \
    --max-ctx "$context" \
    --prefix-cache-slots 0 \
    --prefill-cache-slots 0 \
    --host "${DS4_HOST:-127.0.0.1}" \
    --port "$port" \
    "$@"
