#!/usr/bin/env bash
set -euo pipefail

# Reproduce the qualified R9700 + Strix Halo q=5 attention split.
# Caller must provide the same model/profile paths required by
# qualify_ds4_q5_amd.sh. Every setting remains overrideable for explicit A/Bs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="${CHECKOUT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$CHECKOUT/server/build-hip-dual}"

cache="$BUILD_DIR/CMakeCache.txt"
if [[ ! -f "$cache" ]]; then
    echo "missing build cache: $cache" >&2
    echo "configure the HIP build with -DGGML_HIP_GRAPHS=ON" >&2
    exit 2
fi
if ! grep -qx 'GGML_HIP_GRAPHS:BOOL=ON' "$cache"; then
    echo "attention TP qualification requires GGML_HIP_GRAPHS=ON" >&2
    echo "reconfigure with: cmake -S $CHECKOUT/server -B $BUILD_DIR -DGGML_HIP_GRAPHS=ON" >&2
    exit 2
fi

export CHECKOUT BUILD_DIR
export TARGETS="${TARGETS:-2048}"
export WARMUP="${WARMUP:-2}"
export RUNS="${RUNS:-7}"
export MAX_TOKENS="${MAX_TOKENS:-128}"

export CRITICAL_PATH_PLACEMENT="${CRITICAL_PATH_PLACEMENT:-1}"
export MAIN_TO_PEER_RATE="${MAIN_TO_PEER_RATE:-4.4}"
export EXPERT_BUDGET_MB="${EXPERT_BUDGET_MB:-14350}"
export DYNAMIC_ROUTE_BALANCE="${DYNAMIC_ROUTE_BALANCE:-1}"
export DYNAMIC_MAIN_SLOTS="${DYNAMIC_MAIN_SLOTS:-3}"
export DYNAMIC_MAIN_SLOTS_X4="${DYNAMIC_MAIN_SLOTS_X4:-13}"
export FUSED_OWNER_RESIDUAL="${FUSED_OWNER_RESIDUAL:-1}"

# Two of eight output groups = 16 of 64 heads (25%). Ratio-4 layers are the
# only layers whose longer attention span amortized the heterogeneous fork.
export ATTENTION_TP_GROUPS="${ATTENTION_TP_GROUPS:-2}"
export ATTENTION_TP_RATIO="${ATTENTION_TP_RATIO:-4}"
export ATTENTION_TP_VALUES="${ATTENTION_TP_VALUES:-1}"
export ATTENTION_TP_DIRECT_KV="${ATTENTION_TP_DIRECT_KV:-1}"
export ATTENTION_TP_FLASH="${ATTENTION_TP_FLASH:-0}"
export ATTENTION_TP_OUTPUT_B="${ATTENTION_TP_OUTPUT_B:-1}"
export ATTENTION_TP_PROJECTION_ONLY="${ATTENTION_TP_PROJECTION_ONLY:-0}"
export ATTENTION_TP_CORE_ONLY="${ATTENTION_TP_CORE_ONLY:-0}"
export DST_STREAM_PEER_COPIES="${DST_STREAM_PEER_COPIES:-1}"
export ATTENTION_TP_PACKED_STAGE="${ATTENTION_TP_PACKED_STAGE:-1}"

# The packed destination-stream fork and deferred result join already provide
# the two required lifetime edges. Global generation fences only add overhead.
export SINGLE_COPY_EVENT_FENCES="${SINGLE_COPY_EVENT_FENCES:-0}"
export FORCE_GRAPH_REPLAY="${FORCE_GRAPH_REPLAY:-0}"

exec bash "$SCRIPT_DIR/qualify_ds4_q5_amd.sh"
