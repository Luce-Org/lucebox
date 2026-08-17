#!/usr/bin/env bash
set -euo pipefail

root="/home/lucebox5"
experiment="$root/tool-spec-cpu-20260813"
launcher="${LUCEBOX_LAUNCHER:-$experiment/run-deepseek-0731-cpu-tool.sh}"
executor="${TOOL_SPEC_EXECUTOR:-$experiment/cpu_sparse_tool_executor}"
profile="${TOOL_SPEC_PROFILE:-$experiment/profiles/lucebox5-cpu-lane-qualified.json}"
allowed="${TOOL_SPEC_ALLOW:-benchmark_cpu_sparse}"
native_wrapper_dir="${NATIVE_WRAPPER_DIR:-$experiment/native-wrapper}"
candidate_build="${CANDIDATE_BUILD:-$experiment/engine-ooo-spec/server/build-hip-dual}"
predictor_model="${PREDICTOR_MODEL:-$experiment/models/Qwen3-0.6B-Q8_0.gguf}"

for required in \
    "$launcher" \
    "$executor" \
    "$profile" \
    "$native_wrapper_dir/dflash_server" \
    "$candidate_build/dflash_server" \
    "$candidate_build/backend_ipc_daemon" \
    "$predictor_model"; do
    [[ -e "$required" ]] || {
        printf 'missing required path: %s\n' "$required" >&2
        exit 2
    }
done
if pgrep -x dflash_server >/dev/null; then
    printf 'a dflash_server is already running; refusing to overlap it\n' >&2
    exit 75
fi
if fuser -s /dev/kfd 2>/dev/null; then
    printf '/dev/kfd already has an owner; refusing to overlap it\n' >&2
    fuser -v /dev/kfd >&2 || true
    exit 75
fi

exec env \
    HOME="$root" \
    USER="lucebox5" \
    PATH="$root/.local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    ENGINE_DIR="$root/lucebox-engine-0731" \
    BUILD_DIR="$native_wrapper_dir" \
    CANDIDATE_BUILD="$candidate_build" \
    PREDICTOR_MODEL="$predictor_model" \
    PREDICTOR_GPU="${PREDICTOR_GPU:-1}" \
    PREDICTOR_MAX_CTX="${PREDICTOR_MAX_CTX:-4096}" \
    PREDICTOR_MAX_TOKENS="${PREDICTOR_MAX_TOKENS:-256}" \
    PREDICTOR_CONFIDENCE="${PREDICTOR_CONFIDENCE:-0.75}" \
    PREDICTOR_SCHEDULE="${PREDICTOR_SCHEDULE:-before-model}" \
    PREFIX_CACHE_SLOTS_OVERRIDE="${PREFIX_CACHE_SLOTS_OVERRIDE:-}" \
    QUALIFIED_CONFIG_DIR="/opt/lucebox-manage/qualified/r9700_deepseek/runtime-config" \
    TARGET_MODEL="$root/lucebox-models/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf" \
    DRAFT_MODEL="$root/lucebox-models/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf" \
    SERVER_PORT="18145" \
    MODEL_CPU_AFFINITY="0-13,16-29" \
    TOOL_SPEC_EXECUTOR="$executor" \
    TOOL_SPEC_PROFILE="$profile" \
    TOOL_SPEC_ALLOW="$allowed" \
    TOOL_SPEC_CPU_AFFINITY="14-15,30-31" \
    TOOL_SPEC_MAX_MODEL_SLOWDOWN="1.05" \
    LUCEBOX_INFERENCE_PROFILE="quality" \
    "$launcher"
