#!/usr/bin/env bash
set -euo pipefail

root="/home/lucebox5"
experiment="$root/tool-spec-cpu-20260813"
launcher="$experiment/run-deepseek-0731-cpu-tool.sh"
executor="$experiment/cpu_sparse_tool_executor"
profile="$experiment/profiles/lucebox5-cpu-lane-qualified.json"

for required in "$launcher" "$executor" "$profile"; do
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
    BUILD_DIR="$root/codex-ds4-tool-spec-fix-20260812/build-tool-spec" \
    QUALIFIED_CONFIG_DIR="/opt/lucebox-manage/qualified/r9700_deepseek/runtime-config" \
    TARGET_MODEL="$root/lucebox-models/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf" \
    DRAFT_MODEL="$root/lucebox-models/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf" \
    SERVER_PORT="18145" \
    MODEL_CPU_AFFINITY="0-13,16-29" \
    TOOL_SPEC_EXECUTOR="$executor" \
    TOOL_SPEC_PROFILE="$profile" \
    TOOL_SPEC_ALLOW="benchmark_cpu_sparse" \
    TOOL_SPEC_CPU_AFFINITY="14-15,30-31" \
    TOOL_SPEC_MAX_MODEL_SLOWDOWN="1.05" \
    LUCEBOX_INFERENCE_PROFILE="quality" \
    "$launcher"
