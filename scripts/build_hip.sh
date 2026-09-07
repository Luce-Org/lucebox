#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
build_env="${LUCEBOX_BUILD_ENV:-.lucebox/build.env}"
if [[ -f "$build_env" ]]; then source "$build_env"; fi
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
: "${HIP_ARCHITECTURES:?Set HIP_ARCHITECTURES to your GPU target(s), e.g. gfx1100}"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build_dir="${LUCEBOX_BUILD_DIR:-server/build-hip}"
cmake -S server -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip -DDFLASH27B_HIP_ARCHITECTURES="$HIP_ARCHITECTURES" \
  -DDFLASH27B_HIP_SM80_EQUIV="${HIP_SM80_EQUIV:-ON}" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DROCM_PATH="$ROCM_PATH"
cmake --build "$build_dir" --target dflash_server -j "${BUILD_JOBS:-6}"
