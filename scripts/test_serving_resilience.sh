#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
build_env="${LUCEBOX_BUILD_ENV:-.lucebox/build.env}"
if [[ -f "$build_env" ]]; then source "$build_env"; fi
if [[ -n "${ROCM_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
./scripts/test_memory_guard.sh
build_dir="${LUCEBOX_BUILD_DIR:-server/build-hip}"
cmake --build "$build_dir" --target test_server_unit -j "${BUILD_JOBS:-6}"
"$build_dir/test_server_unit"
"${LUCEBOX_PYTHON:-python3}" -m unittest discover -s server/test -p test_model_router.py -v
