#!/usr/bin/env bash
set -euo pipefail

# Launch dflash_server under a delayed rocprofv3 collection window. This is a
# diagnostic wrapper for model-load-heavy runs where ptrace attachment is not
# permitted by the host policy.

PROFILED_SERVER_BIN="${PROFILED_SERVER_BIN:?set PROFILED_SERVER_BIN}"
ROCPROF_OUTPUT_DIR="${ROCPROF_OUTPUT_DIR:?set ROCPROF_OUTPUT_DIR}"
ROCPROF_START_SECONDS="${ROCPROF_START_SECONDS:-180}"
ROCPROF_DURATION_SECONDS="${ROCPROF_DURATION_SECONDS:-90}"

if [[ ! "$ROCPROF_START_SECONDS" =~ ^[0-9]+$ ]] ||
   [[ ! "$ROCPROF_DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROCPROF_START_SECONDS must be non-negative and " \
         "ROCPROF_DURATION_SECONDS must be positive integer seconds" >&2
    exit 2
fi

mkdir -p "$ROCPROF_OUTPUT_DIR"
exec rocprofv3 \
    --kernel-trace \
    --memory-copy-trace \
    --group-by-queue true \
    --collection-period \
        "${ROCPROF_START_SECONDS}:${ROCPROF_DURATION_SECONDS}:1" \
    --output-format csv \
    --output-directory "$ROCPROF_OUTPUT_DIR" \
    --output-file trace \
    -- "$PROFILED_SERVER_BIN" "$@"
