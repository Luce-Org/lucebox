#!/usr/bin/env bash
set -euo pipefail

# Preserve every argument from the qualified 0731 launcher and add the native
# Qwen3 tool-prediction lane on the Strix GPU. The qualified launcher clears
# ambient variables, so an adjacent `candidate-build` symlink is the durable
# deployment override; direct launches may still use CANDIDATE_BUILD.
wrapper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_candidate="/home/lucebox5/tool-spec-cpu-20260813/engine-ooo-spec/server/build-hip-dual"
if [[ -d "${wrapper_dir}/candidate-build" ]]; then
  default_candidate="${wrapper_dir}/candidate-build"
fi
CANDIDATE_BUILD="${CANDIDATE_BUILD:-${default_candidate}}"
PREDICTOR_MODEL="${PREDICTOR_MODEL:-/home/lucebox5/tool-spec-cpu-20260813/models/Qwen3-0.6B-Q8_0.gguf}"
PREDICTOR_IPC_BIN="${PREDICTOR_IPC_BIN:-${CANDIDATE_BUILD}/backend_ipc_daemon}"
PREDICTOR_GPU="${PREDICTOR_GPU:-1}"
PREDICTOR_MAX_CTX="${PREDICTOR_MAX_CTX:-4096}"
PREDICTOR_MAX_TOKENS="${PREDICTOR_MAX_TOKENS:-256}"
PREDICTOR_TIMEOUT_MS="${PREDICTOR_TIMEOUT_MS:-2000}"
PREDICTOR_CONFIDENCE="${PREDICTOR_CONFIDENCE:-0.75}"
# The qualified 0731 launcher disables caches for cold throughput benchmarks.
# Tool-using agent loops need turn-boundary reuse; this later CLI flag wins
# without modifying the qualified model/DSpark arguments.
PREFIX_CACHE_SLOTS_OVERRIDE="${PREFIX_CACHE_SLOTS_OVERRIDE:-32}"

cache_args=()
if [[ -n "${PREFIX_CACHE_SLOTS_OVERRIDE}" ]]; then
  [[ "${PREFIX_CACHE_SLOTS_OVERRIDE}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    printf 'invalid PREFIX_CACHE_SLOTS_OVERRIDE: %s\n' \
      "${PREFIX_CACHE_SLOTS_OVERRIDE}" >&2
    exit 2
  }
  (( PREFIX_CACHE_SLOTS_OVERRIDE <= 64 )) || {
    printf 'PREFIX_CACHE_SLOTS_OVERRIDE exceeds the 64-slot engine limit\n' >&2
    exit 2
  }
  cache_args+=(--prefix-cache-slots "${PREFIX_CACHE_SLOTS_OVERRIDE}")
fi

for binary in "${CANDIDATE_BUILD}/dflash_server" "${PREDICTOR_IPC_BIN}"; do
  [[ -f "${binary}" && -x "${binary}" ]] || {
    printf 'Qwen tool-predictor binary is not executable: %s\n' "${binary}" >&2
    exit 2
  }
done
[[ -f "${PREDICTOR_MODEL}" ]] || {
  printf 'Qwen tool-predictor model is not a regular file: %s\n' \
    "${PREDICTOR_MODEL}" >&2
  exit 2
}

export LD_LIBRARY_PATH="${CANDIDATE_BUILD}/deps/llama.cpp/ggml/src:${CANDIDATE_BUILD}/deps/llama.cpp/ggml/src/ggml-hip:${LD_LIBRARY_PATH:-}"
export LUCE_MMVQ_MAX_NCOLS=5

exec "${CANDIDATE_BUILD}/dflash_server" "$@" \
  --tool-hint-native-model "${PREDICTOR_MODEL}" \
  --tool-hint-native-ipc-bin "${PREDICTOR_IPC_BIN}" \
  --tool-hint-native-gpu "${PREDICTOR_GPU}" \
  --tool-hint-native-max-ctx "${PREDICTOR_MAX_CTX}" \
  --tool-hint-max-tokens "${PREDICTOR_MAX_TOKENS}" \
  --tool-hint-timeout-ms "${PREDICTOR_TIMEOUT_MS}" \
  --tool-hint-execution-confidence "${PREDICTOR_CONFIDENCE}" \
  "${cache_args[@]}"
