#!/usr/bin/env bash
# Wrap the qualified 0731 launcher's dflash_server with speculative tool
# execution for real-API agent loops:
#   EARLY_DISPATCH=1     launch each tool call the moment its block closes
#                        in the token stream (default on)
#   END_TURN_SNAPSHOT=1  cache prompt+output KV across tool turns (default on)
#   PREFETCH_PREFILL=1   prefill the deterministic next tool turn before the
#                        client requests it (default on)
set -euo pipefail
wrapper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_candidate="/home/lucebox5/tool-spec-cpu-20260813/engine-ooo-spec/server/build-hip-dual"
if [[ -d "${wrapper_dir}/candidate-build" ]]; then
  default_candidate="${wrapper_dir}/candidate-build"
fi
CANDIDATE_BUILD="${CANDIDATE_BUILD:-${default_candidate}}"
EARLY_DISPATCH="${EARLY_DISPATCH:-1}"
END_TURN_SNAPSHOT="${END_TURN_SNAPSHOT:-1}"
PREFETCH_PREFILL="${PREFETCH_PREFILL:-1}"
PREFIX_CACHE_SLOTS_OVERRIDE="${PREFIX_CACHE_SLOTS_OVERRIDE:-32}"
for toggle in EARLY_DISPATCH END_TURN_SNAPSHOT PREFETCH_PREFILL; do
  [[ "${!toggle}" == "0" || "${!toggle}" == "1" ]] || {
    printf '%s must be 0 or 1\n' "${toggle}" >&2
    exit 2
  }
done
[[ -x "${CANDIDATE_BUILD}/dflash_server" ]] || {
  printf 'dflash_server is not executable: %s\n' "${CANDIDATE_BUILD}/dflash_server" >&2
  exit 2
}
extra_args=()
[[ "${EARLY_DISPATCH}" == "1" ]] && extra_args+=(--early-dispatch)
[[ "${END_TURN_SNAPSHOT}" == "1" ]] && extra_args+=(--end-turn-snapshot)
[[ "${PREFETCH_PREFILL}" == "1" ]] && extra_args+=(--prefetch-prefill)
if [[ -n "${PREFIX_CACHE_SLOTS_OVERRIDE}" ]]; then
  [[ "${PREFIX_CACHE_SLOTS_OVERRIDE}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    printf 'invalid PREFIX_CACHE_SLOTS_OVERRIDE: %s\n' "${PREFIX_CACHE_SLOTS_OVERRIDE}" >&2
    exit 2
  }
  extra_args+=(--prefix-cache-slots "${PREFIX_CACHE_SLOTS_OVERRIDE}")
fi
export LD_LIBRARY_PATH="${CANDIDATE_BUILD}/deps/llama.cpp/ggml/src:${CANDIDATE_BUILD}/deps/llama.cpp/ggml/src/ggml-hip:${LD_LIBRARY_PATH:-}"
export LUCE_MMVQ_MAX_NCOLS=5
exec "${CANDIDATE_BUILD}/dflash_server" "$@" "${extra_args[@]}"
