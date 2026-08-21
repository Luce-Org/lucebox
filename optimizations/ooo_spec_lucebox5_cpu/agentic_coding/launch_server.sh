#!/usr/bin/env bash
# Add agent-turn caching to an otherwise complete dflash_server command.
# Early read-only tool dispatch is an independent, explicit opt-in.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
early_dispatch="${DFLASH_ENABLE_EARLY_DISPATCH:-0}"

if [[ $# -eq 0 ]]; then
  printf 'usage: %s <dflash_server> <normal server arguments...>\n' "$0" >&2
  exit 2
fi
[[ "${early_dispatch}" == "0" || "${early_dispatch}" == "1" ]] || {
  printf 'DFLASH_ENABLE_EARLY_DISPATCH must be 0 or 1\n' >&2
  exit 2
}

for argument in "$@"; do
  case "${argument}" in
    --agent-turn-cache|--end-turn-snapshot)
      printf 'the coding wrapper owns server option %s; remove the duplicate\n' "${argument}" >&2
      exit 2
      ;;
    --early-dispatch|--prefetch-prefill|--tool-spec-executor|--tool-spec-read-only|--tool-spec-cpu-affinity|--tool-spec-timeout-ms|--tool-spec-max-executors)
      if [[ "${early_dispatch}" == "1" ]]; then
        printf 'the coding wrapper owns server option %s when early dispatch is enabled; remove the duplicate\n' "${argument}" >&2
        exit 2
      fi
      ;;
  esac
done

extra_args=(--agent-turn-cache)

if [[ "${early_dispatch}" == "1" ]]; then
  workspace="${DFLASH_TOOL_WORKSPACE:-${script_dir}/fixtures/repo}"
  executor="${DFLASH_CODING_EXECUTOR:-${script_dir}/coding_tool_executor.py}"
  timeout_ms="${DFLASH_TOOL_TIMEOUT_MS:-5000}"
  max_executors="${DFLASH_TOOL_MAX_EXECUTORS:-8}"
  tool_cpus="${DFLASH_TOOL_CPU_AFFINITY:-}"

  [[ -d "${workspace}" ]] || { printf 'workspace is not a directory: %s\n' "${workspace}" >&2; exit 2; }
  [[ -x "${executor}" ]] || { printf 'executor is not executable: %s\n' "${executor}" >&2; exit 2; }
  [[ "${timeout_ms}" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid DFLASH_TOOL_TIMEOUT_MS\n' >&2; exit 2; }
  [[ "${max_executors}" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid DFLASH_TOOL_MAX_EXECUTORS\n' >&2; exit 2; }
  [[ "$(uname -s)" == "Linux" ]] || { printf 'automatic coding-tool dispatch requires Linux CPU isolation\n' >&2; exit 2; }
  [[ -n "${tool_cpus}" ]] || { printf 'DFLASH_TOOL_CPU_AFFINITY is required and must be disjoint from the model process\n' >&2; exit 2; }

  workspace="$(cd -- "${workspace}" && pwd -P)"
  executor_dir="$(cd -- "$(dirname -- "${executor}")" && pwd -P)"
  executor="${executor_dir}/$(basename -- "${executor}")"
  extra_args+=(
    --early-dispatch
    --prefetch-prefill
    --tool-spec-executor "${executor}"
    --tool-spec-read-only read_file
    --tool-spec-read-only search_code
    --tool-spec-read-only list_files
    --tool-spec-cpu-affinity "${tool_cpus}"
    --tool-spec-timeout-ms "${timeout_ms}"
    --tool-spec-max-executors "${max_executors}"
  )
  export DFLASH_TOOL_WORKSPACE="${workspace}"
fi

exec "$@" "${extra_args[@]}"
