#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s {luce-ar|luce-dspark|llama-ar|llama-dspark} TARGET_GGUF DSPARK_GGUF OUTPUT_JSON\n' \
        "${0##*/}" >&2
}

if (( $# != 4 )); then
    usage
    exit 2
fi

mode=$1
model_path=$2
draft_path=$3
output_json=$4
case "$mode" in
    luce-ar|luce-dspark|llama-ar|llama-dspark) ;;
    *) usage; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
server_dir=$(cd -- "$script_dir/.." && pwd)
repo_dir=$(cd -- "$server_dir/.." && pwd)
source "$script_dir/ling3_bench_safety.sh"

luce_bin=${LING_SERVER_BINARY:-$server_dir/build-ling3-cuda/dflash_server}
llama_bin=${LLAMA_SERVER_BINARY:-llama-server}
harness_client=${LING_HELDOUT_CLIENT:-$repo_dir/harness/client_test_runner.py}
prompts_dir=${LING_HELDOUT_PROMPTS_DIR:-$repo_dir/harness/benchmarks/prompts}
port=${LING_BENCH_PORT:-18085}
max_ctx=${LING_BENCH_MAX_CTX:-8192}
suites=${LING_BENCH_SUITES:-he,gsm,math}
n_sample=${LING_BENCH_N_SAMPLE:-3}
chat_template_file=${LING_CHAT_TEMPLATE_FILE:-}
host_log="${output_json}.host.log"
server_log="${output_json}.server.log"
server_pid=
chat_template_args=()

[[ -r "$model_path" && -r "$draft_path" ]] || {
    printf 'Missing target or DSpark model.\n' >&2
    exit 2
}
[[ -r "$harness_client" && -d "$prompts_dir" ]] || {
    printf 'Missing held-out client or prompts.\n' >&2
    exit 2
}
ling3_refuse_existing_artifacts "$output_json"
ling3_validate_positive_integer LING_BENCH_MAX_CTX "$max_ctx"
ling3_validate_positive_integer LING_BENCH_N_SAMPLE "$n_sample"
if [[ -n "$chat_template_file" ]]; then
    [[ -r "$chat_template_file" ]] || {
        printf 'Missing chat template: %s\n' "$chat_template_file" >&2
        exit 2
    }
    chat_template_args=(--chat-template-file "$chat_template_file")
fi
if pgrep -x dflash_server >/dev/null || pgrep -x llama-server >/dev/null; then
    printf 'Refusing to benchmark while an inference server is running.\n' >&2
    exit 2
fi

cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ -e "$host_log" ]]; then
        ling3_host_snapshot "$host_log" cleanup
    fi
}
trap cleanup EXIT

ling3_safe_host_preflight "$output_json"

common_luce_args=(
    --host 127.0.0.1
    --port "$port"
    --max-ctx "$max_ctx"
    --default-max-tokens 2048
    --cache-type-k q4_0
    --cache-type-v q4_0
    --prefix-cache-slots 0
    --disk-prefix-cache off
    --model-name ling-3.0-flash-lucebox
    "${chat_template_args[@]}"
)
common_llama_args=(
    --model "$model_path"
    --alias ling-3.0-flash-lucebox
    --host 127.0.0.1
    --port "$port"
    --ctx-size "$max_ctx"
    --gpu-layers all
    --split-mode none
    --flash-attn on
    --cache-type-k q4_0
    --cache-type-v q4_0
    --parallel 1
    --jinja
    "${chat_template_args[@]}"
    --no-cache-prompt
    --no-webui
)

case "$mode" in
    luce-ar)
        [[ -x "$luce_bin" ]] || { printf 'Missing LuceBox server.\n' >&2; exit 2; }
        ling3_start_guarded_server env \
            DFLASH_BAILING_MTP=0 \
            GGML_CUDA_PDL="${LING_GGML_CUDA_PDL:-0}" \
            "$luce_bin" "$model_path" "${common_luce_args[@]}" \
            >"$server_log" 2>&1
        ;;
    luce-dspark)
        [[ -x "$luce_bin" ]] || { printf 'Missing LuceBox server.\n' >&2; exit 2; }
        ling3_start_guarded_server env \
            DFLASH_BAILING_MTP=0 \
            DFLASH_BAILING_ROUTER_BF16=1 \
            DFLASH_DSPARK_VERIFY_WIDTH="${LING_DSPARK_VERIFY_WIDTH:-5}" \
            DFLASH_CUDA_MMVQ_GROUPED_TPG="${LING_GROUPED_TPG:-4}" \
            GGML_CUDA_PDL="${LING_GGML_CUDA_PDL:-0}" \
            "$luce_bin" "$model_path" --draft "$draft_path" \
            "${common_luce_args[@]}" >"$server_log" 2>&1
        ;;
    llama-ar|llama-dspark)
        if [[ "$llama_bin" == */* ]]; then
            [[ -x "$llama_bin" ]] || { printf 'Missing llama-server.\n' >&2; exit 2; }
        elif ! command -v "$llama_bin" >/dev/null; then
            printf 'llama-server is not on PATH.\n' >&2
            exit 2
        fi
        if [[ "$mode" == llama-dspark ]]; then
            common_llama_args+=(
                --spec-draft-model "$draft_path"
                --spec-type draft-dspark
                --spec-draft-n-max "${LING_LLAMA_DRAFT_N_MAX:-8}"
                --spec-draft-ngl all
            )
        fi
        ling3_start_guarded_server "$llama_bin" "${common_llama_args[@]}" \
            >"$server_log" 2>&1
        ;;
esac
server_pid=$LING3_GUARDED_SERVER_PID

for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"
        exit 1
    fi
    sleep 2
done
curl -fsS "http://127.0.0.1:${port}/health" >/dev/null

python3 "$harness_client" bench \
    --url "http://127.0.0.1:${port}" \
    --model ling-3.0-flash-lucebox \
    --suite "$suites" \
    --n-sample "$n_sample" \
    --prompts-dir "$prompts_dir" \
    --json-out "$output_json"

printf 'Mode:   %s\nResult: %s\n' "$mode" "$output_json"
