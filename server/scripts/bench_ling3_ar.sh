#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s {luce|llama} TARGET_GGUF OUTPUT_JSON\n' "${0##*/}" >&2
}

if (( $# != 3 )); then
    usage
    exit 2
fi

engine=$1
model_path=$2
output_json=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
server_dir=$(cd -- "$script_dir/.." && pwd)
source "$script_dir/ling3_bench_safety.sh"
luce_bin=${LING_SERVER_BINARY:-$server_dir/build-ling3-cuda/dflash_server}
llama_bin=${LLAMA_SERVER_BINARY:-llama-server}
port=${LING_BENCH_PORT:-18084}
max_ctx=${LING_BENCH_MAX_CTX:-32768}
chat_template_file=${LING_CHAT_TEMPLATE_FILE:-}
server_pid=
host_log="${output_json}.host.log"
chat_template_args=()

if [[ "$engine" != luce && "$engine" != llama ]]; then
    usage
    exit 2
fi
if [[ ! -r "$model_path" ]]; then
    printf 'Missing target model: %s\n' "$model_path" >&2
    exit 2
fi
ling3_refuse_existing_artifacts "$output_json"
ling3_validate_positive_integer LING_BENCH_MAX_CTX "$max_ctx"
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

cd "$server_dir"
if [[ "$engine" == luce ]]; then
    [[ -x "$luce_bin" ]] || {
        printf 'Missing LuceBox server: %s\n' "$luce_bin" >&2
        exit 2
    }
    engine_name='LuceBox strict AR'
    ling3_start_guarded_server env DFLASH_BAILING_MTP=0 \
        GGML_CUDA_PDL="${LING_GGML_CUDA_PDL:-0}" \
        "$luce_bin" "$model_path" \
            --host 127.0.0.1 \
            --port "$port" \
            --max-ctx "$max_ctx" \
            --default-max-tokens 256 \
            --cache-type-k q4_0 \
            --cache-type-v q4_0 \
            --prefix-cache-slots 0 \
            --disk-prefix-cache off \
            --model-name ling-3.0-flash-lucebox \
            "${chat_template_args[@]}" \
        >"${output_json}.server.log" 2>&1
else
    if [[ "$llama_bin" == */* ]]; then
        [[ -x "$llama_bin" ]] || {
            printf 'Missing llama-server: %s\n' "$llama_bin" >&2
            exit 2
        }
    elif ! command -v "$llama_bin" >/dev/null; then
        printf 'llama-server is not on PATH.\n' >&2
        exit 2
    fi
    engine_name='llama.cpp current strict AR'
    ling3_start_guarded_server "$llama_bin" \
        --model "$model_path" \
        --alias ling-3.0-flash-lucebox \
        --host 127.0.0.1 \
        --port "$port" \
        --ctx-size "$max_ctx" \
        --gpu-layers all \
        --split-mode none \
        --flash-attn on \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        --parallel 1 \
        --jinja \
        "${chat_template_args[@]}" \
        --no-cache-prompt \
        --no-webui \
        >"${output_json}.server.log" 2>&1
fi
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

python3 scripts/bench_ling3_flash.py \
    --url "http://127.0.0.1:${port}" \
    --engine "$engine_name" \
    --server-max-context "$max_ctx" \
    --prompt-profile "${LING_BENCH_PROMPT_PROFILE:-official-chat}" \
    --decode-workload "${LING_BENCH_DECODE_WORKLOAD:-beta}" \
    --sections "${LING_BENCH_SECTIONS:-decode}" \
    --decode-tokens "${LING_BENCH_DECODE_TOKENS:-128}" \
    --decode-runs "${LING_BENCH_DECODE_RUNS:-7}" \
    --warmups "${LING_BENCH_WARMUPS:-3}" \
    --expected-output-sha256 "${LING_EXPECTED_SHA256:-}" \
    --context-tokens "${LING_BENCH_CONTEXT_TOKENS:-256,1024,4096,8192}" \
    --context-runs "${LING_BENCH_CONTEXT_RUNS:-2}" \
    --decode-context-tokens "${LING_BENCH_DECODE_CONTEXT_TOKENS:-256,1024,4096,8192,16384}" \
    --decode-context-runs "${LING_BENCH_DECODE_CONTEXT_RUNS:-3}" \
    --decode-context-warmups "${LING_BENCH_DECODE_CONTEXT_WARMUPS:-1}" \
    --output "$output_json"

printf 'Engine: %s\nResult: %s\n' "$engine_name" "$output_json"
