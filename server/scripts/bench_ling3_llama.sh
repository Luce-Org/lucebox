#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s TARGET_GGUF DSPARK_GGUF OUTPUT_JSON\n' "${0##*/}" >&2
}

if (( $# != 3 )); then
    usage
    exit 2
fi

model_path=$1
draft_path=$2
output_json=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
server_dir=$(cd -- "$script_dir/.." && pwd)
source "$script_dir/ling3_bench_safety.sh"
llama_bin=${LLAMA_SERVER_BINARY:-llama-server}
port=${LLAMA_BENCH_PORT:-18083}
max_ctx=${LING_BENCH_MAX_CTX:-32768}
chat_template_file=${LING_CHAT_TEMPLATE_FILE:-}
server_pid=
host_log="${output_json}.host.log"
chat_template_args=()

if [[ ! -r "$model_path" || ! -r "$draft_path" ]]; then
    printf 'Missing target or DSpark model.\n' >&2
    exit 2
fi
if [[ "$llama_bin" == */* ]]; then
    [[ -x "$llama_bin" ]] || {
        printf 'Missing llama-server: %s\n' "$llama_bin" >&2
        exit 2
    }
elif ! command -v "$llama_bin" >/dev/null; then
    printf 'llama-server is not on PATH.\n' >&2
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
if pgrep -x llama-server >/dev/null; then
    printf 'Refusing to benchmark while another llama-server is running.\n' >&2
    exit 2
fi
if pgrep -x dflash_server >/dev/null; then
    printf 'Refusing to benchmark while dflash_server is running.\n' >&2
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

ling3_start_guarded_server "$llama_bin" \
    --model "$model_path" \
    --spec-draft-model "$draft_path" \
    --spec-type draft-dspark \
    --spec-draft-n-max 8 \
    --spec-draft-ngl all \
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

cd "$server_dir"
python3 scripts/bench_ling3_flash.py \
    --url "http://127.0.0.1:${port}" \
    --engine "llama.cpp current + DSpark" \
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
    --output "$output_json"

printf 'Engine: llama.cpp + DSpark\nResult: %s\n' "$output_json"
