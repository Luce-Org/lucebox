#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s {baseline|router-bf16|router-auto} TARGET_GGUF DSPARK_GGUF OUTPUT_JSON\n' \
        "${0##*/}" >&2
}

if (( $# != 4 )); then
    usage
    exit 2
fi

variant=$1
model_path=$2
draft_path=$3
output_json=$4
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
server_dir=$(cd -- "$script_dir/.." && pwd)
source "$script_dir/ling3_bench_safety.sh"
server_bin=${LING_SERVER_BINARY:-$server_dir/build-ling3-cuda/dflash_server}
port=${LING_BENCH_PORT:-18082}
max_ctx=${LING_BENCH_MAX_CTX:-32768}
chat_template_file=${LING_CHAT_TEMPLATE_FILE:-}
server_pid=
host_log="${output_json}.host.log"
chat_template_args=()

case "$variant" in
    baseline)
        router_bf16=0
        ;;
    router-bf16)
        router_bf16=1
        ;;
    router-auto)
        router_bf16=auto
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ ! -r "$model_path" || ! -r "$draft_path" || ! -x "$server_bin" ]]; then
    printf 'Missing target, DSpark model, or server binary.\n' >&2
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
if pgrep -x dflash_server >/dev/null; then
    printf 'Refusing to benchmark while another dflash_server is running.\n' >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null; then
    printf 'Refusing to benchmark while llama-server is running.\n' >&2
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
server_env=(
    env
    DFLASH_BAILING_MTP=0
    GGML_CUDA_PDL="${LING_GGML_CUDA_PDL:-0}"
)
if [[ "$router_bf16" != auto ]]; then
    server_env+=(DFLASH_BAILING_ROUTER_BF16="$router_bf16")
fi
ling3_start_guarded_server "${server_env[@]}" "$server_bin" "$model_path" \
    --draft "$draft_path" \
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
server_pid=$LING3_GUARDED_SERVER_PID

for _ in $(seq 1 150); do
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

printf 'Variant: %s\nResult:  %s\n' "$variant" "$output_json"
