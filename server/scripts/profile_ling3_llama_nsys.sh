#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s TARGET_GGUF DSPARK_GGUF OUTPUT_BASE\n' "${0##*/}" >&2
}

if (( $# != 3 )); then
    usage
    exit 2
fi

model_path=$1
draft_path=$2
output_base=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
server_dir=$(cd -- "$script_dir/.." && pwd)
llama_bin=${LLAMA_SERVER_BINARY:-llama-server}
nsys_bin=${NSYS_BIN:-/usr/local/bin/nsys}
port=${LLAMA_PROFILE_PORT:-18084}
context_tokens=${LING_PROFILE_CONTEXT_TOKENS:-16384}
profile_sections=${LING_PROFILE_SECTIONS:-context}
profile_mode=${LING_PROFILE_MODE:-dspark}
max_ctx=${LING_PROFILE_MAX_CTX:-4096}
chat_template_file=${LING_CHAT_TEMPLATE_FILE:-}
session="llamaar_${$}"
launch_pid=
server_pid=
session_started=0
draft_args=()
chat_template_args=(--jinja)
expected_hash_args=()

if [[ ! -r "$model_path" || ! -x "$llama_bin" || ! -x "$nsys_bin" ]]; then
    printf 'Missing target model, llama-server, or Nsight Systems binary.\n' >&2
    exit 2
fi
case "$profile_mode" in
    ar) ;;
    dspark)
        [[ -r "$draft_path" ]] || {
            printf 'Missing DSpark model.\n' >&2
            exit 2
        }
        draft_args=(
            --spec-draft-model "$draft_path"
            --spec-type draft-dspark
            --spec-draft-n-max 8
            --spec-draft-ngl all
        )
        ;;
    *)
        printf 'LING_PROFILE_MODE must be ar or dspark.\n' >&2
        exit 2
        ;;
esac
if [[ -n "$chat_template_file" ]]; then
    [[ -r "$chat_template_file" ]] || {
        printf 'Missing chat template: %s\n' "$chat_template_file" >&2
        exit 2
    }
    chat_template_args+=(--chat-template-file "$chat_template_file")
fi
if [[ -n "${LING_EXPECTED_SHA256:-}" ]]; then
    expected_hash_args=(--expected-output-sha256 "$LING_EXPECTED_SHA256")
fi
if [[ -e "${output_base}.nsys-rep" ]]; then
    printf 'Refusing to overwrite %s.nsys-rep\n' "$output_base" >&2
    exit 2
fi
if pgrep -x llama-server >/dev/null; then
    printf 'Refusing to profile while another llama-server is running.\n' >&2
    exit 2
fi
if pgrep -x dflash_server >/dev/null; then
    printf 'Refusing to profile while dflash_server is running.\n' >&2
    exit 2
fi

cleanup() {
    if (( session_started )); then
        "$nsys_bin" stop --session="$session" >/dev/null 2>&1 || true
        session_started=0
    fi
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
    fi
    if [[ -n "$launch_pid" ]]; then
        wait "$launch_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

"$nsys_bin" launch \
    --session-new="$session" \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    "$llama_bin" \
        --model "$model_path" \
        "${draft_args[@]}" \
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
        "${chat_template_args[@]}" \
        --no-cache-prompt \
        --no-webui \
    >"${output_base}.server.log" 2>&1 &
launch_pid=$!

for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$launch_pid" 2>/dev/null; then
        wait "$launch_pid"
        exit 1
    fi
    sleep 2
done
curl -fsS "http://127.0.0.1:${port}/health" >/dev/null
server_pid=$(pgrep -n -x llama-server)

cd "$server_dir"
case "$profile_sections" in
    decode)
        bench_args=(
            --sections decode
            --decode-tokens "${LING_PROFILE_DECODE_TOKENS:-128}"
            --decode-runs 1
            --warmups 0
        )
        ;;
    context)
        bench_args=(
            --sections context
            --context-tokens "$context_tokens"
            --context-runs 1
            --warmups 0
        )
        ;;
    decode-context)
        bench_args=(
            --sections decode-context
            --decode-context-tokens "$context_tokens"
            --decode-context-runs 1
            --decode-context-warmups 0
            --decode-tokens "${LING_PROFILE_DECODE_TOKENS:-128}"
            --warmups 0
        )
        ;;
    *)
        printf 'LING_PROFILE_SECTIONS must be decode, context, or decode-context.\n' >&2
        exit 2
        ;;
esac

if [[ "$profile_sections" == decode ]]; then
    python3 scripts/bench_ling3_flash.py \
        --url "http://127.0.0.1:${port}" \
        --engine "llama.cpp llama-server" \
        --sections decode \
        --decode-tokens "${LING_PROFILE_DECODE_TOKENS:-128}" \
        --decode-runs 3 \
        --warmups 2 \
        --server-max-context "$max_ctx" \
        --prompt-profile "${LING_BENCH_PROMPT_PROFILE:-official-chat}" \
        --decode-workload "${LING_BENCH_DECODE_WORKLOAD:-beta}" \
        "${expected_hash_args[@]}" \
        --output "${output_base}.warmup.json" \
        >/dev/null
elif [[ "$profile_sections" == context ]]; then
    python3 scripts/bench_ling3_flash.py \
        --url "http://127.0.0.1:${port}" \
        --engine "llama.cpp llama-server" \
        "${bench_args[@]}" \
        --output "${output_base}.warmup.json" \
        >/dev/null
else
    python3 scripts/bench_ling3_flash.py \
        --url "http://127.0.0.1:${port}" \
        --engine "llama.cpp llama-server" \
        --sections decode-context \
        --decode-context-tokens "$context_tokens" \
        --decode-context-runs 1 \
        --decode-context-warmups 1 \
        --decode-tokens "${LING_PROFILE_DECODE_TOKENS:-128}" \
        --output "${output_base}.warmup.json" \
        >/dev/null
fi

"$nsys_bin" start --session="$session" --output="$output_base"
session_started=1
python3 scripts/bench_ling3_flash.py \
    --url "http://127.0.0.1:${port}" \
    --engine "llama.cpp llama-server" \
    "${bench_args[@]}" \
    --server-max-context "$max_ctx" \
    --prompt-profile "${LING_BENCH_PROMPT_PROFILE:-official-chat}" \
    --decode-workload "${LING_BENCH_DECODE_WORKLOAD:-beta}" \
    "${expected_hash_args[@]}" \
    --output "${output_base}.json" \
    >/dev/null
"$nsys_bin" stop --session="$session"
session_started=0

kill "$server_pid"
wait "$launch_pid" 2>/dev/null || true
server_pid=
launch_pid=

printf 'Profile: %s.nsys-rep\nResult:  %s.json\n' \
    "$output_base" "$output_base"
