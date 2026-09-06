#!/usr/bin/env bash
set -euo pipefail

# Lucebox Gemma 4 / RTX 4090 backend launcher.
#
# This path intentionally uses the Gemma 4 MTP-enabled llama.cpp checkout on
# this workstation. The native Lucebox DFlash binary is hand-shaped for
# Qwen/Laguna graphs; Gemma 4 support needs libllama's Gemma 4 + MTP runtime.

MODEL="${LUCEBOX_GEMMA4_MODEL:-/mnt/c/Users/adyba/Downloads/gemma-4-31B-it-abliterated-Q4_K_M.gguf}"
MTP_MODEL="${LUCEBOX_GEMMA4_MTP_MODEL:-/home/tdamre/models/AtomicChat-gemma-4-31B-it-assistant-GGUF/gemma-4-31B-it-assistant.Q4_K_S.gguf}"
LLAMA_SERVER="${LUCEBOX_LLAMA_SERVER:-/home/tdamre/src/atomic-llama-cpp-turboquant/build-cuda124/bin/llama-server}"
MTP_STYLE="${LUCEBOX_GEMMA4_MTP_STYLE:-atomic}"

HOST="${LUCEBOX_GEMMA4_HOST:-127.0.0.1}"
PORT="${LUCEBOX_GEMMA4_PORT:-18191}"
CTX_SIZE="${LUCEBOX_GEMMA4_CTX_SIZE:-70080}"
DRAFT_CTX_SIZE="${LUCEBOX_GEMMA4_DRAFT_CTX_SIZE:-2048}"
DRAFT_N_MAX="${LUCEBOX_GEMMA4_DRAFT_N_MAX:-4}"
DRAFT_BLOCK_SIZE="${LUCEBOX_GEMMA4_DRAFT_BLOCK_SIZE:-4}"
BATCH_SIZE="${LUCEBOX_GEMMA4_BATCH_SIZE:-2048}"
UBATCH_SIZE="${LUCEBOX_GEMMA4_UBATCH_SIZE:-512}"
CACHE_TYPE_K="${LUCEBOX_GEMMA4_CACHE_TYPE_K:-turbo4}"
CACHE_TYPE_V="${LUCEBOX_GEMMA4_CACHE_TYPE_V:-turbo4}"
DRAFT_CACHE_TYPE_K="${LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_K:-$CACHE_TYPE_K}"
DRAFT_CACHE_TYPE_V="${LUCEBOX_GEMMA4_DRAFT_CACHE_TYPE_V:-$CACHE_TYPE_V}"
GPU_LAYERS_DRAFT="${LUCEBOX_GEMMA4_GPU_LAYERS_DRAFT:-all}"
DRAFT_OVERRIDE_TENSOR="${LUCEBOX_GEMMA4_DRAFT_OVERRIDE_TENSOR:-token_embd.weight=CUDA0}"
CACHE_RAM="${LUCEBOX_GEMMA4_CACHE_RAM:-0}"
NO_KV_OFFLOAD="${LUCEBOX_GEMMA4_NO_KV_OFFLOAD:-0}"
NO_OP_OFFLOAD="${LUCEBOX_GEMMA4_NO_OP_OFFLOAD:-0}"
NO_MMAP="${LUCEBOX_GEMMA4_NO_MMAP:-0}"
MLOCK="${LUCEBOX_GEMMA4_MLOCK:-0}"
POLL="${LUCEBOX_GEMMA4_POLL:-100}"
POLL_BATCH="${LUCEBOX_GEMMA4_POLL_BATCH:-1}"
PRIORITY="${LUCEBOX_GEMMA4_PRIORITY:-2}"
PRIORITY_BATCH="${LUCEBOX_GEMMA4_PRIORITY_BATCH:-2}"
THREADS_HTTP="${LUCEBOX_GEMMA4_THREADS_HTTP:-1}"
THREADS="${LUCEBOX_GEMMA4_THREADS:-}"
THREADS_BATCH="${LUCEBOX_GEMMA4_THREADS_BATCH:-$THREADS}"
RUN_DIR="${LUCEBOX_GEMMA4_RUN_DIR:-$HOME/lucebox-runs}"
PID_FILE="${LUCEBOX_GEMMA4_PID_FILE:-$RUN_DIR/lucebox-gemma4-mtp-server.pid}"
LOG_FILE="${LUCEBOX_GEMMA4_LOG_FILE:-$RUN_DIR/lucebox-gemma4-mtp-server-$(date +%Y%m%d-%H%M%S).log}"

url() {
    printf 'http://%s:%s' "$HOST" "$PORT"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

validate_paths() {
    [[ -x "$LLAMA_SERVER" ]] || die "llama-server not executable: $LLAMA_SERVER"
    [[ -f "$MODEL" ]] || die "target GGUF missing: $MODEL"
    [[ -f "$MTP_MODEL" ]] || die "Gemma 4 MTP assistant missing: $MTP_MODEL"
}

read_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(tr -dc '0-9' < "$PID_FILE")"
    [[ -n "$pid" ]] || return 1
    printf '%s\n' "$pid"
}

is_our_process() {
    local pid="$1"
    local args
    args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$args" == *"llama-server"* && "$args" == *"--host $HOST"* && "$args" == *"--port $PORT"* && "$args" == *"--spec-type mtp"* ]]
}

is_running() {
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    [[ -n "$pid" ]] && is_our_process "$pid"
}

health() {
    curl -fsS "$(url)/health"
}

wait_ready() {
    local timeout_s="${1:-300}"
    local start
    start="$(date +%s)"
    while true; do
        if health >/tmp/lucebox-gemma4-health.json 2>/tmp/lucebox-gemma4-health.err; then
            printf 'ready: %s\n' "$(url)"
            cat /tmp/lucebox-gemma4-health.json
            printf '\n'
            return 0
        fi
        if (( $(date +%s) - start >= timeout_s )); then
            printf 'timed out waiting for %s\n' "$(url)" >&2
            [[ -f "$LOG_FILE" ]] && tail -160 "$LOG_FILE" >&2
            return 1
        fi
        sleep 1
    done
}

run_foreground() {
    validate_paths
    mkdir -p "$RUN_DIR"
    printf '%s\n' "$$" > "$PID_FILE"
    printf 'log=%s\n' "$LOG_FILE"
    printf 'url=%s\n' "$(url)"
    local args=(
        -m "$MODEL" \
        -ngl 999 \
        -c "$CTX_SIZE" \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        --flash-attn on \
        --cache-type-k "$CACHE_TYPE_K" \
        --cache-type-v "$CACHE_TYPE_V" \
        -np 1 \
        --host "$HOST" \
        --port "$PORT" \
        --jinja \
        --reasoning off \
        --metrics \
        --poll "$POLL" \
        --poll-batch "$POLL_BATCH" \
        --prio "$PRIORITY" \
        --prio-batch "$PRIORITY_BATCH" \
        --threads-http "$THREADS_HTTP"
        -ngld "$GPU_LAYERS_DRAFT"
    )
    if [[ -n "$DRAFT_OVERRIDE_TENSOR" && "$DRAFT_OVERRIDE_TENSOR" != "none" && "$DRAFT_OVERRIDE_TENSOR" != "0" && "$DRAFT_OVERRIDE_TENSOR" != "false" ]]; then
        args+=(--override-tensor-draft "$DRAFT_OVERRIDE_TENSOR")
    fi
    if [[ -n "$THREADS" ]]; then
        args+=(--threads "$THREADS")
    fi
    if [[ -n "$THREADS_BATCH" ]]; then
        args+=(--threads-batch "$THREADS_BATCH")
    fi
    case "$MTP_STYLE" in
        atomic)
            args+=(--spec-type mtp --mtp-head "$MTP_MODEL" --draft-block-size "$DRAFT_BLOCK_SIZE")
            ;;
        llama-cpp|llama_cpp|spec-draft)
            args+=(
                --spec-type mtp
                --spec-draft-model "$MTP_MODEL"
                --spec-draft-n-max "$DRAFT_N_MAX"
                --spec-draft-ngl all
                --spec-draft-ctx-size "$DRAFT_CTX_SIZE"
                --spec-draft-type-k "$DRAFT_CACHE_TYPE_K"
                --spec-draft-type-v "$DRAFT_CACHE_TYPE_V"
            )
            ;;
        *)
            die "unsupported LUCEBOX_GEMMA4_MTP_STYLE: $MTP_STYLE"
            ;;
    esac
    if [[ -n "$CACHE_RAM" ]]; then
        args+=(--cache-ram "$CACHE_RAM")
    fi
    if [[ "$NO_KV_OFFLOAD" == "1" || "$NO_KV_OFFLOAD" == "true" || "$NO_KV_OFFLOAD" == "yes" ]]; then
        args+=(--no-kv-offload)
    fi
    if [[ "$NO_OP_OFFLOAD" == "1" || "$NO_OP_OFFLOAD" == "true" || "$NO_OP_OFFLOAD" == "yes" ]]; then
        args+=(--no-op-offload)
    fi
    if [[ "$NO_MMAP" == "1" || "$NO_MMAP" == "true" || "$NO_MMAP" == "yes" ]]; then
        args+=(--no-mmap)
    fi
    if [[ "$MLOCK" == "1" || "$MLOCK" == "true" || "$MLOCK" == "yes" ]]; then
        args+=(--mlock)
    fi
    exec "$LLAMA_SERVER" "${args[@]}" > "$LOG_FILE" 2>&1
}

start_background() {
    validate_paths
    mkdir -p "$RUN_DIR"
    if is_running; then
        printf 'already running: pid=%s url=%s\n' "$(read_pid)" "$(url)"
        return 0
    fi
    LUCEBOX_GEMMA4_LOG_FILE="$LOG_FILE" nohup "$0" run > "$LOG_FILE" 2>&1 &
    printf '%s\n' "$!" > "$PID_FILE"
    printf 'pid=%s\nlog=%s\nurl=%s\n' "$!" "$LOG_FILE" "$(url)"
    wait_ready 300
}

stop_server() {
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    [[ -n "$pid" ]] || {
        printf 'not running: no pid file\n'
        return 0
    }
    if ! is_our_process "$pid"; then
        printf 'not stopping pid=%s because it is not this Gemma 4 server\n' "$pid" >&2
        ps -p "$pid" -o pid,ppid,comm,args 2>/dev/null || true
        return 1
    fi
    kill "$pid"
    for _ in $(seq 1 30); do
        if ! ps -p "$pid" >/dev/null 2>&1; then
            rm -f "$PID_FILE"
            printf 'stopped pid=%s\n' "$pid"
            return 0
        fi
        sleep 1
    done
    kill -9 "$pid"
    rm -f "$PID_FILE"
    printf 'force-stopped pid=%s\n' "$pid"
}

status_server() {
    if is_running; then
        local pid
        pid="$(read_pid)"
        printf 'running pid=%s url=%s\n' "$pid" "$(url)"
        ps -p "$pid" -o pid,ppid,etimes,comm,args
        health || true
        printf '\n'
    else
        printf 'not running\n'
        return 1
    fi
}

case "${1:-status}" in
    run)
        run_foreground
        ;;
    start)
        start_background
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server || true
        start_background
        ;;
    status)
        status_server
        ;;
    wait)
        wait_ready "${2:-300}"
        ;;
    *)
        die "usage: $0 {run|start|stop|restart|status|wait [seconds]}"
        ;;
esac
