#!/usr/bin/env bash
# DS4V-1 baseline launch and proof script for DeepSeek V4 Flash serving on
# Strix Halo plus RX 7900 XT. Flags mirror the qualified PR 604 profile in
# server/scripts/serve_ds4_dual_rocm_128k.sh and docs/ds4v-baseline.md.
#
# This script never kills anything by process name. It only removes the
# container id or PID it recorded itself under artifacts/ds4v-1/$RUN_ID.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

PORT=${PORT:-8216}
HOST=${HOST:-127.0.0.1}
MODELS_DIR=${MODELS_DIR:-"$REPO_ROOT/models"}
TARGET=${TARGET:-"$MODELS_DIR/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf"}
DRAFT=${DRAFT:-"$MODELS_DIR/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf"}
RUN_ID=${LUCEBOX_VERIFY_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
EVIDENCE_DIR=${LUCEBOX_EVIDENCE_DIR:-"$REPO_ROOT/artifacts/ds4v-1/$RUN_ID"}
BUILD_DIR=${DFLASH_BUILD_DIR:-"$REPO_ROOT/server/build-hip-dual"}
BINARY=${DFLASH_SERVER_BIN:-"$BUILD_DIR/dflash_server"}
DS4_MAIN_DEVICE=${DS4_MAIN_DEVICE:-hip:0}
DS4_PEER_DEVICE=${DS4_PEER_DEVICE:-1}
DS4_DRAFT_DEVICE=${DS4_DRAFT_DEVICE:-1}
DS4_CONTEXT=${DS4_CONTEXT:-135168}
DS4_EXPERT_BUDGET_MB=${DS4_EXPERT_BUDGET_MB:-10200}
MODEL_PROMPT=${LUCEBOX_SMOKE_PROMPT:-"Reply with exactly one word. Ready."}
MAX_TOKENS=${LUCEBOX_SMOKE_MAX_TOKENS:-32}

PID_FILE="$EVIDENCE_DIR/server.pid"
LOG_FILE="$EVIDENCE_DIR/server.log"
CONTAINER_FILE="$EVIDENCE_DIR/container.id"

usage() {
    cat <<'EOF'
DS4V-1 baseline script. Launch and proof steps for the DeepSeek V4 Flash
asymmetric baseline on Strix Halo plus RX 7900 XT.

usage: scripts/ds4v-baseline.sh <subcommand>

subcommands:
  launch      Start dflash_server with the qualified PR 604 flags.
  doctor      Curl /props.build and pretty print the JSON with jq.
  chat-smoke  POST /v1/chat/completions. Require HTTP 200 and non empty content.
  spec-flag   Report whether speculative decode ran (usage.spec_decode_ran).
  cleanup     Remove only the container or PID this script recorded. Never
              kills by process name. Evidence stays under artifacts/ds4v-1/RUN_ID.

environment knobs:
  LUCEBOX_VERIFY_RUN_ID   Evidence run id. Default is a UTC timestamp.
  PORT                    Server port. Default 8216.
  MODELS_DIR              Directory holding the GGUF files. Default ./models.
  TARGET                  Target GGUF path. Default $MODELS_DIR/DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf
  DRAFT                   Draft GGUF path. Default $MODELS_DIR/DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf

With no subcommand this usage text is printed and the exit code is 0.
EOF
}

require_files() {
    local path
    for path in "$TARGET" "$DRAFT"; do
        if [[ ! -f $path ]]; then
            echo "missing required file: $path" >&2
            exit 1
        fi
    done
    if [[ ! -x $BINARY ]]; then
        echo "missing executable: $BINARY" >&2
        echo "build it with the flags in docs/ds4v-baseline.md" >&2
        exit 1
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

record_model_shas() {
    {
        echo "target $TARGET $(sha256_of "$TARGET")"
        echo "draft $DRAFT $(sha256_of "$DRAFT")"
    } > "$EVIDENCE_DIR/model-sha256.txt"
}

cmd_launch() {
    require_files
    mkdir -p "$EVIDENCE_DIR"
    record_model_shas

    export DFLASH_DS4_MOE_TP=1
    export DFLASH_DS4_MOE_TP_INPROC=1
    export DFLASH_DS4_MOE_TP_GPU=$DS4_PEER_DEVICE
    export DFLASH_EXPERT_BUDGET_MB=$DS4_EXPERT_BUDGET_MB
    export DFLASH_DS4_TP_MAIN_TO_PEER_RATE=${DFLASH_DS4_TP_MAIN_TO_PEER_RATE:-100}
    export DFLASH_DS4_TP_BALANCE_MIN_HOT=${DFLASH_DS4_TP_BALANCE_MIN_HOT:-31}
    export DFLASH_DS4_TP_BALANCE_MAX_HOT=${DFLASH_DS4_TP_BALANCE_MAX_HOT:-50}

    # A calibrated routing profile enables the critical-path placement policy.
    # Without one the server stays usable but placement is uncalibrated.
    if [[ -n ${DFLASH_DS4_HOTNESS_CSV:-} ]]; then
        if [[ ! -f $DFLASH_DS4_HOTNESS_CSV ]]; then
            echo "missing routing profile: $DFLASH_DS4_HOTNESS_CSV" >&2
            exit 1
        fi
        export DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT=${DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT:-1}
    else
        echo "warning: DFLASH_DS4_HOTNESS_CSV is unset; using uncalibrated placement" >&2
    fi

    export DFLASH_DS4_LONG_CONTEXT_CHUNK=${DFLASH_DS4_LONG_CONTEXT_CHUNK:-2048}
    export DFLASH_DS4_DISABLE_LONG_CONTEXT_ARENA_HANDOFF=${DFLASH_DS4_DISABLE_LONG_CONTEXT_ARENA_HANDOFF:-1}
    export DFLASH_CUDA_MMQ_FP2_AFFINE_PREFILL_ONLY=${DFLASH_CUDA_MMQ_FP2_AFFINE_PREFILL_ONLY:-1}
    export DFLASH_CUDA_MMQ_FP2_AFFINE_CAPTURE=${DFLASH_CUDA_MMQ_FP2_AFFINE_CAPTURE:-1}
    export DFLASH_MOE_PREFILL_MASKED_COLD=${DFLASH_MOE_PREFILL_MASKED_COLD:-0}
    export DFLASH_DS4_HYBRID_PREFILL_GPU_HC=${DFLASH_DS4_HYBRID_PREFILL_GPU_HC:-1}
    export DFLASH_DS4_HYBRID_PREFILL_EAGER=${DFLASH_DS4_HYBRID_PREFILL_EAGER:-1}
    export DFLASH_MOE_EXPERT_MAJOR_PINNED_OUTPUT=${DFLASH_MOE_EXPERT_MAJOR_PINNED_OUTPUT:-1}
    export LUCE_MMVQ_MAX_NCOLS=${LUCE_MMVQ_MAX_NCOLS:-4}
    export DFLASH_HIP_NO_AUTO_UMA=${DFLASH_HIP_NO_AUTO_UMA:-1}
    export DFLASH_DS4_TP_GROUPED_MMVQ=${DFLASH_DS4_TP_GROUPED_MMVQ:-1}
    export DFLASH_MMID_GROUPED=${DFLASH_MMID_GROUPED:-1}
    export DFLASH_MMID_GROUPED_TYPES=${DFLASH_MMID_GROUPED_TYPES:-15}
    export DFLASH_CUDA_MMVQ_MOE_FP2_PACKED32=${DFLASH_CUDA_MMVQ_MOE_FP2_PACKED32:-1}
    export DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24=${DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24:-1}
    export DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24_DECODE_ONLY=${DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24_DECODE_ONLY:-1}
    export DFLASH_CUDA_MMVQ_FP4_X4=${DFLASH_CUDA_MMVQ_FP4_X4:-1}
    export DFLASH_DS4_TP_MASKED_ROUTES=${DFLASH_DS4_TP_MASKED_ROUTES:-1}
    export DFLASH_DS4_TP_DEVICE_JOIN=${DFLASH_DS4_TP_DEVICE_JOIN:-1}
    export DFLASH_DS4_TP_NATIVE_ROUTE_WIDTH=${DFLASH_DS4_TP_NATIVE_ROUTE_WIDTH:-1}
    export DFLASH_DS4_TP_SPLIT_COUNT=${DFLASH_DS4_TP_SPLIT_COUNT:-1}
    export DFLASH_DS4_TP_ROUTE_PREFORK=${DFLASH_DS4_TP_ROUTE_PREFORK:-1}
    export DFLASH_DS4_TP_DEVICE_JOIN_SPLIT=${DFLASH_DS4_TP_DEVICE_JOIN_SPLIT:-1}
    export DFLASH_DS4_TP_FUSED_HC_JOIN=${DFLASH_DS4_TP_FUSED_HC_JOIN:-1}
    export DFLASH_DS4_TP_MAIN_ROUTE_WEIGHTS=${DFLASH_DS4_TP_MAIN_ROUTE_WEIGHTS:-1}
    export DFLASH_DS4_TP_COARSE_OWNER=${DFLASH_DS4_TP_COARSE_OWNER:-1}
    export DFLASH_DS4_TP_COARSE_OWNER_SPLIT=${DFLASH_DS4_TP_COARSE_OWNER_SPLIT:-0}
    export GGML_BATCH_PEER_COPIES=${GGML_BATCH_PEER_COPIES:-1}
    export DFLASH_CUDA_MMVQ_MOE_ROWS_PER_BLOCK=${DFLASH_CUDA_MMVQ_MOE_ROWS_PER_BLOCK:-2}
    export DFLASH_DS4_TP_CAPTURE_CACHE_SLOTS=${DFLASH_DS4_TP_CAPTURE_CACHE_SLOTS:-4}
    export DFLASH_DS4_TP_FUSED_CACHE_SLOTS=${DFLASH_DS4_TP_FUSED_CACHE_SLOTS:-9}
    export DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY=${DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY:-1}
    export DFLASH_DS4_GPU_ARGMAX_VERIFY=${DFLASH_DS4_GPU_ARGMAX_VERIFY:-1}
    export DFLASH_DS4_SPEC=1
    export DFLASH_DS4_DRAFT=$DRAFT
    export DFLASH_DS4_DRAFT_GPU=$DS4_DRAFT_DEVICE
    export DFLASH_DS4_SPEC_Q=${DFLASH_DS4_SPEC_Q:-4}
    export DFLASH_DS4_PINNED_ROLLBACK=${DFLASH_DS4_PINNED_ROLLBACK:-1}
    export DFLASH_DS4_FUSED_VERIFY=${DFLASH_DS4_FUSED_VERIFY:-1}
    export DFLASH_DS4_TOPK=${DFLASH_DS4_TOPK:-6}

    echo "launching dflash_server on port $PORT with run id $RUN_ID"
    "$BINARY" "$TARGET" \
        --target-device "$DS4_MAIN_DEVICE" \
        --peer-access \
        --ds4-expert-top-k "$DFLASH_DS4_TOPK" \
        --ds4-prefill sparse \
        --chunk 2048 \
        --max-ctx "$DS4_CONTEXT" \
        --prefix-cache-slots 0 \
        --prefill-cache-slots 0 \
        --host "$HOST" \
        --port "$PORT" \
        "$@" > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    echo "started pid $pid. log $LOG_FILE. pid file $PID_FILE"
    echo "evidence directory $EVIDENCE_DIR"
}

chat_body() {
    printf '{"model":"dflash","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0}' \
        "$MODEL_PROMPT" "$MAX_TOKENS"
}

chat_request() {
    local out_file=$1
    local http_code
    http_code=$(curl -sS -o "$out_file" -w '%{http_code}' \
        -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data "$(chat_body)")
    echo "$http_code"
}

cmd_doctor() {
    mkdir -p "$EVIDENCE_DIR"
    local out="$EVIDENCE_DIR/props.json"
    local http_code
    http_code=$(curl -sS -o "$out" -w '%{http_code}' "http://$HOST:$PORT/props.build")
    if [[ $http_code != 200 ]]; then
        echo "doctor failed. /props.build returned HTTP $http_code" >&2
        exit 1
    fi
    jq . "$out"
    echo "doctor passed. HTTP 200 from /props.build. saved $out"
}

cmd_chat_smoke() {
    mkdir -p "$EVIDENCE_DIR"
    local out="$EVIDENCE_DIR/chat-smoke.json"
    local http_code
    http_code=$(chat_request "$out")
    if [[ $http_code != 200 ]]; then
        echo "chat-smoke failed. HTTP $http_code. body $out" >&2
        exit 1
    fi
    local content
    content=$(jq -r '.choices[0].message.content // empty' "$out")
    if [[ -z $content ]]; then
        echo "chat-smoke failed. HTTP 200 but content is empty. body $out" >&2
        exit 1
    fi
    echo "chat-smoke passed. HTTP 200. content $content"
}

cmd_spec_flag() {
    mkdir -p "$EVIDENCE_DIR"
    local out="$EVIDENCE_DIR/spec-flag.json"
    local http_code
    http_code=$(chat_request "$out")
    if [[ $http_code != 200 ]]; then
        echo "spec-flag failed. HTTP $http_code. body $out" >&2
        exit 1
    fi
    # jq's alternative operator would collapse a JSON false into the
    # fallback, so test the key with has() first.
    local ran
    ran=$(jq -r '.usage | if has("spec_decode_ran") then (.spec_decode_ran | tostring) else "absent" end' "$out")
    if [[ $ran == true ]]; then
        echo "speculative decode ran. usage.spec_decode_ran is true. saved $out"
    elif [[ $ran == false ]]; then
        echo "speculative decode did NOT run. usage.spec_decode_ran is false. saved $out"
        return 1
    else
        echo "spec flag absent from response. server predates PR 604. saved $out" >&2
        return 1
    fi
}

cmd_cleanup() {
    mkdir -p "$EVIDENCE_DIR"
    if [[ -f $CONTAINER_FILE ]]; then
        local container
        container=$(cat "$CONTAINER_FILE")
        echo "removing recorded container $container"
        docker rm -f "$container" || echo "docker rm failed for $container" >&2
    fi
    if [[ -f $PID_FILE ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        # Kill only the exact PID this script recorded. Never a process name.
        if kill -0 "$pid" 2>/dev/null; then
            echo "stopping recorded pid $pid"
            kill "$pid" 2>/dev/null || true
            for _ in 1 2 3 4 5 6 7 8 9 10; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        else
            echo "recorded pid $pid is not running"
        fi
    fi
    echo "evidence kept under $EVIDENCE_DIR"
    ls -la "$EVIDENCE_DIR"
}

main() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 0
    fi
    case $1 in
        launch) shift; cmd_launch "$@" ;;
        doctor) cmd_doctor ;;
        chat-smoke) cmd_chat_smoke ;;
        spec-flag) cmd_spec_flag ;;
        cleanup) cmd_cleanup ;;
        -h|--help|help) usage; exit 0 ;;
        *) echo "unknown subcommand: $1" >&2; usage; exit 2 ;;
    esac
}

main "$@"
