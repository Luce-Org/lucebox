#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/.harness-runs}"
STAMP="${STAMP:-generation-baseline-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$RUN_DIR/$STAMP"

TARGET="${TARGET:-$REPO_DIR/dflash/models/Qwen3.6-27B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-$REPO_DIR/dflash/models/draft/dflash-draft-3.6-q8_0.gguf}"
DFLASH_BIN="${DFLASH_BIN:-$REPO_DIR/dflash/build/test_dflash}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$REPO_DIR/dflash/deps/llama.cpp/build/bin/llama-server}"

HOST="${HOST:-127.0.0.1}"
LUCEBOX_PORT="${LUCEBOX_PORT:-18080}"
LLAMA_PORT="${LLAMA_PORT:-18082}"
MAX_CTX="${MAX_CTX:-32768}"
MAX_TOKENS="${MAX_TOKENS:-256}"
BUDGET="${BUDGET:-22}"
VERIFY_MODE="${VERIFY_MODE:-ddtree}"
FA_WINDOW="${FA_WINDOW:-2048}"
CACHE_TYPE_K="${CACHE_TYPE_K:-tq3_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-tq3_0}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---lazy-draft}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-999}"
MODEL_ID="${MODEL_ID:-luce-dflash}"
LLAMA_MODEL_ID="${LLAMA_MODEL_ID:-llama-cpp}"
API_KEY="${API_KEY:-sk-lucebox}"
PROMPTS="${PROMPTS:-$SCRIPT_DIR/prompts/generation_smoke.jsonl}"

mkdir -p "$LOG_DIR"

wait_health() {
  local url="$1"
  local pid="$2"
  local log="$3"
  for _ in $(seq 1 300); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "process exited early; log: $log" >&2
      tail -n 160 "$log" >&2 || true
      return 1
    fi
  done
  echo "process did not become healthy; log: $log" >&2
  tail -n 160 "$log" >&2 || true
  return 1
}

stop_pid() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

resolve_llama_server() {
  if [[ -x "$LLAMA_SERVER_BIN" ]]; then
    echo "$LLAMA_SERVER_BIN"
    return 0
  fi
  if command -v llama-server >/dev/null 2>&1; then
    command -v llama-server
    return 0
  fi
  echo "llama-server not found. Set LLAMA_SERVER_BIN=/path/to/llama-server." >&2
  return 1
}

LLAMA_JSON="$LOG_DIR/llamacpp.json"
LUCEBOX_JSON="$LOG_DIR/lucebox.json"
COMPARE_JSON="$LOG_DIR/compare.json"
REPORT_MD="$LOG_DIR/report.md"

LLAMA_LOG="$LOG_DIR/llamacpp-server.log"
LLAMA_SERVER_BIN="$(resolve_llama_server)"
"$LLAMA_SERVER_BIN" \
  --model "$TARGET" \
  --alias "$LLAMA_MODEL_ID" \
  --ctx-size "$MAX_CTX" \
  --n-gpu-layers "$LLAMA_N_GPU_LAYERS" \
  --host "$HOST" \
  --port "$LLAMA_PORT" \
  > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
trap 'stop_pid "${LLAMA_PID:-}"; stop_pid "${LUCEBOX_PID:-}"' EXIT
wait_health "http://$HOST:$LLAMA_PORT/health" "$LLAMA_PID" "$LLAMA_LOG"

python3 "$SCRIPT_DIR/generation_benchmark.py" run \
  --name llama.cpp \
  --url "http://$HOST:$LLAMA_PORT/v1" \
  --model "$LLAMA_MODEL_ID" \
  --prompts "$PROMPTS" \
  --max-tokens "$MAX_TOKENS" \
  --json-out "$LLAMA_JSON"

stop_pid "$LLAMA_PID"
LLAMA_PID=""

LUCEBOX_LOG="$LOG_DIR/lucebox-server.log"
cd "$REPO_DIR"
extra_args=()
if [[ -n "$EXTRA_SERVER_ARGS" ]]; then
  read -r -a extra_args <<< "$EXTRA_SERVER_ARGS"
fi
python3 -u dflash/scripts/server.py \
  --host "$HOST" \
  --port "$LUCEBOX_PORT" \
  --target "$TARGET" \
  --draft "$DRAFT" \
  --bin "$DFLASH_BIN" \
  --budget "$BUDGET" \
  --verify-mode "$VERIFY_MODE" \
  --max-ctx "$MAX_CTX" \
  --fa-window "$FA_WINDOW" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  --prefix-cache-slots 0 \
  --prefill-cache-slots 0 \
  "${extra_args[@]}" \
  > "$LUCEBOX_LOG" 2>&1 &
LUCEBOX_PID=$!
wait_health "http://$HOST:$LUCEBOX_PORT/health" "$LUCEBOX_PID" "$LUCEBOX_LOG"

python3 "$SCRIPT_DIR/generation_benchmark.py" run \
  --name lucebox \
  --url "http://$HOST:$LUCEBOX_PORT/v1" \
  --api-key "$API_KEY" \
  --model "$MODEL_ID" \
  --prompts "$PROMPTS" \
  --max-tokens "$MAX_TOKENS" \
  --json-out "$LUCEBOX_JSON"

python3 "$SCRIPT_DIR/generation_benchmark.py" compare \
  --baseline "$LLAMA_JSON" \
  --candidate "$LUCEBOX_JSON" \
  --json-out "$COMPARE_JSON" \
  --md-out "$REPORT_MD"

echo "run_dir=$LOG_DIR"
echo "report=$REPORT_MD"
