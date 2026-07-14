#!/usr/bin/env bash
set -euo pipefail

# Source this file from a client harness launcher on the RTX 3090 host.
# Assumes the repo and client package cache already exist on that machine.

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEFAULT_REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
CLIENT_WORK_DIR="${CLIENT_WORK_DIR:-$REPO_DIR/.harness-work}"
RUN_DIR="${RUN_DIR:-$CLIENT_WORK_DIR/runs}"
AUTO_INSTALL_CLIENTS="${AUTO_INSTALL_CLIENTS:-1}"

DEFAULT_TARGET="$REPO_DIR/server/models/Qwen3.6-27B-Q4_K_M.gguf"
DEFAULT_DRAFT="$REPO_DIR/server/models/draft/dflash-draft-3.6-q4_k_m.gguf"
TARGET_WAS_EXPLICIT=0
if [[ -n "${TARGET+x}" ]]; then
  TARGET_WAS_EXPLICIT=1
elif [[ -n "${DFLASH_TARGET+x}" ]]; then
  TARGET="$DFLASH_TARGET"
  TARGET_WAS_EXPLICIT=1
else
  TARGET="$DEFAULT_TARGET"
fi
if [[ -n "${DRAFT+x}" ]]; then
  :
elif [[ -n "${DFLASH_DRAFT+x}" ]]; then
  DRAFT="$DFLASH_DRAFT"
elif [[ "$TARGET_WAS_EXPLICIT" == "1" ]]; then
  # A custom target may be Gemma/Laguna/Qwen3 or another model without a
  # matching DFlash draft. Avoid attaching the default Qwen draft silently.
  DRAFT=""
else
  DRAFT="$DEFAULT_DRAFT"
fi
MODEL_SERVER="${MODEL_SERVER:-lucebox}"
DFLASH_SERVER_BIN="${DFLASH_SERVER_BIN:-$REPO_DIR/server/build/dflash_server}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-$CLIENT_WORK_DIR/llama-cpp-server-build}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_BUILD_DIR/bin/llama-server}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-999}"
LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-1}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_CACHE_RAM="${LLAMA_CACHE_RAM:-0}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
LLAMA_COMPAT_PROXY="${LLAMA_COMPAT_PROXY:-}"
LLAMA_COMPAT_MAX_TOKENS="${LLAMA_COMPAT_MAX_TOKENS:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
LLAMA_UPSTREAM_PORT="${LLAMA_UPSTREAM_PORT:-$((PORT + 1))}"
MAX_CTX="${MAX_CTX:-16384}"
VERIFY_MODE="${VERIFY_MODE:-seq}"
BUDGET="${BUDGET:-22}"
# FA_WINDOW: 0 = full attention. A finite window is known to break tool
# calling (Qwen3.6) - only set it explicitly for experiments.
FA_WINDOW="${FA_WINDOW:-0}"
# CACHE_TYPE_K/V: empty = let the server pick the model family's default
# (laguna q8_0, q4_0 base; forcing tq3_0/q4_0 on laguna garbles its
# output). Set explicitly to experiment.
CACHE_TYPE_K="${CACHE_TYPE_K:-}"
CACHE_TYPE_V="${CACHE_TYPE_V:-}"
LLAMA_CACHE_TYPE_K="${LLAMA_CACHE_TYPE_K:-${CACHE_TYPE_K:-q8_0}}"
LLAMA_CACHE_TYPE_V="${LLAMA_CACHE_TYPE_V:-${CACHE_TYPE_V:-q8_0}}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

MODEL_ID="${MODEL_ID:-luce-dflash}"
API_KEY="${API_KEY:-sk-lucebox}"
MARKER="${MARKER:-lucebox-client-ok}"
PROMPT="${PROMPT:-Reply with exactly: $MARKER}"
PROMPT_FILE="${PROMPT_FILE:-}"

if [[ -n "$PROMPT_FILE" ]]; then
  PROMPT="$(<"$PROMPT_FILE")"
fi

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
BASE_URL="http://$HOST:$PORT"
LOG_DIR="$RUN_DIR/$STAMP"
SERVER_LOG="$LOG_DIR/server.log"

mkdir -p "$LOG_DIR"

require_client_binary() {
  local label="$1"
  local path="$2"
  local client="$3"
  local env_var="$4"
  if [[ -x "$path" ]]; then
    return 0
  fi
  if [[ "$AUTO_INSTALL_CLIENTS" == "1" ]]; then
    echo "$label binary not found; installing client package into $CLIENT_WORK_DIR" >&2
    if ! python3 "$REPO_DIR/harness/client_test_runner.py" \
      --work-dir "$CLIENT_WORK_DIR" \
      install \
      --clients "$client" >&2; then
      echo "$label auto-install failed." >&2
    fi
    if [[ -x "$path" ]]; then
      return 0
    fi
  fi
  echo "$label binary not found or not executable: $path" >&2
  echo "Install it with:" >&2
  echo "  python3 $REPO_DIR/harness/client_test_runner.py --work-dir $CLIENT_WORK_DIR install --clients $client" >&2
  echo "or set $env_var=/path/to/$(basename "$path")." >&2
  return 1
}

draft_enabled() {
  [[ -n "${DRAFT:-}" && "$DRAFT" != "none" && "$DRAFT" != "off" && "$DRAFT" != "0" ]]
}

start_lucebox_server() {
  if [[ "$MODEL_SERVER" == "llamacpp" ]]; then
    start_llamacpp_server
    return
  fi
  if [[ "$MODEL_SERVER" != "lucebox" ]]; then
    echo "unknown MODEL_SERVER=$MODEL_SERVER; expected lucebox or llamacpp" >&2
    return 1
  fi
  start_dflash_native_server
}

start_dflash_native_server() {
  if [[ ! -x "$DFLASH_SERVER_BIN" ]]; then
    echo "dflash_server not found or not executable: $DFLASH_SERVER_BIN" >&2
    echo "Build it first, for example:" >&2
    echo "  cmake -S $REPO_DIR/dflash -B $REPO_DIR/server/build -DGGML_CUDA=ON" >&2
    echo "  cmake --build $REPO_DIR/server/build --target dflash_server -j\$(nproc)" >&2
    return 1
  fi
  if [[ ! -f "$TARGET" ]]; then
    echo "target GGUF not found: $TARGET" >&2
    echo "Set TARGET=/path/to/model.gguf or DFLASH_TARGET=/path/to/model.gguf, or download the default:" >&2
    echo "  hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir $REPO_DIR/server/models/" >&2
    return 1
  fi
  if draft_enabled && [[ ! -f "$DRAFT" ]]; then
    echo "DFlash draft not found: $DRAFT" >&2
    echo "Set DRAFT=/path/to/dflash-draft.gguf or DFLASH_DRAFT=/path/to/dflash-draft.gguf, or download the default:" >&2
    echo "  hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir $REPO_DIR/server/models/draft/" >&2
    return 1
  fi
  local draft_args=()
  if draft_enabled; then
    draft_args=(--draft "$DRAFT")
  fi
  local extra_args=()
  if [[ -n "$EXTRA_SERVER_ARGS" ]]; then
    read -r -a extra_args <<< "$EXTRA_SERVER_ARGS"
  fi
  local ddtree_args=()
  if [[ "$VERIFY_MODE" == "ddtree" ]]; then
    ddtree_args=(--ddtree --ddtree-budget "$BUDGET")
  fi
  local fa_args=()
  if [[ -n "$FA_WINDOW" ]] && [[ "$FA_WINDOW" != "0" ]]; then
    fa_args=(--fa-window "$FA_WINDOW")
  fi
  # Export KV cache type env vars for the C++ server to pick up (only when
  # explicitly requested: the per-axis envs override family defaults).
  if [[ -n "$CACHE_TYPE_K" ]]; then export DFLASH27B_KV_K="$CACHE_TYPE_K"; fi
  if [[ -n "$CACHE_TYPE_V" ]]; then export DFLASH27B_KV_V="$CACHE_TYPE_V"; fi
  "$DFLASH_SERVER_BIN" "$TARGET" \
    "${draft_args[@]}" \
    --host "$HOST" \
    --port "$PORT" \
    --max-ctx "$MAX_CTX" \
    --max-tokens "$MAX_TOKENS" \
    --model-name "$MODEL_ID" \
    "${ddtree_args[@]}" \
    "${fa_args[@]}" \
    "${extra_args[@]}" \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
}

start_llamacpp_server() {
  if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
    echo "llama-server not found or not executable: $LLAMA_SERVER_BIN" >&2
    echo "Build it from an external llama.cpp checkout or set LLAMA_SERVER_BIN to an existing binary." >&2
    echo "For example:" >&2
    echo "  cmake -S /path/to/llama.cpp -B $LLAMA_BUILD_DIR -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF" >&2
    echo "  cmake --build $LLAMA_BUILD_DIR --target llama-server -j2" >&2
    return 1
  fi
  if [[ ! -f "$TARGET" ]]; then
    echo "target GGUF not found: $TARGET" >&2
    echo "Set TARGET=/path/to/model.gguf or DFLASH_TARGET=/path/to/model.gguf, or download the default:" >&2
    echo "  hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir $REPO_DIR/server/models/" >&2
    return 1
  fi
  local extra_args=()
  if [[ -n "$LLAMA_EXTRA_ARGS" ]]; then
    read -r -a extra_args <<< "$LLAMA_EXTRA_ARGS"
  fi
  local flash_args=()
  if [[ "$LLAMA_FLASH_ATTN" == "1" ]]; then
    flash_args=(--flash-attn on)
  fi
  local llama_port="$PORT"
  if [[ -n "$LLAMA_COMPAT_PROXY" ]]; then
    llama_port="$LLAMA_UPSTREAM_PORT"
  fi
  "$LLAMA_SERVER_BIN" \
    --model "$TARGET" \
    --alias "$MODEL_ID" \
    --ctx-size "$MAX_CTX" \
    --n-gpu-layers "$LLAMA_N_GPU_LAYERS" \
    --cache-type-k "$LLAMA_CACHE_TYPE_K" \
    --cache-type-v "$LLAMA_CACHE_TYPE_V" \
    --parallel "$LLAMA_PARALLEL" \
    --cache-ram "$LLAMA_CACHE_RAM" \
    --host "$HOST" \
    --port "$llama_port" \
    "${flash_args[@]}" \
    "${extra_args[@]}" \
    > "$SERVER_LOG" 2>&1 &
  LLAMA_SERVER_PID=$!

  if [[ -n "$LLAMA_COMPAT_PROXY" ]]; then
    for _ in $(seq 1 300); do
      if curl -fsS "http://$HOST:$LLAMA_UPSTREAM_PORT/health" >/dev/null 2>&1; then
        break
      fi
      sleep 1
      if ! kill -0 "$LLAMA_SERVER_PID" 2>/dev/null; then
        echo "llama-server exited early; log: $SERVER_LOG" >&2
        tail -n 160 "$SERVER_LOG" >&2 || true
        return 1
      fi
    done
    python3 -u "$SCRIPT_DIR/llamacpp_compat_proxy.py" \
      --host "$HOST" \
      --port "$PORT" \
      --upstream "http://$HOST:$LLAMA_UPSTREAM_PORT" \
      --model "$MODEL_ID" \
      --max-tokens-cap "$LLAMA_COMPAT_MAX_TOKENS" \
      >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
  else
    SERVER_PID=$LLAMA_SERVER_PID
  fi
}

wait_lucebox_server() {
  for _ in $(seq 1 300); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited early; log: $SERVER_LOG" >&2
      tail -n 160 "$SERVER_LOG" >&2 || true
      return 1
    fi
  done
  echo "server did not become healthy; log: $SERVER_LOG" >&2
  tail -n 160 "$SERVER_LOG" >&2 || true
  return 1
}

stop_lucebox_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "${LLAMA_SERVER_PID:-}" ]] && [[ "${LLAMA_SERVER_PID:-}" != "${SERVER_PID:-}" ]] && kill -0 "$LLAMA_SERVER_PID" 2>/dev/null; then
    kill "$LLAMA_SERVER_PID" 2>/dev/null || true
    wait "$LLAMA_SERVER_PID" 2>/dev/null || true
  fi
}

finish_report() {
  local client_out="$1"
  local rc="$2"
  echo "rc=$rc"
  echo "model_server=$MODEL_SERVER"
  echo "run_dir=$LOG_DIR"
  echo "client_out=$client_out"
  echo "server_log=$SERVER_LOG"
  echo "--- marker check ---"
  grep -n "$MARKER" "$client_out" || true
  echo "--- server tail ---"
  tail -n 120 "$SERVER_LOG" || true
  echo "--- gpu ---"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
}
