#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CLIENT_WORK_DIR="${CLIENT_WORK_DIR:-$REPO_DIR/.harness-work}"
RUN_ROOT="${RUN_ROOT:-${RUN_DIR:-$CLIENT_WORK_DIR/runs/claude-llamacpp-decode-check}}"
TARGET="${TARGET:-$REPO_DIR/server/models/Qwen3.6-27B-Q4_K_M.gguf}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-$CLIENT_WORK_DIR/llama-cpp-server-build}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_BUILD_DIR/bin/llama-server}"
STAMP="${STAMP:-q8_32k_decode_check}"
PROMPT="${PROMPT:-Reply with exactly: OK_DONE}"
MARKER="${MARKER:-OK_DONE}"

mkdir -p "$RUN_ROOT"

REPO_DIR="$REPO_DIR" \
RUN_DIR="$RUN_ROOT" \
STAMP="$STAMP" \
TARGET="$TARGET" \
MODEL_SERVER=llamacpp \
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" \
LLAMA_COMPAT_PROXY=anthropic \
LLAMA_COMPAT_MAX_TOKENS=16 \
LLAMA_CACHE_TYPE_K=q8_0 \
LLAMA_CACHE_TYPE_V=q8_0 \
LLAMA_FLASH_ATTN=1 \
LLAMA_N_GPU_LAYERS=999 \
LLAMA_PARALLEL=1 \
LLAMA_CACHE_RAM=0 \
LLAMA_EXTRA_ARGS="--no-warmup" \
MAX_CTX=32768 \
CLAUDE_TIMEOUT=600 \
PROMPT="$PROMPT" \
MARKER="$MARKER" \
"$SCRIPT_DIR/run_claude_code.sh" > "$RUN_ROOT/$STAMP.wrapper.log" 2>&1 || true
