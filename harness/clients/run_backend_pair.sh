#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-/workspace/lucebox-client-harness-runs}"
CLIENT="${CLIENT:-opencode}"
PAIR_STAMP="${PAIR_STAMP:-backend-pair-$CLIENT-$(date +%Y%m%d-%H%M%S)}"
PAIR_DIR="$RUN_DIR/$PAIR_STAMP"

case "$CLIENT" in
  claude|claude_code) CLIENT_SCRIPT="$SCRIPT_DIR/run_claude_code.sh" ;;
  codex) CLIENT_SCRIPT="$SCRIPT_DIR/run_codex.sh" ;;
  hermes) CLIENT_SCRIPT="$SCRIPT_DIR/run_hermes.sh" ;;
  opencode) CLIENT_SCRIPT="$SCRIPT_DIR/run_opencode.sh" ;;
  openclaw) CLIENT_SCRIPT="$SCRIPT_DIR/run_openclaw.sh" ;;
  openwebui) CLIENT_SCRIPT="$SCRIPT_DIR/run_openwebui.sh" ;;
  openwebui_tools) CLIENT_SCRIPT="$SCRIPT_DIR/run_openwebui_tools.sh" ;;
  pi) CLIENT_SCRIPT="$SCRIPT_DIR/run_pi.sh" ;;
  *)
    echo "unknown CLIENT=$CLIENT" >&2
    exit 2
    ;;
esac

mkdir -p "$PAIR_DIR"
PAIR_LOG="$PAIR_DIR/pair.log"

run_backend() {
  local backend="$1"
  local stamp="$PAIR_STAMP-$backend"
  echo "[$(date -Is)] backend=$backend client=$CLIENT script=$CLIENT_SCRIPT" | tee -a "$PAIR_LOG"
  set +e
  MODEL_SERVER="$backend" \
  RUN_DIR="$PAIR_DIR" \
  STAMP="$stamp" \
  "$CLIENT_SCRIPT" 2>&1 | tee "$PAIR_DIR/$backend.out"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[$(date -Is)] backend=$backend rc=$rc" | tee -a "$PAIR_LOG"
  return "$rc"
}

# llama.cpp is run first so Lucebox numbers are not helped by residual client
# caches from a previous Lucebox run. Each backend starts and stops its own
# server, so only one 27B model is resident on the 24 GB card at a time.
LLAMA_RC=0
LUCEBOX_RC=0
run_backend llamacpp || LLAMA_RC=$?
run_backend lucebox || LUCEBOX_RC=$?

python3 "$SCRIPT_DIR/summarize_backend_pair.py" "$PAIR_DIR" | tee "$PAIR_DIR/summary.md" || true

echo "pair_dir=$PAIR_DIR"
echo "llamacpp_rc=$LLAMA_RC"
echo "lucebox_rc=$LUCEBOX_RC"

if [[ "$LLAMA_RC" -ne 0 || "$LUCEBOX_RC" -ne 0 ]]; then
  exit 1
fi
