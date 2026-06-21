#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/home/peppi/Dev/lucebox-hub
cd "$REPO_DIR"

# GPU exclusion lock — never contend on the same GPU
exec 9>/tmp/lucebox_gpu.lock
flock -x 9

# Gate-local log dir (separate from the timestamped run dir common.sh creates)
LOG_DIR="$REPO_DIR/bench/qwen35moe_dflash/runlog"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"
# NOTE: common.sh always sets LOG_DIR="$RUN_DIR/$STAMP" internally; we cannot
# override it via env.  Instead we capture the run_dir= from driver output below.

export DFLASH_SERVER_BIN="${DFLASH_SERVER_BIN:-$REPO_DIR/server/build/dflash_server_specfix6}"
export TARGET="${TARGET:-/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf}"
export DRAFT="${DRAFT:-/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf}"
export MODEL_ID="${MODEL_ID:-luce-dflash}"
export MARKER="${MARKER:-OK_DONE}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-18080}"
export MAX_CTX="${MAX_CTX:-32768}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export FA_WINDOW="${FA_WINDOW:-0}"
export CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
export CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
export FORCE_TEMPERATURE="${FORCE_TEMPERATURE:-0}"
export PROMPT_FILE="${PROMPT_FILE:-$REPO_DIR/harness/clients/prompts/repo_inspection.txt}"
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---lazy-draft --chat-template-file /home/peppi/models/qwen3-coder-chat-template.jinja}"
export CLAUDE_TIMEOUT="${CLAUDE_TIMEOUT:-120}"

GATE_LOG="$LOG_DIR/gate_driver.log"

echo "[gate] starting run — $(date --iso-8601=seconds)"
echo "[gate] PORT=$PORT  MAX_CTX=$MAX_CTX  FORCE_TEMPERATURE=$FORCE_TEMPERATURE"

t0=$(date +%s.%N)
set +e
bash "$REPO_DIR/harness/clients/run_claude_code.sh" 2>&1 | tee "$GATE_LOG"
DRIVER_RC=${PIPESTATUS[0]}
set -e
t1=$(date +%s.%N)
wall=$(echo "$t1 - $t0" | bc)

echo "[gate] driver exit=$DRIVER_RC  wall=${wall}s"

# Resolve run_dir from driver output (common.sh prints "run_dir=<path>")
RUN_DIR_LINE=$(grep '^run_dir=' "$GATE_LOG" | tail -1 || true)
if [[ -z "$RUN_DIR_LINE" ]]; then
  echo "[gate] ERROR: could not find run_dir= in driver output" >&2
  exit 1
fi
ACTUAL_RUN_DIR="${RUN_DIR_LINE#run_dir=}"
SERVER_LOG="$ACTUAL_RUN_DIR/server.log"
CLIENT_OUT="$ACTUAL_RUN_DIR/claude-code.out"

echo "[gate] SERVER_LOG=$SERVER_LOG"
echo "[gate] CLIENT_OUT=$CLIENT_OUT"

# ── parse + assert ────────────────────────────────────────────────────────────

# TOOLS / MULTITURN: chat DONE count
chat_done_count=0
if [[ -f "$SERVER_LOG" ]]; then
  chat_done_count=$(grep -c 'chat DONE' "$SERVER_LOG" || true)
fi
if [[ "$chat_done_count" -ge 2 ]]; then
  tool_gate="PASS"
else
  tool_gate="FAIL"
fi

# COMPLETION: marker in client output
ok_done="N"
if [[ -f "$CLIENT_OUT" ]] && grep -q "$MARKER" "$CLIENT_OUT" 2>/dev/null; then
  ok_done="Y"
fi

# DFLASH ENGAGED: [spec-decode] present
spec_decode_yn="N"
if [[ -f "$SERVER_LOG" ]] && grep -q '\[spec-decode\]' "$SERVER_LOG" 2>/dev/null; then
  spec_decode_yn="Y"
elif [[ -f "$SERVER_LOG" ]] && grep -q '\[ar-decode\]' "$SERVER_LOG" 2>/dev/null; then
  echo "[gate] NOTE: only [ar-decode] found — dFlash did NOT engage (temperature gate?)"
fi

# DECODE TPS: mode-agnostic — max from `chat DONE ... decode=Xs(Ytok/s)`
decode_tps="0"
accept_pct="0"
if [[ -f "$SERVER_LOG" ]]; then
  decode_tps=$(grep -oE 'decode=[0-9.]+s\(([0-9.]+)tok/s\)' "$SERVER_LOG" \
    | grep -oE '\(([0-9.]+)tok' | tr -d '(tok' | sort -n | tail -1 || echo "0")
  decode_tps="${decode_tps:-0}"
  # accept% from last accepted=A/T (P%) line (spec-decode only)
  accept_line=$(grep -oE 'accepted=[0-9]+/[0-9]+ \([0-9.]+%\)' "$SERVER_LOG" | tail -1 || true)
  if [[ -n "$accept_line" ]]; then
    accept_pct=$(echo "$accept_line" | grep -oE '\([0-9.]+%\)' | tr -d '()%')
  fi
fi
if awk "BEGIN{exit !($decode_tps > 66)}"; then
  decode_gate="PASS"
else
  decode_gate="FAIL"
fi

# CACHING: restore=true with prefix_len>0
restore_count=0
if [[ -f "$SERVER_LOG" ]]; then
  restore_count=$(grep -c 'restore=true' "$SERVER_LOG" || true)
fi
prefix_hit=0
if [[ -f "$SERVER_LOG" ]] && grep -qE 'prefix_len=[1-9]' "$SERVER_LOG" 2>/dev/null; then
  prefix_hit=1
fi
if [[ "$restore_count" -ge 1 && "$prefix_hit" -eq 1 ]]; then
  cache_gate="PASS"
else
  cache_gate="FAIL"
fi

# WALL gate
if awk "BEGIN{exit !($wall < 40)}"; then
  wall_gate="PASS"
else
  wall_gate="FAIL"
fi

# OVERALL — spec-decode only required when SPEC_REQUIRED=1 (dFlash arm)
spec_ok="Y"
if [[ "${SPEC_REQUIRED:-0}" == "1" && "$spec_decode_yn" != "Y" ]]; then
  spec_ok="N"
fi
if [[ "$tool_gate" == "PASS" && "$decode_gate" == "PASS" && "$cache_gate" == "PASS" && "$wall_gate" == "PASS" && "$ok_done" == "Y" && "$spec_ok" == "Y" ]]; then
  overall="PASS"
else
  overall="FAIL"
fi

echo ""
echo "=== GATE RESULT ==="
echo "wall_s=${wall}  chat_done=${chat_done_count}  ok_done=${ok_done}  spec_decode=${spec_decode_yn}  decode_tps=${decode_tps}  accept=${accept_pct}%  restore=${restore_count}"
echo "tool_gate:   ${tool_gate}"
echo "decode_gate: ${decode_gate} (>66)"
echo "cache_gate:  ${cache_gate}"
echo "wall_gate:   ${wall_gate} (<40s)"
echo "OVERALL: ${overall}"

[[ "$overall" == "PASS" ]]
