#!/usr/bin/env bash
# Isolation 2x2 bench: cross DRAFT_CTX_MAX x FEAT_RING_CAP
# Runs 4 arms × 5 ctx sizes; saves table to isolation_2x2.md
set -uo pipefail

BENCH_DIR="/home/peppi/Dev/lucebox-hub/bench/qwen35moe_dflash/ctxsweep"
LAUNCH_SCRIPT="$BENCH_DIR/launch_arm.sh"
PORT=18081
BASE_URL="http://127.0.0.1:${PORT}"

# ctx sizes to test (in order)
CTX_SIZES=(2048 4096 8192 16384 32768)

# ARM definitions
ARMS=(ARM_A_uncap ARM_B_recipe ARM_C_ring4k ARM_D_both4k)
declare -A ARM_DRAFT_CTX_MAX=(
  [ARM_A_uncap]=40960
  [ARM_B_recipe]=2048
  [ARM_C_ring4k]=40960
  [ARM_D_both4k]=2048
)
declare -A ARM_FEAT_RING_CAP=(
  [ARM_A_uncap]=40960
  [ARM_B_recipe]=40960
  [ARM_C_ring4k]=4096
  [ARM_D_both4k]=4096
)

TABLE_FILE="$BENCH_DIR/isolation_2x2.md"
BUILD_HASH=$(git -C /home/peppi/Dev/lucebox-hub rev-parse --short HEAD)
BUILD_MD5=$(md5sum /home/peppi/Dev/lucebox-hub/server/build/dflash_server | awk '{print $1}')

# Write table header
{
  echo "# Isolation 2x2 Benchmark"
  echo ""
  echo "Build: ${BUILD_HASH}  md5: ${BUILD_MD5}"
  echo "Date: $(date -u)"
  echo ""
  echo "| arm | DRAFT_CTX_MAX | FEAT_RING_CAP | banner cap= | ctx | prompt_tok | accept% | avg_commit | decode tok/s |"
  echo "|-----|---------------|---------------|-------------|-----|------------|---------|------------|--------------|"
} > "$TABLE_FILE"

wait_healthy() {
  local url="$1" timeout="$2"
  local elapsed=0
  while [ $elapsed -lt $timeout ]; do
    if curl -fsS "${url}/health" >/dev/null 2>&1; then
      echo "healthy"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "[wait] ${elapsed}s..." >&2
  done
  echo "timeout"
  return 1
}

for ARM in "${ARMS[@]}"; do
  DRAFT_CTX="${ARM_DRAFT_CTX_MAX[$ARM]}"
  RING_CAP="${ARM_FEAT_RING_CAP[$ARM]}"
  LOGFILE="$BENCH_DIR/iso_${ARM}.log"

  echo ""
  echo "=============================="
  echo "ARM: $ARM  DRAFT_CTX_MAX=$DRAFT_CTX  FEAT_RING_CAP=$RING_CAP"
  echo "=============================="

  # Clear old log for this arm
  > "$LOGFILE"

  # Acquire GPU lock
  exec 9>/tmp/lucebox_gpu.lock
  flock -x 9

  # Launch server (nohup, background; stdout/stderr go through launch_arm.sh into LOGFILE)
  nohup bash "$LAUNCH_SCRIPT" "$ARM" "$DRAFT_CTX" "$RING_CAP" >/dev/null 2>&1 &
  SERVER_PID=$!
  echo "[bench] Server PID=$SERVER_PID"

  # Wait for health
  echo "[bench] Waiting for server health (up to 180s)..."
  STATUS=$(wait_healthy "$BASE_URL" 180)
  if [ "$STATUS" != "healthy" ]; then
    echo "[bench] ERROR: Server $ARM failed to become healthy within 180s — check $LOGFILE"
    kill -9 "$SERVER_PID" 2>/dev/null || true
    flock -u 9
    for CTX in "${CTX_SIZES[@]}"; do
      echo "| $ARM | $DRAFT_CTX | $RING_CAP | LAUNCH_FAIL | $CTX | - | - | - | - |" >> "$TABLE_FILE"
    done
    continue
  fi
  echo "[bench] Server healthy at $(date)"

  # Wait a beat for banner to flush
  sleep 2

  # Extract banner cap= from server log (first occurrence)
  BANNER_CAP=$(grep -m1 "cap=" "$LOGFILE" 2>/dev/null | grep -oP 'cap=\K[0-9]+' || echo "?")
  echo "[bench] Banner cap=$BANNER_CAP"

  # Run prompts for each ctx size
  for CTX in "${CTX_SIZES[@]}"; do
    PROMPT_FILE="$BENCH_DIR/ctx_$(printf '%06d' $CTX).json"
    if [ ! -f "$PROMPT_FILE" ]; then
      echo "[bench] MISSING prompt file: $PROMPT_FILE"
      echo "| $ARM | $DRAFT_CTX | $RING_CAP | $BANNER_CAP | $CTX | MISSING | - | - | - |" >> "$TABLE_FILE"
      continue
    fi

    echo "[bench] ctx=$CTX prompt=$PROMPT_FILE ..."

    # Count lines BEFORE request so we can grab the NEW spec/ar-decode line
    LINES_BEFORE=$(wc -l < "$LOGFILE")

    # POST request — wait for completion
    curl -s "${BASE_URL}/v1/chat/completions" \
      -H 'content-type: application/json' \
      -d @"$PROMPT_FILE" \
      -o /dev/null \
      --max-time 600 2>/dev/null || true

    # Give server a beat to flush the DONE line
    sleep 1

    # Get lines added since the request
    LINES_AFTER=$(wc -l < "$LOGFILE")
    NEW_LINES=$(tail -n $((LINES_AFTER - LINES_BEFORE)) "$LOGFILE")

    echo "[bench] New log lines for ctx=$CTX:"
    echo "$NEW_LINES" | grep -E "spec-decode|ar-decode|chat (START|DONE)" | head -8 || echo "  (none)"

    # Prefer spec-decode line, fall back to ar-decode
    SPEC_LINE=$(echo "$NEW_LINES" | grep "\[spec-decode\] tokens=" | tail -1 || echo "")
    AR_LINE=$(echo "$NEW_LINES" | grep "\[ar-decode\] tokens=" | tail -1 || echo "")
    DONE_LINE=$(echo "$NEW_LINES" | grep "\[server\] chat DONE" | tail -1 || echo "")

    if [ -n "$SPEC_LINE" ]; then
      DECODE_LINE="$SPEC_LINE"
      DECODE_TYPE="spec"
    elif [ -n "$AR_LINE" ]; then
      DECODE_LINE="$AR_LINE"
      DECODE_TYPE="ar"
    else
      DECODE_LINE=""
      DECODE_TYPE="none"
    fi

    echo "[bench] decode_type=$DECODE_TYPE"

    if [ -n "$DECODE_LINE" ]; then
      SPEED=$(echo "$DECODE_LINE" | grep -oP 'speed=\K[0-9.]+' || echo "-")
      ACCEPT_PCT=$(echo "$DECODE_LINE" | grep -oP '\(([0-9.]+)%\)' | grep -oP '[0-9.]+' || echo "-")
      AVG_COMMIT=$(echo "$DECODE_LINE" | grep -oP 'avg_commit=\K[0-9.]+' || echo "-")
    else
      SPEED="-"
      ACCEPT_PCT="-"
      AVG_COMMIT="-"
    fi

    # Prompt tokens from DONE line
    if [ -n "$DONE_LINE" ]; then
      PROMPT_TOK=$(echo "$DONE_LINE" | grep -oP ' in=\K[0-9]+' || echo "$CTX")
      # Use decode TPS from DONE line if speed not found above
      if [ "$SPEED" = "-" ]; then
        SPEED=$(echo "$DONE_LINE" | grep -oP 'decode=\S+s\(\K[0-9.]+' || echo "-")
      fi
    else
      PROMPT_TOK="$CTX"
    fi

    # For AR-decode, accept% is 0 (no speculation) -- mark it
    if [ "$DECODE_TYPE" = "ar" ]; then
      ACCEPT_PCT="AR"
      AVG_COMMIT="1"
    fi

    echo "| $ARM | $DRAFT_CTX | $RING_CAP | $BANNER_CAP | $CTX | $PROMPT_TOK | $ACCEPT_PCT | $AVG_COMMIT | $SPEED |" | tee -a "$TABLE_FILE"

  done

  # Kill server
  echo "[bench] Killing server PID=$SERVER_PID"
  kill -9 "$SERVER_PID" 2>/dev/null || true
  sleep 4

  # Kill any straggler on the port
  STRAGGLER=$(pgrep -f "port ${PORT}" 2>/dev/null || true)
  if [ -n "$STRAGGLER" ]; then
    echo "[bench] Killing straggler PID=$STRAGGLER on port $PORT"
    kill -9 "$STRAGGLER" 2>/dev/null || true
    sleep 2
  fi

  # Release GPU lock
  flock -u 9

  echo "[bench] ARM $ARM complete."
done

echo ""
echo "=============================="
echo "Benchmark complete."
echo "Table saved to: $TABLE_FILE"
echo "=============================="
echo ""
cat "$TABLE_FILE"
