#!/usr/bin/env bash
# Task #47: ee7 + --prefill-skip-park empirical test at 32K
# Minimal script, avoids python tokenizer network calls

set -uo pipefail

REPO="/home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath"
OUT="$REPO/dflash/bench/results/2026-05-22_skip_park_32k"
NIAH="/tmp/niah_32768.jsonl"
SERVER_LOG="$OUT/skip_park_server.log"
VRAM_CSV="$OUT/vram.csv"
PORT=18100
BIN="$REPO/dflash/build/dflash_server"
TARGET="$HOME/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFTER="$HOME/models/Qwen3-0.6B-BF16.gguf"

mkdir -p "$OUT"

# Acquire GPU lock (blocking)
exec 200>/tmp/dflash_gpu.lock
flock 200
echo "[task47] GPU lock acquired $(date)"

# Start server
GGML_CUDA_NO_VMM=1 \
DFLASH_DRAFTER_EARLY_EXIT_N=7 \
DFLASH_DRAFTER_SCORE_LAYERS=7 \
"$BIN" "$TARGET" \
  --host 127.0.0.1 --port $PORT --max-ctx 36864 \
  --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --prefill-compression always --prefill-keep-ratio 0.05 \
  --prefill-drafter "$DRAFTER" \
  --prefill-skip-park \
  >"$SERVER_LOG" 2>&1 &
SPID=$!
echo "[task47] server pid=$SPID"

# Wait up to 120s for listening or crash
STARTUP=fail
for i in $(seq 1 120); do
  sleep 1
  if ! kill -0 $SPID 2>/dev/null; then
    echo "[task47] server died at ${i}s"
    STARTUP=crash
    break
  fi
  if grep -qF '[server] listening' "$SERVER_LOG" 2>/dev/null; then
    echo "[task47] server listening after ${i}s"
    STARTUP=ok
    break
  fi
done

# Check crash signature
if grep -qE 'cuMemSetAccess|NOT_READY|CUDA_ERROR_NOT_READY' "$SERVER_LOG" 2>/dev/null; then
  echo "[task47] VERDICT C: cuMemSetAccess crash"
  kill $SPID 2>/dev/null; flock -u 200; exit 2
fi

if [ "$STARTUP" != "ok" ]; then
  echo "[task47] server did not start, checking log..."
  tail -30 "$SERVER_LOG"
  kill $SPID 2>/dev/null; flock -u 200; exit 1
fi

# VRAM monitor
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=1 >>"$VRAM_CSV" &
NMPID=$!

# Run 3 NIAH cases
PASS=0
DFWDS=""
IDX=0

while IFS= read -r line; do
  PROMPT=$(echo "$line" | jq -r '.prompt')
  EXPECTED=$(echo "$line" | jq -r '.answer')
  echo "[task47] case $IDX expected=$EXPECTED"

  PAYLOAD=$(jq -n --arg p "$PROMPT" \
    '{model:"local",prompt:$p,max_tokens:64,temperature:0}')

  RESP=$(curl -s --max-time 300 \
    -H 'Content-Type: application/json' \
    -d "$PAYLOAD" \
    "http://127.0.0.1:$PORT/v1/completions" || echo '{}')

  echo "$RESP" > "$OUT/case${IDX}.json"

  ANSWER=$(echo "$RESP" | jq -r '.choices[0].text // ""' 2>/dev/null | tr -d '\n')
  echo "[task47] case $IDX answer='$ANSWER'"

  PFIRST8="${EXPECTED:0:8}"
  if echo "$ANSWER" | grep -qF "$PFIRST8"; then
    PASS=$((PASS+1))
    echo "[task47] case $IDX PASS"
  else
    echo "[task47] case $IDX FAIL"
  fi

  # Extract last drafter_fwd from log
  DFW=$(grep -oP 'drafter_fwd=\K[0-9.]+' "$SERVER_LOG" | tail -1)
  DFWDS="$DFWDS $DFW"
  echo "[task47] drafter_fwd=${DFW}s"

  IDX=$((IDX+1))
done < "$NIAH"

# Stop monitors
kill $NMPID 2>/dev/null || true

PEAK_MIB=$(sort -n "$VRAM_CSV" 2>/dev/null | tail -1)
PEAK_GB=$(awk "BEGIN{printf \"%.1f\", ${PEAK_MIB:-0}/1024}")

# Mean drafter_fwd
MEAN_DFW=$(echo "$DFWDS" | tr ' ' '\n' | grep -v '^$' | \
  awk '{s+=$1;n++} END{if(n>0) printf "%.2f",s/n; else print "N/A"}')

kill $SPID 2>/dev/null || true
sleep 2
flock -u 200

echo "[task47] DONE: NIAH=$PASS/3, peak=${PEAK_GB}GB, mean_dfwd=${MEAN_DFW}s"

# Decide verdict
if [ "$PASS" -ge 2 ] && [ "$(echo "$PEAK_GB < 23.5" | bc -l)" = "1" ]; then
  VERDICT="(A) ee7 + skip-park works at 32K — recommend as opt-in config for <=32K workloads"
elif [ "$MEAN_DFW" = "N/A" ]; then
  VERDICT="(C) Server crashed mid-request — skip_park unsafe"
else
  VERDICT="(B) VRAM OOM or quality issue — keep park/unpark at 32K"
fi

# Baseline drafter_fwd at 32K was 1.44s
DELTA=$(awk "BEGIN{d=${MEAN_DFW}-1.44; printf \"%+.2fs (%+.0f%%)\", d, d/1.44*100}" 2>/dev/null || echo "N/A")

cat >"$OUT/SUMMARY.md" <<EOF
# ee7 + --prefill-skip-park 32K Experiment (Task #47)

Binary: d3fbad3 (layer-subset VRAM fix f157274 + guard bug fix d3fbad3)
GPU: NVIDIA GeForce RTX 3090 (24 GB)
Condition: ee7 (EARLY_EXIT_N=7, SCORE_LAYERS=7) + --prefill-skip-park + GGML_CUDA_NO_VMM=1
Context: 32768 tokens (reusing /tmp/niah_32768.jsonl, seeds 42/43/44, ~32764 tok each)
prefill-keep-ratio=0.05, tq3_0 KV (--cache-type-k/v tq3_0)

## Results

| Metric | Value |
|---|---|
| Server startup | OK |
| Cases completed | $IDX of 3 |
| NIAH retrieval | $PASS/3 |
| Peak VRAM | ${PEAK_GB} GB |
| Mean drafter_fwd | ${MEAN_DFW} s |
| ee7-with-park baseline (32K, d3fbad3) | 1.44 s |
| Delta vs baseline | ${DELTA} |

## Verdict

$VERDICT
EOF

cat "$OUT/SUMMARY.md"
echo "[task47] written $OUT/SUMMARY.md"
