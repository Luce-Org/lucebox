#!/usr/bin/env bash
# Task #47: ee7 + --prefill-skip-park empirical test at 32K
# Single condition, flock-protected GPU access

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
OUT_DIR="$REPO_ROOT/dflash/bench/results/2026-05-22_skip_park_32k"
NIAH_FILE="/tmp/niah_skip_park_32k.jsonl"
SERVER_LOG="$OUT_DIR/skip_park_server.log"
VRAM_LOG="/tmp/vram_skip_park_32k.csv"
SERVER_PORT=18100
SERVER_BIN="$REPO_ROOT/dflash/build/dflash_server"
TARGET_MODEL="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFTER_MODEL="/home/peppi/models/Qwen3-0.6B-BF16.gguf"

mkdir -p "$OUT_DIR"

# Clear stale VRAM log from a previous interrupted run.
: > "$VRAM_LOG"

echo "[task47] Generating NIAH cases (32K, n=3, seed=42)..."
python3 "$REPO_ROOT/pflash/tests/niah_gen.py" \
  --target-tokens 32768 --n 3 \
  --tokenizer Qwen/Qwen3.6-27B \
  --seed 42 \
  --out "$NIAH_FILE"
echo "[task47] NIAH generation done: $(wc -l < "$NIAH_FILE") cases"

echo "[task47] Waiting for GPU lock..."
exec 200>/tmp/dflash_gpu.lock
flock 200
echo "[task47] GPU lock acquired at $(date)"

echo "[task47] Starting server with --prefill-skip-park..."
GGML_CUDA_NO_VMM=1 \
DFLASH27B_KV_K=tq3_0 DFLASH27B_KV_V=tq3_0 \
DFLASH_DRAFTER_EARLY_EXIT_N=7 \
DFLASH_DRAFTER_SCORE_LAYERS=7 \
"$SERVER_BIN" "$TARGET_MODEL" \
  --host 127.0.0.1 --port $SERVER_PORT --max-ctx 36864 \
  --pflash-mode always --pflash-keep-ratio 0.05 \
  --pflash-drafter "$DRAFTER_MODEL" \
  --prefill-skip-park \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[task47] Server PID: $SERVER_PID"

# Wait for server to be ready (or crash)
echo "[task47] Waiting for server startup..."
STARTUP_OK=0
for i in $(seq 1 120); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[task47] SERVER CRASHED during startup (iteration $i)"
    break
  fi
  if grep -q "\[server\] listening" "$SERVER_LOG" 2>/dev/null; then
    echo "[task47] Server listening after ${i}s"
    STARTUP_OK=1
    break
  fi
  sleep 1
done

# Check for cuMemSetAccess crash signature
if grep -q "cuMemSetAccess\|NOT_READY\|CUDA_ERROR" "$SERVER_LOG" 2>/dev/null; then
  echo "[task47] VERDICT C: cuMemSetAccess NOT_READY crash detected"
  kill $SERVER_PID 2>/dev/null || true
  echo "VERDICT_C: cuMemSetAccess crash on startup" > "$OUT_DIR/VERDICT.txt"
  flock -u 200
  exit 1
fi

if [ "$STARTUP_OK" -eq 0 ]; then
  echo "[task47] Server failed to start in 120s"
  tail -50 "$SERVER_LOG"
  kill $SERVER_PID 2>/dev/null || true
  flock -u 200
  exit 1
fi

# Start VRAM monitoring
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=1 >> "$VRAM_LOG" &
NVMON_PID=$!
echo "[task47] VRAM monitor PID: $NVMON_PID"

# Run 3 NIAH cases
NIAH_PASS=0
DRAFTER_FWDS=()
CASE_IDX=0
CRASH_MID=0

while IFS= read -r line; do
  PROMPT=$(echo "$line" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['prompt'])")
  EXPECTED=$(echo "$line" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['answer'])")

  echo "[task47] Running case $CASE_IDX (expected: $EXPECTED)..."

  CASE_LOG="$OUT_DIR/case${CASE_IDX}_response.json"
  RESP=$(curl -s --max-time 300 \
    http://127.0.0.1:$SERVER_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json, sys
prompt = sys.stdin.read()
print(json.dumps({'model': 'gpt-3.5-turbo', 'prompt': prompt, 'max_tokens': 64, 'temperature': 0}))
" <<< "$PROMPT")" 2>/dev/null || echo '{}')

  echo "$RESP" > "$CASE_LOG"

  # Check if server is still alive
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[task47] SERVER CRASHED during case $CASE_IDX"
    CRASH_MID=1
    break
  fi

  # Extract answer and check NIAH
  ANSWER=$(echo "$RESP" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    text = d.get('choices', [{}])[0].get('text', '')
    print(text.strip())
except:
    print('')
" 2>/dev/null || echo "")

  echo "[task47] Case $CASE_IDX answer: '$ANSWER' (expected: '$EXPECTED')"
  if echo "$ANSWER" | grep -qi "$(echo "$EXPECTED" | head -c 8)"; then
    NIAH_PASS=$((NIAH_PASS + 1))
    echo "[task47] Case $CASE_IDX: PASS"
  else
    echo "[task47] Case $CASE_IDX: FAIL"
  fi

  # Extract drafter_fwd from server log for this case
  DRAFTER_FWD=$(grep "drafter_fwd" "$SERVER_LOG" | tail -5 | grep -oP 'drafter_fwd=\K[0-9.]+' | tail -1 || echo "N/A")
  DRAFTER_FWDS+=("$DRAFTER_FWD")
  echo "[task47] Case $CASE_IDX drafter_fwd: ${DRAFTER_FWD}s"

  CASE_IDX=$((CASE_IDX + 1))
done < "$NIAH_FILE"

# Stop VRAM monitor
kill $NVMON_PID 2>/dev/null || true

# Get peak VRAM
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" 2>/dev/null | tail -1 || echo "0")
PEAK_VRAM_GB=$(python3 -c "print(f'{${PEAK_VRAM_MIB:-0}/1024:.1f}')" 2>/dev/null || echo "N/A")

# Compute mean drafter_fwd
MEAN_DFW=$(python3 -c "
vals = [float(x) for x in '${DRAFTER_FWDS[*]}'.split() if x not in ('N/A', '')]
print(f'{sum(vals)/len(vals):.2f}' if vals else 'N/A')
" 2>/dev/null || echo "N/A")

# Stop server
kill $SERVER_PID 2>/dev/null || true
sleep 2
flock -u 200

echo "[task47] Server stopped"
echo "[task47] Results: NIAH $NIAH_PASS/3, peak VRAM: ${PEAK_VRAM_GB} GB, mean drafter_fwd: ${MEAN_DFW}s"

# Write summary
cat > "$OUT_DIR/SUMMARY.md" << SUMMARY_EOF
# ee7 + --prefill-skip-park 32K Experiment (Task #47)

Binary: d3fbad3 (layer-subset VRAM fix f157274 + guard bug fix d3fbad3)
GPU: NVIDIA GeForce RTX 3090 (24 GB)
Condition: ee7 (EARLY_EXIT_N=7, SCORE_LAYERS=7) + --prefill-skip-park + GGML_CUDA_NO_VMM=1
Context: 32768 tokens, pflash-keep-ratio=0.05, tq3_0 KV

## Results

| Metric | Value |
|---|---|
| Server startup | $([ "$STARTUP_OK" -eq 1 ] && echo "OK" || echo "FAILED") |
| Cases completed | $CASE_IDX of 3 |
| NIAH retrieval | $NIAH_PASS/3 |
| Peak VRAM | ${PEAK_VRAM_GB} GB |
| Mean drafter_fwd | ${MEAN_DFW} s |
| Mid-request crash | $([ "$CRASH_MID" -eq 1 ] && echo "YES" || echo "NO") |
| ee7-with-park baseline (32K, d3fbad3) | 1.44 s |
| Delta vs baseline | $(python3 -c "
m = '${MEAN_DFW}'
try:
    d = float(m) - 1.44
    pct = d / 1.44 * 100
    sign = '+' if d > 0 else ''
    print(f'{sign}{d:.2f}s ({sign}{pct:.0f}%)')
except:
    print('N/A')
" 2>/dev/null || echo "N/A") |

## Drafter fwd per case
$(for i in "${!DRAFTER_FWDS[@]}"; do echo "- Case $i: ${DRAFTER_FWDS[$i]} s"; done)

## Verdict

$(if [ "$STARTUP_OK" -eq 0 ] || grep -q "cuMemSetAccess\|NOT_READY" "$SERVER_LOG" 2>/dev/null; then
  echo "(C) cuMemSetAccess NOT_READY crash — historical skip_park bug recurs, choreography stays"
elif [ "$CRASH_MID" -eq 1 ]; then
  echo "(C) Server crashed mid-request — skip_park unsafe at 32K with ee7"
elif [ "$NIAH_PASS" -lt 2 ] || python3 -c "v='${PEAK_VRAM_GB}'; exit(0 if v == 'N/A' or float(v) >= 23 else 1)" 2>/dev/null; then
  echo "(B) VRAM OOM or quality degradation — keep park/unpark at 32K"
else
  echo "(A) ee7 + skip-park works at 32K — recommend as opt-in config for ≤32K workloads"
fi)

## Server Log Tail (last 30 lines)
\`\`\`
$(tail -30 "$SERVER_LOG" 2>/dev/null || echo "(no log)")
\`\`\`
SUMMARY_EOF

echo "[task47] SUMMARY written to $OUT_DIR/SUMMARY.md"
cat "$OUT_DIR/SUMMARY.md"
