#!/usr/bin/env bash
# KV cache precision sweep: prefill / decode / accept(AL) at f16/q8_0/q4_0/tq3_0.
cd /home/peppi/Dev/lucebox-hub
OUT=bench/qwen35moe_dflash/ctxsweep/kvsweep.txt; : > "$OUT"
PROMPT=bench/qwen35moe_dflash/ctxsweep/ctx_032768.json
TGT=/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
DD=/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf
for KV in f16 q8_0 q4_0 tq3_0; do
  LOG=bench/qwen35moe_dflash/ctxsweep/srv_kv_${KV}.log
  DFLASH_DRAFT_CTX_MAX=2048 DFLASH_FEAT_RING_CAP=65536 server/build/dflash_server \
    "$TGT" --draft "$DD" --host 127.0.0.1 --port 18080 --max-ctx 40960 --max-tokens 200 \
    --fa-window 0 --cache-type-k "$KV" --cache-type-v "$KV" --model-name luce-dflash --lazy-draft > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 150); do curl -fsS http://127.0.0.1:18080/health >/dev/null 2>&1 && { ok=1; break; }; sleep 1; done
  if [ "$ok" != 1 ]; then
    echo "KV=$KV FAILED: $(grep -oiE 'unsupported[^\n]*|error[^\n]*|abort[^\n]*' "$LOG" | head -1)" | tee -a "$OUT"
    kill -9 $PID 2>/dev/null; sleep 2; continue
  fi
  kv_gib=$(grep -oE 'kv_cache=[0-9.]+ GiB' "$LOG" | head -1)
  curl -s http://127.0.0.1:18080/v1/chat/completions -H 'content-type: application/json' -d @"$PROMPT" -o /dev/null
  line=$(grep -oE 'prefill=[0-9.]+s decode=[0-9.]+s\([0-9.]+tok/s\)' "$LOG" | tail -1)
  acc=$(grep -oE 'accepted=[0-9]+/[0-9]+ \([0-9.]+%\) avg_commit=[0-9.]+' "$LOG" | tail -1)
  echo "KV=$KV | $kv_gib | $line | $acc" | tee -a "$OUT"
  kill -9 $PID 2>/dev/null; sleep 3
done
echo "KVSWEEP DONE" | tee -a "$OUT"
