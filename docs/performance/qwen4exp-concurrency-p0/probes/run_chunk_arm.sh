#!/usr/bin/env bash
set -euo pipefail
BIN=$1 MODEL=$2 CHUNK=$3 PORT=$4 OUT=$5 PY=$6
mkdir -p "$OUT"
state() {
  date -Is
  printf 'platform_profile='; cat /sys/firmware/acpi/platform_profile
  rocm-smi -d 1 --showperflevel --showclocks --showpower --showtemp --showmeminfo gtt || true
  for k in sclk fclk mclk; do echo "pp_dpm_$k"; cat "/sys/class/drm/card2/device/pp_dpm_$k" 2>/dev/null || true; done
}
state > "$OUT/power_before.txt" 2>&1
export HIP_VISIBLE_DEVICES=1 LUCE_HIP_NO_AUTO_UMA=1 DFLASH_HIP_NO_AUTO_UMA=1 QWEN4EXP_FA_PAD256=1
export GGML_CUDA_MMB=1 QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5
export LUCE_MMB_SHADOW=1 LLAMA_MMB_HC16=2 QWEN4EXP_DENSE_TABLE=1
export QWEN4EXP_HC_TILE16=1 QWEN4EXP_LAST_TOKEN_FFN=1
"$BIN" "$MODEL" --host 127.0.0.1 --port "$PORT" --target-device hip:0 \
  --max-ctx 32768 --chunk "$CHUNK" > "$OUT/server.log" 2>&1 &
PID=$!
MON=
cleanup() {
  if [ -n "$MON" ]; then kill "$MON" 2>/dev/null || true; fi
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}
trap cleanup EXIT
for i in $(seq 1 120); do
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then break; fi
  if ! kill -0 "$PID" 2>/dev/null; then tail -n 35 "$OUT/server.log"; exit 1; fi
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"
tr '\0' '\n' < "/proc/$PID/environ" | grep -E '^(HIP_VISIBLE_DEVICES|LUCE_HIP_NO_AUTO_UMA|QWEN4EXP_FA_PAD256|QWEN4EXP_QSA|QWEN4EXP_MMB_CUBLAS|LUCE_MMB_SHADOW|GGML_CUDA_MMB|LLAMA_MMB_HC16|QWEN4EXP_DENSE_TABLE|QWEN4EXP_HC_TILE16|QWEN4EXP_LAST_TOKEN_FFN)=' > "$OUT/server_env.txt"
(while kill -0 "$PID" 2>/dev/null; do state; sleep 2; done) > "$OUT/power_during.txt" 2>&1 &
MON=$!
python3 "$PY" "http://127.0.0.1:$PORT" "$OUT/chunk.json" 29900 > "$OUT/chunk.stdout" 2>&1
kill "$MON" 2>/dev/null || true
MON=
state > "$OUT/power_after.txt" 2>&1
tail -n 8 "$OUT/server.log" > "$OUT/server_tail.txt"
cat "$OUT/chunk.stdout"
cleanup
trap - EXIT
