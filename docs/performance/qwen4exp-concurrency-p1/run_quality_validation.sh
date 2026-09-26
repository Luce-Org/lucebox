#!/usr/bin/env bash
set -euo pipefail
ROOT=/tmp/qwen4exp-p1-src
BIN=/tmp/qwen4exp-p1-build/luce_server
MODEL=/home/duster/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf
OUT=/tmp/qwen4exp-p1-validation/quality
PORT=8878
export ROOT BIN MODEL OUT PORT
mkdir -p "$OUT"
guard() {
  test -z "$(pgrep -x luce_server || true)"
  test -z "$(pgrep -x dflash_server || true)"
  local gtt avail
  gtt=$(cat /sys/class/drm/card2/device/mem_info_gtt_used)
  avail=$(free -g | grep '^Mem:' | sed 's/^ *//' | tr -s ' ' | cut -d' ' -f7)
  test "$gtt" -lt 1073741824
  test "$avail" -ge 90
  echo "guard server_count=0 gtt_bytes=$gtt available_gib=$avail"
}
state() {
  date -Is
  printf 'platform_profile='; cat /sys/firmware/acpi/platform_profile
  for name in sclk fclk mclk; do
    echo "pp_dpm_$name"
    cat "/sys/class/drm/card2/device/pp_dpm_$name" 2>/dev/null || true
  done
  rocm-smi --showtemp 2>&1 || true
}
guard
flock -x /tmp/qwen-perf/gpu.lock bash -c 'set -euo pipefail
  guard() {
    test -z "$(pgrep -x luce_server || true)"
    test -z "$(pgrep -x dflash_server || true)"
    gtt=$(cat /sys/class/drm/card2/device/mem_info_gtt_used)
    avail=$(free -g | grep "^Mem:" | sed "s/^ *//" | tr -s " " | cut -d" " -f7)
    test "$gtt" -lt 1073741824 && test "$avail" -ge 90
    echo "locked_guard server_count=0 gtt_bytes=$gtt available_gib=$avail"
  }
  guard
  export HIP_VISIBLE_DEVICES=1 LUCE_HIP_NO_AUTO_UMA=1
  export GGML_CUDA_MMB=1 QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5 LUCE_MMB_SHADOW=1
  export LLAMA_MMB_HC16=2 QWEN4EXP_LAST_TOKEN_FFN=1 QWEN4EXP_DENSE_TABLE=1 QWEN4EXP_HC_TILE16=1 QWEN4EXP_FA_PAD256=1
  unset QWEN4EXP_BATCHED_DECODE
  state() { date -Is; printf "platform_profile="; cat /sys/firmware/acpi/platform_profile; for n in sclk fclk mclk; do echo "pp_dpm_$n"; cat "/sys/class/drm/card2/device/pp_dpm_$n" 2>/dev/null || true; done; rocm-smi --showtemp 2>&1 || true; }
  state > "$OUT/power_before.txt" 2>&1
  "$BIN" "$MODEL" --host 127.0.0.1 --port "$PORT" --target-device hip:0 --max-ctx 40000 --chunk 16384 > "$OUT/server.log" 2>&1 &
  pid=$!
  cleanup() { kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
  trap cleanup EXIT
  for _ in $(seq 1 240); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then tail -60 "$OUT/server.log"; exit 1; fi
    sleep 2
  done
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"
  tr "\0" "\n" < "/proc/$pid/environ" > "$OUT/server-environ.txt"
  (while kill -0 "$pid" 2>/dev/null; do state; sleep 15; done) > "$OUT/power_during.txt" 2>&1 &
  mon=$!
  set +e
  cd "$ROOT"
  python3 -u harness/client_test_runner.py bench --url "http://127.0.0.1:$PORT" --suite he,gsm,math,recall --model luce --json-out "$OUT/quality.json" > "$OUT/quality.stdout" 2>&1
  rc=$?
  set -e
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  state > "$OUT/power_after.txt" 2>&1
  tail -25 "$OUT/quality.stdout"
  exit "$rc"'
