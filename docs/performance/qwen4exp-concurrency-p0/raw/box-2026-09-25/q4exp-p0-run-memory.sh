#!/usr/bin/env bash
set -euo pipefail
exec 9>/tmp/qwen-perf/gpu.lock
flock 9
OUT=/tmp/q4exp-p0
MODEL=/home/duster/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf
BIN=/home/duster/lucebox-qwen4exp/server/build-hip/qwen4exp_p0_memory_probe
mkdir -p "$OUT"
state() {
  date -Is
  printf 'platform_profile='; cat /sys/firmware/acpi/platform_profile
  rocm-smi -d 1 --showclocks 2>&1 || true
  rocm-smi -d 1 --showpower 2>&1 || true
  rocm-smi -d 1 --showtemp 2>&1 || true
  rocm-smi -d 1 --showperflevel 2>&1 || true
  for k in sclk fclk mclk; do echo "pp_dpm_$k"; cat "/sys/class/drm/card2/device/pp_dpm_$k" 2>/dev/null || true; done
  cat /sys/class/drm/card2/device/mem_info_gtt_used 2>/dev/null | sed 's/^/gtt_used_bytes=/' || true
  grep '^MemAvailable:' /proc/meminfo
}
state > "$OUT/memory-power-before.txt"
printf 'env\n' > "$OUT/memory-env.txt"
export HIP_VISIBLE_DEVICES=1 DFLASH_HIP_NO_AUTO_UMA=1 LUCE_HIP_NO_AUTO_UMA=1
export GGML_CUDA_MMB=1 QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5
export DFLASH_MMB_SHADOW=1 LUCE_MMB_SHADOW=1 LLAMA_MMB_HC16=2
export QWEN4EXP_LAST_TOKEN_FFN=1 QWEN4EXP_DENSE_TABLE=1 QWEN4EXP_HC_TILE16=1
export QWEN4EXP_FA_PAD256=1 QWEN4EXP_DECODE_REUSE=1 QWEN4EXP_DECODE_STABLEGRAPH=1
tr '\0' '\n' </proc/$$/environ >> "$OUT/memory-env.txt"
"$BIN" "$MODEL" > "$OUT/memory.log" 2>&1 &
PID=$!
(
  while kill -0 "$PID" 2>/dev/null; do
    printf '%s,' "$(date +%s.%N)"
    cat /sys/class/drm/card2/device/mem_info_gtt_used 2>/dev/null || echo NA
    awk '/^MemAvailable:/ {print $2*1024}' /proc/meminfo
    awk '/^VmRSS:/ {print $2*1024}' "/proc/$PID/status" 2>/dev/null || echo NA
    cat "/proc/$PID/status" 2>/dev/null | awk '/^Name:/ {printf "%s,",$2}'
    echo
    sleep 0.25
  done
) > "$OUT/memory-trace.csv" &
MON=$!
set +e
wait "$PID"
RC=$?
set -e
wait "$MON" || true
printf 'probe_exit=%s\n' "$RC" >> "$OUT/memory.log"
state > "$OUT/memory-power-after.txt"
exit "$RC"
