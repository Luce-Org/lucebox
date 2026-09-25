#!/usr/bin/env bash
set -euo pipefail
label=${1:?label required}
bin=${2:?probe binary required}
model=${3:?IQ4_NL shard required}
out=${4:?output directory required}
mkdir -p "$out"
{
    date -Is
    printf 'platform_profile='; cat /sys/firmware/acpi/platform_profile 2>/dev/null || true
    for f in /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_fclk /sys/class/drm/card*/device/pp_dpm_mclk; do
        [ -r "$f" ] && { echo "[$f]"; cat "$f"; }
    done
    rocm-smi --showtemp 2>&1 || true
} > "$out/$label-power-before.txt"
export HIP_VISIBLE_DEVICES=1 LUCE_HIP_NO_AUTO_UMA=1
export GGML_CUDA_MMB=1 QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5
export LUCE_MMB_SHADOW=1 LLAMA_MMB_HC16=2 QWEN4EXP_LAST_TOKEN_FFN=1
export QWEN4EXP_DENSE_TABLE=1 QWEN4EXP_HC_TILE16=1 QWEN4EXP_FA_PAD256=1
export QWEN4EXP_BATCHED_DECODE=1
"$bin" "$model" 16 32768 > "$out/$label.log" 2>&1 &
pid=$!
tr '\0' '\n' < "/proc/$pid/environ" > "$out/$label-environ.txt"
while kill -0 "$pid" 2>/dev/null; do
    {
        date -Is
        rocm-smi --showtemp 2>&1 || true
    } >> "$out/$label-power-during.txt"
    sleep 15
done
set +e
wait "$pid"
rc=$?
set -e
{
    date -Is
    printf 'platform_profile='; cat /sys/firmware/acpi/platform_profile 2>/dev/null || true
    for f in /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_fclk /sys/class/drm/card*/device/pp_dpm_mclk; do
        [ -r "$f" ] && { echo "[$f]"; cat "$f"; }
    done
    rocm-smi --showtemp 2>&1 || true
} > "$out/$label-power-after.txt"
echo "$rc" > "$out/$label-exit.txt"
exit "$rc"
