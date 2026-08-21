#!/usr/bin/env bash
# Launch the slim PR#614 build (DeepSeek-V4-0731 + DSpark on R9700+Strix) with
# model-emitted early dispatch and the public-REST executor on the reserved CPU
# lane. The superseded native Qwen predictor is intentionally not part of this
# path.
set -euo pipefail
root="/home/lucebox5"
experiment="$root/tool-spec-cpu-20260813"
tb="$root/tbspec"
launcher="$experiment/run-deepseek-0731-cpu-tool.sh"
executor="${TBSPEC_EXECUTOR:-$tb/tb_tool_executor.py}"
profile="$root/codex-pr614-simplify-20260818/optimizations/ooo_spec_lucebox5_cpu/profiles/lucebox5-cpu-lane-qualified.json"
allowed="${TBSPEC_ALLOW:-geocode_city,get_weather,country_info,wikipedia_summary,exchange_rate}"
native_wrapper_dir="$tb/wrapper"
candidate_build="$(readlink -f "$native_wrapper_dir/candidate-build")"
for f in "$launcher" "$executor" "$native_wrapper_dir/dflash_server" "$candidate_build/dflash_server" "$candidate_build/backend_ipc_daemon"; do
  [[ -x "$f" ]] || { echo "not executable: $f" >&2; exit 2; }
done
[[ -f "$profile" ]] || { echo "missing data file: $profile" >&2; exit 2; }
if pgrep -x dflash_server >/dev/null; then echo "dflash_server already running" >&2; exit 75; fi
if [[ -e /dev/kfd ]]; then
  command -v fuser >/dev/null || { echo "cannot verify /dev/kfd ownership: fuser missing" >&2; exit 2; }
  if fuser -s /dev/kfd; then echo "/dev/kfd is already in use" >&2; exit 75; fi
fi
exec env -i \
  HOME="$root" USER="lucebox5" \
  PATH="$root/.local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  ENGINE_DIR="$root/lucebox-engine-0731" \
  BUILD_DIR="$native_wrapper_dir" \
  QUALIFIED_CONFIG_DIR="/opt/lucebox-manage/qualified/r9700_deepseek/runtime-config" \
  TARGET_MODEL="$root/lucebox-models/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf" \
  DRAFT_MODEL="$root/lucebox-models/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf" \
  SERVER_PORT="18145" \
  MODEL_CPU_AFFINITY="0-13,16-29" \
  TOOL_SPEC_EXECUTOR="$executor" \
  TOOL_SPEC_PROFILE="$profile" \
  TOOL_SPEC_ALLOW="$allowed" \
  TOOL_SPEC_CPU_AFFINITY="14-15,30-31" \
  TOOL_SPEC_MAX_MODEL_SLOWDOWN="1.05" \
  LUCEBOX_INFERENCE_PROFILE="${TBSPEC_PROFILE:-quality}" \
  "$launcher"
