#!/usr/bin/env bash
# Paired, fresh-process Qwen3.6 concurrency benchmark for Lucebox and llama.cpp.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}"
CLIENT="${CLIENT:-$SCRIPT_DIR/concurrent_benchmark.py}"
GENERATOR="${GENERATOR:-$SCRIPT_DIR/generate_prompts.py}"
SUMMARIZER="${SUMMARIZER:-$SCRIPT_DIR/summarize_concurrency.py}"

MODEL="${MODEL:-}"
LUCE_SERVER_BIN="${LUCE_SERVER_BIN:-$REPO/server/build-hip/dflash_server}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$(command -v llama-server 2>/dev/null || true)}"
OUT="${OUT:-$REPO/.harness-runs/qwen36-concurrency-$(date -u +%Y%m%dT%H%M%SZ)}"
REPEATS="${REPEATS:-1}"
WORKLOADS="${WORKLOADS:-short,medium,long}"
VARIANTS="${VARIANTS:-luce-k8,luce-k1,llama}"
CLIENTS="${CLIENTS:-2,4,8,16}"
SLOTS="${SLOTS:-}"
GPU_DEVICE="${GPU_DEVICE:-0}"
EXPECTED_GPU_ARCH="${EXPECTED_GPU_ARCH:-}"
PORT="${PORT:-18114}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-3}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-600}"
MAX_TOKENS="${MAX_TOKENS:-64}"
WARMUP_TOKENS="${WARMUP_TOKENS:-8}"
PREFILL_FIRST_BURST_STEPS="${PREFILL_FIRST_BURST_STEPS:-0}"
IDLE_PREFILL_TOKENS="${IDLE_PREFILL_TOKENS:-4096}"

usage() {
  cat <<'EOF'
Usage: MODEL=/path/model.gguf [REPEATS=5] run_qwen36_concurrency.sh

Runs fresh-server, same-concurrency warmup + measurement cases for luce-k8,
luce-k1, and llama at C=2/4/8/16. Defaults to one repeat for screening; use at
least five paired repeats for publication. For a decode-heavy comparison, set
WORKLOADS=short MAX_TOKENS=256 VARIANTS=luce-k8,llama. OUT must not already
exist. GPU_DEVICE is the physical ROCr device exposed exclusively to both
servers and defaults to device 0. Pass GPU_DEVICE=1 for Strix Halo on this
dual-GPU benchmark host. CLIENTS can contain any distinct positive levels;
SLOTS defaults to at least 16 and grows to the largest requested level. Set
EXPECTED_GPU_ARCH (for example gfx1151) to fail closed unless isolated
rocminfo identifies that architecture. After health, Luce must report that
architecture in its own log. llama.cpp must expose a ROCm device through this
exact binary's --list-devices output and report a positive full GPU offload
(offloaded N/N layers). The runner uses -lv 4 so revision 4cb22cd emits those
startup proof lines. Evidence for every check is retained in the case dir.
PREFILL_FIRST_BURST_STEPS forwards the experimental Lucebox TTFT policy; 0
preserves continuous decode, while positive N permits at most N consecutive
prefill-only traversals before a mandatory decode traversal.
IDLE_PREFILL_TOKENS forwards Luce's idle/prefill-only token budget and defaults
to 4096. Valid values are 1..16384, the current maximum effective K8 pure
prefill batch (8 lanes x 2048 tokens); llama.cpp does not receive this setting.
EOF
}

if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -ne 0 ]]; then usage >&2; exit 2; fi
for cmd in python3 curl sha256sum; do command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 2; }; done
[[ -r "$MODEL" ]] || { echo "set MODEL to a readable GGUF" >&2; exit 2; }
[[ -x "$LUCE_SERVER_BIN" ]] || { echo "missing Lucebox server: $LUCE_SERVER_BIN" >&2; exit 2; }
[[ -x "$LLAMA_SERVER_BIN" ]] || { echo "missing llama.cpp server: $LLAMA_SERVER_BIN" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 2; }
[[ -z "$EXPECTED_GPU_ARCH" || "$EXPECTED_GPU_ARCH" =~ ^gfx[0-9a-f]+$ ]] || { echo "EXPECTED_GPU_ARCH must be empty or a gfx architecture" >&2; exit 2; }
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] || { echo "GPU_DEVICE must be a physical ROCr device index" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
if ! [[ "$PREFILL_FIRST_BURST_STEPS" =~ ^[0-9]{1,4}$ ]] || (( 10#$PREFILL_FIRST_BURST_STEPS > 1024 )); then
  echo "PREFILL_FIRST_BURST_STEPS must be an integer in range 0..1024" >&2
  exit 2
fi
if ! [[ "$IDLE_PREFILL_TOKENS" =~ ^[1-9][0-9]{0,4}$ ]] || (( 10#$IDLE_PREFILL_TOKENS > 16384 )); then
  echo "IDLE_PREFILL_TOKENS must be an integer in range 1..16384" >&2
  exit 2
fi
ambient_tuning="$(env | grep -E '^(GGML_|DFLASH_|LUCE_|HIP_|ROCR_|HSA_|LD_PRELOAD=|LD_LIBRARY_PATH=)' \
  | grep -v '^LUCE_SERVER_BIN=' || true)"
if [[ -n "$ambient_tuning" ]]; then
  echo "refusing ambient GPU/backend tuning variables:" >&2
  echo "$ambient_tuning" >&2
  exit 2
fi
MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"

IFS=, read -r -a workload_list <<< "$WORKLOADS"
IFS=, read -r -a variant_list <<< "$VARIANTS"
IFS=, read -r -a client_list <<< "$CLIENTS"
declare -A prompt_offsets=()
declare -A seen_clients=()
next_prompt_offset=0
max_clients=0
for c in "${client_list[@]}"; do
  [[ "$c" =~ ^[1-9][0-9]*$ ]] || { echo "CLIENTS must contain positive integers" >&2; exit 2; }
  [[ -z "${seen_clients[$c]+yes}" ]] || { echo "CLIENTS levels must be distinct" >&2; exit 2; }
  seen_clients[$c]=1
  prompt_offsets[$c]="$next_prompt_offset"
  next_prompt_offset=$((next_prompt_offset + c))
  if (( c > max_clients )); then
    max_clients="$c"
  fi
done
if [[ -z "$SLOTS" ]]; then
  SLOTS=16
  if (( max_clients > SLOTS )); then
    SLOTS="$max_clients"
  fi
fi
[[ "$SLOTS" =~ ^[1-9][0-9]*$ ]] || { echo "SLOTS must be positive" >&2; exit 2; }
(( SLOTS >= max_clients )) || { echo "SLOTS must be at least the largest CLIENTS level" >&2; exit 2; }
for v in "${variant_list[@]}"; do
  [[ "$v" == luce-k8 || "$v" == luce-k1 || "$v" == llama ]] || { echo "unknown variant $v" >&2; exit 2; }
done

mkdir -p "$OUT/prompts"
for workload in "${workload_list[@]}"; do
  python3 "$GENERATOR" --profile "$workload" --clients "$CLIENTS" \
    --out "$OUT/prompts/$workload.jsonl"
done

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT
trap 'exit 130' INT TERM

wait_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    kill -0 "$server_pid" 2>/dev/null || return 1
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

write_metadata() {
  local path="$1" variant="$2" workload="$3" clients="$4" repeat="$5" binary="$6" max_prefills="$7" command_file="$8"
  local prompt_offset="$9" client_levels="${10}" slots="${11}" rocr_visible_devices="${12}" prefill_first_burst_steps="${13}" expected_gpu_arch="${14}" idle_prefill_tokens="${15}"
  python3 -c 'import hashlib,json,pathlib,subprocess,sys
p,variant,workload,clients,repeat,binary,max_prefills,cmd_file,model_sha,prompts,repo,prompt_offset,client_levels,slots,rocr_visible,prefill_first_burst_steps,expected_gpu_arch,idle_prefill_tokens=sys.argv[1:]
digest=lambda x: hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest()
libs={}
for line in subprocess.run(["ldd",binary],text=True,capture_output=True).stdout.splitlines():
    fields=line.replace("=>"," ").split()
    paths=[x for x in fields if x.startswith("/") and pathlib.Path(x).is_file()]
    for lib in paths: libs[str(pathlib.Path(lib).resolve())]=digest(lib)
lucebox_git_head=subprocess.run(["git","-C",repo,"rev-parse","HEAD"],text=True,capture_output=True).stdout.strip() or None
server_version=None
if variant == "llama":
    version=subprocess.run([binary,"--version"],text=True,capture_output=True,timeout=30)
    server_version="\n".join(x.strip() for x in (version.stdout,version.stderr) if x.strip()) or None
    if version.returncode != 0 or server_version is None:
        raise RuntimeError(f"cannot identify llama.cpp source version from {binary} --version")
obj={"variant":variant,"workload":workload,"clients":int(clients),"repeat":int(repeat),
     "max_concurrent_prefills":int(max_prefills),"server_binary":str(pathlib.Path(binary).resolve()),
     "server_binary_sha256":digest(binary),"model_sha256":model_sha,
     "prompt_file_sha256":digest(prompts),"server_command":pathlib.Path(cmd_file).read_text().strip(),
     "client_levels":[int(x) for x in client_levels.split(",")],
     "prompt_offset":int(prompt_offset),"server_slots":int(slots),
     "rocr_visible_devices":rocr_visible,
     "expected_gpu_arch":expected_gpu_arch or None,
     "resolved_shared_library_sha256":libs,
     "prefill_first_burst_steps":int(prefill_first_burst_steps) if variant != "llama" else None,
     "idle_prefill_tokens":int(idle_prefill_tokens) if variant != "llama" else None,
     "lucebox_git_head":lucebox_git_head if variant != "llama" else None,
     "server_version":server_version}
pathlib.Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")' \
    "$path" "$variant" "$workload" "$clients" "$repeat" "$binary" "$max_prefills" "$command_file" \
    "$MODEL_SHA256" "$OUT/prompts/$workload.jsonl" "$REPO" "$prompt_offset" "$client_levels" "$slots" "$rocr_visible_devices" "$prefill_first_burst_steps" "$expected_gpu_arch" "$idle_prefill_tokens"
}

run_case() {
  local repeat="$1" workload="$2" clients="$3" variant="$4"
  local max_ctx timeout capacity max_prefills binary model_id
  if [[ "$workload" == long ]]; then
    max_ctx=8192; timeout=1800
  else
    max_ctx=4096; timeout=1200
  fi
  capacity=$((SLOTS * max_ctx))
  local case_dir="$OUT/$workload/c$clients/r$repeat/$variant"
  mkdir -p "$case_dir"
  if [[ "$variant" == llama ]]; then
    binary="$LLAMA_SERVER_BIN"; model_id=qwen36-llama; max_prefills=0
  else
    binary="$LUCE_SERVER_BIN"; model_id=qwen36-luce
    [[ "$variant" == luce-k8 ]] && max_prefills=8 || max_prefills=1
  fi
  if [[ -n "$EXPECTED_GPU_ARCH" ]]; then
    if ! command -v rocminfo >/dev/null; then
      echo "EXPECTED_GPU_ARCH requires rocminfo" >&2
      return 1
    fi
    if ! env ROCR_VISIBLE_DEVICES="$GPU_DEVICE" rocminfo 2>/dev/null | grep -F -- "$EXPECTED_GPU_ARCH" > "$case_dir/gpu-identity.txt"; then
      echo "isolated ROCr device did not report expected architecture $EXPECTED_GPU_ARCH" >&2
      return 1
    fi
  fi
  if [[ "$variant" == llama ]]; then
    printf '%q ' env ROCR_VISIBLE_DEVICES="$GPU_DEVICE" "$binary" --list-devices \
      > "$case_dir/llama-list-devices-command.txt"
    printf '\n' >> "$case_dir/llama-list-devices-command.txt"
    if ! env ROCR_VISIBLE_DEVICES="$GPU_DEVICE" "$binary" --list-devices \
      > "$case_dir/llama-list-devices.txt" 2>&1; then
      echo "llama.cpp --list-devices failed under isolated ROCr device $GPU_DEVICE" >&2
      return 1
    fi
    if ! grep -Eq '^[[:space:]]*ROCm[0-9]+:[[:space:]]+[^[:space:]]' \
      "$case_dir/llama-list-devices.txt"; then
      echo "llama.cpp did not expose a ROCm device under isolated ROCr device $GPU_DEVICE" >&2
      return 1
    fi
  fi
  local -a command launch_command
  if [[ "$variant" == llama ]]; then
    command=("$binary" -m "$MODEL" -ngl all -lv 4 --reasoning off --reasoning-format none
      --parallel "$SLOTS" -c "$capacity"
      -b 2048 -ub 512 --cont-batching --no-context-shift --no-mmap -fa on
      -ctk q4_0 -ctv q4_0 --no-cache-prompt --host 127.0.0.1 --port "$PORT" --alias "$model_id")
  else
    command=("$binary" "$MODEL" --target-device hip:0 --paged-attention
      --max-concurrency "$SLOTS" --kv-pool-tokens "$capacity" --max-ctx "$max_ctx"
      --cache-type-k q4_0 --cache-type-v q4_0 --fa-window 0
      --prefix-cache-slots 0 --prefill-cache-slots 0 --admission-coalesce-ms 5
      --host 127.0.0.1 --port "$PORT" --model-name "$model_id")
  fi
  if [[ "$variant" == llama ]]; then
    launch_command=(env ROCR_VISIBLE_DEVICES="$GPU_DEVICE" "${command[@]}")
  else
    launch_command=(env ROCR_VISIBLE_DEVICES="$GPU_DEVICE" DFLASH_IGNORE_EOS=1
      DFLASH_MIN_TOKENS="$WARMUP_TOKENS"
      DFLASH_PREFILL_FIRST_BURST_STEPS="$PREFILL_FIRST_BURST_STEPS"
      DFLASH_IDLE_PREFILL_TOKENS="$IDLE_PREFILL_TOKENS"
      DFLASH_MAX_CONCURRENT_PREFILLS="$max_prefills" "${command[@]}")
  fi
  printf '%q ' "${launch_command[@]}" > "$case_dir/server-command.txt"; printf '\n' >> "$case_dir/server-command.txt"
  local offset="${prompt_offsets[$clients]}"
  write_metadata "$case_dir/server-metadata.json" "$variant" "$workload" "$clients" "$repeat" "$binary" "$max_prefills" \
    "$case_dir/server-command.txt" "$offset" "$CLIENTS" "$SLOTS" "$GPU_DEVICE" "$PREFILL_FIRST_BURST_STEPS" "$EXPECTED_GPU_ARCH" "$IDLE_PREFILL_TOKENS"

  echo "[run] $workload C=$clients repeat=$repeat variant=$variant"
  "${launch_command[@]}" > "$case_dir/server.log" 2>&1 &
  server_pid=$!
  if ! wait_health; then
    tail -n 80 "$case_dir/server.log" >&2 || true
    stop_server
    sleep "$COOLDOWN_SECONDS"
    return 1
  fi

  if [[ "$variant" == llama ]]; then
    if ! python3 -c 'import pathlib,re,sys
source=pathlib.Path(sys.argv[1])
matches=[]
for line in source.read_text(encoding="utf-8",errors="replace").splitlines():
    match=re.search(r"\boffloaded ([1-9][0-9]*)/([1-9][0-9]*) layers to GPU\b",line)
    if match and match.group(1) == match.group(2): matches.append(line)
if not matches: raise SystemExit(1)
pathlib.Path(sys.argv[2]).write_text("\n".join(matches)+"\n",encoding="utf-8")' \
      "$case_dir/server.log" "$case_dir/server-gpu-proof.txt"; then
      echo "llama.cpp server did not report a positive full GPU offload" >&2
      tail -n 80 "$case_dir/server.log" >&2 || true
      stop_server
      sleep "$COOLDOWN_SECONDS"
      return 1
    fi
  elif [[ -n "$EXPECTED_GPU_ARCH" ]]; then
    if ! grep -F -- "$EXPECTED_GPU_ARCH" "$case_dir/server.log" \
      > "$case_dir/server-gpu-proof.txt"; then
      echo "Lucebox server did not report expected GPU architecture $EXPECTED_GPU_ARCH" >&2
      tail -n 80 "$case_dir/server.log" >&2 || true
      stop_server
      sleep "$COOLDOWN_SECONDS"
      return 1
    fi
  fi


  local prompts="$OUT/prompts/$workload.jsonl"
  local status=0
  if ! python3 "$CLIENT" --base-url "http://127.0.0.1:$PORT/v1" --model "$model_id" \
    --clients "$clients" --prompt-file "$prompts" --prompt-offset "$offset" \
    --require-distinct-prompts --max-tokens "$WARMUP_TOKENS" --temperature 0 \
    --ignore-eos --timeout "$timeout" --cooldown 0 --out "$case_dir/warmup.json" \
    --label "$variant $workload C=$clients warmup" > "$case_dir/warmup.txt"; then
    status=1
  fi
  if (( status == 0 )) && ! python3 "$CLIENT" --base-url "http://127.0.0.1:$PORT/v1" --model "$model_id" \
    --clients "$clients" --prompt-file "$prompts" --prompt-offset "$offset" \
    --require-distinct-prompts --max-tokens "$MAX_TOKENS" --temperature 0 \
    --ignore-eos --timeout "$timeout" --cooldown 0 \
    --server-metadata-json "$case_dir/server-metadata.json" --out "$case_dir/bench.json" \
    --label "$variant $workload C=$clients repeat=$repeat" | tee "$case_dir/bench.txt"; then
    status=1
  fi
  stop_server
  sleep "$COOLDOWN_SECONDS"
  return "$status"
}

case_failures=0
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  for workload in "${workload_list[@]}"; do
    for c_index in "${!client_list[@]}"; do
      clients="${client_list[$c_index]}"
      # Rotate start variant by case so one engine is not always hot or cold.
      shift_by=$(((repeat + c_index) % ${#variant_list[@]}))
      for ((i=0; i<${#variant_list[@]}; i++)); do
        variant="${variant_list[$(((i + shift_by) % ${#variant_list[@]}))]}"
        if ! run_case "$repeat" "$workload" "$clients" "$variant"; then
          echo "[run] failed: $workload C=$clients repeat=$repeat variant=$variant" >&2
          case_failures=$((case_failures + 1))
        fi
      done
    done
  done
done

summary_status=0
python3 "$SUMMARIZER" "$OUT" --out "$OUT/summary.md" || summary_status=$?
if (( case_failures > 0 || summary_status != 0 )); then
  echo "[run] completed with $case_failures failed case(s)" >&2
  exit 1
fi
echo "[run] complete: $OUT"
