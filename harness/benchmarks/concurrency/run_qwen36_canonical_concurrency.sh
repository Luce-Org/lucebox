#!/usr/bin/env bash
# Canonical repository suites under fixed-concurrency waves, including blog parity.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}"
CLIENT="${CLIENT:-$SCRIPT_DIR/canonical_concurrent_benchmark.py}"
GENERATOR="${GENERATOR:-$SCRIPT_DIR/generate_prompts.py}"
SUMMARIZER="${SUMMARIZER:-$SCRIPT_DIR/summarize_concurrency.py}"
MODEL="${MODEL:-}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
SERVER_BIN="${SERVER_BIN:-$REPO/server/build-hip/dflash_server}"
OUT="${OUT:-$REPO/.harness-runs/qwen36-canonical-$(date -u +%Y%m%dT%H%M%SZ)}"
SUITES="${SUITES:-he-raw,he,gsm,math,agent}"
VARIANTS="${VARIANTS:-ar}"
REPEATS="${REPEATS:-1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP_TOKENS="${WARMUP_TOKENS:-8}"
CLIENTS="${CLIENTS:-}"
CASE_LIMIT="${CASE_LIMIT:-}"
SLOTS="${SLOTS:-16}"
GPU_DEVICE="${GPU_DEVICE:-0}"
EXPECTED_GPU_ARCH="${EXPECTED_GPU_ARCH:-}"
PORT="${PORT:-18116}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-600}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-3}"
PREFILL_FIRST_BURST_STEPS="${PREFILL_FIRST_BURST_STEPS:-0}"
IDLE_PREFILL_TOKENS="${IDLE_PREFILL_TOKENS:-4096}"

usage() {
  cat <<'EOF'
Usage: MODEL=/path/Qwen3.6-27B-Q4_K_M.gguf \
       run_qwen36_canonical_concurrency.sh

Runs the exact raw 10-prompt HumanEval-style blog corpus plus the repository's
HumanEval-chat, GSM8K, Math500, and agent prompt suites. Each level processes
the entire suite in full fixed-width waves: C=1/2/5/10 for ten-case suites and
C=1/2/3/6 for the six-case agent suite. No prompt is duplicated within a run.

The default AR variant measures the concurrent server path.
Set VARIANTS=blog-ddtree with a readable DRAFT_MODEL to run the optional
DDTree variant; it requires a server build that emits per-response
[concurrency-metrics] telemetry. The server still uses paged attention
because this script measures the concurrent implementation, including C=1.
GPU_DEVICE is the physical ROCr device exposed exclusively to the server and
defaults to device 0. Pass GPU_DEVICE=1 for Strix Halo on this dual-GPU
benchmark host. The resolved value is stored in each case's command and metadata.
Set EXPECTED_GPU_ARCH=gfx1151 to require that literal architecture in the
server startup log; an empty value disables the check. The expectation is
stored in each case's metadata alongside the selected physical device.
PREFILL_FIRST_BURST_STEPS forwards the experimental Lucebox TTFT policy; 0
preserves continuous decode, while positive N permits at most N consecutive
prefill-only traversals before a mandatory decode traversal. Valid values are
0 through 1024, matching the server-side limit.
IDLE_PREFILL_TOKENS forwards Luce's idle/prefill-only token budget and defaults
to 4096. Valid values are 1..16384, the current maximum effective K8 pure
prefill batch (8 lanes x 2048 tokens).
EOF
}

if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -ne 0 ]]; then usage >&2; exit 2; fi
for cmd in python3 curl sha256sum ldd; do command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 2; }; done
[[ -r "$MODEL" ]] || { echo "set MODEL to a readable target GGUF" >&2; exit 2; }
[[ -x "$SERVER_BIN" ]] || { echo "missing server: $SERVER_BIN" >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 2; }
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_TOKENS must be positive" >&2; exit 2; }
[[ -z "$EXPECTED_GPU_ARCH" || "$EXPECTED_GPU_ARCH" =~ ^gfx[0-9a-f]+$ ]] || { echo "EXPECTED_GPU_ARCH must be empty or a gfx architecture" >&2; exit 2; }
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] || { echo "GPU_DEVICE must be a physical ROCr device index" >&2; exit 2; }
if ! [[ "$PREFILL_FIRST_BURST_STEPS" =~ ^[0-9]{1,4}$ ]] || (( 10#$PREFILL_FIRST_BURST_STEPS > 1024 )); then
  echo "PREFILL_FIRST_BURST_STEPS must be an integer in range 0..1024" >&2
  exit 2
fi
if ! [[ "$IDLE_PREFILL_TOKENS" =~ ^[1-9][0-9]{0,4}$ ]] || (( 10#$IDLE_PREFILL_TOKENS > 16384 )); then
  echo "IDLE_PREFILL_TOKENS must be an integer in range 1..16384" >&2
  exit 2
fi
if ! [[ "$SLOTS" =~ ^[1-9][0-9]*$ ]] || (( SLOTS < 16 )); then
  echo "SLOTS must be an integer >= 16" >&2
  exit 2
fi
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
ambient_tuning="$(env | grep -E '^(GGML_|DFLASH_|LUCE_|HIP_|ROCR_|HSA_|LD_PRELOAD=|LD_LIBRARY_PATH=)' || true)"
if [[ -n "$ambient_tuning" ]]; then
  echo "refusing ambient GPU/backend tuning variables:" >&2
  echo "$ambient_tuning" >&2
  exit 2
fi

IFS=, read -r -a suite_list <<< "$SUITES"
IFS=, read -r -a variant_list <<< "$VARIANTS"
for suite in "${suite_list[@]}"; do
  [[ "$suite" =~ ^(he-raw|he|gsm|math|agent)$ ]] || { echo "unknown suite $suite" >&2; exit 2; }
done
for variant in "${variant_list[@]}"; do
  [[ "$variant" =~ ^(ar|blog-ddtree|adaptive-ddtree)$ ]] || { echo "unknown variant $variant" >&2; exit 2; }
done
if [[ "$VARIANTS" == *ddtree* ]]; then
  [[ -r "$DRAFT_MODEL" ]] || { echo "blog-ddtree requires readable DRAFT_MODEL" >&2; exit 2; }
fi

mkdir -p "$OUT/prompts"
python3 "$GENERATOR" --profile he-raw --out "$OUT/prompts/he-raw.jsonl"
for suite in he gsm math agent; do
  cp "$REPO/harness/benchmarks/prompts/bench_${suite}.jsonl" "$OUT/prompts/$suite.jsonl"
done

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
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

levels_for_suite() {
  if [[ -n "$CLIENTS" ]]; then echo "${CLIENTS//,/ }"; return; fi
  [[ "$1" == agent ]] && echo "1 2 3 6" || echo "1 2 5 10"
}

run_case() {
  local repeat="$1" suite="$2" clients="$3" variant="$4"
  local max_ctx=4096
  [[ "$suite" == agent ]] && max_ctx=32768
  local capacity=$((SLOTS * max_ctx))
  local case_dir="$OUT/$suite/c$clients/r$repeat/$variant"
  mkdir -p "$case_dir"
  local -a command launch client_common bench_options
  command=("$SERVER_BIN" "$MODEL" --target-device hip:0 --paged-attention
    --max-concurrency "$SLOTS" --kv-pool-tokens "$capacity" --max-ctx "$max_ctx"
    --cache-type-k q4_0 --cache-type-v q4_0 --fa-window 0
    --prefix-cache-slots 0 --prefill-cache-slots 0
    --admission-coalesce-ms 5 --host 127.0.0.1 --port "$PORT" --model-name qwen36)
  if [[ "$suite" == he-raw ]]; then
    command+=(--chat-template-file "$SCRIPT_DIR/raw_prompt_identity.jinja")
  fi
  if [[ "$variant" == *ddtree ]]; then
    local adaptive=0
    [[ "$variant" == adaptive-ddtree ]] && adaptive=1
    command+=(--draft "$DRAFT_MODEL" --draft-device hip:0 --ddtree
      --ddtree-budget 22 --fast-rollback --draft-residency persistent)
    launch=(env ROCR_VISIBLE_DEVICES="$GPU_DEVICE" DFLASH_IGNORE_EOS=1 DFLASH27B_DRAFT_SWA=2048 DFLASH_DDTREE_ADAPTIVE="$adaptive"
      DFLASH_PREFILL_FIRST_BURST_STEPS="$PREFILL_FIRST_BURST_STEPS"
      DFLASH_IDLE_PREFILL_TOKENS="$IDLE_PREFILL_TOKENS"
      DFLASH_MIN_TOKENS="$WARMUP_TOKENS" stdbuf -oL -eL "${command[@]}")
  else
    launch=(env ROCR_VISIBLE_DEVICES="$GPU_DEVICE" DFLASH_IGNORE_EOS=1 DFLASH_MIN_TOKENS="$WARMUP_TOKENS"
      DFLASH_PREFILL_FIRST_BURST_STEPS="$PREFILL_FIRST_BURST_STEPS"
      DFLASH_IDLE_PREFILL_TOKENS="$IDLE_PREFILL_TOKENS"
      stdbuf -oL -eL "${command[@]}")
  fi
  printf '%q ' "${launch[@]}" > "$case_dir/server-command.txt"
  printf '\n' >> "$case_dir/server-command.txt"
  python3 -c 'import hashlib,json,pathlib,sys
import subprocess
out,variant,suite,c,repeat,binary,model,draft,prompts,cmd,n_gen,repo,rocr_visible,slots,prefill_first_burst_steps,expected_gpu_arch,idle_prefill_tokens=sys.argv[1:]
digest=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if p else None
libs={}
for line in subprocess.run(["ldd",binary],text=True,capture_output=True).stdout.splitlines():
    fields=line.replace("=>"," ").split()
    paths=[x for x in fields if x.startswith("/") and pathlib.Path(x).is_file()]
    for lib in paths: libs[str(pathlib.Path(lib).resolve())]=digest(lib)
git_head=subprocess.run(["git","-C",repo,"rev-parse","HEAD"],text=True,capture_output=True).stdout.strip() or None
obj={"variant":variant,"suite":suite,"clients":int(c),"repeat":int(repeat),
"server_binary":str(pathlib.Path(binary).resolve()),"server_binary_sha256":digest(binary),
"model_sha256":digest(model),"draft_model_sha256":digest(draft),
"lucebox_git_head":git_head,
"resolved_shared_library_sha256":libs,
"prompt_file_sha256":digest(prompts),"server_command":pathlib.Path(cmd).read_text().strip(),
"rocr_visible_devices":rocr_visible,"server_slots":int(slots),
"prefill_first_burst_steps":int(prefill_first_burst_steps),
"idle_prefill_tokens":int(idle_prefill_tokens),
"expected_gpu_arch":expected_gpu_arch or None,
"blog_decode_settings":({"draft_quant":"Q8_0","draft_swa":2048,"ddtree_budget":22,
"fast_rollback":True,"adaptive":variant == "adaptive-ddtree","n_gen":int(n_gen)} if variant.endswith("ddtree") else None)}
pathlib.Path(out).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")' \
    "$case_dir/server-metadata.json" "$variant" "$suite" "$clients" "$repeat" \
    "$SERVER_BIN" "$MODEL" "$([[ "$variant" == *ddtree ]] && echo "$DRAFT_MODEL")" \
    "$OUT/prompts/$suite.jsonl" "$case_dir/server-command.txt" "$MAX_TOKENS" "$REPO" \
    "$GPU_DEVICE" "$SLOTS" "$PREFILL_FIRST_BURST_STEPS" "$EXPECTED_GPU_ARCH" "$IDLE_PREFILL_TOKENS"

  echo "[canonical] $suite C=$clients repeat=$repeat variant=$variant"
  "${launch[@]}" > "$case_dir/server.log" 2>&1 &
  server_pid=$!
  if ! wait_health; then tail -n 80 "$case_dir/server.log" >&2 || true; stop_server; return 1; fi
  if [[ -n "$EXPECTED_GPU_ARCH" ]]; then
    if ! grep -F -- "$EXPECTED_GPU_ARCH" "$case_dir/server.log" > "$case_dir/gpu-identity.txt"; then
      echo "server log does not identify expected GPU architecture: $EXPECTED_GPU_ARCH" >&2
      tail -n 80 "$case_dir/server.log" >&2 || true
      stop_server
      sleep "$COOLDOWN_SECONDS"
      return 1
    fi
  fi
  client_common=(--base-url "http://127.0.0.1:$PORT/v1" --model qwen36
    --suite "$suite" --clients "$clients" --prompt-file "$OUT/prompts/$suite.jsonl"
    --temperature 0 --seed 1 --ignore-eos --timeout 1800 --retire-log "$case_dir/server.log")
  [[ -n "$CASE_LIMIT" ]] && client_common+=(--case-limit "$CASE_LIMIT")
  bench_options=()
  [[ "$variant" == *ddtree ]] && bench_options+=(--ddtree-proof)
  local status=0
  python3 "$CLIENT" "${client_common[@]}" --max-tokens "$WARMUP_TOKENS" \
    --out "$case_dir/warmup.json" --label "$variant $suite C=$clients warmup" \
    > "$case_dir/warmup.txt" || status=1
  if (( status == 0 )); then
    python3 "$CLIENT" "${client_common[@]}" --max-tokens "$MAX_TOKENS" \
      --server-metadata-json "$case_dir/server-metadata.json" "${bench_options[@]}" \
      --out "$case_dir/bench.json" \
      --label "$variant $suite C=$clients repeat=$repeat" | tee "$case_dir/bench.txt" || status=1
  fi
  stop_server
  sleep "$COOLDOWN_SECONDS"
  return "$status"
}

failures=0
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  for suite in "${suite_list[@]}"; do
    for clients in $(levels_for_suite "$suite"); do
      (( clients <= SLOTS )) || { echo "C=$clients exceeds SLOTS=$SLOTS" >&2; exit 2; }
      for variant in "${variant_list[@]}"; do
        run_case "$repeat" "$suite" "$clients" "$variant" || failures=$((failures + 1))
      done
    done
  done
done
(( failures == 0 )) || { echo "$failures case(s) failed" >&2; exit 1; }
python3 "$SUMMARIZER" "$OUT" --format canonical --out "$OUT/summary.md"
echo "[canonical] complete: $OUT"
