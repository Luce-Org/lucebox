#!/usr/bin/env bash
# Start the routing trial topology as separate luce_server processes:
#   qwen38-27b  (+DFlash2 draft)  R9700      hip:0  :8421
#   brick-max   (classifier)      R9700      hip:0  :8411
#   qwen35-2b                     Strix Halo hip:1  :8402
# Each process gets an explicit --kv-pool-tokens so the first one started
# does not size its pool from all free memory.
#
#   launch_routing.sh [name ...]   (default: all three; names: 27b brick 2b)
# Env: LUCE_SERVER (binary), MODELS (~/models), FORCE=1 to start even if
# another process holds /dev/kfd.
# Stop with stop_small.sh (pidfiles in runs/servers/).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LUCE_SERVER="${LUCE_SERVER:-$REPO/server/build-routing/luce_server}"
MODELS="${MODELS:-$HOME/models}"
RUN_DIR="$HERE/runs/servers"
mkdir -p "$RUN_DIR"

[[ -x "$LUCE_SERVER" ]] || { echo "no luce_server at $LUCE_SERVER (build server/build-routing first)" >&2; exit 1; }
if [[ "${FORCE:-0}" != 1 ]] && holders="$(lsof -t /dev/kfd 2>/dev/null)" && [[ -n "$holders" ]]; then
  echo "/dev/kfd is held by pid(s): $holders ($(ps -o comm= -p ${holders//$'\n'/,} | sort -u | tr '\n' ' '))" >&2
  echo "another session may be using the GPUs; rerun with FORCE=1 if the memory budget allows" >&2
  exit 1
fi

start() {
  local name="$1"; shift
  if [[ -f "$RUN_DIR/$name.pid" ]] && kill -0 "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null; then
    echo "$name already running ($(cat "$RUN_DIR/$name.pid"))"; return
  fi
  nohup "$LUCE_SERVER" "$@" > "$RUN_DIR/$name.log" 2>&1 &
  echo $! > "$RUN_DIR/$name.pid"
  echo "started $name ($!) -> $RUN_DIR/$name.log"
}

wait_ready() {
  local name="$1" port="$2"
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { echo "$name ready on :$port"; return; }
    kill -0 "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null || { echo "$name exited; tail of log:" >&2; tail -20 "$RUN_DIR/$name.log" >&2; return 1; }
    sleep 1
  done
  echo "$name not ready after 120 s" >&2; return 1
}

want=("$@"); [[ ${#want[@]} -gt 0 ]] || want=(27b brick 2b)
for w in "${want[@]}"; do
  case "$w" in
    27b)
      start qwen38-27b "$MODELS/Qwen3.8-27B-UD-IQ4_XS.gguf" \
        --draft "$MODELS/qwen38-dflash2-q8_0.gguf" --draft-block-size 16 \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        --target-device hip:0 --max-concurrency 4 --kv-pool-tokens 131072 \
        --max-ctx 65536 --max-tokens 32768 \
        --host 127.0.0.1 --port 8421 --model-name qwen3.8-27b
      wait_ready qwen38-27b 8421 ;;
    brick)
      start brick-max "$MODELS/routing/brick-complexity-2-max-Q8_0.gguf" \
        --target-device hip:0 --max-concurrency 4 --kv-pool-tokens 131072 --max-ctx 32768 \
        --host 127.0.0.1 --port 8411 --model-name brick-max
      wait_ready brick-max 8411 ;;
    2b)
      start qwen35-2b "$MODELS/routing/Qwen3.5-2B-Q8_0.gguf" \
        --target-device hip:1 --max-concurrency 4 --kv-pool-tokens 262144 --max-ctx 122880 \
        --host 127.0.0.1 --port 8402 --model-name qwen35-2b
      wait_ready qwen35-2b 8402 ;;
    *) echo "unknown name $w (27b brick 2b)" >&2; exit 2 ;;
  esac
done
