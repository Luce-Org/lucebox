#!/usr/bin/env bash
# Start the routing trial topology as ONE luce_server process on :8420 that
# hosts every model; a request picks its model by the OpenAI `model` field
# (--model-routing name):
#   qwen3.8-27b (+DFlash2 draft)  R9700      hip:0  4 slots
#   brick-max   (classifier)      R9700      hip:0  4 slots
#   qwen35-2b                     Strix Halo hip:1  4 slots
# Each model block gets an explicit --kv-pool-tokens so the first model loaded
# does not size its pool from all free memory on the shared R9700.
# Omitted/"auto" model names balance across the blocks in order (27B first);
# unknown names get 404 with the list of loaded names.
#
#   launch_routing.sh
# TOPOLOGY=strix starts only brick-max and qwen35-2b, both on hip:1, for when
# another session holds the R9700.
# Env: LUCE_SERVER (binary), MODELS (~/models), PORT (8420), FORCE=1 to start
# even if another process holds /dev/kfd. LUCE_MULTI_MODEL_GRAPHS=1 passes
# through to keep GPU graphs on (experimental; see MODEL_LOAD_BALANCING.md).
# Stop with stop_small.sh (pidfile runs/servers/routing.pid).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LUCE_SERVER="${LUCE_SERVER:-$REPO/server/build-routing/luce_server}"
MODELS="${MODELS:-$HOME/models}"
PORT="${PORT:-8420}"
RUN_DIR="$HERE/runs/servers"
NAME=routing
mkdir -p "$RUN_DIR"

[[ $# -eq 0 ]] || { echo "usage: $0 (one process hosts all models; no per-model names)" >&2; exit 2; }
[[ -x "$LUCE_SERVER" ]] || { echo "no luce_server at $LUCE_SERVER (build server/build-routing first)" >&2; exit 1; }
if [[ -f "$RUN_DIR/$NAME.pid" ]] && kill -0 "$(cat "$RUN_DIR/$NAME.pid")" 2>/dev/null; then
  echo "$NAME already running ($(cat "$RUN_DIR/$NAME.pid"))"; exit 0
fi
if [[ "${FORCE:-0}" != 1 ]] && holders="$(lsof -t /dev/kfd 2>/dev/null)" && [[ -n "$holders" ]]; then
  echo "/dev/kfd is held by pid(s): $holders ($(ps -o comm= -p ${holders//$'\n'/,} | sort -u | tr '\n' ' '))" >&2
  echo "another session may be using the GPUs; rerun with FORCE=1 if the memory budget allows" >&2
  exit 1
fi

BRICK_DEVICE=hip:0
# The first model is positional; process-wide options go in its block.
args=(--model-routing name)
common=(--host 127.0.0.1 --port "$PORT" --routing-queue-limit 64)
case "${TOPOLOGY:-full}" in
  full)
    args+=(
      "$MODELS/Qwen3.8-27B-UD-IQ4_XS.gguf" "${common[@]}" --model-name qwen3.8-27b
        --target-device hip:0
        --draft "$MODELS/qwen38-dflash2-q8_0.gguf" --draft-device hip:0 --draft-block-size 16
        --cache-type-k q8_0 --cache-type-v q8_0
        --max-concurrency 4 --kv-pool-tokens 131072 --max-ctx 65536 --max-tokens 32768) ;;
  strix) BRICK_DEVICE=hip:1 ;;
  *) echo "TOPOLOGY must be full or strix" >&2; exit 2 ;;
esac
if [[ "${TOPOLOGY:-full}" == strix ]]; then
  args+=("$MODELS/routing/brick-complexity-2-max-Q8_0.gguf" "${common[@]}")
else
  args+=(--model "$MODELS/routing/brick-complexity-2-max-Q8_0.gguf")
fi
args+=(
  --model-name brick-max
    --target-device "$BRICK_DEVICE"
    --max-concurrency 4 --kv-pool-tokens 131072 --max-ctx 32768
    --chat-template-file "$HERE/config/brick_chatml.jinja"
  --model "$MODELS/routing/Qwen3.5-2B-Q8_0.gguf" --model-name qwen35-2b
    --target-device hip:1
    --max-concurrency 4 --kv-pool-tokens 262144 --max-ctx 122880
)
nohup "$LUCE_SERVER" "${args[@]}" > "$RUN_DIR/$NAME.log" 2>&1 &
echo $! > "$RUN_DIR/$NAME.pid"
echo "started $NAME ($!) -> $RUN_DIR/$NAME.log"

# The listener opens only after every model has loaded.
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "$NAME ready on :$PORT"
    curl -sf "http://127.0.0.1:$PORT/v1/models" |
      python3 -c 'import json,sys; print("models:", ", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' || true
    exit 0
  fi
  kill -0 "$(cat "$RUN_DIR/$NAME.pid")" 2>/dev/null || { echo "$NAME exited; tail of log:" >&2; tail -20 "$RUN_DIR/$NAME.log" >&2; exit 1; }
  sleep 1
done
echo "$NAME not ready after 300 s" >&2; exit 1
