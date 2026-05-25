#!/usr/bin/env bash
# Sindri RTX 3090 Ti @ 225W — perf sweep for Qwen3.6-27B Q4_K_M + dflash-draft-3.6.
# Defensive: no set -e. Each config logs whatever it gets to CSV.
# Note: current binary (May 23) does NOT emit usage.timings; columns NaN.
#       It does emit "[spec-decode] tokens=" summary lines to server.log.

SWEEP_DIR=/tmp/perf-sweep
PORT=1236
HOST=127.0.0.1
SERVER=/home/erik/Projects/lucebox-hub/dflash/build/dflash_server
MODEL=/home/erik/Projects/lucebox-hub/dflash/models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT=/home/erik/Projects/lucebox-hub/dflash/models/draft/dflash-draft-3.6-q8_0.gguf
CSV="$SWEEP_DIR/results.csv"
LOG_ROOT="$SWEEP_DIR/runs"
mkdir -p "$LOG_ROOT"

PROMPT="$(cat <<'EOF'
Solve this problem step by step, showing all your reasoning clearly.

A bag contains 12 red balls, 15 blue balls, and 18 green balls. Three balls
are drawn at random without replacement. What is the probability that:
(a) all three balls are the same color,
(b) exactly two balls are blue,
(c) at least one ball is green?

Show all calculations as fractions, then convert to decimals. Explain each
step. Conclude with a summary table of all three answers.
EOF
)"

MAX_TOKENS=1024

# CSV header (always overwrite)
echo "config_name,prefix_cache_slots,ddtree_budget,max_ctx,chunk,cache_type_v,wall_s,completion_tokens,prompt_tokens,specdec_tokens,specdec_speed_tps,specdec_accept_pct,specdec_avg_commit" > "$CSV"

CONFIGS=(
  "baseline             0  22  131072  512  q4_0"
  "prefix-16            16 22  131072  512  q4_0"
  "budget-8             0  8   131072  512  q4_0"
  "budget-14            0  14  131072  512  q4_0"
  "budget-32            0  32  131072  512  q4_0"
  "max-ctx-32k          0  22  32768   512  q4_0"
  "chunk-2048           0  22  131072  2048 q4_0"
  "cache-v-q8           0  22  131072  512  q8_0"
)

kill_server() {
  local pid="$1"
  # SIGTERM first; let it flush logs
  kill "$pid" 2>/dev/null
  for i in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then return; fi
    sleep 1
  done
  # SIGKILL fallback
  kill -9 "$pid" 2>/dev/null
  sleep 3
}

run_one() {
  local name="$1" prefix="$2" budget="$3" mctx="$4" chunk="$5" vtype="$6"
  local run_dir="$LOG_ROOT/$name"
  mkdir -p "$run_dir"
  local server_log="$run_dir/server.log"
  local resp_json="$run_dir/response.json"

  echo "=== [$(date +%H:%M:%S)] $name ==="
  echo "    prefix=$prefix budget=$budget max-ctx=$mctx chunk=$chunk v=$vtype"

  # Boot server (stdbuf -oL = line-buffered stdout so logs flush promptly)
  stdbuf -oL "$SERVER" "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --max-ctx "$mctx" \
    --prefix-cache-slots "$prefix" \
    --cache-type-k q4_0 --cache-type-v "$vtype" \
    --draft "$DRAFT" \
    --ddtree --ddtree-budget "$budget" \
    --chunk "$chunk" \
    > "$server_log" 2>&1 &
  local SPID=$!
  echo "    server pid=$SPID"

  # Wait for /props
  local up=0
  for i in $(seq 1 90); do
    if curl -s --max-time 1 "http://$HOST:$PORT/props" > /dev/null 2>&1; then
      up=1; break
    fi
    sleep 1
  done
  if [ "$up" -ne 1 ]; then
    echo "    SERVER FAILED TO START — skipping"
    echo "$name,$prefix,$budget,$mctx,$chunk,$vtype,SERVER_BOOT_FAILED,,,,,," >> "$CSV"
    kill_server "$SPID"
    return
  fi
  echo "    server ready"

  # Send request
  local req_body
  req_body=$(jq -nc \
    --arg p "$PROMPT" \
    --argjson mt "$MAX_TOKENS" \
    '{model:"dflash", messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:1.0, thinking:{type:"disabled"}, stream:false}')

  local t0=$(date +%s.%N)
  curl -s --max-time 600 -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$req_body" \
    -o "$resp_json"
  local curl_rc=$?
  local t1=$(date +%s.%N)
  local wall
  wall=$(awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.3f", a-b}')
  echo "    wall=${wall}s curl_rc=$curl_rc"

  if [ "$curl_rc" -ne 0 ] || [ ! -s "$resp_json" ]; then
    echo "    curl failed or empty response"
    echo "$name,$prefix,$budget,$mctx,$chunk,$vtype,$wall,CURL_FAIL,,,,," >> "$CSV"
    kill_server "$SPID"
    return
  fi

  # Parse response (jq with fallbacks — never let it tank set -e)
  local ctok ptok
  ctok=$(jq -r '.usage.completion_tokens // ""' "$resp_json" 2>/dev/null)
  ptok=$(jq -r '.usage.prompt_tokens // ""' "$resp_json" 2>/dev/null)
  ctok=${ctok:-NaN}; ptok=${ptok:-NaN}

  # IMPORTANT: kill server FIRST so the spec-decode summary line flushes.
  echo "    killing server to flush logs..."
  kill_server "$SPID"

  # Now grep the (post-flush) server log
  local sd_line
  sd_line=$(grep -E "^\[spec-decode\] tokens=" "$server_log" 2>/dev/null | tail -1)
  local sd_tokens sd_speed sd_accept sd_commit
  sd_tokens=$(echo "$sd_line" | sed -nE 's/.*tokens=([0-9]+).*/\1/p')
  sd_speed=$(echo "$sd_line" | sed -nE 's/.*speed=([0-9.]+) tok.*/\1/p')
  sd_accept=$(echo "$sd_line" | sed -nE 's/.*\(([0-9.]+)%\).*/\1/p')
  sd_commit=$(echo "$sd_line" | sed -nE 's/.*avg_commit=([0-9.]+).*/\1/p')
  sd_tokens=${sd_tokens:-NaN}
  sd_speed=${sd_speed:-NaN}
  sd_accept=${sd_accept:-NaN}
  sd_commit=${sd_commit:-NaN}

  echo "    ctok=$ctok ptok=$ptok spec=$sd_speed tok/s accept=$sd_accept% commit=$sd_commit"
  echo "$name,$prefix,$budget,$mctx,$chunk,$vtype,$wall,$ctok,$ptok,$sd_tokens,$sd_speed,$sd_accept,$sd_commit" >> "$CSV"
}

for line in "${CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  run_one $line
done

echo
echo "=== sweep complete; results: $CSV ==="
column -t -s, "$CSV"
