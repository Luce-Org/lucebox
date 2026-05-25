#!/usr/bin/env bash
# Sindri RTX 3090 Ti @ 225W — deeper perf sweep (iter 2).
# - temp=0 (exercises spec-decode for configs with a draft)
# - 3 trials per config (washes out wall noise)
# - longer prompt (~700 token) to exercise prefill realistically
# - [spec-decode] / [ar-decode] summary lines now both available (binary patched)
# - prefix-cache-16 test exploits 3 identical trials to show 2nd+3rd speedup

SWEEP_DIR=/tmp/perf-sweep
PORT=1236
HOST=127.0.0.1
SERVER=/home/erik/Projects/lucebox-hub/dflash/build/dflash_server
MODEL=/home/erik/Projects/lucebox-hub/dflash/models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT_Q8=/home/erik/Projects/lucebox-hub/dflash/models/draft/dflash-draft-3.6-q8_0.gguf
DRAFT_Q4=/home/erik/Projects/lucebox-hub/dflash/models/draft/dflash-draft-3.6-q4_k_m.gguf
CSV="$SWEEP_DIR/results2.csv"
LOG_ROOT="$SWEEP_DIR/runs2"
mkdir -p "$LOG_ROOT"

# Larger prompt: a coding-style problem with multiple sub-asks.
PROMPT="$(cat <<'EOF'
You are implementing a thread-safe LRU cache in C++17. The class signature is:

    template<typename K, typename V, size_t Cap>
    class LRUCache {
    public:
        void   put(const K& key, V value);
        std::optional<V> get(const K& key);   // also promotes to most-recent
        size_t size() const;
        void   clear();
    };

Requirements:
  1. Reads and writes must both be safe from multiple threads.
  2. get() must be lock-free on a cache hit if the hot element is already at
     the front of the recency list. On hits that require promotion, taking
     the write lock is allowed.
  3. Eviction must happen during put() when size() would exceed Cap.
  4. The internal storage must use a hash map + intrusive doubly-linked list,
     not std::list (to avoid allocating per insert).

Tasks:
  (a) Outline your data structures, including all member fields and their
      mutexes / atomics. Justify each synchronization primitive choice.
  (b) Write the put() function, including the eviction case. Explain how
      you avoid races between concurrent put() calls that evict the same
      victim.
  (c) Write the get() function. Show how the lock-free-fast-path branch
      decides whether promotion is needed; explain the memory order of
      every atomic operation involved.
  (d) Discuss the trade-off between this design and a sharded approach
      where you'd split the cache into N independent shards each with its
      own mutex.
  (e) Suggest a benchmark plan (workload, metrics, suite) that would
      verify the lock-free-fast-path optimization is actually paying off
      vs. always taking the write lock.

Be thorough. Show code where appropriate. The goal is a design doc that a
senior engineer could review and approve.
EOF
)"

MAX_TOKENS=1024
TRIALS=3

# CSV
echo "config,trial,prefix_cache_slots,ddtree_budget,chunk,cache_type_v,draft,wall_s,ctok,ptok,decode_path,decode_tokens,decode_speed_tps,specdec_accept_pct,specdec_avg_commit" > "$CSV"

# config_name prefix budget chunk vtype draft_path
# draft_path: "q8" / "q4" / "none"
CONFIGS=(
  "baseline    0  22  512   q4_0  q8"
  "prefix-16   16 22  512   q4_0  q8"
  "budget-8    0  8   512   q4_0  q8"
  "budget-32   0  32  512   q4_0  q8"
  "budget-48   0  48  512   q4_0  q8"
  "chunk-2048  0  22  2048  q4_0  q8"
  "cache-v-q8  0  22  512   q8_0  q8"
  "draft-q4    0  22  512   q4_0  q4"
  "no-draft    0  22  512   q4_0  none"
)

kill_server() {
  local pid="$1"
  kill "$pid" 2>/dev/null
  for i in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then return; fi
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null
  sleep 3
}

run_one() {
  local name="$1" prefix="$2" budget="$3" chunk="$4" vtype="$5" draft_choice="$6"
  local run_dir="$LOG_ROOT/$name"
  mkdir -p "$run_dir"
  local server_log="$run_dir/server.log"

  echo "=== [$(date +%H:%M:%S)] $name ==="
  echo "    prefix=$prefix budget=$budget chunk=$chunk v=$vtype draft=$draft_choice"

  # Choose draft + spec-decode flags
  local draft_args=()
  case "$draft_choice" in
    q8)   draft_args=(--draft "$DRAFT_Q8" --ddtree --ddtree-budget "$budget") ;;
    q4)   draft_args=(--draft "$DRAFT_Q4" --ddtree --ddtree-budget "$budget") ;;
    none) draft_args=() ;;
  esac

  # Boot
  stdbuf -oL "$SERVER" "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --max-ctx 131072 \
    --prefix-cache-slots "$prefix" \
    --cache-type-k q4_0 --cache-type-v "$vtype" \
    --chunk "$chunk" \
    "${draft_args[@]}" \
    > "$server_log" 2>&1 &
  local SPID=$!

  # Wait for /props
  local up=0
  for i in $(seq 1 120); do
    if curl -s --max-time 1 "http://$HOST:$PORT/props" > /dev/null 2>&1; then up=1; break; fi
    sleep 1
  done
  if [ "$up" -ne 1 ]; then
    echo "    SERVER FAILED TO START"
    for t in $(seq 1 $TRIALS); do
      echo "$name,$t,$prefix,$budget,$chunk,$vtype,$draft_choice,SERVER_FAIL,,,,,," >> "$CSV"
    done
    kill_server "$SPID"
    return
  fi
  echo "    server ready (pid $SPID)"

  # Request body (temp=0 → spec-decode if draft present, AR otherwise)
  local req_body
  req_body=$(jq -nc \
    --arg p "$PROMPT" \
    --argjson mt "$MAX_TOKENS" \
    '{model:"dflash", messages:[{role:"user",content:$p}], max_tokens:$mt, temperature:0, top_p:1.0, thinking:{type:"disabled"}, stream:false}')

  for t in $(seq 1 $TRIALS); do
    local resp_json="$run_dir/response-t$t.json"
    local pre_log_lines
    pre_log_lines=$(wc -l < "$server_log")

    local t0=$(date +%s.%N)
    curl -s --max-time 600 -X POST "http://$HOST:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$req_body" \
      -o "$resp_json"
    local curl_rc=$?
    local t1=$(date +%s.%N)
    local wall
    wall=$(awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.3f", a-b}')

    if [ "$curl_rc" -ne 0 ] || [ ! -s "$resp_json" ]; then
      echo "    trial $t: curl FAIL"
      echo "$name,$t,$prefix,$budget,$chunk,$vtype,$draft_choice,$wall,CURL_FAIL,,,,,," >> "$CSV"
      continue
    fi

    local ctok ptok
    ctok=$(jq -r '.usage.completion_tokens // ""' "$resp_json" 2>/dev/null); ctok=${ctok:-NaN}
    ptok=$(jq -r '.usage.prompt_tokens // ""' "$resp_json" 2>/dev/null); ptok=${ptok:-NaN}

    # Look at lines APPENDED to server.log since this trial started — get the
    # most recent [spec-decode] or [ar-decode] line.
    local new_lines
    new_lines=$(sed -n "$((pre_log_lines+1)),\$p" "$server_log")
    local path tokens speed accept commit
    local sd_line ar_line
    sd_line=$(echo "$new_lines" | grep -E "^\[spec-decode\] tokens=" | tail -1)
    ar_line=$(echo "$new_lines" | grep -E "^\[ar-decode\] tokens="   | tail -1)
    if [ -n "$sd_line" ]; then
      path=spec
      tokens=$(echo "$sd_line" | sed -nE 's/.*tokens=([0-9]+).*/\1/p')
      speed=$(echo "$sd_line"  | sed -nE 's/.*speed=([0-9.]+) tok.*/\1/p')
      accept=$(echo "$sd_line" | sed -nE 's/.*\(([0-9.]+)%\).*/\1/p')
      commit=$(echo "$sd_line" | sed -nE 's/.*avg_commit=([0-9.]+).*/\1/p')
    elif [ -n "$ar_line" ]; then
      path=ar
      tokens=$(echo "$ar_line" | sed -nE 's/.*tokens=([0-9]+).*/\1/p')
      speed=$(echo "$ar_line"  | sed -nE 's/.*speed=([0-9.]+) tok.*/\1/p')
      accept=
      commit=
    else
      path=unknown
      tokens=; speed=; accept=; commit=
    fi
    tokens=${tokens:-NaN}; speed=${speed:-NaN}
    accept=${accept:-NaN}; commit=${commit:-NaN}

    echo "    trial $t: wall=${wall}s ctok=$ctok path=$path tokens=$tokens speed=${speed}tok/s accept=${accept}% commit=$commit"
    echo "$name,$t,$prefix,$budget,$chunk,$vtype,$draft_choice,$wall,$ctok,$ptok,$path,$tokens,$speed,$accept,$commit" >> "$CSV"
  done

  echo "    killing server..."
  kill_server "$SPID"
}

for line in "${CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  run_one $line
done

echo
echo "=== sweep complete ==="
column -t -s, "$CSV"
