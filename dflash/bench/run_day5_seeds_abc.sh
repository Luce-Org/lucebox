#!/usr/bin/env bash
# 3-seed Day-5 A/B/C run for PR #264 variance evidence.
# Usage: run_day5_seeds_abc.sh <seed_label> <prompt_file> <session_suffix>
#   seed_label:     seed1 | seed2 | seed3
#   prompt_file:    basename of prompt file under harness/clients/prompts/
#   session_suffix: unique string appended to session_id for condition C
#
# Example:
#   ./run_day5_seeds_abc.sh seed1 decode_check.txt day5s1
#   ./run_day5_seeds_abc.sh seed2 repo_inspection.txt day5s2
#   ./run_day5_seeds_abc.sh seed3 math_check.txt day5s3
set -euo pipefail

SEED_LABEL="${1:?Usage: $0 <seed_label> <prompt_file> <session_suffix>}"
PROMPT_BASENAME="${2:?Usage: $0 <seed_label> <prompt_file> <session_suffix>}"
SESSION_SUFFIX="${3:?Usage: $0 <seed_label> <prompt_file> <session_suffix>}"

WORKTREE="/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto"
RESULTS_BASE="$WORKTREE/dflash/bench/results/2026-05-23_day5_seeds"
RESULTS_DIR="$RESULTS_BASE/$SEED_LABEL"
SERVER_BIN="$WORKTREE/dflash/build/dflash_server"
TARGET="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT="/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf"
PFLASH_DRAFTER="/home/peppi/models/Qwen3-0.6B-BF16.gguf"
HARNESS_DIR="$WORKTREE/harness/clients"
PROMPT_FILE="$HARNESS_DIR/prompts/$PROMPT_BASENAME"
CLAUDE_BIN="${CLAUDE_BIN:-/home/peppi/.local/bin/claude}"
MARKER="OK_DONE"
CLAUDE_TIMEOUT=600

HOST=127.0.0.1
PORT=18080
PROXY_PORT=18082
MODEL_ID="luce-dflash"
API_KEY="sk-lucebox"
BASE_URL="http://$HOST:$PORT"

mkdir -p "$RESULTS_DIR"
echo "=== Day 5 Seeds A/B/C [$SEED_LABEL] prompt=$PROMPT_BASENAME start $(date -Is) ===" | tee "$RESULTS_DIR/run.log"

# ─── run_condition ──────────────────────────────────────────────────────────
# Args: LABEL KEEP_RATIO SESSION_ID(or empty)
run_condition() {
    local label="$1"
    local keep="$2"
    local sid="$3"
    local cdir="$RESULTS_DIR/$label"
    mkdir -p "$cdir"

    local slog="$cdir/server.log"
    local plog="$cdir/proxy.log"
    local cout="$cdir/client.out"
    local mfile="$cdir/metrics.txt"

    echo "--- [$SEED_LABEL/$label] keep=$keep sid='$sid' $(date -Is) ---" | tee -a "$RESULTS_DIR/run.log"
    local t0; t0=$(date +%s)

    _SID="$sid" _KEEP="$keep" _SLOG="$slog" _PLOG="$plog" _COUT="$cout" \
    _CHOME="$cdir/claude_home" \
    _PROMPT_FILE="$PROMPT_FILE" \
    flock /tmp/dflash_gpu.lock bash <<'INNER'
set -eo pipefail
export DFLASH27B_KV_K=tq3_0
export DFLASH27B_KV_V=tq3_0
export GGML_CUDA_NO_VMM=1
SERVER_BIN="/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/build/dflash_server"
TARGET="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT="/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf"
PFLASH_DRAFTER="/home/peppi/models/Qwen3-0.6B-BF16.gguf"
HARNESS_DIR="/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/harness/clients"
CLAUDE_BIN="/home/peppi/.local/bin/claude"
HOST=127.0.0.1
PORT=18080
PROXY_PORT=18082
MODEL_ID="luce-dflash"
API_KEY="sk-lucebox"
BASE_URL="http://$HOST:$PORT"
CLAUDE_TIMEOUT=600

# ── Start dflash server ──────────────────────────────────────────────────
"$SERVER_BIN" "$TARGET" \
    --draft "$DRAFT" \
    --prefill-drafter "$PFLASH_DRAFTER" \
    --host $HOST --port $PORT \
    --max-ctx 98304 --max-tokens 512 \
    --model-name "$MODEL_ID" \
    --ddtree --ddtree-budget 16 \
    --prefill-compression always \
    --prefill-keep-ratio "$_KEEP" \
    > "$_SLOG" 2>&1 &
SPID=$!

# Wait for server health
for i in $(seq 1 120); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then break; fi
    sleep 1
    if ! kill -0 "$SPID" 2>/dev/null; then
        echo "server died" >&2; tail -n 40 "$_SLOG" >&2; exit 1
    fi
    if [[ $i -eq 120 ]]; then echo "server timeout" >&2; exit 1; fi
done
echo "server up (pid=$SPID)"

# ── Optionally start session-inject proxy ────────────────────────────────
PPID_VAR=""
CLIENT_URL="$BASE_URL"
if [[ -n "$_SID" ]]; then
    python3 "$HARNESS_DIR/session_inject_proxy.py" \
        --host $HOST \
        --port $PROXY_PORT \
        --upstream "$BASE_URL" \
        --session-id "$_SID" \
        >> "$_PLOG" 2>&1 &
    PPID_VAR=$!
    for i in $(seq 1 10); do
        if curl -fsS "http://$HOST:$PROXY_PORT/health" >/dev/null 2>&1; then break; fi
        sleep 1
        if ! kill -0 "$PPID_VAR" 2>/dev/null; then
            echo "proxy died" >&2; cat "$_PLOG" >&2; exit 1
        fi
    done
    CLIENT_URL="http://$HOST:$PROXY_PORT"
    echo "proxy up on $CLIENT_URL (session=$_SID)"
fi

# ── Run claude CLI against server (or proxy) ─────────────────────────────
PROMPT="$(<"$_PROMPT_FILE")"
mkdir -p "$_CHOME"
HOME="$_CHOME" \
ANTHROPIC_API_KEY="$API_KEY" \
ANTHROPIC_BASE_URL="$CLIENT_URL" \
CLAUDE_CODE_API_BASE_URL="$CLIENT_URL" \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
CLAUDE_CODE_DISABLE_TELEMETRY=1 \
CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
timeout "${CLAUDE_TIMEOUT}s" "$CLAUDE_BIN" \
    --print --output-format json \
    --model "$MODEL_ID" --tools none \
    --permission-mode dontAsk --no-session-persistence \
    "$PROMPT" </dev/null > "$_COUT" 2>&1 || true

# ── Tear down proxy + server ─────────────────────────────────────────────
if [[ -n "$PPID_VAR" ]] && kill -0 "$PPID_VAR" 2>/dev/null; then
    kill "$PPID_VAR" 2>/dev/null || true
    wait "$PPID_VAR" 2>/dev/null || true
fi
kill "$SPID" 2>/dev/null || true
wait "$SPID" 2>/dev/null || true
INNER

    local t1; t1=$(date +%s)
    local wall=$((t1 - t0))

    # OK_DONE marker
    local ok_done="NO"
    if grep -q "$MARKER" "$cout" 2>/dev/null; then ok_done="YES"; fi

    # accept_rate
    local ar; ar=$(grep 'spec-decode' "$slog" 2>/dev/null | \
        grep -oE '\(([0-9.]+)%\)' | tail -1 | tr -d '()%' || echo "N/A")
    [[ -z "$ar" ]] && ar="N/A"

    # drafter_fwd timing
    local dfwd; dfwd=$(grep '\[drafter\] forward+score in' "$slog" 2>/dev/null | \
        grep -oE 'in [0-9.]+s' | awk '{s+=$2*1000; n++} END{if(n) printf "%.0f ms (n=%d)",s/n,n; else print "N/A"}' || echo "N/A")
    [[ -z "$dfwd" ]] && dfwd="N/A"

    # bandit log lines
    local bandit; bandit=$(grep '\[pflash-bandit\]' "$slog" 2>/dev/null || echo "none")

    {
        echo "seed=$SEED_LABEL"
        echo "prompt=$PROMPT_BASENAME"
        echo "label=$label"
        echo "keep_ratio=$keep"
        echo "session_id=$sid"
        echo "wall_s=$wall"
        echo "ok_done=$ok_done"
        echo "accept_rate=$ar"
        echo "mean_drafter_fwd_ms=$dfwd"
        echo "bandit_log:"
        echo "$bandit"
    } | tee "$mfile" | tee -a "$RESULTS_DIR/run.log"

    echo "[$SEED_LABEL/$label] wall=${wall}s ok=$ok_done ar=$ar" | tee -a "$RESULTS_DIR/run.log"
}

# ─── Run the three conditions ────────────────────────────────────────────────
run_condition "A_fixed_low"  "0.05" ""
run_condition "B_fixed_high" "0.20" ""
run_condition "C_bandit"     "0.10" "claude_code_${SESSION_SUFFIX}"

echo "=== Day 5 Seeds [$SEED_LABEL] done $(date -Is) ===" | tee -a "$RESULTS_DIR/run.log"

# ─── Print summary table ─────────────────────────────────────────────────────
echo ""
echo "=== SUMMARY [$SEED_LABEL] ==="
printf "%-18s %10s %8s %12s %8s  %s\n" "Condition" "wall_s" "ok_done" "accept_rate" "keep" "bandit"
for cond in A_fixed_low B_fixed_high C_bandit; do
    mf="$RESULTS_DIR/$cond/metrics.txt"
    if [[ -f "$mf" ]]; then
        wall=$(grep "^wall_s=" "$mf" | cut -d= -f2)
        ok=$(grep "^ok_done=" "$mf" | cut -d= -f2)
        ar=$(grep "^accept_rate=" "$mf" | cut -d= -f2)
        keep=$(grep "^keep_ratio=" "$mf" | cut -d= -f2)
        sid=$(grep "^session_id=" "$mf" | cut -d= -f2)
        bandit_note=""
        if [[ -n "$sid" ]]; then bandit_note="yes"; else bandit_note="-"; fi
        printf "%-18s %10s %8s %12s %8s  %s\n" "$cond" "$wall" "$ok" "$ar" "$keep" "$bandit_note"
    fi
done
