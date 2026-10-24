#!/usr/bin/env bash
# GPU e2e: KVFlash pooled-session restore (LUCE_KVFLASH_RESTORE_CONSUME) identity test.
#
# For each turn-1 size N, runs a 3-turn greedy conversation twice against a
# fresh server (--kvflash 8192 --max-ctx 131072):
#   LUCE_KVFLASH_RESTORE_CONSUME=0  reference: pooled snapshots are not saved, every
#                              turn re-prefills its whole prompt
#   LUCE_KVFLASH_RESTORE_CONSUME=1  turns 2 and 3 restore the pooled snapshot and
#                              prefill only the new tokens
# Asserts (a) the turn-3 text is identical across the two runs and (b) with =1,
# turns 2 and 3 prefill suffix-only: restore=true with a non-zero prefix_len in
# the "[server] chat CACHE" line, and the "[kvflash] pooled prefill: X tokens"
# line reports X close to (effective_prompt - prefix_len), not the whole prompt.
#
# Usage:
#   bash pooled_restore_identity.sh <luce_server binary> <model.gguf> [N ...]
# Default N: 20000 35000 64000. Env: PORT (18090), OUT_DIR (default
# $HOME/kvflash-pooled-restore-e2e), SLACK (suffix slack tokens, default 1024),
# CHAT_TEMPLATE (optional --chat-template-file), EXTRA_ARGS (extra server flags).
# Never writes outside the repo or $HOME.
set -uo pipefail

SERVER_BIN="${1:?usage: $0 <server binary> <model.gguf> [N ...]}"
MODEL="${2:?usage: $0 <server binary> <model.gguf> [N ...]}"
shift 2
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(20000 35000 64000)

HOST=127.0.0.1
PORT="${PORT:-18090}"
OUT_DIR="${OUT_DIR:-$HOME/kvflash-pooled-restore-e2e}"
SLACK="${SLACK:-1024}"
POOL=8192
MAX_CTX=131072
case "$OUT_DIR" in "$HOME"/*|"$PWD"/*) ;; *) echo "OUT_DIR must be under \$HOME or the repo" >&2; exit 2;; esac
[ -x "$SERVER_BIN" ] || { echo "no such server binary: $SERVER_BIN" >&2; exit 2; }
[ -f "$MODEL" ]      || { echo "no such model: $MODEL" >&2; exit 2; }
command -v jq >/dev/null || { echo "jq required" >&2; exit 2; }
mkdir -p "$OUT_DIR"

SERVER_PID=""
stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    SERVER_PID=""
}
trap stop_server EXIT

FAILS=0
fail() { echo "FAIL: $*" >&2; FAILS=$((FAILS + 1)); }

start_server() {  # $1 = consume flag, $2 = log file
    local args=("$MODEL" --host "$HOST" --port "$PORT" --max-ctx "$MAX_CTX"
                --kvflash "$POOL" --model-name luce)
    [ -n "${CHAT_TEMPLATE:-}" ] && args+=(--chat-template-file "$CHAT_TEMPLATE")
    # shellcheck disable=SC2206
    [ -n "${EXTRA_ARGS:-}" ] && args+=($EXTRA_ARGS)
    LUCE_KVFLASH_RESTORE_CONSUME="$1" "$SERVER_BIN" "${args[@]}" >"$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 300); do
        curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 2
    done
    tail -30 "$2" >&2
    return 1
}

# Deterministic filler of roughly $1 tokens (about 16 tokens per record) with
# one needle in the middle.
make_filler() {
    local n="$1" records=$(( $1 / 16 ))
    awk -v r="$records" 'BEGIN {
        for (i = 0; i < r; i++) {
            if (i == int(r / 2)) printf "NEEDLE: the secret code is QX-7319-ZEBRA. ";
            printf "Record %d: alpha-%d bravo-%d. ", i, (i * 7) % 97, (i * 13) % 89;
            if (i % 8 == 7) printf "\n";
        }
    }'
}

chat() {  # $1 = messages json file; prints the response json
    jq -n --slurpfile m "$1" \
        '{model: "luce", messages: $m[0], temperature: 0, seed: 1, max_tokens: 64, stream: false}' \
        | curl -fsS "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}

# run_conversation <consume> <N> <tag>  -> writes $OUT_DIR/<tag>.{log,t1,t2,t3,usage}
run_conversation() {
    local consume="$1" n="$2" tag="$3"
    local log="$OUT_DIR/$tag.log" msgs="$OUT_DIR/$tag.msgs.json"
    start_server "$consume" "$log" || { fail "$tag: server did not start"; stop_server; return 1; }
    make_filler "$n" > "$OUT_DIR/$tag.filler.txt"
    jq -n --rawfile f "$OUT_DIR/$tag.filler.txt" \
        '[{role: "user", content: ($f + "\nQuestion: what is the secret code? Answer in one sentence.")}]' > "$msgs"
    local follow=("Repeat the secret code, then name one record number you saw."
                  "Spell the secret code one character at a time.")
    local t resp text
    : > "$OUT_DIR/$tag.usage"
    for t in 1 2 3; do
        resp="$(chat "$msgs")" || { fail "$tag turn $t: request failed"; stop_server; return 1; }
        text="$(jq -r '.choices[0].message.content // empty' <<<"$resp")"
        printf '%s' "$text" > "$OUT_DIR/$tag.t$t"
        jq -c '{turn: '"$t"', prompt_tokens: .usage.prompt_tokens, completion_tokens: .usage.completion_tokens}' \
            <<<"$resp" >> "$OUT_DIR/$tag.usage"
        [ -n "$text" ] || { fail "$tag turn $t: empty completion"; stop_server; return 1; }
        if [ "$t" -lt 3 ]; then
            jq --arg a "$text" --arg q "${follow[$((t - 1))]}" \
                '. + [{role: "assistant", content: $a}, {role: "user", content: $q}]' "$msgs" > "$msgs.new" \
                && mv "$msgs.new" "$msgs"
        fi
    done
    stop_server
}

# Per-turn facts from the server log (one entry per request, in order).
cache_lines()   { grep -E '^\[server\] chat CACHE ' "$1"; }
prefill_tokens() { sed -nE 's/^\[kvflash\] pooled prefill: ([0-9]+) tokens.*/\1/p' "$1"; }

check_suffix_only() {  # $1 = tag (consume=1 run)
    local log="$OUT_DIR/$1.log" n_cache n_pref
    n_cache="$(cache_lines "$log" | wc -l)"
    mapfile -t pref < <(prefill_tokens "$log")
    n_pref=${#pref[@]}
    if [ "$n_cache" -lt 3 ] || [ "$n_pref" -lt 3 ]; then
        fail "$1: expected 3 chat CACHE lines and 3 pooled-prefill lines, got $n_cache / $n_pref"
        return
    fi
    local t line restore prefix eff delta
    for t in 2 3; do
        line="$(cache_lines "$log" | sed -n "${t}p")"
        restore="$(sed -nE 's/.* restore=([a-z]+) .*/\1/p' <<<"$line")"
        prefix="$(sed -nE 's/.* prefix_len=([0-9]+) .*/\1/p' <<<"$line")"
        eff="$(sed -nE 's/.* effective_prompt=([0-9]+) .*/\1/p' <<<"$line")"
        delta=$(( eff - prefix ))
        echo "  $1 turn $t: restore=$restore prefix_len=$prefix effective_prompt=$eff prefilled=${pref[$((t - 1))]} (new tokens $delta)"
        [ "$restore" = "true" ] && [ "${prefix:-0}" -gt 0 ] || fail "$1 turn $t: no prefix restore (restore=$restore prefix_len=$prefix)"
        if [ "${pref[$((t - 1))]}" -gt $(( delta + SLACK )) ]; then
            fail "$1 turn $t: prefilled ${pref[$((t - 1))]} tokens, expected about $delta (suffix only)"
        fi
        [ "${pref[$((t - 1))]}" -lt $(( eff / 2 )) ] || fail "$1 turn $t: prefill is not suffix-only (${pref[$((t - 1))]} of $eff)"
    done
    local consumed fallback
    consumed="$(grep -c '^\[kvflash\] pooled restore: consume' "$log")"
    fallback="$(grep -c 'not consumable: full re-prefill' "$log")"
    echo "  $1: consume lines=$consumed fallback lines=$fallback"
    [ "$consumed" -ge 2 ] || fail "$1: expected >=2 'pooled restore: consume' lines, got $consumed (fallbacks: $fallback)"
}

for n in "${SIZES[@]}"; do
    echo "== N=$n =="
    run_conversation 0 "$n" "N$n.consume0" || continue
    run_conversation 1 "$n" "N$n.consume1" || continue
    cat "$OUT_DIR/N$n.consume1.usage"
    if cmp -s "$OUT_DIR/N$n.consume0.t3" "$OUT_DIR/N$n.consume1.t3"; then
        echo "  turn-3 output identical ($(wc -c < "$OUT_DIR/N$n.consume1.t3") bytes)"
    else
        fail "N=$n: turn-3 output differs between CONSUME=0 and =1 ($OUT_DIR/N$n.consume{0,1}.t3)"
    fi
    # turn 1/2 identity is informative, not asserted separately (turn 3 depends on both)
    check_suffix_only "N$n.consume1"
    # Baseline sanity: with =0 nothing is restored from a pooled snapshot, so
    # turn 2 must prefill (about) the whole prompt.
    mapfile -t base < <(prefill_tokens "$OUT_DIR/N$n.consume0.log")
    echo "  consume0 pooled-prefill tokens per turn: ${base[*]:-none}"
done

if [ "$FAILS" -eq 0 ]; then echo "PASS"; exit 0; fi
echo "$FAILS check(s) failed; logs in $OUT_DIR" >&2
exit 1
