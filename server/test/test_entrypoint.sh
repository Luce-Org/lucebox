#!/usr/bin/env bash
# Drives scripts/entrypoint.sh against a fake luce_server that records its argv
# and answers --list-devices from FAKE_PROBE.

set -euo pipefail

ENTRYPOINT="${1:?usage: test_entrypoint.sh <entrypoint.sh>}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

TARGET="$TMP_DIR/model.gguf"
FAKE_SERVER="$TMP_DIR/luce_server"
touch "$TARGET"
mkdir -p "$TMP_DIR/drafts"
touch "$TMP_DIR/drafts/dflash-draft-3.6-q8_0.gguf" \
      "$TMP_DIR/drafts/DeepSeek-V4-Flash-DSpark-draft.gguf" \
      "$TMP_DIR/drafts/model-mmproj-BF16.gguf"

cat >"$FAKE_SERVER" <<'FAKE'
#!/usr/bin/env bash
if [ "${1:-}" = --list-devices ]; then
    printf '%s\n' "${FAKE_PROBE:-}"
    printf 'PROBE_ARG=%s\n' "$@" >&2
    exit 0
fi
printf 'SERVER_ARG=%s\n' "$@"
printf 'ENV_LUCE_TARGET_DEVICE=%s\n' "${LUCE_TARGET_DEVICE:-}"
FAKE
chmod +x "$FAKE_SERVER"

DUAL_AMD_DS4='device hip:0 arch=gfx1201 type=discrete total_mib=32624 free_mib=32000 name=AMD Radeon AI PRO R9700
device hip:1 arch=gfx1151 type=integrated total_mib=98304 free_mib=98000 name=AMD Radeon Graphics
model arch=deepseek4 size_mib=93741 path=/models/ds4.gguf
auto hip:1 fits=yes total_mib=98304 reason=first integrated GPU that fits the model'
DUAL_AMD_QWEN='device hip:0 arch=gfx1201 type=discrete total_mib=32624 free_mib=32000 name=AMD Radeon AI PRO R9700
device hip:1 arch=gfx1151 type=integrated total_mib=98304 free_mib=98000 name=AMD Radeon Graphics
model arch=qwen35 size_mib=13592 path=/models/qwen.gguf
auto hip:0 fits=yes total_mib=32624 reason=first discrete GPU that fits the model'
RTX_24G='device cuda:0 arch=sm_86 type=discrete total_mib=24576 free_mib=24000 name=NVIDIA GeForce RTX 3090
model arch=qwen35 size_mib=15000 path=/models/qwen.gguf
auto cuda:0 fits=yes total_mib=24576 reason=first discrete GPU that fits the model'

run_entrypoint() {  # run_entrypoint [VAR=value ...] -- [entrypoint args...]
    local env_args=() unset_args=() var
    while [ $# -gt 0 ] && [ "$1" != -- ]; do env_args+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    # The entrypoint reads many LUCE_* variables (LUCE_BUDGET, LUCE_PORT, ...);
    # none from the caller's shell may reach it.
    for var in $(compgen -e); do
        case "$var" in LUCE_*) unset_args+=(-u "$var") ;; esac
    done
    env "${unset_args[@]}" \
        LUCE_DIR="$TMP_DIR" \
        LUCE_TARGET="$TARGET" \
        LUCE_DRAFT="$TMP_DIR/no-draft" \
        LUCE_SERVER_BIN="$FAKE_SERVER" \
        "${env_args[@]}" \
        bash "$ENTRYPOINT" "$@" 2>"$TMP_DIR/stderr"
}

fail() {
    echo "FAIL: $*" >&2
    echo "--- entrypoint stderr of the last run:" >&2
    cat "$TMP_DIR/stderr" >&2
    exit 1
}

has_pair() {  # has_pair <output> <flag> <value>
    awk -v f="SERVER_ARG=$2" -v v="SERVER_ARG=$3" '
        prev == f && $0 == v { found = 1 } { prev = $0 }
        END { exit(found ? 0 : 1) }' <<<"$1"
}
has_flag() { grep -Fxq "SERVER_ARG=$2" <<<"$1"; }
last_value() {  # value after the last occurrence of a flag
    awk -v f="SERVER_ARG=$2" 'prev == f { v = $0 } { prev = $0 }
        END { sub(/^SERVER_ARG=/, "", v); print v }' <<<"$1"
}

# ── native cache defaults stay with luce_server ─────────────────────────────
out="$(run_entrypoint --)"
for flag in --prefix-cache-slots --prefill-cache-slots; do
    has_flag "$out" "$flag" && fail "entrypoint overrides the native cache default with $flag"
done
out="$(run_entrypoint LUCE_PREFIX_CACHE_SLOTS=0 --)"
has_pair "$out" --prefix-cache-slots 0 || fail "explicit LUCE_PREFIX_CACHE_SLOTS=0 dropped"
out="$(run_entrypoint LUCE_PREFIX_CACHE_SLOTS=4 LUCE_PREFILL_CACHE_SLOTS=2 -- serve)"
has_pair "$out" --prefix-cache-slots 4 || fail "LUCE_PREFIX_CACHE_SLOTS not forwarded"
has_pair "$out" --prefill-cache-slots 2 || fail "LUCE_PREFILL_CACHE_SLOTS not forwarded"

# ── no arguments serves, with auto device selection by default ─────────────
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" --)"
grep -Fxq "ENV_LUCE_TARGET_DEVICE=auto" <<<"$out" || fail "LUCE_TARGET_DEVICE=auto not exported"
has_pair "$out" --max-ctx 131072 || fail "context not sized from the auto device (96 GB)"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_TARGET_DEVICE=hip:0 --)"
grep -Fxq "ENV_LUCE_TARGET_DEVICE=hip:0" <<<"$out" || fail "explicit LUCE_TARGET_DEVICE replaced"
has_pair "$out" --max-ctx 98304 || fail "context not sized from LUCE_TARGET_DEVICE (31.9 GB)"
out="$(run_entrypoint FAKE_PROBE="$RTX_24G" --)"
has_pair "$out" --max-ctx 98304 || fail "24 GB tier not applied"
out="$(run_entrypoint FAKE_PROBE="" --)"
has_pair "$out" --max-ctx 16384 || fail "fallback context without a probe"

# ── server flags pass through and win ──────────────────────────────────────
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" -- serve --target-device hip:1 --max-ctx 4096)"
has_pair "$out" --target-device hip:1 || fail "--target-device not forwarded"
[ "$(last_value "$out" --max-ctx)" = 4096 ] || fail "operator --max-ctx does not win"
[ "$(grep -c '^SERVER_ARG=--max-ctx$' <<<"$out")" = 1 ] ||
    fail "entrypoint forwards its own --max-ctx next to the operator's"
grep -q "max_ctx=4096" "$TMP_DIR/stderr" || fail "startup log does not show the operator's --max-ctx"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" -- --target-devices hip:0,hip:1)"
has_pair "$out" --target-devices hip:0,hip:1 || fail "--target-devices not forwarded"
grep -q "device=hip:0,hip:1" "$TMP_DIR/stderr" || fail "startup log does not show the --target-devices placement"
grep -q "target hip:1 " "$TMP_DIR/stderr" && fail "startup log claims the auto device with --target-devices"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_QWEN" LUCE_DRAFT="$TMP_DIR/drafts" LUCE_BUDGET=7 LUCE_PORT=9999 --)"
has_pair "$out" --ddtree-budget 7 || fail "LUCE_BUDGET passed to run_entrypoint is ignored"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" -- --chunk 2048)"
has_pair "$out" --chunk 2048 || fail "bare flags do not start the server"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_ARGS="--chunk 1024 --peer-access" --)"
has_pair "$out" --chunk 1024 || fail "LUCE_ARGS not split into flags"
has_flag "$out" --peer-access || fail "LUCE_ARGS boolean flag lost"

# ── profiles own their context size ────────────────────────────────────────
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" -- --profile ds4-strix)"
has_pair "$out" --profile ds4-strix || fail "--profile not forwarded"
has_flag "$out" --max-ctx && fail "entrypoint --max-ctx overrides the profile"
grep -q "device=hip:1" "$TMP_DIR/stderr" && fail "entrypoint reports the auto device over the profile's"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_PROFILE=ds4-strix LUCE_MAX_CTX=65536 --)"
has_pair "$out" --profile ds4-strix || fail "LUCE_PROFILE not forwarded"
has_pair "$out" --max-ctx 65536 || fail "explicit LUCE_MAX_CTX dropped with a profile"

out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_PROFILE=ds4-strix -- --profile ds4-r9700-strix)"
[ "$(grep -Fxc "SERVER_ARG=--profile" <<<"$out")" = 1 ] || fail "two --profile flags reach luce_server"
has_pair "$out" --profile ds4-r9700-strix || fail "command-line --profile does not replace LUCE_PROFILE"

# ── drafts follow the target architecture ──────────────────────────────────
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_DRAFT="$TMP_DIR/drafts" --)"
[ "$(last_value "$out" --draft)" = "$TMP_DIR/drafts/DeepSeek-V4-Flash-DSpark-draft.gguf" ] ||
    fail "DeepSeek V4 did not get its DSpark drafter"
has_flag "$out" --ddtree && fail "DDTree requested for a DSpark drafter"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_QWEN" LUCE_DRAFT="$TMP_DIR/drafts" --)"
[ "$(last_value "$out" --draft)" = "$TMP_DIR/drafts/dflash-draft-3.6-q8_0.gguf" ] ||
    fail "Qwen did not get the DFlash draft"
has_pair "$out" --ddtree-budget 22 || fail "DDTree budget missing for DFlash"
rm "$TMP_DIR/drafts/dflash-draft-3.6-q8_0.gguf"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_QWEN" LUCE_DRAFT="$TMP_DIR/drafts" --)"
has_flag "$out" --draft && fail "DSpark or mmproj file used as a Qwen draft"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_DRAFT=none --)"
has_flag "$out" --draft && fail "LUCE_DRAFT=none still passed a draft"
# DeepSeek V4 paged (concurrent) serving is autoregressive and rejects a drafter.
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_DRAFT="$TMP_DIR/drafts" -- --max-concurrency 4)"
has_flag "$out" --draft && fail "DSpark drafter passed to concurrent DeepSeek V4"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_DRAFT="$TMP_DIR/drafts" \
    LUCE_ARGS="--paged-attention" --)"
has_flag "$out" --draft && fail "DSpark drafter passed to paged DeepSeek V4"
out="$(run_entrypoint FAKE_PROBE="$DUAL_AMD_DS4" LUCE_DRAFT="$TMP_DIR/drafts" -- --max-concurrency 1)"
has_flag "$out" --draft || fail "single-lane DeepSeek V4 lost its DSpark drafter"

# ── retired variables do not reach the server ──────────────────────────────
out="$(run_entrypoint LUCE_THINK_SOFT_CLOSE_MIN_RATIO=0.5 LUCE_DEBUG_THINKING_LOGITS=1 --)"
grep -q "soft-close\|debug-thinking" <<<"$out" && fail "retired flags forwarded"

# ── devices subcommand ─────────────────────────────────────────────────────
probe_args="$(env LUCE_DIR="$TMP_DIR" LUCE_TARGET="$TARGET" LUCE_SERVER_BIN="$FAKE_SERVER" \
    bash "$ENTRYPOINT" devices 2>&1 >/dev/null)"
grep -Fxq "PROBE_ARG=$TARGET" <<<"$probe_args" || fail "devices did not pass the target"

echo "entrypoint: PASS"
