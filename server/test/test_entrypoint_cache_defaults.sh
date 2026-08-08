#!/usr/bin/env bash

set -euo pipefail

ENTRYPOINT="${1:?usage: test_entrypoint_cache_defaults.sh <entrypoint.sh>}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

TARGET="$TMP_DIR/model.gguf"
FAKE_SERVER="$TMP_DIR/dflash_server"
touch "$TARGET"

cat >"$FAKE_SERVER" <<'EOF'
#!/usr/bin/env bash
printf 'SERVER_ARG=%s\n' "$@"
EOF
chmod +x "$FAKE_SERVER"

run_entrypoint() {
    env \
        DFLASH_DIR="$TMP_DIR" \
        DFLASH_TARGET="$TARGET" \
        DFLASH_DRAFT="$TMP_DIR/no-draft" \
        DFLASH_SERVER_BIN="$FAKE_SERVER" \
        "$@" \
        bash "$ENTRYPOINT" serve 2>/dev/null
}

assert_arg_pair() {
    local output="$1"
    local flag="$2"
    local value="$3"
    if ! awk -v expected_flag="SERVER_ARG=$flag" \
             -v expected_value="SERVER_ARG=$value" '
            previous == expected_flag && $0 == expected_value { found = 1 }
            { previous = $0 }
            END { exit(found ? 0 : 1) }
        ' <<<"$output"; then
        echo "missing server argument: $flag $value" >&2
        exit 1
    fi
}

default_output="$(
    unset DFLASH_PREFIX_CACHE_SLOTS DFLASH_PREFILL_CACHE_SLOTS
    run_entrypoint
)"
for flag in --prefix-cache-slots --prefill-cache-slots --think-max-tokens; do
    if grep -Fq "SERVER_ARG=$flag" <<<"$default_output"; then
        echo "entrypoint overrides the native cache default with $flag" >&2
        exit 1
    fi
done

disabled_output="$(run_entrypoint DFLASH_PREFIX_CACHE_SLOTS=0)"
assert_arg_pair "$disabled_output" --prefix-cache-slots 0

configured_output="$(
    run_entrypoint \
        DFLASH_PREFIX_CACHE_SLOTS=4 \
        DFLASH_PREFILL_CACHE_SLOTS=2 \
        DFLASH_THINK_MAX=15488
)"
assert_arg_pair "$configured_output" --prefix-cache-slots 4
assert_arg_pair "$configured_output" --prefill-cache-slots 2
assert_arg_pair "$configured_output" --think-max-tokens 15488

# Grace Hopper / GB10 drivers can report `[N/A]` for unified GPU memory.
# The container must leave the fallback at zero instead of evaluating that
# string as a Bash arithmetic expression before the host topology is applied.
FAKE_BIN="$TMP_DIR/bin"
mkdir -p "$FAKE_BIN"
cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [[ " $* " == *" -L "* ]]; then
    printf 'GPU 0: NVIDIA GB10 (UUID: GPU-test)\n'
else
    printf '[N/A]\n'
fi
EOF
chmod +x "$FAKE_BIN/nvidia-smi"

gb10_stderr="$TMP_DIR/gb10.stderr"
env \
    PATH="$FAKE_BIN:$PATH" \
    DFLASH_DIR="$TMP_DIR" \
    DFLASH_TARGET="$TARGET" \
    DFLASH_DRAFT="$TMP_DIR/no-draft" \
    DFLASH_SERVER_BIN="$FAKE_SERVER" \
    bash "$ENTRYPOINT" serve >/dev/null 2>"$gb10_stderr"
if grep -Fq "syntax error: operand expected" "$gb10_stderr"; then
    echo "entrypoint attempted arithmetic on GB10 [N/A] memory" >&2
    exit 1
fi

echo "entrypoint cache defaults: PASS"
