#!/usr/bin/env bash
# Phase 3 RED test: --draft-method mtp produces byte-identical output to
# --draft-method none (target-only) at temp=0 seed=0.
#
# Before Phase 3 GREEN: --draft-method mtp is unrecognized; binary exits ≥1.
# After Phase 3 GREEN: token sequences match exactly; this passes.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/dflash/build/test_gemma4_dflash"
TARGET="${MTP_TARGET:-$ROOT/models/gemma-4-31B-it-Q4_K_M.gguf}"
MTP="${MTP_HEAD:-$ROOT/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf}"
PROMPT_FILE="${MTP_PROMPT:-$ROOT/dflash/test/data/chat_2k.txt}"

# Skip cleanly if any input is missing
if [ ! -x "$BIN" ]; then
    echo "[skip] test_gemma4_dflash binary missing at $BIN — build first"
    exit 77
fi
if [ ! -f "$TARGET" ]; then
    echo "[skip] target GGUF missing at $TARGET"
    exit 77
fi
if [ ! -f "$MTP" ]; then
    echo "[skip] MTP head missing at $MTP — Phase 0 spike not done"
    exit 77
fi
if [ ! -f "$PROMPT_FILE" ]; then
    echo "[skip] prompt file missing at $PROMPT_FILE"
    exit 77
fi

OUT_BASELINE=/tmp/mtp_e2e_baseline.tokens
OUT_MTP=/tmp/mtp_e2e_mtp.tokens
COMMON_ARGS=(
    --model "$TARGET"
    --kv-k tq3_0 --kv-v tq3_0
    --pflash
    --tokens-file "$PROMPT_FILE"
    --n-predict 64
    --temp 0
    --seed 0
    --print-tokens-only
)

echo "[run] target-only (baseline)"
"$BIN" "${COMMON_ARGS[@]}" --draft-method none > "$OUT_BASELINE" 2>&1 || {
    echo "FAIL: baseline run errored — see $OUT_BASELINE"
    tail -10 "$OUT_BASELINE"
    exit 1
}

echo "[run] --draft-method mtp"
"$BIN" "${COMMON_ARGS[@]}" --draft-method mtp --mtp "$MTP" > "$OUT_MTP" 2>&1 || {
    echo "FAIL: --draft-method mtp errored (expected before Phase 3 GREEN)"
    tail -10 "$OUT_MTP"
    exit 1
}

# Extract just the token IDs (assume binary outputs one token id per line on stdout)
grep -E '^[0-9]+$' "$OUT_BASELINE" > /tmp/mtp_e2e_baseline.ids
grep -E '^[0-9]+$' "$OUT_MTP"      > /tmp/mtp_e2e_mtp.ids

echo "[diff] comparing token streams"
if diff -u /tmp/mtp_e2e_baseline.ids /tmp/mtp_e2e_mtp.ids > /tmp/mtp_e2e_diff.txt ; then
    echo "[red->green] token streams identical (Phase 3 GREEN)"
    exit 0
else
    LEN_B=$(wc -l < /tmp/mtp_e2e_baseline.ids)
    LEN_M=$(wc -l < /tmp/mtp_e2e_mtp.ids)
    DIFF_LINES=$(wc -l < /tmp/mtp_e2e_diff.txt)
    echo "FAIL: streams differ"
    echo "  baseline tokens: $LEN_B"
    echo "  mtp      tokens: $LEN_M"
    echo "  diff lines:     $DIFF_LINES"
    head -30 /tmp/mtp_e2e_diff.txt
    exit 1
fi
