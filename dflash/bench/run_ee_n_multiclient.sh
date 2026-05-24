#!/usr/bin/env bash
# Multi-client bandit-session × {baseline, ee3, ee5, ee7} × {claude_code, hermes, opencode, pi, codex}
# Each server boot is flock-serialized on /tmp/lucebox-gpu.lock.
#
# Usage: dflash/bench/run_ee_n_multiclient.sh [output_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKTREE="/home/peppi/Dev/lucebox-hub/.claude/worktrees/harness-adapters"
DRIVER="$WORKTREE/harness/client_test_runner.py"

TARGET="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFTER="/home/peppi/models/Qwen3-0.6B-BF16.gguf"
BIN="$ROOT/dflash/build/dflash_server"

OUTDIR="${1:-$ROOT/bench/results/2026-05-25_ee_n_multiclient}"
mkdir -p "$OUTDIR"

# Verify driver exists.
if [[ ! -f "$DRIVER" ]]; then
    echo "ERROR: driver not found: $DRIVER"
    echo "Check that the harness-adapters worktree is checked out."
    exit 1
fi

CLIENTS=(claude_code hermes opencode pi codex)
# format: name:EARLY_EXIT_N:SCORE_LAYERS (0 means unset -> full drafter layers)
CONDITIONS=("baseline:0:0" "ee3:3:3" "ee5:5:5" "ee7:7:7")

echo "=== ee N-sweep multi-client start ($(date)) ==="

for cond_spec in "${CONDITIONS[@]}"; do
    name="${cond_spec%%:*}"
    rest="${cond_spec#*:}"
    early_n="${rest%%:*}"
    score_n="${rest#*:}"

    cond_dir="$OUTDIR/$name"
    mkdir -p "$cond_dir"

    for client in "${CLIENTS[@]}"; do
        echo "=== $name x $client ($(date)) ==="

        # Build env vars for early-exit (unset for baseline).
        # DFLASH_SERVER_BIN: overrides harness default cpp binary path.
        # PYTHONPATH: needed for 'from harness.metrics_parser import ...' inside driver.
        export DFLASH_SERVER_BIN="$BIN"
        export PYTHONPATH="$WORKTREE"
        if [[ "$early_n" != "0" ]]; then
            export PFLASH_DRAFTER_EARLY_EXIT_N="$early_n"
            export PFLASH_DRAFTER_SCORE_LAYERS="$score_n"
        else
            unset PFLASH_DRAFTER_EARLY_EXIT_N PFLASH_DRAFTER_SCORE_LAYERS 2>/dev/null || true
        fi

        flock -w 1800 /tmp/lucebox-gpu.lock \
            python3 "$DRIVER" bandit-session \
                --client "$client" \
                --turns 3 \
                --target "$TARGET" \
                --draft "$DRAFTER" \
                --bin "$BIN" \
                --output "$cond_dir/${client}.csv" \
        2>&1 | tee "$cond_dir/${client}.log" \
            || echo "FAIL: $name x $client (see $cond_dir/${client}.log)"

        # Capture server log if the harness wrote one to the standard evidence dir.
        latest_server_log=$(ls -t "$WORKTREE"/dflash/bench/results/*_adaptive_evidence/server.log 2>/dev/null | head -1 || true)
        if [[ -n "$latest_server_log" ]]; then
            cp "$latest_server_log" "$cond_dir/${client}_server.log"
        fi
    done
done

echo "=== multi-client done. Results under $OUTDIR ($(date)) ==="
