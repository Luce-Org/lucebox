#!/usr/bin/env bash
# N-sweep NIAH bench: baseline + ee3 + ee5 + ee7 @ 32K / 64K / 128K
#
# Each server boot is flock-serialized on /tmp/lucebox-gpu.lock.
# Pre-flight: NIAH case files must exist under CASES_DIR (see PLAN.md for gen command).
#
# Usage: dflash/bench/run_ee_n_sweep.sh [output_dir] [cases_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTDIR="${1:-$ROOT/dflash/bench/results/2026-05-25_ee_n_sweep}"
CASES_DIR="${2:-/tmp}"

mkdir -p "$OUTDIR"

echo "=== ee N-sweep NIAH start ($(date)) ==="
echo "    out-dir:   $OUTDIR"
echo "    cases-dir: $CASES_DIR"

# Verify case files exist before acquiring the GPU lock.
for ctx in 32768 65536 131072; do
    f="$CASES_DIR/niah_${ctx}.jsonl"
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing $f"
        echo "Generate with:"
        echo "  python3 $ROOT/pflash/tests/niah_gen.py --context $ctx --n 3 -o $f"
        exit 1
    fi
done

# Run all conditions in a single flock-serialized call.
# The Python script handles per-server serialization internally (one server per case).
flock -w 1800 /tmp/lucebox-gpu.lock -c "
    python3 $ROOT/dflash/bench/run_ee_n_sweep_niah.py \
        --out-dir '$OUTDIR' \
        --cases-dir '$CASES_DIR'
" || { echo "FAIL: GPU lock timeout or sweep error"; exit 1; }

echo "=== N-sweep NIAH done. Results under $OUTDIR ($(date)) ==="
