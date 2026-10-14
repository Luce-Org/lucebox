#!/bin/bash
# usage: KV_GATE_DIR=<dir> KV_MODEL=<model.gguf> run_scenarios.sh BINARY [1|2|3|all] [TAG]
# GPU regression scenarios for KVFlash session restore (see scripts/scenarios.py). One server at a time (flock).
# Required: KV_GATE_DIR, KV_MODEL. Optional: KV_MMPROJ (scenario 3 skips without it), KV_PORT.
# The caller picks the GPU (export CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES); this script sets no device variable.
set -u
: "${KV_GATE_DIR:?set KV_GATE_DIR to the output directory}"
: "${KV_MODEL:?set KV_MODEL to the target model .gguf}"
BIN=$(readlink -f "$1"); WHICH=${2:-all}; TAG=${3:-$(date +%Y%m%d-%H%M%S)}
HERE=$(dirname "$(readlink -f "$0")")
G=$KV_GATE_DIR; D=$G/out/scenarios-$TAG; mkdir -p "$D" "$G/tmp"
export TMPDIR=$G/tmp KV_ROOT=$D KV_BIN=$BIN KV_PORT=${KV_PORT:-18291} KV_MODEL
exec 9>"$G/server.lock"; flock -n 9 || { echo "another gate holds the lock" >&2; exit 3; }
pgrep -x luce_server >/dev/null && { echo "a luce_server is already running" >&2; exit 3; }
echo "bin=$BIN md5=$(md5sum "$BIN" | cut -d' ' -f1) scenarios=$WHICH out=$D" | tee "$D/provenance.txt"
cd "$HERE/scripts" && python3 -I scenarios.py "$WHICH" 2>&1 | tee "$D/scenarios.out"
exit "${PIPESTATUS[0]}"
