#!/bin/bash
# usage: KV_GATE_DIR=<dir> KV_MODEL=<model.gguf> run_gate.sh BINARY MODE [TURNS] [TAG]
#   MODE = plain   : session gate (A LUCE_KVFLASH_SESSION=0 cold re-prefill, B LUCE_KVFLASH_SESSION=1 live restore)
#   MODE = dflash2 : the same with a DFlash2 draft model (needs KV_DRAFT=<draft.gguf>)
# Required: KV_GATE_DIR (output, lock and TMPDIR live under it), KV_MODEL. Optional: KV_DRAFT, KV_PORT.
# The caller picks the GPU (export CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES); this script sets no device variable.
# Parity: default KV type, --max-ctx 131072, --kvflash 8192, temp 0 seed 1, turn-1 ~12500 filler tokens (G4_T1),
# 10 turns, max_tokens 80. One server at a time (flock).
set -u
: "${KV_GATE_DIR:?set KV_GATE_DIR to the output directory}"
: "${KV_MODEL:?set KV_MODEL to the target model .gguf}"
BIN=$(readlink -f "$1"); MODE=$2; TURNS=${3:-10}; TAG=${4:-$(date +%Y%m%d-%H%M%S)}
HERE=$(dirname "$(readlink -f "$0")")
G=$KV_GATE_DIR; D=$G/out/$MODE-$TAG; mkdir -p "$D" "$G/tmp"
export TMPDIR=$G/tmp KV_ROOT=$D KV_BIN=$BIN KV_PORT=${KV_PORT:-18290} G4_TURNS=$TURNS KV_MODEL
if [ "$MODE" = dflash2 ]; then
  [ -n "${KV_DRAFT:-}" ] || { echo "SKIP: mode dflash2 needs KV_DRAFT=<draft.gguf>" >&2; exit 0; }
  export KV_EXTRA="--draft $KV_DRAFT"
elif [ "$MODE" != plain ]; then echo "bad MODE" >&2; exit 2; fi
exec 9>"$G/server.lock"; flock -n 9 || { echo "another gate holds the lock" >&2; exit 3; }
pgrep -x luce_server >/dev/null && { echo "a luce_server is already running" >&2; exit 3; }
echo "bin=$BIN md5=$(md5sum "$BIN" | cut -d' ' -f1) mode=$MODE turns=$TURNS out=$D" | tee "$D/provenance.txt"
cd "$HERE/scripts"
python3 -I gate4_live.py A > "$D/gate4_A.out" 2>&1 || { echo "arm A failed, see $D/gate4_A.out"; exit 1; }
python3 -I gate4_live.py B > "$D/gate4_B.out" 2>&1 || { echo "arm B failed, see $D/gate4_B.out"; exit 1; }
python3 -I cmp_gate.py "$D" | tee "$D/cmp.out"
exit "${PIPESTATUS[0]}"
