#!/bin/bash
# Top-k/top-p LM-head calibration over all prompt bins in calib/.
# Chain verify path (no --ddtree) so target_tok[i] aligns with draft row i+1.
# Each test_dflash run prints its own coverage tables; aggregate with
# calib/aggregate.py over the captured log.
#
#   GPU=0 NGEN=256 bash calib/run_calib.sh | tee calib/out/calib.log
set -u
BIN=${BIN:-server/build/test_dflash}
TARGET=${TARGET:-server/models/Qwen3.6-27B-Q4_K_M.gguf}
DRAFT=${DRAFT:-server/models/draft/dflash-draft-3.6-q4_k_m.gguf}
NGEN=${NGEN:-256}
GPU=${GPU:-0}
MAXCTX=${MAXCTX:-1024}
mkdir -p calib/out
for p in calib/*.bin; do
  name=$(basename "$p" .bin)
  echo "######## $name ########"
  CUDA_VISIBLE_DEVICES=$GPU DFLASH_TOPK_CALIB=1 \
    "$BIN" "$TARGET" "$DRAFT" "$p" "$NGEN" "calib/out/${name}_out.bin" \
    --fast-rollback --max-ctx="$MAXCTX" 2>&1 \
    | grep -E "topk-calib|out of memory|failed|CUDA error"
done
