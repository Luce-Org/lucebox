#!/usr/bin/env bash
# Usage: launch_arm.sh <ARM_NAME> <DRAFT_CTX_MAX> <FEAT_RING_CAP>
# Launches dflash_server on port 18081 with the given knob values.
# Stdout/stderr go to bench/qwen35moe_dflash/ctxsweep/iso_<ARM>.log

set -euo pipefail

ARM="$1"
DRAFT_CTX_MAX="$2"
FEAT_RING_CAP="$3"

LOGFILE="/home/peppi/Dev/lucebox-hub/bench/qwen35moe_dflash/ctxsweep/iso_${ARM}.log"

SERVER="/home/peppi/Dev/lucebox-hub/server/build/dflash_server"
TARGET="/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
DRAFTER="/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf"
TEMPLATE="/home/peppi/models/qwen3-coder-chat-template.jinja"

echo "[launch_arm] ARM=${ARM} DRAFT_CTX_MAX=${DRAFT_CTX_MAX} FEAT_RING_CAP=${FEAT_RING_CAP}" | tee -a "$LOGFILE"
echo "[launch_arm] PID=$$  started at $(date)" | tee -a "$LOGFILE"

DFLASH_DRAFT_CTX_MAX="$DRAFT_CTX_MAX" \
DFLASH_FEAT_RING_CAP="$FEAT_RING_CAP" \
  "$SERVER" \
    "$TARGET" \
    --draft "$DRAFTER" \
    --host 127.0.0.1 --port 18081 \
    --max-ctx 40960 --max-tokens 200 \
    --fa-window 0 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --chat-template-file "$TEMPLATE" \
    --model-name luce-dflash \
    --lazy-draft \
    >> "$LOGFILE" 2>&1
