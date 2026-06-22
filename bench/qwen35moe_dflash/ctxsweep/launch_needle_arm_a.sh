#!/bin/bash
# ARM_A_uncap: DRAFT_CTX_MAX=40960
set -euo pipefail

exec flock -x /tmp/lucebox_gpu.lock \
  env DFLASH_DRAFT_CTX_MAX=40960 DFLASH_FEAT_RING_CAP=40960 \
  /home/peppi/Dev/lucebox-hub/server/build/dflash_server \
  /home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  --draft /home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf \
  --host 127.0.0.1 --port 18081 --max-ctx 40960 --max-tokens 200 \
  --fa-window 0 --cache-type-k q4_0 --cache-type-v q4_0 \
  --chat-template-file /home/peppi/models/qwen3-coder-chat-template.jinja \
  --model-name luce-dflash --lazy-draft
