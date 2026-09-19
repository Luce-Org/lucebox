#!/usr/bin/env bash
set -euo pipefail
export VLLM_QUARK_NATIVE_INT4_VERIFY_TILES=1
export VLLM_USE_TRITON_GEMMA_RMSNORM=0 VLLM_ROCM_NATIVE_HEAD=0
export VLLM_PLUGINS=''
exec /bin/bash /root/vllm-native-test/run-dflash-server.sh \
  --served-model-name qwen3.8-27b-swift \
  --max-model-len -1 --max-num-seqs 4 --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8_per_token_head \
  --kv-offloading-size 8 --kv-offloading-backend native \
  --speculative-config '{"method":"dflash","model":"/code/models/qwen38/dflash2","num_speculative_tokens":7}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32]}' \
  "$@"
