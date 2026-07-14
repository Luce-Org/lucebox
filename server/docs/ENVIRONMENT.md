# Server environment variables

Policy (2026-07): **new features ship as CLI flags or defaults, not env vars.**
Environment variables are reserved for two cases:

1. **Burn-in kill switches** for freshly landed defaults - documented here with
   the intent to delete them once the feature has soaked.
2. **Debug instrumentation** (profilers, stats) - zero-cost when unset, never
   required for correct serving.

Anything else in the inventory below is legacy surface: prefer the CLI flag
where one exists, and treat undocumented variables as internal. The
consolidation of this list into CLI flags is tracked as follow-up work.

## Documented variables

| Variable | Default | Purpose |
|---|---|---|
| `DFLASH_DRAFT_KV` | 1 | KILL SWITCH (remove after burn-in): =0 restores the legacy per-step drafter window recompute instead of the ring cache. |
| `DFLASH_LAGUNA_SWA_RING` | 1 | KILL SWITCH (remove after burn-in): =0 keeps SWA layers on pool-sized caches under KVFlash. |
| `DFLASH_PROF` | unset | DEBUG: comma list of profilers (step,verify,prefill). Replaces DFLASH_LAGUNA_{STEP,VERIFY,PREFILL}_PROF. |
| `GGML_CUDA_GRAPH_STATS` | unset | DEBUG: per-graph CUDA-graph replay/capture/eager counters. |
| `GGML_CUDA_GRAPH_STATS_EVERY` | 200 | DEBUG: print period for the stats above (clamped to >=1). |
| `DFLASH_ADAPTIVE_K_TAU` | 0 = off | Prefer the CLI: --adaptive-experts [tau]. Cumulative combine-weight threshold for per-token expert gating. |
| `DFLASH_ADAPTIVE_K_DENSE` | per-model default | CSV of MoE layers kept dense under adaptive-K (DFlash capture layers). Warned-inert on families that do not thread layer indices yet. |
| `DFLASH_MMID_GROUPED` | unset | Grouped MUL_MAT_ID kernel for small verify batches; candidate for CLI promotion. |
| `DFLASH_KVFLASH` | unset | Prefer the CLI: `--kvflash` (token count or `auto`). |

## Full inventory (generated)

`grep -rE 'getenv\("[A-Z0-9_]+"\)' server/src` - regenerate when adding or removing variables.

- `DFLASH27B_CHUNKED` - qwen35_target_graph.cpp
- `DFLASH27B_DRAFT_FP16` - draft_safetensors_loader.cpp
- `DFLASH27B_DRAFT_SWA` - server_main.cpp
- `DFLASH27B_KV_F16` - kv_quant.cpp
- `DFLASH27B_KV_K` - kv_quant.cpp, laguna_backend.cpp
- `DFLASH27B_KV_Q4` - kv_quant.cpp
- `DFLASH27B_KV_TQ3` - kv_quant.cpp, qwen3_drafter.cpp
- `DFLASH27B_KV_V` - kv_quant.cpp, laguna_backend.cpp
- `DFLASH27B_LM_HEAD_FIX` - http_server.cpp
- `DFLASH27B_PREFILL_UBATCH` - layer_split_daemon.cpp, qwen35_backend.cpp, qwen35_layer_split_adapter.cpp
- `DFLASH_ADAPTIVE_K_DENSE` - mmid_adaptive_k.h
- `DFLASH_ADAPTIVE_K_TAU` - mmid_adaptive_k.h
- `DFLASH_ADAPTIVE_WIDTH_MIN` - adaptive_verify_width.h
- `DFLASH_ADAPTIVE_WIDTH_THETA` - adaptive_verify_width.h
- `DFLASH_COLD_THREADS` - moe_expert_compute_cpu.cpp
- `DFLASH_DISABLE_DRAFT_ATTN` - draft_graph.cpp
- `DFLASH_DISABLE_DRAFT_ATTN_GATE` - draft_graph.cpp
- `DFLASH_DISABLE_DRAFT_AUX_NORMS` - draft_graph.cpp
- `DFLASH_DISABLE_DRAFT_FFN` - draft_graph.cpp
- `DFLASH_DISABLE_DRAFT_SWA` - dflash_draft_kv.cpp, draft_graph.cpp
- `DFLASH_DOMINO_ZERO_START` - domino_head.cpp
- `DFLASH_DRAFT_IPC_SHARED_BYTES` - dflash_draft_ipc.cpp
- `DFLASH_DRAFT_IPC_TRANSPORT` - dflash_draft_ipc.cpp
- `DFLASH_DRAFT_KV` - laguna_backend.cpp, qwen35_backend.cpp
- `DFLASH_DRAFT_PERSIST` - laguna_backend.cpp
- `DFLASH_DROP_COLD` - qwen35moe_backend.cpp, qwen35moe_pipelined_decode.cpp
- `DFLASH_DS4_TIMING` - deepseek4_target_shard_ipc_daemon.cpp
- `DFLASH_EXPERT_BUDGET_MB` - deepseek4_backend.cpp, laguna_backend.cpp, qwen35moe_backend.cpp
- `DFLASH_EXPERT_BUDGET_PCT` - laguna_backend.cpp
- `DFLASH_FEATURE_DTYPE` - dflash_feature_ring.cpp
- `DFLASH_FP_ALPHA` - http_server.cpp, qwen3_graph.cpp, server_main.cpp
- `DFLASH_FP_CHUNK_S` - qwen3_graph.cpp
- `DFLASH_FP_DEBUG_LAYER0` - qwen3_graph.cpp
- `DFLASH_FP_DUMP_COUNTS` - flashprefill.cpp
- `DFLASH_FP_HIP_ROW` - flashprefill_kernels.cu
- `DFLASH_FP_NOPE_TAIL` - qwen3_graph.cpp
- `DFLASH_FP_PROFILE` - flashprefill.cpp
- `DFLASH_FP_SKIP_PREWARM` - qwen3_drafter.cpp
- `DFLASH_FP_USE_BSA` - flashprefill.cpp, http_server.cpp, server_main.cpp
- `DFLASH_G4_BSA_CHUNK` - gemma4_graph.cpp
- `DFLASH_GEMMA4_LAYER_SPLIT_UBATCH` - gemma4_layer_split_adapter.cpp
- `DFLASH_GEMMA4_NO_KVPAD` - gemma4_graph.cpp
- `DFLASH_GPU_ARGMAX` - qwen35_backend.cpp
- `DFLASH_GPU_DRAFT_TOPK` - qwen35_dflash_target.cpp
- `DFLASH_GPU_SAMPLE` - geometric_sampler_cuda.cu
- `DFLASH_GPU_VERIFY_ARGMAX` - qwen35_dflash_target.cpp
- `DFLASH_IGNORE_EOS` - laguna_backend.cpp
- `DFLASH_KVFLASH` - gemma4_backend.cpp, gemma4_layer_split_adapter.cpp, kvflash_pager.h, laguna_backend.cpp, laguna_layer_split_adapter.cpp, qwen35_backend.cpp, qwen35_layer_split_adapter.cpp
- `DFLASH_KVFLASH_DRAFTER` - kvflash_pager.h
- `DFLASH_KVFLASH_MAX_POOL` - kvflash_pager.h
- `DFLASH_KVFLASH_POLICY` - kvflash_pager.h
- `DFLASH_KVFLASH_TAU` - gemma4_backend.cpp, gemma4_layer_split_adapter.cpp, laguna_backend.cpp, laguna_layer_split_adapter.cpp, qwen35_layer_split_adapter.cpp
- `DFLASH_LAGUNA_AUTO_HEAD_MAJOR` - laguna_backend.cpp
- `DFLASH_LAGUNA_CACHE_SLOTS` - laguna_backend.cpp
- `DFLASH_LAGUNA_DRAFT_PAD` - laguna_backend.cpp
- `DFLASH_LAGUNA_DSPARK` - laguna_backend.cpp
- `DFLASH_LAGUNA_DSPARK_CONFIDENCE_THRESHOLD` - laguna_backend.cpp
- `DFLASH_LAGUNA_DSPARK_TREE` - laguna_backend.cpp
- `DFLASH_LAGUNA_EXPERT_CACHE` - moe_hybrid_ffn_eval.cpp
- `DFLASH_LAGUNA_FUSED_DOMINO` - laguna_backend.cpp
- `DFLASH_LAGUNA_FUSED_DSPARK` - laguna_backend.cpp
- `DFLASH_LAGUNA_FUSED_QK` - laguna_target_loader.cpp
- `DFLASH_LAGUNA_FUSE_FFN` - laguna_backend.cpp
- `DFLASH_LAGUNA_GPU_ARGMAX` - laguna_backend.cpp
- `DFLASH_LAGUNA_GPU_REMAP` - moe_hybrid_ffn_eval.cpp
- `DFLASH_LAGUNA_HOTNESS` - laguna_backend.cpp
- `DFLASH_LAGUNA_KV_HEAD_MAJOR` - laguna_backend.cpp, laguna_target_graph.cpp
- `DFLASH_LAGUNA_LAYER_SPLIT_UBATCH` - laguna_layer_split_adapter.cpp
- `DFLASH_LAGUNA_MOE_FUSED_COMBINE` - laguna_target_graph.cpp
- `DFLASH_LAGUNA_MOE_STUB` - laguna_target_graph.cpp
- `DFLASH_LAGUNA_NEXT_PLACEMENT_OUT` - laguna_backend.cpp
- `DFLASH_LAGUNA_NO_KVPAD` - laguna_dflash_target.cpp, laguna_target_graph.cpp
- `DFLASH_LAGUNA_NO_SINGLE_GRAPH` - laguna_backend.cpp
- `DFLASH_LAGUNA_PAD_CPY` - laguna_dflash_target.cpp, laguna_target_graph.cpp
- `DFLASH_LAGUNA_PERSIST_VERIFY` - laguna_target_graph.cpp
- `DFLASH_LAGUNA_PREGATE_MAX` - laguna_backend.cpp
- `DFLASH_LAGUNA_PREGATE_TRACE` - laguna_backend.cpp
- `DFLASH_LAGUNA_PROFILE` - laguna_backend.cpp
- `DFLASH_LAGUNA_SWAP_MAX` - laguna_backend.cpp
- `DFLASH_LAGUNA_SWAP_MIN_GAIN` - laguna_backend.cpp
- `DFLASH_LAGUNA_SWA_RING` - laguna_backend.cpp
- `DFLASH_LAGUNA_TELEMETRY` - laguna_backend.cpp
- `DFLASH_LAGUNA_VERIFY_WIDTH` - laguna_backend.cpp
- `DFLASH_LAGUNA_VERIFY_WIDTH_MAX` - laguna_backend.cpp
- `DFLASH_MAX_CONTEXT` - laguna_backend.cpp, qwen35moe_backend.cpp
- `DFLASH_MMQ_FULL_BATCH_MIN` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MMQ_SUB_BATCH` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MODEL_CARDS_DIR` - model_card.cpp
- `DFLASH_MOE_COLD_BACKEND` - deepseek4_loader.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_DTYPE` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_PROFILE` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_SHARED_BYTES` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_TRANSPORT` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_THREADS` - moe_expert_compute_cpu.cpp
- `DFLASH_MOE_FIXED_SLOT_GRAPHS` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_FIXED_SLOT_MAX` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_PREFILL_HOT_SUB_BATCH` - moe_hybrid_ffn_eval.cpp
- `DFLASH_NO_MASK` - laguna_backend.cpp
- `DFLASH_NO_MOE_ROUTER_FUSE` - qwen35moe_ffn.cpp
- `DFLASH_NO_MOE_SWIGLU_FUSE` - qwen35moe_ffn.cpp
- `DFLASH_NO_PREAD` - deepseek4_loader.cpp
- `DFLASH_PROF` - prof_env.h
- `DFLASH_QWEN35MOE_CACHE_SLOTS` - qwen35moe_backend.cpp
- `DFLASH_QWEN35MOE_HOTNESS` - qwen35moe_backend.cpp
- `DFLASH_QWEN35MOE_NEXT_PLACEMENT_OUT` - qwen35moe_backend.cpp
- `DFLASH_QWEN35MOE_NO_KVPAD` - qwen35moe_pipelined_decode.cpp
- `DFLASH_QWEN35MOE_NO_ROUTED` - qwen35moe_pipelined_decode.cpp
- `DFLASH_QWEN35MOE_RUNTIME_STATS_OUT` - qwen35moe_backend.cpp
- `DFLASH_QWEN35MOE_SWAP_MAX` - qwen35moe_backend.cpp
- `DFLASH_QWEN35MOE_SWAP_MIN_GAIN` - qwen35moe_backend.cpp
- `DFLASH_QWEN35MOE_TELEMETRY` - qwen35moe_backend.cpp
- `DFLASH_QWEN35_NO_KVPAD` - graph_builders.cpp
- `DFLASH_SAMPLED_VERIFY` - laguna_backend.cpp, qwen35_backend.cpp
- `DFLASH_SHARE_DIR` - http_server.cpp
- `DFLASH_SPARK` - laguna_backend.cpp, qwen35moe_backend.cpp
- `DFLASH_SPARK_VRAM_MB` - laguna_backend.cpp, qwen35moe_backend.cpp
- `DFLASH_SV_DEBUG` - qwen35_backend.cpp
- `DFLASH_TARGET_SHARD_IPC_SHARED_BYTES` - target_shard_ipc.cpp
- `DFLASH_TARGET_SHARD_IPC_TRANSPORT` - target_shard_ipc.cpp
- `DFLASH_TOPK_PROFILE` - geometric_draft_topk_cuda.cu
- `DFLASH_TOPK_SPLIT` - geometric_draft_topk_cuda.cu
- `DFLASH_VERIFY_WIDTH` - qwen35moe_backend.cpp
- `FAST_ROLLBACK_DIAG` - qwen35_dflash_target.cpp
- `HOME` - spark_corpus.cpp
- `LUCE_QK_FUSE_LAYERS` - laguna_target_graph.cpp
- `LUCE_QK_FUSE_MODE` - laguna_target_graph.cpp
- `PFLASH_DRAFTER_EARLY_EXIT_N` - qwen3_graph.cpp
- `PFLASH_DRAFTER_SCORE_LAYERS` - qwen3_graph.cpp
- `PFLASH_FREEZE_HOT_WINDOW` - http_server.cpp
- `TMPDIR` - backend_ipc.cpp, moe_expert_compute_ipc.cpp
