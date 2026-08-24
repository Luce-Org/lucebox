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
| `DFLASH_MMID_GROUPED_TYPES` | 7 | Grouped-kernel type mask; bit 3 (`8`) opts ROCmFP2/ROCmFP3 into the path. |
| `DFLASH_MMID_GROUPED_DEVICE` | -1 | Optional zero-based device restriction; unset/-1 applies to every eligible device. |
| `DFLASH_DS4_MOE_TP` / `DFLASH_DS4_MOE_TP_INPROC` | unset | BURN-IN: enable DeepSeek4 route-owner expert parallelism in one process. |
| `DFLASH_DS4_MOE_TP_BACKEND` / `DFLASH_MOE_TP_BACKEND` | peer runtime in a mixed build; compiled runtime otherwise | Select the in-process cold expert owner backend. |
| `DFLASH_DS4_MOE_TP_GPU` | peer backend device 0 in a mixed build; other local device otherwise | Device index within the cold DeepSeek4 expert backend. |
| `DFLASH_DS4_MOE_TP_CONCENTRATE_COLD` | unset | BURN-IN: use complete peer-owned expert layers to reduce cross-runtime joins; falls back when the placement would exceed the target budget. |
| `DFLASH_DS4_MOE_TP_PEER_HOT` | unset | BURN-IN: with `DFLASH_DS4_HOTNESS_CSV`, reserve the profile's hottest experts for the secondary owner. |
| `DFLASH_DS4_CROSS_VENDOR_OWNER_SUMS` | unset | BURN-IN: reduce each mixed-vendor owner's routed outputs locally before the final owner add. This changes floating-point association and is not the byte-identity mode. |
| `DFLASH_DS4_TP_SCHEDULE_BRANCHES` | unset | BURN-IN: expose independent mixed-vendor expert branches to the common multi-backend scheduler. |
| `DFLASH_DS4_TP_TARGETED_JOIN_SPLIT` / `DFLASH_MOE_TP_TARGETED_JOIN_SPLIT` | unset | BURN-IN: start a main-GPU split only at each peer-result join, avoiding an extra peer fence per MoE layer. |
| `DFLASH_DS4_COMP_PAD_STRIDE` | 16 | BURN-IN: compressed-KV padding bucket (`16`, `32`, `64`, or `128`); wider exact-masked buckets reduce verifier graph recapture churn. |
| `DFLASH_DS4_DISABLE_GROUPED_OUTPUT_PROJECTION` | unset | DEBUG: restore the materialized output projection when diagnosing grouped-view copies across unlike runtimes. |
| `DFLASH_DS4_DRAFT_BACKEND` / `DFLASH_DS4_DRAFT_GPU` | compiled backend / target device | Select the in-process DSpark backend and device. |
| `DFLASH_CUDA_BACKEND_PATH` / `DFLASH_HIP_BACKEND_PATH` | auto-discovered beside the executable | Explicit peer module file path for a mixed CUDA+HIP build. |
| `GGML_BATCH_PEER_COPIES` | unset | BURN-IN: batch peer-runtime copies and unlike-runtime host staging with one source wait per split. `GGML_CUDA_BATCH_PEER_COPIES` remains a compatibility alias. |
| `GGML_SCHED_PROFILE` / `GGML_SCHED_PROFILE_MIN_SPLITS` | unset / 1 | DEBUG: report scheduler splits, copy volume, submission time, and source/destination synchronization time. |
| `DFLASH_DS4_TP_FUSED_CACHE_SLOTS` | 2 | BURN-IN: number of heterogeneous verifier schedulers retained; higher values retain substantially more scratch on both GPUs. |
| `DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY` | unset | OPT-IN: bypass graph property scans only after warmup; scheduler-generation checks remain mandatory. |
| `DFLASH_DS4_ROCTX` | unset | DEBUG: on HIP builds, dynamically load ROCTX and emit semantic DS4 prefill, speculative-decode, and layer-range markers for external rocprof traces. No events, timing, or device synchronization are added. |
| `DFLASH_QWEN35_ROCTX` | unset | DEBUG: on HIP builds, dynamically load ROCTX and mark Qwen concurrent steps, graph compute, and argmax readback with live, padded, and packed-prefill shape metadata. |
| `GGML_DS4_FA_SERIAL_INDEX_SCAN` | unset | DEBUG/A-B: restore the serial indexed-attention mask scan instead of the long-context HIP parallel scan. |
| `DFLASH_MOE_PREFILL_PERSISTENT_OWNER_ALLOC` | 1 for qualified long heterogeneous prefill | KILL SWITCH: =0 restores per-layer route/owner scratch allocation. |
| `DFLASH_MOE_TP_*` / `DFLASH_MOE_HYBRID_PREFILL_EAGER` | unset | BURN-IN: model-neutral names for common heterogeneous-MoE scheduling and kernel policy. Existing `DFLASH_DS4_*` names remain compatibility aliases. |
| `DFLASH_MMID_TELEMETRY` | unset | DEBUG: report MUL_MAT_ID dispatch, MMVQ variant, and per-node graph compatibility. |
| `DFLASH_KVFLASH` | unset | Prefer the CLI: `--kvflash` (token count or `auto`). |
| `DFLASH_PREFIX_CACHE_SLOTS` | 32 | Container-entrypoint equivalent of `--prefix-cache-slots`; not read directly by the native binary. |
| `DFLASH_PREFILL_CACHE_SLOTS` | 0 | Container-entrypoint equivalent of `--prefill-cache-slots`; not read directly by the native binary. |
| `DFLASH_SPLIT_FAST_ROLLBACK` | unset | OPT-IN: exact F32 checkpoints and replay-free rollback for local qwen35 target layer splits. Prefer `--target-split-fast-rollback`; adds checkpoint VRAM (~1.65 GiB for the measured Qwen3.6-27B q=16 split). |
| `DFLASH_STALL_TOOL_PREFIX` | unset | OPT-IN: recover a stalled tool call by injecting the prepared tool prefix when generation stops after an action suffix. |
| `DFLASH_DS4_SPEC` / `DFLASH_DS4_DRAFT` / `DFLASH_DS4_DRAFT_BACKEND` / `DFLASH_DS4_DRAFT_GPU` | unset | OPT-IN: enable DeepSeek4 DSpark, select its draft GGUF, and optionally select the local drafter backend/device. See `DS4.md`. |
| `DFLASH_DS4_CUDA_LAYERS` | auto | Override the DeepSeek4 heterogeneous layer-split heuristic. See `DS4.md`. |

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
- `DFLASH_CUDA_BACKEND_PATH` - dynamic_backend.cpp
- `DFLASH_CUDA_MMVQ_MOE_ALIGN_SHARED_IDS` - moe_hybrid_ffn_eval.cpp
- `DFLASH_CUDA_MMVQ_MOE_KERNEL` - moe_hybrid_ffn_eval.cpp
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
- `DFLASH_DS4_ADAPTIVE_WIDTH` - deepseek4_dspark_spec.cpp
- `DFLASH_DS4_COMP_PAD_STRIDE` - deepseek4_graph.cpp
- `DFLASH_DS4_CROSS_VENDOR_OWNER_SUMS` - deepseek4_fused_verify.inc
- `DFLASH_DS4_CUDA_LAYERS` - deepseek4_layer_split_adapter.cpp
- `DFLASH_DS4_DENSE_TP_MASK` - deepseek4_loader.cpp
- `DFLASH_DS4_DENSE_TP_STRIX_FRACTION` - deepseek4_loader.cpp
- `DFLASH_DS4_DISABLE_GROUPED_OUTPUT_PROJECTION` - deepseek4_graph.cpp
- `DFLASH_DS4_DRAFT` - deepseek4_backend.cpp
- `DFLASH_DS4_DRAFT_BACKEND` - deepseek4_backend.cpp
- `DFLASH_DS4_DRAFT_GPU` - deepseek4_backend.cpp
- `DFLASH_DS4_DSPARK_DEBUG` - deepseek4_graph.cpp
- `DFLASH_DS4_FUSED_VERIFY` - deepseek4_dspark_spec.cpp, deepseek4_loader.cpp
- `DFLASH_DS4_HOTNESS_CSV` - deepseek4_backend.cpp
- `DFLASH_DS4_MOE_TP` - deepseek4_backend.cpp
- `DFLASH_DS4_MOE_TP_BACKEND` - deepseek4_backend.cpp
- `DFLASH_DS4_MOE_TP_CONCENTRATE_COLD` - deepseek4_backend.cpp
- `DFLASH_DS4_MOE_TP_GPU` - deepseek4_backend.cpp
- `DFLASH_DS4_MOE_TP_INPROC` - deepseek4_backend.cpp
- `DFLASH_DS4_MOE_TP_PEER_HOT` - deepseek4_backend.cpp
- `DFLASH_DS4_ROUTING_STATS_OUT` - deepseek4_backend.cpp
- `DFLASH_DS4_ROCTX` - deepseek4_roctx.cpp
- `DFLASH_QWEN35_ROCTX` - qwen35_roctx.cpp
- `DFLASH_DS4_SEQ_VERIFY` - deepseek4_dspark_spec.cpp
- `DFLASH_DS4_SPEC` - deepseek4_backend.cpp
- `DFLASH_DS4_SPEC_REFERENCE_EXACT` - deepseek4_dspark_spec.cpp
- `DFLASH_DS4_SPEC_Q` - deepseek4_dspark_spec.cpp
- `DFLASH_DS4_TIMING` - deepseek4_backend.cpp, deepseek4_target_shard_ipc_daemon.cpp
- `DFLASH_DS4_TP_CAPTURE_CACHE_SLOTS` - deepseek4_fused_verify.inc
- `DFLASH_DS4_TP_FUSED_CACHE_SLOTS` - deepseek4_fused_verify.inc
- `DFLASH_DS4_TP_SCHEDULE_BRANCHES` - deepseek4_fused_verify.inc
- `DFLASH_DS4_TP_TARGETED_JOIN_SPLIT` - moe_hybrid_ffn_eval.cpp
- `DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY` - deepseek4_fused_verify.inc
- `DFLASH_DS4_TOPK` - deepseek4_graph.cpp
- `DFLASH_DYN_CONV_FUSED` - draft_graph.cpp (=0 expands the DFlash2 dynamic convs instead of the fused kernel)
- `DFLASH_EXPERT_BUDGET_MB` - deepseek4_backend.cpp, laguna_backend.cpp, qwen35moe_backend.cpp
- `DFLASH_EXPERT_BUDGET_PCT` - laguna_backend.cpp
- `DFLASH_FAST_ROLLBACK_THRESHOLD` - chain_rollback_policy.h
- `DFLASH_FEATURE_DTYPE` - dflash_feature_ring.cpp
- `DFLASH_KV_ROTATE` - qwen35_target_graph.cpp (set to 1 to force FWHT K rotation on; off by default for f16/q8_0 caches, on for narrower types)
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
- `DFLASH_HIP_BACKEND_PATH` - dynamic_backend.cpp
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
- `DFLASH_MOE_TP_TARGETED_JOIN_SPLIT` - moe_hybrid_ffn_eval.cpp
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
- `DFLASH_MMID_TELEMETRY` - ggml-cuda.cu, mmvq.cu
- `DFLASH_MMQ_FULL_BATCH_MIN` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MMQ_SUB_BATCH` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MODEL_CARDS_DIR` - model_card.cpp
- `DFLASH_MOE_COLD_BACKEND` - deepseek4_loader.cpp
- `DFLASH_MOE_COMPACT_MATERIALIZED` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_DUPLICATE_HOT_ON_COLD` - moe_hybrid_storage.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_DAEMON_TOKEN_LOOP` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_DTYPE` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU` - deepseek4_backend.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_PROFILE` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_SHARED_BYTES` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_IPC_TRANSPORT` - moe_expert_compute_ipc.cpp
- `DFLASH_MOE_EXPERT_COMPUTE_THREADS` - moe_expert_compute_cpu.cpp
- `DFLASH_MOE_EXPERT_MAJOR_GPU_REDUCE` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_EXPERT_MAJOR_PREFILL` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_FIXED_SLOT_GRAPHS` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_FIXED_SLOT_MAX` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_FULL_COLD_PARALLEL` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_FUSED_COMBINE` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_PREFILL_DEVICE_INPUT` - deepseek4_graph.cpp
- `DFLASH_MOE_PREFILL_HOT_SUB_BATCH` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_PREFILL_MASKED_COLD` - moe_hybrid_ffn_eval.cpp
- `DFLASH_MOE_PREFILL_PERSISTENT_OWNER_ALLOC` - deepseek4_graph.cpp
- `DFLASH_MOE_TP_BACKEND` - deepseek4_backend.cpp
- `DFLASH_NO_MASK` - laguna_backend.cpp
- `DFLASH_NO_MOE_ROUTER_FUSE` - qwen35moe_ffn.cpp
- `DFLASH_NO_MOE_SWIGLU_FUSE` - qwen35moe_ffn.cpp
- `DFLASH_NO_PREAD` - deepseek4_loader.cpp
- `DFLASH_PROF` - prof_env.h
- `DFLASH_PREFILL_CACHE_SLOTS` - scripts/entrypoint.sh (maps to `--prefill-cache-slots`)
- `DFLASH_PREFILL_TIMING` - qwen35_backend.cpp (DEBUG: per-ubatch prefill build/alloc/compute timing)
- `DFLASH_PREFIX_CACHE_SLOTS` - scripts/entrypoint.sh (maps to `--prefix-cache-slots`)
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
- `DFLASH_QWEN35_AR_BURST` - qwen35_backend.cpp (adaptive plain-decode burst length; default 40)
- `DFLASH_QWEN35_DFLASH2_TREE` - qwen35_backend.cpp (=0 uses raw top-k tree scoring instead of the selector under --ddtree)
- `DFLASH_QWEN35_DSPARK` - qwen35_backend.cpp (=0 disables the DSpark drafter head path)
- `DFLASH_QWEN35_DSPARK_CONF_DEBUG` - qwen35_backend.cpp (DEBUG: print confidence-gate scores)
- `DFLASH_QWEN35_DSPARK_CONFIDENCE_THRESHOLD` - qwen35_backend.cpp (adaptive block-length confidence gate)
- `DFLASH_QWEN35_DSPARK_TREE` - qwen35_backend.cpp (DSpark tree drafting toggle)
- `DFLASH_QWEN35_FUSED_DSPARK` - qwen35_backend.cpp (=0 keeps the DSpark heads off the fused graph)
- `DFLASH_QWEN35_NO_FUSED_KERNELS` - qwen35_target_graph.cpp (KILL SWITCH: =1 rebuilds the pre-fusion decode graph)
- `DFLASH_QWEN35_NO_STACK` - gguf_target_loader.cpp (KILL SWITCH: =1 disables zero-copy stacked weight aliases)
- `DFLASH_QWEN35_SPEC_STEP_RATIO` - qwen35_backend.cpp (adaptive policy spec/plain step-time ratio override)
- `DFLASH_SAMPLED_VERIFY` - laguna_backend.cpp, qwen35_backend.cpp
- `DFLASH_SHARE_DIR` - http_server.cpp
- `DFLASH_SINGLE_CHAIN_CHECKPOINT_F32` - chain_rollback_policy.h
- `DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG` - chain_rollback_policy.h
- `DFLASH_SPARK` - laguna_backend.cpp, qwen35moe_backend.cpp
- `DFLASH_SPARK_VRAM_MB` - laguna_backend.cpp, qwen35moe_backend.cpp
- `DFLASH_SPLIT_CAPTURE_SELFTEST` - qwen35_layer_split_dflash_target.cpp
- `DFLASH_SPLIT_CHAIN_ROLLBACK_DIAG` - qwen35_layer_split_dflash_target.cpp, qwen35_target_graph.cpp
- `DFLASH_SPLIT_FAST_ROLLBACK` - chain_rollback_policy.h
- `DFLASH_STALL_TOOL_PREFIX` - http_server.cpp
- `DFLASH_SV_DEBUG` - qwen35_backend.cpp
- `DFLASH_TARGET_SHARD_IPC_SHARED_BYTES` - target_shard_ipc.cpp
- `DFLASH_TARGET_SHARD_IPC_TRANSPORT` - target_shard_ipc.cpp
- `DFLASH_TOPK_PROFILE` - geometric_draft_topk_cuda.cu
- `DFLASH_TOPK_SPLIT` - geometric_draft_topk_cuda.cu
- `DFLASH_VERIFY_WIDTH` - qwen35moe_backend.cpp
- `FAST_ROLLBACK_DIAG` - qwen35_dflash_target.cpp
- `HOME` - spark_corpus.cpp
- `LUCE_Q8_MEMO` - mmvq.cu (set to 0 to disable q8_1 activation memoisation; on by default)
- `LUCE_MMQ_BIG_PREFILL` - mmq.cu (=0 disables the RDNA4 128-wide MMQ tiles for large prefill batches)
- `LUCE_MMVQ_MAX_NCOLS` - deepseek4_backend.cpp
- `LUCE_QK_FUSE_LAYERS` - laguna_target_graph.cpp
- `LUCE_QK_FUSE_MODE` - laguna_target_graph.cpp
- `PFLASH_DRAFTER_EARLY_EXIT_N` - qwen3_graph.cpp
- `PFLASH_DRAFTER_SCORE_LAYERS` - qwen3_graph.cpp
- `PFLASH_FREEZE_HOT_WINDOW` - http_server.cpp
- `TMPDIR` - backend_ipc.cpp, moe_expert_compute_ipc.cpp
