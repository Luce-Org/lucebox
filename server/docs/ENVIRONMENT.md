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
| `LUCE_ADAPTIVE_SPEC_WIDTH` | unset | BURN-IN: =1 enables the shared acceptance-feedback verify-width controller. Fixed width is the production default; backend-specific overrides take precedence. |
| `LUCE_FA256_MMA` | 1 on RDNA4 | KILL SWITCH (burn-in): =0 restores the generic tile kernel for head-256 attention on RDNA4. With the default 1, prefill-sized head-256 batches take the rocWMMA kernel below ~32K KV and the raw-MMA kernel beyond (rocWMMA requires a `GGML_HIP_ROCWMMA_FATTN` build; without it the raw-MMA kernel covers all shapes). |
| `LUCE_FA256_WMMA` | unset | A/B: =1 forces the rocWMMA kernel on head-256 RDNA4 shapes whose KV length is a multiple of 256, bypassing the KV-length crossover (requires a `GGML_HIP_ROCWMMA_FATTN` build). |
| `LUCE_FA256_WMMA_MAX_KV` | 32768 | KV length above which the head-256 tensor-core route switches from the rocWMMA kernel to the raw-MMA kernel in `GGML_HIP_ROCWMMA_FATTN` builds (measured crossover on gfx1201). |
| `LUCE_PAGED_WMMA` | unset (0) | BURN-IN: =1 routes paged full-attention layers (RDNA4, head 256, F16/Q8_0/Q4_0 KV, non-tree) to the WMMA kernel. Differential-tested against the decode kernel; single-prompt TTFT -21% at 12K and -42% at 44K, batched 8K-pool prefill slightly ahead. |
| `GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS` | unset | DEBUG: force the paged-attention context partition count (both routes) to bisect partition-overlap and overhead behaviour. |
| `LUCE_PREFILL_UBATCH` | backend-dependent (512 in `qwen35_backend.cpp`; 16/384 in `layer_split_daemon.cpp`; `cfg_.chunk` in `qwen35_layer_split_adapter.cpp`) | Prefill ubatch. Under pooled kvflash prefill it is rounded down to a multiple of the pager chunk (never below one chunk) and clamped to the pool, instead of being forced to one chunk per ubatch. |
| `LUCE_DRAFT_KV` | 1 | KILL SWITCH (remove after burn-in): =0 restores the legacy per-step drafter window recompute instead of the ring cache. |
| `LUCE_LAGUNA_SWA_RING` | 1 | KILL SWITCH (remove after burn-in): =0 keeps SWA layers on pool-sized caches under KVFlash. |
| `LUCE_PROF` | unset | DEBUG: comma list of profilers (step,verify,prefill). Replaces LUCE_LAGUNA_{STEP,VERIFY,PREFILL}_PROF. |
| `GGML_CUDA_GRAPH_STATS` | unset | DEBUG: per-graph CUDA-graph replay/capture/eager counters. |
| `GGML_CUDA_GRAPH_STATS_EVERY` | 200 | DEBUG: print period for the stats above (clamped to >=1). |
| `LUCE_ADAPTIVE_K_TAU` | 0 = off | Prefer the CLI: --adaptive-experts [tau]. Cumulative combine-weight threshold for per-token expert gating. |
| `LUCE_ADAPTIVE_K_DENSE` | per-model default | CSV of MoE layers kept dense under adaptive-K (DFlash capture layers). Warned-inert on families that do not thread layer indices yet. |
| `LUCE_MMID_GROUPED` | unset | Grouped MUL_MAT_ID kernel for small verify batches; candidate for CLI promotion. |
| `LUCE_MMID_GROUPED_TYPES` | 7 | Grouped-kernel type mask; bit 3 (`8`) opts ROCmFP2/ROCmFP3 into the path. |
| `LUCE_MMID_GROUPED_DEVICE` | -1 | Optional zero-based device restriction; unset/-1 applies to every eligible device. |
| `LUCE_DS4_MOE_TP` / `LUCE_DS4_MOE_TP_INPROC` | unset | BURN-IN: enable DeepSeek4 route-owner expert parallelism in one process. |
| `LUCE_DS4_MOE_TP_BACKEND` / `LUCE_MOE_TP_BACKEND` | peer runtime in a mixed build; compiled runtime otherwise | Select the in-process cold expert owner backend. |
| `LUCE_DS4_MOE_TP_GPU` | peer backend device 0 in a mixed build; other local device otherwise | Device index within the cold DeepSeek4 expert backend. |
| `LUCE_DS4_MOE_TP_CONCENTRATE_COLD` | unset | BURN-IN: use complete peer-owned expert layers to reduce cross-runtime joins; falls back when the placement would exceed the target budget. |
| `LUCE_DS4_MOE_TP_PEER_HOT` | unset | BURN-IN: with `LUCE_DS4_HOTNESS_CSV`, reserve the profile's hottest experts for the secondary owner. |
| `LUCE_DS4_CROSS_VENDOR_OWNER_SUMS` | unset | BURN-IN: reduce each mixed-vendor owner's routed outputs locally before the final owner add. This changes floating-point association and is not the byte-identity mode. |
| `LUCE_DS4_TP_SCHEDULE_BRANCHES` | unset | BURN-IN: expose independent mixed-vendor expert branches to the common multi-backend scheduler. |
| `LUCE_DS4_TP_TARGETED_JOIN_SPLIT` / `LUCE_MOE_TP_TARGETED_JOIN_SPLIT` | unset | BURN-IN: start a main-GPU split only at each peer-result join, avoiding an extra peer fence per MoE layer. |
| `LUCE_DS4_COMP_PAD_STRIDE` | 16, 128 on `gfx1151` DSpark | BURN-IN: compressed-KV padding bucket (`16`, `32`, `64`, or `128`); wider exact-masked buckets reduce verifier graph recapture churn. |
| `LUCE_DS4_DECODE_ATTN_CACHE_MB` | a quarter of the target GPU's free memory when the first decode attention graph is cached, 256 MiB floor | Byte budget of the per-layer decode attention graph cache (heterogeneous and token-wise paths); least recently used shapes are evicted across layers before a new one is built. A positive value is used as given. |
| `GGML_CUDA_GRAPH_MAX_KEYS` | 4096 | Cap on captured CUDA/HIP graph executables kept per backend context, keyed by graph node address; least recently used entries are retired in batches. It must exceed the number of live graphs (the heterogeneous DeepSeek4 verifier keeps up to 24 scheduler graphs of ~130 splits) or warm graphs stop replaying. `0` disables the cap. |
| `LUCE_DS4_MIX_MMQ_PREFILL` | enabled for DS4 approximate prefill on gfx1151 | BURN-IN KILL SWITCH: =0 disables registry-aware mixed ROCmFP MMQ; =1 explicitly enables it on supported HIP devices. Automatic selection is model/graph-local and never writes the process environment. Exact prefill retains its existing dispatch defaults, including the specialized paired FP2 path. |
| `LUCE_DS4_INCREMENTAL_VERIFY_MASK` | 1 for masks at least 4 MiB | BURN-IN KILL SWITCH: =0 rebuilds and transfers the complete fused-verifier attention mask from the host on every step. |
| `LUCE_DS4_INCREMENTAL_VERIFY_MASK_MIN_BYTES` | 4194304 | DEBUG/A-B: minimum fused-verifier mask size for GPU zeroing plus negative-range updates. |
| `LUCE_DS4_SPARSE_DECODE_FLASH` | 0 | Experimental single-HIP-target verifier attention. Opt in with =1; may change generated tokens. Uses model sparse top-k only when it removes more than half of the compressed rows. |
| `LUCE_DS4_DISABLE_GROUPED_OUTPUT_PROJECTION` | unset | DEBUG: restore the materialized output projection when diagnosing grouped-view copies across unlike runtimes. |
| `LUCE_DS4_DRAFT_BACKEND` / `LUCE_DS4_DRAFT_GPU` | compiled backend / target device | Select the in-process DSpark backend and device. |
| `LUCE_CUDA_BACKEND_PATH` / `LUCE_HIP_BACKEND_PATH` | auto-discovered beside the executable | Explicit peer module file path for a mixed CUDA+HIP build. |
| `LUCE_DS4_TP_GROUPED_MMVQ` / `LUCE_MOE_TP_GROUPED_MMVQ` | unset | OPT-IN: grouped expert MMVQ for `n_tokens > 1` instead of tokenwise ROCmFP2 gate/up dispatch. Qualified for paged R9700 + Strix at concurrency 1–4; the flag itself does not enforce topology or lane limits. The model-neutral name takes precedence. |
| `LUCE_CUDA_MMVQ_FP4_X4` | 1 for monolithic DS4 `gfx1151` paged serving and opt-in HIP DS4 q=5 verification; unset otherwise | Enable dense ROCmFP4 x4 dispatch. Set `0` to restore generic four- and five-column kernels. |
| `LUCE_CUDA_MMVQ_FP4_Q5_X4_PLUS1` | 1 for monolithic DS4 `gfx1151` paged serving and opt-in DS4 q=5 verification on `gfx1201`; unset otherwise | Enable dense five-column x4+1 dispatch when `LUCE_CUDA_MMVQ_FP4_X4=1`. Set `0` to restore the generic five-column kernel. |
| `LUCE_CUDA_MMVQ_MOE_FP3_PACKED24` | 1 for monolithic DS4 `gfx1151` paged serving; unset otherwise | Enable packed 24-bit ROCmFP3 expert dispatch. Set `0` to restore generic expert dispatch. |
| `LUCE_DS4_TP_BATCH_SPLIT_COPIES` | unset | OPT-IN: establish destination readiness once per DS4 scheduler split while retaining each backend copy's dependency publication. The qualified dual-ROCm launcher enables it. |
| `GGML_BATCH_PEER_COPIES` | unset | BURN-IN: additionally combine HIP peer-copy dependency publication. `GGML_CUDA_BATCH_PEER_COPIES` remains a compatibility alias. Keep these event-batching variables unset for the exact qualified profile. |
| `GGML_SCHED_PROFILE` / `GGML_SCHED_PROFILE_MIN_SPLITS` | unset / 1 | DEBUG: report scheduler splits, copy volume, submission time, and source/destination synchronization time. |
| `LUCE_DS4_TP_FUSED_CACHE_SLOTS` | 8, 24 with `LUCE_DS4_Q5_VERIFY` | BURN-IN: number of heterogeneous verifier schedulers retained; higher values retain substantially more scratch on both GPUs. |
| `LUCE_DS4_VERIFY_FORCE_GRAPH_REPLAY` | unset | OPT-IN: bypass graph property scans only after warmup; scheduler-generation checks remain mandatory. |
| `LUCE_DS4_ROCTX` | unset | DEBUG: on HIP builds, dynamically load ROCTX and emit semantic DS4 prefill, speculative-decode, and layer-range markers for external rocprof traces. No events, timing, or device synchronization are added. |
| `LUCE_QWEN35_ROCTX` | unset | DEBUG: on HIP builds, dynamically load ROCTX and mark Qwen concurrent steps, graph compute, and argmax readback with live, padded, and packed-prefill shape metadata. |
| `LUCE_CUDA_MMVF_NARROW_F16` | enabled on qualified gfx1151 narrow F16 matmuls | BURN-IN KILL SWITCH: =0 restores the generic dispatch decision for the narrow F16 projection optimization, unless an explicit `LUCE_MMVF_MAX_NCOLS_F16` ceiling overrides it. |
| `GGML_CUDA_MMQ_X` | unset | DEBUG: force a supported MMQ output-column tile width (8–128) for architecture tuning; invalid or over-budget values fall back to automatic selection. |
| `GGML_CUDA_MMQ_MOE_ADAPTIVE_X` | unset (MIX formats adapt) | On sparse-route gfx1151 grouped MoE MMQ, choose the measured output tile from routed rows per expert. Unset: only the shipped ROCmFP2/ROCmFP3 MIX expert formats adapt (tile 32). `=1` (BURN-IN): the plain ROCmFP2/3/4 formats adapt too. `=0`: none adapt. Ordinary matmuls and other devices are unchanged. |
| `GGML_CUDA_MMQ_MOE_PERSISTENT` | unset | EXPERIMENTAL: on prefill-sized (at least 256-token) sparse grouped ROCmFP2/3/4 MMQ on gfx1151, build a compact device-side expert-tile queue and consume it with bounded persistent workers. Short batches, ordinary matmuls, unmeasured formats, and other devices are unchanged. |
| `GGML_CUDA_MMQ_MOE_PERSISTENT_BLOCKS_PER_CU` | 32 | DEBUG: set the compact grouped-MoE worker budget per gfx1151 CU from 1–32. Invalid values use 32. |
| `GGML_CUDA_MLA_STREAM_TOPK` / `GGML_DS4_FA_STREAM_TOPK` | enabled for maskless ratio-4 sparse prefill; unset otherwise | Use wave32 selected-row D512 indexed attention for eligible long-prefill shapes: eight-head compact-order F32 on HIP, or the existing streaming path for F16 and CUDA. The first name takes precedence. Set `0` to restore the compact fallback, including for maskless prefill. |
| `GGML_CUDA_MLA_STREAM_F32_STAGE` | enabled for maskless ratio-4 sparse prefill; unset otherwise | With F16-input streaming D512 indexed attention, convert aligned F16 pairs once while staging them in shared memory instead of repeating conversion for every head. Applies only when streaming top-k is selected automatically or through `GGML_CUDA_MLA_STREAM_TOPK` / `GGML_DS4_FA_STREAM_TOPK`. Set `0` to restore F16 staging for F16 inputs. This override does not affect the compact-order HIP F32 path. |
| `GGML_CUDA_MLA_STREAM_FAST_EXP` | enabled for maskless ratio-4 sparse prefill; unset otherwise | With FP32-staged streaming D512 indexed attention, use the hardware exponential intrinsic for online softmax. Requires both streaming top-k selection and FP32 staging; setting this flag alone does not enable either. Set `0` to use `expf`. The compact-order HIP F32 path and other attention paths are unchanged. |
| `GGML_CUDA_MLA_SPLIT_KV` / `GGML_DS4_FA_SPLIT_KV` | 1 on gfx1151 indexed decode; unset elsewhere | BURN-IN: =1 forces the reusable split-KV MLA schedule. Unset, empty, or =0 does not force it (the device default still applies). Set `GGML_CUDA_MLA_NO_SPLIT_KV=1` (or legacy `GGML_DS4_FA_NO_SPLIT_KV=1`) to disable it; either kill switch overrides either force flag, while empty/=0 kill switches have no effect. |
| `GGML_DS4_TOPK_BLOCK_RADIX` | 1 on gfx1151 | BURN-IN KILL SWITCH: =0 restores hipCUB full sort for DS4-shaped 512-row top-k selection. |
| `GGML_DS4_FA_SERIAL_INDEX_SCAN` | unset | DEBUG/A-B: restore the serial indexed-attention mask scan instead of the long-context HIP parallel scan. |
| `LUCE_MOE_PREFILL_PERSISTENT_OWNER_ALLOC` | 1 for qualified long heterogeneous prefill | KILL SWITCH: =0 restores per-layer route/owner scratch allocation. |
| `LUCE_MOE_TP_*` / `LUCE_MOE_HYBRID_PREFILL_EAGER` | unset | BURN-IN: model-neutral names for common heterogeneous-MoE scheduling and kernel policy. Existing `LUCE_DS4_*` names remain compatibility aliases. |
| `LUCE_MMID_TELEMETRY` | unset | DEBUG: report MUL_MAT_ID dispatch, MMVQ variant, and per-node graph compatibility. |
| `LUCE_KVFLASH` | unset | Prefer the CLI: `--kvflash` (token count or `auto`). |
| `LUCE_PREFIX_CACHE_SLOTS` | 32 | Container-entrypoint equivalent of `--prefix-cache-slots`; not read directly by the native binary. |
| `LUCE_PREFILL_CACHE_SLOTS` | 0 | Container-entrypoint equivalent of `--prefill-cache-slots`; not read directly by the native binary. |
| `LUCE_MMPROJ` | unset | Container-entrypoint equivalent of `--mmproj` (vision projector path, enables image input); not read directly by the native binary. |
| `LUCE_PREFILL_POOL_TRIM_TOKENS` | unset | OPT-IN: trim cached allocations from legacy CUDA/HIP device pools at completed Qwen3.5 prefill chunk boundaries after each configured token interval. Intended for long, shape-changing prefills on non-VMM devices; each trim synchronizes the target backend and retires captured graphs. |
| `LUCE_SPLIT_FAST_ROLLBACK` | unset | OPT-IN: exact F32 checkpoints and replay-free rollback for local qwen35 target layer splits. Prefer `--target-split-fast-rollback`; adds checkpoint VRAM (~1.65 GiB for the measured Qwen3.6-27B q=16 split). |
| `LUCE_STALL_TOOL_PREFIX` | unset | OPT-IN: recover a stalled tool call by injecting the prepared tool prefix when generation stops after an action suffix. |
| `LUCE_DS4_SPEC` / `LUCE_DS4_DRAFT` / `LUCE_DS4_DRAFT_BACKEND` / `LUCE_DS4_DRAFT_GPU` | unset | OPT-IN: enable DeepSeek4 DSpark, select its draft GGUF, and optionally select the local drafter backend/device. See `DS4.md`. |
| `LUCE_DS4_CUDA_LAYERS` | auto | Override the DeepSeek4 heterogeneous layer-split heuristic. See `DS4.md`. |
| `LUCE_ROCMFP2_ROW4` | 1 on gfx1151 for q>2; legacy two-row kernel elsewhere | BURN-IN KILL SWITCH: =0 restores two-row-per-wave ROCmFP2 verification kernels. |

## Full inventory (generated)

`grep -rE 'getenv\("[A-Z0-9_]+"\)' server/src` - regenerate when adding or removing variables. Also include backend variables and helper-based reads (such as `ds4_env_flag_enabled`) under `server/deps/llama.cpp/ggml/src`.

- `LUCE_CHUNKED` - qwen35_target_graph.cpp
- `LUCE_DRAFT_FP16` - draft_safetensors_loader.cpp
- `LUCE_DRAFT_SWA` - server_main.cpp
- `LUCE_KV_F16` - kv_quant.cpp
- `LUCE_KV_K` - kv_quant.cpp, laguna_backend.cpp
- `LUCE_KV_Q4` - kv_quant.cpp
- `LUCE_KV_TQ3` - kv_quant.cpp, qwen3_drafter.cpp
- `LUCE_KV_V` - kv_quant.cpp, laguna_backend.cpp
- `LUCE_LM_HEAD_FIX` - http_server.cpp
- `LUCE_PAGED_WMMA` - paged-attn.cu (ggml-cuda) (=1 routes paged full-attention layers to the WMMA kernel; RDNA4 only, F16/Q8_0/Q4_0, non-tree)
- `LUCE_PREFILL_UBATCH` - qwen35/prefill_helpers.h
- `LUCE_ADAPTIVE_K_DENSE` - mmid_adaptive_k.h
- `LUCE_ADAPTIVE_K_TAU` - mmid_adaptive_k.h
- `LUCE_ADAPTIVE_SPEC_WIDTH` - adaptive_spec_width.h
- `LUCE_ADAPTIVE_WIDTH_MIN` - adaptive_verify_width.h
- `LUCE_ADAPTIVE_WIDTH_THETA` - adaptive_verify_width.h
- `LUCE_COLD_THREADS` - moe_expert_compute_cpu.cpp
- `LUCE_CUDA_BACKEND_PATH` - dynamic_backend.cpp
- `LUCE_CUDA_MMVF_NARROW_F16` - ggml-cuda/mmvf.cu
- `LUCE_CUDA_MMVQ_FP4_X4` - deepseek4_backend.cpp, mmvq.cu
- `LUCE_CUDA_MMVQ_MOE_ALIGN_SHARED_IDS` - moe_hybrid_ffn_eval.cpp
- `LUCE_CUDA_MMVQ_MOE_FP3_PACKED24` - deepseek4_backend.cpp, mmvq.cu
- `LUCE_CUDA_MMVQ_MOE_KERNEL` - moe_hybrid_ffn_eval.cpp
- `LUCE_CUDA_MMVQ_MOE_ROWS_PER_BLOCK` - mmvq.cu
- `LUCE_DISABLE_DRAFT_ATTN` - draft_graph.cpp
- `LUCE_DISABLE_DRAFT_ATTN_GATE` - draft_graph.cpp
- `LUCE_DISABLE_DRAFT_AUX_NORMS` - draft_graph.cpp
- `LUCE_DISABLE_DRAFT_FFN` - draft_graph.cpp
- `LUCE_DISABLE_DRAFT_SWA` - dflash_draft_kv.cpp, draft_graph.cpp
- `LUCE_DOMINO_ZERO_START` - domino_head.cpp
- `LUCE_DRAFT_IPC_SHARED_BYTES` - dflash_draft_ipc.cpp
- `LUCE_DRAFT_IPC_TRANSPORT` - dflash_draft_ipc.cpp
- `LUCE_DRAFT_KV` - laguna_backend.cpp, qwen35_backend.cpp
- `LUCE_DRAFT_PERSIST` - laguna_backend.cpp
- `LUCE_DROP_COLD` - qwen35moe_backend.cpp, qwen35moe_pipelined_decode.cpp
- `LUCE_DS4_ADAPTIVE_WIDTH` - deepseek4_dspark_spec.cpp
- `LUCE_DS4_CONFIDENCE_WIDTH` - deepseek4_dspark_spec.cpp (KILL SWITCH: =0 falls back from the drafter confidence head to the learned-acceptance width policy)
- `LUCE_DS4_DRAFT_CONTEXT_KV_CACHE` - deepseek4_dspark_spec.cpp (gfx1151 DSpark default =1: pinned drafter context window; =0 restores pageable copies)
- `LUCE_DS4_FUSED_HYBRID_DECODE` - deepseek4_graph.cpp
- `LUCE_DS4_GPU_ARGMAX_VERIFY` - deepseek4_dspark_spec.cpp, deepseek4_fused_verify.inc (gfx1151 DSpark default =1: on-device argmax of verifier logits; =0 copies the logits back)
- `LUCE_DS4_HYBRID_PREFILL_EAGER` - deepseek4_graph.cpp, moe_hybrid_ffn_eval.cpp
- `LUCE_DS4_HYBRID_PREFILL_GPU_HC` - deepseek4_graph.cpp
- `LUCE_DS4_Q5_VERIFY` - deepseek4_backend.cpp, deepseek4_dspark_spec.cpp, deepseek4_fused_verify.inc, deepseek4_graph.cpp (gfx1151 DSpark default =1: five-row fused verifier and the 24-slot cache; =0 restores the q<=4 verifier)
- `LUCE_DS4_PINNED_ROLLBACK` - deepseek4_dspark_spec.cpp (gfx1151 DSpark default =1: pinned host rollback state; =0 restores pageable copies)
- `LUCE_DS4_COMP_PAD_STRIDE` - deepseek4_graph.cpp
- `LUCE_DS4_CROSS_VENDOR_OWNER_SUMS` - deepseek4_fused_verify.inc
- `LUCE_DS4_CUDA_LAYERS` - deepseek4_layer_split_adapter.cpp
- `LUCE_DS4_DECODE_ATTN_CACHE_MB` - deepseek4_graph.cpp
- `LUCE_DS4_DENSE_TP_MASK` - deepseek4_loader.cpp
- `LUCE_DS4_DENSE_TP_STRIX_FRACTION` - deepseek4_loader.cpp
- `LUCE_DS4_DIRECT_CONTIGUOUS_CAUSAL` - deepseek4_backend.cpp, deepseek4_graph.cpp (gfx1151 sparse-prefill default; KILL SWITCH: =0 restores the explicit causal mask)
- `LUCE_DS4_DISABLE_BOUNDARY_CHECKPOINT` - deepseek4_dspark_spec.cpp (KILL SWITCH: =1 replays a two-boundary q5 rejection from a full snapshot instead of the boundary checkpoint)
- `LUCE_DS4_DISABLE_GROUPED_OUTPUT_PROJECTION` - deepseek4_graph.cpp
- `LUCE_DS4_DIRECT_INDEXER_TOPK` - deepseek4_graph.cpp
- `LUCE_DS4_DRAFT` - deepseek4_backend.cpp
- `LUCE_DS4_DRAFT_BACKEND` - deepseek4_backend.cpp
- `LUCE_DS4_DRAFT_GPU` - deepseek4_backend.cpp
- `LUCE_DS4_DSPARK_DEBUG` - deepseek4_graph.cpp
- `LUCE_DS4_FUSED_VERIFY` - deepseek4_dspark_spec.cpp, deepseek4_loader.cpp
- `LUCE_DS4_HOTNESS_CSV` - deepseek4_backend.cpp
- `LUCE_DS4_INCREMENTAL_VERIFY_MASK` - deepseek4_fused_verify.inc
- `LUCE_DS4_INCREMENTAL_VERIFY_MASK_MIN_BYTES` - deepseek4_fused_verify.inc
- `LUCE_DS4_INDEXER_F16_Q` - deepseek4_backend.cpp, deepseek4_graph.cpp (gfx1151 sparse-prefill default; KILL SWITCH: =0 restores F32 indexer queries)
- `LUCE_DS4_MIX_MMQ_PREFILL` - deepseek4_backend.cpp, ggml-cuda/mmq.cu
- `LUCE_DS4_MOE_TP` - deepseek4_backend.cpp
- `LUCE_DS4_MOE_TP_BACKEND` - deepseek4_backend.cpp
- `LUCE_DS4_MOE_TP_CONCENTRATE_COLD` - deepseek4_backend.cpp
- `LUCE_DS4_MOE_TP_GPU` - deepseek4_backend.cpp
- `LUCE_DS4_MOE_TP_INPROC` - deepseek4_backend.cpp
- `LUCE_DS4_MOE_TP_PEER_HOT` - deepseek4_backend.cpp
- `LUCE_DS4_PREFILL_F16_KV_ALL` - deepseek4_backend.cpp, deepseek4_graph.cpp (gfx1151 sparse-prefill default; KILL SWITCH: =0 restores F32 selected-KV transport)
- `LUCE_DS4_ROUTING_STATS_OUT` - deepseek4_backend.cpp
- `LUCE_DS4_ROCTX` - deepseek4_roctx.cpp
- `LUCE_DS4_TOKEN_TRACE` - deepseek4_dspark_spec.cpp (DIAGNOSTIC: per-token speculative trace on stderr)
- `LUCE_DS4_TP_COARSE_OWNER` - moe_hybrid_ffn_eval.cpp
- `LUCE_DS4_TP_DEVICE_JOIN` - moe_hybrid_ffn_eval.cpp
- `LUCE_DS4_TP_DEVICE_JOIN_SPLIT` - deepseek4_fused_verify.inc
- `LUCE_DS4_TP_FUSED_HC_JOIN` - deepseek4_fused_verify.inc
- `LUCE_DS4_TP_MAIN_ROUTE_WEIGHTS` - deepseek4_fused_verify.inc
- `LUCE_DS4_TP_MASKED_ROUTES` - deepseek4_fused_verify.inc
- `LUCE_DS4_TP_NATIVE_ROUTE_WIDTH` - deepseek4_fused_verify.inc
- `LUCE_DS4_TP_ROUTE_PREFORK` - moe_hybrid_ffn_eval.cpp
- `LUCE_DS4_VERIFY_BUILD_TIMING` - deepseek4_fused_verify.inc (DIAGNOSTIC: fused-verify graph build timing)
- `LUCE_DSPARK_NO_CHAIN_GRAPH_CACHE` - dspark_head.cpp (KILL SWITCH: =1 rebuilds the DSpark Markov chain graph on every call)
- `LUCE_MMID_GROUPED` - deepseek4_backend.cpp, mmvq.cu
- `LUCE_MMID_GROUPED_DEVICE` - mmvq.cu
- `LUCE_MMID_GROUPED_TYPES` - deepseek4_backend.cpp, mmvq.cu
- `LUCE_MOE_TP_DYNAMIC_MAIN_SLOTS_X4` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_TP_DYNAMIC_ROUTE_BALANCE` - moe_hybrid_ffn_eval.cpp
- `LUCE_QWEN35_ROCTX` - qwen35_roctx.cpp
- `LUCE_DS4_SEQ_VERIFY` - deepseek4_dspark_spec.cpp
- `LUCE_ROCMFP2_ROW4` - rocmfp2_mix.cu
- `LUCE_DS4_SPEC` - deepseek4_backend.cpp
- `LUCE_DS4_SPEC_REFERENCE_EXACT` - deepseek4_dspark_spec.cpp
- `LUCE_DS4_SPEC_Q` - deepseek4_dspark_spec.cpp
- `LUCE_DS4_SPARSE_DECODE_FLASH` - deepseek4_fused_verify.inc, deepseek4_graph.cpp
- `LUCE_DS4_TIMING` - deepseek4_backend.cpp, deepseek4_target_shard_ipc_daemon.cpp
- `LUCE_DS4_TP_CAPTURE_CACHE_SLOTS` - deepseek4_fused_verify.inc
- `LUCE_DS4_TP_FUSED_CACHE_SLOTS` - deepseek4_fused_verify.inc
- `LUCE_DS4_TP_SCHEDULE_BRANCHES` - deepseek4_fused_verify.inc
- `LUCE_DS4_TP_TARGETED_JOIN_SPLIT` - moe_hybrid_ffn_eval.cpp
- `LUCE_DS4_VERIFY_FORCE_GRAPH_REPLAY` - deepseek4_fused_verify.inc
- `LUCE_DS4_TOPK` - deepseek4_graph.cpp
- `LUCE_DYN_CONV_FUSED` - draft_graph.cpp (=0 expands the DFlash2 dynamic convs instead of the fused kernel)
- `LUCE_EXPERT_BUDGET_MB` - deepseek4_backend.cpp, laguna_backend.cpp, qwen35moe_backend.cpp
- `LUCE_EXPERT_BUDGET_PCT` - laguna_backend.cpp
- `LUCE_FAST_ROLLBACK_THRESHOLD` - chain_rollback_policy.h
- `LUCE_FEATURE_DTYPE` - dflash_feature_ring.cpp
- `LUCE_KV_ROTATE` - qwen35_target_graph.cpp (set to 1 to force FWHT K rotation on; off by default for f16/q8_0 caches, on for narrower types)
- `LUCE_FP_ALPHA` - http_server.cpp, qwen3_graph.cpp, server_main.cpp
- `LUCE_FP_CHUNK_S` - qwen3_graph.cpp
- `LUCE_FP_DEBUG_LAYER0` - qwen3_graph.cpp
- `LUCE_FP_DUMP_COUNTS` - flashprefill.cpp
- `LUCE_FP_HIP_ROW` - flashprefill_kernels.cu
- `LUCE_FP_NOPE_TAIL` - qwen3_graph.cpp
- `LUCE_FP_PROFILE` - flashprefill.cpp
- `LUCE_FP_SKIP_PREWARM` - qwen3_drafter.cpp
- `LUCE_FP_USE_BSA` - flashprefill.cpp, http_server.cpp, server_main.cpp
- `LUCE_G4_BSA_CHUNK` - gemma4_graph.cpp
- `LUCE_GEMMA4_LAYER_SPLIT_UBATCH` - gemma4_layer_split_adapter.cpp
- `LUCE_GEMMA4_NO_KVPAD` - gemma4_graph.cpp
- `LUCE_GPU_ARGMAX` - qwen35_backend.cpp
- `LUCE_GPU_DRAFT_TOPK` - qwen35_dflash_target.cpp
- `LUCE_GPU_SAMPLE` - geometric_sampler_cuda.cu
- `LUCE_GPU_VERIFY_ARGMAX` - qwen35_dflash_target.cpp
- `LUCE_HIP_BACKEND_PATH` - dynamic_backend.cpp
- `LUCE_IGNORE_EOS` - laguna_backend.cpp
- `LUCE_KVFLASH` - gemma4_backend.cpp, gemma4_layer_split_adapter.cpp, kvflash_pager.h, laguna_backend.cpp, laguna_layer_split_adapter.cpp, qwen35_backend.cpp, qwen35_layer_split_adapter.cpp
- `LUCE_KVFLASH_DRAFTER` - kvflash_pager.h
- `LUCE_KVFLASH_MAX_POOL` - kvflash_pager.h
- `LUCE_KVFLASH_POLICY` - kvflash_pager.h
- `LUCE_KVFLASH_TAU` - gemma4_backend.cpp, gemma4_layer_split_adapter.cpp, laguna_backend.cpp, laguna_layer_split_adapter.cpp, qwen35_layer_split_adapter.cpp
- `LUCE_LAGUNA_AUTO_HEAD_MAJOR` - laguna_backend.cpp
- `LUCE_LAGUNA_CACHE_SLOTS` - laguna_backend.cpp
- `LUCE_LAGUNA_DRAFT_PAD` - laguna_backend.cpp
- `LUCE_LAGUNA_DSPARK` - laguna_backend.cpp
- `LUCE_LAGUNA_DSPARK_CONFIDENCE_THRESHOLD` - laguna_backend.cpp
- `LUCE_LAGUNA_DSPARK_TREE` - laguna_backend.cpp
- `LUCE_LAGUNA_EXPERT_CACHE` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_TP_TARGETED_JOIN_SPLIT` - moe_hybrid_ffn_eval.cpp
- `LUCE_LAGUNA_FUSED_DOMINO` - laguna_backend.cpp
- `LUCE_LAGUNA_FUSED_DSPARK` - laguna_backend.cpp
- `LUCE_LAGUNA_FUSED_QK` - laguna_target_loader.cpp
- `LUCE_LAGUNA_FUSE_FFN` - laguna_backend.cpp
- `LUCE_LAGUNA_GPU_ARGMAX` - laguna_backend.cpp
- `LUCE_LAGUNA_GPU_REMAP` - moe_hybrid_ffn_eval.cpp
- `LUCE_LAGUNA_HOTNESS` - laguna_backend.cpp
- `LUCE_LAGUNA_KV_HEAD_MAJOR` - laguna_backend.cpp, laguna_target_graph.cpp
- `LUCE_LAGUNA_LAYER_SPLIT_UBATCH` - laguna_layer_split_adapter.cpp
- `LUCE_LAGUNA_MOE_FUSED_COMBINE` - laguna_target_graph.cpp
- `LUCE_LAGUNA_MOE_STUB` - laguna_target_graph.cpp
- `LUCE_LAGUNA_NEXT_PLACEMENT_OUT` - laguna_backend.cpp
- `LUCE_LAGUNA_NO_KVPAD` - laguna_dflash_target.cpp, laguna_target_graph.cpp
- `LUCE_LAGUNA_NO_SINGLE_GRAPH` - laguna_backend.cpp
- `LUCE_LAGUNA_PAD_CPY` - laguna_dflash_target.cpp, laguna_target_graph.cpp
- `LUCE_LAGUNA_PERSIST_VERIFY` - laguna_target_graph.cpp
- `LUCE_LAGUNA_PREGATE_MAX` - laguna_backend.cpp
- `LUCE_LAGUNA_PREGATE_TRACE` - laguna_backend.cpp
- `LUCE_LAGUNA_PROFILE` - laguna_backend.cpp
- `LUCE_LAGUNA_SWAP_MAX` - laguna_backend.cpp
- `LUCE_LAGUNA_SWAP_MIN_GAIN` - laguna_backend.cpp
- `LUCE_LAGUNA_SWA_RING` - laguna_backend.cpp
- `LUCE_LAGUNA_TELEMETRY` - laguna_backend.cpp
- `LUCE_LAGUNA_VERIFY_WIDTH` - laguna_backend.cpp
- `LUCE_LAGUNA_VERIFY_WIDTH_MAX` - laguna_backend.cpp
- `LUCE_MAX_CONTEXT` - laguna_backend.cpp, qwen35moe_backend.cpp
- `LUCE_MMID_TELEMETRY` - ggml-cuda.cu, mmvq.cu
- `LUCE_MMQ_FULL_BATCH_MIN` - moe_hybrid_ffn_eval.cpp
- `LUCE_MMQ_SUB_BATCH` - moe_hybrid_ffn_eval.cpp
- `LUCE_MMVF_MAX_NCOLS_F16` - ggml-cuda/mmvf.cu
- `LUCE_MODEL_CARDS_DIR` - model_card.cpp
- `LUCE_MOE_COLD_BACKEND` - deepseek4_loader.cpp
- `LUCE_MOE_COMBINE_VEC4` - ggml-cuda/moe-fused.cu
- `LUCE_MOE_COMPACT_MATERIALIZED` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_DUPLICATE_HOT_ON_COLD` - moe_hybrid_storage.cpp
- `LUCE_MOE_EXPERT_COMPUTE_DAEMON_TOKEN_LOOP` - moe_expert_compute_ipc.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY` - moe_expert_compute_ipc.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_BIN` - moe_expert_compute.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_DTYPE` - moe_expert_compute_ipc.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_GPU` - deepseek4_backend.cpp, moe_expert_compute.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_MODE` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_PROFILE` - moe_expert_compute_ipc.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_REQUIRED` - moe_expert_compute.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_SHARED_BYTES` - moe_expert_compute_ipc.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_TRANSPORT` - moe_expert_compute_ipc.cpp
- `LUCE_MOE_EXPERT_COMPUTE_IPC_WORK_DIR` - moe_expert_compute.cpp
- `LUCE_MOE_EXPERT_COMPUTE_THREADS` - moe_expert_compute_cpu.cpp
- `LUCE_MOE_EXPERT_MAJOR_GPU_REDUCE` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_EXPERT_MAJOR_PREFILL` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_FIXED_SLOT_GRAPHS` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_FIXED_SLOT_MAX` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_FULL_COLD_PARALLEL` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_FUSED_COMBINE` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_PREFILL_DEVICE_INPUT` - deepseek4_graph.cpp
- `LUCE_MOE_PREFILL_HOT_SUB_BATCH` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_PREFILL_MASKED_COLD` - moe_hybrid_ffn_eval.cpp
- `LUCE_MOE_PREFILL_PERSISTENT_OWNER_ALLOC` - deepseek4_graph.cpp
- `LUCE_MOE_TP_BACKEND` - deepseek4_backend.cpp
- `LUCE_NO_MASK` - laguna_backend.cpp
- `LUCE_NO_MOE_ROUTER_FUSE` - qwen35moe_ffn.cpp
- `LUCE_NO_MOE_SWIGLU_FUSE` - qwen35moe_ffn.cpp
- `LUCE_NO_PREAD` - deepseek4_loader.cpp
- `LUCE_PROF` - prof_env.h
- `LUCE_MMPROJ` - scripts/entrypoint.sh (maps to `--mmproj`)
- `LUCE_PREFILL_CACHE_SLOTS` - scripts/entrypoint.sh (maps to `--prefill-cache-slots`)
- `LUCE_PREFILL_POOL_TRIM_TOKENS` - qwen35_backend.cpp (OPT-IN: trim legacy device pools during long prefills)
- `LUCE_PREFILL_TIMING` - qwen35_backend.cpp (DEBUG: per-ubatch prefill build/alloc/compute timing)
- `LUCE_PREFIX_CACHE_SLOTS` - scripts/entrypoint.sh (maps to `--prefix-cache-slots`)
- `LUCE_QWEN35MOE_CACHE_SLOTS` - qwen35moe_backend.cpp
- `LUCE_QWEN35MOE_HOTNESS` - qwen35moe_backend.cpp
- `LUCE_QWEN35MOE_NEXT_PLACEMENT_OUT` - qwen35moe_backend.cpp
- `LUCE_QWEN35MOE_NO_KVPAD` - qwen35moe_pipelined_decode.cpp
- `LUCE_QWEN35MOE_NO_ROUTED` - qwen35moe_pipelined_decode.cpp
- `LUCE_QWEN35MOE_RUNTIME_STATS_OUT` - qwen35moe_backend.cpp
- `LUCE_QWEN35MOE_SWAP_MAX` - qwen35moe_backend.cpp
- `LUCE_QWEN35MOE_SWAP_MIN_GAIN` - qwen35moe_backend.cpp
- `LUCE_QWEN35MOE_TELEMETRY` - qwen35moe_backend.cpp
- `LUCE_QWEN35_NO_KVPAD` - graph_builders.cpp
- `LUCE_QWEN35_AR_BURST` - qwen35_backend.cpp (adaptive plain-decode burst length; default 40)
- `LUCE_QWEN35_DFLASH2_TREE` - qwen35_backend.cpp (=0 uses raw top-k tree scoring instead of the selector under --ddtree)
- `LUCE_QWEN35_DSPARK` - qwen35_backend.cpp (=0 disables the DSpark drafter head path)
- `LUCE_QWEN35_DSPARK_CONF_DEBUG` - qwen35_backend.cpp (DEBUG: print confidence-gate scores)
- `LUCE_QWEN35_DSPARK_CONFIDENCE_THRESHOLD` - qwen35_backend.cpp (adaptive block-length confidence gate)
- `LUCE_QWEN35_DSPARK_TREE` - qwen35_backend.cpp (DSpark tree drafting toggle)
- `LUCE_QWEN35_FUSED_DSPARK` - qwen35_backend.cpp (=0 keeps the DSpark heads off the fused graph)
- `LUCE_QWEN35_NO_FUSED_KERNELS` - qwen35_target_graph.cpp (KILL SWITCH: =1 rebuilds the pre-fusion decode graph)
- `LUCE_QWEN35_NO_STACK` - gguf_target_loader.cpp (KILL SWITCH: =1 disables zero-copy stacked weight aliases)
- `LUCE_QWEN35_SPEC_STEP_RATIO` - qwen35_backend.cpp (adaptive policy spec/plain step-time ratio override)
- `LUCE_ROCMFP3_WIDE_TWO_PASS` - rocmfp3_mix.cu
- `LUCE_ROCMFP3_ROW3` - rocmfp3_mix.cu (KILL SWITCH: =0 restores two-row tiles on gfx1151)
- `LUCE_SAMPLED_VERIFY` - laguna_backend.cpp, qwen35_backend.cpp
- `LUCE_SHARE_DIR` - http_server.cpp
- `LUCE_SINGLE_CHAIN_CHECKPOINT_F32` - chain_rollback_policy.h
- `LUCE_SINGLE_CHAIN_ROLLBACK_DIAG` - chain_rollback_policy.h
- `LUCE_SPARK` - laguna_backend.cpp, qwen35moe_backend.cpp
- `LUCE_SPARK_VRAM_MB` - laguna_backend.cpp, qwen35moe_backend.cpp
- `LUCE_SPLIT_CAPTURE_SELFTEST` - qwen35_layer_split_dflash_target.cpp
- `LUCE_SPLIT_CHAIN_ROLLBACK_DIAG` - qwen35_layer_split_dflash_target.cpp, qwen35_target_graph.cpp
- `LUCE_SPLIT_FAST_ROLLBACK` - chain_rollback_policy.h
- `LUCE_STALL_TOOL_PREFIX` - http_server.cpp
- `LUCE_SV_DEBUG` - qwen35_backend.cpp
- `LUCE_TARGET_SHARD_IPC_SHARED_BYTES` - target_shard_ipc.cpp
- `LUCE_TARGET_SHARD_IPC_TRANSPORT` - target_shard_ipc.cpp
- `LUCE_TOPK_PROFILE` - geometric_draft_topk_cuda.cu
- `LUCE_TOPK_SPLIT` - geometric_draft_topk_cuda.cu
- `LUCE_VERIFY_WIDTH` - qwen35moe_backend.cpp
- `FAST_ROLLBACK_DIAG` - qwen35_dflash_target.cpp
- `LUCE_FA256_MMA` - fattn.cu (ggml-cuda) (=0 restores the tile kernel for head-256 on RDNA4)
- `LUCE_FA256_WMMA` - fattn.cu (ggml-cuda) (=1 forces the rocWMMA kernel on every head-256 RDNA4 shape)
- `LUCE_FA256_WMMA_MAX_KV` - fattn.cu (ggml-cuda) (rocWMMA/raw-MMA crossover KV length in flag builds)
- `GGML_HIP_ROCWMMA_FATTN` - server/CMakeLists.txt (BUILD OPTION, not an env var: compiles the rocWMMA fattn kernel; required by `LUCE_FA256_WMMA` and the sub-32K head-256 prefill route)
- `GGML_CUDA_BATCH_PEER_COPIES` - ggml-cuda.cu (ggml-cuda), deepseek4_fused_verify.inc, moe_hybrid_ffn_eval.cpp
- `GGML_CUDA_GRAPH_MAX_KEYS` - common.cuh (ggml-cuda)
- `GGML_CUDA_MLA_DENSE_HIGH_RATIO` - fattn.cu, deepseek4_backend.cpp, deepseek4_graph.cpp
- `GGML_CUDA_MLA_DENSE_WMMA` - fattn.cu, deepseek4_backend.cpp, deepseek4_graph.cpp
- `GGML_CUDA_MLA_NO_SPLIT_KV` - ds4-env.cuh (fattn.cu)
- `GGML_CUDA_MLA_SEGMENTED_KV` - deepseek4_graph.cpp
- `GGML_CUDA_MLA_SPARSE_VALUE_SKIP` - fattn.cu
- `GGML_CUDA_MLA_STREAM_WMMA` - fattn.cu, deepseek4_backend.cpp
- `GGML_CUDA_MLA_STREAM_WMMA_HEAD_GROUPS` - fattn.cu, deepseek4_backend.cpp
- `GGML_DS4_INDEXER_M32` - ds4-indexer.cu (KILL SWITCH: =0 disables the rocWMMA m32 indexer kernel on RDNA 3.5)
- `GGML_DS4_INDEXER_M32_CACHE_B` - ds4-indexer.cu, deepseek4_backend.cpp (gfx1151 default: cached B operand)
- `GGML_DS4_INDEXER_M32_PREFILL` - ds4-indexer.cu (DIAGNOSTIC: m32 prefill crossover override)
- `GGML_DS4_INDEXER_M32_DIRECT_B` - ds4-indexer.cu (DIAGNOSTIC: m32 direct-B crossover override)
- `GGML_CUDA_MMQ_X` - ggml-cuda/mmq.cuh
- `GGML_CUDA_MMQ_MOE_ADAPTIVE_X` - ggml-cuda/mmq.cuh
- `GGML_CUDA_MMQ_MOE_PERSISTENT` - ggml-cuda/mmq.cuh
- `GGML_CUDA_MMQ_MOE_PERSISTENT_BLOCKS_PER_CU` - ggml-cuda/mmq.cuh
- `GGML_CUDA_MLA_STREAM_TOPK` - ggml-cuda/fattn.cu
- `GGML_DS4_FA_STREAM_TOPK` - ggml-cuda/fattn.cu (compatibility alias)
- `GGML_CUDA_MLA_STREAM_F32_STAGE` - ggml-cuda/fattn.cu
- `GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS` - ggml-cuda/paged-attn.cu (DIAGNOSTIC: force the paged-attention partition count on both routes)
- `GGML_CUDA_MLA_STREAM_FAST_EXP` - ggml-cuda/fattn.cu
- `GGML_CUDA_MLA_SPLIT_KV` - ds4-env.cuh (fattn.cu)
- `GGML_DS4_FA_NO_SPLIT_KV` - ds4-env.cuh (fattn.cu)
- `GGML_DS4_FA_SPLIT_KV` - ds4-env.cuh (fattn.cu)
- `GGML_DS4_TOPK_BLOCK_RADIX` - top-k.cu
- `HOME` - spark_corpus.cpp
- `LUCE_CUDA_I32_REPEAT` - ggml-cuda.cu (ggml-cuda)
- `LUCE_Q8_MEMO` - mmvq.cu (set to 0 to disable q8_1 activation memoisation; on by default)
- `LUCE_MMQ_BIG_PREFILL` - mmq.cu (=0 disables the RDNA4 128-wide MMQ tiles for large prefill batches)
- `LUCE_MMVQ_MAX_NCOLS` - deepseek4_backend.cpp
- `LUCE_QK_FUSE_LAYERS` - laguna_target_graph.cpp
- `LUCE_QK_FUSE_MODE` - laguna_target_graph.cpp
- `PFLASH_DRAFTER_EARLY_EXIT_N` - qwen3_graph.cpp
- `PFLASH_DRAFTER_SCORE_LAYERS` - qwen3_graph.cpp
- `PFLASH_FREEZE_HOT_WINDOW` - http_server.cpp
- `TMPDIR` - backend_ipc.cpp, moe_expert_compute_ipc.cpp
