# Environment Variables Reference

Summary of `LUCE_*` environment variables recognized across the
codebase, grouped by subsystem. Most are runtime toggles read via `getenv` /
`os.environ`; a few are build/compile-time or harness knobs (noted where relevant).

> Policy (see `server/docs/ENVIRONMENT.md`): new features should ship as CLI flags
> or defaults. Env vars are reserved for burn-in kill switches and debug
> instrumentation. Treat undocumented variables as internal.

### Status legend

Tags in the tables below flag variables that are **not** part of the intended
long-term serving surface:

- 🐛 **debug** — profiling/telemetry/ablation instrumentation. Zero-cost when
  unset and never required for correct serving; safe to ignore in production.
- 🔀 **kill-switch** — burn-in toggle for a landed default, documented with the
  intent to be deleted once the feature has soaked.
- 🧪 **test/bench** — only read by tests, benchmarks, or harness scripts.
- ⚠️ **removal candidate** — legacy/one-off/likely-obsolete; prefer the CLI flag
  or default where one exists.

Untagged variables are operational tuning knobs.

## Server / runtime configuration

| Variable | Purpose |
|---|---|
| `LUCE_HOST` | Server bind host. |
| `LUCE_PORT` | Server bind port. |
| `LUCE_BIN` | 🧪 **test/bench** Path to `test_dflash` for the bench scripts. |
| `LUCE_SERVER_BIN` | Path to `luce_server` (container entrypoint, harness). |
| `LUCE_BIN_AR` | Alternate/AR binary path for benchmarks. |
| `LUCE_DIR` | Base working directory. |
| `LUCE_SHARE_DIR` | Static/share asset directory served by the HTTP server. |
| `LUCE_MODEL_CARDS_DIR` | Directory of model-card definitions. |
| `LUCE_MODEL_NAME` | Model name/identifier. |
| `LUCE_TOKENIZER` | Tokenizer path/identifier. |
| `LUCE_TARGET` | Target model path (container entrypoint, `run.py`). |
| `LUCE_DRAFT` | Draft file or directory (container entrypoint, `run.py`); `none` disables the container draft. |
| `LUCE_TARGET_DEVICE` | Default `--target-device` (`backend:gpu` or `auto`) when no flag or profile names one; the container sets `auto`. |
| `LUCE_PROFILE` / `LUCE_ARGS` | Container entrypoint: `--profile` name and extra `luce_server` flags. |
| `LUCE_IMAGE_INFO_PATH` | Path to image/build info metadata. |
| `LUCE_MAX_CTX` | Container entrypoint `--max-ctx` (default: sized from the GPU's memory). |
| `LUCE_MAX_CONTEXT` | KV sizing override for laguna and qwen35moe expert placement. |
| `LUCE_DEFAULT_MAX_TOKENS` | Default generation token cap. |
| `LUCE_IGNORE_EOS` | Ignore EOS token during generation. |
| `LUCE_LAZY` | Container entrypoint: `1` adds `--lazy-draft` (needs a draft and `LUCE_PREFILL_DRAFTER`). |

## GPU / backend placement

| Variable | Purpose |
|---|---|
| `LUCE_TARGET_GPU` / `LUCE_TARGET_GPUS` | GPU(s) assigned to the target model. |
| `LUCE_TARGET_LAYER_SPLIT` | Layer-split placement across GPUs for the target. |
| `LUCE_DRAFT_GPU` | GPU assigned to the drafter. |
| `LUCE_CUDA_ARCHES` / `LUCE_HIP_ARCHES` | CUDA/HIP architecture targets (build). |
| `LUCE_GPU_BACKEND` / `LUCE_BACKEND_CUDA` / `LUCE_BACKEND_HIP` | Backend selection (build/compile-time). |
| `LUCE_HIP_NO_AUTO_UMA` | Disable automatic HIP UMA (unified memory) selection. |
| `LUCE_HIP_UMA_MIN_FRAC` | Minimum VRAM fraction before HIP UMA kicks in. |
| `LUCE_WAVE_SIZE` | HIP wave size (compile flag, e.g. gfx1151 needs 32). |

## Speculative decoding — drafter

| Variable | Purpose |
|---|---|
| `LUCE_DRAFT_KV` | 🔀 **kill-switch** `=0` restores per-step drafter window recompute instead of the ring cache. |
| `LUCE_DRAFT_PERSIST` | Persist drafter state across steps. |
| `LUCE_DISABLE_DRAFT_ATTN` | 🐛 **debug** Disable drafter attention block (ablation). |
| `LUCE_DISABLE_DRAFT_ATTN_GATE` | 🐛 **debug** Disable drafter attention gate (ablation). |
| `LUCE_DISABLE_DRAFT_AUX_NORMS` | 🐛 **debug** Disable auxiliary norms in the drafter (ablation). |
| `LUCE_DISABLE_DRAFT_FFN` | 🐛 **debug** Disable drafter FFN block (ablation). |
| `LUCE_DISABLE_DRAFT_SWA` | 🐛 **debug** Disable drafter sliding-window attention (ablation). |
| `LUCE_DOMINO_ZERO_START` | Domino head zero-start behavior. |
| `LUCE_DRAFT_FP16` | Load drafter in FP16. |
| `LUCE_DRAFT_SWA` | Enable drafter SWA. |
| `LUCE_DRAFT_CTX_MAX` | Drafter max context. |
| `LUCE_DRAFT_BLOCK_SIZE` / `LUCE_DRAFT_LAYERS` / `LUCE_DRAFT_N_TARGET_LAYERS` / `LUCE_DRAFT_MASK_TOKEN_ID` | Drafter geometry (build/config). |

## Draft IPC transport

| Variable | Purpose |
|---|---|
| `LUCE_DRAFT_IPC_TRANSPORT` | IPC transport for the draft process. |
| `LUCE_DRAFT_IPC_SHARED_BYTES` | Shared-memory size for draft IPC. |
| `LUCE_DRAFT_IPC_RING_CAP` | Ring buffer capacity for draft IPC. |
| `LUCE_DRAFT_IPC_BIN` | Draft IPC daemon binary. |
| `LUCE_DRAFT_IPC_GPU` | GPU for the draft IPC daemon. |
| `LUCE_DRAFT_IPC_WORK_DIR` | Working directory for draft IPC. |

## Verification / sampling

| Variable | Purpose |
|---|---|
| `LUCE_SAMPLED_VERIFY` | Use sampled verification. |
| `LUCE_VERIFY_WIDTH` | Verify batch width. |
| `LUCE_GPU_SAMPLE` | GPU sampling path. |
| `LUCE_GPU_ARGMAX` / `LUCE_GPU_VERIFY_ARGMAX` | GPU argmax for sampling/verification. |
| `LUCE_GPU_DRAFT_TOPK` | GPU draft top-k. |
| `LUCE_TQ3_VERIFY` | TQ3-quantized verify path. |
| `LUCE_N_SAMPLE` / `LUCE_SAMP` | 🧪 **test/bench** Sample count/mode. |
| `LUCE_SAMPLER_BENCH` | 🧪 **test/bench** Sampler benchmark mode. |
| `LUCE_SV_DEBUG` | 🐛 **debug** Sampled-verify debug output. |

## Adaptive experts / adaptive verify width

| Variable | Purpose |
|---|---|
| `LUCE_ADAPTIVE_K_TAU` | Cumulative combine-weight threshold for per-token expert gating (prefer `--adaptive-experts`). |
| `LUCE_ADAPTIVE_K_DENSE` | CSV of MoE layers kept dense under adaptive-K. |
| `LUCE_ADAPTIVE_WIDTH_MIN` | Minimum adaptive verify width. |
| `LUCE_ADAPTIVE_WIDTH_THETA` | Threshold controlling adaptive verify width. |
| `LUCE_HYBRID_HOT_PCT` | Hot-expert percentage for hybrid MoE. |

## KVFlash (KV cache pager)

| Variable | Purpose |
|---|---|
| `LUCE_KVFLASH` | Enable KVFlash (prefer CLI `--kvflash`; token count or `auto`). |
| `LUCE_KVFLASH_DRAFTER` | KVFlash for the drafter cache. |
| `LUCE_KVFLASH_MAX_POOL` | Max KVFlash pool size. |
| `LUCE_KVFLASH_POLICY` | KVFlash eviction/placement policy. |
| `LUCE_KVFLASH_TAU` | KVFlash tau threshold. |
| `LUCE_LAGUNA_SWA_RING` | 🔀 **kill-switch** `=0` keeps SWA layers on pool-sized caches under KVFlash. |

## KV cache quantization / dtype

| Variable | Purpose |
|---|---|
| `LUCE_CACHE_TYPE_K` / `LUCE_CACHE_TYPE_V` | KV cache K/V dtype. |
| `LUCE_KV_TYPE` | KV cache type selector. |
| `LUCE_FEATURE_DTYPE` | Draft feature-ring dtype. |
| `LUCE_KV_F16` | F16 KV cache. |
| `LUCE_KV_K` / `LUCE_KV_V` | Per-side (K/V) KV quantization. |
| `LUCE_KV_Q4` / `LUCE_KV_TQ3` / `LUCE_KV_TBQ` | Quantized KV formats (Q4 / TQ3 / TBQ). |
| `LUCE_PFLASH_K_TYPE` | PFlash K dtype. |

## FlashPrefill / prefill

| Variable | Purpose |
|---|---|
| `LUCE_FP_USE_BSA` | Use block-sparse attention in flash prefill. |
| `LUCE_FP_ALPHA` | FlashPrefill alpha parameter. |
| `LUCE_FP_CHUNK_S` | FlashPrefill chunk size. |
| `LUCE_FP_NOPE_TAIL` | NoPE tail handling in flash prefill. |
| `LUCE_FP_HIP_ROW` | HIP row-kernel path for flash prefill. |
| `LUCE_FP_SKIP_PREWARM` | Skip flash-prefill prewarm. |
| `LUCE_FP_PROFILE` / `LUCE_FP_DUMP_COUNTS` / `LUCE_FP_DEBUG_LAYER0` | 🐛 **debug** FlashPrefill profiling/debug. |
| `LUCE_PREFILL_MODE` | Prefill mode selector. |
| `LUCE_PREFILL_THRESHOLD` | Prefill length threshold. |
| `LUCE_PREFILL_DRAFTER` | Drafter participation during prefill. |
| `LUCE_PREFILL_KEEP` | Keep prefill cache across requests. |
| `LUCE_PREFILL_CACHE_SLOTS` / `LUCE_PREFIX_CACHE_SLOTS` | Optional prefill/prefix cache slot override. When unset, the container preserves the native server defaults (prefix: 32; exact prefill: 0). Set either value to `0` for an explicit opt-out. |
| `LUCE_PREFILL_POOL_TRIM_TOKENS` | Opt-in interval for trimming cached legacy CUDA/HIP pool allocations between completed Qwen3.5 prefill chunks. Useful when a single long, shape-changing prefill would otherwise exhaust VRAM before request cleanup. |
| `LUCE_PREFILL_CACHE_TEST_LOG` / `LUCE_PREFILL_CACHE_TEST_PORT` | 🧪 **test/bench** Prefill-cache test harness. |
| `LUCE_LAYER_PREFILL` / `LUCE_PREFILL_UBATCH` | Layer-split prefill / prefill micro-batch. |
| `LUCE_CHUNKED` / `LUCE_CHUNKED_CHUNK` / `LUCE_CHUNKED_Q_BATCH` / `LUCE_CHUNKED_THRESHOLD` | Chunked prefill controls. |
| `LUCE_LAGUNA_CHUNK` | Laguna prefill chunk size. |
| `LUCE_FA_WINDOW` / `LUCE_FA_WINDOW` | Flash-attention window size. |

## Laguna backend

| Variable | Purpose |
|---|---|
| `LUCE_LAGUNA_PROFILE` / `LUCE_LAGUNA_TELEMETRY` | 🐛 **debug** Profiling / telemetry. |
| `LUCE_LAGUNA_AUTO_HEAD_MAJOR` / `LUCE_LAGUNA_KV_HEAD_MAJOR` | Head-major KV layout. |
| `LUCE_LAGUNA_CACHE_SLOTS` | Cache slot count. |
| `LUCE_LAGUNA_DRAFT_PAD` | Drafter padding. |
| `LUCE_LAGUNA_DSPARK` / `LUCE_LAGUNA_DSPARK_TREE` / `LUCE_LAGUNA_DSPARK_CONFIDENCE_THRESHOLD` | DSpark speculative controls. |
| `LUCE_LAGUNA_EXPERT_CACHE` | Expert cache toggle. |
| `LUCE_LAGUNA_FUSED_DOMINO` / `LUCE_LAGUNA_FUSED_DSPARK` / `LUCE_LAGUNA_FUSED_QK` / `LUCE_LAGUNA_FUSE_FFN` / `LUCE_LAGUNA_MOE_FUSED_COMBINE` | Kernel fusion toggles. |
| `LUCE_LAGUNA_GPU_ARGMAX` / `LUCE_LAGUNA_GPU_REMAP` | GPU argmax / expert remap. |
| `LUCE_LAGUNA_HOTNESS` | Expert hotness tracking. |
| `LUCE_LAGUNA_LAYER_SPLIT_UBATCH` | Layer-split micro-batch. |
| `LUCE_LAGUNA_MOE_STUB` | 🐛 **debug** Stub MoE (ablation). |
| `LUCE_LAGUNA_NEXT_PLACEMENT_OUT` | 🐛 **debug** Dump next-placement plan. |
| `LUCE_LAGUNA_NO_KVPAD` / `LUCE_LAGUNA_PAD_CPY` | KV padding controls. |
| `LUCE_LAGUNA_NO_SINGLE_GRAPH` | Disable single-graph capture. |
| `LUCE_LAGUNA_PERSIST_VERIFY` | Persist verify graph. |
| `LUCE_LAGUNA_PREGATE_MAX` / `LUCE_LAGUNA_PREGATE_TRACE` | Pre-gating max; 🐛 **debug** trace. |
| `LUCE_LAGUNA_SWAP_MAX` / `LUCE_LAGUNA_SWAP_MIN_GAIN` | Expert-swap thresholds. |
| `LUCE_LAGUNA_VERIFY_WIDTH` / `LUCE_LAGUNA_VERIFY_WIDTH_MAX` | Verify width limits. |
| `LUCE_LAGUNA_BENCH_NO_LOGITS` | 🧪 **test/bench** Skip logits in benchmarks. |

## Qwen3.5 MoE backend

| Variable | Purpose |
|---|---|
| `LUCE_QWEN35MOE_CACHE_SLOTS` | Cache slot count. |
| `LUCE_QWEN35MOE_HOTNESS` | Expert hotness tracking. |
| `LUCE_QWEN35MOE_SWAP_MAX` / `LUCE_QWEN35MOE_SWAP_MIN_GAIN` | Expert-swap thresholds. |
| `LUCE_QWEN35MOE_TELEMETRY` | 🐛 **debug** Telemetry. |
| `LUCE_QWEN35MOE_NEXT_PLACEMENT_OUT` / `LUCE_QWEN35MOE_RUNTIME_STATS_OUT` | 🐛 **debug** Dump placement / runtime stats. |
| `LUCE_QWEN35MOE_NO_KVPAD` / `LUCE_QWEN35_NO_KVPAD` | KV padding controls. |
| `LUCE_QWEN35MOE_NO_ROUTED` | 🐛 **debug** Disable routed experts (ablation). |
| `LUCE_QWEN35MOE_PREFILL_CHUNK` | Prefill chunk size. |
| `LUCE_QWEN35MOE_HYBRID_SPEC_MIN_ACCEPT_RATE` / `LUCE_QWEN35MOE_HYBRID_SPEC_MIN_STEPS_BEFORE_AR` | Hybrid speculative acceptance thresholds. |

## Gemma4 backend

| Variable | Purpose |
|---|---|
| `LUCE_GEMMA4_LAYER_SPLIT_UBATCH` | Layer-split micro-batch. |
| `LUCE_GEMMA4_NO_KVPAD` | Disable KV padding. |
| `LUCE_G4_BSA_CHUNK` | Block-sparse attention chunk size. |

## DeepSeek4 (DS4) backend

| Variable | Purpose |
|---|---|
| `LUCE_DS4_TIMING` | 🐛 **debug** DS4 timing instrumentation. |
| `LUCE_DS4_CUDA_LAYERS` | Number of DS4 layers on CUDA. |
| `LUCE_DS4_SPEC` / `LUCE_DS4_DRAFT` / `LUCE_DS4_DRAFT_GPU` | Enable the local DSpark drafter, select its GGUF, and choose its HIP device. |
| `LUCE_DS4_MOE_TP` / `LUCE_DS4_MOE_TP_INPROC` / `LUCE_DS4_MOE_TP_GPU` | Burn-in controls for in-process route-owner expert parallelism and the cold-owner HIP device. |
| `LUCE_DS4_HOTNESS_CSV` | Optional per-layer expert routing profile for hot placement. |
| `LUCE_MOE_COLD_BACKEND` | Cold-expert compute backend. |
| `LUCE_NO_PREAD` | Disable pread-based weight loading. |

## MoE expert compute / IPC

| Variable | Purpose |
|---|---|
| `LUCE_MOE_HYBRID_PREFILL_EAGER` / `LUCE_MOE_PREFILL_TRACE` | Model-neutral heterogeneous prefill policy and tracing. The legacy `LUCE_DS4_*` spellings remain aliases. |
| `LUCE_MOE_TP_GROUPED_MMVQ` / `LUCE_MOE_TP_FUSED_GATE_UP` | Model-neutral grouped and fused routed-FFN kernel qualification switches. The legacy `LUCE_DS4_*` spellings remain aliases. |
| `LUCE_MOE_TP_COARSE_OWNER` / `LUCE_MOE_TP_COARSE_OWNER_SPLIT` | Model-neutral owner-op lowering switches. The legacy `LUCE_DS4_*` spellings remain aliases. |
| `LUCE_MOE_TP_DEVICE_JOIN` / `LUCE_MOE_TP_ROUTE_PREFORK` | Model-neutral cross-owner scheduling switches. The legacy `LUCE_DS4_*` spellings remain aliases. |
| `LUCE_MOE_EXPERT_COMPUTE_THREADS` / `LUCE_COLD_THREADS` | CPU threads for expert compute. |
| `LUCE_MOE_EXPERT_COMPUTE_BATCH` / `LUCE_MOE_EXPERT_COMPUTE_BATCH_MAX` | Expert compute batch sizing. |
| `LUCE_MOE_EXPERT_COMPUTE_IPC_MODE` | Expert-compute IPC mode. |
| `LUCE_MOE_EXPERT_COMPUTE_IPC_TRANSPORT` | IPC transport. |
| `LUCE_MOE_EXPERT_COMPUTE_IPC_SHARED_BYTES` | Shared-memory size. |
| `LUCE_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY` | IPC batch capacity. |
| `LUCE_MOE_EXPERT_COMPUTE_IPC_DTYPE` | IPC payload dtype. |
| `LUCE_MOE_EXPERT_COMPUTE_IPC_PROFILE` | 🐛 **debug** IPC profiling. |
| `LUCE_MOE_EXPERT_COMPUTE_IPC_BIN` / `LUCE_MOE_EXPERT_COMPUTE_IPC_GPU` / `LUCE_MOE_EXPERT_COMPUTE_IPC_WORK_DIR` / `LUCE_MOE_EXPERT_COMPUTE_IPC_REQUIRED` | IPC daemon binary / GPU / work dir / required flag. |
| `LUCE_MOE_FIXED_SLOT_GRAPHS` / `LUCE_MOE_FIXED_SLOT_MAX` | Fixed-slot MoE graph controls. |
| `LUCE_MOE_PREFILL_HOT_SUB_BATCH` | Hot-expert prefill sub-batch. |
| `LUCE_MOE_PREFILL_PERSISTENT_OWNER_ALLOC` | Kill switch for persistent long-prefill route and owner arenas. |
| `LUCE_NO_MOE_ROUTER_FUSE` / `LUCE_NO_MOE_SWIGLU_FUSE` | Disable router / SwiGLU fusion. |
| `LUCE_EXPERT_BUDGET_MB` / `LUCE_EXPERT_BUDGET_PCT` | Expert VRAM budget (absolute / percent). |
| `LUCE_DROP_COLD` | Drop cold experts. |
| `LUCE_COLLECT_ROUTING` | 🐛 **debug** Collect routing statistics. |

## Target-shard IPC

| Variable | Purpose |
|---|---|
| `LUCE_TARGET_SHARD_IPC_TRANSPORT` | Transport for target-shard IPC. |
| `LUCE_TARGET_SHARD_IPC_SHARED_BYTES` | Shared-memory size for target-shard IPC. |

## Matmul / MMID / MMVQ kernels

| Variable | Purpose |
|---|---|
| `LUCE_MMID_GROUPED` | Grouped `MUL_MAT_ID` kernel for small verify batches. |
| `LUCE_MMID_GROUPED_TYPES` | Types eligible for the grouped MMID kernel. |
| `LUCE_MMID_GROUPED_DEVICE` | Optional device restriction for the grouped MMID kernel. |
| `GGML_CUDA_BATCH_PEER_COPIES` | Batch ordered CUDA/HIP peer copies behind one dependency per source/destination pair. |
| `LUCE_MMQ_FULL_BATCH_MIN` / `LUCE_MMQ_SUB_BATCH` | MMQ batch thresholds. |
| `LUCE_CUDA_MMVQ_TOKENWISE` / `LUCE_CUDA_MMVQ_MOE_TOKENWISE` / `LUCE_CUDA_MMVQ_MOE_KERNEL` | MMVQ token-wise / MoE kernel selection. |
| `LUCE_GDN_FORCE_GROUPED_COLS` / `LUCE_GDN_NO_GROUPED_COLS` | Gated-delta-net grouped-column control. |
| `LUCE_NO_MASK` | 🐛 **debug** Disable attention masking (ablation). |

## Top-k kernels

| Variable | Purpose |
|---|---|
| `LUCE_TOPK_PROFILE` | 🐛 **debug** Top-k kernel profiling. |
| `LUCE_TOPK_SPLIT` | Top-k split strategy. |
| `LUCE_TOPK_CASE` / `LUCE_TOPK_CONSUME` / `LUCE_TOPK_LAUNCH` | Top-k kernel case/consume/launch tuning. |

## Spark

| Variable | Purpose |
|---|---|
| `LUCE_SPARK` | Enable Spark. |
| `LUCE_SPARK_VRAM_MB` | Spark VRAM budget. |
| `LUCE_SPARK_CLAUDE_DIR` / `LUCE_SPARK_CODEX_DIR` | Spark corpus directories. |

## KV / context compression

| Variable | Purpose |
|---|---|
| `LUCE_COMPRESS_NO_PARK` | Disable parking of compressed blocks. |
| `LUCE_COMPRESS_ANCHOR_RADIUS` / `LUCE_COMPRESS_MAX_ANCHOR_HITS` | Anchor radius / max hits. |
| `LUCE_COMPRESS_HEAD_CHUNKS` / `LUCE_COMPRESS_TAIL_CHUNKS` | Head/tail chunks kept uncompressed. |
| `LUCE_COMPRESS_QUERY_TOKENS` | Query tokens considered for compression. |
| `LUCE_COMPRESS_REPEAT_CHUNKS` / `LUCE_COMPRESS_REPEAT_MIN` / `LUCE_COMPRESS_REPEAT_MAX` | Repeat-chunk detection bounds. |
| `LUCE_COMPRESS_POOL_KERNEL` | Pooling kernel for compression. |

## Generation / thinking control

| Variable | Purpose |
|---|---|
| `LUCE_THINK_MAX` | Max thinking tokens. |
| `LUCE_DEGENERATE_RUN_TOKENS` | Degenerate-run token threshold. |
| `LUCE_STALL_TOOL_PREFIX` | Tool-call stall prefix handling. |
| `LUCE_MIN_TOKENS` | Minimum generated tokens. |
| `LUCE_BUDGET` | Container entrypoint `--ddtree-budget` (default 22). |
| `LUCE_ANTHROPIC_RAW_SYSTEM` / `LUCE_ANTHROPIC_RAW_USER` | Pass raw system/user content on the Anthropic-compatible path. |

## Profiling / debug instrumentation

| Variable | Purpose |
|---|---|
| `LUCE_PROF` | 🐛 **debug** Comma list of profilers (`step,verify,prefill`). |
| `LUCE_TQ3_VERIFY` | See Verification (also a debug quant path). |
| `LUCE_LM_HEAD_FIX` | ⚠️ **removal candidate** LM-head correctness fix toggle. |
| `LUCE_TESTS` | 🧪 **test/bench** Enable test-only code paths (build). |

## Benchmark / harness

| Variable | Purpose |
|---|---|
| `LUCE_BENCH_MIX` | 🧪 **test/bench** Benchmark workload mix. |
| `LUCE_BENCH_SEED` | 🧪 **test/bench** Benchmark RNG seed. |
| `LUCE_CHUNK` | 🧪 **test/bench** Generic chunk-size knob (bench/scripts). |
| `LUCE_HAS_CURL` | 🧪 **test/bench** Whether curl is available (scripts). |
| `LUCE_REQUIRED_ENV` | 🧪 **test/bench** Required-env assertion list (scripts). |
| `LUCE_SERVER_VERSION` | 🧪 **test/bench** Reported server version (scripts). |

---

### Regenerating

Runtime C/C++ variables can be re-listed with:

```sh
grep -rE 'getenv\("LUCE_[A-Z0-9_]*"\)' server/src
```

See `server/docs/ENVIRONMENT.md` for the canonical generated inventory and the
policy on promoting env vars to CLI flags.
