# Environment Variables Reference

Summary of `DFLASH_*` / `DFLASH27B_*` environment variables recognized across the
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
| `DFLASH_HOST` | Server bind host. |
| `DFLASH_PORT` | Server bind port. |
| `DFLASH_BIN` / `DFLASH_SERVER_BIN` | Path to the server binary (harness/scripts). |
| `DFLASH_BIN_AR` | Alternate/AR binary path for benchmarks. |
| `DFLASH_DIR` | Base working directory. |
| `DFLASH_SHARE_DIR` | Static/share asset directory served by the HTTP server. |
| `DFLASH_MODEL_CARDS_DIR` | Directory of model-card definitions. |
| `DFLASH_MODEL_NAME` | Model name/identifier. |
| `DFLASH_TOKENIZER` | Tokenizer path/identifier. |
| `DFLASH_TARGET` | Target model path/spec. |
| `DFLASH_DRAFT` | Draft model path/spec. |
| `DFLASH_IMAGE_INFO_PATH` | Path to image/build info metadata. |
| `DFLASH_MAX_CONTEXT` / `DFLASH_MAX_CTX` | Maximum context length. |
| `DFLASH_DEFAULT_MAX_TOKENS` | Default generation token cap. |
| `DFLASH_IGNORE_EOS` | Ignore EOS token during generation. |
| `DFLASH_LAZY` | Lazy model/weight loading. |
| `DFLASH_VERBOSE` | 🐛 **debug** Verbose logging. |

## GPU / backend placement

| Variable | Purpose |
|---|---|
| `DFLASH_TARGET_GPU` / `DFLASH_TARGET_GPUS` | GPU(s) assigned to the target model. |
| `DFLASH_TARGET_LAYER_SPLIT` | Layer-split placement across GPUs for the target. |
| `DFLASH_DRAFT_GPU` | GPU assigned to the drafter. |
| `DFLASH_CUDA_ARCHES` / `DFLASH_HIP_ARCHES` | CUDA/HIP architecture targets (build). |
| `DFLASH27B_GPU_BACKEND` / `DFLASH27B_BACKEND_CUDA` / `DFLASH27B_BACKEND_HIP` | Backend selection (build/compile-time). |
| `DFLASH_HIP_NO_AUTO_UMA` | Disable automatic HIP UMA (unified memory) selection. |
| `DFLASH_HIP_UMA_MIN_FRAC` | Minimum VRAM fraction before HIP UMA kicks in. |
| `DFLASH_WAVE_SIZE` | HIP wave size (compile flag, e.g. gfx1151 needs 32). |

## Speculative decoding — drafter

| Variable | Purpose |
|---|---|
| `DFLASH_DRAFT_KV` | 🔀 **kill-switch** `=0` restores per-step drafter window recompute instead of the ring cache. |
| `DFLASH_DRAFT_PERSIST` | Persist drafter state across steps. |
| `DFLASH_DISABLE_DRAFT_ATTN` | 🐛 **debug** Disable drafter attention block (ablation). |
| `DFLASH_DISABLE_DRAFT_ATTN_GATE` | 🐛 **debug** Disable drafter attention gate (ablation). |
| `DFLASH_DISABLE_DRAFT_AUX_NORMS` | 🐛 **debug** Disable auxiliary norms in the drafter (ablation). |
| `DFLASH_DISABLE_DRAFT_FFN` | 🐛 **debug** Disable drafter FFN block (ablation). |
| `DFLASH_DISABLE_DRAFT_SWA` | 🐛 **debug** Disable drafter sliding-window attention (ablation). |
| `DFLASH_DOMINO_ZERO_START` | Domino head zero-start behavior. |
| `DFLASH27B_DRAFT_FP16` | Load drafter in FP16. |
| `DFLASH27B_DRAFT_SWA` | Enable drafter SWA. |
| `DFLASH27B_DRAFT_CTX_MAX` | Drafter max context. |
| `DFLASH27B_DRAFT_BLOCK_SIZE` / `DFLASH27B_DRAFT_LAYERS` / `DFLASH27B_DRAFT_N_TARGET_LAYERS` / `DFLASH27B_DRAFT_MASK_TOKEN_ID` | Drafter geometry (build/config). |

## Draft IPC transport

| Variable | Purpose |
|---|---|
| `DFLASH_DRAFT_IPC_TRANSPORT` | IPC transport for the draft process. |
| `DFLASH_DRAFT_IPC_SHARED_BYTES` | Shared-memory size for draft IPC. |
| `DFLASH_DRAFT_IPC_RING_CAP` | Ring buffer capacity for draft IPC. |
| `DFLASH_DRAFT_IPC_BIN` | Draft IPC daemon binary. |
| `DFLASH_DRAFT_IPC_GPU` | GPU for the draft IPC daemon. |
| `DFLASH_DRAFT_IPC_WORK_DIR` | Working directory for draft IPC. |

## Verification / sampling

| Variable | Purpose |
|---|---|
| `DFLASH_SAMPLED_VERIFY` | Use sampled verification. |
| `DFLASH_VERIFY_WIDTH` | Verify batch width. |
| `DFLASH_GPU_SAMPLE` | GPU sampling path. |
| `DFLASH_GPU_ARGMAX` / `DFLASH_GPU_VERIFY_ARGMAX` | GPU argmax for sampling/verification. |
| `DFLASH_GPU_DRAFT_TOPK` | GPU draft top-k. |
| `DFLASH_TQ3_VERIFY` | TQ3-quantized verify path. |
| `DFLASH_N_SAMPLE` / `DFLASH_SAMP` | 🧪 **test/bench** Sample count/mode. |
| `DFLASH_SAMPLER_BENCH` | 🧪 **test/bench** Sampler benchmark mode. |
| `DFLASH_SV_DEBUG` | 🐛 **debug** Sampled-verify debug output. |

## Adaptive experts / adaptive verify width

| Variable | Purpose |
|---|---|
| `DFLASH_ADAPTIVE_K_TAU` | Cumulative combine-weight threshold for per-token expert gating (prefer `--adaptive-experts`). |
| `DFLASH_ADAPTIVE_K_DENSE` | CSV of MoE layers kept dense under adaptive-K. |
| `DFLASH_ADAPTIVE_WIDTH_MIN` | Minimum adaptive verify width. |
| `DFLASH_ADAPTIVE_WIDTH_THETA` | Threshold controlling adaptive verify width. |
| `DFLASH_HYBRID_HOT_PCT` | Hot-expert percentage for hybrid MoE. |

## KVFlash (KV cache pager)

| Variable | Purpose |
|---|---|
| `DFLASH_KVFLASH` | Enable KVFlash (prefer CLI `--kvflash`; token count or `auto`). |
| `DFLASH_KVFLASH_DRAFTER` | KVFlash for the drafter cache. |
| `DFLASH_KVFLASH_MAX_POOL` | Max KVFlash pool size. |
| `DFLASH_KVFLASH_POLICY` | KVFlash eviction/placement policy. |
| `DFLASH_KVFLASH_TAU` | KVFlash tau threshold. |
| `DFLASH_LAGUNA_SWA_RING` | 🔀 **kill-switch** `=0` keeps SWA layers on pool-sized caches under KVFlash. |

## KV cache quantization / dtype

| Variable | Purpose |
|---|---|
| `DFLASH_CACHE_TYPE_K` / `DFLASH_CACHE_TYPE_V` | KV cache K/V dtype. |
| `DFLASH_KV_TYPE` | KV cache type selector. |
| `DFLASH_FEATURE_DTYPE` | Draft feature-ring dtype. |
| `DFLASH27B_KV_F16` | F16 KV cache. |
| `DFLASH27B_KV_K` / `DFLASH27B_KV_V` | Per-side (K/V) KV quantization. |
| `DFLASH27B_KV_Q4` / `DFLASH27B_KV_TQ3` / `DFLASH27B_KV_TBQ` | Quantized KV formats (Q4 / TQ3 / TBQ). |
| `DFLASH_PFLASH_K_TYPE` | PFlash K dtype. |

## FlashPrefill / prefill

| Variable | Purpose |
|---|---|
| `DFLASH_FP_USE_BSA` | Use block-sparse attention in flash prefill. |
| `DFLASH_FP_ALPHA` | FlashPrefill alpha parameter. |
| `DFLASH_FP_CHUNK_S` | FlashPrefill chunk size. |
| `DFLASH_FP_NOPE_TAIL` | NoPE tail handling in flash prefill. |
| `DFLASH_FP_HIP_ROW` | HIP row-kernel path for flash prefill. |
| `DFLASH_FP_SKIP_PREWARM` | Skip flash-prefill prewarm. |
| `DFLASH_FP_PROFILE` / `DFLASH_FP_DUMP_COUNTS` / `DFLASH_FP_DEBUG_LAYER0` | 🐛 **debug** FlashPrefill profiling/debug. |
| `DFLASH_PREFILL_MODE` | Prefill mode selector. |
| `DFLASH_PREFILL_THRESHOLD` | Prefill length threshold. |
| `DFLASH_PREFILL_DRAFTER` | Drafter participation during prefill. |
| `DFLASH_PREFILL_KEEP` | Keep prefill cache across requests. |
| `DFLASH_PREFILL_CACHE_SLOTS` / `DFLASH_PREFIX_CACHE_SLOTS` | Prefill/prefix cache slot count. |
| `DFLASH_PREFILL_CACHE_TEST_LOG` / `DFLASH_PREFILL_CACHE_TEST_PORT` | 🧪 **test/bench** Prefill-cache test harness. |
| `DFLASH27B_LAYER_PREFILL` / `DFLASH27B_PREFILL_UBATCH` | Layer-split prefill / prefill micro-batch. |
| `DFLASH27B_CHUNKED` / `DFLASH27B_CHUNKED_CHUNK` / `DFLASH27B_CHUNKED_Q_BATCH` / `DFLASH27B_CHUNKED_THRESHOLD` | Chunked prefill controls. |
| `DFLASH27B_LAGUNA_CHUNK` | Laguna prefill chunk size. |
| `DFLASH27B_FA_WINDOW` / `DFLASH_FA_WINDOW` | Flash-attention window size. |

## Laguna backend

| Variable | Purpose |
|---|---|
| `DFLASH_LAGUNA_PROFILE` / `DFLASH_LAGUNA_TELEMETRY` | 🐛 **debug** Profiling / telemetry. |
| `DFLASH_LAGUNA_AUTO_HEAD_MAJOR` / `DFLASH_LAGUNA_KV_HEAD_MAJOR` | Head-major KV layout. |
| `DFLASH_LAGUNA_CACHE_SLOTS` | Cache slot count. |
| `DFLASH_LAGUNA_DRAFT_PAD` | Drafter padding. |
| `DFLASH_LAGUNA_DSPARK` / `DFLASH_LAGUNA_DSPARK_TREE` / `DFLASH_LAGUNA_DSPARK_CONFIDENCE_THRESHOLD` | DSpark speculative controls. |
| `DFLASH_LAGUNA_EXPERT_CACHE` | Expert cache toggle. |
| `DFLASH_LAGUNA_FUSED_DOMINO` / `DFLASH_LAGUNA_FUSED_DSPARK` / `DFLASH_LAGUNA_FUSED_QK` / `DFLASH_LAGUNA_FUSE_FFN` / `DFLASH_LAGUNA_MOE_FUSED_COMBINE` | Kernel fusion toggles. |
| `DFLASH_LAGUNA_GPU_ARGMAX` / `DFLASH_LAGUNA_GPU_REMAP` | GPU argmax / expert remap. |
| `DFLASH_LAGUNA_HOTNESS` | Expert hotness tracking. |
| `DFLASH_LAGUNA_LAYER_SPLIT_UBATCH` | Layer-split micro-batch. |
| `DFLASH_LAGUNA_MOE_STUB` | 🐛 **debug** Stub MoE (ablation). |
| `DFLASH_LAGUNA_NEXT_PLACEMENT_OUT` | 🐛 **debug** Dump next-placement plan. |
| `DFLASH_LAGUNA_NO_KVPAD` / `DFLASH_LAGUNA_PAD_CPY` | KV padding controls. |
| `DFLASH_LAGUNA_NO_SINGLE_GRAPH` | Disable single-graph capture. |
| `DFLASH_LAGUNA_PERSIST_VERIFY` | Persist verify graph. |
| `DFLASH_LAGUNA_PREGATE_MAX` / `DFLASH_LAGUNA_PREGATE_TRACE` | Pre-gating max; 🐛 **debug** trace. |
| `DFLASH_LAGUNA_SWAP_MAX` / `DFLASH_LAGUNA_SWAP_MIN_GAIN` | Expert-swap thresholds. |
| `DFLASH_LAGUNA_VERIFY_WIDTH` / `DFLASH_LAGUNA_VERIFY_WIDTH_MAX` | Verify width limits. |
| `DFLASH_LAGUNA_BENCH_NO_LOGITS` | 🧪 **test/bench** Skip logits in benchmarks. |

## Qwen3.5 MoE backend

| Variable | Purpose |
|---|---|
| `DFLASH_QWEN35MOE_CACHE_SLOTS` | Cache slot count. |
| `DFLASH_QWEN35MOE_HOTNESS` | Expert hotness tracking. |
| `DFLASH_QWEN35MOE_SWAP_MAX` / `DFLASH_QWEN35MOE_SWAP_MIN_GAIN` | Expert-swap thresholds. |
| `DFLASH_QWEN35MOE_TELEMETRY` | 🐛 **debug** Telemetry. |
| `DFLASH_QWEN35MOE_NEXT_PLACEMENT_OUT` / `DFLASH_QWEN35MOE_RUNTIME_STATS_OUT` | 🐛 **debug** Dump placement / runtime stats. |
| `DFLASH_QWEN35MOE_NO_KVPAD` / `DFLASH_QWEN35_NO_KVPAD` | KV padding controls. |
| `DFLASH_QWEN35MOE_NO_ROUTED` | 🐛 **debug** Disable routed experts (ablation). |
| `DFLASH_QWEN35MOE_PREFILL_CHUNK` | Prefill chunk size. |
| `DFLASH_QWEN35MOE_HYBRID_SPEC_MIN_ACCEPT_RATE` / `DFLASH_QWEN35MOE_HYBRID_SPEC_MIN_STEPS_BEFORE_AR` | Hybrid speculative acceptance thresholds. |

## Gemma4 backend

| Variable | Purpose |
|---|---|
| `DFLASH_GEMMA4_LAYER_SPLIT_UBATCH` | Layer-split micro-batch. |
| `DFLASH_GEMMA4_NO_KVPAD` | Disable KV padding. |
| `DFLASH_G4_BSA_CHUNK` | Block-sparse attention chunk size. |

## DeepSeek4 (DS4) backend

| Variable | Purpose |
|---|---|
| `DFLASH_DS4_TIMING` | 🐛 **debug** DS4 timing instrumentation. |
| `DFLASH_DS4_CUDA_LAYERS` | Number of DS4 layers on CUDA. |
| `DFLASH_MOE_COLD_BACKEND` | Cold-expert compute backend. |
| `DFLASH_NO_PREAD` | Disable pread-based weight loading. |

## MoE expert compute / IPC

| Variable | Purpose |
|---|---|
| `DFLASH_MOE_EXPERT_COMPUTE_THREADS` / `DFLASH_COLD_THREADS` | CPU threads for expert compute. |
| `DFLASH_MOE_EXPERT_COMPUTE_BATCH` / `DFLASH_MOE_EXPERT_COMPUTE_BATCH_MAX` | Expert compute batch sizing. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE` | Expert-compute IPC mode. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_TRANSPORT` | IPC transport. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_SHARED_BYTES` | Shared-memory size. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY` | IPC batch capacity. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_DTYPE` | IPC payload dtype. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_PROFILE` | 🐛 **debug** IPC profiling. |
| `DFLASH_MOE_EXPERT_COMPUTE_IPC_BIN` / `DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU` / `DFLASH_MOE_EXPERT_COMPUTE_IPC_WORK_DIR` / `DFLASH_MOE_EXPERT_COMPUTE_IPC_REQUIRED` | IPC daemon binary / GPU / work dir / required flag. |
| `DFLASH_MOE_FIXED_SLOT_GRAPHS` / `DFLASH_MOE_FIXED_SLOT_MAX` | Fixed-slot MoE graph controls. |
| `DFLASH_MOE_PREFILL_HOT_SUB_BATCH` | Hot-expert prefill sub-batch. |
| `DFLASH_NO_MOE_ROUTER_FUSE` / `DFLASH_NO_MOE_SWIGLU_FUSE` | Disable router / SwiGLU fusion. |
| `DFLASH_EXPERT_BUDGET_MB` / `DFLASH_EXPERT_BUDGET_PCT` | Expert VRAM budget (absolute / percent). |
| `DFLASH_DROP_COLD` | Drop cold experts. |
| `DFLASH_COLLECT_ROUTING` | 🐛 **debug** Collect routing statistics. |

## Target-shard IPC

| Variable | Purpose |
|---|---|
| `DFLASH_TARGET_SHARD_IPC_TRANSPORT` | Transport for target-shard IPC. |
| `DFLASH_TARGET_SHARD_IPC_SHARED_BYTES` | Shared-memory size for target-shard IPC. |

## Matmul / MMID / MMVQ kernels

| Variable | Purpose |
|---|---|
| `DFLASH_MMID_GROUPED` | Grouped `MUL_MAT_ID` kernel for small verify batches. |
| `DFLASH_MMID_GROUPED_TYPES` | Types eligible for the grouped MMID kernel. |
| `DFLASH_MMQ_FULL_BATCH_MIN` / `DFLASH_MMQ_SUB_BATCH` | MMQ batch thresholds. |
| `DFLASH_CUDA_MMVQ_TOKENWISE` / `DFLASH_CUDA_MMVQ_MOE_TOKENWISE` / `DFLASH_CUDA_MMVQ_MOE_KERNEL` | MMVQ token-wise / MoE kernel selection. |
| `DFLASH_GDN_FORCE_GROUPED_COLS` / `DFLASH_GDN_NO_GROUPED_COLS` | Gated-delta-net grouped-column control. |
| `DFLASH_NO_MASK` | 🐛 **debug** Disable attention masking (ablation). |

## Top-k kernels

| Variable | Purpose |
|---|---|
| `DFLASH_TOPK_PROFILE` | 🐛 **debug** Top-k kernel profiling. |
| `DFLASH_TOPK_SPLIT` | Top-k split strategy. |
| `DFLASH_TOPK_CASE` / `DFLASH_TOPK_CONSUME` / `DFLASH_TOPK_LAUNCH` | Top-k kernel case/consume/launch tuning. |

## Spark

| Variable | Purpose |
|---|---|
| `DFLASH_SPARK` | Enable Spark. |
| `DFLASH_SPARK_VRAM_MB` | Spark VRAM budget. |
| `DFLASH_SPARK_CLAUDE_DIR` / `DFLASH_SPARK_CODEX_DIR` | Spark corpus directories. |

## KV / context compression

| Variable | Purpose |
|---|---|
| `DFLASH_COMPRESS_NO_PARK` | Disable parking of compressed blocks. |
| `DFLASH_COMPRESS_ANCHOR_RADIUS` / `DFLASH_COMPRESS_MAX_ANCHOR_HITS` | Anchor radius / max hits. |
| `DFLASH_COMPRESS_HEAD_CHUNKS` / `DFLASH_COMPRESS_TAIL_CHUNKS` | Head/tail chunks kept uncompressed. |
| `DFLASH_COMPRESS_QUERY_TOKENS` | Query tokens considered for compression. |
| `DFLASH_COMPRESS_REPEAT_CHUNKS` / `DFLASH_COMPRESS_REPEAT_MIN` / `DFLASH_COMPRESS_REPEAT_MAX` | Repeat-chunk detection bounds. |
| `DFLASH_COMPRESS_POOL_KERNEL` | Pooling kernel for compression. |

## Generation / thinking control

| Variable | Purpose |
|---|---|
| `DFLASH_THINK_MAX` | Max thinking tokens. |
| `DFLASH_THINK_SOFT_CLOSE_MIN_RATIO` | Soft-close ratio for thinking blocks. |
| `DFLASH_DEBUG_THINKING_LOGITS` | 🐛 **debug** Debug thinking logits. |
| `DFLASH_DEGENERATE_RUN_TOKENS` | Degenerate-run token threshold. |
| `DFLASH_STALL_TOOL_PREFIX` | Tool-call stall prefix handling. |
| `DFLASH_MIN_TOKENS` | Minimum generated tokens. |
| `DFLASH_BUDGET` | Token/compute budget. |
| `DFLASH_ANTHROPIC_RAW_SYSTEM` / `DFLASH_ANTHROPIC_RAW_USER` | Pass raw system/user content on the Anthropic-compatible path. |

## Profiling / debug instrumentation

| Variable | Purpose |
|---|---|
| `DFLASH_PROF` | 🐛 **debug** Comma list of profilers (`step,verify,prefill`). |
| `DFLASH_TQ3_VERIFY` | See Verification (also a debug quant path). |
| `DFLASH27B_LM_HEAD_FIX` | ⚠️ **removal candidate** LM-head correctness fix toggle. |
| `DFLASH27B_TESTS` | 🧪 **test/bench** Enable test-only code paths (build). |

## Benchmark / harness

| Variable | Purpose |
|---|---|
| `DFLASH_BENCH_MIX` | 🧪 **test/bench** Benchmark workload mix. |
| `DFLASH_BENCH_SEED` | 🧪 **test/bench** Benchmark RNG seed. |
| `DFLASH_CHUNK` | 🧪 **test/bench** Generic chunk-size knob (bench/scripts). |
| `DFLASH_HAS_CURL` | 🧪 **test/bench** Whether curl is available (scripts). |
| `DFLASH_REQUIRED_ENV` | 🧪 **test/bench** Required-env assertion list (scripts). |
| `DFLASH_SERVER_VERSION` | 🧪 **test/bench** Reported server version (scripts). |

---

### Regenerating

Runtime C/C++ variables can be re-listed with:

```sh
grep -rE 'getenv\("DFLASH[A-Z0-9_]*"\)' server/src
```

See `server/docs/ENVIRONMENT.md` for the canonical generated inventory and the
policy on promoting env vars to CLI flags.
