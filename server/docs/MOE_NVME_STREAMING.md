# Routed-MoE NVMe Streaming

This is an inference-engine feature. It streams existing expert-weight bytes
from SSD without changing their format or numerical representation.

## Lucebox data path

The three owners have distinct jobs:

1. **R9700** owns dense layers and the statically hot routed experts.
2. **Strix Halo** owns an adaptive warm-expert cache, the bounded SSD staging
   buffers, and execution of streamed cold experts. Its direct path to system
   memory is faster than staging those experts into the discrete GPU on the
   qualified machine.
3. **NVMe** is the capacity tier for true cache misses that do not fit in the
   Strix safe-memory budget.

`LayerExpertRegions` is the model-adapter contract. It describes the exact
GGUF byte ranges for each layer's gate/up/down tensors (or fused gate+up).
The scheduler therefore has no model names, tensor names, expert dimensions,
or quantization-format assumptions. Each tensor range can also select a model
shard; single-file models use shard zero. See [KIMI_K3_HETERO.md](KIMI_K3_HETERO.md)
for the 14-shard Kimi K3 qualification.

## Scheduler

`MoeNvmeScheduler` provides a bounded asynchronous data plane:

- Linux `io_uring` with a registered model file and fixed, page-locked host
  buffers; a portable `pread` worker-pool and mmap fallback remain available.
- Optional `O_DIRECT`, selected automatically when the model is larger than
  75% of physical RAM, avoids retaining both a model-sized page cache and the
  explicit expert cache.
- Exact per-expert reads. Direct-I/O requests use aligned envelopes while GPU
  copies contain only the logical tensor payload.
- Demand requests outrank speculation. A fixed demand-slot reserve prevents
  prefetches from occupying the entire cache.
- Duplicate in-flight requests are merged and speculative requests can be
  upgraded to demand without issuing a second read.
- A small LFRU-style resident cache protects demand-loaded experts. A
  move-only lease prevents eviction until the asynchronous host-to-device
  copy has completed.
- Each expert becomes available at its final completion event. It does not
  wait behind the rest of an `io_uring` batch, so SSD read N+1 overlaps the
  upload and execution of expert N.
- Two or more rotating GPU slots separate the expert being computed from the
  expert being uploaded.
- Otherwise-unused Strix memory becomes a contiguous model-neutral expert
  cache indexed by `(layer, expert)`. Cache hits issue neither SSD reads nor
  host-to-device copies. LFRU replacement cannot evict a pending upload or an
  expert currently executing.

The native model router remains authoritative. Prediction may only issue a
bounded prefetch; a wrong prediction cannot change model output.

## DeepSeek V4 Flash activation

The existing heterogeneous mode is required. In `auto` mode, SSD streaming is
enabled only when the cold expert stack exceeds current Strix free memory
minus the larger of 2 GiB or 5% of device memory.

```bash
export DFLASH_DS4_MOE_TP=1
export DFLASH_DS4_MOE_TP_INPROC=1
export DFLASH_DS4_MOE_TP_GPU=1
export DFLASH_EXPERT_BUDGET_MB=11700
export DFLASH_MOE_NVME_COLD_TIER=auto

./build-hip-dual/dflash_server /path/to/model.gguf \
  --target-device hip:0 --peer-access
```

`DFLASH_MOE_NVME_COLD_TIER=on` forces the capacity tier for qualification;
`off` requires the old resident-cold path. In `auto`, a model that exceeds
Strix receives all currently usable memory (after reserve) as its adaptive
expert-cache budget, and only the remainder spills to SSD.

## Tuning and diagnostics

Defaults are intentionally small: eight pinned host slots, four fallback I/O
threads, two reserved demand slots, and two GPU slots. More queue depth did
not improve the qualified P310 drive and consumes extra pinned/system memory.

| Variable | Default | Meaning |
|---|---:|---|
| `DFLASH_MOE_NVME_BACKEND` | `auto` | `auto`, `uring`, `pread`, or `mmap` |
| `DFLASH_MOE_NVME_DIRECT` | `auto` | `auto`, `on`, or `off` |
| `DFLASH_MOE_NVME_SLOTS` | `8` | Fixed pinned host slots |
| `DFLASH_MOE_NVME_IO_THREADS` | `4` | Portable pread workers |
| `DFLASH_MOE_NVME_DEMAND_RESERVE` | `2` | Slots unavailable to speculation |
| `DFLASH_MOE_NVME_PREFETCH_BATCH` | `2` | Maximum speculative jobs per ring submission |
| `DFLASH_MOE_NVME_DEVICE_SLOTS` | `2` | Minimum rotating GPU expert buffers |
| `DFLASH_MOE_NVME_DEVICE_CACHE_MB` | automatic/`0` | Override adaptive Strix expert-cache memory; `0` leaves only pipeline slots |

Shutdown telemetry reports logical and physical bytes, measured read service
rate, cache hits, demand wait, de-duplication, dropped speculation, and errors.
The standalone targets `test_moe_nvme_scheduler`, `bench_moe_nvme_io`, and
`bench_moe_nvme_pipeline` test correctness, raw storage, and the complete
SSD-to-GPU path respectively. Benchmarks are read-only.

## Qualification result (2026-07-30)

On the Lucebox P310 and a realistic 24 MiB expert working set:

| Path | Direct-I/O throughput |
|---|---:|
| raw SSD scheduler | 4.202 GiB/s |
| SSD to R9700 pipeline | 2.940 GiB/s |
| SSD to Strix pipeline | **4.275 GiB/s** |

The Strix path completed 182.39 experts/s with p50 42.279 ms and p95 44.069
ms per eight-expert group and zero I/O errors. The early-completion pipeline
raised the R9700 result from 1.887 to 2.940 GiB/s. These are storage-pipeline
measurements, not end-to-end token rates.

The adaptive-cache qualification used a 1.584 GiB Strix cache with 64 experts
of 24 MiB each across four rounds. It produced exactly 64 cold misses followed
by 192 GPU hits (75% hit rate), raised effective service to 723.9 expert
accesses/s, and issued only 1.5 GiB of SSD traffic instead of 6 GiB. A separate
16-slot stress case completed 112 evictions at 4.431 GiB/s active I/O with zero
errors.

With the real 95.3 GiB DeepSeek V4 Flash ROCmFP2 model, an 8 GiB Strix cache
and every HTTP/disk prefix cache disabled, the first five-token prompt loaded
6.068 GiB of unique cold-expert data in 2.406 seconds. The identical second
prompt issued no additional SSD reads and completed in 0.842 seconds: a 2.86x
warm-request improvement with identical output. Final counters were 731 cache
misses, 1,463 Strix hits, and zero I/O errors.

## Research lineage and next optimization

The bounded priority/cache design follows the lessons of MoE-Infinity and
HOBBIT; exact per-prefill loading and frequency/recency admission follow
FlashMoE. SpecPrefetch motivates a future shared next-layer transfer predictor
that leaves the native router untouched. MoE-SpAc motivates compile-time
expert layout and I/O coalescing. Tutti's slack-aware `io_uring` scheduling is
relevant when persistent KV traffic shares the device.

The Kimi qualification now has a persistent reusable expert graph and proves
that its small expert math can hide under SSD service. The production DS4 path
still constructs a graph per streamed expert. Its next implementation step is
to move that qualification design into the common evaluator, then overlap
hot-owner work with cold-owner streaming. Any learned predictor comes after
that deterministic path is qualified.
