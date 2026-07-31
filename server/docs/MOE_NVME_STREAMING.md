# Routed-MoE NVMe Streaming

This is an inference-engine feature. It streams existing expert-weight bytes
from SSD without changing their format or numerical representation.

## Deployment shapes

On a full Lucebox, the three owners have distinct jobs:

1. **R9700** is the primary compute owner. It runs dense layers and the
   profile-selected hot routed branch from its own SSD-backed expert cache.
2. **Strix Halo** is the capacity owner. It runs the remaining routed branch
   from a separate adaptive cache and bounded SSD staging pipeline. Its direct
   path to system memory is faster than staging every cold expert into the
   discrete GPU on the qualified machine.
3. **NVMe** is the capacity tier for true cache misses that do not fit in the
   two safe-memory budgets.

The two routed branches launch concurrently. Each native route has exactly one
owner, and their activation-sized partials are added after both complete, so
the routed-layer target is `max(R9700 branch, Strix branch)` rather than their
sum. A persistent secondary worker avoids thread creation in the layer loop.
This route-owner split does not itself solve placement of a non-routed core
larger than R9700 VRAM; that remains an independent layer/tensor-plan decision.

On a Strix Halo-only machine, the same device owns dense/static-hot weights,
the warm cache, and streamed-expert execution. The planner reserves KV and
runtime headroom before assigning otherwise-unused UMA to the cache. Nothing
in the scheduler assumes an R9700 or peer access.

## Model integration contract

A model adapter supplies three independent descriptions:

1. `LayerExpertRegions` describes physical bytes and model shards. Ordinary
   GGUF uses tensor-major gate/up/down regions. An optional expert-major record
   stores all components of one expert contiguously, reducing three reads to
   one without changing weight values. Component offsets must respect the
   target backend's tensor alignment.
2. `MoeStreamExpertSpec` describes dimensions, tensor types, scales, and the
   gated activation. It supports separate or fused gate/up, different routed
   input/output widths, SwiGLU, clamped SwiGLU, and SiTU.
3. `MoeStreamRouteBatch` carries the native router IDs, weights, and F32 input
   activations. It contains no architecture-specific tensor names.

`eval_moe_streamed_experts` validates the byte layout against the numerical
specification before compute. It then uses a bounded cache of persistent graphs
keyed by the full specification and active token width. New MoE families only
need an adapter that fills these descriptors; the storage scheduler, cache,
and compute pipeline remain unchanged. Each file range can select a different
shard, while single-file models use shard zero. See
[KIMI_K3_HETERO.md](KIMI_K3_HETERO.md) for the 14-shard Kimi K3 qualification.

`MoeStreamDualOwnerExecutor` composes two such engines without adding a model
router. `MoeHybridPlacement` can select the primary owner's experts per layer;
otherwise a deterministic hash is only a bring-up fallback. Duplicate routes
to one expert retain one owner, and a partition test verifies that the two
weight masks reconstruct the original route batch exactly.

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
- Otherwise-unused device memory becomes a contiguous model-neutral expert
  cache indexed by `(layer, expert)`. Cache hits issue neither SSD reads nor
  host-to-device copies. LFRU replacement cannot evict a pending upload or an
  expert currently executing.
- Persistent compute graphs remove graph construction and activation-buffer
  allocation from the steady-state expert loop. A bounded LRU supports models
  whose layers use more than one expert shape or quantization format.

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

### Strix Halo only

Do not enable the MoE-TP variables. A single-device Strix machine normally
exposes its GPU as `hip:0`:

```bash
unset DFLASH_DS4_MOE_TP DFLASH_DS4_MOE_TP_INPROC DFLASH_DS4_MOE_TP_GPU
export DFLASH_MOE_NVME_COLD_TIER=on

./build-hip/dflash_server /path/to/model.gguf \
  --target-device hip:0 --max-ctx 8192
```

`on` keeps at least one expert per layer in the SSD tier even if the model
would otherwise be fully resident, making the path directly testable. For a
model that genuinely exceeds UMA, `auto` chooses the partial placement itself.
`DFLASH_EXPERT_BUDGET_MB` can additionally cap static routed-weight residency;
`DFLASH_MOE_NVME_DEVICE_CACHE_MB` can override the adaptive cache budget.

## Kimi K3 activation

Pass the first file of a standard split GGUF and select the Strix device.
Kimi's routed gate/up/down stacks remain file-backed by default; the loader
allocates only dense, shared, routing, latent-projection, and cache tensors.

```bash
export DFLASH_MOE_NVME_BACKEND=auto

./build-hip/dflash_server \
  /path/to/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf \
  --target-device hip:0 --max-ctx 8192
```

No DeepSeek MoE-TP variables are required. Cache memory is chosen only after
the resident model and recurrent/KV state have been allocated. The automatic
budget reserves the larger of 2 GiB or 5% of device memory and never exceeds
the complete routed pool. `DFLASH_MOE_NVME_DEVICE_CACHE_MB` remains an explicit
override, also capped by the routed pool.

Kimi's native router remains authoritative. The current text backend is
correctness-first and sequential; captured per-layer graphs and the vision
tower are separate optimizations.

On a two-GPU Lucebox, put the compute-intensive primary path on the R9700 and
use Strix as the secondary capacity owner. Both devices receive independent
SSD slots/caches and execute their selected routed experts concurrently:

```bash
export DFLASH_MOE_TP_GPU=<strix-gpu>

./build-hip-dual/dflash_server \
  /path/to/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf \
  --target-device hip:<r9700-gpu> --max-ctx 8192
```

This is functional expert ownership, not a contiguous layer split, so do not
use `--target-devices`. `DFLASH_MOE_PLACEMENT` may point at an offline
`MoeHybridPlacement` JSON; its hot expert IDs become R9700-owned and
all other selected routes become Strix-owned. Without a plan,
`DFLASH_MOE_PRIMARY_SHARE_PER_MILLE` controls a deterministic bring-up split
(default 500). Leaving `DFLASH_MOE_TP_GPU` unset preserves the single-device
path, including Strix-only systems. The current Kimi join still uses
activation-sized host staging; a device-resident peer join remains the next
throughput optimization.

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
| `DFLASH_MOE_NVME_DEMAND_TIMEOUT_MS` | `30000` | Maximum wait for a demanded expert; `0` disables the guard |
| `DFLASH_MOE_NVME_DEVICE_SLOTS` | `2` | Minimum rotating GPU expert buffers |
| `DFLASH_MOE_NVME_DEVICE_CACHE_MB` | automatic/`0` | Override adaptive device expert-cache memory; `0` leaves only pipeline slots |
| `DFLASH_MOE_NVME_GRAPH_CACHE` | `8` | Persistent expert-graph variants retained per stream engine; `0` is a diagnostic no-cache mode |
| `DFLASH_MOE_NVME_REFERENCE_EVAL` | unset | Diagnostic only: `1` restores the allocation-heavy reference evaluator for numerical/performance A/B |
| `DFLASH_MOE_TP_GPU` | primary GPU | Optional secondary GPU; enables concurrent route ownership when different from the primary |
| `DFLASH_MOE_PLACEMENT` | unset | Offline placement JSON; listed experts belong to the primary GPU |
| `DFLASH_MOE_PRIMARY_SHARE_PER_MILLE` | `500` | Bring-up hash split used only when no placement is supplied |
| `DFLASH_MOE_DUAL_STREAM_TRACE` | unset | Debug per-layer owner counts and branch/wall timing |

Shutdown telemetry reports logical and physical bytes, measured read service
rate, cache hits, demand wait/timeouts, de-duplication, dropped speculation,
errors, and persistent-graph builds/hits/evictions. The scheduler rejects
truncated shards at bind time and accepts a valid short direct-I/O completion
only when it covers the complete logical payload at an unaligned file tail.
`test_moe_stream_compute` generates
tiny experts and checks both tensor-major and expert-major GPU results against
a CPU oracle. It defaults to GPU 0, so it runs directly on Strix-only systems;
`DFLASH_TEST_GPU` selects another device on multi-GPU hosts. The standalone
targets `test_moe_nvme_scheduler`, `bench_moe_nvme_io`, and
`bench_moe_nvme_pipeline` test scheduling, raw storage, and the complete
SSD-to-GPU path. Benchmarks are read-only.

The external `smoke_kimi_k3_forward` target accepts
`[stream_experts=0|1] [expert_gpu=-1]`. This is an A/B and placement oracle for
small Kimi fixtures; production Kimi uses streaming and `-1` resolves the
environment/default owner.

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

The single-Strix production path was also qualified with no MoE-TP variables,
a 12 GiB static-expert cap, a 255 MiB device cache, and 79.95 GiB left in the
SSD tier. Two identical seven-token prompts returned identical output in 3.762
and 3.503 seconds of prefill. Across both requests the engine moved 25.367 GiB
at 4.350 GiB/s active I/O, reported zero errors, and used one graph build for
3,056 launches (3,055 graph-cache hits). The allocation-heavy reference path
took 3.816 and 3.599 seconds on the same run shape, so persistence improved
this deliberately cold, storage-bound case by 1.4-2.7%; its larger value is
removing thousands of allocations when more of the route set is warm.

The native Kimi path was qualified with a real two-shard 0.40B MXFP4
architecture fixture. Resident and streamed execution produced the same eight
greedy output tokens. The `io_uring` run completed 168 selected-expert
launches with one graph build, 167 graph-cache hits, and zero I/O errors. A
1 MiB device cache forced 163 evictions, demonstrating that the result came
through the split-GGUF SSD path rather than accidental full residency.

The dual-owner extension was then qualified on the same fixture with the
R9700 primary and Strix secondary. Stable route ownership exercised both
engines and both issued real `io_uring` traffic; some top-2 layers naturally
landed on only one owner and bypassed the rendezvous. The eight greedy token
IDs matched the single-R9700 oracle exactly. After warm-up, dual-branch
routed-layer wall time matched the slower branch to within a few microseconds.
The tiny experts do not provide a meaningful end-to-end speed claim; full-size
expert geometry and a quiescent machine are required for that comparison.

## Research lineage and next optimization

The bounded priority/cache design follows the lessons of MoE-Infinity and
HOBBIT; exact per-prefill loading and frequency/recency admission follow
FlashMoE. SpecPrefetch motivates a future shared next-layer transfer predictor
that leaves the native router untouched. MoE-SpAc motivates compile-time
expert layout and I/O coalescing. Tutti's slack-aware `io_uring` scheduling is
relevant when persistent KV traffic shares the device.

The persistent graph, model-neutral evaluator, and exact concurrent owner
split are now shared infrastructure. The next high-value work is a repacked
expert-major artifact and a device-resident fork/join that removes Kimi's host
activation boundary. Any learned predictor comes after that deterministic path
is qualified, and may only prefetch routes selected later by the native router.
