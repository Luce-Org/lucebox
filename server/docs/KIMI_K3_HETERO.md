# Kimi K3 on heterogeneous Lucebox

This note separates measured facts from estimates. The qualification target is
Unsloth `Kimi-K3-UD-IQ1_S`, currently the smallest published Kimi K3 GGUF: 14
files and 594 GB.

## What is implemented

Kimi K3 now uses the same model-neutral SSD data plane as DeepSeek, with a
small architecture adapter around it:

- A tensor region carries a model-shard index. `MoeNvmeScheduler` can register
  and read any number of GGUF shard file descriptors with `io_uring`, including
  an expert whose gate, up, and down tensors live in different files.
- The streamed expert dimension can differ from the model hidden dimension.
  This represents Kimi's `7168 -> 3584` routed projection without pretending
  its experts consume full-width hidden states.
- The common streamed graph supports SiTU as well as SwiGLU and DS4's clamped
  SwiGLU.
- The native Kimi text backend executes KDA, absorbed MLA, Attention Residuals,
  dense layer 0, latent routed MoE, shared experts, final projection,
  tokenization, and sampling. Its own router IDs and weights are copied across
  the activation-sized boundary into `eval_moe_streamed_experts`; the SSD
  scheduler never substitutes or predicts a contributing expert.
- The loader accepts standard split GGUFs, even though only shard 1 contains
  global metadata. It keeps routed gate/up/down stacks file-backed while
  selectively allocating all other tensors on the target device.
- Device-cache sizing happens after resident weights and recurrent/KV state
  are allocated. It reserves the larger of 2 GiB or 5% of device memory and is
  capped by the actual routed pool, so a tiny model cannot accidentally request
  a model-sized cache.
- Routed-expert ownership is independent of contiguous layer splitting.
  With a second HIP device, two common stream engines partition the exact
  native routes and run concurrently. The primary owns profile-selected hot
  experts; the secondary owns the remaining capacity routes. Each engine has
  independent SSD slots, adaptive cache, and persistent expert graphs.
- `MoeStreamDualOwnerExecutor` keeps one secondary worker alive for the model
  lifetime. This removes thread create/join from every MoE layer and makes
  routed wall time track the slower branch instead of their sum.
- `bench_kimi_k3_hetero` runs Kimi's exact IQ1_S routed-expert geometry through
  the same model-neutral persistent evaluator used by production adapters:
  896 experts, top-16, 92 MoE layers,
  `3584 -> 3072 -> 3584`, and SiTU (`beta=4`, `linear_beta=25`). It overlaps
  SSD/H2D for expert N+1 with compute for expert N.

Seven scheduler tests pass on the AMD Lucebox, including expert-major one-read
records, mmap, and real-file multi-shard reads. A separate numerical test
matches both tensor-major and expert-major streamed GPU execution against a
CPU oracle on gfx1151 and gfx1201. Existing single-file DS4 descriptors remain
source index zero and need no model-specific change.

The SSD text path is implemented, but two qualification boundaries remain:

- The backend is correctness-first and token-sequential. Its per-layer graph
  boundaries are not yet fused/captured for full-model speed.
- The heterogeneous routed branches overlap, but their partial outputs still
  cross a host-visible, activation-sized boundary at every routed layer. It
  does not yet use PR-505's device-resident peer join.

The vision encoder is out of scope for this text-only path.

## End-to-end split-GGUF qualification, 2026-07-31

The released `inference-optimization/Kimi-K3-0.40B-MXFP4` architecture fixture
was converted to two standard GGUF shards. The routed tensors remained native
MXFP4. On Strix Halo, the same prompt was run once with all experts resident
and once with all routed stacks file-backed:

```text
prompt IDs: 18805 308 799 5624 12524
output IDs: 318 57195 11 1459 387 1495 2189 261
text: According to all known laws of aviation, there is no way a
```

The token IDs matched exactly. The automatic Linux path selected `io_uring`,
read all selected experts from the correct GGUF shards, and reported 168
expert launches, one graph build, 167 graph-cache hits, and zero I/O errors.
With a deliberately tiny 1 MiB device cache it moved 0.033 GiB; this forces
evictions and proves the result is not an all-resident accident.

## Exact routed-weight demand

IQ1_S stores 50 bytes for every 256 values. All three matrices of one Kimi K3
routed expert therefore occupy:

```text
gate  = row_size(IQ1_S, 3584) * 3072 = 2,150,400 bytes
up    = row_size(IQ1_S, 3584) * 3072 = 2,150,400 bytes
down  = row_size(IQ1_S, 3072) * 3584 = 2,150,400 bytes
expert                                      = 6,451,200 bytes
```

That produces the following deployment constants:

| Quantity | Exact value |
|---|---:|
| One routed expert | 6.152344 MiB |
| Routed expert calls per decoded token | 92 x 16 = 1,472 |
| Fully cold bytes per token | 8.843994 GiB |
| Complete routed-expert pool | 495.263672 GiB |
| Approximate non-routed part of the 594 GB GGUF | 57.94 GiB |

The routed pool alone is much larger than Lucebox memory. SSD streaming is
therefore required for this published quant; ordinary CPU/GPU offload cannot
make it resident.

## Measured result, 2026-07-30

The benchmark ran read-only on the R9700 + Strix Halo Lucebox and its P310
NVMe. It used the exact Kimi byte ranges and IQ1_S+SiTU graph. An existing large
local model supplied the bytes so downloading Kimi's 594 GB was not required;
the byte values do not affect transfer volume or kernel shape.

| Owner of streamed experts | Scenario | Pipeline | Routed-core rate |
|---|---|---:|---:|
| Strix Halo | 3 tokens, balanced cold routes, common evaluator | **3.804 GiB/s** | **0.430 token/s** |
| R9700 | 1 token, balanced cold routes, compute on | 2.091 GiB/s | 0.236 token/s |
| Strix Halo | 2 tokens, 10 GiB cache, unrelated routes | 3.638 GiB/s, 0.82% hits | 0.415 token/s |
| Strix Halo | 2 tokens, 10 GiB cache, identical routes | 3.557 GiB/s, 50% aggregate hits | 0.804 token/s |

The sustained Strix run moved 26.531982 GiB in 6.974184 seconds, evaluated
4,416 routed experts with one graph build and 4,415 graph-cache hits, and
reported zero I/O errors. Adding the exact expert math did not reduce the cold
result materially: compute is hidden behind the 8.84 GiB/token storage path.
The R9700 is the wrong cold owner because the extra discrete-GPU upload path
cuts end-to-end throughput.

The repeated-route result is deliberately a best case, not a prediction.
Kimi K3 was designed for balanced expert use. With unrelated balanced routes,
a cache only helps in proportion to its share of the 495 GiB routed pool.

## Practical Lucebox placement

The machine has about 125.08 GiB of system/UMA memory plus 31.86 GiB on the
R9700, or 156.94 GiB of unique physical weight capacity before runtime
reserves. The implemented capacity-safe Strix-only starting point is:

1. Strix/system memory: all non-routed text weights, shared experts, latent
   projections, recurrent/KV state, workspace, and the routed-expert cache.
2. NVMe: all routed expert stacks, with actual route misses read directly into
   pinned slots and evaluated on Strix.

This works unchanged on a Strix-only machine. When the non-routed execution
plan fits the R9700 budget, the full-Lucebox speed topology instead makes R9700
the primary and sets `DFLASH_MOE_TP_GPU` to Strix. An offline placement
file assigns hot routes to R9700 while Strix executes the remaining routes and
serves the larger capacity cache. Both branches run concurrently; blindly
sending every streamed expert to R9700 remains slower because its cold SSD
upload path measured below Strix.

After approximately 57.94 GiB of non-routed weights plus OS, workspace, and a
moderate context reserve, roughly 70-85 GiB may remain for routed experts.
Under a uniform balanced-routing assumption this covers about 14-17% of the
routed pool. At the measured 3.804 GiB/s, the storage-only ceiling is then
approximately 0.50-0.51 token/s. Full Strix-only inference will be lower
because dense/recurrent work shares the same device; a later R9700 split may
recover part of that gap through overlap.

So the honest expectation for this quant is **roughly one token every two to
three seconds**, not interactive multi-token-per-second generation. Real
router locality can move that estimate; only a route trace from the real model
can establish it.

## Next full-model milestone

The implementation no longer needs another generic cache, a second Kimi
router, or per-layer worker creation. Full-scale qualification requires:

1. Stage all 14 IQ1_S shards and run a short token-for-token comparison against
   the upstream Kimi implementation.
2. Record real `(layer, expert)` routes on a calibration prompt suite and let
   the existing placement planner allocate the measured best cache under the
   chosen context budget.
3. Compare end-to-end output and token rate against ordinary llama.cpp
   CPU/GPU offload.
4. Replace the correctness-first host partial join with a device-resident peer
   join, then tune the placement until the two owner branches balance.

The full 594 GB model cannot currently be staged on the qualification box,
which currently has about 513 GB free. It needs at least about 650 GB of safe
free space for all shards plus logging/headroom; existing user models should
not be deleted implicitly.
