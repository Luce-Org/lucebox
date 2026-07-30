# Kimi K3 on heterogeneous Lucebox

This note separates measured facts from estimates. The qualification target is
Unsloth `Kimi-K3-UD-IQ1_S`, currently the smallest published Kimi K3 GGUF: 14
files and 594 GB.

## What is implemented

The common MoE data plane now has the Kimi-specific capabilities that were
missing without making the scheduler Kimi-specific:

- A tensor region carries a model-shard index. `MoeNvmeScheduler` can register
  and read any number of GGUF shard file descriptors with `io_uring`, including
  an expert whose gate, up, and down tensors live in different files.
- The streamed expert dimension can differ from the model hidden dimension.
  This represents Kimi's `7168 -> 3584` routed projection without pretending
  its experts consume full-width hidden states.
- The common streamed graph supports SiTU as well as SwiGLU and DS4's clamped
  SwiGLU.
- `bench_kimi_k3_hetero` runs Kimi's exact IQ1_S routed-expert geometry through
  a reusable graph: 896 experts, top-16, 92 MoE layers,
  `3584 -> 3072 -> 3584`, and SiTU (`beta=4`, `linear_beta=25`). It overlaps
  SSD/H2D for expert N+1 with compute for expert N.

Six scheduler tests pass on the AMD Lucebox, including mmap and real-file
multi-shard reads. Existing single-file DS4 descriptors remain source index
zero and need no model-specific change.

This is not yet a complete Kimi K3 backend. KDA/MLA, Attention Residuals, the
vision encoder, the latent projections around the routed core, tokenizer, and
sampling still need a model adapter. The current upstream llama.cpp Kimi K3
text-model implementation is also not merged, so it should be treated as a
reference implementation rather than a stable dependency.

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
| Strix Halo | 3 tokens, balanced cold routes, compute on | **3.716 GiB/s** | **0.420 token/s** |
| R9700 | 1 token, balanced cold routes, compute on | 2.091 GiB/s | 0.236 token/s |
| Strix Halo | 2 tokens, 10 GiB cache, unrelated routes | 3.638 GiB/s, 0.82% hits | 0.415 token/s |
| Strix Halo | 2 tokens, 10 GiB cache, identical routes | 3.557 GiB/s, 50% aggregate hits | 0.804 token/s |

The sustained Strix run moved 26.531982 GiB in 7.139575 seconds, evaluated
4,416 routed experts, and reported zero I/O errors. Adding the exact expert
math did not reduce the cold result materially: compute is hidden behind the
8.84 GiB/token storage path. The R9700 is the wrong cold owner because the
extra discrete-GPU upload path cuts end-to-end throughput.

The repeated-route result is deliberately a best case, not a prediction.
Kimi K3 was designed for balanced expert use. With unrelated balanced routes,
a cache only helps in proportion to its share of the 495 GiB routed pool.

## Practical Lucebox placement

The machine has about 125.08 GiB of system/UMA memory plus 31.86 GiB on the
R9700, or 156.94 GiB of unique physical weight capacity before runtime
reserves. A simple placement is:

1. R9700: attention/KDA/MLA and other dense matrices that fit its 32 GiB.
2. Strix/system memory: remaining non-routed weights, shared experts, latent
   projections, recurrent/KV state, workspace, and the routed-expert cache.
3. NVMe: all routed expert stacks, with actual route misses read directly into
   pinned slots and evaluated on Strix.

After approximately 57.94 GiB of non-routed weights plus OS, workspace, and a
moderate context reserve, roughly 70-85 GiB may remain for routed experts.
Under a uniform balanced-routing assumption this covers about 14-17% of the
routed pool. At the measured 3.716 GiB/s, the storage-only ceiling is then
approximately 0.49-0.50 token/s. Full inference will be lower unless dense
R9700 work overlaps almost completely with Strix expert service.

So the honest expectation for this quant is **roughly one token every two to
three seconds**, not interactive multi-token-per-second generation. Real
router locality can move that estimate; only a route trace from the real model
can establish it.

## Next end-to-end milestone

The next useful step is one narrow Kimi adapter, not another generic cache:

1. Import the correctness-first text graph from the upstream Kimi K3 work.
2. Populate shard-indexed expert regions directly from the GGUF tensor table.
3. Place dense attention on R9700 and the latent/shared/routed MoE path on
   Strix; keep only activation-sized transfers at the boundary.
4. Record real `(layer, expert)` routes on a calibration prompt suite and let
   the existing placement planner allocate the measured best cache under the
   chosen context budget.
5. Compare end-to-end output and token rate against ordinary llama.cpp
   CPU/GPU offload.

The full 594 GB model cannot currently be staged on the qualification box,
which has substantially less free SSD space. It needs at least about 650 GB of
safe free space for all shards plus conversion/logging headroom; existing user
models should not be deleted implicitly.
