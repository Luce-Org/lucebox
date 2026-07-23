<p align="left">
  <a href="../../README.md">← lucebox-hub</a>
</p>

<h1 align="center">Luce Paged Attention</h1>

<p align="center">
  <strong>Exact fixed-block K/V management for concurrent long-context serving.</strong><br/>
  Decode reads reusable physical K/V blocks directly through per-sequence block tables.
</p>

---

Lucebox's paged-attention path stores full-attention K/V rows in fixed
16-token blocks. Each active sequence owns a logical block table, while the
K/V tensors are addressed by reusable physical block IDs. Decode attention
walks that table directly instead of requiring a sequence's cache to occupy
one contiguous range.

This is separate from [KVFlash](../kvflash/README.md). Paged attention is an
exact layout and allocation mechanism: live blocks remain on the device and
no tokens are evicted. KVFlash is a bounded-residency policy that can evict
cold chunks and changes which context is resident.

The initial integration targets dense Qwen3.5/Qwen3.6 27B (`qwen35`) on one
CUDA or HIP device:

```bash
./build/bin/dflash-server model.gguf \
  --paged-attention \
  --max-ctx 131072
```

The prompt is prefetched with the existing exact, chunked attention path.
Because the first sequence receives blocks from a freshly reset pool, its
prefill layout is identity-mapped. Single-token autoregressive decode then
uses the block table and context length for every full-attention layer.

## Current compatibility

- One local Qwen3.5/Qwen3.6 dense CUDA target.
- Autoregressive decode with full attention (`--fa-window 0`).
- Fixed 16-token blocks.
- Exact block allocation with transactional exhaustion handling and stale
  sequence-handle detection.
- A multi-sequence GPU attention primitive and host allocator, ready for the
  scheduler integration described below.

Startup rejects speculative drafts, DDTree, target layer splitting, and
KVFlash when paged attention is selected. Prefix-cache snapshots are also
disabled in this mode because the existing snapshot format assumes contiguous
K/V rows.

Qwen3.6 is a hybrid model: 48 of its 64 layers use recurrent Gated DeltaNet
state rather than an attention cache. That state is currently stored once per
backend, so admitting multiple live requests would mix their recurrent state
even though their K/V blocks were isolated. Continuous batching therefore
requires two additional pieces before it is safe:

1. make DeltaNet and convolution state sequence-indexed, including state
   routing on every scheduled step;
2. replace the HTTP server's run-to-completion worker with a token-step
   scheduler that emits block tables, valid K/V sequence lengths, and write
   slots for each batch.

Page-aware prefix snapshots, copy-on-write prefix sharing, paged prefill,
speculative verification, and layer-split execution remain follow-up work.
The implementation follows the fixed-block/block-table shape described in
the [llama.cpp paged-attention discussion][llama-discussion], while keeping
these unsupported combinations explicit.

## Recommended roadmap

1. **Current implementation:** decode-only `ggml_paged_attn`, block manager,
   and contiguous-vs-paged benchmarks.
2. **Baseline continuous batching:** iteration-level scheduler, admission
   control, cancellation, and dynamic decode batches.
3. **Batched prefill:** process multiple prompts in one forward pass,
   preferably with variable-length batching.
4. **Chunked prefill:** add a token budget, decode prioritization, and
   prefill/decode interleaving.
5. **Paged-aware prefill:** read prefixes and preceding chunks directly from
   the paged K/V pool.
6. **Prefix caching/CoW:** share reusable prefix blocks with reference counting
   and copy-on-write.

## GPU attention comparison microbenchmark

`bench_paged_attention` compares one native batched contiguous-attention
operation with one paged-attention operation at `n_seq=1,2,4,8`, using Qwen's
D=256, Hq=24, and Hkv=4 dimensions. Its default run also includes one
representative ragged eight-request capacity case:

The paged kernel, Qwen integration, numerical test, and benchmark build on
both CUDA and HIP. HIP numerical coverage has been validated on gfx1151.

```bash
cmake --build "$BUILD_DIR" --target bench_paged_attention
"$BUILD_DIR/bench_paged_attention" \
  --context 4096 \
  --k-type q4_0 \
  --v-type q4_0
```

Both layouts receive the same logical queries and identical quantized K/V
rows. The paged inputs use private, interleaved physical blocks for every
sequence. Before timing, both GPU outputs are compared with a double-precision
CPU oracle, and neither path may exceed `5e-3` maximum absolute error. The
focused paged-attention numerical test uses a tighter `5e-4` bound. Direct
layout error remains a diagnostic because different GPU reduction trees can
diverge at long context. Each sample window queues repeated graph executions
and synchronizes once.

CTest runs the numerical test twice: the default case covers partitioned
long-context attention plus negative/over-capacity length clamps, while
`paged_attention_direct` forces one partition to cover the scratch-free
normalization and empty-sequence path independently of device occupancy.

The CSV reports median step time, paged overhead, aggregate attention-query
throughput, exact K/V and paging-metadata bytes, memory saving, layout error,
and both paths' CPU-oracle error. Automatic calibration gives each layout an
approximately equal-duration sample window.

The default run emits uniform `n_seq=1,2,4,8` rows for direct kernel/layout
comparison, followed by a ragged eight-request capacity row with one
`--context` request and seven requests at `--context / 16`. At the default 4K
context this is `4096 + 7 x 256`; it has the same 16:1 long/short mix and
82.03% K/V saving as the documented `128K + 7 x 8K` profile while remaining
quick to run.

The CUDA decode kernel groups the six Qwen query heads that share each K/V
head in one thread block. It also divides long contexts across independently
computed partitions and merges their softmax statistics with a stable
max/sum reduction. The partition count is occupancy-aware for ordinary
contexts and grows with cache capacity for long contexts, without reading a
device context length on the host or changing CUDA graph topology.

The following reference result was measured on an RTX 3090 with CUDA graphs,
Q4_0 K/V, a 4096-token context, 10 warmups, and 7 automatically calibrated
samples. Times are median microseconds per attention step; the ratio is
`paged / contiguous`, so values above 1 are paged-kernel overhead.

| Sequences | Contiguous (us) | Paged (us) | Ratio |
|----------:|----------------:|-----------:|------:|
| 1 | 81.549 | 106.759 | 1.31x |
| 2 | 142.287 | 209.283 | 1.47x |
| 4 | 254.292 | 393.838 | 1.55x |
| 8 | 486.698 | 733.469 | 1.51x |

To expose the allocation trade-off separately, pass a comma-separated set of
live context lengths:

```bash
"$BUILD_DIR/bench_paged_attention" \
  --ragged-contexts 131072,8192,8192,8192,8192,8192,8192,8192 \
  --k-type q4_0 \
  --v-type q4_0
```

This runs one ragged decode step for each sequence. The native baseline is
specifically a **padded per-sequence contiguous** layout: every sequence owns
`align256(max(contexts))` K/V slots and an F16 mask hides its unused suffix.
The paged layout allocates only `sum(align16(context_i))` slots and interleaves
the sequences' unique physical blocks in that compact pool. Both paths receive
identical logical Q/K/V values. As in the uniform benchmark, both paths must
pass the benchmark's CPU-oracle ceiling before timing; direct parity remains
diagnostic. The CUDA contiguous kernel derives each live prefix from the mask
and skips fully masked tail tiles, so the timing comparison does not charge it
for all padded tokens even though its K/V allocation remains padded.

The ragged CSV reports live, contiguous-padded, and paged-pool token counts;
kernel timings and output error; and exact one-layer K/V and paging-metadata
bytes. The Qwen3.6 projection below multiplies K/V storage by its 16
full-attention layers and excludes model weights, activations, recurrent-layer
state, allocator workspace, and other runtime memory, so it is a capacity
comparison rather than a claim that one layout will or will not fit on a
particular GPU.

For Q4_0 K/V, the eight-user profile above projects to 18.000 GiB for padded
per-sequence contiguous K/V and 3.234 GiB for paged K/V: a 5.57x reduction,
or 82.03% less K/V storage. Uniform `8 x 128K` contexts would use 18.000 GiB
in both layouts, apart from small paging metadata and block rounding.

On the same RTX 3090 setup, the ragged case measured 3.122 ms for contiguous
attention and 4.095 ms for paged attention (`1.31x`), while reducing the
one-layer K/V allocation from 1152 MiB to 207 MiB. This shows both sides of
the trade-off: the current paged kernel still has compute overhead, but it
keeps the highly ragged eight-request workload close to the native kernel
while admitting it with 82.03% less K/V storage.

A second GGML design can use one unified contiguous arena plus a
sequence-aware mask. It can approach sum-sized storage while compact, but
each query attends over a shared high-water span, and request churn can leave
holes that must be reused or defragmented. That alternative is not measured
here. Paging targets sum-sized stable allocation while each sequence follows
only its own block table.

Neither mode measures the full Qwen graph, HTTP scheduling, or
continuous-batching throughput.

[llama-discussion]: https://github.com/ggml-org/llama.cpp/discussions/21961
