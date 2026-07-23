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
D=256, Hq=24, and Hkv=4 dimensions:

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
sequence, and an untimed output comparison must pass before timing starts.
Each sample window queues repeated graph executions and synchronizes once.

The CSV reports median and p95 window-averaged step time, paged overhead
relative to contiguous attention, aggregate attention-query throughput,
scaling efficiency, K/V memory, block-table metadata, and output error.
Automatic calibration gives each layout an approximately equal-duration
sample window. All sequences have the same context length, so this is a direct
kernel/layout comparison.

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
identical logical Q/K/V values, and output parity is mandatory before either
path is timed. The CUDA contiguous kernel derives each live prefix from the
mask and skips fully masked tail tiles, so the timing comparison does not
charge it for all padded tokens even though its K/V allocation remains padded.

The ragged CSV reports live, contiguous-padded, and paged-pool token counts;
kernel timings and output error; exact one-layer K/V bytes; and a K/V-only
projection across Qwen3.6's 16 full-attention layers. The projection excludes
model weights, activations, recurrent-layer state, allocator workspace, and
other runtime memory, so it is a capacity comparison rather than a claim that
one layout will or will not fit on a particular GPU.

For Q4_0 K/V, the eight-user profile above projects to 18.000 GiB for padded
per-sequence contiguous K/V and 3.234 GiB for paged K/V: a 5.57x reduction,
or 82.03% less K/V storage. Uniform `8 x 128K` contexts would use 18.000 GiB
in both layouts, apart from small paging metadata and block rounding.

A second GGML design can use one unified contiguous arena plus a
sequence-aware mask. It can approach sum-sized storage while compact, but
each query attends over a shared high-water span, and request churn can leave
holes that must be reused or defragmented. That alternative is not measured
here. Paging targets sum-sized stable allocation while each sequence follows
only its own block table.

Neither mode measures the full Qwen graph, HTTP scheduling, or
continuous-batching throughput.

[llama-discussion]: https://github.com/ggml-org/llama.cpp/discussions/21961
