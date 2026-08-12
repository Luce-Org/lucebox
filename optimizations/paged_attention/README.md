<p align="left">
  <a href="../../README.md">← lucebox-hub</a>
</p>

<h1 align="center">Luce Paged Attention</h1>

<p align="center">
  <strong>Exact fixed-block K/V for long-context serving.</strong><br/>
  Every sequence owns a block table; decode reads reusable 16-token physical blocks straight through it.<br/>
  Ragged 8-request profile (<code>128K + 7x8K</code>) on one RTX 3090:
  <strong>82% less K/V storage and 1.35x faster attention</strong>.
</p>

---

## How it works

- **Blocks.** Full-attention K/V rows live in fixed 16-token physical blocks in
  one pool. Nothing is evicted and nothing is approximated — this is a layout
  and allocation mechanism, not a residency policy.
- **Block tables.** Each sequence owns a logical→physical map. Decode walks it,
  so a sequence never has to occupy one contiguous K/V range, and the pool only
  ever holds live, block-rounded tokens.
- **One decode op.** `GGML_OP_PAGED_ATTN` on CUDA and HIP: D=256 GQA, F16/Q4_0/Q8_0
  K/V. One warp covers all query heads of a K/V group per row loaded, and long
  contexts split into partitions merged by a stable split-softmax.
- **Resident metadata.** The block table and sequence length live in the
  persistent target cache next to the pool, so a decode step uploads 4 bytes per
  newly filled block instead of the whole table every token.
- **Prefill is unchanged.** The prompt runs through the existing exact chunked
  path against a freshly reset pool, so its layout is identity-mapped. Paging
  begins at autoregressive decode.

Not to be confused with [KVFlash](../kvflash/README.md): that one bounds how much
context stays resident and evicts cold chunks. Paging keeps every token.

## Usage

```bash
./server/build/dflash_server model.gguf --paged-attention --max-ctx 131072
```

## Compatibility

| | |
|---|---|
| Architecture | `qwen35` — dense Qwen3.5 / Qwen3.6 |
| Placement | one local CUDA or HIP device |
| Attention | full only (`--fa-window 0`) |
| K/V types | F16, Q4_0, Q8_0 |
| Decode | autoregressive, one live sequence |
| Block size | 16 tokens, fixed |

**Rejected at startup** (exit 2, with the reason): `--draft`, `--ddtree`,
`--target-devices`, `--target-shard-ipc-bin`, a non-zero `--fa-window`,
`--prefill-compression`, and `DFLASH_KVFLASH`. The rules live with every other
launch-admission rule in `check_feature_compatibility()`; which architecture and
placement they are allowed on is one row in `model_capabilities.h`.

**Disabled, not rejected:** prefix, prefill, and disk snapshots. Their format
assumes contiguous K/V rows, so `--paged-attention` zeroes the caps and says so.

**Why one sequence.** Qwen3.6 is hybrid: 48 of its 64 layers hold recurrent Gated
DeltaNet state rather than an attention cache, and that state is backend-global
today. Two live requests would mix it even with perfectly isolated K/V blocks.
Making it sequence-indexed is step 2 below.

## Numbers

Attention ops only, Q4_0 K/V, Qwen dims (`D=256, Hq=24, Hkv=4`). Uniform batches
of 1/2/4/8 sequences, as `paged ÷ contiguous` throughput — above 1.0x is paged
winning.

| Context | RTX 3090 (CUDA) | Radeon 8060S (HIP) |
|---|---|---|
| 4K | 1.10–1.20x | 4.28–5.60x |
| 32K | 1.12–1.23x | 4.72–5.16x |
| 128K | 1.23–1.97x | 5.00–6.42x |

Ragged 8-request profile, `128K + 7x8K` on the RTX 3090. Paging stores only live,
block-rounded tokens (188,416) instead of padding every sequence to 128K
(1,048,576 slots):

| | contiguous | paged |
|---|---|---|
| attention step | 3.007 ms | **2.235 ms** (1.35x) |
| K/V, one layer | 1152 MiB | **207 MiB** |
| K/V, 16 full-attn layers | 18.0 GiB | **3.23 GiB** (−82%) |

Two caveats. The HIP ratios are inflated by a weak native HIP
`flash_attn_ext_vec` baseline — don't read them as whole-model throughput. And
small ragged batches are still behind (`4K + 7x256` measures 0.48x), because
short sequences carry dead partitions sized by the longest sequence; the stream-K
kernel in step 4 is what fixes that.

## Roadmap

1. **Done.** Decode-only `ggml_paged_attn`, block manager, CUDA/HIP integration,
   benchmarks.
2. **Scheduler groundwork.** Sequence-indexed DeltaNet/conv state, iteration-level
   scheduling, dynamic decode batches.
3. **Paged-aware chunk prefill.** Arbitrary block-table writes, multi-token
   queries over preceding paged K/V, causal attention within the chunk.
4. **Continuous admission and interleaving.** Token budgets, decode
   prioritization, admission control, cancellation, prefill/decode interleaving,
   plus the stream-K decode kernel for ragged batches.
5. **Variable-length batched prefill.**
6. **Prefix caching / CoW.** Shared prefix blocks with reference counting and
   copy-on-write.

## Benchmarks and tests

```bash
cmake --build "$BUILD_DIR" --target bench_paged_attention
"$BUILD_DIR/bench_paged_attention" --context 131072 --k-type q4_0 --v-type q4_0
```

Each run compares one native padded-contiguous attention op against one paged op
at `n_seq=1,2,4,8`, then adds a ragged 8-request row. Both layouts get identical
logical Q/K/V and identical quantized K/V rows, and both must pass a
double-precision CPU oracle before anything is timed. The CSV reports median step
time, aggregate query throughput, exact K/V and metadata bytes, and each path's
oracle error. It measures attention ops — not the full Qwen graph, HTTP
scheduling, or continuous-batching throughput.

`ctest -R paged` covers the host allocator plus the nine F16/Q4_0/Q8_0 K/V
combinations, in both the partitioned and the forced-single-partition kernel
paths.

The fixed-block/block-table shape follows the
[llama.cpp paged-attention discussion][llama-discussion].

[llama-discussion]: https://github.com/ggml-org/llama.cpp/discussions/21961
