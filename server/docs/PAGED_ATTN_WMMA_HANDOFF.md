# paged_attn_wmma — handoff

Status: LANDED, env-gated (`DFLASH27B_PAGED_WMMA`, default off), RDNA4-only.
Stage-1 kernel: `paged_attn_wmma` in `paged-attn.cu`, a paged port of the
contiguous `fattn-mma-f16.cuh` machinery. This file keeps the non-obvious
invariants, the bug classes already hit, and the qualification state. Read the
landed code; do not re-derive the assembly.

## Invariants (load-bearing — do not "harmonize" with the decode kernel)

- **Log2 domain**: `exp2f` everywhere; `scale_h2 = scale*log2(e)`. `KQ_max`
  inits at `-FLT_MAX/2.0f` (NOT `-FLT_MAX`): a fully-masked paged pass must
  produce `exp2 = 0`, never `-inf - (-inf) = NaN`.
- 2-arg `mma(D,A,B)` **accumulates** on AMD; zero `KQ_C`/`VKQ_C` explicitly.
- Tiles are `tile<16,16,float>` / `tile<16,8,half2>` with DEFAULT I_MAJOR —
  on RDNA4 that already IS the transposed A/B layout. Do NOT declare J_MAJOR.
- Tile config is compile-time: D=256, ncols1=4, ncols2=8, np=4, nbatch_fa=64,
  nbatch_combine=32, tile_stride=36. The launcher gates `gqa_ratio <= 8` and
  `block_size == 16`; without the first, heads 8+ silently never compute.
- Rows in a block may belong to different sequences: enumerate the <=4 unique
  seqs **block-uniformly** into smem (a divergent per-warp enumeration
  deadlocks the `__syncthreads`), then run one iter pass per sequence.

## Bug classes already hit (each reproduced before fixing)

1. **Partition overlap** — clamp the iter tile to the partition's `token_end`:
   `k_VKQ_sup = clamp(token_end - (token_begin + kb0*nbatch_fa), 0, 64)`.
   Without it, tokens are staged by every spanning partition and softmax runs
   over a multiset (differential 0.18 -> 0.07).
2. **`partial_meta` cross-block race** — the meta write needs the
   `c >= gqa_ratio` guard (ncols2=8 > gqa_ratio=6) or zero-Q padding columns
   write into the next `kv_head`'s slots. Also apply `FORCE_PARTITIONS` after
   the occupancy floor, not before.
3. **Invalid ragged rows** — a row with `query_positions` present, no tree, and
   `pos < 0` is a dead row (empty context, zero output); it must not fall
   through to the full `kv_len`. `test_paged_attn_wmma`'s mixed `qpos=-1` row
   guards this.
4. **Gather** — resolve the block table once per thread (one thread owns 16
   contiguous half2s of one row). First attempt compared `k_VKQ_sup` against a
   global token instead of the chunk-local row, and used Q8_0 block stride 18
   instead of 34.

## Write-back guards (the silent-corruption surface)

- Skip dead columns: `c >= gqa_ratio || row >= n_rows || row invalid`.
- Single partition divide: `inv_sum = qk_sum > 0 ? 1/qk_sum : 0` (dead rows
  yield 0, not NaN).
- Multi-partition: `partial_acc`/`partial_meta` exactly as `paged_attn_decode`
  (pre-normalized acc, log2 meta, `(-FLT_MAX, 0)` sentinel); reuse
  `paged_attn_partitions` + `paged_attn_combine`. An unwritten sentinel reads
  recycled pool memory as a plausible `float2`.

## Qualification

`test_paged_attn_wmma` diffs the WMMA route (env=1, pinned by the
`g_paged_attn_wmma256` launch counter) against V_DOT2 (env=0, counter must stay
0). The CPU backend ABORTS on `GGML_OP_PAGED_ATTN` (`ggml-cpu.c:2247`), so the
two GPU routes are compared against each other. Cases: 512/8192 prefill, mixed
ragged, sparse multi-partition, decode row, F16/Q8_0/Q4_0.

Differential green: max 2.08e-3 (q8_0) / 4.28e-3 (q4_0) against 3e-3 / 6e-3
tolerances, deterministic. `compare_paged_attn.py` reports one global max;
per-case headroom is >=4x except **mixed-q4 at 1.40x** (short-extent rows carry
the larger error). If a codegen change flips mixed-q4, split the comparator per
case rather than widening `--tol`.

Debugging traps that cost hours:
- numpy reshape of a sliced view silently keeps wrong strides — index flat or
  reshape to the true memory order.
- Ungated debug dumps race across blocks; gate to one (kv_head, partition,
  group).
- Host ground truth dumps pre-quantization q/kv dim-major (see the test).

## Performance

Isolated op (`bench_paged_attn_wmma`, q8_0, TFLOP/s): paged WMMA **20.6-22.4**
vs paged V_DOT2 **6.4-6.8** (3.1-3.3x), matching the contiguous MMA template.
End-to-end server A/B (q8_0): batched 8K parity to +6%; single 12K -20%
(14.5 -> 11.6 s), 44K -42% (93.6 -> 54.4 s). The kernel-level 3x is bounded by
attention's share of prefill (~8% at 44K, ~4% at 12K end-to-end).

Reverted: software double-buffer (measured **+12.6%** at 44K with 3 blocks/CU —
gather latency was already hidden; `cp.async` is `!GGML_USE_HIP`-gated).

## Verify on lucebox4

Build with the ROCm CI flags, then run both routes and the comparator:

```
ctest -R '^test_paged_attn_wmma$'                     # V_DOT2
ctest -R '^paged_attn_wmma_route$'                    # DFLASH27B_PAGED_WMMA=1
python3 server/test/compare_paged_attn.py \
    paged_attn_out_vdot2.bin paged_attn_out_wmma.bin
```
