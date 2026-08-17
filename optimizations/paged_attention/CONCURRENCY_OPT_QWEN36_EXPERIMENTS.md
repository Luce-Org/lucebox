# Qwen3.6 Concurrency Optimization Experiments

Last reconciled: 2026-08-12.

This ledger consolidates the Codex sessions and local notes covering Qwen35/36, paged attention, concurrent serving, follow-up experiments on the PR #596 benchmark stack, and the latest Strix Halo profiling. It distinguishes retained production changes from negative screens, inconclusive results, and ideas that have not been implemented.

Unless noted otherwise, measurements used Qwen3.6-27B Q4_K_M on a Radeon 8060S (`gfx1151`) with Q4_0 K/V. Most isolated optimization figures are engineering screens rather than five-repeat publication results.

The central architectural result is already in PR #595: ragged prompt slices and live decode rows share one packed token axis and one model traversal. Dense projections, normalization, FFNs, full-attention lowering, and output projections operate over that packed width. Only sequence-dependent attention/recurrent work is lowered by segment. This is also why the large DeepSeek4 gathered-concurrency optimization does not have a second direct port here: Qwen already performs the equivalent whole-batch projections and does not issue DeepSeek4's layer-by-layer, lane-by-lane metadata uploads or duplicated MLA/indexer gathers.

## Retained optimizations

- **Paged KV pool, block tables, and batched decode:** positive. The benefit is strongest at C8/C16 and cannot be attributed to paged attention alone.
- **Ragged packed prefill:** positive; shares the graph, projections, and FFNs across requests.
- **Fused prefill/decode and continuous-batching scheduler:** positive, roughly +1-2% over the previous branch.
- **Admission burst/coalescing:** 5 ms is effectively identical to 0 ms; it is not the main source of the gain.
- **In-place Gated DeltaNet state and compact decode batches:** retained as part of the concurrency path.
- **Automatic hipBLASLt on gfx1151:** positive, mainly at C16: approximately +2.9% aggregate throughput, +3.9% per-request decode, and +4.4% output-window throughput.
- **Specialized DeltaNet transpose/concat:** the strongest improvement found: approximately +3.6% at C4, +4.6% at C8, and +7.4% at C16; retained.
- **Workload-sized KV pool:** same throughput with lower memory allocation; positive as a capacity optimization.
- **Axis-major M-RoPE fix:** not a performance optimization, but it improved serial/concurrent consistency from 0/16 to 8/16 matching outputs.
- **gfx1151 Q4_K 64x64/four-warp MMQ tile:** retained for the narrow MMQ route. An initial C8 decode-facing screen measured 64.66 versus 59.88 output-window tok/s (about +8%) against the 128x128 tile. This is not a wide-prefill result: packed prefill normally crosses to dequantization plus hipBLASLt above 256 columns.
- **Concurrent ROCTX regions and route/tile diagnostics:** retained as observability or build-time specialization where present; they are not performance claims by themselves.

## Tried and rejected

- **K=2 "parallel prefill":** two separate serial graphs with synchronization and duplicated staging; only modest gains, not true GPU batching.
- **Larger prefill budgets:** approximately +1.2-1.4% at C4/C8, but capacity problems at C16; reverted.
- **HIP graphs/replay:** approximately 1% or flat; launch overhead is not dominant.
- **Async tensor transfers, vector-allocation hoisting, and metadata caching:** no isolated gain demonstrated.
- **Native wide Q4_K WMMA:** at most 0.48x the hipBLAS path; removed.
- **Wider MMVQ/GEMV paths:** substantially slower, approximately -20/23% to -42/69%.
- **Non-grouped DeltaNet fallback:** -3.4%.
- **Equal-length DeltaNet segment grouping:** +1.5% at C8 and +0.5% at C16; too limited, removed.
- **Direct scatter instead of concat:** flat.
- **Removing explicit DeltaNet Q/K repeats:** approximately +0.56% on medium, within noise; reverted.
- **One-partition paged attention:** -2.9% at C8 and -1.1% at C16.
- **Q-tiled paged attention (Q_TILE=4):** correctness tests passed, but end-to-end A/B was flat and sometimes regressed at C16; not retained.
- **Removing paged Q/K/V copies:** -2.9% in the first test, then essentially flat; not retained as a performance change.
- **Batched logits readback:** -1.4% in the non-greedy test.
- **Packed metadata upload:** -2.1%.
- **SwiGLU split fusion:** -1.9% with no stable gain; rejected as a performance optimization.
- **Native consumer output layout:** -2.1%.
- **Fusing the sigmoid gate into the attention epilogue:** not reproducible.
- **Batching slot resets and block-table updates:** not reproducible.
- **Merging graph compute and argmax synchronization:** slightly slower.
- **"Tall block" DeltaNet retile:** +0.95% against the immediate control; considered noise.
- **Persistent graph reuse:** measured maximum overhead recovery of approximately 0.6% at C8; not worth the complexity.
- **HipBLASLt plan cache by shape:** implemented as an experiment, but not yet validated with a clean A/B.
- **Prefill-width K sweep:** K8 was the best max-TTFT/goodput setting in the tested matrix. Medium C8 prompt throughput was 303.0/330.6/352.7/359.6 tok/s for K1/K2/K4/K8. Medium C16 favored K8 over K16 (350.9 versus 346.4), and long C8 favored K8 over K4 (345.7 versus 337.6). Lower K sometimes improved median TTFT but worsened tail latency and throughput.
- **Forcing BLAS for narrow Q4_K decode:** decisively worse in the route screen (14.28 versus 59.88 C8 output-window tok/s). Narrow concurrent decode remains on MMQ.
- **Raising the Q4_K MMQ cutoff:** rejected for packed prefill. The final fresh-process screen kept the production cutoff at 256 columns:

  | Q4_K MMQ ceiling | Short C8 | Medium C8 | Medium C16 | Long C16 |
  | ---: | ---: | ---: | ---: | ---: |
  | 256 baseline | 26.72 | 15.28 | 16.00 | 6.57 |
  | 512 | - | 14.90 | - | - |
  | 1024 | 26.67 | 15.12 | 15.92 | 6.53 |
  | 2048 | - | 14.34 | - | - |

  The 1024 candidate was slower everywhere (-0.2%, -1.0%, -0.5%, and -0.6%). Earlier isolated pairs that appeared favorable did not survive fresh paired repeats.
- **Forcing FP32 GEMM output/compute to remove conversions:** hard rejection. Medium-C1 TTFT increased from about 3.43 to 10.70 seconds.
- **Direct hipBLASLt heuristic selection:** the direct API could not reproduce the server's fast FP16-compute compatibility path with valid output on the installed ROCm version. Its supported FP32-compute mode was already much slower.
- **Fixed rocBLAS/hipBLASLt solution index:** individual shapes improved by as much as 24-26%, but a fixed solution improved weighted GEMM time only about 0.8% and regressed other shapes.
- **Shape-aware hipBLASLt solution cache/autotune:** even optimistic best-solution-per-shape selection reduced weighted GEMM time only about 3.3%. With GEMM near 58% of prefill time, the end-to-end ceiling was roughly 2% before tuning overhead; rejected.

## Inconclusive or opt-in-only experiments

- **Qwen prefill quantum via `--chunk`:** wiring the previously dropped setting into dense Qwen concurrent planning made 1024 a modest Strix candidate, but not a justified default. Three paired medium-C8 runs showed 24.865 versus 25.201 seconds max TTFT for q1024 versus q512 (about 1.3%); long C1/C8 improved about 2%. q256 regressed, q1536 delayed shorter requests despite a slightly better maximum, and q2048 added no benefit. A later q1024 run was a significant outlier. Keep 512 pending a broader repeated study.
- **Full-generation numerical topology dependence:** first-token C16 became stable after the M-RoPE correction, but later greedy tokens can still differ as packed/decode topology changes. This is a robustness/publication issue, not evidence of cross-request state leakage by itself.

## Latest packed-prefill profile

The latest trace-only packed-prefill windows show that the optimization boundary has moved away from scheduling and DeltaNet. Kernel times overlap and therefore do not sum to wall time.

| Workload | Packed tokens | Q4_K dequant | hipBLASLt GEMM | DeltaNet | Transpose/concat | Paged attention | Padding/dead-row | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short C8 | 3,743 | 0.428 s | 6.588 s | 0.449 s | 0.249 s | 0.118 s | 0.003 s | 2.874 s |
| Medium C8 | 9,126 | 1.286 s | 14.719 s | 1.085 s | 0.553 s | 0.683 s | 0.013 s | 7.073 s |
| Medium C16 | 18,276 | 2.365 s | 30.545 s | 2.254 s | 1.009 s | 1.442 s | 0.030 s | 14.390 s |
| Long C16 | 54,158 | 5.388 s | 93.364 s | 6.768 s | 3.269 s | 11.776 s | 0.066 s | 42.272 s |

No direct Q4_K MMQ dispatch appeared in these complete packed-prefill windows; MMQ remains relevant to decode and smaller scheduling intervals.

Representative counters reinforce the route decision:

- Q4_K MMQ `<12,80>`: 184 VGPR, 128 SGPR, 23.1% occupancy, 39.7% MemUnitBusy, and 51.5% L2 hit.
- Q4_K dequantization: 16 VGPR, 128 SGPR, approximately 96.5% occupancy, 99.2% MemUnitBusy, and 47.3% L2 hit.

The dequantizer is traffic-heavy; direct MMQ is register/occupancy limited. ROCm 7.2.4 did not expose a usable MFMA/WMMA utilization counter for this gfx1151 study.

## Untried candidates

These are proposals, not measured results.

1. **Wide FFN gate/up super-op.** Both `[5120,17408]` Q4_K projections consume the same activation in all 64 layers. The meaningful experiment is to share activation conversion, materialize both weights into adjacent FP16 workspace, issue one tall hipBLASLt GEMM, expose gate/up as views, and fuse the FP16/FP32/SwiGLU boundary where numerically legal. This is materially different from the rejected elementwise-only SwiGLU experiment.
2. **DeltaNet QKV plus gate/z multi-output projection.** It affects 48 layers but mixes Q6_K and Q4_K, requiring heterogeneous dequantization into a shared FP16 workspace. Its surface is smaller and implementation more complex than FFN gate/up.
3. **True variable-length ragged DeltaNet operation.** Pass segment offsets, lengths, and slot IDs to one conv/recurrent operation instead of launching per segment. This is different from rejected equal-length grouping, but DeltaNet's profile share limits the ceiling.
4. **Attention K/V projection packing.** Low priority: K and V use mixed Q4_K/Q6_K and appear in only 16 full-attention layers.

The FFN super-op should remain opt-in and wide-prefill-only until it clears at least a 5% end-to-end gate on its target workload.

## Benchmark gate for the next candidate

PR #596 is a useful burst benchmark, but source-change A/Bs should compare two Luce binaries directly.

- Use baseline and candidate Luce binaries with identical K8 configuration and libraries.
- Run five paired AB/BA repeats.
- Use short/256 at C1/C4/C8/C16 and medium/64 at C8/C16 when prefill code changes.
- Require exact forced work, accepted per-prompt output behavior, a median paired gain at target concurrency, and less than 2% C1 regression.
- Record actual packed/live/decode widths, route, graph-build/reuse, clocks, power, temperature, and throttling.
- Add a cold-burst pair and a staggered-admission/slot-reuse/cancellation soak.

## Summary

The proven gains come from packed concurrency and scheduling, gfx1151 hipBLASLt selection, the narrow Q4_K tile specialization, and especially the DeltaNet concat specialization. Metadata, graph, scheduler-width, attention micro-fusion, elementwise fusion, and wider-MMQ experiments have now been screened extensively.

For wide packed prefill, hipBLASLt GEMM is dominant and the Q4_K cutoff should remain 256. The next substantial experiment must remove traffic across the quantized-materialization/GEMM/FFN boundary; another scheduler or paged-attention micro-optimization is unlikely to produce a large result.
