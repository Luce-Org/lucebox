# Early-Exit Forward Bench — 2026-05-21

Binary: `dflash/build/dflash_server` built 2026-05-21 21:48  
Patch: `PFLASH_DRAFTER_EARLY_EXIT_N` support in `dflash/src/qwen3/qwen3_graph.cpp` (uncommitted)  
GPU: RTX 3090 (24 GiB), TQ3_0 KV cache, BF16 drafter only

## Conditions

| Condition | PFLASH_DRAFTER_EARLY_EXIT_N | PFLASH_DRAFTER_SCORE_LAYERS |
|---|---|---|
| baseline_ee | unset (28 layers) | unset (28 layers) |
| ee14 | 14 | unset (28, clamped to 14) |
| ee7 | 7 | 7 (BUG: empty scoring range — see notes) |

## Results Table

| Condition | ctx | drafter_fwd_cold | drafter_fwd_warm | tail_score_warm | A_compute_warm | NIAH | warm_speedup_vs_baseline |
|---|---|---|---|---|---|---|---|
| baseline_ee | 32K | 3.650s | 3.520s | 0.570s | 0.530s | 3/3 | 1.00x |
| baseline_ee | 64K | 7.350s | 7.280s | 1.145s | 1.045s | 3/3 | 1.00x |
| ee14 | 32K | 1.910s | 1.840s | 0.290s | 0.265s | 3/3 | 1.91x |
| ee14 | 64K | 3.760s | 3.785s | 0.595s | 0.530s | 3/3 | 1.92x |
| ee7 | 32K | 0.920s | 0.830s | 0.000s* | 0.130s | 3/3 | 4.24x |
| ee7 | 64K | 1.690s | 1.745s | 0.000s* | 0.275s | 3/3 | 4.17x |

*ee7 tail_score=0.000s because scoring range is empty (layers 7-6 inverted — see bug note).

warm_fwd_warm is p50(rep1, rep2). warm_speedup = baseline_warm_p50 / cond_warm_p50.

## Per-rep detail (cold=rep0, warm=rep1,rep2)

### baseline_ee
- 32K: rep0=3.650s rep1=3.490s rep2=3.520s  A_compute: 0.620/0.520/0.530  FP: 0.510/0.500/0.510  tail: 0.590/0.570/0.570
- 64K: rep0=7.350s rep1=7.330s rep2=7.220s  A_compute: 1.060/1.050/1.040  FP: 1.250/1.210/1.210  tail: 1.170/1.150/1.140

### ee14 (14/28 layers)
- 32K: rep0=1.910s rep1=1.820s rep2=1.840s  A_compute: 0.340/0.260/0.270  FP: 0.240/0.250/0.250  tail: 0.310/0.290/0.290
- 64K: rep0=3.760s rep1=3.780s rep2=3.790s  A_compute: 0.520/0.530/0.530  FP: 0.590/0.600/0.590  tail: 0.590/0.590/0.600

### ee7 (7/28 layers, scoring range bug)
- 32K: rep0=0.920s rep1=0.830s rep2=0.830s  A_compute: 0.210/0.130/0.130  FP: 0.120/0.120/0.120  tail: 0/0/0
- 64K: rep0=1.690s rep1=1.790s rep2=1.700s  A_compute: 0.260/0.280/0.270  FP: 0.290/0.310/0.290  tail: 0/0/0

## Key Findings

**Early-exit does NOT trigger the warm-path regression seen in layer-subset (349cbc5).** A_compute scales proportionally to the number of layers executed:
- baseline_ee warm 32K A_compute = 0.525s (28 layers)
- ee14 warm 32K A_compute = 0.265s (14 layers) → 1.98x reduction, matches 2x layer ratio
- ee7 warm 32K A_compute = 0.130s (7 layers) → 4.04x reduction, matches 4x layer ratio

This confirms early-exit skips layers cleanly without the warm-cache inflation that plagued `score_layer_start > 0`.

**ee7 scoring bug**: With `PFLASH_DRAFTER_EARLY_EXIT_N=7` and `PFLASH_DRAFTER_SCORE_LAYERS=7`, the clamp logic sets `score_layer_start = min(28-7, 7) = 7` and `score_layer_end = fwd_layer_limit = 7`, producing an empty range [7, 7). The tail-score runs zero iterations — quality is undefined. NIAH still passes 3/3 because with keep_ratio=0.05 at 32K/64K the compression is aggressive enough that any result is effectively random-but-lucky; this cannot be trusted for real-world quality. **ee7 as currently configured is broken for scoring.**

**ee14 is the viable condition**: 1.91-1.92x warm speedup with proper scoring (14 layers, range 0-13), NIAH 3/3 at both 32K and 64K. Cold run shows ~1.91x as well — no cold penalty vs warm, unlike the 128K layer-subset run.

**Headline**: ee14 delivers ~1.91x drafter_fwd speedup at 32K and 64K with NIAH preserved. ee7 is ~4.2x but the scoring range bug makes it untrustworthy until the `score_layer_start` clamping logic is fixed.

## No warm-path regression

Unlike the layer-subset 128K rebench (349cbc5) where A_compute inflated 5.4x on warm runs, early-exit shows stable warm-run performance. The bug in layer-subset (score_layer_start > 0 invalidating the warm optimization) is NOT triggered by early-exit because early-exit always starts scoring from layer 0 (score_layer_start=0 for baseline and ee14).
