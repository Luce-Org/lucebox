# Early-Exit Forward Design

## Problem

At 32K-64K context, `A_compute` (Q/K/V projections, RoPE, chunked) and `FP` (FlashPrefill kernel) dominate the drafter forward wall. From the Tier 1 spike at 23K tokens:

- A_compute: ~2.15s total (28 layers x ~77ms each)
- FP kernel: ~1.95s total (28 layers x ~70ms each)
- tail-score: ~1.96s (28 layers)

If we exit at layer N, we save (28-N)/28 of A_compute and FP costs.

## Code Locations

**dflash/src/qwen3/qwen3_graph.cpp** -- `forward_qwen3_drafter_model()`:

- Line 381: `for (int il = 0; il < w.n_layer; ++il)` -- the per-layer forward loop.
  Insert early-exit check at top of this loop body.
- Line 799: `const int score_layer_start = ...` -- the scoring head reads `score_layers` env var.
  The scoring loop (line 805) iterates `for (int il = score_layer_start; il < w.n_layer; ++il)`.
  With early-exit at N, layers N..27 have no K/Q data. The scoring loop must be capped at `early_exit_n`.

**Key interaction**: `PFLASH_DRAFTER_SCORE_LAYERS=S` sets `score_layer_start = w.n_layer - S`.
With early-exit at N, the effective scoring window is `score_layer_start .. early_exit_n - 1`.
The two vars compose cleanly: scoring only touches layers that were actually computed.

## Env Var

`PFLASH_DRAFTER_EARLY_EXIT_N=N` -- only run the first N layers of the forward (default 28, no change).

Pattern mirrors `PFLASH_DRAFTER_SCORE_LAYERS`: static int initialized from getenv at first call.

## Changes

1. **qwen3_graph.cpp line ~381** -- add static early_exit_n read, then at top of layer loop body:
   `if (il >= early_exit_n) break;`

2. **qwen3_graph.cpp line ~805** -- cap scoring loop end:
   `for (int il = score_layer_start; il < std::min(w.n_layer, early_exit_n); ++il)`
   This ensures scoring never reads K/Q_norope from layers that were not computed.

3. **Persistent buffer allocation** -- unchanged. We still allocate K_curr/V_curr/Q_last/K_norope/Q_norope
   for ALL w.n_layer. The early-exit layers have uninitialized buffers but they are never read by the
   capped scoring loop. VRAM stays the same.

## Expected Wall Savings (extrapolated from 23K per-layer cost)

Per-layer cost at 23K tokens (A_compute + FP): ~(2.15+1.95)/28 = ~145ms/layer

| Condition | Layers computed | A+FP saved | fwd @ 23K | fwd @ 46K (x2 scaling) |
|-----------|-----------------|------------|-----------|------------------------|
| baseline  | 28              | 0          | ~11s      | ~27s                   |
| ee14      | 14              | ~2.0s      | ~9s       | ~20s                   |
| ee7       | 7               | ~3.0s      | ~8s       | ~16s                   |

Score cost also drops proportionally (already < 2s at baseline).

## Quality Risk

The tail-score uses K from layers 0..N-1. At N=14, scoring has 14 layers of attention signal vs 28.
The NoPE fix (pre-RoPE K) still applies. At N=7, only 1/4 of the model depth contributes.

Gate: NIAH must remain 3/3 correct (100%) per cell to call a condition passing.
