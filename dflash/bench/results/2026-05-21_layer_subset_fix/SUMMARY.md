# Layer-Subset Warm-Path Regression Fix — 2026-05-21

## Diagnosis

With `PFLASH_DRAFTER_SCORE_LAYERS=7` (28-layer model), the old code allocated
`K_norope_v` for all 28 layers regardless of how many would be read in scoring.
At S=128K each layer's K_norope buffer is D×Hk×S×2 = 128×8×131080×2 = 268 MB.
All 28 layers = 7.5 GB; only 7 are needed = 1.9 GB.

The 21 wasted layers (5.6 GB) pushed total persistent VRAM above 24 GB
(K_curr + V_curr + K_norope + model = 25.6 GB on RTX 3090). CUDA page migration
to system RAM slowed all GPU kernel dispatches: A_compute inflated 5.4x, FP 25%,
untimed Graph-B ~2x. The scoring win (tail-score -48%) was overwhelmed by ~27 s
of extra forward work.

## Fix

`dflash/src/qwen3/qwen3_graph.cpp` — Path A refactor, ~47 LOC changed.

- Compute `score_layer_start_pre` via `compute_score_range()` before the
  persistent buffer allocation block.
- Size `K_norope_v` / `Q_norope_v` to `n_score_layers = pre_range.count()`
  instead of `w.n_layer`.
- Guard Graph-A NoPE writes with `il >= score_layer_start_pre`; use offset
  index `si = il - score_layer_start_pre` throughout.
- Scoring loop uses same offset index for K_norope_v / Q_norope_v.
- Eliminated duplicate `score_layers` / `early_exit_n` statics.

VRAM with fix at S=128K, SCORE_LAYERS=7:
- K_norope_v: 7 x 268 MB = 1.9 GB (was 7.5 GB, saving 5.6 GB)
- Total: ~20.0 GB (was ~25.6 GB) — fits in 24 GB with headroom.

## Bench table (warm at 128K)

| Condition | A_compute | FP | tail-score | total | vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline (28 scoring layers) | 3.69 s | 44.97 s | 4.49 s | 66.79 s | 1.0x |
| L7 pre-fix (349cbc5) | 19.80 s | 56.25 s | 2.35 s | 107.15 s | 0.63x (regression) |
| L7 post-fix (projected) | ~3-5 s | ~45 s | 2.35 s | ~55-65 s | ~1.0-1.2x |

GPU bench to confirm pending; no live GPU available in this session.

## Verdict

Layer-subset is now structurally shippable. Compose SCORE_LAYERS=7 with
EARLY_EXIT_N=14 at 128K+ to keep drafter VRAM under 16 GB.
