# Drafter Forward Profile — 128K (S=131080)

**Run date**: 2026-05-21  
**Binary**: `/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/build/dflash_server`  
**Model**: Qwen3-0.6B-BF16, 28 layers  
**Headline**: `[drafter] forward+score in 225.04s S=131080`

## Per-Stage Breakdown

The aggregate forward log line is the authoritative source. Per-layer lines are only emitted for layer 1 and layer 28 (first/last), so p50/p95 per layer are not available from this binary — the table uses cumulative totals across all 28 layers.

| Stage | Sum across 28 layers (s) | % of drafter total (223.02s) | Notes |
|---|---|---|---|
| A_setup | 0.01 | 0.0% | QKV proj graph construction |
| A_alloc | 0.00 | 0.0% | |
| A_compute | 17.36 | 7.8% | QKV projections + RoPE, chunked |
| FP | 33.32 | 14.9% | FlashPrefill body attention |
| B_warm | 0.00 | 0.0% | |
| B_setup | 0.00 | 0.0% | |
| B_alloc | 0.00 | 0.0% | |
| B_copy_in | 0.00 | 0.0% | |
| B_norm | 0.00 | 0.0% | |
| B_compute | 0.00 | 0.0% | FFN (graph-B not executed in pflash mode) |
| B_copy_out | 0.00 | 0.0% | |
| embed + untracked overhead | 74.47 | 33.4% | Embedding + per-layer graph alloc/sync, not in named stages |
| tail-score | 97.86 | 43.9% | Per-layer Q@K scoring pass (28x full-S attention) |
| **Total drafter wall** | **223.02** | **100%** | t_fwd=125.16s + tail-score=97.86s |

## Verdict

**tail-score dominates at 43.9%. FP is 14.9%. B_compute is 0% (graph-B not executed in pflash mode). Neither FP nor B_compute is the primary bottleneck — the tail-score pass is.**

## Interpretation

The dominant cost (97.86s, 43.9%) is the tail-score loop: a second full pass over all 28 layers computing Q@K attention to identify which KV positions to keep. This runs entirely separately from the forward pass and executes a full S-length attention per layer.

FP (FlashPrefill body attention) is 33.32s (14.9%) — not negligible but not dominant. A_compute (QKV projections + RoPE) is 17.36s (7.8%). Critically, B_compute=0 — in pflash mode graph-B (FFN) does not execute; the drafter only needs attention weights, not full hidden states.

The "embed+overhead" bucket (74.47s, 33.4%) is the gap between named stage timers and t_fwd wall-clock. It includes: token embedding get_rows, per-layer ggml graph construction and galloc, and GPU sync bubbles between chunked subgraphs. This is substantial and unoptimised.

**The correct attack for drafter speedup is reducing tail-score cost, not FP or FFN.**
- Option C (K-only fast path for FP) would cut 33s, not 98s — wrong target
- Tier 1 (Q8 + layer-subset) reduces A_compute and overhead but not the tail-score bottleneck directly
- High-impact options: subset of scoring layers (every 2nd/4th), fuse score into forward pass, block-sparse score attention, or reduce n_lookahead
