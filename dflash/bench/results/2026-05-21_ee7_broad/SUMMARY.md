# ee7 Broad Context (1K-16K) + Agentic-Coding Validation — 2026-05-21

Binary: `dflash/build/dflash_server` (rebuilt after alloc-loop bug fix)
GPU: RTX 3090 24 GiB, TQ3_0 KV, Qwen3.6-27B Q4_K_M + Qwen3-0.6B-BF16 drafter
Branch: feat/pflash-drafter-fastpath

## Bug Found and Fixed During Validation

The warm-path commit (f157274) introduced a K_norope_v alloc-loop bounds bug:
the guard `il >= score_layer_start_pre` lacked the upper bound `il < fwd_layer_limit_pre`,
causing out-of-bounds vector access for any condition where n_score_layers < n_layer.
Fix: add `&& il < fwd_layer_limit_pre` to the alloc loop guard.
T7 (3 subcases) added to test_drafter_warm_path_regression to cover this boundary.

## Pass A: NIAH Broad Context (1K-16K)

| ctx | condition | drafter_fwd_p50 | tail_score | NIAH | speedup_vs_baseline |
|---|---|---|---|---|---|
| 1024 | baseline | 0.310s | 0.060s | 1/3 | 1.00x |
| 4096 | baseline | 0.770s | 0.130s | 1/3 | 1.00x |
| 8192 | baseline | 1.340s | 0.220s | 2/3 | 1.00x |
| 16384 | baseline | 2.530s | 0.415s | 2/3 | 1.00x |
| 1024 | ee14 | 0.210s | 0.040s | 1/3 | 1.48x |
| 4096 | ee14 | 0.440s | 0.080s | 1/3 | 1.75x |
| 8192 | ee14 | 0.745s | 0.125s | 2/3 | 1.80x |
| 16384 | ee14 | 1.360s | 0.215s | 2/3 | 1.86x |
| 1024 | ee7 | 0.170s | 0.030s | 1/3 | 1.82x |
| 4096 | ee7 | 0.290s | 0.050s | 1/3 | 2.66x |
| 8192 | ee7 | 0.460s | 0.080s | 2/3 | 2.91x |
| 16384 | ee7 | 0.800s | 0.120s | 2/3 | 3.16x |

## Pass B: Agentic-Coding Harness (Claude Code client)

| condition | drafter_fwd | accept_rate | OK_DONE | speedup_vs_baseline |
|---|---|---|---|---|
| baseline | 1.72s | 30.6% (88/288) | YES | 1.00x |
| ee14 | 0.93s | 32.6% (99/304) | YES | 1.85x |
| ee7 | 0.56s | 41.7% (80/192) | YES | 3.07x |

## Verdict

ee7 is equivalent to ee14 on NIAH at every context tested and preserves agentic OK_DONE.
Speedup over ee14: 1.82x/2.66x/2.91x/3.16x at 1K/4K/8K/16K vs ee14 1.48x/1.75x/1.80x/1.86x.
Pass B agentic: 3.07x vs ee14 1.85x.
Recommendation: promote ee7 to production default for RTX 3090.
