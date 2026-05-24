# ee N-sweep NIAH: baseline / ee3 / ee5 / ee7 @ 32K / 64K / 128K

## Results

| ctx | condition | drafter_p50 | tail_score | NIAH | speedup_vs_baseline |
|---|---|---|---|---|---|
| 32768 | baseline | 5.240s | 0.840s | 3/3 | 1.00x |
| 65536 | baseline | 10.660s | 1.630s | 3/3 | 1.00x |
| 131072 | baseline | 69.520s | 14.850s | 3/3 | 1.00x |
| 32768 | ee3 | 0.760s | 0.100s | 3/3 | 6.89x |
| 65536 | ee3 | 1.400s | 0.170s | 3/3 | 7.61x |
| 131072 | ee3 | 2.860s | 0.350s | 3/3 | 24.31x |
| 32768 | ee5 | 1.110s | 0.150s | 3/3 | 4.72x |
| 65536 | ee5 | 2.120s | 0.280s | 3/3 | 5.03x |
| 131072 | ee5 | 6.540s | 2.740s | 3/3 | 10.63x |
| 32768 | ee7 | 1.480s | 0.220s | 3/3 | 3.54x |
| 65536 | ee7 | 2.810s | 0.400s | 3/3 | 3.79x |
| 131072 | ee7 | 8.420s | 3.180s | 3/3 | 8.26x |

## Pivot table (by context)

| ctx | baseline NIAH | ee3 NIAH | ee5 NIAH | ee7 NIAH | baseline drafter | ee3 drafter | ee5 drafter | ee7 drafter |
|-----|---------------|----------|----------|----------|------------------|-------------|-------------|-------------|
| 32K | 3/3 | 3/3 | 3/3 | 3/3 | 5.240s | 0.760s | 1.110s | 1.480s |
| 64K | 3/3 | 3/3 | 3/3 | 3/3 | 10.660s | 1.400s | 2.120s | 2.810s |
| 128K | 3/3 | 3/3 | 3/3 | 3/3 | 69.520s | 2.860s | 6.540s | 8.420s |

## Verdict

Gate criteria per PLAN.md: smallest N where vs ee7 — NIAH within ±1 needle, drafter_wall ≤ ee7 wall, zero crashes.

**ee3 passes all NIAH gates: 3/3 at every context.** Drafter wall is 6.9x/7.6x/24.3x vs baseline and strictly faster than ee5 and ee7 at every context. Zero ggml_view_3d asserts, zero server OOM.

Decision: **ee3 is the NIAH gate winner.** Pending multi-client accept_rate validation (within ±2 pp of ee7 mean across 5 clients).

