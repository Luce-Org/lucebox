# Qwen3.6-27B prefix_cache regression — bragi — 2026-05-31

Empirical test of `prefix_cache_slots=32` (server default) on Qwen3.6-27B,
quantifying the known-reliability concern from `autotune.py`.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
* **Model**: Qwen3.6-27B-Q4_K_M, optimal config + prefix_cache_slots=32
* **Image**: `dc20057e-cuda12`

## Result: -19pp regression on agent_recorded

| metric | prefix_cache=0 (baseline) | prefix_cache=32 | delta |
|--------|--------------------------|-----------------|-------|
| agent_recorded | **42.3% (11/26)** | 23.1% (6/26) | **-19pp** |

The 6 passing cases with prefix_cache=32 are a strict subset of the 11
baseline passes. No new cases were unlocked; 5 previously-passing cases
regressed.

**Conclusion: `prefix_cache_slots` must stay at 0 for Qwen3.6 tool-use
workloads.** Reverted to 0 immediately after confirming the regression.

## Root cause (autotune.py comment)

```
Prefix cache remains an explicit sweep tunable, but the automatic baseline
keeps it off because tool prompts currently exercise a daemon snapshot path
that is not reliable with prefix slots enabled.
```

The regression confirms this comment. The prefix cache's KV snapshot path
does not correctly handle tool-calling conversations: it likely restores a
cached KV state at a turn boundary that doesn't account for the tool call
context, causing the model to lose track of prior tool invocations.

The word "currently" in the comment suggests this is a known bug (not a
fundamental limitation), but it is not fixed in dc20057e-cuda12.

## Pass/fail detail

| case | prefix=0 | prefix=32 |
|------|----------|-----------|
| 1 | PASS | FAIL (regression) |
| 2 | PASS | PASS |
| 3 | PASS | FAIL (regression) |
| 4 | FAIL | FAIL |
| 5 | PASS | PASS |
| 6 | FAIL | FAIL |
| 7–11 | FAIL | FAIL |
| 12 | PASS | PASS |
| 13 | PASS | FAIL (regression) |
| 14–18 | FAIL | FAIL |
| 17 | PASS | FAIL (regression) |
| 19 | PASS | PASS |
| 20–21 | FAIL | FAIL |
| 22 | PASS | PASS |
| 23 | FAIL | FAIL |
| 24 | PASS | PASS |
| 25–26 | FAIL | FAIL |
| 9 | PASS | FAIL (regression) |

5 regressions (cases 1, 3, 9, 13, 17), 0 new passes.

## Smoke test result

Smoke (3 simple arithmetic/factual cases) passed 100% with prefix_cache=32.
The reliability issue is specific to multi-turn tool conversations, not
single-turn generation.

## Speed note

The speed benefit of prefix caching would be significant for multi-turn
conversations (reusing KV for prior turns instead of re-prefilling from
scratch). At 24–25 tok/s, a 3000-token prior context costs ~2.5s prefill
per turn. With a working prefix cache, subsequent turns would skip that cost.

The performance upside is real — the fix needs to land in the server before
prefix caching can be safely enabled for tool-use workloads.

## Recommendation

Keep `prefix_cache_slots = 0` until the daemon snapshot path bug is fixed.
When the fix lands, re-run this test (agent_recorded nothink) to verify
recovery to the 42.3% baseline before enabling for production use.
