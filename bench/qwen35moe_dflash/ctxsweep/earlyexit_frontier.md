> ⚠️ SUPERSEDED (2026-06-22): these runs varied `DFLASH_DRAFT_CTX_MAX` (IGNORED on the MoE path — draft_ctx is pinned at 4096) and/or used sequential requests (warm-EMA inflated accept). Treat the draft_ctx columns and any >77% accept as artifacts. Authoritative cold numbers: `clean_rebaseline.md`. The surviving real levers are `FEAT_RING_CAP=max_ctx` and f32 mirror.

# Early-Exit Frontier Benchmark

**Binary:** `server/build/dflash_server`  
**md5:** `950eae1d4962e8b0d0df29acf63a1a2e`  
**Date:** 2026-06-22  
**Config:** Q3_K_XL target, BF16 drafter, port 18081, q4_0 KV, fa-window=0, max-ctx=40960, max-tokens=200, DFLASH_FEATURE_DTYPE=f16, DFLASH_FEAT_RING_CAP=40960

Drafter has 6 layers: [SWA,SWA,SWA,SWA,SWA,FULL]. EARLY_EXIT_N=K runs layers 0..K-1. N=5 skips the FULL attention layer (layer 5). N=0 or unset = all 6 layers.

---

## Q1: DRAFT_CTX_MAX=8192

| N (layers) | prompt     | accept% | avg_commit (AL) | decode tok/s | recalled | gate_floor |
|------------|------------|---------|-----------------|--------------|----------|------------|
| 0 (all 6)  | ctx_8192   |   6.2%  | ---             |  70.5        | Y        | slow       |
| 0 (all 6)  | needle_06k |  76.8%  | 12.86           |  89.8        | Y        | -          |
| 5          | ctx_8192   |   6.2%  | ---             |  68.3        | Y        | slow       |
| 5          | needle_06k |   7.8%  | ---             |  51.2        | Y        | slow       |
| 4          | ctx_8192   |   6.2%  | ---             |  90.8        | N        | slow       |
| 4          | needle_06k |   7.8%  | ---             |  74.4        | Y        | slow       |
| 3          | ctx_8192   |   7.8%  | ---             |  91.5        | N        | slow       |
| 3          | needle_06k |   6.2%  | ---             |  77.5        | Y        | slow       |

**Gate floor** = spec-gate fell back to AR-only for the rest of the request after the initial warm-up period.  
**AL = ---** in floored cases means the held-spec block never executed (no avg_commit logged).  
**recalled N** on ctx_8192 at N=4 and N=3: the model output the marker but with wrong surroundings (content quality degraded).

### Q1 Verdict

No N delivers a decode tok/s gain over the N=0 baseline at dc=8192 while keeping the spec gate open.

- N=0 on needle_06k is the only case where the spec gate HELD: accept=76.8%, AL=12.86, 89.8 tok/s.
- N=5 on needle_06k: gate floors (reason=slow), drops to 51.2 tok/s — a 43% regression vs N=0.
- N=4 and N=3 on needle_06k: gate floors, 74–78 tok/s — small gains over the N=0 floored-gate path on ctx_8192 but the gate-held N=0 needle_06k at 89.8 tok/s is still better.
- ctx_8192 gates floor for ALL N including N=0 (the drafter produces low-quality drafts on this code content); early exit makes this worse or neutral.
- The AR-only decode speed when floored (the token generation after the gate floors) is ~91-118 tok/s raw; the reported chat-DONE decode speed is slower because it amortises the spec probe overhead.

**Early exit at dc=8192 is NOT worth shipping.** The spec gate is the governing constraint — early exit degrades draft quality faster than it saves drafter latency, resulting in earlier gate floors that wipe the spec-decode benefit.

---

## Q2: DRAFT_CTX_MAX=16384

| N (layers) | dc    | prompt     | accept% | AL    | decode tok/s | fit | gate_floor | deep-marker recalled |
|------------|-------|------------|---------|-------|--------------|-----|------------|----------------------|
| 0 (all 6)  | 16384 | needle_12k |  42.2%  | ---   |  47.8        | OK  | slow       | Y                    |
| 0 (all 6)  | 16384 | ctx_8192   |  92.7%  | 14.83 | 109.2        | OK  | -          | Y                    |
| 5          | 16384 | needle_12k |   7.8%  | ---   |  57.6        | OK  | slow       | Y                    |
| 5          | 16384 | ctx_8192   |   7.8%  | ---   |  71.1        | OK  | slow       | Y                    |
| 4          | 16384 | needle_12k |   7.8%  | ---   |  56.6        | OK  | slow       | Y                    |
| 4          | 16384 | ctx_8192   |   6.2%  | ---   |  71.7        | OK  | slow       | Y                    |
| 3          | 16384 | needle_12k |   7.8%  | ---   |  58.3        | OK  | slow       | Y                    |
| 3          | 16384 | ctx_8192   |   6.2%  | ---   |  71.8        | OK  | slow       | Y                    |

**No OOM at dc=16384 for any N.** All cells loaded and served.

### Q2 Observations

**N=0 (all 6 layers) at dc=16384 — the baseline:**
- needle_12k: gate floors after ~140 AR tokens + 57 spec tokens (ema_ratio=0.90, just below hold threshold). Accept=42.2% and marker RECALLED — the full attention layer is accessing context ~12K tokens back. This confirms the architectural prediction: you need layer 5 (full-attention) AND a large draft_ctx to get deep recall.
- ctx_8192: gate HELD with accept=92.7%, AL=14.83, 109.2 tok/s — the best single result in the grid.

**N=5 at dc=16384 — skip the full layer:**
- needle_12k: accept collapses to 7.8%, gate floors immediately (ema_ratio=0.35). Despite dc=16384, the SWA-only drafter sees only the last 4096 tokens — exactly the architectural prediction. The marker is still RECALLED in the response because the *target model* can still do deep attention; the drafter just can't speculate about what it will say.
- ctx_8192: same collapse. Floored at same ema_ratio.

**N=4 and N=3:** Identical collapse pattern to N=5. Removing more layers does not recover quality and provides no additional speed benefit — the gate floors so fast that drafter speed is irrelevant.

**Architectural prediction confirmed:** At N≤5 (skip layer 5), dc=16384 gives zero benefit over dc=4096 on deep recall. The SWA window caps out at 4096 regardless of dc. The extra VRAM allocation for dc=16384 is wasted.

**dc=16384 + N=0 is NOT a win either:** N=0 at dc=16384 gives a partial spec benefit on deep-ctx content (42.2% accept vs the prior work baseline of ~33.8%), but the gate still floors, and the decode speed at 47.8 tok/s is worse than N=0 at dc=8192 needle_06k (89.8 tok/s) because the larger draft context slows the drafter itself.

### Q2 Verdict

No N makes dc=16384 a real deep-recall win in the sense required. N=0 partially uses the larger window (42.2% vs 33.8% baseline) but the gate still floors and decode speed drops. N>0 collapses immediately — skipping the full attention layer destroys deep-context speculation, confirming the architectural prediction. The extra VRAM for dc=16384 + N>0 is pure waste.

---

## Bottom Line

**Early exit is NOT worth shipping at any N or draft_ctx combination tested.**

The governing constraint is the spec-gate floor (reason=slow), not drafter latency. Early exit saves drafter forward time but produces worse drafts, causing earlier gate floors. The one configuration where the gate stays open (N=0, dc=8192, recent-target content) already has optimal throughput at 89.8 tok/s with AL=12.86. All early-exit variants either match floored-AR performance or regress it. Draft context expansion to 16384 is orthogonal to early exit and provides marginal gain only under N=0 with fresh/structured content — a different question from this benchmark.

---

## Raw Log Files

- `earlyexit_q1_N0_ctx_8192.log` — Q1 N=0 all6 ctx_8192
- `earlyexit_q1_N0_needle_06k.log` — Q1 N=0 all6 needle_06k
- `earlyexit_q1_N5_ctx_8192.log` — Q1 N=5 ctx_8192
- `earlyexit_q1_N5_needle_06k.log` — Q1 N=5 needle_06k
- `earlyexit_q1_N4_ctx_8192.log` — Q1 N=4 ctx_8192
- `earlyexit_q1_N4_needle_06k.log` — Q1 N=4 needle_06k (orphan server, same env)
- `earlyexit_q1_N3_server.log` — Q1 N=3 both prompts (chatcmpl_0 = ctx_8192, chatcmpl_1 = needle_06k)
- `earlyexit_q2_N0_server.log` — Q2 N=0 both prompts
- `earlyexit_q2_N5_server.log` — Q2 N=5 both prompts
- `earlyexit_q2_N4_server.log` — Q2 N=4 both prompts
- `earlyexit_q2_N3_server.log` — Q2 N=3 both prompts
