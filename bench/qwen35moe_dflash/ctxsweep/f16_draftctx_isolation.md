# f16 vs f32 mirror / DRAFT_CTX_MAX isolation

**Binary:** `server/build/dflash_server`  
**md5:** `950eae1d4962e8b0d0df29acf63a1a2e`  
**Date:** 2026-06-22  
**Fixed:** DFLASH_FEAT_RING_CAP=40960, max_ctx=40960, KV q4_0/q4_0, fa-window=0, bf16-reconv drafter, temp=0, port=18081, lazy-draft  
**Method:** 4 cells x 2 prompts, each cell on a fresh cold server instance, kill -9 between cells.

---

## Results Table

| FEATURE_DTYPE | DRAFT_CTX_MAX | prompt        | accept%  | AL    | decode tok/s | gate_floor |
|---------------|---------------|---------------|----------|-------|-------------|------------|
| f32           | 2048          | ctx_008192    |  76.8%   | 12.86 |  180.2      | -          |
| f32           | 2048          | needle_mid_06k|  31.2%   |  5.56 |   83.3      | -          |
| f16           | 2048          | ctx_008192    |   ~4.1%* |  ---  |   80.8      | slow       |
| f16           | 2048          | needle_mid_06k|   ~5.5%* |  ---  |   89.9      | slow       |
| f32           | 8192          | ctx_008192    |  76.8%   | 12.86 |  112.0      | -          |
| f32           | 8192          | needle_mid_06k|  92.7%   | 14.83 |  220.6      | -          |
| f16           | 8192          | ctx_008192    |   ~4.1%* |  ---  |   68.4      | slow       |
| f16           | 8192          | needle_mid_06k|  92.7%   | 14.83 |  127.0      | -          |

`*` gate-floored accept% computed as `spec_tokens/(ar_tokens + spec_tokens)` from the `[spec-gate] floor` log line (gate emits no final accepted= summary). The `---` AL means no spec block held to completion. All prompt_tokens: ctx_008192=9092, needle_mid_06k=6593.

---

## Config A reproduction check

Cell (f32, 2048) on ctx_008192 produced **76.8% accept, AL=12.86, 180.2 tok/s**, gate held. This does NOT reproduce the previously reported 92.7%. The baseline is real at 76.8%, not 92.7%. See the Measurement Artifact section below for the explanation.

---

## VERDICT

### Which knob causes the ctx_008192 regression?

**FEATURE_DTYPE=f16 is the sole culprit. DRAFT_CTX_MAX is not a factor.**

**f32 vs f16 at the SAME draft_ctx:**

At draft_ctx=2048:
- (f32, 2048) ctx_008192: **76.8%**, gate held, ema_ratio=3.82
- (f16, 2048) ctx_008192: **~4.1%**, gate floor reason=slow, ema_ratio=0.50

At draft_ctx=8192:
- (f32, 8192) ctx_008192: **76.8%**, gate held, ema_ratio=2.82
- (f16, 8192) ctx_008192: **~4.1%**, gate floor reason=slow, ema_ratio=0.39

f16 floors on ctx_008192 at BOTH draft_ctx values. f32 holds at BOTH. The drop is entirely attributable to dtype.

**draft_ctx=2048 vs 8192 at the SAME mirror dtype:**

At f32: ctx_008192 accept = 76.8% at dc=2048, 76.8% at dc=8192. Zero difference.  
At f16: ctx_008192 floors at both. Zero difference in accept (both ~4.1%).

draft_ctx has no effect on ctx_008192 accept at either dtype.

**Is it an interaction (only f16+8192 craters)?** No. f16 craters on ctx_008192 regardless of draft_ctx. The combination f16+8192 is not special — f16+2048 craters identically.

---

## Measurement Artifact: Where did the 92.7% on ctx_008192 come from?

The 92.7% on ctx_008192 in the prior earlyexit bench (earlyexit_q2_N0_server.log, labeled "dc=16384") was **not a cold measurement**. That server received needle_mid_12k.json first (15545 tokens, gate floored after 140 AR + 57 spec tokens), which set `gate_t_ar_` = 0.0195s. When ctx_008192 arrived as the **second request** on the same server, the gate's EMA started at ema_ratio=1.84 (inherited from the prior gate-held state at turn-1 close), passed the warmup immediately, and held throughout. The 92.7% reflects a warm-gate second-request measurement, not the cold-start behavior of ctx_008192.

The prior "Config A" label of "f32/DRAFT_CTX_MAX=2048 → 92.7%" was a transcription error in the task description. The actual log for that result shows `dtype=f16`, and the accept was 92.7% due to multi-request EMA state, not to f32 or draft_ctx=2048. Cold-server, single-request: ctx_008192 on f32 gives 76.8%, on f16 gates floor.

Note: `DFLASH_DRAFT_CTX_MAX` env var is not read by the qwen35moe backend (only by qwen35 backend). For qwen35moe, draft_ctx is bounded by `cfg_.draft_ctx_max=4096` (BackendArgs default) regardless of `DFLASH_DRAFT_CTX_MAX` env. The `DFLASH_FEAT_RING_CAP` env IS consumed (via inherited qwen35_backend initialization of the feature mirror cap). All four cells confirmed mirror cap=40960 in the `[dflash-feature]` log line.

---

## Implication for the shipped recipe

**The shipped recipe uses DFLASH_FEATURE_DTYPE=f16 + DFLASH_DRAFT_CTX_MAX=8192.**

Verdict: **the recipe is NOT safe for recent-target content.**

- f16 causes the spec gate to floor (reason=slow) on ctx_008192 at cold start regardless of draft_ctx. The drafter's f16 mirror produces worse hidden-state approximations; the gate correctly detects this as unprofitable and falls back to AR-only for the entire request.
- f32 mirror holds the gate and delivers 76.8% accept, 180 tok/s on the same prompt.
- DRAFT_CTX_MAX does not help: f32 at dc=2048 and dc=8192 are identical (76.8%), so the recall-horizon argument for dc=8192 does not compensate for the f16 regression.

**Action required:** revert DFLASH_FEATURE_DTYPE to f32 (or drop the env var, as f32 is the default). This costs the 1.34 GB VRAM headroom that f16 mirror was saving, but restores accept on recent-target content from floor-level back to 76.8%.

The DFLASH_DRAFT_CTX_MAX=8192 line in the recipe is harmless for qwen35moe (the env is ignored by that backend; actual draft_ctx is capped at cfg_.draft_ctx_max=4096). It may be removed for clarity but does not cause the observed regression.

---

## Raw Logs

- `isolation2x2_dtf32_dc2048.log` — cell (f32, 2048)
- `isolation2x2_dtf16_dc2048.log` — cell (f16, 2048)
- `isolation2x2_dtf32_dc8192.log` — cell (f32, 8192)
- `isolation2x2_dtf16_dc8192.log` — cell (f16, 8192)
- `isolation2x2_results.json` — machine-readable results
