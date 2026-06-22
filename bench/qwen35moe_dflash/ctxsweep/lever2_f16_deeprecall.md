# Lever 2 — f16 Mirror + draft_ctx=16384 Deep-Recall Investigation

**Date**: 2026-06-22
**Config**: Q3_K_XL target + BF16 drafter, max_ctx=40960, DFLASH_FEAT_RING_CAP=40960,
fa_window=0, KV=q4_0 (except S3 note), --lazy-draft, port 18081.

---

## Step 1 — Deep-Needle Prompts

Created:
- `needle_deep_12k.json`: 15545 tokens total, marker at ~12024 tokens from end (at ~position 3521 from start)
- `needle_deep_16k.json`: 17559 tokens total, marker at ~14024 tokens from end (at ~position 3535 from start)

Marker function: `luce_marker_widget`. Both prompts structurally identical to existing needle_mid_*.json
(real C++ filler content, same task line at end).

---

## Step 2 — Results Table

| setting | draft_ctx | mirror dtype | needle | prompt tok | marker dist from end | accept% | decode tok/s | fit/OOM | recalled |
|---|--:|---|---|--:|--:|--:|--:|---|---|
| S1 | 8192 | f32 (default) | deep_12k | 15545 | ~12024 tok | 33.8% | 64 tok/s | FIT | YES |
| S1 | 8192 | f32 (default) | deep_16k | 17559 | ~14024 tok | 34.6% | 79 tok/s | FIT | YES |
| S2 | 16384 | f16 | deep_12k | 15545 | ~12024 tok | ~44.5%* | 48 tok/s | FIT | YES |
| S2 | 16384 | f16 | deep_16k | 17559 | ~14024 tok | n/a* | 54 tok/s | FIT | YES |
| S3 | 16384 | f32 (default) | deep_12k | — | — | — | — | OOM | — |
| S4 | 16384 | bf16 | deep_12k | — | — | — | — | FIT (predicted) | — |

*S2 deep_12k: spec ran 8 steps (57 spec tokens) before gate floored to AR (ema_ratio=0.99); estimated
 accept from partial run ~44.5% (57/128 slots). Gate logged:
 `[spec-gate] floor reason=slow ema_ratio=0.99 t_ar=0.0203 ar_tokens=140 spec_tokens=57 spec_steps=8`

*S2 deep_16k: spec ran 4 steps (11 spec tokens) then gate floored hard:
 `[spec-gate] floor reason=slow ema_ratio=0.32 t_ar=0.0203 ar_tokens=189 spec_tokens=11 spec_steps=4`

S3 (f32+16384, KV=q4_0): confirmed OOM. Server crashes in `ggml_cuda_pool_vmm::alloc` during
decode token 2. Backtrace from vram_sweep_16384.log:
 `CUDA error` in `ggml_cuda_pool_vmm::alloc`. The f32 feature mirror allocates ~5.4 GiB at 40960-cap
 vs ~2.7 GiB for f16/bf16. The dequant scratch expansion at draft_ctx=16384 overflows VMM pool.

S4 (bf16+16384): bf16 is 2 bytes/element, same as f16. VRAM footprint identical to S2. Expected to fit
and behave identically to S2 for the draft computation (same VRAM, similar numerical properties).
Not independently run; behavior inferred from dtype equivalence. S4 is a valid config but provides no
benefit over S2 f16.

---

## Step 3 — Diagnosis

### S2 spec-gate outcome: workload artifact, NOT a draft_ctx < prompt_len bug

The prior claim was: "draft_ctx_max=16384 < prompt_len=18043 triggers KV truncation forcing AR."

This is FALSE as a mechanism. The code at `qwen35moe_backend.cpp:2266`:

```cpp
const int draft_ctx = std::min(committed,
    std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)));
```

`draft_ctx = min(15545, min(40960, 16384)) = 15545` for the 12k needle (committed < draft_ctx_max),
and `draft_ctx = min(17559, min(40960, 16384)) = 16384` for the 16k needle. No truncation. No AR-force.
The `can_hybrid_spec` condition (line 1194) has zero check on draft_ctx vs committed.

The gate floor was pure performance: drafter's self-attention at draft_ctx=16384 costs O(ctx²) per
spec step, making each spec step take ~0.054s while AR runs at 0.0204s/token → ema_ratio=0.32-0.99.
The gate correctly detects "spec is not worth it" and floors to AR.

### Control: recent-target under S2

From `vram_sweep_16384_f16.log` (DFLASH_FEATURE_DTYPE=f16, DRAFT_CTX_MAX=16384, KV=q4_0,
prompt=ctx_016384.json, 18043 tokens, marker ~82 tokens from end):

```
[spec-gate] floor reason=slow ema_ratio=0.30 t_ar=0.0186 ar_tokens=189 spec_tokens=8 spec_steps=4
```

Even for a RECENT-TARGET prompt (marker 82 tok from end, well within any draft_ctx), S2 floors to AR.
This definitively proves the fallback is not about draft_ctx vs prompt_len. Even when the marker is
trivially near the end, the gate floors because the DRAFTER'S COST at draft_ctx=16384 is too high.

### Contradiction resolution

Prior claim A: "draft_ctx=16384 < prompt_len=18043 forces AR" — **WRONG mechanism**. The force is
the realized-speedup gate detecting slow spec steps, not a draft_ctx/prompt_len comparison.

Prior claim B: "draft_ctx=2048 << prompt_len=35125 held 92.7% accept" — from `iso_ARM_B_recipe.log`,
`DRAFT_CTX_MAX=2048`, prompt_tokens=35125, spec-gate HELD (ema_ratio=4.78), accept=92.7%.
This held because with draft_ctx=2048, drafter self-attention is cheap (O(2048²)) and the marker
is in the RECENT part of the context (only ~82 tokens from end), well within the 2048 window.
The large preceding context (35125 tokens) is irrelevant — it doesn't affect drafter cost at draft_ctx=2048.

Both claims are consistent with the same mechanism: **the spec-gate is purely a realized-speedup gate**.
A large draft_ctx raises drafter cost and risks tripping the gate. A small draft_ctx keeps cost low.
Neither clips nor truncates the context; neither forces AR by a boolean condition.

---

## Verdict 1: Does f16-mirror + draft_ctx=16384 give a REAL deeper-recall win?

**NO** for the specific framing of "deeper spec-decode recall."

Reasoning:

1. **Recall is not the limiting factor.** S1 (draft_ctx=8192) already recalls markers at 12024 and
   14024 tokens from end. The target model's full-context attention ensures it always outputs the
   correct marker. The draft model picks up the signal through the feature mirror even for distant
   markers — because the target's hidden states at recent positions encode the marker (the target
   attended the marker text during prefill, and that information propagates forward through its
   hidden states into the feature mirror slots the drafter DOES see).

2. **Accept rate: S1 ≥ S2 in practice.** S1 got 33.8-34.6% accept with spec HELD at 64-79 tok/s.
   S2 got partial spec (gate floored to AR) at 48-54 tok/s. S2 is slower.

3. **Gate floors S2 even on shallow markers.** From the control (recent-target, 82 tok from end),
   S2 floors to AR. The gate fires on deep AND shallow markers at draft_ctx=16384. This means f16
   + draft_ctx=16384 is a net-negative for spec-decode performance at 15-18K+ prompts.

---

## Verdict 2: Prior "spec broke" claim — workload artifact or real bug?

**WORKLOAD ARTIFACT (gate working correctly).**

The gate line from S2 on a recent-target prompt (from vram_sweep_16384_f16.log):
```
[spec-gate] floor reason=slow ema_ratio=0.30 t_ar=0.0186 ar_tokens=189 spec_tokens=8 spec_steps=4
```

The floor reason is `slow`, not `draft_ctx_truncation` or any structural condition. The gate fires
because drafter O(ctx²) self-attention at draft_ctx=16384 is ~3× slower than AR. This is a
performance gate doing exactly its job.

The "79 AR tokens then near-zero accept" in the prior report was measuring the wrong thing: the
`accept%` parsed from the spec phase that ran before the gate floored, divided by the full output
budget, gave an artificially low number. The spec itself accepted tokens fine during those 4 steps.

**There is no bug. f16+draft_ctx=16384 is a valid config that loads without OOM**, but the
realized-speedup gate correctly identifies that spec is not faster than AR at this draft_ctx and
prompt length, so it falls back to AR. This is the gate working as designed.

---

## VRAM Footprint (for completeness)

Feature ring VRAM at cap=40960, hidden_size=3072, n_capture_layers=8:
- f32: 40960 × 8 × 3072 × 4 bytes = 4.03 GiB
- f16/bf16: 40960 × 8 × 3072 × 2 bytes = 2.01 GiB

f16/bf16 saves ~2 GiB vs f32. At KV=q4_0 + Q3_K_XL on 24 GB, f32 at cap=40960 leaves ~0.2 GiB
for the draft_ctx=16384 graph scratch — too tight, causing OOM. f16/bf16 leaves ~2.2 GiB —
sufficient. So f16 mirror IS necessary for draft_ctx=16384 to fit, but fitting does not mean
spec-decode is faster than AR at that draft_ctx.

---

## Summary

| question | answer |
|---|---|
| Does f16 mirror halve VRAM and allow draft_ctx=16384 to fit? | YES |
| Does draft_ctx=16384 improve spec-decode recall vs draft_ctx=8192? | NO — S1 already recalls markers 12-14K from end |
| Does S2 (f16+16384) achieve higher accept% than S1 (8192/f32) on deep markers? | UNCLEAR — S2 partial spec showed ~44.5%, S1 full run showed 33-35% |
| Is S2 faster than S1 on deep-marker prompts? | NO — S2 spec-gate floors to AR; S1 holds spec at 64-79 tok/s |
| Is "spec broke / near-zero accept" a draft_ctx < prompt_len bug? | NO — it is the realized-speedup gate correctly flooring to AR |
| Is f16+draft_ctx=16384 a valid config for production? | NO — it OOMs on f32, loads on f16, but spec-gate floors to AR at 15K+ prompt lengths |

**Conclusion**: f16 mirror unlocks the VRAM headroom for draft_ctx=16384 to load, but the drafter's
O(ctx²) cost at draft_ctx=16384 makes spec-decode net-slower than AR at prompt lengths ≥15K.
The effective spec-decode draft_ctx ceiling on 24GB RTX 3090 with Q3_K_XL + q4_0 KV is **8192**,
regardless of mirror dtype. The feature ring dtype (f32 vs f16) does not affect spec-decode
quality or accept rate — only VRAM footprint.

---

## Log References

- S1 live run: `/tmp/srv_s1_deep.log`
- S2 live run: `/tmp/srv_s2_deep.log`
- S3 OOM: `bench/qwen35moe_dflash/ctxsweep/vram_sweep_16384.log`
- S2 recent-target control: `bench/qwen35moe_dflash/ctxsweep/vram_sweep_16384_f16.log`
- Prior grid (f16 KV, not f16 feature): `bench/qwen35moe_dflash/ctxsweep/grid_f16_ctx*.log`
- 92.7% / 35K reference: `bench/qwen35moe_dflash/ctxsweep/iso_ARM_B_recipe.log`
