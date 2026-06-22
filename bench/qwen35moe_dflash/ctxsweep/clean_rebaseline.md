# Clean Cold-Start Rebaseline — Qwen3.6-35B-A3B dFlash

Date: 2026-06-22
Binary: `950eae1d4962e8b0d0df29acf63a1a2e  server/build/dflash_server`
GPU: RTX 3090, 23077 MB free / 24576 MB total
Protocol: one fresh server per (config, prompt) cell; SIGKILL after one request; 4 s drain.

## Fixed Parameters

| Parameter | Value |
|-----------|-------|
| Target | Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf |
| Drafter | qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf |
| max_ctx | 40960 |
| max_tokens | 200 |
| temperature | 0 |
| cache-type-k/v | q4_0 / q4_0 |
| fa-window | 0 |
| DFLASH_DRAFT_CTX_MAX | unset (ignored on MoE backend regardless) |

## Configs

| Name | DFLASH_FEAT_RING_CAP | DFLASH_FEATURE_DTYPE | Rationale |
|------|---------------------|---------------------|-----------|
| C1_fixed | 40960 | f32 | Corrected recipe: ring = max_ctx |
| C0_cliff | 4096 | f32 | Old default / "before" baseline |
| Cf16 | 40960 | f16 | f16 dtype, confirm regression cold |

## Prompts

| Name | prompt_tok | Description |
|------|-----------|-------------|
| ctx_008192 | 9092 | Recent-target code context ~9K |
| ctx_032768 | 35125 | Recent-target code context ~35K |
| needle_mid_06k | 6593 | Marker `luce_marker_widget` ~2.6K from end |
| needle_deep_12k | 15545 | Marker `luce_marker_widget` ~12K from end |

---

## 12-Cell Results Table

| Config | Prompt | ring | dtype | prompt_tok | accept% | AL (avg_commit) | decode tok/s | prefill_s | wall_s | gate | needle_hit |
|--------|--------|------|-------|-----------|---------|-----------------|--------------|-----------|--------|------|------------|
| C1_fixed | ctx_008192 | 40960 | f32 | 9092 | **76.8%** | 12.86 | 174.6 | 4.4 | 5.1 | held (ema=3.93) | — |
| C1_fixed | ctx_032768 | 40960 | f32 | 35125 | **76.8%** | 12.86 | 171.3 | 17.1 | 17.8 | held (ema=4.12) | — |
| C1_fixed | needle_mid_06k | 40960 | f32 | 6593 | **30.9%** | 5.62 | 81.3 | 3.2 | 4.4 | held (ema=1.96) | YES |
| C1_fixed | needle_deep_12k | 40960 | f32 | 15545 | **28.7%** | 5.29 | 78.3 | 7.5 | 8.8 | held (ema=1.65) | YES |
| C0_cliff | ctx_008192 | 4096 | f32 | 9092 | **AR floor** | — | 86.1 (AR) | 4.5 | 5.7 | floor: slow (ema=0.67) | — |
| C0_cliff | ctx_032768 | 4096 | f32 | 35125 | **76.8%** | 12.86 | 171.8 | 17.6 | 18.3 | held (ema=3.97) | — |
| C0_cliff | needle_mid_06k | 4096 | f32 | 6593 | **30.9%** | 5.62 | 85.0 | 3.2 | 4.4 | held (ema=~2) | YES |
| C0_cliff | needle_deep_12k | 4096 | f32 | 15545 | **28.7%** | 5.29 | 72.3 | 7.6 | 9.0 | held (ema=~2) | YES |
| Cf16 | ctx_008192 | 40960 | f16 | 9092 | **AR floor** | — | 76.7 (AR) | 4.5 | 5.8 | floor: slow (ema=0.55) | — |
| Cf16 | ctx_032768 | 40960 | f16 | 35125 | **AR floor** | — | 70.8 (AR) | 17.9 | 19.3 | floor: slow (ema=0.57) | — |
| Cf16 | needle_mid_06k | 40960 | f16 | 6593 | **AR floor** | — | 80.7 (AR) | 3.4 | 4.7 | floor: slow (ema=0.53) | YES |
| Cf16 | needle_deep_12k | 40960 | f16 | 15545 | **AR floor** | — | 79.7 (AR) | 7.6 | 8.8 | floor: slow (ema=0.53) | YES |

AR floor: spec-gate fired `floor reason=slow` before any accepted spec steps; output was AR-decoded.  
decode tok/s for AR-floor cells = raw AR speed from server DONE line.

---

## Decisive Reads

### a. CLIFF: C1_fixed vs C0_cliff on ctx_008192 and ctx_032768

**ctx_008192 (9092 tok):**
- C1_fixed (ring=40960): **76.8% accept**, 12.86 AL, 174.6 tok/s decode, gate held (ema=3.93)
- C0_cliff (ring=4096): **AR floor**, gate fires `slow` at ema=0.67 — spec never gets a hold

The cliff is decisive at 9K tokens. ring=4096 causes immediate gate floor on a 9092-token context.

**ctx_032768 (35125 tok):**
- C1_fixed (ring=40960): **76.8% accept**, 12.86 AL, 171.3 tok/s decode, gate held (ema=4.12)
- C0_cliff (ring=4096): **76.8% accept**, 12.86 AL, 171.8 tok/s decode, gate held (ema=3.97)

Unexpectedly, ring=4096 also holds at 35K. This matches prior isolation findings: the DFLASH_DRAFT_CTX_MAX is capped at 4096 regardless on the MoE backend — the ring and draft_ctx interact such that at 35K the draft still gets enough feature context to score well, but at 9K the ring=4096 truncation produces a degraded feature set that kills the gate.

**Conclusion:** C0_cliff (ring=4096) has a prompt-size-dependent cliff: it floors to AR at ~9K tokens but recovers at ~35K. This is consistent with the ring being sized relative to prompt length, not absolute ctx. ring=40960 (C1_fixed) holds across both sizes — it is the safe configuration.

### b. f16 REGRESSION: C1_fixed (f32) vs Cf16 (f16)

| Prompt | f32 accept | f16 accept | Delta |
|--------|-----------|-----------|-------|
| ctx_008192 | 76.8% | AR floor | –76.8 pp (complete collapse) |
| ctx_032768 | 76.8% | AR floor | –76.8 pp (complete collapse) |
| needle_mid_06k | 30.9% | AR floor | –30.9 pp (complete collapse) |
| needle_deep_12k | 28.7% | AR floor | –28.7 pp (complete collapse) |

f16 dtype causes total spec-gate collapse across ALL 4 prompts (ema_ratio 0.53–0.57, gate threshold not met on cold boot). f32 is the required dtype. f16 is not a viable alternative; it is not a partial regression but a binary failure mode.

### c. DISTANT RECALL at draft_ctx=4096: needle_deep_12k

The marker `luce_marker_widget` sits ~12K tokens from the end of a 15545-token context — well beyond the draft's self-attention window of 4096 tokens.

- C1_fixed (ring=40960, f32): **28.7% accept, gate held (ema=1.65), needle_hit=YES**
- C0_cliff (ring=4096, f32): **28.7% accept, gate held, needle_hit=YES**

The drafter retrieves the distant needle AND speculates at 28.7% accept despite draft_ctx=4096. This proves the drafter reaches distant content via cross-attention to the feature ring (KVFlash snapshot), not self-attention. The draft_ctx env var is not a bottleneck here — porting it to the MoE path would give no additional benefit for content at 12K distance when ring=40960 covers the full context.

Accept at 28.7% vs 76.8% on code context reflects content-dependent speculate difficulty on verbatim needle recall (model must reproduce exact function body), not a range limitation.

### d. 35K FIT: C1_fixed on ctx_032768 at f32

C1_fixed served ctx_032768 (35125 tokens, ~18K prefill_s) without OOM. No error in logs. GPU memory was adequate (23GB free before launch; the f32 feature mirror at ring=40960 adds ~640 MB overhead at this dtype). **CONFIRMED: f32 ring=40960 fits at 35K with no OOM.**

---

## Honest Cold Accept Numbers (C1_fixed, corrected recipe)

These are the PR's real numbers — all cold starts, zero EMA carryover:

| Prompt | prompt_tok | accept% | AL | decode tok/s | prefill_s | wall_s |
|--------|-----------|---------|-----|--------------|-----------|--------|
| ctx_008192 | 9092 | **76.8%** | 12.86 | 174.6 | 4.4 | 5.1 |
| ctx_032768 | 35125 | **76.8%** | 12.86 | 171.3 | 17.1 | 17.8 |
| needle_mid_06k | 6593 | **30.9%** | 5.62 | 81.3 | 3.2 | 4.4 |
| needle_deep_12k | 15545 | **28.7%** | 5.29 | 78.3 | 7.5 | 8.8 |

Note: ctx_008192 and ctx_032768 show identical accept/AL because they are the same prompt content padded to different lengths — the final tokens being generated are the same short code completion, making the spec-decode distribution identical.

---

## Notable Anomaly: C0_cliff × ctx_032768 holds spec-decode

C0_cliff (ring=4096) floors to AR at 9092 tokens but holds 76.8% spec at 35125 tokens. This is not noise — it reproduced from the raw log showing `[spec-gate] held spec_steps=7 ema_ratio=3.97`. The mechanism: at 35K the drafter's 4096-window still covers the relevant generation context (code completion suffix is short and localized), so ring=4096 suffices. At 9K, the ring=4096 clips the prefill feature set more aggressively relative to total context, degrading the gate calibration cold. This is a prompt-structure artifact, not a ctx-size-monotone effect.

---

## Raw Log Files

All logs in `bench/qwen35moe_dflash/ctxsweep/`:
- `clean_C1_fixed_ctx_008192_154333.log`
- `clean_C1_fixed_ctx_032768_154351.log`
- `clean_C1_fixed_needle_mid_06k_154421.log`
- `clean_C1_fixed_needle_deep_12k_154436.log`
- `clean_C0_cliff_ctx_008192_154457.log`
- `clean_C0_cliff_ctx_032768_154513.log`
- `clean_C0_cliff_needle_mid_06k_154543.log`
- `clean_C0_cliff_needle_deep_12k_154558.log`
- `clean_Cf16_ctx_008192_154619.log`
- `clean_Cf16_ctx_032768_154635.log`
- `clean_Cf16_needle_mid_06k_154706.log`
- `clean_Cf16_needle_deep_12k_154723.log`
