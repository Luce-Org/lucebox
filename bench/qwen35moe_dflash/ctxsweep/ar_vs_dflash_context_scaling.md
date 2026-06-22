# AR-baseline vs dFlash spec-decode — Context Scaling (Dense 27B & MoE 35B-A3B)

**Fabrication-free.** Every number below carries a `file:line` or `jq path` provenance.
Build: git `42278139ce85`, branch `pr/kvflash-moe-prefill-snapshot`, binary sha256
`717778c8…7160e047` (agentic runs) / md5 `950eae1d4962e8b0d0df29acf63a1a2e` (cold rebaselines).
GPU: RTX 3090 24 GB. temp=0.

## How to read "BEFORE" vs "AFTER"

dFlash spec-decode is **gated**. On novel/varied content the gate floors to **AR** (the
"before" autoregressive baseline); on copy/structured content it **holds spec** (the
"after"). Many logs therefore contain BOTH a `[spec-decode]` (held) and an `[ar-decode]`
(floored) row. Both are reported honestly below. `decode_mode` = `ar` is the BEFORE; `spec`
is the AFTER. A floored cell means dFlash gave **no decode benefit** on that content type —
it correctly fell back to AR rather than wasting draft cycles.

**Provenance caveat (carried from `agentic_bestconfig_dense_vs_moe.md:12-15`):** the
`*_smoke_raw.json` `provenance.cache_type_k/v` field reads `tq3_0` for both arms — this is a
harness-template metadata bug. The **actual** KV loaded by the binary is q4_0 (27B,
`srv_BENCH_27B_20260622_100403.log:48-49`) and f16 (35B,
`srv_BENCH_35B_20260622_101002.log:63-64`), matching the arm defs
(`replay_harness.py:415,435`). Tables use the real values.

---

## TABLE 1 — DENSE 27B (Qwen3.6-27B-Q4_K_M) — AR vs dFlash by context tier

Drafter `dflash-draft-3.6-bf16-reconverted.gguf` `--draft-swa 2048`. Cold = fresh server,
one request, prefix_len=0. Warm = prefix-cache restore=true (agentic replay).

| context (tok) | scenario | AR decode tok/s | dFlash decode tok/s (accept%) | prefill_s | gate verdict | source |
|---|---|---|---|---|---|---|
| 9,088 (cold code) | cold | — | **15.9 (80.4%)** spec held | 240.3 | held ema=1.71 | `dense27b_ctx_008192_174618.log:74,78` |
| 35,121 (cold code) | cold | — | **26.3 (80.4%)** spec held | 425.5 | held ema=2.14 | `dense27b_ctx_032768_175122.log:74,78` |
| 15,541 (cold needle) | cold | **15.1 (AR)** | floored → no spec | 180.3 | floor slow ema=0.53 | `dense27b_needle_deep_12k_180103.log:73,77,81` |
| 34,135 (warm agentic t1) | cold t1 | **15.8 (AR)** | floored t1 | 53.0 | ar | `BENCH_27B_…100403_smoke_raw.json` per_turn_median t1 |
| 34,965 (warm agentic t2) | warm restore | — | **14.8 (38.6%)** spec held | 3.6 | spec | per_turn_median t2 |
| 37,699 (warm agentic t3) | warm restore | **11.5 (AR)** | floored t3 | 12.5 | ar | per_turn_median t3 |

Notes:
- The cold code cells (9K/35K) HOLD spec at 80.4% accept because the goldgate code content is
  repetitive (the 9K and 35K prompts are the same content padded; final completion identical →
  identical spec distribution, `dense27b_rebaseline.md:36`).
- The needle (novel verbatim recall) FLOORS to AR — this is the honest "before" on varied content.
- 35K cold decode (26.3) > 9K cold decode (15.9): the short 94-token decode at 9K absorbed CUDA-graph
  warmup resets; 35K is more representative of steady state (`dense27b_rebaseline.md:36`).
- **Prefill is brutal on dense 27B**: 240s @ 9K, 425s @ 35K cold — all 64 layers full-attention,
  compute-bound (`dense27b_rebaseline.md:38`). Warm prefix-cache collapses this to 3.6s @ turn-2.

---

## TABLE 2 — MoE 35B-A3B (Qwen3.6-35B-A3B-UD-Q3_K_XL, all-hot) — AR vs dFlash by context tier

Drafter `qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf` `--draft-swa 2048`.

| context (tok) | scenario | KV | AR decode tok/s | dFlash decode tok/s (accept%) | prefill_s | gate | source |
|---|---|---|---|---|---|---|---|
| 9,092 (cold code) | cold | q4_0 | — | **174.6 (76.8%)** spec held | 4.4 | held ema=3.93 | `clean_rebaseline.md:44` |
| 35,125 (cold code) | cold | q4_0 | — | **171.3 (76.8%)** spec held | 17.1 | held ema=4.12 | `clean_rebaseline.md:45` |
| 6,593 (cold needle) | cold | q4_0 | — | **81.3 (30.9%)** spec held | 3.2 | held ema=1.96 | `clean_rebaseline.md:46` |
| 15,545 (cold needle) | cold | q4_0 | — | **78.3 (28.7%)** spec held | 7.5 | held ema=1.65 | `clean_rebaseline.md:47` |
| 9,092 (cold code) | cold | **f16** | **76.7 (AR)** | floored | 4.5 | floor ema=0.55 | `clean_rebaseline.md:52` |
| 35,125 (cold code) | cold | **f16** | **70.8 (AR)** | floored | 17.9 | floor ema=0.57 | `clean_rebaseline.md:53` |
| 34,135 (warm agentic t1) | cold t1 | f16 | **64.7 (AR)** | floored t1 | 23.4 | ar | `BENCH_35B_…101002` per_turn_median t1 |
| 34,965 (warm agentic t2) | warm restore | f16 | **80.7 (AR)** | floored t2 | 0.8 | ar | per_turn_median t2 |
| 37,699 (warm agentic t3) | warm restore | f16 | **55.3 (AR)** | floored t3 | 2.9 | ar | per_turn_median t3 |
| 34,135 (warm agentic t2 spec arm) | warm restore | f16 | — | **66.4 (25.0%)** spec held | — | spec | `agentic_f16_40k.log:17` |

Notes:
- **f16 KV cold-floors spec across the board** (`clean_rebaseline.md:80-89`): –76.8 pp collapse
  vs f32-feature-mirror config. f32 feature dtype is required for cold-boot spec to hold; f16 is a
  binary failure (this is the *feature mirror dtype*, distinct from KV precision).
- **On MoE, spec is roughly decode-neutral**: f16 AR decode is already 55-80 tok/s, so the f16_40k
  spec arm (66.4 @ 25% accept) ≈ the AR numbers. The fastest MoE numbers don't even need spec
  (`agentic_bestconfig_dense_vs_moe.md:70-72`).
- q4_0 KV ≈ f16 KV on accept/AL and is a free VRAM saver (`NOTES.md:47-52`): f16 174 / q4_0 167 /
  q8_0 143 (anomalous low accept 66.4) / tq3_0 109 tok/s @ 35K.

---

## TABLE 3 — Long-context scaling: does dFlash hold to 128K+?

The only ≥64K agentic run is `agentic_kvflash_131k.log` (MoE 35B, max_ctx=131072, `--kvflash auto`
**forced on** because max_ctx exceeds all-hot fit). Compare to the 34K all-hot run (Table 2).

| max_ctx | turn | prompt_tok | restore | prefill_s (tps) | decode tok/s | mode | accept% | source |
|---|---|---|---|---|---|---|---|---|
| 40,960 (all-hot) | t1 | 34,135 | false (cold) | 23.4 (1458.8) | 64.7 | ar | — | Table 2 / per_turn_median |
| 40,960 (all-hot) | t2 | 34,965 | **true (warm)** | **0.8** (43706) | 80.7 | ar | — | per_turn_median t2 |
| **131,072** (kvflash) | t1 | 34,135 | false | 49.7 (686.8) | **20.6** | **spec** | **22.3** | `agentic_kvflash_131k.log:46,80,83` |
| **131,072** (kvflash) | t2 | 34,965 | **false (MISS)** | **77.6** (450.6) | 29.2 | ar | — | `agentic_kvflash_131k.log:47,90,95` |
| **131,072** (kvflash) | t3 | 37,699 | — | — | — | ERROR conn closed | `agentic_kvflash_131k.log:18,48` |

**Verdict on 128K scaling (HONEST, the run FAILED its smoke gate):**
- At 131K the kvflash 16384-token pool **evicts** (`agentic_kvflash_131k.log:72,88` "pooled prefill
  … through a 16384-token pool, evicting"), which **breaks prefix-cache restore** — every turn is
  `restore=false / prefix_len=0` (`:71,87,99`). Turn-2 warm prefill that was 0.8s all-hot becomes a
  full re-prefill of **77.6s**. The gate FAILED ("cache reuse not visible", `:40`) and the server
  closed the connection on turn 3.
- Decode at 131K spec held is **20.6 tok/s @ 22.3% accept** vs 64.7-80.7 tok/s at 34K all-hot — a
  **3-4× decode regression** from pool eviction + larger-ctx attention, NOT a spec-decode win.
- This corroborates `NOTES.md:53-56`: **KVFlash trades prefix-caching + decode speed for unbounded
  context.** For 34K coding sessions that fit in q4_0 KV all-hot and NEED caching, KVFlash is
  net-negative. It is only justified for 128K-256K that physically won't fit all-hot — and even then
  loses warm-restore on this all-hot MoE path.
- **No clean warm-cache result exists above 40,960 max_ctx for either model.**

---

## TABLE 4 — Headline BEFORE→AFTER at representative agentic ctx (~34K), real novel content

| model | metric | BEFORE (AR) | AFTER (dFlash) | net effect on real agentic |
|---|---|---|---|---|
| **DENSE 27B** | warm-turn decode | 11.5-15.8 (AR, t1/t3 floored) | 14.8 @ 38.6% (t2 held) | **TIES / marginal** — gate holds only on the cache-hit turn; floors on novel t1/t3. Mean 14.03 tok/s (`arm_aggregate`). spec_engagement 0.333. |
| **DENSE 27B** | warm prefill (cache) | 53.0s cold t1 | 3.6s warm t2 | **14.7× prefill win** — but this is prefix-cache, not spec-decode |
| **MoE 35B-A3B** | warm-turn decode | 55.3-80.7 (AR) | 66.4 @ 25% (spec arm) | **FLOORS to AR / neutral** — best MoE arm is pure AR (spec_engagement 0.0, `arm_aggregate`); f16 AR already 55-80 tok/s so spec adds nothing |
| **MoE 35B-A3B** | warm prefill (cache) | 23.4s cold t1 | 0.8s warm t2 | **29× prefill win** — prefix-cache, not spec |
| **DENSE vs MoE** | cold decode @ 9K/35K | — | 15.9/26.3 vs 174.6/171.3 | MoE **6.5-6.7× faster decode** (A3B ~3B active vs 27B dense, `dense27b_rebaseline.md:58`) |
| **DENSE vs MoE** | cold prefill @ 9K/35K | — | 240/425s vs 4.4/17s | MoE **25-54× faster prefill** (sparse FFN, `dense27b_rebaseline.md:60`) |

### The honest one-line conclusions
1. **dFlash spec-decode net-helps on COPY/structured content, ties-or-floors on NOVEL content.**
   On goldgate code (repetitive) it holds 76-80% accept; on needle/novel-turn content it floors to
   AR. On real multi-turn agentic the gate engages on ≤1 of 3 turns.
2. **The big agentic win is PREFIX-CACHE prefill (15-29×), not spec-decode.** Warm-turn prefill
   drops from 23-53s to 0.8-3.6s. This is the lever for the user's "short prompt on large growing
   context" workload — decode tok/s barely moves; cache-served prefill is what matters.
3. **MoE 35B-A3B dominates dense 27B for this workload**: 6-7× decode, 25-54× prefill, at the cost
   of Q3 quant. Dense 27B prefill (240-425s cold) is impractical for interactive agentic without a
   warm cache.

---

## MISSING tiers (what should be run)

- **Dense 27B at 64K and 131K**: NO dense long-context run exists at all. Only MoE has a 131K log
  (and it failed its gate). A dense 27B 131K run would almost certainly be unserviceable on 24 GB
  (15 GB target + drafter + 131K KV), but it is untested — MISSING.
- **Dense 27B agentic at f16 KV with per-turn data**: `agentic_27b_f16.log` is config-header-only
  (truncated). Dense vs MoE decode gap is confounded by KV format (dense=q4_0, MoE=f16). MISSING for
  a like-for-like KV comparison (`agentic_bestconfig_dense_vs_moe.md:115-120`).
- **A clean warm-cache run above 40,960 max_ctx for either model**: the 131K kvflash path destroys
  restore. No config currently delivers BOTH 128K context AND warm prefix-cache on this all-hot MoE
  path — MISSING and is the real open problem for 128K+ agentic.
- **MoE 35B 64K / 98K tiers**: only 34K (all-hot) and 131K (kvflash) exist; the 40K→131K cliff in
  decode (80→20 tok/s) is unsampled in between — MISSING the 64K/96K data points to locate where
  pool-eviction starts hurting.
