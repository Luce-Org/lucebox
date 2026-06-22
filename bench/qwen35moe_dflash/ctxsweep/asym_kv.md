# Asymmetric KV Precision Sweep — Qwen3.6-35B-A3B dFlash

**Binary:** `dflash_server` md5=`950eae1d4962e8b0d0df29acf63a1a2e`
**Model:** `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`
**Drafter:** `qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf` (modal drafter, all layers — EARLY_EXIT unset)
**Fixed env:** `DFLASH_FEAT_RING_CAP=40960 DFLASH_FEATURE_DTYPE=f16 DFLASH_DRAFT_CTX_MAX=8192`
**Fixed flags:** `--max-ctx 40960 --fa-window 0 --max-tokens 200 --lazy-draft`, `temperature=0`

## Results

### Prompt: ctx_008192 (8K code context, ~9092 tokens)

Note: all cells fell back to AR mode (spec-gate `floor reason=slow`, ema_ratio ≈0.42). The 8K code content is below the spec-gate's speedup threshold. Decode figures below reflect AR-mode speed under the dFlash feature overhead, not spec-decode.

| cache-type-k | cache-type-v | kv_cache est. | accept% | AL  | decode tok/s | prefill_s |
|:-------------|:-------------|:-------------|:--------|:----|:------------|:----------|
| q4_0         | q4_0         | 3.12 GiB*    | AR (no spec) | — | 67.7 | 4.3 |
| q8_0         | q4_0         | 3.12 GiB*    | AR (no spec) | — | 76.1 | 4.2 |
| f16          | q4_0         | 3.12 GiB*    | AR (no spec) | — | 73.7 | 4.3 |
| q8_0         | q8_0         | 3.12 GiB*    | AR (no spec) | — | 70.0 | 4.3 |
| f16          | f16          | 3.12 GiB*    | AR (no spec) | — | 76.4 | 4.2 |

*kv_cache estimate from placement planner, hardcoded f16 sizing regardless of actual KV type. Real runtime VRAM is lower for q4/q8.

### Prompt: needle_mid_06k (6K needle-in-haystack, ~6593 tokens)

All cells ran spec-decode. Accept and AL are identical across all 5 cells — the spec-gate engaged and performance is content-ceiling-limited, not KV-type-limited.

| cache-type-k | cache-type-v | kv_cache est. | accept% | AL    | decode tok/s | prefill_s |
|:-------------|:-------------|:-------------|:--------|:------|:------------|:----------|
| q4_0         | q4_0         | 3.12 GiB*    | 92.7%   | 14.83 | 116.8       | 3.0       |
| q8_0         | q4_0         | 3.12 GiB*    | 92.7%   | 14.83 | 127.9       | 3.1       |
| f16          | q4_0         | 3.12 GiB*    | 92.7%   | 14.83 | 122.6       | 3.1       |
| q8_0         | q8_0         | 3.12 GiB*    | 92.7%   | 14.83 | 127.3       | 3.0       |
| f16          | f16          | 3.12 GiB*    | 92.7%   | 14.83 | 133.5       | 3.1       |

## Theoretical VRAM (runtime, not planner estimate)

Model has 10 attention layers (30/40 are SSM, no KV). GQA: 8 KV heads, head_dim=128, max_ctx=40960.

| config         | K GiB | V GiB | K+V total |
|:---------------|:------|:------|:----------|
| q4_0 / q4_0   | 0.195 | 0.195 | 0.39 GiB  |
| q8_0 / q4_0   | 0.391 | 0.195 | 0.59 GiB  |
| f16  / q4_0   | 0.781 | 0.195 | 0.98 GiB  |
| q8_0 / q8_0   | 0.391 | 0.391 | 0.78 GiB  |
| f16  / f16    | 0.781 | 0.781 | 1.56 GiB  |

The planner always reserves 3.12 GiB (f16 upper bound) regardless of actual KV type — it does not vary expert placement across these cells.

## Verdicts

### 1. Does k=q8/v=q4 give higher accept/AL than symmetric q4/q4?

No detectable difference. Both cells show accept=92.7% / AL=14.83 on the needle prompt. On the ctx_008192 code prompt, spec-decode was gated off by the spec-floor (ema_ratio=0.42) in all cells — KV type had no influence on the gating decision.

**Accept/AL are identical across all 5 cells** at these context lengths (≤8K). KV quantization type does not move the needle on spec-decode quality at short-mid contexts where the content ceiling dominates.

### 2. Does k=q8/v=q4 avoid the symmetric-q8 accept regression?

The prior "symmetric q8 regression (~66% vs ~77% at f16)" was measured at longer contexts (reported at 32K+). At 6-8K context, all cells hit the content ceiling (92.7%). The regression hypothesis cannot be confirmed or denied at these context lengths — the ceiling masks any precision sensitivity.

At these short contexts: k=q8/v=q4 (127.9 tok/s) and k=q8/v=q8 (127.3 tok/s) are within noise. No asymmetric-K benefit observed.

### 3. Best accept-per-GiB config and whether asymmetric beats symmetric q4

All cells tie on accept/AL. The decode throughput ordering on needle (spec-decode active) is:

```
f16/f16: 133.5 > q8/q4: 127.9 ≈ q8/q8: 127.3 > f16/q4: 122.6 > q4/q4: 116.8 tok/s
```

Faster decode with higher KV precision is expected: f16 KV enables higher-fidelity attention score computation, yielding better accept (unseen here — ceiling-limited) and faster decode path. However, f16/f16 costs 4× the KV VRAM of q4/q4 (1.56 GiB vs 0.39 GiB) with no accept gain at these contexts.

**q4/q4 is the correct recipe** at ≤8K: lowest VRAM cost (0.39 GiB), no quality penalty vs higher precision types, decode ceiling is content-limited not KV-limited. Asymmetric k=q8/v=q4 (+10 tok/s over q4/q4 on needle) offers marginal throughput upside but at 50% more KV VRAM with no accept improvement.

## Caveats

- Both prompts are at ≤8K. The prior symmetric-q8 regression was observed at 32K+. A longer-context sweep is needed to see whether KV type affects accept at >16K contexts where attention becomes bandwidth-bound and quantization error compounds over more attended positions.
- The kv_cache GiB banner line from the server is a planner estimate (always f16-sized), not runtime VRAM usage. Actual savings are in GPU memory not reflected in the log.
- The spec-gate gated off spec-decode on ctx_008192 (AR fallback) for all cells, so that prompt cannot distinguish KV type effects on spec-decode quality.
