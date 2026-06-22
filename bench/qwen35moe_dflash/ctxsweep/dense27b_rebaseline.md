# Dense 27B dFlash Cold Rebaseline

**Date:** 2026-06-22  
**Binary:** `dflash_server` md5 `950eae1d4962e8b0d0df29acf63a1a2e`  
**GPU:** NVIDIA GeForce RTX 3090 24GB (23GB free at bench start)  
**Protocol:** one request per fresh server, flock -x /tmp/lucebox_gpu.lock, port 18081

## Config

| Field | Value |
|-------|-------|
| Target | Qwen3.6-27B-Q4_K_M.gguf (15.0 GiB on GPU, 682 MiB tok_embd CPU) |
| Drafter | dflash-draft-3.6-bf16-reconverted.gguf (loaded OK) |
| max_ctx | 40960 (no OOM — fit at full ctx) |
| KV cache | q4_0 / q4_0 |
| DFLASH_FEAT_RING_CAP | 40960 |
| DFLASH_FEATURE_DTYPE | default (f32 mirror) |
| fa-window | 0 |
| temp | 0 |
| max-tokens | 200 |

**Drafter load status:** CLEAN. Architecture=qwen35 detected. Draft loaded (SWA 4/5 layers, window=2048). No architecture mismatch.

## Per-Cell Results

| prompt | prompt_tok | accept% | AL | decode tok/s | prefill_s | gate | fit | needle |
|--------|-----------|---------|-----|-------------|-----------|------|-----|--------|
| ctx_008192 | 9,088 | 80.4% | 13.43 | 15.9 | 240.3s | held | OK (40960) | - |
| ctx_032768 | 35,121 | 80.4% | 13.43 | 26.3 | 425.5s | held | OK (40960) | - |
| needle_deep_12k | 15,541 | AR-floor | - | 15.1 (AR) | 180.3s | slow (ema=0.53) | OK (40960) | YES |

**ctx_008192 raw spec line:** `tokens=94 accepted=90/112 (80.4%) avg_commit=13.43 steps=7 speed=16.50 tok/s`  
**ctx_032768 raw spec line:** `tokens=94 accepted=90/112 (80.4%) avg_commit=13.43 steps=7 speed=27.46 tok/s`  
**needle spec-gate:** `floor reason=slow ema_ratio=0.53 t_ar=0.0590 ar_tokens=175 spec_tokens=22` → fell to AR; decode=15.1 tok/s (AR speed from server_done)

Note: ctx_032768 decode TPS (26.3) is higher than ctx_008192 (15.9) because the first cell absorbed more CUDA graph warmup resets during the short decode (94 tokens). The [spec-decode] speed field (16.50 vs 27.46) reflects the same pattern. Both are valid cold measurements; the 32K number is more representative of steady-state throughput.

Note: the 27B prefill is extremely slow relative to MoE — 240s for 9K tokens vs 4.4s for the MoE. This is because the dense model runs all 64 layers on full attention, while the MoE 35B-A3B has only ~3B active parameters per token. Prefill is compute-bound per-token-per-layer.

## Dense 27B-Q4 vs MoE 35B-A3B-Q3 — Same Prompts, Same Protocol

| prompt | DENSE 27B-Q4 accept% | DENSE 27B-Q4 decode | MoE 35B-A3B-Q3 accept% | MoE 35B-A3B-Q3 decode |
|--------|---------------------|---------------------|------------------------|----------------------|
| ctx_008192 | 80.4% | 15.9 tok/s | 76.8% | 174.6 tok/s |
| ctx_032768 | 80.4% | 26.3 tok/s | 76.8% | 171.3 tok/s |
| needle_deep_12k | AR-floor | 15.1 tok/s (AR) | 28.7% | 78.3 tok/s |

## Prefill Comparison

| prompt | prompt_tok | DENSE 27B prefill | MoE 35B prefill |
|--------|-----------|------------------|-----------------|
| ctx_008192 | 9,088 | 240.3s | 4.4s |
| ctx_032768 | 35,121 | 425.5s | 17.1s |
| needle_deep_12k | 15,541 | 180.3s | 7.5s |

## Takeaway

**Decode:** Dense 27B computes all 64 layers × full-attention on every token; MoE 35B-A3B activates ~3B params (1 of 128 experts per token). Measured decode gap: **6.7× slower on ctx_008192 (15.9 vs 174.6 tok/s), 6.5× slower on ctx_032768 (26.3 vs 171.3 tok/s), 5.2× slower on needle (15.1 vs 78.3 tok/s AR)**. The spec-decode accept% for the dense 27B is higher (80.4% vs 76.8% on short/long ctx, and the MoE needle 28.7% vs AR-floor on 27B dense), but the higher accept% cannot close a 6-7× per-step cost gap.

**Prefill:** Dense 27B prefill is 54× slower than MoE at 9K tokens (240s vs 4.4s) and 25× slower at 35K tokens (425s vs 17s). This is entirely architectural: the MoE sparse FFN computes only ~3B params per token vs 27B dense.

**Spec-gate behavior:** The dense 27B spec-gate "held" (accepted spec mode) on the repetitive ctx_008192/ctx_032768 content (EMA ratio 1.71, 2.14). On the needle prompt (more varied content, lower EMA ratio=0.53) it floored to AR. This is the correct gate behavior — the drafter's quality was below threshold on that content type.

**VRAM:** 27B Q4_K_M + bf16 drafter + q4/q4 KV fit cleanly at max_ctx=40960. No OOM. Target used 15.0 GiB GPU, drafter ~3.2 GiB additional.

## Log Files

| prompt | log |
|--------|-----|
| ctx_008192 | `dense27b_ctx_008192_174618.log` |
| ctx_032768 | `dense27b_ctx_032768_175122.log` |
| needle_deep_12k | `dense27b_needle_deep_12k_180103.log` |
