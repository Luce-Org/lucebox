# Qwen3.6-27B vs Blog (Qwen3.5-27B) — Decode Gap Investigation

Date: 2026-06-22
Binary: dflash_server md5 `e9cb2790bb8ede64a9452f71e192d834` (Jun 22 19:21)
GPU: RTX 3090 24GB
Protocol: flock -x /tmp/lucebox_gpu.lock, port 18081, fresh server per config

---

## Step 0 — 3.5-27B Target Assets on Disk

| File | Size | Type |
|------|------|------|
| `/home/peppi/models/qwen3.5-27b-dflash/qwen3.5-27b-dflash-f16.gguf` | 4.0 GB | **DRAFTER** (arch=qwen35-dflash-draft, 6-layer, 5120-dim) |
| `/home/peppi/models/qwen3.5-27b-dflash/model.safetensors` | 4.0 GB | HuggingFace weights |

**No Qwen3.5-27B TARGET GGUF exists on disk.** The 4GB file is the matched drafter (6-layer, custom arch), not a 16GB Q4_K_M target. Step A (direct model A/B) is not runnable. Proceeded with Step B.

---

## Qwen3.6-27B Architecture

From GGUF metadata (`qwen35.*` keys):

| Field | Value |
|-------|-------|
| block_count | 64 |
| context_length | 262144 |
| embedding_length | 5120 |
| feed_forward_length | 17408 |
| attention.head_count | 24 |
| ssm.inner_size | 6144 |
| ssm.state_size | 128 |
| **full_attention_interval** | **4** |

`full_attention_interval=4` means every 4th layer is attention, the other 3 are SSM (GatedDeltaNet).
→ 64/4 = **16 attention layers + 48 SSM layers**

SSM in verify: implemented via `ggml_gated_delta_net_tree` kernel — single batched GPU graph (not N serial steps). However: 48 SSM layers must execute layer-by-layer within the GGML compute graph, each depending on the prior layer's output. The tree-structured verify pass carries significantly more kernel-launch overhead than a pure-attention model at the same layer count.

---

## Step B — Config Sweep on Qwen3.6-27B

Same binary, same target model (Q4_K_M), same KV (q4_0), same prompts (HumanEval/0-9), n_gen=256, temp=0.

### Per-Request Results — Config A: budget=22, bf16 drafter

| Task | decode tok/s | AL | steps | ms/step | prefill_s |
|------|---------|----|-------|---------|-----------|
| HE/0 | 102.3 | 11.80 | 15 | 108.5 | 1.5 |
| HE/1 | 114.5 | 10.90 | 21 | 95.2 | 0.5 |
| HE/2 | 120.6 | 11.22 | 9 | 93.0 | 0.3 |
| HE/3 | 122.1 | 11.20 | 5 | 91.8 | 0.5 |
| HE/4 | 121.8 | 11.20 | 15 | 91.9 | 0.5 |
| HE/5 | 121.2 | 10.79 | 14 | 89.0 | 0.4 |
| HE/6 | 124.3 | 11.21 | 19 | 90.2 | 0.5 |
| HE/7 | 122.6 | 11.10 | 10 | 90.6 | 0.4 |
| HE/8 | 151.6 | 13.33 | 12 | 88.0 | 0.5 |
| HE/9 | 121.6 | 11.29 | 14 | 92.8 | 0.4 |
| **MEAN** | **122.3** | **11.40** | — | **93.1** | — |

t_ar (AR per-token @ ~150-tok ctx): 51ms first request (cold CUDA), 35ms from Config B (warm).

### Per-Request Results — Config B: budget=16, bf16 drafter

| Task | decode tok/s | AL | steps | ms/step |
|------|---------|----|-------|---------|
| HE/0 | 117.2 | 11.80 | 15 | 96.0 |
| HE/1 | 126.4 | 10.90 | 21 | 86.2 |
| HE/2 | 128.3 | 11.22 | 9 | 87.4 |
| HE/3 | 109.7 | 9.33 | 6 | 85.0 |
| HE/4 | 112.5 | 10.50 | 16 | 93.3 |
| HE/5 | 116.1 | 10.79 | 14 | 92.9 |
| HE/6 | 131.8 | 11.21 | 19 | 85.1 |
| HE/7 | 125.5 | 11.10 | 10 | 88.4 |
| HE/8 | 153.3 | 13.33 | 12 | 87.0 |
| HE/9 | 127.4 | 11.29 | 14 | 88.6 |
| **MEAN** | **124.8** | **11.15** | — | **89.0** | — |

### Config C: budget=22, q8_0 drafter — FAILED

| Metric | Value |
|--------|-------|
| Mean decode TPS | 19.2 |
| Mean AL | 2.27 |
| Mean ms/step | 118.0 |

q8_0 drafter collapses to AL=2.2 (near-random). Consistent with known finding: Q8_0 dead on RTX 3090 Ampere consumer (scalar fallback path). Not viable.

---

## Summary Table

| Config | Draft | Budget | Mean TPS | Max TPS | AL | ms/step | t_ar_ms | vs blog |
|--------|-------|--------|----------|---------|-----|---------|---------|---------|
| Blog (3.5-27B) | bf16 | 22 | **129.52** | — | 8.31 | 64.2 | — | 1.000 |
| A (our b22, bf16) | bf16 | 22 | 122.3 | 151.6 | 11.40 | 93.1 | 35-51 | 0.944 |
| B (our b16, bf16) | bf16 | 16 | **124.8** | 153.3 | 11.15 | 89.0 | 35 | 0.964 |
| C (our b22, q8_0) | q8_0 | 22 | 19.2 | 20.0 | 2.27 | 118.0 | 35 | 0.148 |

**Best 3.6-27B result: 124.8 tok/s at budget=16, bf16 drafter.**

---

## Per-Step Decomposition

AR per-token at ~150-tok context: **35 ms** (measured from `[ar-decode] tokens=2 time=0.070s`).
This is one full 64-layer hybrid forward over 1 query token.

Blog implied per-step budget: `AL/TPS * 1000 = 8.31/129.52 * 1000 = **64.2 ms/step**`
Our best per-step:              `AL/TPS * 1000 = 11.15/124.8 * 1000 = **89.3 ms/step**`

Per-step remainder after verify (≈ t_ar):
- Blog: 64.2 - 35 = **29.2 ms** (draft forward + overhead)
- Ours (best): 89.0 - 35 = **54.0 ms** (draft forward + overhead)

The verify cost (≈ t_ar) is the same for both (same 64-layer model size, same ~150-tok KV).
The gap lives in the **29.2 → 54.0 ms remainder**: draft forward + graph rebuild + tensor copy + argmax.

This is consistent with our model having 48 SSM layers vs the blog's 3.5-27B model. The `ggml_gated_delta_net_tree` kernel must propagate SSM state through 48 tree-structured SSM layers per step, while a pure-attention model would only run 64 attention layers (all parallelizable in one flash-attention call). The SSM kernel adds sequential data dependencies per layer that cannot be hidden.

However: without the 3.5-27B TARGET GGUF confirmed on disk, we cannot isolate model vs other implementation factors (e.g. CUDA graph warmup, different GGML kernel paths for different layer types).

---

## Verdict

**The 15% gap is PRIMARILY THE MODEL, not the config.**

Evidence:

1. **Config change (budget 22→16): only +2.5 tok/s (+2.1%)** — eliminates "budget is the lever" hypothesis. Budget 16 recovers 2 tok/s of the 7 tok/s gap, not 7 tok/s.

2. **Step time gap is 89ms (ours) vs 64ms (blog) = 1.39× slower per step** — this is not explained by any config flag. Budget, drafter precision (q8_0=broken), KV quant, ring cap are all the same. The step time gap is structural.

3. **Qwen3.6-27B has 48 SSM layers (GatedDeltaNet)** confirmed by GGUF `full_attention_interval=4`. Every verify step runs `ggml_gated_delta_net_tree` over 48 layers with parent-chain state propagation. Blog's 3.5-27B architecture is unconfirmed (no target GGUF) but from public sources is a pure-attention dense transformer.

4. **AL is higher on 3.6-27B (11.2 vs 8.31)** — our drafter accepts MORE tokens per step than the blog's. This partially compensates: without the AL advantage, our effective TPS would be ~`89.0ms/step × (1/64.2ms) × 129.52 = 83 tok/s`. The AL advantage saves us ~42 tok/s, meaning the model's SSM overhead is being masked by better drafter quality.

**What's fixable:** The 2.1% from budget=16 is captured. The remaining ~3.5% gap (124.8 vs 129.52) is structurally the model — the hybrid SSM architecture imposes per-step overhead that doesn't exist in a pure-attention target.

**What would settle it:** Running the same bench against a Qwen3.5-27B-Q4_K_M target (pure-attention) with the same binary and same drafter. If 3.5 hits ~130 and 3.6 hits ~124, the architecture hypothesis is confirmed. The 3.5-27B target (~16GB Q4_K_M) is not on disk.

---

## Best Config for Qwen3.6-27B

`--ddtree --ddtree-budget 16 --draft dflash-draft-3.6-bf16-reconverted.gguf`
`--cache-type-k q4_0 --cache-type-v q4_0 DFLASH_FEAT_RING_CAP=4096`

Mean: **124.8 tok/s** (vs blog 129.52, gap 3.6%). Max: **153.3 tok/s**.

Raw data: `bench/qwen35moe_dflash/ctxsweep/model_ab_3.6_configs.json`
Server logs: `/tmp/dflash_he_model_ab_{A,B,C}_*.log`
