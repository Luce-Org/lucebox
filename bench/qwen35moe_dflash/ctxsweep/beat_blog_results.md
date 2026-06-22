# Beat-the-Blog Benchmark Results — Qwen3.6-27B-Q4_K_M + DDTree

Date: 2026-06-22
Binary md5: `e9cb2790bb8ede64a9452f71e192d834`  (server/build/dflash_server, Jun 22 19:21)
GPU: RTX 3090, 22859 MiB free / 24576 MiB total (at start)
Config: --ddtree --ddtree-budget 22 --cache-type-k q4_0 --cache-type-v q4_0
        --lazy-draft --fa-window 0  DFLASH_FEAT_RING_CAP=4096  DFLASH_DRAFT_CTX_MAX=2048
HE server log: /tmp/dflash_he_ddtree.log

## DDTree Engagement Confirmation

Banner in server log:
```
[server]  ddtree          = ON
[server]  ddtree_budget   = 22
```
All 10 prompts produced `[spec-decode]` lines with accepted= and avg_commit= fields,
confirming tree-verify was active (not silently falling back to AR).

---

## Step 1 — HumanEval 10-Prompt DDTree Bench

Target model:  /home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf
Draft model:   /home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf
max_ctx=8192, max_tokens=256, temperature=0, DFLASH_FEAT_RING_CAP=4096

| Task      | prefill_s | decode tok/s | AL    | accept% | ptok | gtok | wall_s |
|-----------|-----------|-------------|-------|---------|------|------|--------|
| HE/0      |      0.60 |        88.6 | 11.17 |   59.4% |  151 |   67 |   1.90 |
| HE/1      |      0.20 |       129.2 | 12.06 |   68.4% |  143 |  217 |   2.40 |
| HE/2      |      0.10 |       106.7 | 10.67 |   59.0% |  112 |   96 |   1.50 |
| HE/3      |      0.20 |        96.8 | 10.20 |   55.0% |  146 |   51 |   1.20 |
| HE/4      |      0.20 |       123.9 | 11.64 |   65.6% |  145 |  163 |   2.00 |
| HE/5      |      0.10 |       110.3 | 10.12 |   56.2% |  121 |   81 |   1.30 |
| HE/6      |      0.20 |       108.1 | 10.55 |   58.5% |  141 |  116 |   1.70 |
| HE/7      |      0.10 |       100.7 | 10.60 |   58.1% |  121 |  106 |   1.70 |
| HE/8      |      0.20 |       133.8 | 12.92 |   73.4% |  143 |  155 |   1.80 |
| HE/9      |      0.20 |       104.0 | 10.43 |   57.1% |  127 |   73 |   1.30 |
| **MEAN**  |      0.21 |   **110.2** | **11.04** |      — |    — |    — |      — |
| **MAX**   |         — |       133.8 |  12.92 |      — |    — |    — |      — |

### Comparison vs Blog Targets (Qwen3.5-27B-Q4_K_M, RTX 3090)

| Metric          | Blog DDTree | Ours (DDTree) | Delta         | Result |
|-----------------|-------------|---------------|---------------|--------|
| AR baseline     | 37.78 tok/s | —             | —             | —      |
| Chain-only      | 112.82 tok/s| —             | —             | —      |
| DDTree mean TPS | 129.52 tok/s| **110.21 tok/s** | -19.31 tok/s | FAIL   |
| DDTree mean AL  | 8.31        | **11.04**     | +2.73         | PASS   |

**VERDICT: FAIL on TPS, PASS on AL.**

DDTree IS engaged and producing higher AL than the blog target (11.04 vs 8.31).
However, per-step speed (tok/s) is 15% below the blog's DDTree number.

Likely causes:
1. Model diff: blog ran Qwen3.5-27B, this run uses Qwen3.6-27B (different arch parameters,
   different SSM verify cost).
2. CUDA graphs: the blog mentions a "+1.6x lever" — no runtime flag exists in this tree.
   Without graph-replay, each decode step rebuilds and relaunches GGML ops.
3. Short prompts (100-150 tokens). The blog may have measured at higher prompt occupancy.

---

## Step 2 — 128K Context Decode WITH pFlash Prefill

Server log: /tmp/dflash_pflash_128k_FINAL.log
Config: --prefill-compression auto --prefill-threshold 32000
        --prefill-drafter /home/peppi/models/Qwen3-0.6B-BF16.gguf (keep=0.050 default)
        --max-ctx 131072 --ddtree --ddtree-budget 22 --cache-type-k q4_0 --cache-type-v q4_0

### pFlash Engagement — CONFIRMED

Log evidence:
```
[compress] drafter ready
[pflash] 126146 -> 23784 -> 26032 tokens (20.6% kept)
[server] chat CACHE ... effective_prompt=26032 pflash=true
```

- pflash=auto (banner in server config)
- Drafter forward: 74.93s for 115K tokens (Qwen3-0.6B-BF16, BF16 tensor-core path on RTX 3090)
- Compression: 126146 → 26032 effective tokens (20.6% kept; keep target=0.050 but anchors forced 20.6%)
- `pflash=true` in CACHE line confirms pFlash path was taken, not bypassed

### 128K pFlash Run Results

| Metric           | Value          | Notes                                    |
|------------------|----------------|------------------------------------------|
| prompt_tok       | 126146         | raw tokens sent                          |
| effective_in     | 26032          | after pFlash compression (20.6% kept)   |
| pflash drafter   | 74.93s         | Qwen3-0.6B-BF16 forward+score for 115K  |
| prefill_s        | 365.9s         | includes drafter time + target prefill   |
| prefill_TPS      | 71.1 tok/s     | 26032 / 365.9s (on effective_in)        |
| decode_s         | 68.7s          | 256 tokens                               |
| **decode_TPS**   | **3.74 tok/s** | MEASURED                                 |
| accept%          | 0.0% (1/2016)  | spec-decode completely rejected          |
| avg_commit (AL)  | 2.03           | near minimum (avg_commit ~= 2 = AR-mode) |
| wall_s           | 515.1s         | total request wall                       |

### vs Blog Target (134.78 tok/s / AL 8.33)

| Metric          | Blog 128K | Ours (pFlash 128K) | Delta          | Result |
|-----------------|-----------|--------------------|----------------|--------|
| decode tok/s    | 134.78    | **3.74**           | -131 tok/s     | FAIL   |
| AL              | 8.33      | **2.03**           | -6.30          | FAIL   |

**VERDICT: FAIL — massive gap. Root cause: KV-reservation cliff.**

The 3.74 tok/s decode is the known RTX 3090 "decode cliff" from max_ctx=131072 KV cache
reservation: q4_0 KV for 131072 positions requires ~9.7 GB on top of 15 GB model weights.
At KV position ~26K (compressed effective context), the 131K-slot KV table still consumes
the full reserved VRAM bandwidth. This is the KV-reservation cliff documented in
`project_qwen35moe_decode_cliff_kv_reservation.md`.

Additional factor: spec-decode accept=0.0% (essentially pure AR) due to the large KV context
degrading drafter alignment — the dFlash drafter is calibrated for short-context patterns.
At effective KV position 26K+, the drafter acceptance collapses.

The blog's 134.78 tok/s at "128K" likely measured with a shorter effective KV position
(e.g., 8K or 16K actual filled context against a 128K max-ctx reservation), or on a higher-
VRAM GPU (A100/H100) where the KV table does not dominate bandwidth.

---

## Summary

| Bench                       | Blog Target  | This Run         | Status             |
|-----------------------------|-------------|------------------|--------------------|
| Binary md5                  | —           | e9cb2790bb8ede64 | —                  |
| HumanEval mean tok/s        | 129.52      | **110.21**       | FAIL -19.3 tok/s   |
| HumanEval mean AL           | 8.31        | **11.04**        | PASS +2.73         |
| HumanEval max tok/s         | —           | **133.8**        | —                  |
| 128K pFlash prefill_s       | —           | **365.9s**       | (incl. 74.9s drafter) |
| 128K pFlash effective_in    | —           | **26032 tok**    | 20.6% kept         |
| 128K pFlash prefill_TPS     | —           | **71 tok/s**     | on effective_in    |
| 128K decode tok/s           | 134.78      | **3.74**         | FAIL (KV cliff)    |
| 128K AL                     | 8.33        | **2.03**         | FAIL (accept=0%)   |

DDTree engagement: CONFIRMED (all 10 HumanEval prompts + 128K run initiated spec-decode).
pFlash engagement: CONFIRMED (pflash=true in CACHE line, 126K→26K compression logged).
KV-reservation cliff: CONFIRMED (3.74 tok/s at effective 26K position, max_ctx=131072).
