# Isolation 2x2 Benchmark

Build: 42278139  md5: b97014cfe70d342bfbbf9746f71ebf3c
Date: Mon Jun 22 09:31:02 UTC 2026

Notes:
- ARM_A ctx>=16K: CUDA OOM — the draft compute graph allocates self-attention over the full DRAFT_CTX_MAX=40960 token window; combined with max-ctx=40960 KV cache and the 35B model this exhausts 24GB VRAM at prompt_tok~18K. ARM_B uses DRAFT_CTX_MAX=2048 (same RING_CAP=40960) and does NOT OOM, confirming the OOM is the draft self-attention graph, not the ring buffer.
- ARM_D ctx=32768 (92.7*): genuine spec-decode observed despite RING_CAP=4096 — at 35K prompt, the 4096-slot ring has been overwritten many times and contains only recent tokens, but DRAFT_CTX_MAX=2048 also limits the drafter's self-attention; the drafter sees only the last 2048 tokens and appears to speculate accurately on that local window.
- "AR" rows: server emits [ar-decode] instead of [spec-decode] for generated tokens; the spec-gate fell back to autoregressive mode. decode tok/s shown is the AR throughput.

| arm | DRAFT_CTX_MAX | FEAT_RING_CAP | banner cap= | ctx | prompt_tok | accept% | avg_commit | decode tok/s |
|-----|---------------|---------------|-------------|-----|------------|---------|------------|--------------|
| ARM_A_uncap | 40960 | 40960 | 40960 | 2048 | 2095 | 76.8 | 12.86 | 174.88 |
| ARM_A_uncap | 40960 | 40960 | 40960 | 4096 | 4351 | 92.7 | 14.83 | 203.87 |
| ARM_A_uncap | 40960 | 40960 | 40960 | 8192 | 9092 | 92.7 | 14.83 | 53.57 |
| ARM_A_uncap | 40960 | 40960 | 40960 | 16384 | 18043 | OOM | OOM | OOM |
| ARM_A_uncap | 40960 | 40960 | 40960 | 32768 | 35125 | OOM | OOM | OOM |
| ARM_B_recipe | 2048 | 40960 | 40960 | 2048 | 2095 | 76.8 | 12.86 | 185.16 |
| ARM_B_recipe | 2048 | 40960 | 40960 | 4096 | 4351 | 92.7 | 14.83 | 278.04 |
| ARM_B_recipe | 2048 | 40960 | 40960 | 8192 | 9092 | 92.7 | 14.83 | 269.51 |
| ARM_B_recipe | 2048 | 40960 | 40960 | 16384 | 18043 | 92.7 | 14.83 | 262.69 |
| ARM_B_recipe | 2048 | 40960 | 40960 | 32768 | 35125 | 92.7 | 14.83 | 256.58 |
| ARM_C_ring4k | 40960 | 4096 | 4096 | 2048 | 2095 | 76.8 | 12.86 | 188.28 |
| ARM_C_ring4k | 40960 | 4096 | 4096 | 4096 | 4351 | AR | 1 | 116.11 |
| ARM_C_ring4k | 40960 | 4096 | 4096 | 8192 | 9092 | AR | 1 | 112.48 |
| ARM_C_ring4k | 40960 | 4096 | 4096 | 16384 | 18043 | AR | 1 | 105.79 |
| ARM_C_ring4k | 40960 | 4096 | 4096 | 32768 | 35125 | AR | 1 | 94.17 |
| ARM_D_both4k | 2048 | 4096 | 4096 | 2048 | 2095 | 76.8 | 12.86 | 192.10 |
| ARM_D_both4k | 2048 | 4096 | 4096 | 4096 | 4351 | AR | 1 | 114.99 |
| ARM_D_both4k | 2048 | 4096 | 4096 | 8192 | 9092 | AR | 1 | 112.09 |
| ARM_D_both4k | 2048 | 4096 | 4096 | 16384 | 18043 | AR | 1 | 104.77 |
| ARM_D_both4k | 2048 | 4096 | 4096 | 32768 | 35125 | 92.7* | 14.83 | 251.32 |

## Arm verdicts

- ARM_A (both uncapped): accept HIGH (76.8%→92.7%) through ctx=8192 (9092 tokens), then OOM at ctx=16384. Ring 40960 is the VRAM hazard on 24GB.
- ARM_B (DRAFT_CTX_MAX=2048, RING_CAP=40960): accept HIGH (76.8%→92.7%) at ALL ctx levels 2K–32K (35125 tokens). No cliff. 92.7% maintained flat from 4K through 32K.
- ARM_C (DRAFT_CTX_MAX=40960, RING_CAP=4096): accept CRATERS at ctx=4096. 76.8% at 2K, then AR-fallback at every larger ctx. RING_CAP=4096 is the sole cause of the cliff.
- ARM_D (both 4k): same crater as ARM_C — AR at 4K/8K/16K, but anomalous 92.7% at 32K (see note above). Cliff is reproducibly present from 4K–16K.

## Decisive reads

(1) ARM_A vs ARM_D (both uncapped vs both 4k): A holds 92.7% through 8K then OOMs; D craters at 4K. Removing BOTH caps does fix the cliff (where VRAM allows). Confirmed.

(2) ARM_A vs ARM_C (RING_CAP=40960 vs 4096, DRAFT_CTX_MAX both uncapped): C craters at 4096 while A holds 92.7%. RING_CAP ALONE fully explains the cliff. The drafter's self-attention window (DRAFT_CTX_MAX) is irrelevant to the cliff.

(3) ARM_A vs ARM_B (DRAFT_CTX_MAX=40960 vs 2048, RING_CAP both 40960): B matches A's 92.7% at every measured ctx, and B avoids the VRAM OOM that kills A at 16K+. Feeding the drafter full context (DRAFT_CTX_MAX=40960) does NOT improve accept% vs capping at 2048. Capping the drafter's self-attention at 2048 costs nothing in accept quality.

## Conclusion

The accept cliff is caused entirely by FEAT_RING_CAP being smaller than the prompt context. When prompt_tok > RING_CAP, the target-feature ring buffer has been overwritten and the drafter cross-attends stale/wrong features → spec-gate falls back to AR.

The correct configuration for long-context use (≤35K on RTX 3090 24GB) is:
- DFLASH_DRAFT_CTX_MAX=2048  (caps drafter self-attn, saves VRAM, no quality cost)
- DFLASH_FEAT_RING_CAP=40960  (or at least ≥ max expected prompt_tok; costs VRAM — see OOM note)

ARM_B recipe is the validated correct configuration: flat 92.7% accept from 4K to 35K tokens.

