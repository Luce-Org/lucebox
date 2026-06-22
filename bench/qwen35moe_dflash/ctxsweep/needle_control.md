> ⚠️ SUPERSEDED (2026-06-22): these runs varied `DFLASH_DRAFT_CTX_MAX` (IGNORED on the MoE path — draft_ctx is pinned at 4096) and/or used sequential requests (warm-EMA inflated accept). Treat the draft_ctx columns and any >77% accept as artifacts. Authoritative cold numbers: `clean_rebaseline.md`. The surviving real levers are `FEAT_RING_CAP=max_ctx` and f32 mirror.

# Distant-Recall Control Benchmark

Build: server/build/dflash_server (2026-06-22)
GPU: RTX 3090 24GB (23163 MB free at start)
Target: Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
Draft: qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf
KV cache: q4_0/q4_0, fa-window=0, max-ctx=40960, max-tokens=200, temp=0

## Design

Marker function `luce_marker_widget` is embedded at the MIDDLE of the context,
not at the tail. The drafter capped at DRAFT_CTX_MAX=2048 cannot self-attend
to any token before the last 2048 positions; both prompts place the marker
well outside that window (>2000 tok from end on 6k, >4000 tok on 8k).

- needle_mid_06k: ~6000 tokens, marker at ~token 3000 (>2048 from end)
- needle_mid_08k: ~8000 tokens, marker at ~token 4000 (>4000 from end)

## Results

| arm | DRAFT_CTX_MAX | prompt | prompt_tok | marker_pos (approx) | accept% | avg_commit | decode tok/s | output_correct |
|-----|---------------|--------|------------|---------------------|---------|------------|--------------|----------------|
| ARM_A_uncap | 40960 | needle_mid_06k | 6593 | ~3000 | 76.8% | 12.86 | 56.6 | Y |
| ARM_A_uncap | 40960 | needle_mid_08k | 8854 | ~4000 | 44.8% | 7.42 | 41.4 | Y |
| ARM_B_cap2048 | 2048 | needle_mid_06k | 6593 | ~3000 | 30.9% | 5.62 | 87.2 | Y |
| ARM_B_cap2048 | 2048 | needle_mid_08k | 8854 | ~4000 | 34.2% | 5.93 | 99.3 | Y |

Source log lines:
- ARM_A 06k: [spec-decode] tokens=90 speed=56.57 steps=7 accepted=86/112 (76.8%) avg_commit=12.86
- ARM_A 08k: [spec-decode] tokens=89 speed=41.43 steps=12 accepted=86/192 (44.8%) avg_commit=7.42
- ARM_B 06k: [spec-decode] tokens=90 speed=87.24 steps=16 accepted=79/256 (30.9%) avg_commit=5.62
- ARM_B 08k: [spec-decode] tokens=89 speed=99.28 steps=15 accepted=82/240 (34.2%) avg_commit=5.93

## Decisive read

ARM_B (draft_ctx=2048) shows substantially LOWER accept than ARM_A (draft_ctx=40960)
on these middle-marker prompts:
- 6k prompt: 76.8% (A) vs 30.9% (B) — a drop of 45.9 pp
- 8k prompt: 44.8% (A) vs 34.2% (B) — a drop of 10.6 pp

Both arms reproduced the marker function correctly (output_correct=Y), so the
target (35B) can retrieve the distant marker regardless — but the DRAFTER's
speculation quality collapses when its self-attention is capped to the last 2048
tokens and the answer token region is thousands of tokens away.

VERDICT: draft_ctx=2048 is NOT free for distant-recall content.
The drafter genuinely uses the distant context window. Capping at 2048 costs
~46 pp accept on 6k mid-marker prompts. The prior 2x2 isolation result
(ARM_B matching ARM_A accept at 92.7%) held only because those prompts had
the answer in the TAIL (recent) tokens, squarely within the 2048-token window
the capped drafter sees. The claim "draft_ctx=2048 is free even for distant
content" is FALSE; it is only free when the prediction target is within the
last 2048 tokens of the prompt.

NOTE on decode tok/s reversal: ARM_B decode is faster (87-99 tok/s vs 56-41
tok/s for ARM_A) because with lower accept the drafter issues more, smaller
verify passes with less KV recompute per committed token — decode wall is
dominated by target verify cost, not accept rate alone at this ctx.
