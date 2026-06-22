# Qwen3.6-35B-A3B + dFlash — validated launch recipe (RTX 3090)

Validated 2026-06-22. Supersedes earlier draft_ctx/f16 findings — see `ctxsweep/clean_rebaseline.md`.
All numbers are **cold, one-prompt-per-server** on a single RTX 3090 (24 GB), Q3_K_XL all-hot, build 950eae1d.

## TL;DR launch

```bash
DFLASH_FEAT_RING_CAP=40960 \
server/build/dflash_server \
  /home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  --draft /home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf \
  --host 127.0.0.1 --port 18099 \
  --max-ctx 40960 \
  --fa-window 0 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --chat-template-file /home/peppi/models/qwen3-coder-chat-template.jinja \
  --model-name luce-dflash
```

**KV = q4_0 is the recommended default** — parity accept/AL at ¼ the KV VRAM.

**Model choice for long-context coding sessions: the 35B-A3B MoE, not 27B dense.** The
MoE computes only ~3B active params/token → 170 tok/s decode on copy content; 27B dense computes
all 27B → only 11–16 tok/s at the same context.

Send requests at **temperature 0** (dFlash spec-decode is hard-gated to temp==0;
temp>0 silently falls back to AR).

## Why each flag

| Setting | Why | Cost of getting it wrong |
|---|---|---|
| **`DFLASH_FEAT_RING_CAP=max_ctx`** | MANDATORY — enables the drafter's cross-attention to the full-context feature ring (this is the distant-reach mechanism) AND prevents the wrap-to-stale cliff | ring < prompt_tokens → spec floors to AR unpredictably |
| **`--cache-type-k/v q4_0`** | ¼ KV VRAM at parity accept/AL | `tq3_0` → 10× slower prefill; `q8_0` → accept regression |
| **`--max-ctx 40960`** (size to the session) | KV is reserved at load from max-ctx; right-sized fits all-hot | `65536` reserves ~5 GiB KV → with the BF16 drafter overflows 24 GB → spills to host → 186s cold prefill |
| **`--fa-window 0`** | full attention — preserves tool/system context for agentic tool-calling | windowed → drops tool defs at long ctx |
| **commit-EMA gate (default on)** | auto-calibrated: runs spec when it beats AR, floors to AR otherwise — never slower | — |
| **Unsloth `qwen3-coder` chat template** | tool-calling works under heavy agentic system prompts | official template breaks tool-calls under claude-code's prompt |

## Measured (cold, one-prompt-per-server)

Config: ring=40960, f32 mirror (default), q4_0 KV, Q3_K_XL, max_ctx=40960.

| scenario | accept% | AL | decode tok/s | prefill |
|---|--:|--:|--:|--:|
| copy/recent 9K ctx (ctx_008192) | 76.8% | 12.86 | 174.6 | 4.4s |
| copy/recent 35K ctx (ctx_032768) | 76.8% | 12.86 | 171.3 | 17.1s |
| distant needle 2.6K-back | 30.9% | 5.62 | 81.3 | — |
| distant needle 12K-back | 28.7% | 5.29 | 78.3 | — |
| AR baseline (no spec / cliff-floored) | — | — | ~86 | — |

Copy content: **~2.0× decode** over AR. Distant needle at 12K-back: marker reproduced at 28.7% accept.

Distant reach works at the fixed draft_ctx=4096 because the drafter uses **cross-attention to the
target-feature ring**, not self-attention to its own KV. `FEAT_RING_CAP=max_ctx` is the only lever
that matters for reach.

## Footguns / do-NOT

- **Do NOT set `DFLASH_FEATURE_DTYPE=f16`** — floors spec-decode to AR on every prompt (ema 0.53–0.57). f32 (the default) is required; do not override it.
- **`DFLASH_DRAFT_CTX_MAX` is IGNORED on the MoE backend** — draft_ctx is pinned at 4096 in `qwen35moe_backend.cpp` (lines 2267-2270) regardless of the env var (which only exists in the dense qwen35 backend). Setting it does nothing here.
- **Distant recall is via feature-ring cross-attention**, not drafter self-attention window. `FEAT_RING_CAP=max_ctx` is the only lever; there is no self-attn window to tune.
- ❌ `DFLASH_FEAT_RING_CAP` < max_ctx — wraps the feature ring → accept cliff.
- ❌ oversized `--max-ctx` — size it to the session; oversizing spills VRAM with the BF16 drafter.
- ❌ temp > 0 if you want dFlash (silent AR fallback).
- ❌ Q4_K_M target all-hot on 24 GB (doesn't fit → hybrid path → can't cache).

See `ctxsweep/clean_rebaseline.md` for the full cold sweep. `GOTCHAS.md` (same dir) for additional footguns.
