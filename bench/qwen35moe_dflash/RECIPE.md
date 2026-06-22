# Qwen3.6-35B-A3B + dFlash — validated launch recipe (RTX 3090)

Validated 2026-06-22. This is the config that gives **fast prefill + intact caching +
decode ≥ AR (and up to ~170 tok/s on code)** on a long-context agentic coding session.
All numbers on a single RTX 3090 (24 GB), Q3_K_XL all-hot.

## TL;DR launch

```bash
DFLASH_FEAT_RING_CAP=40960 \
DFLASH_DRAFT_CTX_MAX=8192 \
DFLASH_FEATURE_DTYPE=f16 \
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

**KV = q4_0 is the recommended default** — the sweep showed it's *performance-equal*
to f16 (138-145 vs 174 tok/s decode, identical accept / AL) at **¼ the KV VRAM**, so
it fits everywhere. Use f16 only if you have spare VRAM and want the last ~4%.

**`DFLASH_FEAT_RING_CAP=40960` (= max_ctx) is mandatory.** When prompt_tokens >
ring_cap the target-feature ring wraps → drafter cross-attends stale features → the
commit-EMA gate correctly floors to AR. Ring cap must be ≥ max_ctx or accept craters.

**`DFLASH_DRAFT_CTX_MAX=8192`** is the VRAM-max uncrippled drafter self-attention
window on a 24 GB GPU at max_ctx=40960. It recovers distant recall to ~8K-from-end
(vs 2048's ~2K). The ceiling is draft-graph decode-time dequant scratch, not KV
reservation. OOMs at draft_ctx≥16384 on this config.

**`DFLASH_FEATURE_DTYPE=f16`** halves the feature-mirror VRAM (2.68→1.34 GB) with
zero quality cost at draft_ctx=8192 — free headroom.

**Model choice for long-context coding sessions: the 35B-A3B MoE, not 27B dense.** The
MoE computes only ~3B active params/token → 55–67 tok/s decode at 34K; 27B dense computes
all 27B → only 11–16 tok/s at the same context. Dense-27B's published 177 tok/s is a
*short-context* result; it loses badly at long context.
Send requests at **temperature 0** (dFlash spec-decode is hard-gated to temp==0;
temp>0 silently falls back to AR).

## Why each flag

| Setting | Why | Cost of getting it wrong |
|---|---|---|
| **`--cache-type-k/v q4_0`** | ¼ KV VRAM at parity accept/AL; 138-145 tok/s decode | `tq3_0` → 10× slower prefill, slower decode; `q8_0` → accept regression |
| **`--max-ctx 40960`** (size to the session, don't oversize) | KV is reserved at load from max-ctx; right-sized fits all-hot | `65536` reserves ~5 GiB KV → with the 3.3 GB bf16 drafter overflows 24 GB → spills to host → 186s cold prefill |
| **`DFLASH_DRAFT_CTX_MAX=8192`** | VRAM-max drafter self-attention window at max_ctx=40960 on 24 GB; recovers recall to ~8K-from-end | `2048` craters distant recall 46pp (needle@2.6K-from-end: 30.9% accept vs 76.8% at draft_ctx=8192). `≥16384` OOMs (draft-graph scratch, not KV). |
| **`DFLASH_FEAT_RING_CAP=<max-ctx>`** | target-feature ring must cover the context or it wraps → stale features → accept cliff | any value < prompt_tokens → accept craters; must be ≥ max_ctx |
| **`DFLASH_FEATURE_DTYPE=f16`** | halves feature-mirror VRAM (2.68→1.34 GB), zero quality cost | `bf16` wastes headroom that limits draft_ctx ceiling |
| **`--fa-window 0`** | full attention — preserves tool/system context for agentic tool-calling | windowed → drops tool defs at long ctx |
| **commit-EMA gate (default on)** | auto-calibrated: runs spec when it beats AR, floors to AR otherwise — never slower | — |
| **Unsloth `qwen3-coder` chat template** | tool-calling works under heavy agentic system prompts | official template breaks tool-calls under claude-code's prompt |

## Recall-horizon table (draft_ctx=8192, q4_0 KV, Q3_K_XL, max_ctx=40960)

| marker dist from end | accept% | decode tok/s | recalled |
|---|--:|--:|---|
| ~2.6K | 76.8% | 130-145 | yes |
| ~4.9K | 44.8% | 93-101 | yes |
| ~8.0K | 34.2% | 76-78 | yes |

draft_ctx=2048 control: 30.9% accept at just 2.6K-from-end — a 46pp amputation.

draft_ctx beyond 8192 is **self-defeating, not an open lever**: `DFLASH_FEATURE_DTYPE=f16`
(halves the feature-mirror VRAM) does make draft_ctx=16384 fit, but the larger drafter
self-attention makes each spec step slower than AR, so the commit-EMA gate correctly floors
to AR (`reason=slow`) — no throughput win. Measured: draft_ctx=8192 already recalls
12-14K-deep markers at 33-35% with the gate held; 16384 lifts raw accept to ~44% but floors
to AR. **8192 is the throughput sweet spot.**

## KV-quant ceiling table (max_ctx=40960, draft_ctx=8192)

| KV | KV cache GiB @40960 | max-fit draft_ctx | decode tok/s |
|---|--:|--:|--:|
| q4_0 | 0.78 | 8192 | 138-145 |
| q8_0 | 1.56 | 8192 | 142 |
| tq3_0 | 0.58 | 8192 | 116 |
| f16 | 3.12 | 32768 (spec-gate floors to AR → no benefit) | 39 |

Ceiling is 8192 regardless of KV quant — set by the draft compute graph's decode-time
dequant scratch (VMM overflow at draft_ctx≥16384), not KV reservation.

## KV precision options (measured, 35K verbatim)

| KV | prefill | decode | accept% | AL | use when |
|---|--:|--:|--:|--:|---|
| **f16** | 18.0s | **174** | 76.8 | 12.86 | max decode speed, VRAM-rich only |
| **q4_0** | 18.0s | 138-145 | 76.8 | 12.86 | **recommended default** — parity accept/AL, ¼ KV VRAM |
| q8_0 | 18.1s | 142 | — | — | ❌ no gain over q4_0, double the KV VRAM |
| tq3_0 | 23.6s | 116 | 76.8 | 12.86 | ❌ only if VRAM-desperate (256K) |

## Measured session result (goldgate, f16 + max-ctx 40960)

| turn | prefill | decode | restore |
|---|--:|--:|---|
| 1 (cold) | 18.3s | 67 | false |
| 2 (warm) | 0.8s | 66 | **true** |
| 3 (warm) | 2.9s | 55 | **true** |

vs the `tq3_0` + 65536 regression: 186s cold prefill, 20 tok/s decode.

## When dFlash beats AR (vs floors to it)
Accept is content-dependent. dFlash **wins** on copy-derivable output (refactors,
edits, repeating recent code → 100–200 tok/s) where the relevant context is in the
last ~8192 tokens. It **floors to AR** (correctly, never slower) on novel generation
(prose, fresh answers) — high entropy. The gate makes this automatic.

## Hard "don'ts"
- ❌ `tq3_0` / `q8_0` KV (see table) — `f16` or `q4_0` only.
- ❌ oversized `--max-ctx` — size it to the session; oversizing spills VRAM with the bf16 drafter.
- ❌ temp > 0 if you want dFlash (silent AR fallback).
- ❌ Q4_K_M target all-hot on 24 GB (doesn't fit → hybrid path → can't cache).
- ❌ `DFLASH_FEAT_RING_CAP` < max_ctx — wraps the feature ring → accept cliff.
- ❌ `DFLASH_DRAFT_CTX_MAX` < 8192 — amputates distant recall (see recall-horizon table).
- ❌ a different `draft_ctx`/ring/rope without re-checking accept — these are the documented footguns (see GOTCHAS.md).

See `GOTCHAS.md` (same dir) for the full footgun list, `charbench/NOTES.md` and
`ctxsweep/NOTES.md` for the supporting measurements.
