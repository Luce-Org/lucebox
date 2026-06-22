> ⚠️ SUPERSEDED (2026-06-22): these runs varied `DFLASH_DRAFT_CTX_MAX` (IGNORED on the MoE path — draft_ctx is pinned at 4096) and/or used sequential requests (warm-EMA inflated accept). Treat the draft_ctx columns and any >77% accept as artifacts. Authoritative cold numbers: `clean_rebaseline.md`. The surviving real levers are `FEAT_RING_CAP=max_ctx` and f32 mirror.

# Draft-ctx VRAM Ceiling — Full KV Quant Grid

**Config**: Q3_K_XL target + BF16 drafter, max_ctx=40960, DFLASH_FEAT_RING_CAP=40960,
fa_window=0, port 18081, --lazy-draft. RTX 3090 24GB.
Measured 2026-06-22. Step A prompt: ctx_016384.json (~18043 tokens).

---

## 1. VRAM Ceiling — max-fitting draft_ctx at ~18K prompt

| KV quant | KV cache GiB (target @40960 ctx) | OOM at draft_ctx | Max-fitting draft_ctx | decode tok/s @max-fitting |
|---|--:|--:|--:|--:|
| **f16** | 3.12 GiB | none in {8192..32768} | **32768** | 39 tok/s† |
| **q8_0** | 1.56 GiB | 16384+ | **8192** | 142 tok/s |
| **q4_0** | 0.78 GiB | 16384+ | **8192** | 138 tok/s |
| **tq3_0** | 0.58 GiB | 16384+ | **8192** | 116 tok/s |

† f16 loads at 32768 but spec-decode gates to AR (decode 39 tok/s = AR speed) — not a usable win.

KV cache GiB = actual runtime VRAM per quant at 40960 ctx (f16=16b, q8_0=8b, q4_0=4b, tq3_0=~3b).
The server's `kv_cache=3.12 GiB` banner is the f16 baseline used for placement budget; actual
quantized-KV savings are per the table above.

### Why f16 KV is the OOM exception (mechanism)

The OOM at draft_ctx≥16384 fires in `ggml_cuda_pool_vmm::alloc` during CUDA graph compute
at decode token 2 (post-warmup). For q8_0/q4_0/tq3_0 KV, FlashAttention must dequantize
the target KV tensor before each compute step — at draft_ctx=16384 the dequant scratch
expansion overflows the remaining VMM pool. For f16 KV, FA reads tensors natively (no
dequant intermediate), so scratch stays flat at any draft_ctx. This is a dequant-scratch
cliff, not a static VRAM reservation.

### f16 at large draft_ctx: loads but spec-gate floors to AR

| draft_ctx | accept% | decode tok/s | spec-gate outcome |
|--:|--:|--:|---|
| 8192 | 76.8% | 127 tok/s | spec active |
| 16384 | 76.8% | 36 tok/s | spec active but 3.5× slower |
| 24576 | — | 39 tok/s | gate floors to AR (ema_ratio=0.86) |
| 32768 | — | 39 tok/s | gate floors to AR |

At draft_ctx≥24576 the drafter's O(ctx²) self-attention cost exceeds AR speed; the spec-gate
auto-detects and floors the session to AR. f16 *loads* at 32768 but the effective spec-decode
ceiling is still **8192** for any meaningful decode TPS gain.

Full Step A sweep across all KV quants (all serve the ~18K prompt OK if not OOM):

| KV | draft_ctx | status | accept% | decode tok/s |
|---|--:|---|--:|--:|
| f16 | 8192 | OK | 76.8% | 127 tok/s |
| f16 | 16384 | OK | 76.8% | 36 tok/s |
| f16 | 24576 | OK (AR) | — | 39 tok/s |
| f16 | 32768 | OK (AR) | — | 39 tok/s |
| q8_0 | 8192 | OK | 76.8% | 142 tok/s |
| q8_0 | 16384 | SERVE_OOM | — | — |
| q8_0 | 24576 | SERVE_OOM | — | — |
| q8_0 | 32768 | SERVE_OOM | — | — |
| q4_0 | 8192 | OK | 76.8% | 138 tok/s |
| q4_0 | 16384 | SERVE_OOM | — | — |
| q4_0 | 24576 | SERVE_OOM | — | — |
| q4_0 | 32768 | SERVE_OOM | — | — |
| tq3_0 | 8192 | OK | 76.8% | 116 tok/s |
| tq3_0 | 16384 | SERVE_OOM | — | — |
| tq3_0 | 24576 | SERVE_OOM | — | — |
| tq3_0 | 32768 | SERVE_OOM | — | — |

---

## 2. Recall Horizon at max-fitting draft_ctx per KV

Three needle prompts; draft_ctx=32768 for f16 (its max-fitting), 8192 for q8_0/q4_0/tq3_0.

| KV quant | draft_ctx | needle | marker dist from end | accept% | avg_commit | decode tok/s | recalled |
|---|--:|---|--:|--:|--:|--:|--:|
| **f16** | 32768 | 06k | ~2593 tok | 76.8% | 12.86 | 146 tok/s | YES |
| **f16** | 32768 | 08k | ~4854 tok | 41.3% | 6.85 | 90 tok/s | YES |
| **f16** | 32768 | 12k | ~7987 tok | 44.8% | 7.42 | 63 tok/s | YES |
| **q8_0** | 8192 | 06k | ~2593 tok | 76.8% | 12.86 | 148 tok/s | YES |
| **q8_0** | 8192 | 08k | ~4854 tok | 44.8% | 7.42 | 101 tok/s | YES |
| **q8_0** | 8192 | 12k | ~7987 tok | 34.2% | 5.93 | 78 tok/s | YES |
| **q4_0** | 8192 | 06k | ~2593 tok | 76.8% | 12.86 | 145 tok/s | YES |
| **q4_0** | 8192 | 08k | ~4854 tok | 44.8% | 7.42 | 101 tok/s | YES |
| **q4_0** | 8192 | 12k | ~7987 tok | 34.2% | 5.93 | 77 tok/s | YES |
| **tq3_0** | 8192 | 06k | ~2593 tok | 76.8% | 12.86 | 132 tok/s | YES |
| **tq3_0** | 8192 | 08k | ~4854 tok | 41.3% | 6.85 | 83 tok/s | YES |
| **tq3_0** | 8192 | 12k | ~7987 tok | 37.1% | 6.36 | 69 tok/s | YES |

All 12 needle probes recalled `luce_marker_widget` correctly. Recall depth is equivalent
across all KV quants at their respective max-fitting draft_ctx.

Note: f16 at draft_ctx=32768 shows the 08k/12k needles via spec-decode (spec-gate re-engages
at shorter needle contexts where drafter cost is acceptable). The recall is equal to q4_0 at 8192.

---

## 3. Per-KV Recommendation

**q4_0 + draft_ctx=8192**: max-fitting 8192, recall to ~8K from end, 77-145 tok/s, 0.78 GiB KV.
Equal accept/AL to f16 at same draft_ctx. Saves 2.34 GiB vs f16. **Recommended default.**

**q8_0 + draft_ctx=8192**: same draft_ctx ceiling. Recall equivalent. Decode slightly better at
06k (148 vs 145 tok/s), near-identical at 08k/12k. Accept anomaly (66.4% at 35K verbatim, per
prior kvsweep) persists. No VRAM advantage over q4_0; not recommended.

**f16 + draft_ctx=8192**: fastest decode (146 tok/s 06k vs 145 q4_0). Same recall depth.
Uses 2.34 GiB more KV. Only justified on >24 GB GPUs or if that extra margin buys a larger max_ctx.
At draft_ctx=32768 with f16, recall is equivalent but decode on 12k needle drops to 63 tok/s
(spec-gate floors to AR at 18K main prompt; only engages spec on the shorter needle prompts).

**tq3_0 + draft_ctx=8192**: recall equivalent. Decode 15-17% slower than q4_0 (69-132 vs 77-145).
Saves only 0.20 GiB vs q4_0 (0.58 vs 0.78 GiB KV). The decode penalty is not justified on 24 GB.

In one sentence each:
- **tq3_0**: frees most KV VRAM but decode 15% slower with identical recall — not worth it on 24 GB.
- **f16**: fastest decode, same recall as others at 8192, costs 2.34 GiB extra KV — use on bigger GPUs.
- **q8_0**: avoid — accept anomaly at long verbatim, no advantage over q4_0.
- **q4_0**: best 24-GB balance — same accept as f16, same recall, lowest usable-quant KV VRAM.

---

## 4. Single Best Production Config

**q4_0 KV + DFLASH_DRAFT_CTX_MAX=8192** on RTX 3090 24GB.

- Recall depth: ~8K tokens from end (all needles 06k/08k/12k correct)
- Decode TPS: 77-145 tok/s (content-dependent)
- KV VRAM: 0.78 GiB at max_ctx=40960
- Accept / AL: 76.8% / 12.86 at 06k needle
- Hard OOM ceiling: 16384 — never set draft_ctx above 8192

The draft_ctx ceiling is **8192 for every KV quant that avoids the dequant-scratch cliff**
(q8_0, q4_0, tq3_0). f16 KV uniquely avoids the cliff but gains nothing in spec-decode
at large draft_ctx (spec-gate floors to AR). Smaller KV quant does NOT unlock larger draft_ctx.

---

## 5. f16 Mirror Lever Finding (retained from prior run)

`DFLASH_FEATURE_DTYPE=f16` (env, default=f32) halves the feature ring from 2.68 GiB to 1.34 GiB,
which allows draft_ctx=16384 to load without OOM on non-f16 KV configurations. However,
at 16384 draft_ctx the drafter's per-step cost grows enough that spec-gate floors to AR at
typical 18K+ agentic prompts. The effective spec-decode ceiling is still 8192 regardless of
whether the feature ring is f32 or f16.

The lever to recover recall at distances >8K from end is **not KV quant or feature dtype** —
it is fixing the drafter's effective context (currently hard-limited to ~2048 tokens due to
the rope/conversion bug confirmed by the prior full-attn drafter test in NOTES.md).

---

## Raw Log References

Prior run (q4_0 only):
- `vram_sweep_8192.log`, `vram_sweep_16384.log` — original Step 1 (q4_0)

Full grid run (2026-06-22):
- `grid_{kv}_{ctx}.log` — Step A: 16 load+serve tests
- `grid_{kv}_needle_{ctx}.log` — Step B: 12 needle probes (3 per KV quant)
- `kv_grid_results.json` — machine-readable results
- `grid_sweep_master.log` — full Python sweep stdout
