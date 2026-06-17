# Spark results

All on a single **RTX 3090 (24 GB)**, target **Laguna-XS.2 Q4_K_M** (MoE, 3B
active / 33B total, 256 experts/layer, top-8, 40 layers). Calibration corpus:
**333 chunks / ~171K tokens** of routing extracted from real Claude Code
sessions across 6 projects. Held-out: **60 chunks from sessions never seen
during calibration** (split by session hash, no leakage).

Decode tok/s is single-stream. "Cold-hit rate" = fraction of the 8 routed
experts/layer that land on the CPU (cold) tier per token. Sections 1-2 isolate
the placement and cache levers on the per-layer decode path; the shipped default
**single-graph fused decode** (section 4) lifts that operating point to **~100
tok/s** (`spark/bench.py`, bit-identical to all-GPU at full residency).

## 1. Placement: calibration is the big lever

Held-out Claude Code sessions, varying the hot budget. `0%` = all experts on GPU
(no offload).

| Config (held-out) | decode tok/s | % of all-GPU speed | cold-hit | hot VRAM | VRAM saved |
|---|---:|---:|---:|---:|---:|
| All-GPU (100% hot) | 111 | 100% | n/a | 18.8 GiB | 0 |
| Uniform 60% | 66 | 59% | 36% | 10.6 GiB | ~7 GiB |
| **Calibrated 60%** | **81** | **73%** | **6.6%** | 10.6 GiB | ~7 GiB |
| Calibrated 70% | 85 | 76% | lower | 12.4 GiB | ~5.3 GiB |
| Calibrated 80% | 87 | 78% | lower | 14.1 GiB | ~3.5 GiB |

Calibration cuts the cold-hit rate from **36% → 6.6%** and lifts tok/s **66 →
81** at the same VRAM, and it generalizes to sessions it never saw. A static
frequency tier beyond this is just "raise the budget" (coverage grows ~linearly
with size), so the next lever is the cache, not a bigger hot set.

Notes:
- Decode-aware calibration (`--n-gen 24`) did not beat prefill-only (~81 both).
- Calibrate on the traffic you serve: a pure-code prompt scored against a
  conversation-heavy profile hit 16% cold instead of 6.6%.

## 2. Bounded expert cache: drives cold -> 0 in fixed VRAM

A fixed ring of spare GPU slots per layer. On a cold hit the expert is swapped
in (LRU evict) and served on-GPU for the rest of the session. The distinct cold
set saturates within a session, so a small cache eliminates cold after warmup.
Steady-state on a long generation:

| cache slots/layer | tok/s | cold-hit/tok | cache VRAM |
|---:|---:|---:|---:|
| 0 (no cache) | 74.5 | 39.7 | 0 |
| 16 | 80.2 | 5.6 | +1.1 GiB |
| 32 | 85.0 | 0.8 | +2.1 GiB |
| 48 | 88.2 | 0.4 | +3.2 GiB |

(Sweep run on a code prompt where the cold rate starts high, to show the cache
mechanism; on matched held-out sessions cold starts at 6.6% and 16-32 slots
already drive it to ~0.)

## 3. VRAM: stays under 16 GiB

Measured peak for the target operating point (**60% hot + 32 cache slots**):

| Component | VRAM |
|---|---:|
| Core (non-expert) weights | ~1.35 GiB |
| 60% hot experts | 10.61 GiB |
| 32 cache slots | ~2.1 GiB |
| KV cache (ctx 4096) | 0.62 GiB |
| CUDA context / overhead | ~0.5 GiB |
| **Measured peak** | **14.59 GiB** |

A 33B-total MoE on **14.6 GiB**, vs 18.8 GiB to hold it all. The sweep above is
the per-layer decode path; with the default single-graph fused decode the same
operating point runs at **~100 tok/s** (section 4). Trade cache slots against
context length to keep headroom under 16.

## 4. Single-graph fused decode (shipped) + the last gap (research)

The per-layer hybrid (sections 1-2) topped out near 82% of all-GPU because each
layer ran as a separate small graph. The **single-graph fused decode**
(`laguna_step_hybrid`, default-on) collapses the whole token into one graph and
lifts the operating point to **~100 tok/s (~85% of all-GPU)**, bit-identical to
all-GPU at full residency (128/128 tokens). No predictor needed for that.

Closing the **last ~15%** is the hard part: it needs the routed experts known
*before* the FFN runs so cold fetches prefetch under compute, i.e. a predictor
accurate enough to be useful. We captured `(block-input hidden -> selected experts)` traces (1.2 GB,
150K samples) and trained per-layer pre-gates:

| predictor | recall@8 | recall@16 | recall@24 |
|---|---:|---:|---:|
| temporal (last token) | ~24% | -- | -- |
| linear (ridge) | 49.6% | 64.6% | 72.0% |
| MLP (512, GELU) | 52.8% | 68.3% | 75.9% |

The MLP beats linear by only ~3%, so the limit is **fundamental, not capacity**:
the pre-gate sees the pre-attention hidden, but the router decides on the
post-attention hidden. Clean prefetch wants ~95%+ recall at small K. A
fitted-from-traces predictor does not get there. Reaching it means relocating
and fine-tuning the model's gate (Pre-gated MoE), for which these traces are the
dataset. The placement + cache results above need no model change.

The table above trains with per-expert binary cross-entropy. recall@K is a
ranking metric, so `spark/train_pregate.py` now defaults to a ranking-aware
objective (`--loss listnet|ranknet`) that optimizes the *order* of the expert
scores instead of each expert independently (ListNet, Cao et al. 2007; RankNet,
Burges et al. 2005; ranking-aware pre-gating, arXiv:2511.10676). `--loss all`
trains every objective on the same split and init and prints the recall@K
comparison. The ranking objective is the right one for top-K prefetch, but it
cannot supply the post-attention information the pre-attention input lacks, so it
does not move the **fundamental** cap — it tightens the surrogate, not the
ceiling. Re-measure on the full trace with `--loss all` to quantify the delta
over the BCE baseline.
