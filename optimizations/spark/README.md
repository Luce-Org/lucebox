<p align="left">
  <a href="../../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce Spark</h1>

<p align="center">
  <strong>Run a big MoE on a smaller GPU by keeping only the active units lit.</strong><br/>
  Calibrate which experts stay hot from real traffic, swap the rest through a bounded GPU cache.<br/>
  Laguna-XS.2 (3B active / <strong>33B total</strong>) on a single RTX 3090: a 33B-total MoE in <strong>14.6 GiB</strong>,<br/>
  decoding at <strong>~100 tok/s</strong> with one fused graph, near the <strong>119 tok/s all-GPU ceiling</strong>, vs 66 for naive offload.<br/><br/>
  <a href="https://lucebox.com">lucebox.com</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a>
</p>

---

```
Laguna-XS.2 Q4_K_M (33B total MoE) · RTX 3090 · decode at 60% residency

                                tok/s   % all-GPU
  all on GPU (needs >16 GB)      119       100%
  naive offload (uniform)         66        55%
  Spark, calibrated               81        68%
  Spark + cache + fused decode    100        85%

  reproduce: python -m spark.bench --bin ../../server/build/test_dflash ...
```

> A 33B-total mixture-of-experts only fires ~8 of 256 experts per token, but a naive hot/cold offload still pays for it: pick the wrong experts to keep resident and you hit the CPU tier a third of the time. Spark exploits **activation sparsity** as a product: it calibrates the hot set from the traffic you actually serve, then a bounded GPU cache swaps the long tail in and out so cold-misses fall to ~zero, all inside a fixed VRAM budget.

<p align="center">
  <img src="demo.gif" width="760" alt="Naive expert offload vs Luce Spark: same 33B MoE, same RTX 3090, same 60% GPU residency, same output. Spark decodes at 100 tok/s vs 66 for naive offload." />
</p>

<p align="center"><sub>Same model, same card, same 60% residency, same output. Spark finishes first: 66 to 100 tok/s, 1.5x the decode.</sub></p>

## What Spark is

Spark is the **placement + caching layer** on top of the merged hot/cold MoE
offload engine in [`../../server/`](../../server/). The engine knows how to split
experts hot (GPU) / cold (CPU) and compute the cold ones. Spark makes that
actually fast on a small card:

1. **Calibrated placement.** The expert that *should* be hot is the one your
   traffic routes to most. Spark replays real sessions through the daemon,
   accumulates per-(layer, expert) routing frequencies, and emits a placement
   profile. Loading it cuts the cold-hit rate from **36% (uniform) to 6.6%** on
   held-out sessions and lifts decode **66 → 81 tok/s** at the same VRAM.
2. **A bounded expert cache.** Calibration is static; real generations still hit
   a per-session tail. A fixed ring of spare GPU slots swaps cold experts in on
   first use (LRU evict) and serves them on-GPU afterward. The distinct cold set
   saturates within a session, so a small cache drives cold-misses to **~0**
   without growing VRAM, removing the cold-tail penalty on throughput.
3. **Per-token fused decode.** Under offload the engine was building 40 separate
   per-layer GPU graphs per token; that submission overhead, not where the experts
   live, was the remaining gap. Spark folds the routed FFN into the attention graph
   and runs the whole token as **one fused graph** (`laguna_step_hybrid`, default-on):
   **bit-identical to all-GPU at full residency (119 tok/s)** and **~100 tok/s at
   60% residency** on the decode bench. Closing the last ~15% needs the next experts
   known one step early; token-level prediction caps near **53% recall**, so that
   part stays open research, not a shipped win.

It generalizes beyond experts: the same hot/cold residency applies to
fine-grain neuron sparsity (Deja Vu / PowerInfer style). MoE experts are the
coarse-grain case Spark ships today.

## Results

Single RTX 3090, Laguna-XS.2 Q4_K_M. Calibrated from a 333-chunk / ~171K-token
corpus of real Claude Code sessions; cold-hit rates validated on 60 held-out
sessions. Decode tok/s reproduce with [`spark/bench.py`](spark/bench.py); full
methodology in [RESULTS.md](RESULTS.md).

| Config (60% resident) | decode tok/s | % all-GPU | cold-hit | VRAM |
|---|---:|---:|---:|---:|
| All-GPU (needs >16 GB) | 119 | 100% | - | 18.8 GiB |
| Naive offload (uniform) | 66 | 55% | 36% | ~10.6 GiB |
| **Spark, calibrated** | **81** | 68% | 6.6% | ~10.6 GiB |
| **Spark + cache + fused decode** | **100** | **85%** | **~0%** | **14.6 GiB** |

Calibration drives the cold-hit rate from 36% (uniform) to 6.6%, the bounded
cache to ~0; the single-graph fused decode (`laguna_step_hybrid`, default-on) is
**bit-identical to all-GPU at full residency** (128/128 tokens, 119 tok/s) and
holds **~100 tok/s at 60% residency**. Peak VRAM at the operating point (60% hot
+ 32 cache slots) measured at **14.59 GiB**: a 33B-total MoE under 16 GiB.

## How it works

```
  ┌─ offline, once per traffic profile ───────────────────────────────┐
  │  sessions (*.jsonl) ─ extract ─► corpus ─ calibrate ─► placement.csv │
  └────────────────────────────────────────────────────────────────────┘
                                    │  DFLASH_LAGUNA_HOTNESS=placement.csv
                                    ▼
  ┌─ serve time, per layer, per token ──────────────────────────────────┐
  │  router picks 8 experts                                              │
  │     ├─ hot (calibrated, pinned on GPU) ───────────► GPU FFN          │
  │     ├─ warm (in the cache ring) ──────────────────► GPU FFN          │
  │     └─ cold miss ─ swap into a spare slot (LRU evict) ─► GPU FFN      │
  │                     (rare after warmup; bounded VRAM)                │
  └──────────────────────────────────────────────────────────────────────┘
```

The hot tier is pinned from calibration; the cache ring is a small over-allocation
of the hot expert stack, so a swap is "copy 3 weight tensors into a spare slot +
update one routing LUT entry" and the existing GPU FFN serves it with no special
path. Cold experts that are never cached fall back to the engine's CPU path.

## Serve it: one self-tuning command

For production, you don't run any of the offline pipeline. `dflash_server`
auto-tunes from its own traffic:

```bash
dflash_server <model.gguf> --spark                   # use the card, auto-size everything
dflash_server <model.gguf> --spark --spark-vram 14   # cap total VRAM at 14 GiB
```

The only knob is `--spark-vram <GiB>`: the total VRAM Spark may use. From that
target it sizes everything itself. It fits the non-expert weights, the KV cache
and a safety margin, pins as many calibrated-hot experts as the rest allows, and
carves an auto-sized cache ring out of the budget (you never set a slot count).
Omit it and Spark sizes to the whole card.

That one flag aside, `--spark`:
- enables the **bounded expert cache** (auto-tunes the working set at serve time),
- **auto-loads** a learned placement profile from `<model>.gguf.spark.csv` if present,
- keeps **persisting** that profile after every request from live routing.

First boot starts uniform and warms the cache within a session; each restart
loads a better profile and starts warmer. No corpus, no CSV juggling, no env
vars. Verified on RTX 3090 for **both** Laguna-XS.2 and Qwen3.6-35B-A3B:
`--spark-vram 13`/`15` holds peak VRAM at 11.6/13.8 GiB, the profile is written
and reloaded (`source=hotness:...`), and generation stays coherent under offload.
(`--spark-slots N` forces an explicit cache size if you ever want to override the
auto-sizing.)

The offline pipeline below is for **bootstrapping** a profile before first serve
(e.g. from your own Claude Code and Codex session history) and for
**evaluation**. It is optional.

## Bootstrapping / eval pipeline

The tooling here drives the dflash daemon (`test_dflash`); build it from
[`../../server/`](../../server/) first.

```bash
cd optimizations/spark
uv sync                                   # tokenizers (+ gguf/torch optional extras)

# 0. one tokenizer, extracted from the GGUF (gpt2 byte-level BPE)
python -m spark.tokenizer --gguf laguna-xs2-Q4_K_M.gguf --out laguna_tok.json

# 1. corpus from your agent sessions, by session split. Pulls Claude Code
#    (~/.claude/projects) and Codex (~/.codex/sessions) by default; --source
#    claude|codex|both to pick.
python -m spark.extract_sessions --source both --out-dir ./corpus

# 2. calibrate the placement profile (routing is placement-independent, so a
#    high budget calibrates fast)
python -m spark.calibrate \
    --bin ../../server/build/test_dflash --gguf laguna-xs2-Q4_K_M.gguf \
    --tok laguna_tok.json --corpus corpus/train.jsonl --out-profile spark_profile.csv

# 3. validate on held-out sessions: uniform vs calibrated vs + cache
python -m spark.validate --bin ../../server/build/test_dflash \
    --gguf laguna-xs2-Q4_K_M.gguf --tok laguna_tok.json \
    --corpus corpus/test.jsonl --budget-pct 60                       # uniform
python -m spark.validate ... --budget-pct 60 --hotness spark_profile.csv        # calibrated
python -m spark.validate ... --budget-pct 60 --hotness spark_profile.csv --cache-slots 32
```

## Reproduce the decode numbers

```
python -m spark.bench --bin ../../server/build/test_dflash \
    --gguf laguna-xs2-Q4_K_M.gguf --tok laguna_tok.json \
    --hotness laguna-xs2-Q4_K_M.gguf.spark.csv --budget-pct 48 --cache-slots 32
```

Reports steady-state decode tok/s and asserts the single-graph hybrid path is
token-for-token identical to all-GPU at full residency. On an RTX 3090,
laguna-xs2 Q4_K_M: all-GPU 119 tok/s, single-graph @100% 118.8 (128/128 exact),
Spark offload @~60% residency 99 tok/s (83% of all-GPU). This is the same
`LagunaBackend` decode path that `dflash_server --spark` runs.

Deploy: run the daemon with `DFLASH_EXPERT_BUDGET_PCT=60
DFLASH_LAGUNA_HOTNESS=spark_profile.csv` and, for the cache,
`DFLASH_LAGUNA_EXPERT_CACHE=1 DFLASH_LAGUNA_CACHE_SLOTS=32`.

## Engine knobs

Placement uses the merged hybrid-offload engine; the cache + trace are the Spark
engine additions. All on the daemon process.

| Env var | Purpose |
|---|---|
| `DFLASH_EXPERT_BUDGET_PCT` | hot-tier VRAM budget (% of experts resident) |
| `DFLASH_LAGUNA_HOTNESS` | placement profile CSV from `spark.calibrate` |
| `DFLASH_LAGUNA_NEXT_PLACEMENT_OUT` | accumulate + dump a routing profile (calibration) |
| `DFLASH_LAGUNA_EXPERT_CACHE` / `_CACHE_SLOTS` | enable the bounded cache + slots/layer |
| `DFLASH_LAGUNA_GPU_REMAP` | unified GPU FFN the cache serves through |
| `DFLASH_LAGUNA_PREGATE_TRACE` / `_MAX` | capture `(hidden -> experts)` traces (research) |

## What's ours, what isn't

The hot/cold MoE offload engine (placement, storage, swap, cold compute) is the
substrate, merged in [`../../server/src/common/moe_hybrid_*`](../../server/src/common/).
Spark is the calibration + bounded cache + predictor on top.

The sparsity-offload ideas are not ours:
- [**Pre-gated MoE**](https://arxiv.org/abs/2308.12066) (Hwang et al, ISCA 2024): decouple gating from execution, prefetch next-block experts.
- [**ProMoE**](https://arxiv.org/abs/2410.22134) / [**Fate**](https://arxiv.org/abs/2502.12224): learned predictor + proactive GPU caching, cross-layer gate prefetch.
- [**Deja Vu**](https://arxiv.org/abs/2310.17157) (Liu et al, ICML 2023) / [**PowerInfer**](https://arxiv.org/abs/2312.12456) / [**LLM in a Flash**](https://arxiv.org/abs/2312.11514): contextual / neuron-grain activation sparsity with hot/cold residency.

What we built:
- A traffic-calibrated placement pipeline keyed on real agent sessions, with a held-out split.
- A bounded LRU expert cache (spare-slot swap) wired through the engine's GPU FFN, so cold-misses fall to ~0 in fixed VRAM.
- A pre-gate trace capture + trainer, and the measurement of where fitted prediction tops out.

## Scope and limits

- **Single 24 GB GPU**, Laguna-XS.2 Q4_K_M is the validated target. Other MoEs need re-calibration.
- **Calibrate on the traffic you serve.** A profile from one distribution under-performs on another (conversation-calibrated vs pure-code: 6.6% vs 16% cold).
- **Locality required.** The cache wins because agent traffic has a small, stable working set; a no-locality workload thrashes toward PCIe-bound.
- **Full all-GPU speed needs a model change.** A fitted pre-gate caps at ~53% recall@8 (the pre/post-attention information gap); closing it means fine-tuning the gate, not a bigger predictor. See [RESULTS.md](RESULTS.md) §4.

## Citation

```bibtex
@software{luce_spark_2026,
  title  = {Luce Spark: traffic-calibrated hot/cold expert residency for MoE inference on consumer GPUs},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/optimizations/spark},
  year   = {2026}
}

@inproceedings{pregated_moe_2024,
  title     = {Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference},
  author    = {Hwang, Ranggi and others},
  booktitle = {ISCA},
  year      = {2024}
}
```

---

Apache 2.0 · [Lucebox](https://lucebox.com) · [Discord](https://discord.gg/yHfswqZmJQ)
