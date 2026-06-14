<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce KVFlash</h1>

<p align="center">
  <strong>Lookahead sparse attention for dflash. Bounded KV residency on one GPU.</strong><br/>
  The attention KV cache lives in a fixed pool of slots; cold 64-token chunks page to host RAM, bit-exact and recallable.
  With pflash, its drafter doubles as a Memory Indexer that recalls the context the generation needs next.<br/>
  Qwen3.6-27B Q4_K_M on a single RTX 3090: <strong>native 256K context at 38.6 tok/s with 72 MiB of resident KV</strong>,
  needle recall 88-100% at 6% residency, harness accuracy unchanged (36/36 vs full cache).
</p>

---

```
                         decode tok/s   KV in VRAM (Q8_0)   needle (d=10/50/90%)
full cache  @  64K            27.8        1152 MiB        16/16
full cache  @ 128K            19.6        2304 MiB        16/16
full cache  @ 256K            13.1        4608 MiB        16/16
KVFlash 4K  @  64K            38.6          72 MiB        14/16
KVFlash 4K  @ 128K            38.6          72 MiB        14/16
KVFlash 4K  @ 256K            38.6          72 MiB        15/16
```

Decode speed is flat at any context length (the per-step KV read is pool-sized,
not context-sized), prefill is up to 2.8x faster, and a 256K prompt that costs
4.6 GiB of VRAM as a full cache costs 72 MiB resident + 4.2 GiB of host RAM.
(The full-cache 256K rows are measured, not extrapolated: they fit the 24 GB
card only thanks to Q8_0 KV; with F16 KV the cache alone is 9.2 GiB and 256K
does not fit at all.)

## Usage

```bash
# recommended: drafter-scored residency, pool auto-sized from VRAM.
# pass --prefill-drafter so the drafter is guaranteed (no silent LRU fallback).
dflash_server model.gguf --max-ctx 32768 --kvflash auto \
    --prefill-drafter /opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf

# drop the path to auto-probe (model dir, drafter/, draft/, /opt/lucebox/models/drafter/);
# falls back to LRU if none is found, so check the banner reads policy=drafter
dflash_server model.gguf --max-ctx 32768 --kvflash auto

# explicit pool size, recency-only LRU
dflash_server model.gguf --max-ctx 32768 --kvflash 8192 --kvflash-policy lru
```

Drafter-scored residency is the DEFAULT policy on every model family:
the server probes for `Qwen3-0.6B-BF16.gguf` next to the model (same
dir, `drafter/`, `draft/`, then `/opt/lucebox/models/drafter/`) and
lazy-loads it on the first reselect; `--prefill-drafter` overrides the
location, prefill compression can stay off either way. Qwen-family
targets feed the drafter their ids directly; laguna and gemma4 bridge
the tokenizer gap with `KvFlashCrossTokScorer` (relevance is a property
of the TEXT, so the target's history is detokenized, re-tokenized for
the drafter, scored, and mapped back to chunk boundaries by character
spans). LRU is the fallback when no drafter is found (the banner says
which policy you got) or the explicit choice via `--kvflash-policy lru`.
`auto` sizes the pool from the GPU, not a fixed fraction: half of the
free VRAM left after weights (minus a reserve for compute buffers and
the drafter), converted at the model's KV density, capped where decode
speed stays near the flat optimum (16384 tokens by default,
`DFLASH_KVFLASH_MAX_POOL` to override) and at `--max-ctx`. Bigger pools
mean more resident chunks and fewer forced evictions of useful context;
the cap keeps the per-step KV read small enough that decode stays near
the small-pool speed.

- `--kvflash <tokens|auto>`: resident pool size (rounded to 256; clamped to
  `--max-ctx`; floored at the protected minimum — 512 for qwen-family and
  gemma4, larger on laguna where the SWA window stays resident — so
  eviction always has a victim). Env: `DFLASH_KVFLASH`.
- `--kvflash-tau <N>`: reselect interval floor (default 64; the effective
  interval grows with history so rescore overhead stays ~15% of decode).
  Env: `DFLASH_KVFLASH_TAU`.

Sizing rule: without a drafter, pool >= prompt + generation headroom
(LRU is recency-only memory — an undersized pool can evict the question
itself). With pflash's drafter attached, 25% of the expected context is a
conservative default and 6-9% is measured safe for retrieval workloads.

## Model support

`--kvflash` works on every architecture the daemon serves:

| arch | models | decode path | policy | notes |
|---|---|---|---|---|
| qwen35 | Qwen3.5/3.6-27B | masked set_rows decode + slot-mapped spec verify | LRU or pflash drafter | reference integration; all RESULTS.md numbers |
| qwen35moe | Qwen3.6-35B-A3B | pipelined hybrid decode (Spark) + all-GPU | LRU or pflash drafter | maskless pool span (zero-row approximation, same as production padding); hybrid spec decode runs on the pool |
| laguna | Laguna-XS.2 | single-graph hybrid + all-GPU, slot-space full+SWA masks | LRU or drafter (cross-tok, untuned) | pager covers all 40 layers; protected tail >= sliding_window keeps SWA exact |
| gemma4 | Gemma4 26B-A4B / 31B | masked decode + slot-mapped spec verify, slot-space full mask | LRU or drafter (cross-tok, untuned) | pools FULL-attention layers only (SWA layers already ring-buffer) |

Non-qwen targets use the cross-tokenizer scorer (detokenize target ids,
re-tokenize for the drafter, score, map back by char spans); the
`KvFlashScorer` seam stays open for native indexers.

## How it works

- **Pool**: attention KV tensors are allocated at pool size; a pager maps
  logical positions to slots at 64-token chunk granularity. Cold chunks
  move to a host backing store (~0.6 ms/chunk) and return bit-exact.
- **Mask**: attention spans the pool with a slot-validity mask, uploaded
  before every compute. Exact, and free (25.10 vs 25.52 ms/step maskless).
- **Reselect**: every tau decoded tokens the scorer re-ranks all chunks
  (resident or host-backed) and `reselect()` repages the pool — the
  lookahead loop from FlashMemory (arXiv 2606.09079), with the pflash
  drafter standing in for their trained indexer, and a hard capacity cap
  their threshold mechanism lacks.
- **Spec decode**: chain-mode verify is slot-mapped (per-token
  `kv_write_rows` + slot-space mask); rejected drafts need no rollback —
  their slots are excluded by the validity rule until rewritten.
  Acceptance parity with the full cache (15.4-15.6% vs 15.3%), with or
  without the --ddtree configuration (fast rollback only snapshots
  DeltaNet state, which is never pooled).
- **Prefill**: prompts larger than the pool prefill in 64-token chunks at
  constant VRAM (linear time; 256K in ~5.9 min on the 3090).

Quality verdict (harness ground truth, base-vs-base control included):
full results in [RESULTS.md](RESULTS.md). Outputs are not guaranteed
byte-identical to the full cache on long generations (the masked kernel
path rounds differently — a different deterministic lineage), but
correctness is identical: 36/36 vs 36/36 across HumanEval, GSM, MATH, and
agent suites.

## Files

- `server/src/common/kvflash_pager.h` — pool, page table, host store, reselect
- `server/src/common/kvflash_scorer.h` — chunk-relevance policy interface
- `server/src/qwen3/qwen3_kvflash_scorer.{h,cpp}` — pflash-drafter scorer
  (tail attention; bisects on allocation pressure)
- `server/src/qwen35/*` — cache `ctx_alloc`, masked pooled decode, slot-mapped
  spec verify, daemon flags
- `server/test/test_kvflash.cpp` — verification suite (A-F), `--niah`,
  `--niah256`, `--longab`
- [DESIGN.md](DESIGN.md) — mechanism details and tuning notes
