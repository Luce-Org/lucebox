# KVFlash design notes

Mechanism details and tuning data behind [README.md](README.md); measured
tables in [RESULTS.md](RESULTS.md).

FlashMemory-style (arXiv 2606.09079) decode-time KV paging for the qwen35
target, designed to compose with pflash. Goal: the GPU footprint of the
full-attention KV cache is a hard O(pool) constant regardless of logical
context length, with paged-out chunks recallable bit-exact from host.

## Division of labor with pflash

pflash and the pager own different resources and compose cleanly:

| concern | owner |
|---|---|
| which prompt chunks the target ever elaborates | pflash (drafter scores, evict at prefill) |
| which elaborated chunks occupy GPU slots | KvFlashPager (this module) |
| prefill compute sparsity | pflash BSA kernels |
| decode-time KV growth (generated tokens) | KvFlashPager (page out cold generated chunks) |

pflash keeps the target from reading the huge context; the pager keeps
what the target HAS elaborated inside a fixed VRAM budget and makes every
eviction reversible. The drafter's chunk scores plug into
`KvFlashPager::score_hook` as the residency policy (LRU fallback in the
prototype).

## Mechanism

- Cache tensors are allocated at `pool_tokens` (e.g. 1024) instead of
  `max_ctx` (e.g. 131072). That allocation delta IS the memory saving:
  a mask over a full-size cache would save nothing.
- Logical positions map to physical pool slots at 64-token chunk
  granularity. The mapping rides the existing step-invariant
  `ggml_set_rows` KV append (`kv_write_rows` carries the physical slot;
  the `positions` input keeps the logical position for M-RoPE).
- Decode FA spans the whole pool with an EXACT slot-validity mask
  (`KvFlashPager::fill_slot_mask`): resident slots 0, free/paged-out -inf.
  The host-side mask rebuilds only when the pager epoch moves; the device
  upload happens before EVERY compute. That upload is mandatory, not an
  optimization: input tensors live in the gallocr compute buffer, whose
  regions are reused during graph execution, so a once-uploaded mask is
  garbage by the next step (this masqueraded as a "fattn NaN kernel bug"
  for a while — all-NaN logits from the second step on; production never
  hit it because its prefill refills masks per chunk). `--no-mask` falls
  back to maskless + zeroed freed slots (exp(-max) ~ 0, production's
  padded-span approximation, measured ~1% argmax flips).
- Page-out copies a chunk's quantized rows (per layer x K/V x head
  segments) to a host backing store and zeroes the slots; page-in writes
  them back. Quantized bytes + baked-in RoPE means the roundtrip is
  bit-exact and relocation is position-independent.
- Eviction protects sinks (first chunk) and the trailing window, mirrors
  FlashMemory's always-resident floor (their last-8K + decoded window).
  Unlike their sigmoid-threshold fetch (which leaks footprint at 500K,
  their §3.3.1), a fixed slot pool is a hard budget by construction.
- DeltaNet/conv recurrent state is fixed-size and never paged.

## What the prototype verifies (test_kvflash)

A. Baseline at logical ctx 128K: reference greedy sequence + KV bytes.
B. Relocation proof: same workload in a small pool with SHUFFLED block
   placement, teacher-forced — argmax must track the baseline.
C. Live paging: pool ≪ prompt+gen, eviction engaged; bit-exact
   page_out/page_in roundtrip; decode completes; KV bytes vs A ≥ 90% cut.

## Reselect (τ-step lookahead)

`KvFlashPager::reselect()` rebuilds the resident set as the top-pool chunks by
`score_hook` over all materialized chunks (resident or host-backed),
keeping sinks and the trailing window unconditionally. Page-outs run
first so recalls always find free blocks. This is the FlashMemory τ=64
loop's mechanism; the production caller invokes it every τ decoded
tokens with fresh drafter scores. Verified in test run D: an evicted
chunk recalled by a score flip, decode continues across the residency
change.

## Measured (lucebox RTX 3090, Qwen3.6-27B Q4_K_M, Q8_0 KV, 2026-06-11)

All gates PASS (exit 0). 64 timed steps per profile row, junk KV so the
FA span traffic is bandwidth-realistic:

| config | FA span | ms/step p50 | tok/s |
|---|---|---|---|
| baseline 8K   | 8192   | 35.1 | 28.5 |
| baseline 32K  | 32768  | 30.1 | 33.1 |
| baseline 128K | 131072 | 45.1 | 22.1 |
| pool 1K @128K logical | 1024 | 25.1 | 39.6 |
| pool 4K @128K logical | 4096 | 25.7 | 38.7 |

- attn-KV memory: 2304.0 -> 18.0 MiB (99.2% cut); whole cache buffer
  2653.6 -> 217.6 MiB, confirmed by VRAM deltas.
- At 128K-logical decode the pool is 1.8x FASTER than the full cache
  (45.1 -> 25.1 ms/step): FA cost is span-bound, the pool caps the span.
- Paging: page_out p50 1.26 ms, page_in p50 0.63 ms per 64-token chunk
  (~2.2 MiB, synchronous); 12 evictions over 1200 generated tokens
  amortize to ~0.01 ms/token. reselect() recalling with 20 page events
  took 21.3 ms — at τ=64 that is ~1% of decode time worst-case.
- Relocation equivalence: 0.83% argmax flips over 1200 teacher-forced
  tokens at shuffled placement (gate: ≤1%).
- Open harness question: the C-loop (live eviction) measured ~34 ms/step
  vs 25 ms for the identical config in the E-loop; suspected interaction
  of sustained-load GPU clocks with run ordering, not paging cost (12
  sync page events explain only ~0.01 ms/token). Re-measure under the
  production decode loop during integration.

## Full LSA loop (drafter as Memory Indexer) — measured

Test run F implements the paper's complete inference paradigm with the
pflash drafter (Qwen3-0.6B, `/opt/lucebox/models/drafter/`) standing in
for the trained indexer: prompt (2048) larger than the pool (1024) so
prefill itself evicts, then every τ=64 decoded tokens the drafter
rescores the full sequence (tail attention = indexer query, chunk means
via `drafter_chunk_scores`), `score_hook` receives the fresh scores, and
`reselect()` repages the pool.

Measured (RTX 3090, target Qwen3.6-27B Q4_K_M + drafter co-resident):
- 31.2 tok/s with the loop active; 12 rescores over 768 generated tokens
- 43 genuine drafter-driven recalls of previously evicted context
- indexer rescore p50 = 245 ms (full 0.6B re-prefill at ~2-2.8K tokens —
  ~12% decode overhead at τ=64; drops to ~ms once the drafter's own KV
  is persisted and only the new τ tokens are pushed through it)
- reselect p50 = 7.5 ms

vs the paper: their indexer is a trained <0.1% projection head (cheaper
queries, backbone-supervised labels); ours is the existing 0.6B drafter
(training-free, already shipped for pflash). Their sigmoid threshold
leaks footprint at scale (their §3.3.1); our fixed pool is a hard cap.

## Production integration (daemon)

The pool is wired into the qwen35 backend behind `--kvflash <tokens>`
(env `DFLASH_KVFLASH`; rounded to a 256 multiple) + `--kvflash-tau <N>`
(env `DFLASH_KVFLASH_TAU`, default 64). Pieces:

- `create_target_cache(..., ctx_alloc)`: attention tensors allocated at
  pool capacity; `cache.max_ctx` stays the logical bound.
- `do_prefill`: prompts that fit the pool land identity-mapped
  (`kvflash_sync_prefill` rebuilds the pager map per request/restore);
  LARGER prompts switch to pooled chunked prefill — pager-chunk batches,
  slot-mapped set_rows writes, a slot-space mask per chunk, live
  eviction. Constant VRAM, linear time (qwen35 only so far).
- `do_ar_decode`: `build_target_step(..., kvflash_mask=true)` keeps the
  step-invariant set_rows write active alongside the slot mask;
  `kv_write_rows` carries the pool slot; the mask uploads per step;
  every τ generated tokens `kvflash_maybe_reselect` rescores + repages.
- Policy is agnostic by construction: `KvFlashScorer` (common/) is the
  interface; with no scorer the pager runs pure LRU (zero pflash
  dependency). When pflash loads its drafter, `KvFlashDrafterScorer`
  (qwen3/) attaches automatically and reselect becomes drafter-driven.
- Spec decode (chain mode) runs ON the pool: verify_batch slot-maps the
  draft block via per-token kv_write_rows and builds a slot-space mask
  (resident committed positions + causal among draft tokens). Rejected
  drafts need no rollback: the pos < base_pos validity rule excludes
  their slots until the replay rewrites them. All four spec KV-write
  sites (verify, both replays, stall-prefix) route through this one
  function. Verified on the daemon: accept_rate 15.4-15.6% pooled vs
  15.3% pool-off (matched avg_commit 3.47 vs 3.45), coherent output
  through a mid-generation pool wrap with live eviction. The daemon's
  --ddtree config (chain verify + fast rollback) also runs on the pool
  (accept 14.6% pooled vs 13.9% off); only the harness-only tree-verify
  graphs (test_dflash) remain not pool-aware.
- LAYOUT TRAP (cost a day of debugging): kv_write_rows is
  [n_tokens, n_head_kv] ne0-major — element (token i, head h) lives at
  i + h*n_tokens (ggml_set_rows asserts b->ne[1] == c->ne[0]). A
  transposed fill scrambles per-head row targets for every multi-token
  write while single-token fills (all entries equal) hide the bug
  completely.
- Post-generation snapshots are skipped once cur_pos exceeds the pool
  (pooled snapshots need page-table serialization; prefill-time
  snapshots still work).

## Production smokes (dflash_server on lucebox 3090, 2026-06-11)

1. WITHOUT pflash (agnostic LRU): `dflash_server <27B> --kvflash 1024`.
   41-token prompt + 1400 generated = 1441 logical through a 1024-slot
   pool (live LRU eviction mid-request). Coherent story end to end,
   36.9 tok/s, clean finish. Second request (per-request pager reset) ok.
2. WITH pflash: `--kvflash 2048 --prefill-compression always
   --prefill-threshold 256 --prefill-drafter <Qwen3-0.6B>`. Compression
   1468 -> 60 tokens, then `[kvflash] drafter scorer attached (tau=64)`
   automatically; 400 coherent tokens answering from the compressed
   context. Same binary, zero pflash-specific configuration on the pool.

Ops note: the init banner is flushed now, but generally `nohup` +
redirected stdout block-buffers printf output — kill the process (atexit
flush) before concluding a code path didn't run.

## Quality matrix (synthetic NIAH, needle recall /16, teacher-forced)

| context | residency | LRU d=10/50/90% | drafter d=10/50/90% | control |
|---|---|---|---|---|
| 8K   | 25%   | 0 / 0 / 16 | 15 / 15 / 16 | 16/16 |
| 8K   | 9%    | 0 / 0 / 0  | 15 / 15 / 15 | 16/16 |
| 32K  | 25%   | 0 / 0 / 16 | 15 / 15 / 16 | 16/16 |
| 32K  | 9%    | 0 / 0 / 0  | 15 / 15 / 15 | 15-16/16 |
| 256K | 6.25% | 0 (d=0.5); 16/16 in-window | 14 / 15 / 15 | (in-window LRU = control) |

Drafter-scored residency retains 88-100% of perfect needle recall at every
depth down to 6-9% residency from 8K to the model's native 256K maximum;
recency-only LRU retains zero outside its tail window. 256K logistics on
the RTX 3090: ~6.5 min linear pooled prefill, 4.22 GiB host backing,
~18 GiB VRAM total, 46 s bisected rescore (drafter forward ceiling ~65K
per segment).

## Tuned defaults (from the matrix)

- Ship drafter scoring whenever a drafter is available; pure-LRU mode is
  recency-only and must be documented as such.
- Pool ~25% of expected context is the conservative default; 9% measured
  safe for retrieval-style work.
- tau adapts: rescore costs ~0.11 ms/history-token, so the effective
  reselect interval is max(configured tau, history/45), capping rescore
  overhead near 15% of decode time.

## Per-architecture integration

The pager core is architecture-blind; each backend routes its own KV writes
and masks through it. What differs per arch:

- **qwen35** (reference): masked set_rows decode, slot-mapped chain-spec
  verify, drafter scorer auto-attach. Everything in RESULTS.md.
- **qwen35moe** (Qwen3.6-35B-A3B): inherits the qwen35 path all-GPU. The
  Spark hybrid pipelined decode keeps its per-layer cached CUDA graphs:
  `pipelined_decode_one_token` takes a `kv_slot`, the cached FA span clamps
  to the pool (so the graph stops rebuilding once the window hits pool
  size), and the pool span stays MASKLESS like the rest of that path — the
  pager zeroes freed blocks (page-out and `zero_free_blocks()` on request
  reset), so evicted slots contribute exp(-max) ~ 0, production's own
  padded-span approximation. Hybrid spec decode (literal-offset KV writes)
  falls back to pipelined AR under kvflash.
- **laguna**: ALL 40 layers pooled (full + SWA share the pager).
  `laguna_step` / `laguna_step_hybrid` take a const pager; both masks are
  built in SLOT space via `fill_slot_pos` (the causal / sliding-window
  conditions evaluate on the position each slot holds). SWA exactness:
  `tail_window_chunks >= sliding_window/64 + 1`, so positions inside the
  window are never evicted. The per-layer hybrid decode fallback and
  NO_KVPAD / PAD_CPY / no_mask ablations are refused under kvflash.
- **gemma4**: pools FULL-attention layers only — SWA layers already use
  sliding-window ring buffers and KV-reuse layers share their source's
  tensors. The full mask is slot-space; the SWA ring path is untouched.
  `--fa-window` (sparse full-attn) and kvflash are mutually exclusive.
  DFlash spec verify is slot-mapped (gemma4_verify_batch gains set_rows
  inputs + the slot-space causal mask; its KV-truncation rejection
  semantics map directly onto the pool's validity rule). Measured:
  identical acceptance pooled vs full (407/3104 = 13.1%, avg_commit
  3.09, identical text).

Policy: drafter-scored residency is the default on all four archs. The
server probes for the Qwen3-0.6B next to the model (or --prefill-drafter)
and lazy-loads it at the first reselect; `--kvflash-policy lru` opts out.
qwen35/qwen35moe feed the drafter target ids directly; laguna/gemma4 use
KvFlashCrossTokScorer (detokenize -> re-tokenize -> score -> map back by
char spans; functional but untuned, see RESULTS). `--kvflash auto` sizes
the pool from free VRAM at the model's KV density, capped at the decode
speed knee (16384 default).

Snapshots on laguna/gemma4 are refused once a chunk has relocated
(page_outs > 0); identity-layout snapshots before that still work.

## Follow-ups

Done since the prototype: pooled chunked prefill in the qwen35 daemon
(prompt > pool, eviction during prefill), spec-decode chain verify on the
pool, VRAM-aware auto sizing, cross-tokenizer scoring for laguna/gemma4.

Open:
1. Drafter KV persistence for the indexer (incremental rescore: push
   only the new τ tokens through the drafter; kills the ~240 ms re-prefill).
2. Pooled chunked prefill for laguna/gemma4 (qwen35-only today).
3. Pooled snapshot save/restore (serialize the page table + host store).
4. Async paging on a copy stream (currently synchronous
   ggml_backend_tensor_get/set between steps).
5. Teacher-forced NIAH harness for non-qwen archs + cross-tok scorer
   tuning (tail window, normalization).
