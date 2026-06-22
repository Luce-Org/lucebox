# Unify disk-snapshot + KVFlash page-table ledger so the QK residency scorer composes with PR #372

Date: 2026-06-22
Branch context: pr/kvflash-moe-prefill-snapshot (commit 2192409b just landed the chunk-aligned serialize fix)
Scope: SCOPING ONLY — root-cause + token-sized plan. No implementation.

---

## TL;DR

They still do NOT compose. The just-landed chunk-aligned `serialize(max_chunks)` fixed the
*save* boundary, but the **restore path throws away both the relevance ordering and the QK
pool**. Two independent gaps, one shared root cause: the snapshot blob is a **raw, position-
ordered KV byte dump with no residency ledger and no pooled-key sidecar**. On restore:

1. **Prefill is NOT collapsed on a pooled hit.** `restore_and_generate_impl` deserializes the
   pager (server/src/qwen35/qwen35_backend.cpp:982-990) and then, for any prompt longer than
   the snapshot, calls `do_prefill(req.prompt, ...)` with `cache_.cur_pos = 0` over the FULL
   prompt (:1029-1032). The deserialized KV is overwritten, not consumed. This is exactly the
   "restore=true but prefill_s does not drop" symptom in memory UPDATE 2026-06-13 13:33.

2. **The QK pool is never rebuilt from the restored blob.** Deserialize restores raw KV bytes
   but `kvflash_qk_pool_` (the pooled, L2-normalized post-RoPE keys the scorer reads) is left
   empty; `kvflash_qk_pooled_upto_` is not reset to re-pool. The scorer that was load-bearing
   on turn N (needle 15/16) starts turn N+1 blind — every chunk scores `missing_score=-2.0`
   (kvflash_qk.h:50) until re-pooled, which only happens incidentally if a fresh re-prefill
   walks those chunks resident again (kvflash_qk_pool_to, :1455-1466).

So restore re-pages-in by **raw logical position** (deserialize loop :549-566 calls
`slot_for(c * chunk_tokens)` in chunk order 0,1,2,…), **losing the QK-relevance ordering**
the scorer chose on the previous turn. There is no ledger recording "chunk c had rank r".

---

## ROOT CAUSE (file:line)

### The "page-table ledger" today = `std::vector<ChunkState>`, residency-only
`KvFlashPager::chunks_` (kvflash_pager.h:720) is the ledger. Per chunk it records:
`{ int block (physical slot, -1=not resident); bool on_host; uint64_t last_use; host_data }`
(:575-584). It records **physical residency**, NOT relevance. Relevance lives entirely
**outside** the pager, transiently, in `kvflash_scores_` (the `score_hook` closure,
:1547-1549) which is recomputed every τ steps from `kvflash_qk_pool_` + the live decode query.

### `serialize()` captures KV bytes only — no ledger, no scores, no pooled keys
`serialize(max_chunks)` (kvflash_pager.h:471-519) writes an 8-byte magic + 6×uint32 header
(`nc, chunk_tokens, n_head_kv, k_seg_bytes, v_seg_bytes, chunk_bytes`) then `chunk_bytes_`
of raw KV per chunk. It serializes **neither** `block`/residency **nor** any QK score **nor**
the `KvFlashQkPool` normalized keys. `deserialize()` (:528-572) reads those same KV bytes and
re-allocates blocks in **naive ascending chunk order** (:549-565). The restored residency is
"chunks 0..nc-1 all want to be resident, assigned blocks in order" — i.e. **LRU/identity, not
QK-ranked**. The scorer's turn-N decision is gone.

### Restore re-prefills instead of consuming the blob
restore_and_generate_impl (:1029-1036): `if (snap_pooled && kvflash_active())` →
`reset_recurrent_state; cache_.cur_pos = 0; committed = do_prefill(req.prompt, ...)`. The
full-prompt re-prefill is the documented "pooled restore with kv_offset!=0 unsupported once
the pool is full" limitation (do_prefill :1152-1164). The deserialized KV is immediately
clobbered by `kvflash_pager_.reset()` inside do_prefill's pooled branch (:1167). Net: the
disk snapshot's only realized benefit on a long-context pooled hit is **nothing** — prefill_s
does not drop.

### PR #372 q8/both-KV interaction — NOT a byte-format conflict, but a silent-correctness trap
PR #372 set the KV cache dtype for BOTH K and V (`kv_k = kv_v = GGML_TYPE_Q8_0`,
qwen35_backend.cpp:165, resolved via `dflash::resolve_kv_types`). The serialize format is
**dtype-agnostic by construction**: it copies `chunk_bytes_` raw bytes where
`chunk_bytes_ = (k_seg_bytes_ + v_seg_bytes_) * n_head_kv * n_layers` derived from the live
tensor `nb[1]` strides (kvflash_pager.h:133-135). The deserialize header guard
(:542-545) rejects a blob whose `k_seg_bytes/v_seg_bytes/chunk_bytes` differ from the current
cache — so a snapshot taken under q8 and restored under a different KV dtype is **refused**,
not silently corrupted. GOOD. BUT the header does **not** record the dtype *enum* itself, only
the byte sizes; two dtypes with equal row size (none today, but a future q8↔some-8bit) would
pass the guard and corrupt. Low-probability, note it.
The QK scorer is already q8-correct: KvFlashQkPool::pool_chunk dequantizes via
`ggml_get_type_traits(t->type)->to_float` (kvflash_qk.h:131,141) and the basis note (:9-13)
documents FWHT-rotation invariance. So PR #372's q8 path does NOT conflict with the QK math —
it conflicts only by **sharing the same snapshot that drops the ledger**.

### THE SEAM (the exact incompatibility, quoted)
- (a) YES: `serialize()` writes a residency-blind, score-blind blob; the QK scorer's runtime
  path (`reselect()` driven by `score_hook`) is never re-seeded on restore. The ledger the
  snapshot *could* carry is ignored because it does not exist in the format.
- (b) PARTIALLY: there are effectively **two** representations — the pager's `ChunkState`
  (residency) and the scorer's `kvflash_qk_pool_` + `kvflash_scores_` (relevance). The
  snapshot serializes a projection of the *first* (KV bytes) and **none** of the second.
- (c) YES: deserialize re-pages-in by raw position (`slot_for(c*chunk_tokens)` ascending,
  :565), so even the residency it does restore is position-ordered, not QK-rank-ordered.
  Combined with the re-prefill at :1032, the QK-relevance ordering is doubly lost.

---

## THE UNIFICATION — one shared page-table ledger

Define a single serialized ledger that the pager, the snapshot, AND the QK scorer agree on.
Per chunk `c` the blob must record:

| field            | source today                          | why                                           |
|------------------|---------------------------------------|-----------------------------------------------|
| chunk index `c`  | implicit (array position)             | logical position = c * chunk_tokens           |
| KV bytes         | already serialized                    | bit-exact recall                              |
| `was_resident`   | `ChunkState.block >= 0`               | restore the SAME resident set, not all-want   |
| `qk_rank`/`score`| `kvflash_scores_[c]` (transient)      | restore relevance ordering w/o re-scoring     |
| pooled QK keys   | `kvflash_qk_pool_.data(c)` (L*Hkv*D f32, normalized) | scorer hot on turn N+1 with no re-pool |
| kv dtype enum    | `attn_k_[0]->type` (NOT recorded today)| guard against equal-rowsize dtype swap        |

### Files / structs to change (server-side only — fast relink, NO ggml-cuda rebuild)
1. **server/src/common/kvflash_pager.h** — extend the serialize/deserialize format:
   - Bump magic (e.g. "KVFLASH1"), add header fields: `kv_k_type`, `kv_v_type` (uint32 enum),
     `has_ledger` flag, `qk_dims` (n_layers, n_kv_heads, head_dim), `pooled_key_bytes`.
   - Per chunk append: `uint8 was_resident`, `float score`, and (when has_ledger) the
     L*Hkv*D pooled-key floats. Keep KV-byte section unchanged so old guards still work.
   - `deserialize()` honors `was_resident` (only those get blocks; rest stay host-backed) and
     reselects in **descending score order**, not ascending chunk order.
2. **server/src/common/kvflash_qk.h** — add `KvFlashQkPool::serialize_keys(out)` /
   `restore_keys(c, ptr)` (raw memcpy of the per-chunk float vector; format already flat at
   keys_[c], :166). Pure, testable.
3. **server/src/qwen35/qwen35_backend.cpp**:
   - `snapshot_save_pooled_at` (:510) — pass the QK pool + current `kvflash_scores_` into
     serialize so the ledger is captured.
   - restore path (:982-990) — after deserialize, call a new
     `kvflash_qk_restore_from_blob()` that repopulates `kvflash_qk_pool_` and sets
     `kvflash_qk_pooled_upto_ = nc`, then seed `score_hook` from restored scores so the FIRST
     reselect is QK-ranked without a re-prefill.
   - **The prefill-collapse fix (the real perf win):** make the pooled restore consume the
     deserialized KV for the shared prefix instead of re-prefilling from 0. Either (i) support
     `kv_offset != 0` in the pooled prefill path (do_prefill :1152-1164 currently refuses it),
     or (ii) restore the pooled prefix resident + only prefill the delta. This is the
     load-bearing change; the ledger fields are what make it *correct* (right chunks resident).

### INVARIANT
> A chunk the QK scorer kept resident on turn N, with its relevance rank, is restored
> resident on turn N+1 under PR #372's q8 KV layout, WITHOUT re-scoring from scratch and
> WITHOUT re-prefilling the shared prefix. Formally: for the top-`n_blocks` chunks by
> turn-N QK score, `is_resident(c)` is true immediately after `deserialize()`, and
> `kvflash_qk_pool_.has(c)` is true, before the first decode step of turn N+1.

---

## TOKEN-SIZED PLAN (phases, owners, budgets)

Build constraint: ALL changes are server-side headers + qwen35_backend.cpp → **fast relink**
against pre-built CUDA artifacts (per project_dflash_server_build_oom_workaround). No
ggml-cuda kernel changes, so NO slow full rebuild. Each phase ends green on
server/test/test_kvflash_*.

- **Phase 1 — Ledger format (Claude/Opus design + sisyphus-junior impl).** ~1.5K tok design,
  ~4K tok impl. Extend serialize/deserialize in kvflash_pager.h with was_resident + score +
  dtype enum (NO pooled keys yet). Add `test_kvflash_pager` cases: round-trip preserves
  residency set + score ordering; old-magic/dtype-mismatch still rejected. Pure, no GPU.
- **Phase 2 — QK pool persistence (sisyphus-junior).** ~3K tok. KvFlashQkPool serialize/
  restore_keys + wire into snapshot_save_pooled_at and the restore branch. Extend
  test_kvflash_qk: restored pool gives bit-exact scores vs freshly-pooled (the 8e-08 bound
  from memory 13:33 is the gate).
- **Phase 3 — Prefill collapse on pooled hit (Codex or GLM5.2 for the do_prefill kv_offset
  path; Opus reviews the FP-determinism risk).** ~6K tok — the hard one. Make pooled restore
  consume restored KV for [0, snap_pos) and prefill only the delta, keeping QK-ranked
  residency. Must preserve the chunk-aligned bit-identity contract (do_prefill :1201-1211).
- **Phase 4 — Merge gate (Opus orchestrates, sisyphus-junior runs).** 128K multi-turn
  agentic run (the 7-turn tq3_0 harness from memory UPDATE 2026-06-13). Asserts below.

### MERGE GATE (must all hold on ONE 128K multi-turn agentic run, q8/both-KV + --kvflash-policy qk)
1. **QK residency preserved across restore:** log `is_resident(c)` for top-rank chunks before
   turn N+1 decode == the turn-N resident set (≥ 95% overlap; sinks/tail exact).
2. **No re-score from scratch:** turn N+1 first reselect uses restored scores (assert
   `kvflash_qk_pooled_upto_ == nc` immediately post-restore; rescore wall ≈ 0 on the restored
   prefix).
3. **Warm-prefill cheap:** `prefix_len > 0` AND `prefill_s` drops vs cold (the symptom that
   failed on 2026-06-13: prefill_s must actually fall, not just restore=true).
4. **Decode quality held:** needle ≥ 15/16 (the standalone QK number) AND decode tok/s within
   noise of QK-alone (~27-30 tok/s); wall NOT 3× the baseline (the 253s→1836s regression must
   be gone).
5. **PR #372 win intact:** q8/both-KV decode +% preserved (both features ON simultaneously).

---

## RISKS

1. **#1 likely-to-NOT-compose-cleanly: Phase 3 FP-determinism.** The pooled-prefill path is
   deliberately chunk-aligned to stay bit-identical to the no-cache path (do_prefill
   :1201-1211 FIX(bug2)). Consuming restored KV for a partial prefix and prefilling only the
   delta changes batch shapes at the seam → greedy output can diverge on cache hits (exactly
   the bug2 class). Mitigation: keep the snapshot boundary chunk-aligned (already done by
   2192409b) and prefill the delta in the SAME chunk geometry; gate on greedy bit-identity at
   the restore seam, not just "output looks fine."
2. **Restored residency vs live tail/sink protections.** deserialize must not restore a
   resident set that violates `min_pool_tokens` or the deadlock guard (pin_range :245). A
   turn-N resident set + turn-N+1 tail window may exceed the pool. Mitigation: clamp restored
   resident set to `n_blocks - (sink + tail + 2)` by score before assigning blocks.
3. **Pooled-key blob size at 128K.** L*Hkv*D f32 per chunk × (128K/64) chunks ≈ 16 layers ×
   4 × 256 × 4B × 2048 chunks ≈ 134 MB of f32 sidecar per snapshot. Acceptable on disk but
   audit the PREFIX_SLOTS in-memory copy (snap.kvflash_blob is held for same-request restore,
   :538). Consider re-pooling from restored KV instead of serializing keys if memory-bound
   (re-pool is cheap: pool_chunk reads resident KV, the bytes we already restore).
4. **dtype-enum guard is additive only.** Recording kv_k_type/kv_v_type prevents the
   equal-rowsize silent-corruption trap but old blobs (new magic) are correctly rejected —
   confirm the server tolerates a cold-start cache miss when format version bumps.

## NOT-stale check
Memory project_qk_scorer_kvflash_win UPDATE 2026-06-13 is CURRENT. Commit 2192409b fixed only
the SAVE boundary (chunk-aligned serialize). The RESTORE-side gaps (re-prefill + dropped QK
pool + position-ordered re-page-in) are unaddressed in the benched tree. They do NOT compose
today. The real remaining gap is Phase 3 (prefill collapse) gated by the ledger (Phases 1-2).
