# MoE DDTree tree-verify — scoping & implementation plan

Status: **draft / scoping** (no functional code yet). Tracks wiring DDTree
tree-verify into the MoE backend (`qwen35moe`), the way it now works for
qwen35-dense and laguna in PR #436.

## Motivation

qwen35-dense and laguna run DDTree tree-verify (including under KVFlash) after
PR #436. MoE (`qwen35moe`, e.g. Qwen3.6-35B-A3B) has **no tree-verify at all**:
its spec path is chain-only, and measured chain accept-length (~1.87) barely
beats plain AR. DDTree is the path to a real spec win on MoE.

## Why MoE is not a port

Unlike qwen35/laguna, MoE has none of the tree infrastructure:

- No `DFlashTarget` abstraction — `qwen35moe_backend` is inline (no `target->`).
- Decode runs through the **pipelined hot/cold expert path**
  (`qwen35moe_pipelined_decode.cpp`, `pipe_state_`, `pipelined_decode_one_token`),
  built for single-token / chain decode.
- Spec is chain-only: draft → chain `verify_batch` → DeltaNet/conv SSM
  snapshot/restore rollback.
- There is no `verify_tree` / `rollback_to_tree` / `project_hidden_to_topk`.

So this is a feature build, not a gate flip.

## Plan

1. **`project_hidden_to_topk` (MoE)** — CPU `extract_draft_topk` (shared
   `common/ddtree.*`), exactly as qwen35/laguna. Low risk. (Note: laguna's GPU
   `ggml_top_k` over the full vocab overflows HIP shared-mem-per-block on
   gfx1151 — see PR #436 — so MoE must use the CPU path from the start.)

2. **`verify_tree` (MoE)** — tree-batched forward over N DFS-ordered nodes with
   ancestor-masked attention **and MoE expert routing per node**. This is the
   hard part: the pipelined hot/cold path is not tree-batched. Options:
   - (a) **v1 — full-expert batched forward** for the tree step (reuse
     `qwen35moe_ffn`, all experts resident), gated to the KVFlash-identity /
     in-pool case (context ≤ pool). Matches the qwen35/laguna constraint and
     keeps routing simple. **Recommended for v1.**
   - (b) extend the pipelined hot/cold path to tree-batched residency (per-node
     expert sets). Larger; defer.

3. **`rollback_to_tree` (MoE)** — DeltaNet/conv SSM snapshot/restore already
   exists (pool-neutral); add KV compaction to the accepted path (slot-mapped
   under KVFlash; identity case = contiguous).

4. **KVFlash identity gating** — reuse the PR #436 pattern: allow tree-verify
   while the pager is identity, `identity_prefix_covers()` guard + pool-bound at
   the call site, `alloc_span()` registration of tree-committed positions, prompt
   prefix registration. (MoE KVFlash already has pooled chunked prefill, so the
   >pool path stays on chain, same as qwen35.)

5. **Wire into the spec loop** — `ddtree_mode && supports_tree_verify()` → tree
   path; reuse shared `build_ddtree` / `follow_verified_tree`.

## Reuse

- DDTree core (`build_ddtree`, `follow_verified_tree`, `extract_draft_topk`) —
  `server/src/common/ddtree.*`.
- `KvFlashPager::identity_prefix_covers()` — added in PR #436.
- qwen35 `verify_tree` / `rollback_to_tree` — structural reference.

## Verification plan (Strix Halo gfx1151, ROCm)

- Build HIP; model `Qwen3.6-35B-A3B-UD-Q4_K_M`.
- AR vs chain-spec vs tree-spec: tok/s + accept on HumanEval + a long prompt.
- KVFlash on/off parity (in-pool); >pool stays on chain.
- Correctness: tree output matches chain/AR greedy on high-confidence tokens;
  needle recall.

## Risks / effort

- Main risk: the tree-batched MoE forward + expert routing for N nodes.
- Scoping v1 to in-pool / KVFlash-identity (full experts resident) keeps it
  tractable; the pipelined tree-batched variant is a follow-up.
- Materially larger than the qwen35/laguna ports (which reused existing tree
  infra) — this builds the infra for MoE.

## Checklist

- [ ] `project_hidden_to_topk` (CPU `extract_draft_topk`)
- [ ] `verify_tree` v1 (full-expert batched forward, ancestor-masked attn)
- [ ] `rollback_to_tree` (SSM snapshot/restore + KV compaction)
- [ ] KVFlash identity gating + `alloc_span` registration + prefix registration
- [ ] spec-loop tree path wiring
- [ ] bench + parity + needle on Qwen3.6-35B-A3B (gfx1151)
