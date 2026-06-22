# Cached-AR-graph replay refactor — team-of-3 plan (token-sized)

Status: Phase 1 started 2026-06-22. Branch: pr/kvflash-moe-prefill-snapshot.

## Goal
CUDA graphs are compiled-on and running (server/build GGML_CUDA_GRAPHS=ON) but re-capture every token: do_ar_decode rebuilds the target graph per step (qwen35_backend.cpp do_ar_decode loop), and kv_start baked into the KV-write offset changes the data pointer each step, so ggml_cuda_graph_update_required invalidates the graph ("warmup reset" churn in logs). Make the AR-decode graph build ONCE and REPLAY → kill the ~42ms/tok launch overhead → ~1.7x decode at the 32-71K band.

## Honest scope
Launch-overhead lever only. Does NOT fix: 128K-on-24GB (KV-bandwidth wall → KVFlash pooling's job; real 128K fill = 3.74 tok/s on 24GB) or the HumanEval-short gap (per-step model cost, 3.6 vs 3.5).

## Merge gate (non-negotiable)
Token-for-token bit-identity at temp 0 across 4K/32K/71K before any stage merges. A graph captured at kv_start=N replayed naively corrupts KV silently, worst at 71K. NEVER merge a stage that fails identity. Report 3 speeds + nsys kernels/token drop (~2020 -> ~O(1)).

## Blockers, owners, token budgets
- A — build the AR graph once, cache on backend, feed inp_embed/positions via ggml_backend_tensor_set only (kill per-token build_target_step). Owner: Codex GPT5.5. ~350K tokens.
- C — kv_start -> SET_ROWS index tensor (fixed-base cache_k/v, dest row set per-step, read-at-launch not baked). Owner: Claude. Hottest correctness path. ~250K tokens.
- D — bucket FA read-window to a 4096 stride (re-capture once/4096 tok). Owner: GLM5.2. ~120K tokens.
- gate — bit-identity harness 4K/32K/71K token-for-token temp-0 + nsys. Owner: Claude. ~100K tokens.
- int — integrate A+C+D, per-stage gate, nsys verify, review. Owner: Claude. ~150K tokens.
- B — build flag: DONE (server/build GRAPHS=ON).
Total ~970K tokens.

## Phasing
- Phase 1 (parallel, 3 worktrees): A (Codex) || D (GLM5.2) || gate-harness (Claude). A touches do_ar_decode/graph_builders; D touches the FA-window READ in qwen35_target_graph.cpp; gate is bench-only.
- Phase 2: C (Claude) stacks on A (the reused graph must route kv_start through SET_ROWS).
- Phase 3: integrate -> bit-identity gate per stage (A -> +C -> +D) -> nsys confirm.

## Real bottleneck
Each ggml-cuda change needs a full ggml-cuda recompile (30-90 min, WSL2 OOM risk -> ninja -j1). That serial rebuild — not tokens — is the wall-clock constraint. Batch each owner's changes per rebuild; Claude owns the shared rebuild+gate cycle.

## Expected payoff
~1.7x decode at 32-71K. Combined with the shipped cache-persistence fix (commit 2192409b), the honest path to fast real long-context decode — distinct from the bandwidth-capped 128K reservation number.
