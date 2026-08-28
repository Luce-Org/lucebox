# PR 651 draft batching performance follow-ups

> Historical planning note: PR 659 implements the batched append projections and
> packed dynamic-convolution coefficient projections described below. Keeping draft
> hidden states on the device was evaluated but is not part of PR 659; the existing
> host handoff remains intentional until a separately measured change replaces it.

## Scope

PR 651 packs the main drafter projections across concurrent lanes. It keeps each lane's cache writes, RoPE, masks, and attention separate.

This note covers three follow-up optimizations that do not belong in the ownership cleanup:

- Batch the append projections.
- Pack the dynamic-convolution coefficient projections.
- Keep draft hidden states on the device through chain selection.

Treat each item as a measured change. Do not combine all three into one patch.

## Keep these invariants

- Keep `build_draft_kv_steps()` as the shared C1-C6 implementation.
- Preserve lane-local positions, masks, cache writes, RoPE, attention, and dynamic-convolution history.
- Rebuild a cached graph when the backend or the ordered lane-state pointers change.
- Keep C1 output equivalent to the existing single-lane graph.
- Keep dummy lanes after real lanes, and discard dummy proposals.

## Batch the append projections

`draft_kv_batch_build()` currently calls `build_draft_kv_append()` once per lane. Each call runs `draft_fuse_features()`, then the per-layer `wk` and `wv` projections.

Pack every lane's `ap_feat` columns before those shared-weight matrix multiplications. Split the projected columns before RoPE and `ggml_set_rows()`, because positions, destination rows, and caches remain lane-local.

The packed append path must preserve the fixed `a_step` width. Padded append rows must still write only to each lane's trash slot.

Suggested shape:

```cpp
bool build_draft_kv_appends(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const DraftWeights & weights,
    const std::vector<DraftKvAppendLane> & lanes);
```

Pass one lane from the normal graph and C1-C6 lanes from the batched graph. Delete the old singular builder after migrating both callers.

Prove the change with the existing multilane output comparison. Add cases with zero, partial, and full append counts so padding and trash-slot writes are covered. Measure draft compute separately at C1, C2, C3, C5, and C6.

## Pack dynamic-convolution coefficient projections

`build_draft_kv_steps()` calls `draft_dyn_conv_kernel()` once per lane for both attention and MLP. The function projects normalized hidden columns with shared weights. The temporal convolution in `draft_dyn_conv_apply()` is lane-local and must remain separate.

Pack the normalized columns before the coefficient projection. Slice the projected coefficients back into lanes, then call `draft_dyn_conv_apply()` per lane.

Do not pack the convolution history or apply step. Adjacent packed columns belong to different requests and must not influence each other.

Verify exact lane isolation with distinct inputs and histories. Run the existing single-lane-versus-packed comparison with dynamic convolution enabled. Record kernel count and draft compute time before and after the change.

## Keep draft hidden states on the device

The current boundary copies every lane's draft hidden block to the host in `draft_kv_batch_compute()`. `Qwen35SeqEngine::prepare_chain_drafts()` converts the host vectors to pointers. `dflash2_select_chains_batched()` then packs the candidates and uploads them to both the projection graph and the selector graph.

Replace that round trip with a device-resident contract. The batch graph should expose a packed hidden tensor or stable per-lane tensor views. The batched selector should accept those device tensors on the same backend.

Keep only token IDs, top-K scores, and final proposals as host results. Do not expose an unowned device pointer whose lifetime is shorter than either consumer graph.

This change crosses the draft and selector APIs, so ship it separately from append or dynamic-convolution packing. It also needs an explicit fallback when the draft and selector backends differ.

Verification must compare complete proposals, not only projected hidden values. Cover C1-C6, reordered active slots, bucket padding, selector-graph reuse, and graph rebuilds. Measure transfer bytes, synchronization count, selector time, and complete-round latency.

## Suggested order

1. Batch append projections. This extends the packing pattern already introduced by PR 651.
2. Pack dynamic-convolution coefficient projections. This is local to `build_draft_kv_steps()`.
3. Remove the host hidden-state handoff in its own PR. This changes ownership across the draft and selector graphs.

For every step, retain the previous implementation long enough to run an A/B output comparison. Delete it before merging the step.
