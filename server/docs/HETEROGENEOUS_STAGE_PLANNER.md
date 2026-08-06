# Generic Heterogeneous Stage Planner

This note records the reusable two-owner stage partition added to the common
MoE runtime, its correctness contract, and the measured decisions that should
not be rediscovered on each model or hardware pair.

## Scope

`heterogeneous_stage_planner.{h,cpp}` is independent of ggml, model family,
quantization, and device vendor. A caller supplies:

- a splittable work width;
- the alignment required by the tensor format or kernel;
- either an explicit peer fraction or calibrated owner rates;
- optional work already assigned to each owner.

The output is a complete, aligned partition. Fractions `0` and `1` represent
whole-stage ownership. Interior fractions keep at least one aligned unit on
both owners. Invalid or unsplittable inputs conservatively remain on the main
owner.

The balanced plan minimizes the predicted critical path:

```text
max((main_fixed_work + main_width) / main_rate,
    (peer_fixed_work + peer_width) / peer_rate)
```

Work and rate units are deliberately abstract. An adapter may use bytes and
bytes/us, FLOPs and FLOPs/us, or calibrated route-equivalents, provided every
input uses the same unit.

## Common MoE integration

The first consumer is the shared/dense SwiGLU FFN in
`moe_hybrid_storage.cpp` and `moe_hybrid_ffn_eval.cpp`:

1. Storage validates the gate/up/down tensor shapes, contiguity, and block
   alignment.
2. It copies only the selected peer rows into peer-owned weight storage.
3. Both owners compute their full gate/up/activation/down chain concurrently.
4. Each owner reduces its local routed and shared contributions before the
   heterogeneous boundary.
5. One deferred peer copy and one final join remain per layer.

The implementation supports an aligned width split and both whole-stage
endpoints. It deliberately refuses a shared FFN with an input-dependent output
gate: applying that gate independently to partial down projections would alter
the floating-point graph and is not bit-identical.

`DFLASH_MOE_TP_FUSED_OWNER_RESIDUAL=1` lets a supported coarse routed-owner
kernel consume its owner-local shared/dense result in the existing final
reduction. Unsupported backends keep the ordinary add, so the common scheduler
does not require a model-specific kernel.

## Policy controls

These are burn-in controls, not a permanent public configuration API:

| Variable | Meaning |
|---|---|
| `DFLASH_MOE_TP_SHARED_FFN_PEER_FRACTION` | `0` disables, `(0,1)` requests an aligned split, `1` assigns the stage to peer, and `auto` uses the rate model. |
| `DFLASH_MOE_TP_MAIN_RATE` | Calibrated main-owner rate for `auto`. |
| `DFLASH_MOE_TP_PEER_RATE` | Calibrated peer-owner rate for `auto`. |
| `DFLASH_MOE_TP_MAIN_TO_PEER_RATE` | Shorthand ratio when separate rates are unavailable. |
| `DFLASH_MOE_TP_MAIN_FIXED_WORK` | Work already assigned to main when the stage begins. |
| `DFLASH_MOE_TP_PEER_FIXED_WORK` | Work already assigned to peer when the stage begins. |
| `DFLASH_MOE_TP_FUSED_OWNER_RESIDUAL` | Fuse an owner-local dense/shared result into a supported routed-owner reduction. |

The legacy `DFLASH_DS4_*` spellings remain compatibility aliases. New model
adapters should use the common names or populate the equivalent
`MoeHybridConfig` fields.

## Qualification result

All entries below used the same q=5, 2K-context, 128-token exact-output run.
The expected response SHA-256 was
`0f785a7ffa406498aafb14553966eaed0f52220fed0f7cc016b66921d104d194`.

| Plan | Median tok/s | Decision |
|---|---:|---|
| Established route balance, shared stage on main | 88.602 | Qualified default |
| Owner-local residual fusion, no shared split | 88.637 | Exact final binary; neutral, retained as opt-in primitive |
| 25% shared-width peer shard, materialized owner adds | 86.750 | Reject |
| 12.5% shared-width peer shard, materialized owner adds | 86.557 | Reject |
| 12.5% shared-width peer shard, four-output join | 85.159 | Reject; implementation removed |
| 12.5% shared-width peer shard, single-join owner fusion | 86.636 | Reject |
| Repeated-expert route-slot alignment | 85.117 | Reject; remains disabled |
| Complete shared stage on peer, all routed work on main | 84.022 | Reject; unstable in this profile |

The narrow shared shards lose more kernel efficiency than their extra overlap
can recover. The complete peer stage also extends verification latency. These
results mean width sharding must remain disabled by default; the planner is a
portable mechanism to qualify other shapes and hardware, not evidence that one
fraction is universally beneficial.

## Current critical path

The qualified route-owner profile measured main-owner overlap at 82.63%, peer
overlap at 100%, launch skew at 8.41 us, and a main tail of about 58.35 us per
MoE layer (roughly 2.5 ms per speculative step). Verification is about 54.7 ms
of a 65.3 ms speculative step. Removing the measured owner tail alone is not
enough to reach 100 tok/s; the next optimization must reduce full-width expert
kernel time or overlap another independent stage, not add synchronization or
small matrix shards.

The timestamped two-device kernel profile, attention-split attribution, and
subsequent rejected expert weight-reuse experiments are recorded in
[`DS4_HETEROGENEOUS_PROFILE_20260806.md`](DS4_HETEROGENEOUS_PROFILE_20260806.md).

## Reproduction

Use `scripts/qualify_ds4_q5_amd.sh`. The runner records the source commit,
binary checksum, all policy controls, exact response hashes, individual runs,
and summary statistics. Relevant switches are:

```bash
SHARED_FFN_PEER_FRACTION=0
FUSED_OWNER_RESIDUAL=1
ALIGN_SHARED_IDS=0
EXPERT_TOP_K=4
DYNAMIC_ROUTE_BALANCE=1
DYNAMIC_MAIN_SLOTS_X4=13
```

Do not promote a candidate from a microbenchmark alone. It must pass unit/GPU
oracles, the exact response hash, warmups, and at least three measured model
runs. Keep rejected candidates disabled and record their result here.
