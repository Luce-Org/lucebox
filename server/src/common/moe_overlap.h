// MoE overlap steps: one step of a model whose MoE layers are split between two GPUs of one runtime, the target
// (attention, dense work, the shared expert and the routed experts it holds) and the expert GPU (the routed experts
// the target does not hold). Used by DeepSeek V4.1's paged steps (deepseek4); written for any such split, so the
// Qwen3.8-Flash-Next split can move onto it.
//
// ggml's scheduler copies a split's inputs from other devices before any node of the split runs, so the two GPUs
// only work at the same time when
//   1. the expert GPU's nodes come right after the routes in graph order: its routed sum is queued as soon as the
//      routes are known, ahead of the target's own routed experts and shared expert (moe_overlap_queue_expert), and
//   2. the target's split ends right before the join, the first node that reads the expert GPU's sum, so the
//      target's experts before it do not wait for that copy (moe_overlap_join, moe_overlap_mark_join).
// Both are scheduling: every node computes the same bytes as in one serial graph.
//
// A step of several rows (sequences side by side) whose products are batch-invariant
// (ggml_backend_cuda_set_mmvq_batch_invariant: each column of a product gets the arithmetic it gets alone) also
// shares the expert weights between its rows:
//   3. each GPU builds the one-row expert branch once for all rows, one product per projection over every row's
//      routes, and the rows routed to one expert share its weight reads (moe_overlap_expert_product). A row's bytes
//      are those it gets alone, at every width and row order.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

namespace luce::common {

// 1: queues the expert GPU's routed sum, and the nodes it needs, in gf. Call once the routes are built and before the
// target's routed experts and shared expert are queued.
inline void moe_overlap_queue_expert(ggml_cgraph * gf, ggml_tensor * expert_sum) {
    if (gf && expert_sum) ggml_build_forward_expand(gf, expert_sum);
}

// The join: the expert GPU's routed sum plus the target's sum (both F32, on the target).
inline ggml_tensor * moe_overlap_join(ggml_context * ctx, ggml_tensor * expert_sum, ggml_tensor * target_sum) {
    return ggml_add(ctx, expert_sum, target_sum);
}

// 2: the scheduler ends the target's split right before `join` (its input from the expert GPU is a new one there), and
// only there: other nodes split as before. Call for every join of the graph after the scheduler is reset and before
// it allocates the graph.
inline void moe_overlap_mark_join(ggml_backend_sched_t sched, ggml_tensor * join) {
    ggml_backend_sched_add_late_cross_input_split_node(sched, join);
}

// 3: one routed-expert projection of an overlap step for all of its rows, ggml_mul_mat_id(ctx, experts, rows, ids),
// whose rows share each routed expert's weight reads (ggml_backend_cuda_mul_mat_id_grouped).
inline ggml_tensor * moe_overlap_expert_product(ggml_context * ctx, ggml_tensor * experts, ggml_tensor * rows,
                                                ggml_tensor * ids) {
    ggml_tensor * product = ggml_mul_mat_id(ctx, experts, rows, ids);
    ggml_backend_cuda_mul_mat_id_grouped(product);
    return product;
}

}  // namespace luce::common
