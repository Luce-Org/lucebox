// MoE split steps: one step of a model whose MoE layers are split between two GPUs of one runtime, the target
// (attention, dense work, the shared expert and the routed experts it holds) and the expert device (the routed
// experts the target does not hold). Used by DeepSeek V4.1 (deepseek4); written for any such split, so the
// Qwen3.8-Flash-Next split can move onto it.
//
// ggml's scheduler copies a split's cross-device inputs before any node of the split runs, so the two devices only
// work at the same time when
//   1. the expert device's nodes come right after the routes in graph order: its split is queued as soon as the
//      routes are known, ahead of the target's own expert and dense work (moe_split_queue_expert), and
//   2. the target's split ends right before the join, the first node that reads the expert device's sum, so the
//      target's work in between does not wait for that copy (moe_split_sched_init, moe_split_join).
// All of it is scheduling: every node computes the same bytes as in one serial graph.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

namespace luce::common {

// 1: queues the expert device's routed sum, and the nodes it needs, in gf. Call once the routes are built and before
// the target's experts are queued.
void moe_split_queue_expert(ggml_cgraph * gf, ggml_tensor * expert_sum);

// The join: the expert device's routed sum (an F16 sum is cast to F32 first) plus the target's sum.
ggml_tensor * moe_split_join(ggml_context * ctx, ggml_tensor * expert_sum, ggml_tensor * target_sum);

// 2 for every split step `sched` runs: a split ends before a node that brings in a new cross-device input
// (ggml_backend_sched_set_late_cross_input_split).
void moe_split_sched_init(ggml_backend_sched_t sched);

}  // namespace luce::common
