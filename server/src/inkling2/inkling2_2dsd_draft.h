// 2DSD Loop-Context 5-Layer Qwen3-DFlash + Vanilla DSpark Markov + Confidence Head Drafter.
//
// Committed prefix tokens use loop-R (depth 3) features cached incrementally in `tree.draft_k/v[0..4]`.
// Uncommitted in-flight ancestor spans use loop-r features gathered on device from `tree.node_features`
// and projected per draft batch. Multi-leaf batches replay from static bucketed HIP/CUDA graphs.

#pragma once

#include "inkling2_cache.h"
#include "inkling2_internal.h"

#include <cstdint>
#include <vector>

namespace luce::common {

struct Inkling2TwoDSDDraftReq {
    int32_t          head_token = 0;
    int              anchor_pos = 0; // parent_pos + 1 = prefix_len + span_slots.size()
    int              depth      = 3; // 1, 2, or 3 (drafting parent's readout depth)
    std::vector<int> span_slots;     // in-flight ancestor tree slots [cursor .. parent], empty when depth == 3
};

struct Inkling2TwoDSDDraftOut {
    std::vector<int32_t> tokens;     // size = block_size - 1 (7)
    std::vector<int32_t> alt_tokens; // optional 2nd-best Markov token per slot (for confidence-guided branching)
    std::vector<float>   confidence; // size = block_size - 1 (7), probabilities in [0, 1]
};

// Append fused in-graph projection of loop-R (depth 3) features [5*H, T] into the 5-layer
// drafter prefix KV cache (`tc.draft_k/v`) using `rope_pos` (true absolute RoPE positions)
// and `write_pos` (compact KV write slots = rope_pos - tc.draft_pref_start).
// Eliminates separate gallocr alloc, device-to-device copy, and extra sync during prefill!
void inkling2_2dsd_append_prefix_kv_graph(
    ggml_context * c,
    ggml_cgraph * gf,
    const Inkling2TwoDSDWeights & td,
    const Inkling2TreeCache & tc,
    ggml_tensor * feats_dev,
    ggml_tensor * rope_pos,
    ggml_tensor * write_pos);

// Commit loop-R (depth 3) prompt features [5*H, n_tokens] (F32 or F16) into the 5-layer
// drafter prefix KV cache at positions [pos0 .. pos0 + n_tokens - 1].
bool inkling2_2dsd_commit_prefix_kv_from_feats(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    ggml_tensor * feats_dev,
    int pos0,
    int n_tokens);

// Commit accepted tree slots [0 .. n_commit - 1] (loop-R features gathered directly on GPU
// from `cache.tree.node_features[2, slot]`) into the 5-layer drafter prefix KV cache.
bool inkling2_2dsd_commit_prefix_kv_from_slots(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const int * commit_slots,
    int n_commit,
    int pos0);

// Draft `requests.size()` blocks in ONE batched multi-leaf HIP/CUDA graph call.
// Returns `block_size - 1` (7) drafted tokens and confidence scores per request.
bool inkling2_2dsd_draft_batch(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const std::vector<Inkling2TwoDSDDraftReq> & requests,
    std::vector<Inkling2TwoDSDDraftOut> & out);

} // namespace luce::common
