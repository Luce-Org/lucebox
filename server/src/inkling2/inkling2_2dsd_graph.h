// Mixed-depth static-bucketed 2DSD target runner (Prelude / Fused Multi-Pass Core /
// Fused Multi-Depth Coda / Commit) and 60-plane multi-depth Prefill for Inkling 2 Mini.

#pragma once

#include "inkling2_cache.h"
#include "inkling2_internal.h"

#include <cstdint>
#include <vector>

namespace luce::common {

struct Inkling2TwoDSDNodeDesc {
    int32_t          token = 0;
    int              slot  = 0;        // tree scratch slot in [0..max_nodes-1]
    int              pos   = 0;        // absolute token position
    int              depth = 0;        // completed loops BEFORE this call (0, 1, or 2)
    std::vector<int> ancestors;        // uncommitted ancestor tree slots (oldest first, excluding self)
};

// Prefill `n_tokens` starting at `pos0` into all 60 planes of `cache.tree` (prelude, 3 core passes,
// and all 3 coda depths {1, 2, 3}), commit loop-3 5-tap features into the 5-layer drafter KV cache,
// and optionally return the greedy argmax token at the final prompt position (from depth-3 coda).
bool inkling2_2dsd_prefill_chunk(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const int32_t * tokens,
    int n_tokens,
    int pos0,
    bool is_last_chunk,
    int32_t & out_greedy_tok);

// Advance `nodes` by +1 loop in up to 3 static-bucketed HIP/CUDA graphs:
//   - Prelude (L0..5) for nodes with depth == 0
//   - Fused Core (L6..17) for ALL nodes across passes 0/1/2 simultaneously (1 weight read!)
//   - Fused Coda (L18..23) for nodes whose new depth (depth+1) is in `readout_mask` (bit d = 1<<d)
// Writes greedy argmax tokens into `out_readout_rows` (indices into `nodes`) and `out_readout_argmax`.
bool inkling2_2dsd_advance_nodes(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
    uint32_t readout_mask,
    bool allow_split,
    std::vector<int> & out_readout_rows,
    std::vector<int32_t> & out_readout_argmax);

// Commit accepted path slots [0..n_commit-1] from tree scratch into all 60 committed prefix planes
// AND commit their loop-3 features into the 5-layer drafter prefix KV cache.
bool inkling2_2dsd_commit_nodes(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const int * commit_slots,
    int n_commit);

} // namespace luce::common
