// Mixed-depth static-bucketed 2DSD target runner (Prelude / Fused Multi-Pass Core /
// Fused Multi-Depth Coda / Commit) and 60-plane multi-depth Prefill for Inkling 2 Mini.

#pragma once

#include "inkling2_cache.h"
#include "inkling2_internal.h"

#include <cstdint>
#include <vector>

namespace luce::common {

struct Inkling2TwoDSDNodeDesc {
    int32_t          token    = 0;
    int              slot     = 0;     // tree scratch slot in [0..max_nodes-1]
    int              pos      = 0;     // absolute token position
    int              depth    = 0;     // completed loops BEFORE this call (0, 1, or 2)
    int              seq_slot = 0;     // sequence slot in [0..n_seq_slots-1] for batched multi-stream
    std::vector<int> ancestors;        // uncommitted ancestor tree slots (oldest first, excluding self)
};

struct Inkling2TwoDSDAdvanceSampleOut {
    float                fast_temp  = 0.0f; // input: >0 runs subop 32 EXACT_GPU_TOP_CANDS on Coda logits
    bool                 need_pmax  = false; // input: true computes p_max[r] = 1/Z(temp=1) for varloop early-exit
    std::vector<int32_t> top_cands;         // output: [516 * n_readout] when fast_temp > 0 or need_pmax
    std::vector<float>   pmax;              // output: [n_readout] max softmax prob (temp=1) when need_pmax
    ggml_tensor *        logits_dev = nullptr; // output: device softcapped F32 [n_vocab, row_b] for fallback/penalties
    int                  n_vocab    = 0;
};

// Prefill `n_tokens` starting at `pos0` into all 60 planes of `cache.tree` (prelude, 3 core passes,
// and all 3 coda depths {1, 2, 3}), commit loop-3 5-tap features into the 5-layer drafter KV cache,
// and optionally return the greedy argmax token (and optional final-position softcapped logits) at
// the final prompt position (from depth-3 coda).
bool inkling2_2dsd_prefill_chunk(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const int32_t * tokens,
    int n_tokens,
    int pos0,
    bool is_last_chunk,
    int32_t & out_greedy_tok,
    std::vector<float> * out_last_logits = nullptr,
    int capture_q_rows = 0);

// 3-Pass Wavefront 60-Plane Prefill for multi-chunk prompts (`n_tokens > chunk_size`):
// Fuses all 3 core passes (L6..17) AND all 3 coda depths (L18..23) across up to 3 chunks
// simultaneously, cutting MoE weight reads from 60 -> 24 per chunk (2.5x) while populating
// all 60 target planes and 5-layer drafter prefix KV bit-for-bit identically.
bool inkling2_2dsd_prefill_wavefront(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const int32_t * tokens,
    int n_tokens,
    int pos0,
    int chunk_size,
    int32_t & out_greedy_tok,
    std::vector<float> * out_last_logits = nullptr,
    int capture_q_rows = 0,
    bool is_last_chunk = true);

// Advance `nodes` by +1 loop in up to 3 static-bucketed HIP/CUDA graphs:
//   - Prelude (L0..5) for nodes with depth == 0
//   - Fused Core (L6..17) for ALL nodes across passes 0/1/2 simultaneously (1 weight read!)
//   - Fused Coda (L18..23) for nodes whose new depth (depth+1) is in `readout_mask` (bit d = 1<<d)
// Writes greedy argmax tokens into `out_readout_rows` (indices into `nodes`) and `out_readout_argmax`.
// When `out_sample != nullptr`, also populates GPU top-candidates (subop 32) and/or device logits pointer
// for exact non-greedy stochastic sampling with zero full-vocab D2H copies on the fast path.
bool inkling2_2dsd_advance_nodes(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
    uint32_t readout_mask,
    bool allow_split,
    std::vector<int> & out_readout_rows,
    std::vector<int32_t> & out_readout_argmax,
    Inkling2TwoDSDAdvanceSampleOut * out_sample = nullptr);

// Multi-stream batched variant: advances nodes from up to `n_seq_slots` independent sequences
// inside ONE shared Prelude/Core/Coda pass (sharing all 128 MoE experts across all active slots!).
// `prefix_lens[s]` gives the committed prefix length for sequence slot `s`.
bool inkling2_2dsd_advance_nodes_batched(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & multi_cache,
    const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
    const int * prefix_lens,
    int n_seq_slots,
    uint32_t readout_mask,
    bool allow_split,
    std::vector<int> & out_readout_rows,
    std::vector<int32_t> & out_readout_argmax,
    Inkling2TwoDSDAdvanceSampleOut * out_sample = nullptr,
    const Inkling2KvFlash * const * slot_kfs = nullptr);

// Commit accepted path slots [0..n_commit-1] from tree scratch into all 60 committed prefix planes
// AND commit their loop-3 features into the 5-layer drafter prefix KV cache.
bool inkling2_2dsd_commit_nodes(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const int * commit_slots,
    int n_commit);

} // namespace luce::common
