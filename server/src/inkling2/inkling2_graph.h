// Inkling2-Loop forward, verify-rollback, and bundled DSpark draft graph builders.

#pragma once

#include "inkling2_cache.h"
#include "inkling2_internal.h"

#include <cstdint>
#include <vector>

namespace luce::common {

struct Inkling2ForwardOut {
    bool ok = false;
    std::vector<int32_t> greedy; // size = T (or 1 for prefill)
    // Populated only when caller requests full logits (non-greedy fallback or oracle check)
    std::vector<float> logits;
    // Populated when fast_sample_temp > 0: compact [516 * T] I32 GPU-extracted top candidates
    // (count, max_logit, exact full-vocab Z, up to 256 (logit, id) pairs/row -> 400x less PCIe!).
    std::vector<int32_t> top_cands;
    ggml_tensor *        logits_dev = nullptr; // device logits pointer for rare single-row fallback
};

// Run n_tokens starting at pos0 through Inkling 2 Mini (48 effective layers).
// - verify == false, n_tokens == 1: single-token decode (HIP-graph cached, R9700-only fast path)
// - verify == true,  2 <= n_tokens <= 8: DSpark verify (writes conv rollback ring, captures 4 DSpark taps,
//   runs Dual-GPU R9700+Strix parallel experts + LM-head split!)
// - verify == false, n_tokens > 1: chunked prefill (inline rel-bias tiled FlashAttention)
// If capture_taps_for_dspark is true, stores the 4 target hidden taps into the DSpark 3-layer KV cache
// and saves the last token's taps in cache.last_taps_dev.
// When fast_sample_temp > 0, extracts compact exact GPU top-candidates (14 KB) instead of copying 5.6 MB logits!
Inkling2ForwardOut inkling2_forward(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const int32_t * tokens,
    int n_tokens,
    int pos0,
    bool verify,
    bool need_logits,
    bool capture_taps_for_dspark,
    bool allow_split = true,
    float fast_sample_temp = 0.0f);

// 3-Pass Wavefront Loop-Fused Prefill (bit-exact):
// Pipelines consecutive prompt chunks across Pass 2 / Pass 1 / Pass 0 simultaneously so every
// physical core layer (blk.6..17) reads its QKV/O + 128 MoE experts ONCE instead of 3x!
Inkling2ForwardOut inkling2_prefill_wavefront(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    const int32_t * tokens,
    int n_tokens,
    int pos0,
    int chunk_size,
    bool need_logits,
    bool capture_taps_for_dspark);

// Exact O(1)-launch GPU rollback after speculative verify:
// Restores all 48 conv states to post-token `retained - 1` (1 <= retained <= verify_n)
// and updates cache.cur_pos = pos0 + retained.
bool inkling2_verify_rollback(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    int pos0,
    int retained);

// Commit accepted verify taps [0 .. retained-1] into DSpark 3-layer draft KV cache
// and copy tap[retained - 1] into cache.last_taps_dev for the next draft block.
bool inkling2_dspark_commit_taps(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    ggml_tensor * verify_taps_dev, // F32 [n_embd, 4, verify_n]
    int pos0,
    int retained);

// Draft 7 speculative tokens in ONE fused HIP graph (3-layer bidirectional block-8
// transformer + 7-step RNN Markov head with pre-folded lm_head + softcap + GPU argmax).
bool inkling2_dspark_draft(
    ggml_backend_t target_backend,
    const Inkling2Weights & w,
    Inkling2Cache & cache,
    int32_t anchor_token,
    int anchor_pos,
    int n_draft, // 1..7 (default 7)
    std::vector<int32_t> & out_draft_tokens,
    int max_draft = 0);

} // namespace luce::common
