// Qwen4Exp forward graph.
//
// qwen4exp_forward() runs n_tokens new tokens starting at cache position pos0
// through the 48-layer hybrid trunk and writes last-token logits.
// Ported from upstream llama.cpp src/models/qwen4exp.cpp
// (hyper-connections, gated delta net linear attention, dense full attention,
// 512-expert top-10 MoE, per-layer n-gram embedding) into Luzebox's ggml graph
// style.
//
// Single sequence (n_seqs = 1). On gfx1151, multi-row prompt prefill uses QSA
// with F32 accumulation, including below the selection budget. T=1 retains
// dense attention below the budget and selected attention beyond it. This
// last-logit-only trunk has no MTP verification entry point: an MTP port must
// route verify rows through the T=1 attention path, not the prefill promotion.
// The PLE table is read through Qwen4ExpPleReader, never uploaded in full.

#pragma once

#include "qwen4exp_internal.h"
#include "qwen4exp_cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace luce::common {

// QSA indexer block pooling: [idim, nb*r] keys -> [idim, nb], block b = mean of tokens r*b .. r*b+r-1.
ggml_tensor * qwen4exp_pool_blocks(ggml_context * c, ggml_tensor * keys, int64_t r);

struct Qwen4ExpForwardResult {
    bool ok = false;
};

// Host-only input preparation; safe to run for the next prompt chunk while
// the current graph computes. The caller owns the immutable token span.
struct Qwen4ExpInputs {
    bool ok = false;
    std::vector<float> emb, ple;
    std::vector<int32_t> ple_prev;
};
Qwen4ExpInputs qwen4exp_prepare_inputs(const Qwen4ExpWeights & w,
    const int32_t * tokens, int n_tokens, const std::vector<int32_t> & ple_prev);

struct Qwen4ExpGraphMemory {
    size_t graph = 0, inputs = 0, mask = 0, host = 0, scratch = 0;
};
// Allocation plan only: no GPU allocation, input reads, compute or state update.
Qwen4ExpGraphMemory qwen4exp_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache, int n_tokens, int pos0);

// Run the trunk. `tokens` has n_tokens entries, processed as one contiguous
// single-sequence span at positions [pos0, pos0 + n_tokens). On success the
// cache is advanced to pos0 + n_tokens and out_logits holds n_vocab floats
// for the final token.
Qwen4ExpForwardResult qwen4exp_forward(ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache,
                                       const int32_t * tokens,
                                       int n_tokens,
                                       int pos0,
                                       std::vector<float> & out_logits,
                                       const Qwen4ExpInputs * inputs = nullptr);

}  // namespace luce::common
