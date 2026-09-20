// Qwen4Exp forward graph.
//
// One entry point: qwen4exp_forward() runs n_tokens new tokens starting at
// cache position pos0 through the 48-layer hybrid trunk and writes the
// last-token logits. Ported from upstream llama.cpp src/models/qwen4exp.cpp
// (hyper-connections, gated delta net linear attention, dense full attention,
// 512-expert top-10 MoE, per-layer n-gram embedding) into Luzebox's ggml graph
// style.
//
// Single sequence (n_seqs = 1). The learned sparse indexer is built but the
// full-attention path runs dense, which is the numerically exact computation
// (see the strix-halo journey: sparse selection measured break-even at equal
// quality). The PLE table is read through Qwen4ExpPleReader, never uploaded.

#pragma once

#include "qwen4exp_internal.h"
#include "qwen4exp_cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

struct Qwen4ExpForwardResult {
    bool ok = false;
    int  n_tokens = 0;
    int  pos0 = 0;
};

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
                                       std::vector<float> & out_logits);

}  // namespace dflash::common
