// Shared inputs/outputs for the DFlash draft graph builder.
#pragma once

#include "ggml.h"

namespace dflash::common {

struct DraftWeights; // fwd

struct DraftGraphInputs {
    int           ctx_len;          // length of target_hidden_cat along ne[1]
    ggml_tensor * noise_embed;      // [hidden, q_len=16, 1] f32
    ggml_tensor * target_hidden_cat;// [5*hidden, ctx_len, 1] f32
    ggml_tensor * positions_q;      // [q_len] i32   values [ctx_len..ctx_len+q_len-1]
    ggml_tensor * positions_k;      // [ctx_len+q_len] i32   values [0..ctx_len+q_len-1]
    // Optional: if non-null, the graph projects final hidden states through
    // this LM head (shape [hidden, vocab]) and returns logits instead of
    // hidden states. Used for DFlash integration where the draft shares the
    // target's lm_head.
    ggml_tensor * lm_head;
    // Optional: causal mask for SWA layers [kv_pad, q_len] F16.
    // nullptr = all layers non-causal.
    ggml_tensor * causal_mask_swa = nullptr;
};

struct DraftGraphOutputs {
    ggml_tensor * hidden_states;    // [hidden, q_len, 1]  (always set)
    ggml_tensor * logits;           // [vocab, q_len, 1]   (non-null iff lm_head was provided)
};

DraftGraphOutputs build_draft_graph(
    ggml_context *            ctx,
    const DraftWeights &      w,
    const DraftGraphInputs &  in);

} // namespace dflash::common
