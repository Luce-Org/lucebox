// StepGraph — per-forward-call compute graph container.
//
// Holds the ggml context, graph, allocator, and named tensor handles for one
// forward step (prefill chunk, verify batch, or replay). Rebuilt per call
// since kv_len varies, but the persistent CUDA allocator buffer is kept
// alive across steps to avoid cudaMalloc/cudaFree churn.

#pragma once

#include "internal.h"  // DeltaNetCapture

#include "ggml.h"
#include "ggml-alloc.h"

#include <vector>

namespace dflash::common {

struct StepGraph {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;

    // The ctx_len last used for ggml_gallocr_reserve (draft only).
    // When the real ctx_len fits within this, alloc_graph is a no-op.
    int alloc_reserved_ctx = 0;

    // Named inputs
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;     // may be null
    ggml_tensor *   parent_ids = nullptr;    // DDTree tree-mode; null for chain mode
    ggml_tensor *   target_hidden_cat = nullptr;  // draft only
    ggml_tensor *   positions_k = nullptr;        // draft only
    ggml_tensor *   hidden_input = nullptr;        // lm-head projection only

    // Output
    ggml_tensor *   logits = nullptr;
    ggml_tensor *   hidden_states = nullptr;       // draft hidden-only output
    ggml_tensor *   argmax_tokens = nullptr; // [n_tokens] i32, GPU-side argmax of logits
    ggml_tensor *   topk_indices = nullptr;  // [K, n_tokens] i32, GPU-side top-K indices
    ggml_tensor *   ffn_residual = nullptr;  // [hidden, n_tokens] pre-FFN residual
    ggml_tensor *   ffn_post = nullptr;      // [hidden, n_tokens] post-attention norm
    ggml_tensor *   moe_weights = nullptr;   // [n_used, n_tokens] f32

    // Per-delta-net-layer captures (verify only).
    std::vector<DeltaNetCapture> delta_captures;
    std::vector<ggml_tensor *> moe_selected;
};

// Reset the per-call graph state (ctx + graph + tensor handles) but KEEP the
// persistent CUDA buffer in `sg.alloc` alive across steps.
inline void step_graph_free(StepGraph & sg) {
    if (sg.ctx)   { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.gf = nullptr;
    sg.inp_embed = sg.positions = sg.attn_mask = nullptr;
    sg.target_hidden_cat = sg.positions_k = nullptr;
    sg.hidden_input = nullptr;
    sg.parent_ids = nullptr;
    sg.logits = nullptr;
    sg.hidden_states = nullptr;
    sg.argmax_tokens = nullptr;
    sg.topk_indices = nullptr;
    sg.ffn_residual = nullptr;
    sg.ffn_post = nullptr;
    sg.moe_weights = nullptr;
    sg.delta_captures.clear();
    sg.moe_selected.clear();
}

// Full cleanup: release the persistent gallocr + its CUDA buffer.
inline void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    step_graph_free(sg);
}

}  // namespace dflash::common
