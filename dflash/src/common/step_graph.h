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
    // Post-norm hidden for last token [n_embd] f32.  Used by MTP module to
    // seed h_prev_0.  Populated by build_target_step; null otherwise.
    ggml_tensor *   last_norm_hidden = nullptr;
    // Full post-norm hidden sequence [n_embd, n_tokens] f32.  Used by
    // warm_head_kv() to read per-position hiddens during prefill.
    ggml_tensor *   all_norm_hidden = nullptr;
    // Pre-final-output-norm hidden — last token [n_embd] f32 and full
    // sequence [n_embd, n_tokens] f32.  Used by the Qwen3.6 MTP module to
    // seed the NextN head's h_prev WITHOUT double-normalising against the
    // head's own hnorm (mirror of llama.cpp PR #22673 `t_h_pre_norm`).
    // Populated alongside last_/all_norm_hidden when the caller asks for
    // capture_all_norm_hidden.
    ggml_tensor *   last_h_pre_norm = nullptr;
    ggml_tensor *   all_h_pre_norm  = nullptr;

    // Per-delta-net-layer captures (verify only).
    std::vector<DeltaNetCapture> delta_captures;
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
    sg.last_norm_hidden = nullptr;
    sg.all_norm_hidden = nullptr;
    sg.last_h_pre_norm = nullptr;
    sg.all_h_pre_norm  = nullptr;
    sg.delta_captures.clear();
}

// Full cleanup: release the persistent gallocr + its CUDA buffer.
inline void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    step_graph_free(sg);
}

}  // namespace dflash::common
