// StepGraph — per-forward-call compute graph container.
//
// Holds the ggml context, graph, allocator, and named tensor handles for a
// forward topology (prefill chunk, verify batch, or replay). Most paths
// rebuild as shapes vary; topology-stable paths can retain the graph. The
// persistent CUDA allocator buffer stays alive across rebuilds.

#pragma once

#include "internal.h"  // DeltaNetCapture

#include "ggml.h"
#include "ggml-alloc.h"

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace dflash::common {

using TargetPagedTreeGraphKey = std::tuple<
    const TargetWeights *, const TargetCache *, ggml_backend_t,
    int, int, int, int, int, int, int, std::vector<int>>;

struct StepGraph {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;
    ggml_context * commit_ctx = nullptr;
    ggml_backend_buffer_t commit_buffer = nullptr;
    std::optional<TargetPagedTreeGraphKey> paged_tree_key;

    // Persistent metadata arena for the draft graph. Reusing the same arena
    // keeps ggml_tensor addresses stable for CUDA-graph replay.
    std::vector<uint8_t> meta_arena;

    // Paged-tree metadata is retained with the graph. Use uninitialized
    // storage: vector::resize would zero 512 MiB on the first decode round.
    std::shared_ptr<uint8_t> paged_tree_meta_arena;

    // The ctx_len last used for ggml_gallocr_reserve (draft only).
    // When the real ctx_len fits within this, alloc_graph is a no-op.
    int alloc_reserved_ctx = 0;

    // Named inputs
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;     // may be null
    ggml_tensor *   parent_ids = nullptr;    // DDTree tree-mode; null for chain mode
    ggml_tensor *   tree_sizes = nullptr;    // DDTree [n_tree_seqs], 0 = padding
    // SpecLA topology masks ([n_tokens, n_tokens] f32, host-filled; see
    // delta_net_specla.h). Created only when DFLASH_SPECLA capture is active.
    ggml_tensor *   specla_m_strict = nullptr;
    ggml_tensor *   specla_m_incl   = nullptr;
    ggml_tensor *   specla_m_eye    = nullptr;
    ggml_tensor *   specla_hld      = nullptr;
    ggml_tensor *   target_hidden_cat = nullptr;  // draft only
    ggml_tensor *   positions_k = nullptr;        // draft only
    ggml_tensor *   pad_mask_full = nullptr;      // draft only; padded-ctx mask
    // When >0, the draft graph was built with the ctx dimension padded to this
    // size (stable topology for CUDA-graph replay); noise keys start here.
    int             ctx_alloc = 0;
    // True when the built topology reads features through a mirror ring VIEW
    // (no target_hidden_cat input tensor). A view-built graph must never be
    // reused by the copy-mode persistent fast path.
    bool            built_view = false;
    ggml_tensor *   hidden_input = nullptr;        // lm-head projection only
    // [n_tokens,n_head_kv] i64 physical destination rows for ggml_set_rows.
    // Used by contiguous replay, KVFlash, and paged attention; null when the
    // graph uses the legacy contiguous ggml_cpy write.
    ggml_tensor *   kv_write_rows = nullptr;
    // Compact decode row -> physical sequence slot. Padding rows carry -1.
    // state_slot_ids has the same shape but maps padding to a safe readable
    // slot for graph-level conv-state gathers.
    ggml_tensor *   active_slot_ids = nullptr;
    // Recurrent gather rows. Unlike active/paged IDs, padding must name a
    // valid harmless slot (normally 0): ggml_get_rows does not mask -1.
    ggml_tensor *   state_slot_ids = nullptr;
    // Ragged paged read (concurrent prefill): per-row block-table column and
    // inclusive causal position, [n_tokens] i32 each. Padding rows carry -1.
    ggml_tensor *   paged_query_seq_ids = nullptr;
    ggml_tensor *   paged_query_positions = nullptr;
    // DFlash target-feature destination rows. In packed-tree graphs this maps
    // only the durable prefill/AR prefix; tree features remain scratch.
    ggml_tensor *   target_feat_rows = nullptr;
    // Packed-tree direct-commit metadata uploaded after posterior selection.
    ggml_tensor *   accepted_prefixes = nullptr;   // [n_tree_seqs] i32
    ggml_tensor *   commit_slot_ids = nullptr;      // [n_tree_seqs] i32
    ggml_tensor *   commit_rows = nullptr;         // [tree_width,n_tree_seqs] i64
    ggml_tensor *   feature_commit_rows = nullptr; // [tree_width*n_tree_seqs] i32
    // Multi-prompt steps: i32 row indices gathered from the final norm
    // before the LM head (committing rows + decode rows).
    ggml_tensor *   logits_row_indices = nullptr;

    // Output
    ggml_tensor *   logits = nullptr;
    ggml_tensor *   hidden_states = nullptr;       // draft hidden-only output
    ggml_tensor *   argmax_tokens = nullptr; // [n_tokens] i32, GPU-side argmax of logits
    ggml_tensor *   topk_indices = nullptr;  // [K, n_tokens] i32, GPU-side top-K indices
    ggml_tensor *   ffn_residual = nullptr;  // [hidden, n_tokens] pre-FFN residual
    ggml_tensor *   ffn_post = nullptr;      // [hidden, n_tokens] post-attention norm
    ggml_tensor *   moe_weights = nullptr;   // [n_used, n_tokens] f32
    ggml_tensor *   hot_local_lut = nullptr; // [1,n_expert] i32 global->hot-slot (fused FFN)
    ggml_tensor *   valid_lut = nullptr;     // [1,n_expert] f32 1=resident 0=drop (fused FFN)

    // Per-delta-net-layer captures (verify only).
    std::vector<DeltaNetCapture> delta_captures;
    ggml_tensor * tree_features = nullptr;
    std::vector<ggml_tensor *> moe_selected;
};

// Reset the per-call graph state (ctx + graph + tensor handles) but KEEP the
// persistent CUDA buffer in `sg.alloc` alive across steps.
inline void step_graph_free(StepGraph & sg) {
    if (sg.commit_buffer) {
        ggml_backend_buffer_free(sg.commit_buffer);
        sg.commit_buffer = nullptr;
    }
    if (sg.commit_ctx) { ggml_free(sg.commit_ctx); sg.commit_ctx = nullptr; }
    if (sg.ctx)   { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.gf = nullptr;
    sg.inp_embed = sg.positions = sg.attn_mask = nullptr;
    sg.target_hidden_cat = sg.positions_k = nullptr;
    sg.pad_mask_full = nullptr;
    sg.ctx_alloc = 0;
    sg.built_view = false;
    sg.hidden_input = nullptr;
    sg.parent_ids = nullptr;
    sg.tree_sizes = nullptr;
    sg.specla_m_strict = sg.specla_m_incl = sg.specla_m_eye = nullptr;
    sg.specla_hld = nullptr;
    sg.kv_write_rows = nullptr;
    sg.active_slot_ids = nullptr;
    sg.state_slot_ids = nullptr;
    sg.paged_query_seq_ids = nullptr;
    sg.paged_query_positions = nullptr;
    sg.target_feat_rows = nullptr;
    sg.accepted_prefixes = nullptr;
    sg.commit_slot_ids = nullptr;
    sg.commit_rows = nullptr;
    sg.feature_commit_rows = nullptr;
    sg.logits_row_indices = nullptr;
    sg.logits = nullptr;
    sg.hidden_states = nullptr;
    sg.argmax_tokens = nullptr;
    sg.topk_indices = nullptr;
    sg.ffn_residual = nullptr;
    sg.ffn_post = nullptr;
    sg.moe_weights = nullptr;
    sg.hot_local_lut = nullptr;
    sg.valid_lut = nullptr;
    sg.delta_captures.clear();
    sg.tree_features = nullptr;
    sg.moe_selected.clear();
    sg.paged_tree_key.reset();
}

// Full cleanup: release the persistent gallocr + its CUDA buffer.
inline void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    step_graph_free(sg);
    sg.paged_tree_meta_arena.reset();
    sg.meta_arena.clear();
    sg.meta_arena.shrink_to_fit();
    sg.alloc_reserved_ctx = 0;
}

}  // namespace dflash::common
