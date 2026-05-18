// qwen36_mtp_graph.h — CUDA cgraph for Qwen3.6 MTP head step forward.

#pragma once

#include "qwen36_mtp.h"

struct ggml_tensor;
struct ggml_context;
struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr * ggml_gallocr_t;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace dflash27b {
namespace mtp {

// Step graph state for a single MTP head's per-call forward.  Inputs are
// set via ggml_backend_tensor_set; the output (out_x_normed) is read back
// to host (or, when the fused LM-head path is wired, out_argmax_token /
// out_topk_* are read instead and out_x_normed is unused).
//
// Bug #5 fix: graphs are shape-only (no slot/kv_len baked in).  The KV
// slot to write and the FA read indices + mask are runtime inputs, so
// one graph per (head, fa_window, fused_lm_head, topk_k) suffices for
// the entire generation instead of one per draft_pos.
struct Qwen36MtpStepGraph {
    ggml_context * ctx           = nullptr;
    ggml_cgraph  * gf            = nullptr;
    ggml_gallocr_t alloc         = nullptr;

    // Build-time keys (used to detect invalidation).
    int  fa_window  = 0;
    int  fa_max     = 0;   // baked FA window width (rows in inp_kv_idxs_read/inp_kv_mask)
    int  topk_k     = 0;
    bool fused_lm_head = false;

    // Inputs.
    ggml_tensor * inp_embed       = nullptr;   // [n_embd]      f32 — pre-embedded cur_token
    ggml_tensor * inp_h_prev      = nullptr;   // [n_embd]      f32 — backbone hidden h_{base_pos-1}
    ggml_tensor * inp_pos         = nullptr;   // [4]           i32 — MROPE positions (p,p,p,0)
    ggml_tensor * inp_kv_idx_write= nullptr;   // [1]           i64 — slot to write Kcur/Vcur
    ggml_tensor * inp_kv_idxs_read= nullptr;   // [fa_max,n_head_kv] i32 — FA read slots
    ggml_tensor * inp_kv_mask     = nullptr;   // [fa_max,1]    f16 — -INF on inactive rows

    // Outputs.
    ggml_tensor * out_x_normed     = nullptr;  // [n_embd]   f32 — post shared_head_norm hidden
    // Pre-shared_head_norm hidden (post FFN residual `add`).  Mirrors
    // llama.cpp PR #22673's `t_h_pre_norm`: this is the tensor that must
    // be fed back as h_prev for the NEXT autoregressive step.  Re-using
    // `out_x_normed` here causes the next iter's `hnorm` to double-
    // normalise, producing a distribution drift that compounds per depth
    // (D=3 accept collapsed from ~91% to ~67% per-position in our bench;
    // see qwen36_mtp.cpp:1166 for the byte-correct CPU stash of pre-norm).
    ggml_tensor * out_h_pre_norm   = nullptr;  // [n_embd]   f32 — pre shared_head_norm hidden
    // Fused LM-head outputs (only populated when build is called with a
    // non-null lm_head_weight).  out_argmax_token holds the i32 argmax of
    // the logits; out_logits is exposed so the host can compute log-softmax
    // for top-K emission without re-running the LM head matmul.
    ggml_tensor * out_argmax_token = nullptr;  // [1]        i32
    ggml_tensor * out_logits       = nullptr;  // [n_vocab]  f32 — full logits (optional)
};

void qwen36_mtp_step_graph_free(Qwen36MtpStepGraph & sg);

// Batched warmup graph: writes K/V to head_kv slots [slot_start, slot_start+n_tokens)
// in a single backend pass.  Replaces the host-side per-position CPU loop with
// one ggml cgraph using the same quant-aware matmul kernels the step graph uses.
struct Qwen36MtpWarmGraph {
    ggml_context * ctx           = nullptr;
    ggml_cgraph  * gf            = nullptr;
    ggml_gallocr_t alloc         = nullptr;

    ggml_tensor * inp_embed_seq  = nullptr;   // [n_embd, n_tokens] f32 — pre-embedded tokens
    ggml_tensor * inp_h_seq      = nullptr;   // [n_embd, n_tokens] f32 — backbone hiddens
    ggml_tensor * inp_pos        = nullptr;   // [4 * n_tokens]     i32 — MROPE positions
};

void qwen36_mtp_warm_graph_free(Qwen36MtpWarmGraph & sg);

// Build the warmup graph for n_tokens prefill positions writing to slots
// [slot_start, slot_start + n_tokens) of head_k_cache / head_v_cache.
bool build_qwen36_mtp_warm_graph(
        Qwen36MtpWarmGraph & sg,
        const Qwen36MtpHeadWeights & head,
        ggml_tensor * head_k_cache,
        ggml_tensor * head_v_cache,
        ggml_backend_t backend,
        int n_embd,
        int n_head_kv,
        int key_len,
        int val_len,
        int n_rot,
        int rope_sections[4],
        float rope_freq_base,
        float rms_eps,
        int slot_start,
        int n_tokens);

// Build the head's step graph as a SHAPE-ONLY template.  Per-call slot
// (write) and FA read indices/mask are wired as runtime inputs so a single
// graph services every draft_pos.
//   - head_k_cache / head_v_cache: per-head KV cache tensors on the backbone
//     backend; layout [head_dim, n_ctx, n_head_kv].  F16 on device.
//   - n_ctx: cache row count (used to reshape the cache for set_rows/get_rows).
//   - fa_window: if > 0, FA attends fa_max=min(fa_window,n_ctx) rows. If <=0,
//     fa_max=n_ctx (full context).  Rows beyond live kv_len are masked at
//     runtime via inp_kv_mask.
//   - lm_head_weight / lm_head_topk: see prior comment block; behaviour unchanged.
// Returns false on allocation failure.
bool build_qwen36_mtp_step_graph(
        Qwen36MtpStepGraph & sg,
        const Qwen36MtpHeadWeights & head,
        ggml_tensor * head_k_cache,
        ggml_tensor * head_v_cache,
        ggml_backend_t backend,
        int n_embd,
        int n_head,
        int n_head_kv,
        int key_len,
        int val_len,
        int ffn_len,
        int n_rot,
        int rope_sections[4],
        float rope_freq_base,
        float rms_eps,
        int n_ctx,
        int fa_window           = 0,
        ggml_tensor * lm_head_weight = nullptr,
        int lm_head_topk        = 0);

}  // namespace mtp
}  // namespace dflash27b
