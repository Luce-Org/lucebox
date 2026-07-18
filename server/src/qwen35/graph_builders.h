// Graph-building functions for the qwen35 target forward passes.
//
// These create ggml compute graphs for one step (prefill chunk, chain-mode
// verify, tree-mode verify, or LM-head projection). Each function
// allocates tensor descriptors, wires the graph via build_qwen35_graph,
// and reserves the gallocr buffer.
//
// The generic DFlash draft-step graph builder lives in
// common/dflash_draft_graph.h.
//
// The `kq_stride_pad` parameter replaces the old file-scope g_kq_stride_pad
// global — callers pass it explicitly (default KQ_MASK_PAD, or 256 when TBQ
// KV is active).

#pragma once

#include "step_graph.h"
#include "attn_masks.h"       // align_up, KQ_MASK_PAD
#include "internal.h"         // TargetWeights, TargetCache

#include "ggml.h"
#include "ggml-backend.h"

namespace dflash::common {

// Layer-segmented prefill: process one target layer for chunk_start..chunk_start+n_tokens.
bool build_layer_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    ggml_tensor * act_in,
    ggml_tensor * act_out,
    int chunk_start,
    int n_tokens,
    int kv_start,
    bool with_mask,
    bool capture,
    int fa_window = 0,
    int kq_stride_pad = KQ_MASK_PAD,
    bool kvflash = false,
    bool tree_mode = false);

// `kvflash`: pooled mode — KV rows go through a set_rows input
// (sg.kv_write_rows, [n_tokens, n_head_kv] ne0-major slots) and the mask
// (forced on) is sized to the PHYSICAL tensor capacity so the caller can
// fill it in slot space. Caller allocates slots and fills rows + mask.
bool build_layer_prefn_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    int kv_start,
    int n_tokens,
    bool with_mask,
    int fa_window = 0,
    int kq_stride_pad = KQ_MASK_PAD,
    bool kvflash = false);

// Full layer graph for hybrid decode: pre-FFN + MoE FFN + shared + residual in one compute.
// Output: sg.hidden_input = layer_output, sg.moe_selected = router selections.
bool build_hybrid_full_layer_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    int kv_start,
    int n_tokens,
    bool with_mask,
    int fa_window = 0,
    int kq_stride_pad = KQ_MASK_PAD);

// Full target forward: chain mode (all layers, logits + argmax output).
//
// `kvflash_mask`: kvflash pooled mode — keep the set_rows KV write active
// even though a mask is requested (the mask carries pool-slot validity and
// must be re-uploaded by the caller before every compute). Used by both
// single-token decode and multi-token spec verify; requires fa_window == 0.
bool build_target_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    bool with_mask,
    bool capture,
    bool capture_delta_intermediate = false,
    int fa_window = 0,
    bool last_token_logits_only = false,
    int kq_stride_pad = KQ_MASK_PAD,
    bool capture_moe_router = false,
    bool kvflash_mask = false,
    bool capture_qk = false);

// Full target forward: DDTree tree-verify mode.
bool build_target_step_tree(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    int fa_window = 0,
    int kq_stride_pad = KQ_MASK_PAD);

// LM-head projection: project draft hidden states through the target output matrix.
bool build_lm_head_projection_step(
    StepGraph & sg,
    const TargetWeights & w,
    ggml_backend_t backend,
    int n_tokens);

}  // namespace dflash::common
