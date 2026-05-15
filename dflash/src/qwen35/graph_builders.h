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

namespace dflash27b {

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
    int kq_stride_pad = KQ_MASK_PAD);

// Full target forward: chain mode (all layers, logits + argmax output).
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
    int kq_stride_pad = KQ_MASK_PAD);

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

}  // namespace dflash27b
