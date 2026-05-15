// layer_split_forward.h — Multi-GPU layer-split forward pass for qwen35.
//
// The target model is split across GPUs with each shard owning a contiguous
// range of layers. The forward pass runs sequentially through shards,
// transferring activations between GPUs at shard boundaries.

#pragma once

#include "layer_split_types.h"
#include "dflash_draft_ipc.h"
#include "dflash_feature_ring.h"
#include "step_graph.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash27b {

// Compute argmax(logits) for a slice of the activation tensor via
// out_norm + lm_head projection.
bool compute_target_split_argmax(
        StepGraph & sg,
        const TargetWeights & w,
        ggml_backend_t backend,
        ggml_tensor * act,
        int token_offset,
        int n_tokens,
        int hidden,
        int vocab,
        std::vector<int32_t> & argmax_out);

// Run a full forward pass through all shards, writing K/V into each shard's
// cache.  Returns the argmax of the last token in `last_tok`.
// Optionally captures features into `feature_ring` / remote draft.
bool run_target_layer_split_forward(
        std::vector<TargetLayerSplitShard> & shards,
        const TargetWeights & embed_source,
        const std::vector<int32_t> & tokens,
        int base_pos,
        int ubatch,
        int & last_tok,
        int kq_stride_pad,
        int fa_window,
        DraftFeatureMirror * feature_ring = nullptr,
        std::vector<int32_t> * argmax_out = nullptr,
        std::vector<float> * logits_out = nullptr,
        DFlashDraftIpcClient * remote_draft = nullptr);

// Free all shards (weights, cache, backend).
void free_target_layer_split_shards(std::vector<TargetLayerSplitShard> & shards);

} // namespace dflash27b
