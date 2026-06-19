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
#include "qwen35_target_shard_ipc.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

class KvFlashPager;

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

bool compute_target_split_projection(
        StepGraph & sg,
        const TargetWeights & w,
        ggml_backend_t backend,
        ggml_tensor * act,
        int token_offset,
        int n_tokens,
        int hidden,
        int vocab,
        std::vector<int32_t> * argmax_out,
        std::vector<float> * logits_out);

// Run a full forward pass through all shards, writing K/V into each shard's
// cache.  Returns the argmax of the last token in `last_tok`.
// Optionally captures features into `feature_ring` / remote draft.
bool run_qwen35_layer_split_forward(
        std::vector<Qwen35LayerSplitShard> & shards,
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
        DFlashDraftIpcClient * remote_draft = nullptr,
        ggml_type activation_type = GGML_TYPE_F32,
        KvFlashPager * kvflash = nullptr);

bool run_qwen35_layer_split_forward_from_activation(
        std::vector<Qwen35LayerSplitShard> & shards,
        ActivationPair & acts,
        int base_pos,
        int n_tokens_total,
        int ubatch,
        int & last_tok,
        int kq_stride_pad,
        int fa_window,
        std::vector<int32_t> * argmax_out = nullptr,
        std::vector<float> * logits_out = nullptr,
        std::vector<Qwen35TargetCaptureSlice> * captures_out = nullptr,
        KvFlashPager * kvflash = nullptr,
        bool kvflash_preallocated = false);

bool run_qwen35_mixed_layer_split_forward(
        std::vector<Qwen35LayerSplitShard> & local_shards,
        Qwen35TargetShardIpcClient & remote_shard,
        const TargetWeights & embed_source,
        const std::vector<int32_t> & tokens,
        int base_pos,
        int ubatch,
        int & last_tok,
        int kq_stride_pad,
        int fa_window,
        std::vector<int32_t> * argmax_out = nullptr,
        std::vector<float> * logits_out = nullptr,
        DraftFeatureMirror * feature_ring = nullptr,
        DFlashDraftIpcClient * remote_draft = nullptr,
        KvFlashPager * kvflash = nullptr);

// Free all shards (weights, cache, backend).
void free_qwen35_layer_split_shards(std::vector<Qwen35LayerSplitShard> & shards);

} // namespace dflash::common
