// Qwen35LayerSplitDFlashTarget — DFlashTarget adapter for qwen35 multi-GPU
// layer-split inference.
//
// Wraps a vector of TargetLayerSplitShard behind the generic DFlashTarget
// interface so the common spec-decode loop can drive layer-split verify
// without knowing about shard layout, qwen attention parameters, or the
// remote-draft IPC client.
//
// snapshot_kv / restore_kv iterate every shard. verify_batch delegates to
// run_target_layer_split_forward. project_hidden_to_tokens builds the
// LM-head graph on the back shard (where the lm_head weights live).

#pragma once

#include "common/dflash_target.h"
#include "layer_split_types.h"
#include "layer_split_forward.h"
#include "dflash_feature_ring.h"
#include "dflash_draft_ipc.h"
#include "step_graph.h"

#include <vector>

namespace dflash27b {

class Qwen35LayerSplitDFlashTarget : public DFlashTarget {
public:
    // All references/pointers are non-owning; caller controls lifetime.
    Qwen35LayerSplitDFlashTarget(std::vector<TargetLayerSplitShard> & shards,
                                 DraftFeatureMirror * feature_ring,
                                 int kq_stride_pad,
                                 int fa_window,
                                 DFlashDraftIpcClient * remote_draft = nullptr);

    ~Qwen35LayerSplitDFlashTarget() override;

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr) override;

    bool snapshot_kv() override;
    bool restore_kv() override;

    bool is_eos(int token) const override;

    bool embed_tokens(const int32_t * tokens, int n,
                      float * out) const override;

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;

    int hidden_size() const override;
    int mask_token_id() const override;
    const std::vector<int> & capture_layer_ids() const override;

private:
    std::vector<TargetLayerSplitShard> & shards_;
    DraftFeatureMirror *                 feature_ring_;
    int                                  kq_stride_pad_;
    int                                  fa_window_;
    DFlashDraftIpcClient *               remote_draft_;

    std::vector<int> capture_ids_;
    StepGraph        proj_sg_;
};

}  // namespace dflash27b
