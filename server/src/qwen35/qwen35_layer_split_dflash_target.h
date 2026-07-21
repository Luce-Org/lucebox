// Qwen35LayerSplitDFlashTarget — DFlashTarget adapter for qwen35 multi-GPU
// layer-split inference.
//
// Wraps a vector of Qwen35LayerSplitShard behind the generic DFlashTarget
// interface so the common spec-decode loop can drive layer-split verify
// without knowing about shard layout, qwen attention parameters, or the
// remote-draft IPC client.
//
// snapshot_kv / restore_kv iterate every shard. verify_batch delegates to
// run_qwen35_layer_split_forward. project_hidden_to_tokens builds the
// LM-head graph on the back shard (where the lm_head weights live).

#pragma once

#include "common/dflash_target.h"
#include "layer_split_types.h"
#include "layer_split_forward.h"
#include "dflash_feature_ring.h"
#include "dflash_draft_ipc.h"
#include "qwen35_target_shard_ipc.h"
#include "step_graph.h"

#include <string>
#include <vector>

namespace dflash::common {

class KvFlashPager;

class Qwen35LayerSplitDFlashTarget : public DFlashTarget {
public:
    // All references/pointers are non-owning; caller controls lifetime.
    Qwen35LayerSplitDFlashTarget(std::vector<Qwen35LayerSplitShard> & shards,
                                 DraftFeatureMirror * feature_ring,
                                 int kq_stride_pad,
                                 int fa_window,
                                 DFlashDraftIpcClient * remote_draft = nullptr,
                                 Qwen35TargetShardIpcClient * remote_target_shard = nullptr,
                                 KvFlashPager * kvflash = nullptr);

    ~Qwen35LayerSplitDFlashTarget() override;

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override;

    bool snapshot_kv() override;
    bool restore_kv() override;

    bool supports_fast_rollback() const override;
    bool rollback_to(int base_pos, int commit_n) override;
    bool supports_tree_verify() const override;
    bool verify_tree(int committed,
                     const DDTree & tree,
                     const std::vector<int32_t> & flat_tokens,
                     int n_alloc,
                     std::vector<int32_t> & posterior_out,
                     std::vector<float> * logits_out = nullptr) override;
    bool rollback_to_tree(int committed,
                          const DDTree & tree,
                          const std::vector<int> & accepted_dfs) override;
    bool last_rollback_context_fatal() const { return last_rollback_context_fatal_; }

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
    std::vector<Qwen35LayerSplitShard> & shards_;
    DraftFeatureMirror *                 feature_ring_;
    int                                  kq_stride_pad_;
    int                                  fa_window_;
    DFlashDraftIpcClient *               remote_draft_;
    Qwen35TargetShardIpcClient *         remote_target_shard_;
    KvFlashPager *                       kvflash_;

    std::vector<int> capture_ids_;
    StepGraph        proj_sg_;
    bool             split_capture_validated_ = false;
    bool             last_rollback_context_fatal_ = false;
    bool             rollback_poisoned_ = false;
    mutable int      required_tree_slots_ = 1;
};

}  // namespace dflash::common
