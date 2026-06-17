// LagunaDFlashTarget - DFlashTarget adapter for Poolside Laguna-XS.2.
//
// Laguna is a pure-attention iSWA + MoE model, so speculative verification
// follows the Gemma4 chain-verify pattern with no SSM rollback path.

#pragma once

#include "common/dflash_target.h"
#include "laguna_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace dflash::common {

class LagunaDFlashTarget : public DFlashTarget {
public:
    LagunaDFlashTarget(LagunaTargetWeights & w,
                       LagunaTargetCache & cache,
                       ggml_backend_t backend);

    ~LagunaDFlashTarget() override;

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override;

    bool read_verify_logits(int n_tokens, std::vector<float> & out) override;

    void set_kvflash_pager(class KvFlashPager * p) { pager_ = p; }
    void set_keep_verify_logits(bool enabled) { keep_verify_logits_ = enabled; }

    bool snapshot_kv() override;
    bool restore_kv() override;

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

    bool is_eos(int token) const override;

    bool embed_tokens(const int32_t * tokens, int n,
                      float * out) const override;

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;

    bool project_hidden_to_topk(const float * hidden,
                                int n_tokens,
                                int K,
                                float temperature,
                                std::vector<float> & top_log_probs,
                                std::vector<int32_t> & top_token_ids) override;

    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override { return 12; }
    const std::vector<int> & capture_layer_ids() const override;

private:
    LagunaTargetWeights & w_;
    LagunaTargetCache & cache_;
    ggml_backend_t backend_;
    class KvFlashPager * pager_ = nullptr;
    std::vector<int> capture_ids_;
    LagunaCacheSnapshot verify_snap_;
    bool keep_verify_logits_ = false;
    int verify_logits_n_ = 0;
    std::vector<float> verify_logits_;
};

}  // namespace dflash::common
