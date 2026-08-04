#pragma once

#include "common/dflash_target.h"
#include "kimi_k3_internal.h"
#include "step_graph.h"

#include <vector>

namespace dflash::common {

struct DraftFeatureMirror;

// Kimi-specific state/capture adapter for the shared DFlash/DSpark runtime.
// Draft execution, Markov correction, confidence gating, acceptance, and
// scheduling deliberately remain in common/.  This class owns only the Kimi
// forward and ReplaySSM boundary.
class KimiK3DFlashTarget final : public DFlashTarget {
public:
    KimiK3DFlashTarget(
        KimiK3Weights & weights,
        KimiK3Cache & cache,
        ggml_backend_t backend,
        DraftFeatureMirror & feature_ring,
        std::vector<int> capture_layer_ids,
        int mask_token_id,
        bool fast_rollback,
        MoeHybridStreamEngine * stream_engine,
        MoeStreamDualOwnerExecutor * dual_stream_executor,
        const MoeStreamDualOwnerPolicy * stream_owner_policy,
        MoeHybridRoutingStats * routing_stats);
    ~KimiK3DFlashTarget() override;

    KimiK3DFlashTarget(const KimiK3DFlashTarget &) = delete;
    KimiK3DFlashTarget & operator=(const KimiK3DFlashTarget &) = delete;

    // Sequential prefill/AR entry point that shares the same feature-capture
    // code as speculative verification.
    bool forward_token(int32_t token, int position, std::vector<float> & logits);

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override;
    bool snapshot_kv() override;
    bool restore_kv() override;
    bool supports_fast_rollback() const override;
    bool prefer_fast_rollback_over_replay() const override;
    bool rollback_to(int base_pos, int commit_n) override;

    bool is_eos(int token) const override;
    bool embed_tokens(const int32_t * tokens, int n, float * out) const override;
    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;
    bool project_hidden_to_logits(const float * hidden,
                                  int n_tokens,
                                  std::vector<float> & logits_out) override;

    ggml_tensor * lm_head_tensor() override;
    ggml_tensor * gpu_embd_table() override;
    ggml_backend_t fused_head_backend() override;
    int hidden_size() const override;
    int mask_token_id() const override;
    const std::vector<int> & capture_layer_ids() const override;
    int default_adaptive_verify_min_rows() const override;

private:
    bool sync_captures(const KimiK3ForwardResult & result,
                       int base_pos,
                       int n_tokens);
    bool build_embedding_graph(int n_tokens) const;
    bool build_projection_graph(int n_tokens);

    KimiK3Weights & weights_;
    KimiK3Cache & cache_;
    ggml_backend_t backend_ = nullptr;
    DraftFeatureMirror & feature_ring_;
    std::vector<int> capture_layer_ids_;
    int mask_token_id_ = -1;
    bool fast_rollback_ = false;
    MoeHybridStreamEngine * stream_engine_ = nullptr;
    MoeStreamDualOwnerExecutor * dual_stream_executor_ = nullptr;
    const MoeStreamDualOwnerPolicy * stream_owner_policy_ = nullptr;
    MoeHybridRoutingStats * routing_stats_ = nullptr;
    mutable StepGraph embedding_graph_;
    StepGraph projection_graph_;
};

} // namespace dflash::common
