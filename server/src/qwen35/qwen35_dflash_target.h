// Qwen35DFlashTarget — DFlashTarget implementation for qwen35 hybrid models.
//
// Wraps the existing qwen35 target infrastructure (TargetWeights, TargetCache,
// StepGraph, DraftFeatureMirror) behind the generic DFlashTarget interface.
// This adapter enables the generic spec-decode loop to drive qwen35 verification.

#pragma once

#include "common/dflash_target.h"
#include "internal.h"         // TargetWeights, TargetCache, DraftWeights
#include "step_graph.h"
#include "graph_builders.h"
#include "kvflash_pager.h"
#include "gdn_single_commit.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda-gdn-chain.h"

#include <vector>

namespace luce::common {

class Qwen35DFlashTarget : public DFlashTarget {
public:
    // Non-owning references — caller must ensure lifetime.
    Qwen35DFlashTarget(TargetWeights & w,
                       TargetCache & cache,
                       ggml_backend_t backend,
                       StepGraph & sg,
                       int kq_stride_pad,
                       int fa_window);

    ~Qwen35DFlashTarget() override;

    // ── DFlashTarget interface ──────────────────────────────────────

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override;

    bool read_verify_logits(int n_tokens, std::vector<float> & out) override;
    bool read_verify_row(int row, std::vector<float> & out) override;
    void set_verify_top_n(int top_n) override { verify_top_n_ = top_n; }
    bool read_verify_top(int n_rows, std::vector<float> & vals,
                         std::vector<int32_t> & ids, int & top_n,
                         int & vocab) override;

    bool snapshot_kv() override;
    bool restore_kv() override;
    bool supports_fast_rollback() const override;
    bool verify_pool_exhausted() const override { return verify_pool_exhausted_; }
    bool exact_fast_rollback() const override { return specla_active(); }
    bool rollback_failure_is_recoverable() const override { return !specla_active(); }
    bool rollback_to(int base_pos, int commit_n) override;
    bool finish_speculative_state() override;

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
    int mask_token_id() const override;
    ggml_tensor *  lm_head_tensor() override { return w_.output; }
    const std::vector<int> & capture_layer_ids() const override;

    // kvflash mode: verify writes are slot-mapped via the pager and the
    // attention mask carries slot validity (resident committed positions
    // only) plus causal structure among the verify tokens. Rejected draft
    // tokens need no explicit rollback: their slots are excluded by the
    // pos < base_pos validity rule on the next verify and get rewritten.
    // Forces fa_window = 0 (logical windowing is meaningless in slot space).
    void set_kvflash_pager(KvFlashPager * pager) { pager_ = pager; }

    // Enable fast-rollback mode: verify will capture per-step SSM intermediates
    // so rollback_to() can restore recurrent state without replay.
    void set_fast_rollback(bool enabled) { fast_rollback_ = enabled; }

    // Rotary positions run ahead of KV positions by this many after an image
    // (image tokens take 2D positions). Points at the owner's per-request
    // offset, which every prefill sets; null means zero.
    void set_rope_offset(const int * offset) { rope_offset_ = offset; }

    // DDTree servers keep the parent recurrent-state handling: the GDN single
    // commit (gdn_single_commit.h) stays inert there.
    void set_ddtree_mode(bool enabled) { ddtree_mode_ = enabled; }

private:
    TargetWeights & w_;
    TargetCache & cache_;
    ggml_backend_t backend_;
    StepGraph & sg_;
    int kq_stride_pad_;
    int fa_window_;
    int verify_top_n_ = 0;   // set_verify_top_n
    KvFlashPager * pager_ = nullptr;
    bool fast_rollback_ = false;
    bool ddtree_mode_ = false;
    bool verify_pool_exhausted_ = false;   // last verify_batch failed on a pool slot, nothing touched
    const int * rope_offset_ = nullptr;
    int rope_offset() const { return rope_offset_ ? *rope_offset_ : 0; }

    // GDN single commit (gdn_single_commit.h, state machine there): where the
    // committed and the pre-step SSM state live between snapshot_kv, verify
    // and rollback_to / restore_kv. Untouched on the paths the parts
    // exclude.
    gdn_commit::StateBook gdn_book_;
    gdn_commit::PathFacts gdn_path_facts() const;
    // KEEP_BASE decision for one verify_batch call.
    bool gdn_keep_base_verify(bool capture_requested, bool do_capture,
                                int n_tokens) const;
    // A keep-base verify reached `where` without rollback_to or restore_kv.
    void gdn_drop_uncommitted(const char * where);
    // SWAP_SNAPSHOT decision for one snapshot_kv call.
    bool gdn_swap_snapshot() const;
    bool gdn_pairs_exchangeable() const;
    // Take a deferred SSM snapshot (spare <- live, then synchronize, as the
    // parent's snapshot_ssm_state) before something writes the live state.
    void gdn_materialize_snapshot();
    // SINGLE_REPLAY decision for one keep-base verify of n_tokens: the
    // one-slot state map when it records the replay log, else nullptr (the
    // verify keeps the F32 checkpoints).
    ggml_tensor * gdn_single_replay_ids(int n_tokens);
    // The I32 [1] map holding slab 0, allocated on first use.
    ggml_tensor * gdn_replay_slot_map();
    // Fill gdn_replay_layers_ for a replay commit of `plan` after a verify
    // of q_len tokens; false (nothing written) when a capture or state does
    // not match. S_v and H out.
    bool gdn_prepare_replay_commit(const gdn_commit::RollbackPlan & plan,
                                     int q_len, int64_t & S_v, int64_t & H);
    ggml_context * gdn_slot_ctx_ = nullptr;
    ggml_backend_buffer_t gdn_slot_buf_ = nullptr;
    ggml_tensor * gdn_slot_ids_ = nullptr;
    std::vector<ggml_cuda_gdn_chain_commit_layer> gdn_replay_layers_;

    // SpecLA (docs/SPECLA.md): true when the cache was
    // migrated with factor buffers. Capture-verify then runs the
    // topology-masked factor path, never mutates durable SSM/conv state
    // (snapshot/restore become no-ops), and rollback commits via
    // DeltaConstruct instead of dense checkpoint copies.
    bool specla_active() const {
        return fast_rollback_ && pager_ == nullptr && !cache_.factor_k.empty();
    }

    // SpecLA chain commit: DeltaConstruct over the accepted prefix plus a
    // fused shift/append of raw convolution factors.
    bool rollback_to_specla(int base_pos, int commit_n);

    // Cached vector form of capture layer IDs (built once in constructor).
    std::vector<int> capture_ids_;

    // LM-head projection graph (lazily built).
    StepGraph proj_sg_;
};

}  // namespace luce::common
