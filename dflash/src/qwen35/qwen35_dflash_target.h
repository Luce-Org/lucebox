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

#include "ggml.h"
#include "ggml-backend.h"
#include "device_runtime.h"  // cudaStream_t

#include <vector>

namespace dflash27b {

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

    ggml_backend_t backend() const override { return backend_; }

    // Phase B+ fused-LM-head path: the Qwen3.6 MTP head's step graph can
    // append `mul_mat(w_.output, x_normed) -> argmax` directly so it
    // avoids a hidden -> host -> separate-cgraph round trip per call.
    ggml_tensor * lm_head_weight() const override { return w_.output; }

    // Mirror the causal window onto MTP head's flash-attn so it sees the
    // same active context as the target's full-attention blocks.
    int fa_window() const override { return fa_window_; }

    // Enable per-position post-norm hidden capture during verify_batch.
    // Off by default; MTP modules that depend on hidden_at_pos() flip it on
    // in attach() — and ALSO pin capture_mode_ to FULL_SEQ so the runtime
    // toggle below cannot demote it to LAST_ROW_ONLY (which captures the
    // wrong row for partial-accept iterations of the MTP chain and silently
    // returns null from hidden_at_pos_pre_norm).
    void enable_hidden_seq_capture(bool on) override {
        capture_hidden_seq_ = on;
        if (on) {
            capture_mode_ = VerifyCaptureMode::FULL_SEQ;
            capture_pinned_ = true;
        } else {
            capture_pinned_ = false;
        }
    }

    // Hidden-sequence capture granularity.  FULL_SEQ downloads the entire
    // [n_tokens, n_embd] post-norm + pre-norm hidden tensors device->host
    // on every verify_batch (needed by warm_head_kv during prefill, which
    // consumes per-position hiddens via last_hidden_seq()).  LAST_ROW_ONLY
    // downloads only row n_tokens-1 — sufficient for decode-side chain
    // verifies whose only consumer is hidden_at_pos(base_pos - 1), i.e. the
    // last token of the just-verified batch.  Switching to LAST_ROW_ONLY
    // after prefill collapses two ~80 KB device->host syncs (post-norm +
    // pre-norm, hidden_dim=5120, D+1=4 tokens) into two ~20 KB single-row
    // syncs and saves the 2x WSL2 scheduler latency hit per verify.
    enum class VerifyCaptureMode {
        FULL_SEQ,        // default — required during prefill / warm_head_kv
        LAST_ROW_ONLY,   // decode mode — only hidden_at_pos(base_pos-1) used
    };
    // Runtime toggle is a NO-OP once MTP has pinned capture to FULL_SEQ. This
    // makes the partial-accept-returns-null bug impossible by construction —
    // non-MTP callers can still use this freely; MTP callers cannot demote it.
    void set_hidden_capture_mode(VerifyCaptureMode mode) {
        if (capture_pinned_) return;
        capture_mode_ = mode;
    }
    VerifyCaptureMode hidden_capture_mode() const { return capture_mode_; }

    void set_hidden_capture_scope(DFlashTarget::VerifyCaptureScope scope) override {
        if (capture_pinned_) return;
        capture_mode_ = (scope == DFlashTarget::VerifyCaptureScope::LAST_ROW_ONLY)
            ? VerifyCaptureMode::LAST_ROW_ONLY
            : VerifyCaptureMode::FULL_SEQ;
    }

    // ── DFlashTarget interface ──────────────────────────────────────

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr) override;

    // Tree-verify override.  Stage 1 stub: only handles the degenerate
    // single-token case (tree.n_nodes == 0) by dispatching to verify_batch.
    // For real DDTree shapes (n_nodes > 0) it returns false so the harness
    // falls back to chain-verify.  Stage 2 will wire build_target_step_tree
    // + ancestor mask here.
    bool verify_tree(const std::vector<int32_t> & flat_tokens,
                     const DDTree & tree,
                     int base_pos,
                     std::vector<int32_t> & out_argmax,
                     std::vector<float> * out_logits = nullptr) override;

    bool snapshot_kv() override;
    bool restore_kv() override;

    // Rollback DeltaNet SSM/conv + full-attn KV to the accepted-path tail of
    // the most recent verify_tree() call.  Dequantizes per-DFS-slot SSM
    // snapshots; compacts full-attn KV when accepted walk deviates from DFS spine.
    // Postcondition: cache_.cur_pos = root_base_pos + commit_n.
    // Returns false if verify_tree was not called or accepted_dfs is out of range.
    bool restore_kv_at_dfs(const std::vector<int> & accepted_dfs) override;

    // Chain-mode rollback: builds spine [0..accept_n] and dispatches to
    // restore_kv_at_dfs.  Requires chain capture enabled on the prior verify_batch.
    // Returns false if capture was off; caller falls back to snapshot+recommit.
    bool restore_kv_at_chain(int accept_n) override;

    // Enable per-position DeltaNet intermediate capture in verify_batch.
    // Off by default; unsafe when n_tokens > max_verify_tokens.
    // When on, verify_batch populates ssm_intermediate/conv_input_cache and
    // records a linear-chain topology for restore_kv_at_chain.
    void enable_chain_capture(bool on) override { chain_capture_enabled_ = on; }

    // Record linear-chain topology (spine [0..n_tokens-1] at base_pos) so
    // restore_kv_at_chain() can locate the rollback slot.
    void capture_topology_for_chain(int n_tokens, int base_pos) override;

    bool is_eos(int token) const override;

    bool embed_tokens(const int32_t * tokens, int n,
                      float * out) const override;

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;

    bool project_hidden_to_logits(const float * hidden,
                                  int n_tokens,
                                  std::vector<float> & logits_out,
                                  int & out_vocab) override;

    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override;
    const std::vector<int> & capture_layer_ids() const override;

    // Return the backbone's final post-norm hidden state for the last committed
    // token (n_embd floats, F32).  Populated by the most recent verify_batch.
    // Returns nullptr if verify_batch has not been called yet.
    const float * last_hidden() const override { return last_hidden_cpu_.empty() ? nullptr : last_hidden_cpu_.data(); }

    // Full post-norm hidden sequence from the last verify_batch, laid out as
    // [token_0_hidden, ..., token_{n_tokens-1}_hidden], n_tokens * n_embd floats.
    const float * last_hidden_seq(int * out_n_tokens) const override {
        if (out_n_tokens) *out_n_tokens = last_hidden_seq_n_;
        return last_hidden_seq_cpu_.empty() ? nullptr : last_hidden_seq_cpu_.data();
    }

    // Absolute-position accessor: returns &last_hidden_seq[(pos - chunk_start) * n_embd]
    // if pos is within [last_verify_chunk_start_, last_verify_chunk_start_ + n_seq).
    const float * hidden_at_pos(int abs_pos) const override {
        const int rel = abs_pos - last_verify_chunk_start_;
        if (rel < 0 || rel >= last_hidden_seq_n_ ||
            last_hidden_seq_cpu_.empty()) {
            return nullptr;
        }
        return last_hidden_seq_cpu_.data() + (size_t)rel * w_.n_embd;
    }

    // Pre-final-output-norm variant of hidden_at_pos.  Mirrors llama.cpp
    // PR #22673's `t_h_pre_norm`: the Qwen3.6 MTP head's hnorm normalises
    // h_prev internally, so feeding it the post-output-norm tensor double-
    // normalises and compounds per-depth rejection.  The chain GPU loop's
    // outer h_prev_0 seed must use this accessor; the intra-iter re-feed
    // already uses the MTP graph's pre-norm output (state_->last_hidden).
    // Returns nullptr if the most recent verify_batch did NOT capture the
    // pre-norm sequence (enable_hidden_seq_capture(true) flips this on).
    const float * hidden_at_pos_pre_norm(int abs_pos) const override {
        const int rel = abs_pos - last_verify_chunk_start_;
        if (rel < 0 || rel >= last_hidden_seq_n_ ||
            last_hidden_seq_pre_norm_cpu_.empty()) {
            return nullptr;
        }
        return last_hidden_seq_pre_norm_cpu_.data() + (size_t)rel * w_.n_embd;
    }

private:
    TargetWeights & w_;
    TargetCache & cache_;
    ggml_backend_t backend_;
    StepGraph & sg_;
    int kq_stride_pad_;
    int fa_window_;

    // Cached vector form of capture layer IDs (built once in constructor).
    std::vector<int> capture_ids_;

    // LM-head projection graph (lazily built).
    StepGraph proj_sg_;

    // CPU-side copy of the last token's post-norm hidden state.
    // Filled by verify_batch after each GPU graph_compute.
    mutable std::vector<float> last_hidden_cpu_;

    // CPU-side copy of the full [n_tokens, n_embd] post-norm hidden sequence
    // from the last verify_batch.  Used by Qwen3.6 MTP warm_head_kv to seed
    // the head's per-position K/V cache during prefill.
    mutable std::vector<float> last_hidden_seq_cpu_;
    // CPU-side copy of the full [n_tokens, n_embd] PRE-final-output-norm
    // hidden sequence — populated alongside last_hidden_seq_cpu_ when
    // capture_hidden_seq_ is on.  Consumed by hidden_at_pos_pre_norm() so
    // the Qwen3.6 MTP chain's outer h_prev_0 seed dodges double-normalisation.
    mutable std::vector<float> last_hidden_seq_pre_norm_cpu_;
    mutable int                last_hidden_seq_n_ = 0;
    // Absolute position of the FIRST token captured in last_hidden_seq_cpu_.
    // Used by hidden_at_pos to translate an absolute sequence position to an
    // index inside the captured chunk.  -1 means no captured chunk yet.
    mutable int                last_verify_chunk_start_ = -1;

    // Whether verify_batch should request the full post-norm hidden sequence
    // and copy it to the host.  Toggled by enable_hidden_seq_capture().
    bool                       capture_hidden_seq_ = false;

    // Granularity of the device->host download when capture_hidden_seq_ is on.
    // Toggled by set_hidden_capture_mode().  See VerifyCaptureMode docs.
    VerifyCaptureMode          capture_mode_ = VerifyCaptureMode::FULL_SEQ;

    // Pinned by enable_hidden_seq_capture(true) — once an MTP module attaches
    // and enables capture, the mode cannot be demoted to LAST_ROW_ONLY (that
    // captures the wrong row for partial-accept chain iters and silently
    // returns null from hidden_at_pos_pre_norm).
    bool                       capture_pinned_   = false;

#ifdef DFLASH_VERIFY_PROFILE
    // Per-instance accumulators: summed wall-clock (ms) per verify_batch call;
    // dumped from destructor. Zero-cost when flag is off.
    mutable double             vprof_sum_set_         = 0.0;
    mutable double             vprof_sum_compute_     = 0.0;
    mutable double             vprof_sum_get_hidden_  = 0.0;
    mutable double             vprof_sum_get_hpre_    = 0.0;
    mutable double             vprof_sum_get_argmax_  = 0.0;
    mutable double             vprof_sum_total_       = 0.0;
    mutable long long          vprof_n_calls_         = 0;
#endif  // DFLASH_VERIFY_PROFILE

    // ── Stage 3: state captured by verify_tree for restore_kv_at_dfs() ──
    // base_pos of the most recent verify_tree call (= root slot in KV).
    int                        last_tree_base_pos_ = -1;
    // n_nodes of the most recent verify_tree.
    int                        last_tree_n_nodes_  = 0;
    // Copies of tree.parents/depths from the most recent verify_tree so
    // rollback's conv-window walk (which traverses ancestry, not DFS) is
    // self-contained.
    std::vector<int>           last_tree_parents_;
    std::vector<int>           last_tree_depths_;

    // When true, verify_batch will populate per-position DeltaNet
    // ssm_intermediate / conv_input_cache buffers AND record a linear-chain
    // topology in last_tree_* so restore_kv_at_chain() can dispatch to
    // restore_kv_at_dfs() on partial-accept.  Toggled by the MTP chain runner
    // via enable_chain_capture(); off by default (capture is unsafe on
    // n_tokens > max_verify_tokens, e.g. 512-token prefill chunks, where the
    // in-graph ggml_view_3d into the conv_input cache asserts).
    bool                       chain_capture_enabled_ = false;

    // Dedicated CUDA stream for restore_kv_at_dfs copies (bug #3): avoids
    // serializing ~384 per-head launches on the default stream.  Created on
    // first use, destroyed in the dtor.
    mutable cudaStream_t       rollback_stream_ = nullptr;
};

}  // namespace dflash27b
