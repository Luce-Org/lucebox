// Qwen35Backend — ModelBackend implementation for the Qwen3.5 hybrid
// (attention + DeltaNet/SSM) architecture with speculative decoding via a
// DFlash draft model.
//
// Manages two models on potentially different GPUs:
//   - Target: 27B parameter qwen35 hybrid model
//   - Draft:  small DFlash speculative prefill model
//
// Generation strategy: DDTree/chain-mode speculative decoding with SSM state
// rollback and replay.

#pragma once

#include "common/model_backend.h"
#include "common/dflash_target.h"
#include "common/dflash_draft_ipc.h"
#include "placement/placement_config.h"
#include "placement/remote_draft_config.h"
#include "step_graph.h"
#include "ddtree.h"
#include "dflash_feature_ring.h"
#include "internal.h"         // TargetWeights, TargetCache, DraftWeights, PrefixSnapshot
#include "qwen3/qwen3_drafter.h"  // DrafterContext, load_drafter, free_drafter, drafter_score_and_compress

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <cstddef>

namespace dflash::common {

// ── Configuration passed at construction ────────────────────────────────

struct Qwen35Config {
    const char * target_path = nullptr;
    const char * draft_path  = nullptr;
    DevicePlacement device;                // target GPU placement
    int          draft_gpu   = 0;
    RemoteDraftConfig remote_draft;
    int          stream_fd   = -1;

    // FA/KV
    int          fa_window       = 0;  // 0 = full attention. qwen3.6 full-attn layers must see the whole context; a finite window drops the system prompt/tools -> breaks tool calls.
    int          kq_stride_pad   = 32;   // KQ_MASK_PAD or 256 for TBQ

    // Draft
    int          draft_swa_window = 0;
    int          draft_ctx_max    = 4096;

    // Draft YaRN rope scaling (set by caller for models that need it).
    float        draft_yarn_factor    = 0.0f;  // 0 = no YaRN; >1 = enable
    float        draft_yarn_beta_fast = 32.0f;
    float        draft_yarn_beta_slow = 1.0f;
    int          draft_yarn_orig_ctx  = 4096;

    // Speculative decode strategy
    bool         fast_rollback   = false;
    bool         seq_verify      = false;
    bool         ddtree_mode     = false;
    int          ddtree_budget   = 22;
    float        ddtree_temp     = 1.0f;
    bool         ddtree_chain_seed = true;
    bool         use_feature_mirror = false;
};

// ── Backend class ───────────────────────────────────────────────────────

class Qwen35Backend : public ModelBackend {
public:
    explicit Qwen35Backend(const Qwen35Config & cfg);
    ~Qwen35Backend() override;

    // Non-copyable, non-movable (owns GPU resources).
    Qwen35Backend(const Qwen35Backend &) = delete;
    Qwen35Backend & operator=(const Qwen35Backend &) = delete;

    // ── Initialization ───────────────────────────────────────────────
    // Load target + draft models, create KV caches.
    // Returns false on failure (check dflash27b_last_error()).
    bool init();

    // ── ModelBackend interface ────────────────────────────────────────
    void print_ready_banner() const override;

    bool park(const std::string & what) override;
    bool unpark(const std::string & what) override;
    bool is_target_parked() const override { return target_parked_; }

    GenerateResult generate(const GenerateRequest & req,
                            const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate(int slot,
                                        const GenerateRequest & req,
                                        const DaemonIO & io) override;

    SnapshotRef snapshot_ref(int slot) const override;
    bool snapshot_adopt(int slot, ggml_context * ctx,
                        ggml_backend_buffer_t buf, int cur_pos,
                        int32_t last_tok = -1) override;

    CompressResult compress(const CompressRequest & req) override;
    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    bool try_handle_command(const std::string & line,
                            const DaemonIO & io) override;

    bool supports_dflash_spec_decode() const override { return true; }
    DFlashTarget * dflash_target() override;
    bool supports_remote_draft() const override { return true; }

    void shutdown() override;

    // Release oversized scratch buffers (gallocr, BSA cache) between requests
    // to prevent VRAM growth over time.
    void release_scratch() override;

protected:
    virtual bool load_target_model(ggml_backend_t backend, TargetWeights & out);
    virtual bool run_ar_decode_path(int committed, int n_gen,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io);
    virtual bool should_capture_moe_router() const { return false; }
    virtual void after_target_compute(StepGraph &,
                                      int /*kv_start*/,
                                      int /*n_tokens*/) {}

    TargetWeights & target_weights() { return w_; }
    const TargetWeights & target_weights() const { return w_; }
    TargetCache & target_cache() { return cache_; }
    const TargetCache & target_cache() const { return cache_; }
    ggml_backend_t target_backend() const { return target_backend_; }
    StepGraph & target_step_graph() { return sg_; }
    const StepGraph & target_step_graph() const { return sg_; }
    SamplerCfg & sampler_config() { return sampler_; }
    std::mt19937_64 & sampler_rng_engine() { return sampler_rng_; }
    bool prefill_logits_valid() const { return prefill_last_logits_valid_; }
    std::size_t prefill_logits_offset() const { return prefill_last_logits_offset_; }

    // Accessors for draft/spec-decode state (needed by hybrid spec-decode in subclass)
    DraftWeights & draft_weights() { return dw_; }
    const DraftWeights & draft_weights() const { return dw_; }
    ggml_backend_t draft_backend() const { return draft_backend_; }
    DraftFeatureMirror & feature_mirror() { return feature_mirror_; }
    const DraftFeatureMirror & feature_mirror() const { return feature_mirror_; }
    bool is_draft_parked() const { return draft_parked_; }

    // ── Configuration ────────────────────────────────────────────────
    Qwen35Config cfg_;

private:
    // ── GPU backends ─────────────────────────────────────────────────
    ggml_backend_t target_backend_ = nullptr;
    ggml_backend_t draft_backend_  = nullptr;
    ggml_backend_t snap_backend_   = nullptr;  // snapshot storage (CPU or unified)
    bool           split_gpus_     = false;

    // ── Model weights + caches ───────────────────────────────────────
    TargetWeights  w_;
    DraftWeights   dw_;
    TargetCache    cache_;

    // ── Graph containers (persistent gallocr buffers) ────────────────
    StepGraph      sg_;           // target forward (verify / prefill)
    StepGraph      draft_sg_;    // draft forward
    StepGraph      proj_sg_;     // lm-head projection (remote-lm-head mode)

    // ── Draft feature mirror (cross-GPU feature transfer) ────────────
    DraftFeatureMirror feature_mirror_;
    DFlashDraftIpcClient remote_draft_;

    // ── Prefix cache (snapshots) ─────────────────────────────────────
    static constexpr int PREFIX_SLOTS = 64;
    PrefixSnapshot prefix_snapshots_[PREFIX_SLOTS];

    // ── Park state ───────────────────────────────────────────────────
    bool target_parked_ = false;
    bool draft_parked_  = false;

    // ── Pflash drafter (lazy-loaded) ─────────────────────────────────
    DrafterContext drafter_ctx_;
    bool           drafter_loaded_ = false;

    // ── Sampler state ────────────────────────────────────────────────
    SamplerCfg      sampler_;
    std::mt19937_64 sampler_rng_{std::random_device{}()};

    // Last prefill chunk metadata, used to sample the first generated token
    // without deriving a chunk-local offset from absolute KV position.
    std::size_t     prefill_last_logits_offset_ = 0;
    bool            prefill_last_logits_valid_  = false;

    // ── DFlashTarget adapter (lazy-built) ────────────────────────────
    std::unique_ptr<DFlashTarget> dflash_target_;

    // ── Internal helpers ─────────────────────────────────────────────
    // Prefill a prompt and return the number of tokens committed to KV.
    // kv_offset > 0 resumes from a restored snapshot: tokens are placed at
    // KV positions [kv_offset, kv_offset + tokens.size()) instead of [0, N).
    int do_prefill(const std::vector<int32_t> & tokens,
                   const DaemonIO & io,
                   int snap_pos = -1, int snap_slot = -1,
                   int kv_offset = 0);

    // Speculative decode loop: draft → verify → accept until EOS/max.
    // When budget_hook is non-null and (n_gen - generated) drops to the
    // hard-limit boundary, breaks out of the spec-decode loop and tails
    // off via do_ar_decode so the force-close override fires cleanly
    // with KV state intact. Spec-decode itself can't safely inject the
    // close token mid-batch (verify-and-accept assumes the sampled
    // tokens are the ones that got committed), so the boundary switch
    // is the simplest correct integration.
    // out_accept_rate receives accepted/total draft token ratio (0.0 if AR fallback).
    // out_spec_ran is true when spec decode actually ran (even with 0 accepts).
    bool do_spec_decode(int committed, int n_gen,
                        std::vector<int32_t> & out_tokens,
                        const DaemonIO & io,
                        float & out_accept_rate,
                        bool & out_spec_ran,
                        const std::vector<int32_t> * hint_tokens = nullptr,
                        const BudgetHook * budget_hook = nullptr,
                        bool * forced_close_out = nullptr,
                        bool * degenerate_close_out = nullptr,
                        bool * soft_forced_close_out = nullptr);

    // AR decode fallback (no draft model or sampling mode).
    // budget_hook (when close_token_ids is non-empty) overrides the next
    // sampled token(s) with the close-tag sequence once (n_gen - committed)
    // <= hard_limit. For Qwen3.x, close_token_ids is the canonical
    // "Considering the limited time..." summarize-and-stop lead-in (24
    // tokens including `</think>`); for non-qwen arches it's a single
    // close-tag token. Mirrors the trained pathway documented in the
    // Qwen3 technical report (arXiv 2505.09388).
    // forced_close_out, when non-null, is set to true iff the hook injected
    // the close sequence (vs. the model self-closing at the boundary). The
    // server uses this to attribute close_kind=hard correctly — decoding
    // the token stream and grepping for "</think>" cannot distinguish an
    // injected close from a natural one because the bytes are identical.
    bool do_ar_decode(int committed, int n_gen,
                      std::vector<int32_t> & out_tokens,
                      const DaemonIO & io,
                      const BudgetHook & budget_hook = {},
                      bool * forced_close_out = nullptr,
                      bool * degenerate_close_out = nullptr,
                      bool * soft_forced_close_out = nullptr);

    bool sync_remote_draft_features(int start_pos, int n_tokens);

    // Chain-mode verify (single batch of q_len tokens).
    int verify_chain(int committed, const int32_t * draft_tok, int q_len);

    // DDTree tree-mode verify.
    int verify_tree(int committed, const DDTree & tree);
};

}  // namespace dflash::common
