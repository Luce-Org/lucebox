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
#include "placement/placement_config.h"
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
    int          stream_fd   = -1;

    // FA/KV
    int          fa_window       = 2048;
    int          kq_stride_pad   = 32;   // KQ_MASK_PAD or 256 for TBQ

    // Draft
    int          draft_swa_window = 0;
    int          draft_ctx_max    = 4096;

    // Speculative decode strategy
    bool         fast_rollback   = false;
    bool         seq_verify      = false;
    bool         ddtree_mode     = false;
    int          ddtree_budget   = 64;
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

    void shutdown() override;

    // Release oversized scratch buffers (gallocr, BSA cache) between requests
    // to prevent VRAM growth over time.
    void release_scratch() override;

private:
    // ── Configuration ────────────────────────────────────────────────
    Qwen35Config cfg_;

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
    bool do_spec_decode(int committed, int n_gen,
                        std::vector<int32_t> & out_tokens,
                        const DaemonIO & io,
                        const std::vector<int32_t> * hint_tokens = nullptr,
                        const BudgetHook * budget_hook = nullptr,
                        bool * forced_close_out = nullptr,
                        bool * soft_close_out = nullptr);

    // AR decode fallback (no draft model or sampling mode).
    // budget_hook (when close_token_ids is non-empty) overrides the next
    // sampled token(s) with the close-tag sequence once (n_gen - committed)
    // <= hard_limit. Mirrors antirez/ds4 ds4_eval.c's hard_limit_reply_budget.
    // forced_close_out, when non-null, is set to true iff the hook injected
    // the close sequence (vs. the model self-closing at the boundary). The
    // server uses this to attribute close_kind=hard correctly — decoding
    // the token stream and grepping for "</think>" cannot distinguish an
    // injected close from a natural one because the bytes are identical.
    // soft_close_out, when non-null, is set to true iff the soft-limit
    // top-K peek (spec §5.3) accepted the close (close_kind=soft) —
    // distinct from forced_close_out which only fires for hard overrides.
    bool do_ar_decode(int committed, int n_gen,
                      std::vector<int32_t> & out_tokens,
                      const DaemonIO & io,
                      const BudgetHook & budget_hook = {},
                      bool * forced_close_out = nullptr,
                      bool * soft_close_out = nullptr);

    // Chain-mode verify (single batch of q_len tokens).
    int verify_chain(int committed, const int32_t * draft_tok, int q_len);

    // DDTree tree-mode verify.
    int verify_tree(int committed, const DDTree & tree);
};

}  // namespace dflash::common
