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
#include "common/device_placement.h"
#include "step_graph.h"
#include "ddtree.h"
#include "dflash_feature_ring.h"
#include "internal.h"         // TargetWeights, TargetCache, DraftWeights, PrefixSnapshot
#include "qwen3/qwen3_drafter.h"  // DrafterContext, load_drafter, free_drafter, drafter_score_and_compress
#include "qwen36/qwen36_mtp.h"    // Qwen36MtpModule

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash27b {

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

    // MTP (Multi-Token Prediction) speculator — mutually exclusive with draft
    const char * mtp_gguf_path    = nullptr;   // path to fused MTP GGUF (or nullptr = DFlash)
    int          mtp_gamma        = 0;         // max speculation depth
    const char * mtp_draft_source = nullptr;   // "chain" | "mtp_topk" | nullptr -> "chain"
    int          mtp_draft_topk   = 1;         // top-k for mtp_topk mode
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

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    bool try_handle_command(const std::string & line,
                            const DaemonIO & io) override;

    bool supports_dflash_spec_decode() const override { return true; }
    DFlashTarget * dflash_target() override;

    // Test/bench integration hooks for native-head MTP. These keep the
    // Qwen3.6 MTP harness on the backend-owned target/cache/context without
    // exposing the frozen common interfaces.
    bool ensure_decode_cache(int max_verify_tokens);
    ggml_context * tensor_context() const;

    // MTP speculator accessors (ModelBackend interface).
    bool                   supports_mtp() const override { return mtp_module_ != nullptr; }
    mtp::IMtpModule *      mtp()                override { return mtp_module_.get(); }

    void shutdown() override;

private:
    // ── Configuration ────────────────────────────────────────────────
    Qwen35Config cfg_;

    // ── GPU backends ─────────────────────────────────────────────────
    ggml_backend_t target_backend_ = nullptr;
    ggml_backend_t draft_backend_  = nullptr;
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
    static constexpr int PREFIX_SLOTS = 8;
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

    // ── DFlashTarget adapter (lazy-built) ────────────────────────────
    std::unique_ptr<DFlashTarget> dflash_target_;

    // ── MTP speculator (optional, set when cfg_.mtp_gguf_path != nullptr) ──
    std::unique_ptr<mtp::Qwen36MtpModule> mtp_module_;

    // R5: head_kv is only valid for snapshotting after warm_head_kv succeeds.
    // Reset on every generate() entry; flipped true by warm_mtp_for_prompt_
    // and by the orchestrator's post-warm callback. snapshot_save gates the
    // mtp_head_kv capture on this — otherwise a zeroed head_kv would round-trip
    // as a "valid" snapshot.
    bool head_kv_warm_ = false;

    // ── Internal helpers ─────────────────────────────────────────────
    // Prefill a prompt and return the number of tokens committed to KV.
    // kv_offset > 0 resumes from a restored snapshot: tokens are placed at
    // KV positions [kv_offset, kv_offset + tokens.size()) instead of [0, N).
    int do_prefill(const std::vector<int32_t> & tokens,
                   const DaemonIO & io,
                   int snap_pos = -1, int snap_slot = -1,
                   int kv_offset = 0);

    // Speculative decode loop: draft → verify → accept until EOS/max.
    bool do_spec_decode(int committed, int n_gen,
                        std::vector<int32_t> & out_tokens,
                        const DaemonIO & io);

    // AR decode fallback (no draft model or sampling mode).
    bool do_ar_decode(int committed, int n_gen,
                      std::vector<int32_t> & out_tokens,
                      const DaemonIO & io);

    // Chain-mode verify (single batch of q_len tokens).
    int verify_chain(int committed, const int32_t * draft_tok, int q_len);

    // DDTree tree-mode verify.
    int verify_tree(int committed, const DDTree & tree);

    // MTP init: load and attach the Qwen36MtpModule. Called from init() when
    // cfg_.mtp_gguf_path is set. Returns false on failure.
    bool init_mtp_();

    // MTP warm: seed the head KV cache after prefill. Mirrors harness lines
    // 783-792: set_initial_hidden + warm_head_kv. prefill_next is the argmax
    // token produced by the last prefill chunk.
    bool warm_mtp_for_prompt_(const std::vector<int32_t> & prompt,
                              const std::vector<float>    & all_prefill_hidden,
                              int32_t                       prefill_next);

    // MTP prefill: chunked prefill via DFlashTarget::verify_batch, collecting
    // all_prefill_hidden for each chunk so warm_mtp_for_prompt_ can seed the
    // head KV cache. Returns committed KV position (>= 0) or -1 on error.
    // kv_offset > 0 resumes from a snapshot (same semantics as do_prefill).
    int do_mtp_prefill_(const std::vector<int32_t> & tokens,
                        std::vector<float>          & all_prefill_hidden_out,
                        int                           kv_offset = 0);

    // MTP decode: drive MtpChainRunner after prefill. Emits tokens via io.emit
    // (including the terminal -1). Returns false on error.
    bool do_mtp_decode_(int committed, int n_gen,
                        std::vector<int32_t> & out_tokens,
                        const DaemonIO & io);
};

}  // namespace dflash27b
