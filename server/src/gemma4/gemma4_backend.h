// Gemma4Backend — ModelBackend for Gemma4 iSWA+MoE models.
//
// Architecture: iSWA hybrid attention, MoE with shared+routed experts,
// per-layer embeddings, KV sharing, logit softcapping.

#pragma once

#include "common/model_backend.h"
#include "placement/placement_config.h"
#include "common/dflash_feature_ring.h"
#include "common/dflash_draft_graph.h"
#include "gemma4_internal.h"
#include "gemma4_dflash_target.h"
#include "common/sampler.h"
#include "../qwen3/qwen3_drafter.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct Gemma4BackendConfig {
    const char *    model_path = nullptr;
    const char *    draft_path = nullptr;
    int             draft_gpu  = -1;       // GPU for draft model (-1 = same as target)
    int             draft_ctx_max = 2048;  // max context for draft feature mirror
    DevicePlacement device;
    int             stream_fd  = -1;
    int             chunk      = 512;
    int             fa_window  = 0;     // 0 = full attention; >0 = sparse decode window
};

class Gemma4Backend : public ModelBackend {
public:
    explicit Gemma4Backend(const Gemma4BackendConfig & cfg);
    ~Gemma4Backend() override;

    Gemma4Backend(const Gemma4Backend &) = delete;
    Gemma4Backend & operator=(const Gemma4Backend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;

    bool park(const std::string & what) override;
    bool unpark(const std::string & what) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    bool try_handle_command(const std::string & line,
                            const DaemonIO & io) override;

    void shutdown() override;

private:
    Gemma4BackendConfig   cfg_;
    ggml_backend_t        backend_ = nullptr;
    ggml_backend_t        snap_backend_ = nullptr;
    Gemma4Weights         w_;
    Gemma4Cache           cache_;
    bool                  parked_ = false;

    // Sampler
    SamplerCfg            sampler_;
    std::mt19937_64       sampler_rng_{std::random_device{}()};

    // DFlash speculative decode
    ggml_backend_t        draft_backend_ = nullptr;
    DraftWeights          dw_{};
    DraftFeatureMirror    feature_mirror_{};
    Gemma4DFlashTarget *  dflash_target_ = nullptr;
    bool                  draft_parked_ = false;

    // PFlash drafter (compress)
    DrafterContext        drafter_ctx_;
    bool                  drafter_loaded_ = false;

    // Snapshots
    static constexpr int PREFIX_SLOTS = 64;
    Gemma4Snapshot        snapshots_[PREFIX_SLOTS];

    // Prefill prompt tokens in chunks, return absolute committed position.
    // kv_offset: starting KV cache position (0 for fresh prefill, snap_pos for restore).
    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io,
                   int kv_offset = 0);

    // Autoregressive decode loop.
    // budget_hook (when close_token_ids is non-empty) overrides the next
    // sampled token(s) with the close-tag sequence once (n_gen - committed)
    // <= hard_limit. Mirrors qwen35's do_ar_decode. For Gemma4 the close
    // tag is typically `<channel|>` (single token in the gemma4 vocab).
    // forced_close_out, when non-null, is set to true iff the hook injected
    // the close sequence (vs. the model self-closing). See qwen35_backend.h
    // for full rationale.
    bool do_decode(int committed, int n_gen,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io,
                   const BudgetHook & budget_hook = {},
                   bool * forced_close_out = nullptr);

    // DFlash speculative decode loop.
    // When budget_hook is non-null and (n_gen - generated) falls within
    // hard_limit + batch headroom, breaks out and tails via do_decode so
    // the force-close override fires cleanly with KV state intact.
    bool do_spec_decode(int committed, int n_gen,
                        std::vector<int32_t> & out_tokens,
                        const DaemonIO & io,
                        const BudgetHook * budget_hook = nullptr,
                        bool * forced_close_out = nullptr);
};

}  // namespace dflash::common
