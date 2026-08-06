// DeepSeek4Backend — ModelBackend for DeepSeek V4 Flash MLA+MoE models.
//
// Architecture: Multi-head Latent Attention (MLA), KV compression with
// learned compressors, Hierarchical Controller (HC), MoE with hash routing
// (first 3 layers) + top-k routing + shared expert.

#pragma once

#include "common/model_backend.h"
#include "common/moe_expert_compute.h"
#include "common/sampler.h"
#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_stream.h"
#include "deepseek4_internal.h"
#include "deepseek4_dspark.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

// Bounds the sparse heterogeneous prefill arena once accumulated attention
// context dominates its memory footprint. Decode batching is unaffected.
int deepseek4_hybrid_prefill_chunk_tokens(
    int requested_chunk,
    int context_end,
    int current_cap = 0);

class DeepSeek4Backend : public ModelBackend {
public:
    explicit DeepSeek4Backend(const DeepSeek4BackendConfig & cfg);
    ~DeepSeek4Backend() override;

    DeepSeek4Backend(const DeepSeek4Backend &) = delete;
    DeepSeek4Backend & operator=(const DeepSeek4Backend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
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

    void shutdown() override;

    const MoeHybridRoutingStats * get_routing_stats() const override {
        return routing_stats_.get();
    }

private:
    DeepSeek4BackendConfig cfg_;
    ggml_backend_t         backend_      = nullptr;
    ggml_backend_t         snap_backend_ = nullptr;
    ggml_backend_t         expert_backend_ = nullptr;
    DeepSeek4Weights       w_;
    DeepSeek4Cache         cache_;
    bool                   parked_       = false;

    // Sampler
    SamplerCfg             sampler_;
    std::mt19937_64        sampler_rng_{std::random_device{}()};

    // Snapshots
    static constexpr int PREFIX_SLOTS = 64;
    struct SnapshotAux {
        std::vector<float> last_logits;
        std::vector<float> spec_feat_window;
        bool used = false;
    };
    DeepSeek4Snapshot      snapshots_[PREFIX_SLOTS];
    SnapshotAux            snapshot_aux_[PREFIX_SLOTS];
    std::vector<float>     last_logits_;
    // Absolute cache position represented by last_logits_. A snapshot is
    // safe only when this matches cache_.cur_pos.
    int                    last_logits_pos_ = -1;

    // DSpark speculative decode (opt-in: DFLASH_DS4_SPEC=1 + DFLASH_DS4_DRAFT=<gguf>).
    bool                           spec_enabled_ = false;
    bool                           spec_drafter_parked_ = false;
    std::string                    spec_draft_path_;
    ggml_backend_t                 spec_backend_ = nullptr;
    std::unique_ptr<DSparkDrafter> spec_drafter_;
    std::vector<float>             spec_feat_window_;
    // Once a long prompt selects the fragmentation-safe prefill shape, retain
    // it for later requests so the HIP arenas never switch back under load.
    int                            hybrid_prefill_chunk_cap_ = 0;

    bool load_spec_drafter();
    void release_spec_drafter(bool mark_parked);
    void keep_spec_feature_tail(std::vector<float> & features,
                                size_t max_rows) const;
    // True when a wide prefill path returns per-token DSpark features and the
    // caller can retain only the requested capture window without splitting.
    static bool supports_batched_spec_feature_capture(
        bool hybrid,
        PrefillAttentionMode mode,
        int n_tokens);
    // Limit a prefill batch to a region with a uniform DSpark capture policy.
    // Wide GPU paths can capture a subrange without splitting the final
    // feature window; other paths still stop exactly at capture boundaries.
    static int capture_safe_prefill_tokens(int token_offset,
                                           int requested_tokens,
                                           int final_capture_from,
                                           bool batch_final_capture,
                                           bool snapshot_pending,
                                           int snapshot_capture_from,
                                           int snapshot_capture_to);

    // Prefill prompt tokens in chunks, return absolute committed position.
    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io,
                   int kv_offset = 0, int snap_slot = -1, int snap_pos = -1);

    // Generate after either a fresh prefill or a restored prefix. kv_offset is
    // the number of prompt tokens already represented by cache_ and the
    // auxiliary logits/speculative state.
    GenerateResult generate_from_state(const GenerateRequest & req,
                                       const DaemonIO & io,
                                       int kv_offset);
    bool snapshot_restore(int slot);

    // Autoregressive decode loop.
    bool do_decode(int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io,
                   const BudgetHook & budget_hook = {},
                   bool * forced_close_out = nullptr);

    bool load_model();
    bool init_hybrid_model();
    bool requires_monolithic_model() const;
    bool validate_prefill_mode() const;
    bool init_moe_tensor_parallel();
    bool compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                          int max_ctx,
                                          MoeHybridPlacement & out,
                                          MoeHybridPlacement * decode_out,
                                          std::string * err) const;
    void maybe_save_routing_stats();

    std::shared_ptr<MoeHybridStorage> moe_hybrid_;
    MoeHybridPlacement                moe_placement_;
    MoeHybridPlacement                moe_decode_placement_;
    MoeHybridStreamEngine             stream_engine_;
    MoeExpertComputeRuntime            expert_runtime_;
    std::shared_ptr<MoeHybridRoutingStats> routing_stats_;
    std::string                       routing_stats_out_path_;
};

}  // namespace dflash::common
