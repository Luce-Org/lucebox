// DeepSeek4Backend — ModelBackend for DeepSeek V4 Flash MLA+MoE models.
//
// Architecture: Multi-head Latent Attention (MLA), KV compression with
// learned compressors, Hierarchical Controller (HC), MoE with hash routing
// (first 3 layers) + top-k routing + shared expert.

#pragma once

#include "common/model_backend.h"
#include "common/sampler.h"
#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/expert_split_state.h"
#include "../common/moe_expert_compute.h"
#include "deepseek4_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

class DeepSeek4Backend : public ModelBackend {
public:
    explicit DeepSeek4Backend(const DeepSeek4BackendConfig & cfg);
    ~DeepSeek4Backend() override;

    DeepSeek4Backend(const DeepSeek4Backend &) = delete;
    DeepSeek4Backend & operator=(const DeepSeek4Backend &) = delete;

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

    void shutdown() override;

private:
    bool run_step_with_runtime_path(const float * embed,
                                    int n_tokens,
                                    int kv_start,
                                    std::vector<float> & out_logits,
                                    const int32_t * token_ids,
                                    bool want_logits,
                                    bool disable_cached_decode,
                                    DeepSeek4StepTelemetry * telemetry);
    void reset_request_state();
    DeepSeek4BackendConfig cfg_;
    ggml_backend_t         backend_      = nullptr;
    ggml_backend_t         snap_backend_ = nullptr;
    DeepSeek4Weights       w_;
    DeepSeek4Cache         cache_;
    bool                   parked_       = false;

    // Sampler
    SamplerCfg             sampler_;
    std::mt19937_64        sampler_rng_{std::random_device{}()};

    // Snapshots
    static constexpr int PREFIX_SLOTS = 64;
    DeepSeek4Snapshot      snapshots_[PREFIX_SLOTS];
    std::vector<float>     hc_state_;
    std::vector<float>     last_logits_;

    // Prefill prompt tokens in chunks, return absolute committed position.
    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io,
                   int kv_offset = 0);

    // Autoregressive decode loop.
    bool do_decode(int committed, int n_gen,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io,
                   const BudgetHook & budget_hook = {},
                   bool * forced_close_out = nullptr);

    bool init_hybrid_model();
    bool compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                          int max_ctx,
                                          MoeHybridPlacement & out,
                                          std::string * err) const;
    bool load_expert_split_placement(const char * hotness_path,
                                     const DeepSeek4Weights & w,
                                     int max_ctx,
                                     MoeHybridPlacement & out,
                                     std::string * err);
    bool build_expert_split_state_from_hotness(const MoeHybridRoutingStats & hotness,
                                               uint64_t expert_budget_bytes,
                                               uint64_t hot_budget_bytes,
                                               int max_hot_per_layer,
                                               const std::vector<ExpertSplitTarget> & configured_targets,
                                               const DeepSeek4Weights & w,
                                               std::string * err);
    std::unique_ptr<MoeExpertCompute> make_expert_split_local_target_compute(
        const ExpertSplitComputeTargetRuntime & split_target,
        std::string * err) const;
    bool init_single_target_expert_runtime(std::string * err);
    bool init_expert_split_runtime(std::string * err);
    void maybe_save_routing_stats();

    std::shared_ptr<MoeHybridStorage> moe_hybrid_;
    MoeHybridPlacement                moe_placement_;
    ExpertSplitStateComponents        expert_split_state_;
    MoeExpertComputeRuntime           expert_runtime_;
    MoeMultiTargetExpertRuntime       expert_split_runtime_;
    std::vector<uint64_t>             layer_expert_bytes_;
    uint64_t                          expert_split_total_budget_bytes_ = 0;
    uint64_t                          expert_split_hot_budget_bytes_ = 0;
    int                               expert_split_cache_slots_ = 0;
    std::shared_ptr<MoeHybridRoutingStats> routing_stats_;
    std::string                       routing_stats_out_path_;
};

}  // namespace dflash::common
