// Backend planning and arch-detecting ModelBackend construction.
//
// BackendArgs is mutable input. prepare_backend() consumes it, resolves model
// and placement facts, normalizes backend policy, and returns an immutable
// BackendPlan. create_backend() accepts only that plan.

#pragma once

#include "backend_args.h"
#include "gguf_inspect.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace dflash::common {

namespace detail {
class BackendPlanBuilder;
}

class BackendPlan;
struct ModelBackend;

// The sole construction entry point. Architecture configs own any path data
// retained by the returned backend, so the plan may have an independent
// lifetime after construction.
std::unique_ptr<ModelBackend> create_backend(const BackendPlan & plan);

// The grouped field carriers are public so consumers can name their read-only
// views. Only the private plan builder can populate the enclosing plan.
class BackendPlan final {
public:
    struct Model {
        std::string path;
        GgufModelInfo metadata;
    };

    struct Placement {
        DevicePlacement target;
        DevicePlacement draft;
        RemoteDraftConfig remote_draft;
        RemoteTargetShardConfig remote_target_shard;
    };

    struct Cache {
        int fa_window = 0;
        bool paged_attention = false;
        int max_concurrency = 1;
        long long kv_pool_tokens = 0;
        int kq_stride_pad = 32;
        int draft_swa_window = 0;
        int draft_ctx_max = 4096;
    };

    struct Speculation {
        std::optional<std::string> draft_path;
        int draft_block_size = 0;
        bool fast_rollback = true;
        bool seq_verify = false;
        bool specla_mode = false;
        int specla_top_k = 4;
        bool specla_top_k_explicit = false;
        bool ddtree_mode = false;
        int ddtree_budget = 22;
        float ddtree_temp = 1.0f;
        bool ddtree_chain_seed = true;
        float ddtree_tau = std::numeric_limits<float>::infinity();
        int verify_width = 0;
        bool use_feature_mirror = false;
    };

    struct Execution {
        int stream_fd = -1;
        int chunk = 512;
    };

    struct DeepSeek4 {
        PrefillAttentionMode prefill_mode = PrefillAttentionMode::Exact;
        int expert_top_k = 0;
        bool fused_decode = false;
        bool fused_verify_f16_kv = false;
    };

    BackendPlan(BackendPlan &&) noexcept = default;
    BackendPlan & operator=(BackendPlan &&) = delete;
    BackendPlan(const BackendPlan &) = delete;
    BackendPlan & operator=(const BackendPlan &) = delete;

    const Model & model() const { return model_; }
    const Placement & placement() const { return placement_; }
    const Cache & cache() const { return cache_; }
    const Speculation & speculation() const { return speculation_; }
    const Execution & execution() const { return execution_; }
    const DeepSeek4 & deepseek4() const { return deepseek4_; }
    const std::string & arch() const { return model_.metadata.arch; }
    const std::vector<std::string> & warnings() const { return warnings_; }

private:
    enum class SpeclaEnvironmentAction {
        Preserve,
        Enable,
        Disable,
    };

    BackendPlan() = default;

    Model model_;
    Placement placement_;
    Cache cache_;
    Speculation speculation_;
    Execution execution_;
    DeepSeek4 deepseek4_;
    std::vector<std::string> warnings_;
    SpeclaEnvironmentAction specla_environment_ =
        SpeclaEnvironmentAction::Preserve;

    friend class detail::BackendPlanBuilder;
    friend std::unique_ptr<ModelBackend> create_backend(
        const BackendPlan & plan);
};

enum class BackendPreparationError {
    InvalidRequest,
    ModelInspection,
    FeatureCompatibility,
};

struct BackendPreparationFailure {
    BackendPreparationError error;
    std::string message;
    std::vector<std::string> warnings;
};

using BackendPreparation =
    std::variant<BackendPlan, BackendPreparationFailure>;

// Consumes the mutable request. Successful preparation performs one GGUF
// inspection and returns the only value accepted by backend construction.
BackendPreparation prepare_backend(
    BackendArgs args,
    BackendAdmissionContext admission = {});

}  // namespace dflash::common
