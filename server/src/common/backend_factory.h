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

namespace luce::common {

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
        std::optional<std::string> mmproj_path;
        std::optional<DevicePlacement> mmproj_device;
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
        long long kv_pool_tokens = 0;
        ggml_type cache_type_k = GGML_TYPE_COUNT;
        ggml_type cache_type_v = GGML_TYPE_COUNT;
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
        bool ddtree_mode = false;
        int ddtree_budget = 22;
        float ddtree_temp = 1.0f;
        bool ddtree_chain_seed = true;
        float ddtree_tau = std::numeric_limits<float>::infinity();
        int verify_width = 0;
        bool use_feature_mirror = false;
    };

    // How the backend runs: daemon I/O, prefill chunking, decode-slot
    // scheduling, and the deepseek4 decode knobs (inert elsewhere — the gate
    // rejects them on other architectures).
    struct Execution {
        int stream_fd = -1;
        int chunk = 512;
        int max_concurrency = 1;

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
    const std::string & arch() const { return model_.metadata.arch; }
    const std::vector<std::string> & warnings() const { return warnings_; }

private:
    BackendPlan() = default;

    Model model_;
    Placement placement_;
    Cache cache_;
    Speculation speculation_;
    Execution execution_;
    std::vector<std::string> warnings_;

    friend class detail::BackendPlanBuilder;
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

}  // namespace luce::common
