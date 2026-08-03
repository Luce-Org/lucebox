#pragma once

#include "common/model_backend.h"
#include "common/moe_hybrid_routing_stats.h"
#include "common/moe_hybrid_stream.h"
#include "common/moe_storage_policy.h"
#include "kimi_k3_internal.h"
#include "placement/placement_config.h"

#include <random>
#include <memory>
#include <string>

namespace dflash::common {

struct KimiK3BackendConfig {
    const char * model_path = nullptr;
    DevicePlacement device;
    int stream_fd = -1;
    // -1 resolves DFLASH_MOE_TP_GPU and otherwise keeps the primary device
    // index. DFLASH_MOE_TP_BACKEND may select a different in-process runtime
    // (for example CUDA beside a HIP primary). A different backend or device
    // becomes the secondary capacity owner; routed work is partitioned while
    // dense KDA/MLA, recurrent state, and sampling remain primary-owned.
    int expert_gpu = -1;
    // Auto uses Kimi's capacity-safe file-backed routed experts. Resident is
    // retained as a deterministic oracle for small architecture fixtures.
    MoeStoragePolicy moe_storage = MoeStoragePolicy::Auto;
};

class KimiK3Backend final : public ModelBackend {
public:
    explicit KimiK3Backend(const KimiK3BackendConfig & cfg);
    ~KimiK3Backend() override;

    bool init();

    void print_ready_banner() const override;
    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;
    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int snapshot_cur_pos(int slot) const override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override {}
    void shutdown() override;

private:
    bool init_streaming();
    void release_expert_backend();
    void maybe_save_routing_stats();

    int32_t choose_token(const std::vector<float> & logits,
                         const SamplerCfg & sampler,
                         const std::vector<int32_t> & history);

    KimiK3BackendConfig cfg_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t expert_backend_ = nullptr;
    PlacementBackend expert_backend_kind_ = PlacementBackend::Auto;
    int expert_gpu_ = -1;
    KimiK3Weights weights_;
    KimiK3Cache cache_;
    MoeHybridStreamEngine stream_engine_;
    MoeHybridStreamEngine secondary_stream_engine_;
    MoeStreamDualOwnerExecutor dual_stream_executor_;
    MoeHybridPlacement stream_placement_;
    MoeStreamDualOwnerPolicy stream_owner_policy_;
    std::shared_ptr<MoeHybridRoutingStats> routing_stats_;
    std::string routing_stats_out_path_;
    bool parked_ = false;
    std::mt19937_64 rng_{std::random_device{}()};
};

} // namespace dflash::common
