#pragma once

#include "common/model_backend.h"
#include "common/moe_hybrid_stream.h"
#include "kimi_k3_internal.h"
#include "placement/placement_config.h"

#include <random>
#include <string>

namespace dflash::common {

struct KimiK3BackendConfig {
    const char * model_path = nullptr;
    DevicePlacement device;
    int stream_fd = -1;
    // Production Kimi uses file-backed routed experts. The resident mode is
    // retained as a deterministic oracle for small architecture fixtures.
    bool stream_routed_experts = true;
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

    int32_t choose_token(const std::vector<float> & logits,
                         const SamplerCfg & sampler,
                         const std::vector<int32_t> & history);

    KimiK3BackendConfig cfg_;
    ggml_backend_t backend_ = nullptr;
    KimiK3Weights weights_;
    KimiK3Cache cache_;
    MoeHybridStreamEngine stream_engine_;
    bool parked_ = false;
    std::mt19937_64 rng_{std::random_device{}()};
};

} // namespace dflash::common
