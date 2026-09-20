// Qwen4ExpBackend — ModelBackend for Qwen3.8-Flash-Next (arch `qwen4exp`).
//
// Phase-1 scaffold: the loader and weight layout are real, and the backend
// registers so the architecture is dispatchable. The forward graph (HC, PLE,
// 512-expert MoE, hybrid DeltaNet/full attention) lands next; generate()
// currently fails with a clear BackendSpecific error rather than pretending.

#pragma once

#include "common/model_backend.h"
#include "placement/placement_config.h"

#include "qwen4exp_internal.h"
#include "qwen4exp_cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <string>

namespace dflash::common {

struct Qwen4ExpBackendConfig {
    std::string     model_path;
    DevicePlacement device;
    int             stream_fd = -1;
    int             chunk     = 2048;
};

class Qwen4ExpBackend final : public ModelBackend {
public:
    explicit Qwen4ExpBackend(Qwen4ExpBackendConfig cfg);
    ~Qwen4ExpBackend() override;

    Qwen4ExpBackend(const Qwen4ExpBackend &) = delete;
    Qwen4ExpBackend & operator=(const Qwen4ExpBackend &) = delete;

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

private:
    Qwen4ExpBackendConfig cfg_;
    ggml_backend_t        backend_ = nullptr;
    Qwen4ExpWeights       weights_;
    Qwen4ExpCache         cache_;
    bool                  parked_  = false;
};

}  // namespace dflash::common
