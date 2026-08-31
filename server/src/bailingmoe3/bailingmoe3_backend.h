#pragma once

#include "qwen35_backend.h"

namespace dflash::common {

// Configuration intentionally exposes only the features the first native
// Ling backend implements. Speculative decode and expert offload can be added
// after the autoregressive path has a logits-equivalent baseline.
struct BailingMoe3Config {
    const char * model_path = nullptr;
    DevicePlacement device;
    int stream_fd = -1;
};

class BailingMoe3Backend final : public Qwen35Backend {
public:
    explicit BailingMoe3Backend(const BailingMoe3Config & cfg);

    void print_ready_banner() const override;
    bool supports_dflash_spec_decode() const override { return false; }
    bool supports_remote_draft() const override { return false; }

protected:
    bool load_target_model(ggml_backend_t backend, TargetWeights & out) override;
};

}  // namespace dflash::common
