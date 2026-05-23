// Thin qwen35moe backend wrapper over the shared qwen35-family runtime.

#pragma once

#include "qwen35_backend.h"

namespace dflash::common {

class Qwen35MoeBackend : public Qwen35Backend {
public:
    explicit Qwen35MoeBackend(const Qwen35Config & cfg);
    ~Qwen35MoeBackend() override = default;

protected:
    bool load_target_model(ggml_backend_t backend, TargetWeights & out) override;
};

}  // namespace dflash::common
