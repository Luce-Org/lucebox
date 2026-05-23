#include "qwen35moe_backend.h"

namespace dflash::common {

Qwen35MoeBackend::Qwen35MoeBackend(const Qwen35Config & cfg)
    : Qwen35Backend(cfg) {}

bool Qwen35MoeBackend::load_target_model(ggml_backend_t backend, TargetWeights & out) {
    return load_target_gguf(cfg_.target_path, backend, out);
}

}  // namespace dflash::common
