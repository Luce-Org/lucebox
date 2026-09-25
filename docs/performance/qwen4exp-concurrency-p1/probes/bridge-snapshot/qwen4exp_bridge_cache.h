#pragma once

#include "qwen4exp_bridge_state.h"
#include "qwen4exp/qwen4exp_cache.h"
#include "qwen4exp/qwen4exp_internal.h"

namespace dflash::common::qwen4exp_bridge {

bool validate_for_target(const Checkpoint & checkpoint,
                         const Qwen4ExpWeights & weights,
                         const Qwen4ExpCache & cache,
                         std::string * error = nullptr);

bool capture(ggml_backend_t backend, const Qwen4ExpWeights & weights,
             const Qwen4ExpCache & cache, uint32_t cut, uint32_t prompt_len,
             Checkpoint & checkpoint, std::string * error = nullptr);

bool restore(ggml_backend_t backend, const Qwen4ExpWeights & weights,
             Qwen4ExpCache & cache, const Checkpoint & checkpoint,
             std::string * error = nullptr);

} // namespace dflash::common::qwen4exp_bridge
