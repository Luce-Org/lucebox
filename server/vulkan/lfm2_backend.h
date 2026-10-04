#pragma once
#include "common/model_backend.h"
#include <memory>
#include <string>
namespace luce::common {
// Native GGML/Vulkan LFM2 hybrid/MoE target, not a subprocess or llama-server proxy.
std::unique_ptr<ModelBackend> make_lfm2_vulkan(const std::string &path, int context, int chunk, bool flash);
}
