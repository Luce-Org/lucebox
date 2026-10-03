#pragma once
#include "common/model_backend.h"
#include <memory>
namespace luce::common {
// Native GGML/Vulkan Qwen2 target, not a subprocess or llama-server proxy.
std::unique_ptr<ModelBackend> make_qwen2_vulkan(const std::string &path, int context, int chunk, bool flash);
}
