#pragma once
#include <stdexcept>
#include <string>
namespace luce::common {
enum class VulkanModel { Qwen2, Lfm2Moe };
inline VulkanModel vulkan_model(const std::string & arch) {
    if (arch == "qwen2") return VulkanModel::Qwen2;
    if (arch == "lfm2moe") return VulkanModel::Lfm2Moe;
    throw std::runtime_error("unsupported Vulkan model architecture: " + arch);
}
}
