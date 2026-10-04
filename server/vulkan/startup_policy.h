#pragma once
#include "model_selection.h"
#include "gguf.h"
#include <stdexcept>
#include <string>

namespace luce::common {
inline void validate_vulkan_context(VulkanModel model, int context) {
    if (context < 128 || context > 32768)
        throw std::runtime_error("option out of bounds");
    if (model == VulkanModel::Lfm2Moe && context > 4096)
        throw std::runtime_error("LFM Vulkan context exceeds validated limit of 4096");
}

struct VulkanChatTemplate {
    std::string source;
    std::string path;
};
inline VulkanChatTemplate vulkan_chat_template(const gguf_context * meta, VulkanModel model) {
    const auto ti = gguf_find_key(meta, "tokenizer.chat_template");
    if (ti >= 0 && gguf_get_kv_type(meta, ti) == GGUF_TYPE_STRING) {
        const std::string source = gguf_get_val_str(meta, ti);
        if (model == VulkanModel::Lfm2Moe && source.empty())
            throw std::runtime_error("empty GGUF chat template");
        return {source, "GGUF:tokenizer.chat_template"};
    }
    if (model == VulkanModel::Lfm2Moe) throw std::runtime_error("missing GGUF chat template");
    return {};
}
} // namespace luce::common
