#pragma once
#include "model_selection.h"
#include "gguf.h"
#include <stdexcept>
#include <string>

namespace luce::common {
inline void validate_vulkan_context(VulkanModel model, int context) {
    (void)model;
    if (context < 128 || context > 32768)
        throw std::runtime_error("option out of bounds");
}

struct VulkanChatTemplate {
    std::string source;
    std::string path;
};
inline VulkanChatTemplate vulkan_chat_template(const gguf_context * meta, VulkanModel model) {
    const auto ti = gguf_find_key(meta, "tokenizer.chat_template");
    if (ti >= 0 && gguf_get_kv_type(meta, ti) == GGUF_TYPE_STRING)
        return {gguf_get_val_str(meta, ti), "GGUF:tokenizer.chat_template"};
    if (model == VulkanModel::Lfm2Moe) throw std::runtime_error("missing GGUF chat template");
    return {};
}
} // namespace luce::common
