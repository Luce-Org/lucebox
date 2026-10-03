#include "../src/server/vulkan_request_policy.h"
#include "model_selection.h"
#include <iostream>
#include <stdexcept>
using namespace luce::common;
static void check(bool ok) { if (!ok) throw std::runtime_error("policy regression"); }
int main() {
    check(vulkan_model("qwen2") == VulkanModel::Qwen2);
    check(vulkan_model("lfm2moe") == VulkanModel::Lfm2Moe);
    bool rejected = false;
    try { vulkan_model("unsupported"); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected);
    const nlohmann::json requests[] = {
        {{"stream", true}}, {{"temperature", 0.7}}, {{"tools", nlohmann::json::array()}},
        {{"max_tokens", -1}}, {{"max_output_tokens", 1025}}, {{"n", 2}},
        {{"logprobs", true}}, {{"frequency_penalty", 0.5}}, {{"stream", "false"}}
    };
    for (const auto & body : requests) {
        check(unsupported_lfm_vulkan_request("native_lfm2_vulkan", body));
        // Architecture is deliberately absent: CUDA LFM and all other runtimes
        // continue through the original server request parsing path.
        for (const char * backend : {"cuda", "hip", "native_lfm25_cuda", "native_qwen2_vulkan", ""})
            check(!unsupported_lfm_vulkan_request(backend, body));
    }
    check(!unsupported_lfm_vulkan_request("native_lfm2_vulkan", {{"temperature", 0}, {"stream", false}, {"max_tokens", 96}}));
    std::cout << "PASS additive Qwen/LFM selection and backend-scoped request limits\n";
}
