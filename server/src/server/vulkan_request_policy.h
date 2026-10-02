#pragma once
#include <nlohmann/json.hpp>
#include <string>

namespace luce::common {
// This is a runtime capability limit, not a property of the LFM architecture.
inline bool limited_lfm_vulkan(const std::string & runtime_backend) {
    return runtime_backend == "native_lfm2_vulkan";
}
inline bool unsupported_lfm_vulkan_request(const std::string & runtime_backend,
                                           const nlohmann::json & body) {
    if (!limited_lfm_vulkan(runtime_backend)) return false;
    if (body.contains("stream") && (!body["stream"].is_boolean() || body["stream"].get<bool>())) return true;
    if (body.contains("n") && (!body["n"].is_number_integer() || body["n"] != 1)) return true;
    for (const char * key : {"max_tokens", "max_output_tokens", "max_completion_tokens"}) {
        if (body.contains(key) && (!body[key].is_number_integer() || body[key] < 1 || body[key] > 1024)) return true;
    }
    for (const char * key : {"temperature", "presence_penalty", "frequency_penalty"}) {
        if (body.contains(key) && (!body[key].is_number() || body[key] != 0)) return true;
    }
    for (const char * key : {"tools", "tool_choice", "functions", "function_call", "response_format", "logprobs", "top_logprobs", "logit_bias", "repetition_penalty", "repeat_penalty"}) {
        if (body.contains(key)) return true;
    }
    return false;
}
} // namespace luce::common
