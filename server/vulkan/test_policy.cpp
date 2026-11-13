#include "../src/server/vulkan_request_policy.h"
#include "model_selection.h"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace luce::common;
using nlohmann::json;
static int failures = 0;
static void check(bool ok, const std::string & name) {
    std::cout << (ok ? "PASS " : "FAIL ") << name << '\n';
    if (!ok) ++failures;
}
static void request_case(const std::string & name, const json & body, bool reject) {
    check(unsupported_lfm_vulkan_request("native_lfm2_vulkan", body) == reject, name+" LFM");
    for (const char * backend : {"cuda", "hip", "native_lfm25_cuda", "native_qwen2_vulkan", ""})
        check(!unsupported_lfm_vulkan_request(backend, body), name+" unchanged backend="+backend);
}
static void sampler_cases() {
    for (const char * key : {"repetition_penalty", "repeat_penalty", "rep_pen"}) {
        for (const json & value : {json(2),json(1),json(0),json(nullptr),json("2")})
            request_case(std::string("reject alias ")+key+"="+value.dump(), {{key,value}}, true);
    }
    request_case("alias precedence cannot bypass guard", {{"repetition_penalty",1},{"rep_pen",2}}, true);
    request_case("plain greedy defaults", json::object(), false);
    request_case("explicit greedy defaults", {{"temperature",0},{"presence_penalty",0},{"frequency_penalty",0},{"stream",false},{"n",1},{"max_tokens",96}}, false);
}
static void optional_cases() {
    request_case("empty tools", {{"tools",json::array()}}, false);
    request_case("disabled logprobs", {{"logprobs",false}}, false);
    request_case("null response format", {{"response_format",nullptr}}, false);
    request_case("combined no-ops", {{"tools",json::array()},{"logprobs",false},{"response_format",nullptr}}, false);
    // Only these three proven no-op representations are relaxed. Do not
    // silently broaden the feature envelope for other optional fields.
    for (const json & body : {
        json{{"tools",json::array({{{"type","function"},{"function",{{"name","test"}}}}})}},
        json{{"tools",nullptr}}, json{{"tools",json::object()}}, json{{"tools",""}},
        json{{"logprobs",true}}, json{{"logprobs",nullptr}}, json{{"logprobs",0}}, json{{"logprobs","false"}},
        json{{"response_format",{{"type","json_object"}}}}, json{{"response_format",json::object()}},
        json{{"response_format",""}}, json{{"response_format",false}},
        json{{"tools",json::array()},{"tool_choice","required"}},
        json{{"logprobs",false},{"top_logprobs",1}},
        json{{"functions",json::array()}}, json{{"function_call","auto"}}, json{{"logit_bias",{{"1",1}}}}
    }) request_case("reject unsupported optional "+body.dump(),body,true);
}
static void existing_cases() {
    check(vulkan_model("qwen2") == VulkanModel::Qwen2,"select Qwen2");
    check(vulkan_model("lfm2moe") == VulkanModel::Lfm2Moe,"select LFM2 MoE");
    bool rejected = false;
    try { vulkan_model("unsupported"); } catch (const std::runtime_error &) { rejected = true; }
    check(rejected,"reject unsupported architecture");
    for (const json & body : {
        json{{"stream",true}}, json{{"temperature",0.7}}, json{{"max_tokens",-1}},
        json{{"max_output_tokens",1025}}, json{{"n",2}}, json{{"logprobs",true}},
        json{{"frequency_penalty",0.5}}, json{{"presence_penalty",0.5}}, json{{"stream","false"}}
    }) request_case("existing rejection "+body.dump(),body,true);
    for (const char * key : {"max_tokens","max_output_tokens","max_completion_tokens"}) {
        for (int value : {1,1024}) request_case(std::string("budget boundary ")+key+"="+std::to_string(value),{{key,value}},false);
        for (const json & value : {json(0),json(1025),json(1.5),json("1"),json(nullptr)})
            request_case(std::string("invalid budget ")+key+"="+value.dump(),{{key,value}},true);
    }
    request_case("plain supported request", {{"temperature",0},{"stream",false},{"max_tokens",96}},false);
}
int main(int argc, char ** argv) {
    try {
        const std::string group = argc > 1 ? argv[1] : "all";
        if (group == "all" || group == "sampler") sampler_cases();
        if (group == "all" || group == "optional") optional_cases();
        if (group == "all" || group == "existing") existing_cases();
        if (group != "all" && group != "sampler" && group != "optional" && group != "existing") throw std::runtime_error("unknown test group");
        return failures ? 1 : 0;
    } catch (const std::exception & e) { std::cerr << "FAIL policy: " << e.what() << '\n'; return 1; }
}
