#include <startup_policy.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
using namespace luce::common;
static int failures = 0;
static void check(bool ok, const std::string & name) {
    std::cout << (ok ? "PASS " : "FAIL ") << name << '\n';
    if (!ok) ++failures;
}
template<class F> static bool rejected(F f) {
    try { f(); } catch (const std::runtime_error &) { return true; }
    return false;
}
static void context_tests() {
    for (auto model : {VulkanModel::Lfm2Moe, VulkanModel::Qwen2}) {
        const std::string name = model == VulkanModel::Lfm2Moe ? "LFM" : "Qwen";
        for (int n : {128, 4096})
            check(!rejected([&]{validate_vulkan_context(model,n);}), name+" accepts "+std::to_string(n));
        for (int n : {-1, 0, 127, 32769})
            check(rejected([&]{validate_vulkan_context(model,n);}), name+" rejects "+std::to_string(n));
        for (int n : {4097, 32768})
            check(rejected([&]{validate_vulkan_context(model,n);}) == (model == VulkanModel::Lfm2Moe),
                  name+" upper envelope "+std::to_string(n));
    }
}
static void template_tests() {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free);
    if (!meta) throw std::runtime_error("test GGUF allocation failed");
    for (auto model : {VulkanModel::Lfm2Moe, VulkanModel::Qwen2}) {
        const bool lfm = model == VulkanModel::Lfm2Moe;
        const std::string name = lfm ? "LFM" : "Qwen";
        gguf_remove_key(meta.get(), "tokenizer.chat_template");
        check(rejected([&]{vulkan_chat_template(meta.get(),model);}) == lfm, name+" missing template");
        gguf_set_val_u32(meta.get(),"tokenizer.chat_template",1);
        check(rejected([&]{vulkan_chat_template(meta.get(),model);}) == lfm, name+" non-string template");
        gguf_set_val_str(meta.get(),"tokenizer.chat_template","");
        check(rejected([&]{vulkan_chat_template(meta.get(),model);}) == lfm, name+" empty template");
        const std::string text = "{% for m in messages %}{{ m.content }}{% endfor %}";
        gguf_set_val_str(meta.get(),"tokenizer.chat_template",text.c_str());
        const auto result = vulkan_chat_template(meta.get(),model);
        check(result.source == text && result.path == "GGUF:tokenizer.chat_template", name+" nonempty template preserved");
    }
}
int main(int argc, char ** argv) {
    try {
        const std::string group = argc > 1 ? argv[1] : "all";
        if (group == "all" || group == "context") context_tests();
        if (group == "all" || group == "template") template_tests();
        if (group != "all" && group != "context" && group != "template") throw std::runtime_error("unknown test group");
        return failures ? 1 : 0;
    } catch (const std::exception & e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
