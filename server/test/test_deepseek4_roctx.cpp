#include "deepseek4/deepseek4_roctx.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

int failures = 0;
std::vector<std::string> events;
int push_result = 0;
int loader_open_calls = 0;
int loader_close_calls = 0;
int loader_diagnostic_calls = 0;
std::string loader_diagnostic;

#define CHECK(condition) do { \
    if (!(condition)) { \
        ++failures; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    } \
} while (0)

int record_push(const char * message) {
    events.emplace_back(std::string("push:") + message);
    return push_result;
}

int record_pop() {
    events.emplace_back("pop");
    return 0;
}

void * loader_open_fail() {
    ++loader_open_calls;
    return nullptr;
}

void * loader_open_success() {
    ++loader_open_calls;
    return reinterpret_cast<void *>(1);
}

DeepSeek4RoctxPush loader_find_push(void *) {
    return record_push;
}

DeepSeek4RoctxPush loader_find_push_missing(void *) {
    return nullptr;
}

DeepSeek4RoctxPop loader_find_pop(void *) {
    return record_pop;
}

void loader_close(void *) {
    ++loader_close_calls;
}

void loader_diagnose(const char * message) {
    ++loader_diagnostic_calls;
    loader_diagnostic = message;
}

void reset_loader_state() {
    loader_open_calls = 0;
    loader_close_calls = 0;
    loader_diagnostic_calls = 0;
    loader_diagnostic.clear();
}

bool return_failure_with_range() {
    const DeepSeek4RoctxRange range(
        "ds4.layer_range", {"verify", 4, 0, 43, 0}, true,
        {record_push, record_pop});
    return false;
}

void test_env_policy() {
    CHECK(!deepseek4_roctx_env_enabled(nullptr));
    CHECK(!deepseek4_roctx_env_enabled(""));
    CHECK(!deepseek4_roctx_env_enabled("0"));
    CHECK(!deepseek4_roctx_env_enabled("false-ish"));
    CHECK(deepseek4_roctx_env_enabled("1"));
    CHECK(deepseek4_roctx_env_enabled("true"));
    CHECK(deepseek4_roctx_env_enabled("YES"));
    CHECK(deepseek4_roctx_env_enabled("On"));
}

void test_exact_prefill_phase_label() {
    CHECK(std::string(deepseek4_roctx_layer_mode(false, 1, "exact")) ==
          "unspecified");
    {
        const DeepSeek4RoctxPhaseScope prefill("exact");
        CHECK(std::string(deepseek4_roctx_layer_mode(false, 1, "exact")) ==
              "exact");
        CHECK(std::string(deepseek4_roctx_layer_mode(true, 1, "exact")) ==
              "exact");
    }
    CHECK(std::string(deepseek4_roctx_layer_mode(false, 1, "exact")) ==
          "unspecified");
}

void test_disabled_loader_is_silent_and_unopened() {
    reset_loader_state();
    const DeepSeek4RoctxCallbacks callbacks = deepseek4_roctx_load_callbacks(
        false, {loader_open_success, loader_find_push, loader_find_pop,
                loader_close, loader_diagnose});
    CHECK(!callbacks.push && !callbacks.pop);
    CHECK(loader_open_calls == 0);
    CHECK(loader_close_calls == 0);
    CHECK(loader_diagnostic_calls == 0);
}

void test_missing_library_is_diagnosed() {
    reset_loader_state();
    const DeepSeek4RoctxCallbacks callbacks = deepseek4_roctx_load_callbacks(
        true, {loader_open_fail, loader_find_push, loader_find_pop,
               loader_close, loader_diagnose});
    CHECK(!callbacks.push && !callbacks.pop);
    CHECK(loader_open_calls == 1);
    CHECK(loader_close_calls == 0);
    CHECK(loader_diagnostic_calls == 1);
    CHECK(loader_diagnostic.find("library could not be loaded") !=
          std::string::npos);
}

void test_missing_symbol_closes_and_is_diagnosed() {
    reset_loader_state();
    const DeepSeek4RoctxCallbacks callbacks = deepseek4_roctx_load_callbacks(
        true, {loader_open_success, loader_find_push_missing, loader_find_pop,
               loader_close, loader_diagnose});
    CHECK(!callbacks.push && !callbacks.pop);
    CHECK(loader_open_calls == 1);
    CHECK(loader_close_calls == 1);
    CHECK(loader_diagnostic_calls == 1);
    CHECK(loader_diagnostic.find("range symbols are missing") !=
          std::string::npos);
}

void test_disabled_is_silent() {
    events.clear();
    {
        const DeepSeek4RoctxRange range(
            "ds4.prefill", {"exact", 8, 0, 43, 0}, false,
            {record_push, record_pop});
    }
    CHECK(events.empty());
}

void test_metadata_and_balance() {
    events.clear();
    push_result = 0;
    {
        const DeepSeek4RoctxRange range(
            "ds4.layer_range", {"exact", 4, 2, 17, 1}, true,
            {record_push, record_pop});
        CHECK(events.size() == 1);
        CHECK(events[0] ==
              "push:ds4.layer_range mode=exact tokens=4 layer_begin=2 layer_end=17 device=1");
    }
    CHECK(events.size() == 2);
    CHECK(events[1] == "pop");
}

void test_failed_push_is_not_popped() {
    events.clear();
    push_result = -1;
    {
        const DeepSeek4RoctxRange range(
            "ds4.spec_decode", {"reference_exact", 32, -1, -1, 0}, true,
            {record_push, record_pop});
    }
    CHECK(events.size() == 1);
    CHECK(events[0] ==
          "push:ds4.spec_decode mode=reference_exact tokens=32 device=0");
    push_result = 0;
}

void test_early_failure_balances_range() {
    events.clear();
    CHECK(!return_failure_with_range());
    CHECK(events.size() == 2);
    CHECK(events[0] ==
          "push:ds4.layer_range mode=verify tokens=4 layer_begin=0 layer_end=43 device=0");
    CHECK(events[1] == "pop");
}

void test_missing_callback_is_silent() {
    events.clear();
    {
        const DeepSeek4RoctxRange no_push(
            "ds4.prefill", {}, true, {nullptr, record_pop});
        const DeepSeek4RoctxRange no_pop(
            "ds4.prefill", {}, true, {record_push, nullptr});
    }
    CHECK(events.empty());
}

} // namespace

int main() {
    test_env_policy();
    test_exact_prefill_phase_label();
    test_disabled_loader_is_silent_and_unopened();
    test_missing_library_is_diagnosed();
    test_missing_symbol_closes_and_is_diagnosed();
    test_disabled_is_silent();
    test_metadata_and_balance();
    test_failed_push_is_not_popped();
    test_early_failure_balances_range();
    test_missing_callback_is_silent();
    if (failures) {
        std::fprintf(stderr, "FAILED: %d assertion(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "deepseek4_roctx: ok\n");
    return 0;
}
