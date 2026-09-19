#include "deepseek4/deepseek4_roctx.h"
#include "common/io_utils.h"
#include "common/target_shard_ipc_daemon.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace dflash::common;

#if !defined(_WIN32) && defined(DFLASH27B_BACKEND_HIP)
std::vector<std::string> interposed_events;

extern "C" int roctxRangePushA(const char * message) {
    interposed_events.emplace_back(std::string("push:") + message);
    return 0;
}

extern "C" int roctxRangePop() {
    interposed_events.emplace_back("pop");
    return 0;
}
#endif

static_assert(std::is_same_v<std::underlying_type_t<InferencePhase>, int32_t>);
static_assert(std::is_same_v<
    decltype(TargetShardDaemonForwardRequest{}.semantic_phase), InferencePhase>);

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
        "ds4.layer_range", {InferencePhase::Verify, 4, 0, 43, 0}, true,
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
    CHECK(deepseek4_roctx_layer_phase(
              false, 1, InferencePhase::Exact) == InferencePhase::Unspecified);
    {
        const DeepSeek4RoctxPhaseScope prefill(InferencePhase::Exact);
        CHECK(deepseek4_roctx_layer_phase(
                  false, 1, InferencePhase::Exact) == InferencePhase::Exact);
        {
            const DeepSeek4RoctxPhaseScope decode(InferencePhase::Decode);
            CHECK(deepseek4_roctx_current_phase() == InferencePhase::Decode);
        }
        CHECK(deepseek4_roctx_current_phase() == InferencePhase::Exact);
        CHECK(deepseek4_roctx_layer_phase(
                  true, 1, InferencePhase::Exact) == InferencePhase::Exact);
    }
    CHECK(deepseek4_roctx_current_phase() == InferencePhase::Unspecified);
}

void test_phase_mapping_and_wire_roundtrip() {
    const InferencePhase phases[] = {
        InferencePhase::Unspecified,
        InferencePhase::Exact,
        InferencePhase::Dense,
        InferencePhase::Sparse,
        InferencePhase::Decode,
        InferencePhase::Verify,
        InferencePhase::ReferenceExact,
        InferencePhase::Sequential,
        InferencePhase::Batched,
    };
    const char * names[] = {
        "unspecified", "exact", "dense", "sparse", "decode", "verify",
        "reference_exact", "sequential", "batched",
    };
    for (size_t i = 0; i < sizeof(phases) / sizeof(phases[0]); ++i) {
        InferencePhase decoded = InferencePhase::Unspecified;
        CHECK(inference_phase_from_wire_value(
            inference_phase_wire_value(phases[i]), decoded));
        CHECK(decoded == phases[i]);
        CHECK(std::string(deepseek4_roctx_phase_name(decoded)) == names[i]);
    }
    InferencePhase decoded = InferencePhase::Exact;
    CHECK(!inference_phase_from_wire_value(-1, decoded));
    CHECK(!inference_phase_from_wire_value(9, decoded));

    TargetShardDaemonForwardRequest remote_request;
    remote_request.semantic_phase = InferencePhase::Decode;
    {
        const DeepSeek4RoctxPhaseScope remote_phase(remote_request.semantic_phase);
        CHECK(deepseek4_roctx_current_phase() == InferencePhase::Decode);
    }
    CHECK(deepseek4_roctx_current_phase() == InferencePhase::Unspecified);
}

#if !defined(_WIN32)
void test_malformed_phase_does_not_desynchronize_pipe(const char * malformed_phase) {
    int payload_fds[2] = {-1, -1};
    int response_fds[2] = {-1, -1};
    CHECK(pipe(payload_fds) == 0);
    CHECK(pipe(response_fds) == 0);
    if (payload_fds[0] < 0 || response_fds[0] < 0) return;

    const float malformed_activation[] = {1.0f, 2.0f};
    const int32_t malformed_token = 11;
    const float valid_activation[] = {3.0f, 4.0f};
    const int32_t valid_token = 22;
    CHECK(write_exact_fd(payload_fds[1], malformed_activation,
                         sizeof(malformed_activation)));
    CHECK(write_exact_fd(payload_fds[1], &malformed_token,
                         sizeof(malformed_token)));
    CHECK(write_exact_fd(payload_fds[1], valid_activation,
                         sizeof(valid_activation)));
    CHECK(write_exact_fd(payload_fds[1], &valid_token, sizeof(valid_token)));
    close(payload_fds[1]);

    std::ostringstream command_text;
    command_text << "forward_pipe 0 1 0 0 8 1 1 1" << malformed_phase << '\n';
    command_text << "forward_pipe 1 1 0 0 8 1 1 1 "
                 << inference_phase_wire_value(InferencePhase::Decode) << '\n';
    command_text << "quit\n";
    std::istringstream commands(command_text.str());

    int callback_calls = 0;
    TargetShardDaemonCallbacks callbacks;
    callbacks.forward = [&](const TargetShardDaemonForwardRequest & request,
                            TargetShardDaemonForwardResponse & response) {
        ++callback_calls;
        CHECK(request.semantic_phase == InferencePhase::Decode);
        CHECK(request.base_pos == 1);
        CHECK(request.boundary_activation &&
              *request.boundary_activation == std::vector<float>({3.0f, 4.0f}));
        CHECK(request.token_ids &&
              *request.token_ids == std::vector<int32_t>({22}));
        response.last_tok = 77;
        return true;
    };

    std::streambuf * previous_input = std::cin.rdbuf(commands.rdbuf());
    std::cin.clear();
    const int rc = run_target_shard_ipc_daemon_loop(
        2, 8, response_fds[1], payload_fds[0], -1, 0, std::move(callbacks));
    std::cin.rdbuf(previous_input);
    std::cin.clear();
    close(payload_fds[0]);
    close(response_fds[1]);

    int32_t responses[4] = {};
    CHECK(read_exact_fd(response_fds[0], responses, sizeof(responses)));
    close(response_fds[0]);
    CHECK(rc == 0);
    CHECK(callback_calls == 1);
    CHECK(responses[0] == 0);
    CHECK(responses[1] == -1);
    CHECK(responses[2] == 0);
    CHECK(responses[3] == 77);
}

void test_malformed_phases_do_not_desynchronize_pipe() {
    test_malformed_phase_does_not_desynchronize_pipe("");
    test_malformed_phase_does_not_desynchronize_pipe(" 4 extra");
    test_malformed_phase_does_not_desynchronize_pipe(" 99");
}
#endif

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

#if !defined(_WIN32) && defined(DFLASH27B_BACKEND_HIP)
constexpr const char * roctx_interposition_child_arg =
    "--roctx-interposition-child";

bool run_runtime_loader_interposition_child() {
    interposed_events.clear();
    {
        const DeepSeek4RoctxRange range(
            "ds4.interposed", {InferencePhase::Decode, 1, 0, 43, 0});
    }
    return interposed_events == std::vector<std::string>({
        "push:ds4.interposed mode=decode tokens=1 layer_begin=0 layer_end=43 device=0",
        "pop",
    });
}

void test_runtime_loader_prefers_interposed_roctx_symbols(
        const char * executable) {
    // Re-exec so the process-scoped callback cache and environment changes
    // used by this integration check cannot leak into the parent test suite.
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child < 0) return;
    if (child == 0) {
        if (setenv("DFLASH_DS4_ROCTX", "1", 1) != 0) _exit(125);
        execlp(executable, executable, roctx_interposition_child_arg,
               static_cast<char *>(nullptr));
        _exit(126);
    }

    int status = 0;
    pid_t waited = 0;
    constexpr int wait_attempts = 500;
    constexpr unsigned int wait_interval_us = 10000;
    for (int attempt = 0; attempt < wait_attempts; ++attempt) {
        waited = waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        if (waited < 0 && errno != EINTR) break;
        usleep(wait_interval_us);
    }

    if (waited != child) {
        ++failures;
        if (waited == 0) {
            std::fprintf(stderr,
                         "FAIL %s:%d: ROCTX interposition child timed out\n",
                         __FILE__, __LINE__);
        } else {
            std::fprintf(stderr,
                         "FAIL %s:%d: waitpid failed for ROCTX interposition child\n",
                         __FILE__, __LINE__);
        }
        if (kill(child, SIGKILL) == 0) {
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
        }
        return;
    }
    if (WIFSIGNALED(status)) {
        ++failures;
        std::fprintf(stderr,
                     "FAIL %s:%d: ROCTX interposition child terminated by signal %d\n",
                     __FILE__, __LINE__, WTERMSIG(status));
        return;
    }
    CHECK(WIFEXITED(status));
    if (WIFEXITED(status)) CHECK(WEXITSTATUS(status) == 0);
}
#endif

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
            "ds4.prefill", {InferencePhase::Exact, 8, 0, 43, 0}, false,
            {record_push, record_pop});
    }
    CHECK(events.empty());
}

void test_metadata_and_balance() {
    events.clear();
    push_result = 0;
    {
        const DeepSeek4RoctxRange range(
            "ds4.layer_range", {InferencePhase::Exact, 4, 2, 17, 1}, true,
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
            "ds4.spec_decode", {InferencePhase::ReferenceExact, 32, -1, -1, 0}, true,
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

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;
#if !defined(_WIN32) && defined(DFLASH27B_BACKEND_HIP)
    if (argc == 2 &&
        std::string(argv[1]) == roctx_interposition_child_arg) {
        return run_runtime_loader_interposition_child() ? 0 : 1;
    }
#endif
    test_env_policy();
    test_exact_prefill_phase_label();
    test_phase_mapping_and_wire_roundtrip();
#if !defined(_WIN32)
    test_malformed_phases_do_not_desynchronize_pipe();
#endif
#if !defined(_WIN32) && defined(DFLASH27B_BACKEND_HIP)
    test_runtime_loader_prefers_interposed_roctx_symbols(argv[0]);
#endif
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
