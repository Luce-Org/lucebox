#include "CppUnitTestFramework.hpp"
#include "server/tool_speculation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <signal.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <features.h>
#    include <sched.h>
#  endif
#endif

using dflash::common::ApiFormat;
using dflash::common::CanonicalToolInvocation;
using dflash::common::ToolCall;
using dflash::common::ToolSpeculationAttempt;
using dflash::common::ToolSpeculationConfig;
using dflash::common::ToolSpeculationPolicy;
using dflash::common::ToolSpeculationPrediction;
using dflash::common::build_tool_speculation_prediction;
using dflash::common::json;
using dflash::common::parse_tool_speculation_prediction;
using dflash::common::parse_tool_speculation_cpu_affinity;
using dflash::common::qualify_tool_speculation_cpu_affinity;
using dflash::common::render_tool_speculation_sse;

namespace {
struct ToolSpeculationFixture {};

json policy_fixture(bool decode_interference_qualified = true) {
    auto path = [decode_interference_qualified](
                    double hit_task, double miss_task,
                    double slowdown_percent) {
        return json{
            {"decode_interference_qualified",
             decode_interference_qualified},
            {"accelerator_relation", "non_accelerator"},
            {"hit", {
                {"control_task_mean_ms", 100.0},
                {"speculative_task_mean_ms", hit_task},
                {"model_slowdown_percent", slowdown_percent},
            }},
            {"miss", {
                {"control_task_mean_ms", 100.0},
                {"speculative_task_mean_ms", miss_task},
                {"model_slowdown_percent", slowdown_percent},
            }},
        };
    };
    json fixture = {
        {"profile_status", "qualified"},
        {"executor", "child_process"},
        {"path_summary", {
            {"25", path(80.0, 101.0, 2.0)},
            {"50", path(60.0, 110.0, 7.0)},
            {"100", path(50.0, 130.0, 16.0)},
        }},
    };
    return fixture;
}

ToolSpeculationConfig test_config(const std::string & executor = {}) {
    ToolSpeculationConfig config;
    config.executor_path = executor.empty() ? "/unused/executor" : executor;
    config.profile_path = "fixture.json";
    config.allowed_tools = {"lookup"};
    config.timeout_ms = 1000;
    config.cancel_grace_ms = 20;
    config.max_model_slowdown_ratio = 1.20;
    std::string error;
    if (!config.policy.load_json(policy_fixture(), error)) {
        throw std::runtime_error(error);
    }
    return config;
}

ToolSpeculationPrediction prediction(double confidence = 0.9) {
    ToolSpeculationPrediction value;
    std::string error;
    if (!CanonicalToolInvocation::from_parts(
            "lookup", json{{"a", 1}, {"b", 2}}, value.call, error)) {
        throw std::runtime_error(error);
    }
    value.confidence = confidence;
    return value;
}

#if !defined(_WIN32)
std::string make_executor_script(const std::string & body) {
    char path[] = "/tmp/dflash-tool-spec-test-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    const std::string script = "#!/bin/sh\nIFS= read -r request\n" + body;
    size_t offset = 0;
    while (offset < script.size()) {
        const ssize_t count = ::write(
            fd, script.data() + offset, script.size() - offset);
        if (count <= 0) {
            ::close(fd);
            ::unlink(path);
            throw std::runtime_error("script write failed");
        }
        offset += static_cast<size_t>(count);
    }
    if (::fchmod(fd, 0700) != 0) {
        ::close(fd);
        ::unlink(path);
        throw std::runtime_error("chmod failed");
    }
    ::close(fd);
    return path;
}

std::string make_temp_path() {
    char path[] = "/tmp/dflash-tool-spec-observed-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    ::close(fd);
    return path;
}

std::string read_text_file(const std::string & path) {
    FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) return {};
    std::string value;
    char buffer[256];
    while (const size_t count = std::fread(buffer, 1, sizeof(buffer), file)) {
        value.append(buffer, count);
    }
    std::fclose(file);
    return value;
}
#endif
}  // namespace

TEST_CASE(ToolSpeculationFixture, canonical_identity_ignores_argument_order) {
    CanonicalToolInvocation first;
    CanonicalToolInvocation second;
    std::string error;
    CHECK(CanonicalToolInvocation::from_parts(
        "lookup", json{{"b", 2}, {"a", 1}}, first, error));
    CHECK(CanonicalToolInvocation::from_parts(
        "lookup", json{{"a", 1}, {"b", 2}}, second, error));
    CHECK(first == second);
    CHECK(first.arguments_json == R"({"a":1,"b":2})");
}

TEST_CASE(ToolSpeculationFixture, engine_prediction_uses_canonical_boundary) {
    ToolSpeculationPrediction value;
    std::string error;
    CHECK(build_tool_speculation_prediction(
        "lookup", json{{"b", 2}, {"a", 1}}, 0.75, value, error));
    CHECK(error.empty());
    CHECK(value.call.name == "lookup");
    CHECK(value.call.arguments_json == "{\"a\":1,\"b\":2}");
    CHECK(std::fabs(value.confidence - 0.75) < 1e-9);
    CHECK(!build_tool_speculation_prediction(
        "lookup", json::array(), 0.75, value, error));
    CHECK(!build_tool_speculation_prediction(
        "lookup", json::object(), 1.01, value, error));
}

TEST_CASE(ToolSpeculationFixture, prediction_requires_declared_tool) {
    const json tools = json::array({{
        {"type", "function"},
        {"function", {{"name", "lookup"}}},
    }});
    const json request = {
        {"call", {
            {"name", "lookup"},
            {"arguments", {{"key", "x"}}},
        }},
        {"confidence", 0.75},
    };
    ToolSpeculationPrediction parsed;
    std::string error;
    CHECK(parse_tool_speculation_prediction(request, tools, parsed, error));
    CHECK(parsed.call.name == "lookup");

    json undeclared = request;
    undeclared["call"]["name"] = "write_file";
    CHECK(!parse_tool_speculation_prediction(
        undeclared, tools, parsed, error));
    CHECK(error.find("not declared") != std::string::npos);
}

TEST_CASE(ToolSpeculationFixture, cpu_affinity_parser_canonicalizes_ranges) {
    std::vector<int> cpus;
    std::string error;
    CHECK(parse_tool_speculation_cpu_affinity(
        "30-31,15,14-15", cpus, error));
    CHECK(cpus == std::vector<int>({14, 15, 30, 31}));

    CHECK(!parse_tool_speculation_cpu_affinity("14,,15", cpus, error));
    CHECK(cpus.empty());
    CHECK(!parse_tool_speculation_cpu_affinity("15-14", cpus, error));
    CHECK(cpus.empty());
}

TEST_CASE(ToolSpeculationFixture, empirical_policy_selects_resource_by_confidence) {
    ToolSpeculationPolicy policy;
    std::string error;
    CHECK(policy.load_json(policy_fixture(), error));

    const auto deferred = policy.choose(0.0, 1.20);
    CHECK(!deferred.admitted);
    CHECK(deferred.reason == "below_profile_break_even");

    const auto low = policy.choose(0.10, 1.20);
    CHECK(low.admitted);
    CHECK(low.resource_percentage == 25);

    const auto medium = policy.choose(0.50, 1.20);
    CHECK(medium.admitted);
    CHECK(medium.resource_percentage == 50);

    const auto high = policy.choose(0.90, 1.20);
    CHECK(high.admitted);
    CHECK(high.resource_percentage == 100);

    const auto guarded = policy.choose(0.90, 1.10);
    CHECK(guarded.admitted);
    CHECK(guarded.resource_percentage == 50);
}

TEST_CASE(ToolSpeculationFixture, unqualified_resource_lanes_are_deferred) {
    ToolSpeculationPolicy policy;
    std::string error;
    CHECK(policy.load_json(policy_fixture(false), error));

    const auto decision = policy.choose(1.0, 1.20);
    CHECK(!decision.admitted);
    CHECK(decision.reason == "decode_interference_unqualified");
}

TEST_CASE(ToolSpeculationFixture, profile_metadata_is_fail_closed) {
    ToolSpeculationPolicy policy;
    std::string error;
    json fixture = policy_fixture();
    CHECK(policy.load_json(fixture, error));
    CHECK(policy.profile_status() == "qualified");
    CHECK(policy.executor_contract() == "child_process");

    fixture["profile_status"] = "provisional_benchmark_only";
    CHECK(!policy.load_json(fixture, error));
    CHECK(policy.empty());

    fixture["profile_status"] = "unknown";
    CHECK(!policy.load_json(fixture, error));
    CHECK(policy.empty());

    fixture = policy_fixture();
    fixture.erase("executor");
    CHECK(!policy.load_json(fixture, error));
    CHECK(policy.empty());

    fixture = policy_fixture();
    fixture.erase("profile_status");
    CHECK(!policy.load_json(fixture, error));
    CHECK(policy.empty());

    fixture = policy_fixture();
    fixture["path_summary"]["25"].erase("accelerator_relation");
    CHECK(!policy.load_json(fixture, error));
    CHECK(policy.empty());
}

TEST_CASE(ToolSpeculationFixture, same_gpu_profile_is_rejected) {
    ToolSpeculationPolicy policy;
    std::string error;
    json fixture = policy_fixture();
    for (auto & lane : fixture["path_summary"]) {
        lane["accelerator_relation"] = "same_physical_gpu";
    }
    CHECK(!policy.load_json(fixture, error));
    CHECK(error.find("accelerator_relation") != std::string::npos);
    CHECK(policy.empty());
}

TEST_CASE(ToolSpeculationFixture, non_allowlisted_tool_is_deferred) {
    ToolSpeculationConfig config = test_config();
    config.allowed_tools = {"other"};
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_allowlist");
    attempt->start();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    CHECK(metadata["status"] == "deferred");
    CHECK(metadata["reason"] == "tool_not_allowlisted");
    CHECK(!metadata.contains("result"));
}

#if !defined(_WIN32)
TEST_CASE(ToolSpeculationFixture, cpu_affinity_reaches_child_executor) {
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    CHECK(::sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
    int selected_cpu = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            selected_cpu = cpu;
            break;
        }
    }
    CHECK(selected_cpu >= 0);
    const std::string selected = std::to_string(selected_cpu);
    const std::string path = make_executor_script(
        "IFS= read -r control\n"
        "observed=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)\n"
        "printf '{\"ok\":true,\"result\":{\"configured\":\"%s\","
        "\"observed\":\"%s\"}}\\n' "
        "\"$DFLASH_TOOL_SPECULATION_CPU_AFFINITY\" \"$observed\"\n");
    ToolSpeculationConfig config = test_config(path);
    config.cpu_affinity = {selected_cpu};
    config.cpu_affinity_isolated = true;
    CHECK(std::string(config.execution_mode()) ==
          "child_process_cpu_affinity");
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_cpu_affinity");
    attempt->start();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    ::unlink(path.c_str());
    CHECK(metadata["status"] == "hit");
    CHECK(metadata["cpu_affinity_isolated"].get<bool>());
    CHECK(metadata["result"]["configured"] == selected);
    CHECK(metadata["result"]["observed"] == selected);
#else
    CHECK(true);
#endif
}

TEST_CASE(ToolSpeculationFixture, cpu_affinity_qualification_rejects_overlap) {
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    CHECK(::sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
    int selected_cpu = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            selected_cpu = cpu;
            break;
        }
    }
    CHECK(selected_cpu >= 0);
    ToolSpeculationConfig config = test_config();
    config.cpu_affinity = {selected_cpu};
    std::string error;
    CHECK(!qualify_tool_speculation_cpu_affinity(config, error));
    CHECK(error.find("overlaps model CPU") != std::string::npos);
    CHECK(!config.cpu_affinity_isolated);
#else
    CHECK(true);
#endif
}

TEST_CASE(ToolSpeculationFixture, exact_match_exposes_result_and_resource_share) {
    const std::string control_path = make_temp_path();
    const std::string path = make_executor_script(
        "IFS= read -r control\nprintf '%s' \"$control\" > " + control_path + "\n"
        "printf '{\"ok\":true,\"result\":{\"resource\":\"%s\","
        "\"relation\":\"%s\",\"value\":42}}\\n' "
        "\"$DFLASH_TOOL_SPECULATION_RESOURCE_PERCENTAGE\" "
        "\"$DFLASH_TOOL_SPECULATION_ACCELERATOR_RELATION\"\n");
    ToolSpeculationConfig config = test_config(path);
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_hit");
    attempt->start();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"b":2,"a":1})"},
    });
    ::unlink(path.c_str());
    const std::string control = read_text_file(control_path);
    ::unlink(control_path.c_str());

    CHECK(metadata["status"] == "hit");
    CHECK(metadata["resource_percentage"] == 100);
    CHECK(metadata["result"]["resource"] == "100");
    CHECK(metadata["result"]["relation"] == "non_accelerator");
    CHECK(metadata["result"]["value"] == 42);
    CHECK(control.find("\"op\":\"commit\"") != std::string::npos);
    CHECK(control.find("\"authoritative_resource_percentage\":100") !=
          std::string::npos);
}

TEST_CASE(ToolSpeculationFixture, unqualified_tool_never_launches_or_changes_decode) {
    const std::string path = make_executor_script(
        "IFS= read -r control\n"
        "exit 0\n");
    ToolSpeculationConfig config = test_config(path);
    std::string error;
    CHECK(config.policy.load_json(policy_fixture(false), error));
    auto deferred = ToolSpeculationAttempt::create(
        config, prediction(), "request_guarded_decode");
    deferred->start();
    CHECK(!deferred->running());
    const json metadata = deferred->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    CHECK(metadata["status"] == "deferred");
    CHECK(metadata["reason"] == "decode_interference_unqualified");
    ::unlink(path.c_str());
}

TEST_CASE(ToolSpeculationFixture, executor_failure_is_private) {
    ToolSpeculationConfig config = test_config(
        "/definitely/missing/dflash-tool-spec-executor");
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_executor_failure");
    attempt->start();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    CHECK(metadata["status"] == "failed");
    CHECK(metadata["reason"] == "executor_launch_failed");
    CHECK(!metadata.contains("result"));
}

TEST_CASE(ToolSpeculationFixture, finished_result_survives_long_generation) {
    // The result budget starts at commit time: a result that was ready long
    // before the authoritative call arrived (slow target) must still count.
    const std::string path = make_executor_script(
        "printf '{\"ok\":true,\"result\":{\"value\":42}}\\n'\n");
    ToolSpeculationConfig config = test_config(path);
    config.timeout_ms = 25;
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_late_commit");
    attempt->start();
    // Much longer than timeout_ms, shorter than the absolute lifetime cap.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    ::unlink(path.c_str());

    CHECK(metadata["status"] == "hit");
    CHECK(metadata["result"]["value"] == 42);
}

TEST_CASE(ToolSpeculationFixture, closed_tool_blocks_are_found_as_they_stream) {
    using dflash::common::find_closed_tool_call_blocks;
    const std::string partial =
        "Let me check.\n<function_call>\n{\"name\": \"a\", \"arguments\": {}}\n</function_call>\n"
        "<function_call>\n{\"name\": \"b\", \"argu";
    auto blocks = find_closed_tool_call_blocks(partial, 0);
    CHECK(blocks.size() == 1u);
    CHECK(partial.substr(blocks[0].begin, blocks[0].end - blocks[0].begin).find("\"a\"") != std::string::npos);
    // Second block closes later; scanning from the end of the first one finds only it.
    const std::string full = partial + "ments\": {\"x\": 1}}\n</function_call>";
    blocks = find_closed_tool_call_blocks(full, blocks[0].end);
    CHECK(blocks.size() == 1u);
    CHECK(full.substr(blocks[0].begin, blocks[0].end - blocks[0].begin).find("\"b\"") != std::string::npos);
    // Nested wrappers are reported once (outermost block).
    const std::string nested =
        "<tool_call><function=lookup><parameter=a>1</parameter></function></tool_call>";
    blocks = find_closed_tool_call_blocks(nested, 0);
    CHECK(blocks.size() == 1u);
    CHECK(blocks[0].begin == 0u);
    CHECK(blocks[0].end == nested.size());
    // Unclosed opener yields nothing.
    CHECK(find_closed_tool_call_blocks("<function=lookup><parameter=a>1</parameter>", 0).empty());
}

TEST_CASE(ToolSpeculationFixture, dependency_placeholders_resolve_between_calls) {
    using dflash::common::find_tool_call_dependencies;
    using dflash::common::substitute_tool_call_dependencies;
    const json args = json::parse(R"json({"latitude":"$1.latitude","longitude":"$1.value.longitude","label":"city $2.name (#$1.id)","plain":3})json");
    auto deps = find_tool_call_dependencies(args);
    std::sort(deps.begin(), deps.end());
    CHECK(deps.size() == 2u);
    CHECK(deps[0] == 1);
    CHECK(deps[1] == 2);
    const json r1 = json::parse(R"json({"value":{"latitude":35.6769,"longitude":139.7639},"id":7})json");
    const json r2 = json::parse(R"json({"name":"Tokyo"})json");
    json resolved;
    std::string error;
    const bool ok = substitute_tool_call_dependencies(
        args, [&](int index) -> const json * { return index == 1 ? &r1 : index == 2 ? &r2 : nullptr; },
        resolved, error);
    CHECK(ok);
    CHECK(resolved["latitude"] == 35.6769);        // whole-string placeholder keeps the number type
    CHECK(resolved["longitude"] == 139.7639);      // explicit value.path
    CHECK(resolved["label"] == "city Tokyo (#7)"); // embedded placeholders stringify
    CHECK(resolved["plain"] == 3);
    // Missing field fails closed.
    json bad;
    CHECK(!substitute_tool_call_dependencies(
        json::parse(R"json({"x":"$1.nope"})json"), [&](int) { return &r1; }, bad, error));
    CHECK(find_tool_call_dependencies(json::parse(R"json({"a":"costs $5.00 today"})json")).empty());
}

TEST_CASE(ToolSpeculationFixture, executor_cap_defers_excess_attempts) {
    const std::string path = make_executor_script(
        "sleep 2\n"
        "printf '{\"ok\":true,\"result\":{\"value\":1}}\\n'\n");
    ToolSpeculationConfig config = test_config(path);
    config.max_concurrent_executors = 1;
    auto first = ToolSpeculationAttempt::create(config, prediction(), "cap_first");
    first->start();
    CHECK(first->running());
    {
        auto second = ToolSpeculationAttempt::create(
            config, prediction(), "cap_second");
        second->start();
        CHECK(!second->running());
        const json deferred = second->resolve({});
        CHECK(deferred["status"] == "deferred");
        CHECK(deferred["reason"] == "executor_saturated");
    }
    first->cancel("test_teardown");
    first.reset();
    ::unlink(path.c_str());
    CHECK(dflash::common::tool_speculation_running_executors() == 0);
}

TEST_CASE(ToolSpeculationFixture, executor_timeout_starts_at_launch) {
    const std::string path = make_executor_script(
        "sleep 1\n"
        "printf '{\"ok\":true,\"result\":{\"value\":42}}\\n'\n");
    ToolSpeculationConfig config = test_config(path);
    config.timeout_ms = 25;
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_launch_deadline");
    attempt->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto resolve_started = std::chrono::steady_clock::now();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    const double resolve_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - resolve_started).count();
    ::unlink(path.c_str());

    CHECK(metadata["status"] == "failed");
    CHECK(metadata["reason"] == "speculative_executor_failure");
    CHECK(metadata["detail"] == "executor_timeout");
    CHECK(!metadata.contains("result"));
    CHECK(resolve_ms < 500.0);
}

TEST_CASE(ToolSpeculationFixture, executor_timeout_terminates_process_group) {
#if defined(__linux__)
    const std::string marker_path = make_temp_path();
    const std::string path = make_executor_script(
        "sleep 10 &\n"
        "printf '%s' \"$!\" > \"" + marker_path + "\"\n"
        "IFS= read -r control\n"
        "wait\n");
    ToolSpeculationConfig config = test_config(path);
    config.timeout_ms = 100;
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_process_group_deadline");
    attempt->start();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    const std::string child_text = read_text_file(marker_path);
    ::unlink(path.c_str());
    ::unlink(marker_path.c_str());

    CHECK(metadata["status"] == "failed");
    CHECK(metadata["detail"] == "executor_timeout");
    CHECK(!child_text.empty());
    if (child_text.empty()) return;
    const pid_t child = static_cast<pid_t>(std::stol(child_text));
    bool gone = false;
    for (int retry = 0; retry < 50; ++retry) {
        if (::kill(child, 0) != 0 && errno == ESRCH) {
            gone = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(gone);
#else
    CHECK(true);
#endif
}

TEST_CASE(ToolSpeculationFixture, executor_does_not_inherit_server_fds) {
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#  if __GLIBC_PREREQ(2, 34)
    const std::string marker_path = make_temp_path();
    const int marker_fd = ::open(marker_path.c_str(), O_RDONLY);
    CHECK(marker_fd >= 0);
    const std::string path = make_executor_script(
        "IFS= read -r control\n"
        "leaked=false\n"
        "for descriptor in /proc/self/fd/*; do\n"
        "  target=$(readlink \"$descriptor\" 2>/dev/null || true)\n"
        "  if [ \"$target\" = \"" + marker_path + "\" ]; then leaked=true; fi\n"
        "done\n"
        "printf '{\"ok\":true,\"result\":{\"leaked\":%s}}\\n' \"$leaked\"\n");
    ToolSpeculationConfig config = test_config(path);
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_fd_isolation");
    attempt->start();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    ::close(marker_fd);
    ::unlink(path.c_str());
    ::unlink(marker_path.c_str());

    CHECK(metadata["status"] == "hit");
    CHECK(!metadata["result"]["leaked"].get<bool>());
#  else
    CHECK(true);
#  endif
#else
    CHECK(true);
#endif
}

TEST_CASE(ToolSpeculationFixture, executor_environment_is_minimal) {
    const char * previous = std::getenv("DFLASH_TOOL_SPECULATION");
    const bool had_previous = previous != nullptr;
    const std::string previous_value = previous ? previous : "";
    const char * previous_secret = std::getenv("DFLASH_TEST_SERVER_SECRET");
    const bool had_previous_secret = previous_secret != nullptr;
    const std::string previous_secret_value = previous_secret
        ? previous_secret : "";
    CHECK(::setenv("DFLASH_TOOL_SPECULATION", "0", 1) == 0);
    CHECK(::setenv("DFLASH_TEST_SERVER_SECRET", "must-not-leak", 1) == 0);
    const std::string path = make_executor_script(
        "IFS= read -r control\n"
        "secret=false\n"
        "if [ \"${DFLASH_TEST_SERVER_SECRET+x}\" = x ]; then secret=true; fi\n"
        "printf '{\"ok\":true,\"result\":{\"enabled\":\"%s\","
        "\"secret_inherited\":%s}}\\n' "
        "\"$DFLASH_TOOL_SPECULATION\" \"$secret\"\n");
    ToolSpeculationConfig config = test_config(path);
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_clean_environment");
    attempt->start();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":1,"b":2})"},
    });
    ::unlink(path.c_str());
    if (had_previous) {
        CHECK(::setenv(
            "DFLASH_TOOL_SPECULATION", previous_value.c_str(), 1) == 0);
    } else {
        CHECK(::unsetenv("DFLASH_TOOL_SPECULATION") == 0);
    }
    if (had_previous_secret) {
        CHECK(::setenv(
            "DFLASH_TEST_SERVER_SECRET", previous_secret_value.c_str(), 1) == 0);
    } else {
        CHECK(::unsetenv("DFLASH_TEST_SERVER_SECRET") == 0);
    }

    CHECK(metadata["status"] == "hit");
    CHECK(metadata["result"]["enabled"] == "1");
    CHECK(!metadata["result"]["secret_inherited"].get<bool>());
}

TEST_CASE(ToolSpeculationFixture, qualified_lane_keeps_speculative_decode) {
    const std::string path = make_executor_script(
        "IFS= read -r control\n"
        "exit 0\n");
    ToolSpeculationConfig config = test_config(path);
    std::string error;
    CHECK(config.policy.load_json(policy_fixture(), error));
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_qualified_lane");
    attempt->start();
    CHECK(attempt->running());
    const json metadata = attempt->cancel("test_complete");
    CHECK(metadata["decode_interference_qualified"].get<bool>());
    ::unlink(path.c_str());
}

TEST_CASE(ToolSpeculationFixture, mismatch_cancels_and_never_exposes_result) {
    const std::string control_path = make_temp_path();
    const std::string path = make_executor_script(
        "IFS= read -r control\nprintf '%s' \"$control\" > " + control_path + "\n"
        "exit 3\n");
    ToolSpeculationConfig config = test_config(path);
    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_miss");
    attempt->start();
    const auto started = std::chrono::steady_clock::now();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":999})"},
    });
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ::unlink(path.c_str());
    const std::string control = read_text_file(control_path);
    ::unlink(control_path.c_str());

    CHECK(metadata["status"] == "miss");
    CHECK(metadata["reason"] == "invocation_mismatch");
    CHECK(!metadata.contains("result"));
    CHECK(elapsed_ms < 1000.0);
    CHECK(control.find("\"op\":\"cancel\"") != std::string::npos);
}
#endif

TEST_CASE(ToolSpeculationFixture, streaming_extension_keeps_result_explicit) {
    const json metadata = {
        {"status", "hit"},
        {"result", {{"value", 42}}},
    };
    const std::string event = render_tool_speculation_sse(
        ApiFormat::OPENAI_CHAT, "req_1", "model", metadata);
    CHECK(event.find("dflash_tool_speculation") != std::string::npos);
    CHECK(event.find("\"value\":42") != std::string::npos);
}
