#include "CppUnitTestFramework.hpp"
#include "server/tool_speculation.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#  include <sys/stat.h>
#  include <unistd.h>
#endif

using dflash::common::ApiFormat;
using dflash::common::CanonicalToolInvocation;
using dflash::common::ToolCall;
using dflash::common::ToolSpeculationAttempt;
using dflash::common::ToolSpeculationConfig;
using dflash::common::ToolSpeculationExecution;
using dflash::common::ToolSpeculationExecutor;
using dflash::common::ToolSpeculationPolicy;
using dflash::common::ToolSpeculationPrediction;
using dflash::common::json;
using dflash::common::parse_tool_speculation_prediction;
using dflash::common::render_tool_speculation_sse;

namespace {
struct ToolSpeculationFixture {};

struct FakeExecutionState {
    json request;
    std::vector<std::string> controls;
    bool terminated = false;
    bool collected = false;
};

class FakeExecution final : public ToolSpeculationExecution {
public:
    explicit FakeExecution(std::shared_ptr<FakeExecutionState> state)
        : state_(std::move(state)) {}

    bool send_control(const std::string & operation) override {
        state_->controls.push_back(operation);
        return operation == "commit" || operation == "cancel";
    }

    bool collect_result(int timeout_ms,
                        size_t max_result_bytes,
                        json & result,
                        double & wait_ms,
                        std::string & error) override {
        (void) timeout_ms;
        state_->collected = true;
        result = {{"value", 42}};
        wait_ms = 0.0;
        if (result.dump().size() > max_result_bytes) {
            error = "executor_result_too_large";
            return false;
        }
        error.clear();
        return true;
    }

    void terminate(bool allow_control_grace) override {
        (void) allow_control_grace;
        state_->terminated = true;
    }

private:
    std::shared_ptr<FakeExecutionState> state_;
};

class FakeExecutor final : public ToolSpeculationExecutor {
public:
    explicit FakeExecutor(std::shared_ptr<FakeExecutionState> state)
        : state_(std::move(state)) {}

    std::unique_ptr<ToolSpeculationExecution> start(
            const json & request, std::string & error) override {
        state_->request = request;
        error.clear();
        return std::make_unique<FakeExecution>(state_);
    }

    const char * mode_name() const override {
        return "fake_in_process";
    }

private:
    std::shared_ptr<FakeExecutionState> state_;
};

json policy_fixture(bool decode_interference_qualified = true) {
    auto path = [decode_interference_qualified](
                    double hit_task, double miss_task,
                    double slowdown_percent) {
        return json{
            {"decode_interference_qualified",
             decode_interference_qualified},
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
    fixture["profile_status"] = "provisional_benchmark_only";
    fixture["executor"] = "in_process_hip_cu_mask";
    CHECK(policy.load_json(fixture, error));
    CHECK(policy.benchmark_only());
    CHECK(policy.executor_contract() == "in_process_hip_cu_mask");

    fixture["profile_status"] = "unknown";
    CHECK(!policy.load_json(fixture, error));
    CHECK(policy.empty());
}

TEST_CASE(ToolSpeculationFixture, same_gpu_profile_declares_routing_requirements) {
    ToolSpeculationPolicy policy;
    std::string error;
    json fixture = policy_fixture();
    for (auto & lane : fixture["path_summary"]) {
        lane["accelerator_relation"] = "same_physical_gpu";
    }
    CHECK(policy.load_json(fixture, error));
    CHECK(!policy.requires_static_model_routing());
    CHECK(!policy.requires_unique_expert_ownership());

    for (auto & lane : fixture["path_summary"]) {
        lane["requires_static_model_routing"] = true;
    }
    CHECK(policy.load_json(fixture, error));
    CHECK(policy.requires_static_model_routing());
    CHECK(!policy.requires_unique_expert_ownership());

    for (auto & lane : fixture["path_summary"]) {
        lane["requires_unique_expert_ownership"] = true;
    }
    CHECK(policy.load_json(fixture, error));
    CHECK(policy.requires_static_model_routing());
    CHECK(policy.requires_unique_expert_ownership());
    const auto decision = policy.choose(1.0, 2.0);
    CHECK(decision.admitted);
    CHECK(decision.accelerator_relation == "same_physical_gpu");
}

TEST_CASE(ToolSpeculationFixture, same_gpu_profile_obeys_measured_break_even) {
    ToolSpeculationPolicy policy;
    std::string error;
    CHECK(policy.load_json(json{
        {"path_summary", {
            {"100", {
                {"accelerator_relation", "same_physical_gpu"},
                {"requires_static_model_routing", true},
                {"requires_unique_expert_ownership", true},
                {"decode_interference_qualified", true},
                {"hit", {
                    {"control_task_mean_ms", 2670.178},
                    {"speculative_task_mean_ms", 2389.530},
                    {"model_slowdown_percent", 169.109},
                }},
                {"miss", {
                    {"control_task_mean_ms", 2670.178},
                    {"speculative_task_mean_ms", 4171.884},
                    {"model_slowdown_percent", 169.109},
                }},
            }},
        }},
    }, error));

    const auto below = policy.choose(0.84, 3.0);
    CHECK(!below.admitted);
    CHECK(below.reason == "below_profile_break_even");

    const auto above = policy.choose(0.85, 3.0);
    CHECK(above.admitted);
    CHECK(above.resource_percentage == 100);

    const auto guarded = policy.choose(1.0, 1.20);
    CHECK(!guarded.admitted);
    CHECK(guarded.reason == "model_slowdown_guardrail");
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

TEST_CASE(ToolSpeculationFixture, in_process_exact_match_commits_private_result) {
    auto state = std::make_shared<FakeExecutionState>();
    ToolSpeculationConfig config = test_config();
    config.executor_path.clear();
    config.in_process_executor = std::make_shared<FakeExecutor>(state);
    CHECK(config.enabled());
    CHECK(std::string(config.execution_mode()) == "fake_in_process");

    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_in_process_hit");
    attempt->start();
    CHECK(attempt->running());
    CHECK(state->request["call"]["name"] == "lookup");
    CHECK(state->request["resource_percentage"] == 100);

    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"b":2,"a":1})"},
    });
    CHECK(metadata["status"] == "hit");
    CHECK(metadata["result"]["value"] == 42);
    CHECK(state->collected);
    CHECK(state->controls.size() == 1);
    CHECK(state->controls[0] == "commit");
    CHECK(!state->terminated);
}

TEST_CASE(ToolSpeculationFixture, in_process_mismatch_cancels_private_result) {
    auto state = std::make_shared<FakeExecutionState>();
    ToolSpeculationConfig config = test_config();
    config.executor_path.clear();
    config.in_process_executor = std::make_shared<FakeExecutor>(state);

    auto attempt = ToolSpeculationAttempt::create(
        config, prediction(), "request_in_process_miss");
    attempt->start();
    const json metadata = attempt->resolve({
        ToolCall{"call_1", "lookup", R"({"a":999})"},
    });
    CHECK(metadata["status"] == "miss");
    CHECK(metadata["reason"] == "invocation_mismatch");
    CHECK(!metadata.contains("result"));
    CHECK(!state->collected);
    CHECK(state->terminated);
    CHECK(state->controls.size() == 1);
    CHECK(state->controls[0] == "cancel");
}

#if !defined(_WIN32)
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
    CHECK(metadata["result"]["relation"] == "unspecified");
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
