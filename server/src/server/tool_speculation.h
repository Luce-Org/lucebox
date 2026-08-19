// Lossless, confidence-gated speculative tool execution.
//
// The model remains authoritative. A predicted read-only invocation may run
// while inference is in flight, but its result is returned only when the
// emitted tool name and canonical JSON arguments match exactly.

#pragma once

#include "api_types.h"
#include "tool_parser.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

using json = nlohmann::json;

struct CanonicalToolInvocation {
    std::string name;
    json arguments = json::object();
    std::string arguments_json;

    static bool from_parts(const std::string & name,
                           const json & arguments,
                           CanonicalToolInvocation & out,
                           std::string & error);
    static bool from_tool_call(const ToolCall & call,
                               CanonicalToolInvocation & out,
                               std::string & error);

    bool operator==(const CanonicalToolInvocation & other) const {
        return name == other.name && arguments_json == other.arguments_json;
    }
};

struct ToolSpeculationPrediction {
    CanonicalToolInvocation call;
    double confidence = 0.0;
};

// Construct a canonical prediction from an engine-side predictor. This is
// the same validation boundary used for caller-supplied predictions, minus
// the request-schema check performed by the semantic predictor itself.
bool build_tool_speculation_prediction(const std::string & name,
                                       const json & arguments,
                                       double confidence,
                                       ToolSpeculationPrediction & out,
                                       std::string & error);

// Parse the request extension:
//   "tool_speculation": {
//     "call": {"name": "...", "arguments": {...}},
//     "confidence": 0.0..1.0
//   }
// The predicted tool must also be present in the request's `tools` array.
bool parse_tool_speculation_prediction(const json & value,
                                       const json & tools,
                                       ToolSpeculationPrediction & out,
                                       std::string & error);

// Parse a Linux CPU-list such as "14-15,30-31". The result is sorted and
// deduplicated so it can be compared directly with an observed affinity mask.
bool parse_tool_speculation_cpu_affinity(const std::string & value,
                                         std::vector<int> & out,
                                         std::string & error);

struct ToolSpeculationLane {
    // Backend-neutral executor capacity. A CUDA adapter may map this to an
    // MPS share; a ROCm, CPU, I/O, or remote adapter may interpret it using
    // its own measured resource contract.
    int resource_percentage = 0;
    double control_task_ms = 0.0;
    double hit_task_ms = 0.0;
    double miss_task_ms = 0.0;
    double model_slowdown_ratio = 1.0;
    // True only when this exact executor/resource lane passed the model-output
    // interference gate. Missing profile metadata defers tool speculation;
    // token speculation is never disabled or replaced with AR decode.
    bool decode_interference_qualified = false;
    // Physical relationship between the tool accelerator and the model's
    // primary accelerator. Production child executors may use CPU/I/O or a
    // separate physical accelerator, but never the model's accelerator.
    std::string accelerator_relation = "unspecified";
};

struct ToolSpeculationAdmission {
    bool admitted = false;
    int resource_percentage = 0;
    double expected_task_ms = 0.0;
    double expected_speedup = 1.0;
    bool decode_interference_qualified = false;
    std::string accelerator_relation = "unspecified";
    std::string reason;
};

// Runtime policy loaded from a qualification report's `path_summary`. This
// keeps backend-specific interference measurements out of hard-coded engine
// heuristics.
class ToolSpeculationPolicy {
public:
    bool load_file(const std::string & path, std::string & error);
    bool load_json(const json & report, std::string & error);

    ToolSpeculationAdmission choose(
        double confidence,
        double max_model_slowdown_ratio) const;

    bool empty() const { return lanes_.empty(); }
    double baseline_task_ms() const { return baseline_task_ms_; }
    const std::vector<ToolSpeculationLane> & lanes() const { return lanes_; }
    const std::string & profile_status() const { return profile_status_; }
    const std::string & executor_contract() const { return executor_contract_; }

private:
    std::vector<ToolSpeculationLane> lanes_;
    double baseline_task_ms_ = 0.0;
    std::string profile_status_ = "qualified";
    std::string executor_contract_;
};

struct ToolSpeculationConfig {
    std::string executor_path;
    std::string profile_path;
    std::vector<std::string> allowed_tools;
    ToolSpeculationPolicy policy;
    int timeout_ms = 60000;
    int cancel_grace_ms = 100;
    size_t max_result_bytes = 1024 * 1024;
    double max_model_slowdown_ratio = 1.20;
    // Optional child-process CPU lane. Startup verifies that these logical
    // CPUs are disjoint from the model process affinity; every child is pinned
    // and re-read before its request payload is released.
    std::vector<int> cpu_affinity;
    std::vector<int> model_cpu_affinity;
    bool cpu_affinity_isolated = false;
    // An executor plus an allowlist enables the lane. A measured resource
    // profile is only required for speculative (confidence < 1) launches;
    // authoritative calls dispatched from the model's own stream are admitted
    // without one.
    bool enabled() const {
        return !executor_path.empty() && !allowed_tools.empty();
    }
    bool predictive_enabled() const {
        return enabled() && !policy.empty();
    }
    const char * execution_mode() const {
        return executor_path.empty()
            ? "disabled"
            : cpu_affinity.empty()
                ? "child_process"
                : "child_process_cpu_affinity";
    }
    bool allows(const std::string & name) const;
};

// Report whether this build can close every non-protocol descriptor before
// executing an untrusted tool child. Unsupported platforms fail closed.
bool tool_speculation_executor_isolation_supported();

// Capture the model process affinity and fail closed unless it is physically
// disjoint from the configured child executor CPUs. No-op when no CPU lane is
// requested.
bool qualify_tool_speculation_cpu_affinity(ToolSpeculationConfig & config,
                                           std::string & error);

// One request-scoped attempt. The configured executable is invoked without a
// shell and receives one JSON request on stdin. It must emit one JSON envelope
// on stdout: {"ok":true,"result":...}. Stdin remains open for a later
// `commit` (exact match; promote checkpointed remainder to the authoritative
// 100% lane) or `cancel` control record. A thin executable may forward this
// protocol to a persistent warm tool pool, keeping GPU initialization outside
// the request's critical path.
class ToolSpeculationAttempt {
public:
    ToolSpeculationAttempt(const ToolSpeculationAttempt &) = delete;
    ToolSpeculationAttempt & operator=(const ToolSpeculationAttempt &) = delete;
    ~ToolSpeculationAttempt();

    static std::unique_ptr<ToolSpeculationAttempt> create(
        const ToolSpeculationConfig & config,
        const ToolSpeculationPrediction & prediction,
        const std::string & request_id);

    // Launch admitted work. Deferred and launch-failed attempts still return
    // metadata through resolve(), so an opted-in client can see why it must
    // execute the authoritative tool normally.
    void start();

    // Exact-match one authoritative call, expose a successful private result,
    // or discard/cancel it. This method is single-use.
    json resolve(const std::vector<ToolCall> & authoritative_calls);

    // Cancel without exposing a result (disconnect, generation failure, etc.).
    json cancel(const std::string & reason);

    bool admitted() const { return admission_.admitted; }
    bool running() const { return running_; }
private:
    ToolSpeculationAttempt(const ToolSpeculationConfig & config,
                           const ToolSpeculationPrediction & prediction,
                           const std::string & request_id);

    json base_metadata() const;
    bool send_control(const char * operation);
    void terminate_executor(bool allow_control_grace = false);
    bool collect_executor_result(json & result,
                                 double & wait_ms,
                                 std::string & error);

    ToolSpeculationConfig config_;
    ToolSpeculationPrediction prediction_;
    std::string request_id_;
    ToolSpeculationAdmission admission_;
    std::chrono::steady_clock::time_point started_at_{};
    bool started_ = false;
    bool running_ = false;
    bool resolved_ = false;
    std::string launch_error_;
#if !defined(_WIN32)
    int child_stdin_fd_ = -1;
    int child_stdout_fd_ = -1;
    int child_pid_ = -1;
#endif
};

// Locate closed tool-call blocks in streamed text. Supports the
// `<function_call>…</function_call>`, `<tool_call>…</tool_call>` and
// `<function=NAME>…</function>` wrappers. Each result is the [begin, end)
// byte range of one complete outermost block at or after `from`; nested
// wrappers are reported once. Blocks whose closer has not streamed yet are
// not reported, so a caller can dispatch as soon as a block is complete.
struct ClosedToolCallBlock {
    size_t begin = 0;
    size_t end = 0;
};
std::vector<ClosedToolCallBlock> find_closed_tool_call_blocks(
    const std::string & text, size_t from);

// Custom SSE extension emitted only for requests that supplied
// `tool_speculation`. Non-streaming responses use the same object under the
// top-level `dflash_tool_speculation` key.
std::string render_tool_speculation_sse(ApiFormat api_format,
                                        const std::string & request_id,
                                        const std::string & model,
                                        const json & metadata);

}  // namespace dflash::common
