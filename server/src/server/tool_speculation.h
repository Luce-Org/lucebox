// Model-agnostic speculative tool execution.
//
// A trusted external service predicts and starts one allowlisted tool before
// model execution. The model remains authoritative: the private result is
// committed only when the emitted tool call matches exactly.

#pragma once

#include "api_types.h"
#include "tool_parser.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

using json = nlohmann::json;

inline constexpr char kToolSpeculationProtocol[] =
    "dflash.tool-speculation.v1";

struct ToolSpeculationConfig {
    std::string endpoint;
    std::string api_key;
    std::vector<std::string> allowed_tools;
    double min_confidence = 0.75;
    int start_timeout_ms = 2000;
    int finish_timeout_ms = 60000;

    bool enabled() const {
        return !endpoint.empty() && !allowed_tools.empty();
    }
    bool allows(const std::string & name) const;
};

// The engine deliberately knows nothing about the predictor model, tool
// runtime, or hardware placement. A transport posts one protocol request to
// the configured service and returns its JSON response.
using ToolSpeculationTransport = std::function<bool(
    const json & request,
    int timeout_ms,
    json & response,
    std::string & error)>;

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

class ToolSpeculationAttempt {
public:
    ToolSpeculationAttempt(const ToolSpeculationAttempt &) = delete;
    ToolSpeculationAttempt & operator=(const ToolSpeculationAttempt &) = delete;
    ~ToolSpeculationAttempt();

    // Returns null when the request has no allowlisted tools. Otherwise the
    // service call is synchronous: a successful return means the tool has
    // started before model compute begins.
    static std::unique_ptr<ToolSpeculationAttempt> begin(
        const ToolSpeculationConfig & config,
        const std::string & request_id,
        const json & messages,
        const json & tools,
        const json & tool_choice,
        ToolSpeculationTransport transport);

    // Commit on one exact authoritative call; cancel on every other outcome.
    // A speculative result is never included in metadata on a miss.
    json finish(const std::vector<ToolCall> & authoritative_calls);
    json cancel(const std::string & reason);

private:
    ToolSpeculationAttempt(const ToolSpeculationConfig & config,
                           std::string request_id,
                           ToolSpeculationTransport transport);

    void start(const json & messages,
               const json & tools,
               const json & tool_choice);
    void cancel_service(const std::string & reason, json & metadata);
    json base_metadata() const;

    ToolSpeculationConfig config_;
    std::string request_id_;
    ToolSpeculationTransport transport_;
    CanonicalToolInvocation prediction_;
    std::string ticket_;
    std::string start_error_;
    double confidence_ = 0.0;
    double predictor_wall_ms_ = 0.0;
    bool started_ = false;
    bool resolved_ = false;
};

std::string render_tool_speculation_sse(ApiFormat api_format,
                                        const std::string & request_id,
                                        const std::string & model,
                                        const json & metadata);

}  // namespace dflash::common
