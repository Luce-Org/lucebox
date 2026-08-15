#include "tool_speculation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <utility>

namespace dflash::common {
namespace {

constexpr int kCancelTimeoutMs = 250;

bool declared_tool(const json & tools, const std::string & name) {
    if (!tools.is_array() || name.empty()) return false;
    for (const auto & tool : tools) {
        if (!tool.is_object()) continue;
        if (tool.contains("name") && tool["name"].is_string() &&
            tool["name"].get<std::string>() == name) {
            return true;
        }
        if (tool.contains("function") && tool["function"].is_object() &&
            tool["function"].contains("name") &&
            tool["function"]["name"].is_string() &&
            tool["function"]["name"].get<std::string>() == name) {
            return true;
        }
    }
    return false;
}

json allowed_tool_definitions(const json & tools,
                              const ToolSpeculationConfig & config) {
    json allowed = json::array();
    if (!tools.is_array()) return allowed;
    for (const auto & tool : tools) {
        if (!tool.is_object()) continue;
        std::string name;
        std::string description;
        json parameters = json::object();
        if (tool.contains("name") && tool["name"].is_string()) {
            name = tool["name"].get<std::string>();
            if (tool.contains("description") &&
                tool["description"].is_string()) {
                description = tool["description"].get<std::string>();
            }
            if (tool.contains("parameters") &&
                tool["parameters"].is_object()) {
                parameters = tool["parameters"];
            } else if (tool.contains("input_schema") &&
                       tool["input_schema"].is_object()) {
                parameters = tool["input_schema"];
            }
        }
        if (tool.contains("function") && tool["function"].is_object() &&
            tool["function"].contains("name") &&
            tool["function"]["name"].is_string()) {
            const json & function = tool["function"];
            name = function["name"].get<std::string>();
            if (function.contains("description") &&
                function["description"].is_string()) {
                description = function["description"].get<std::string>();
            }
            if (function.contains("parameters") &&
                function["parameters"].is_object()) {
                parameters = function["parameters"];
            }
        }
        if (config.allows(name)) {
            json normalized = {
                {"name", name},
                {"parameters", std::move(parameters)},
            };
            if (!description.empty()) {
                normalized["description"] = std::move(description);
            }
            allowed.push_back(std::move(normalized));
        }
    }
    return allowed;
}

json normalized_tool_choice(const json & choice) {
    if (choice.is_null() || choice.is_string()) return choice;
    if (!choice.is_object()) return nullptr;
    if (choice.contains("name") && choice["name"].is_string()) {
        return {{"name", choice["name"]}};
    }
    if (choice.contains("function") && choice["function"].is_object() &&
        choice["function"].contains("name") &&
        choice["function"]["name"].is_string()) {
        return {{"name", choice["function"]["name"]}};
    }
    if (choice.contains("type") && choice["type"].is_string()) {
        const std::string type = choice["type"].get<std::string>();
        if (type == "any") return "required";
        if (type == "auto" || type == "none" || type == "required") {
            return type;
        }
    }
    return nullptr;
}

std::string transport_failure(const std::string & prefix,
                              const std::string & detail) {
    return detail.empty() ? prefix : prefix + ": " + detail;
}

}  // namespace

bool ToolSpeculationConfig::allows(const std::string & name) const {
    return std::find(allowed_tools.begin(), allowed_tools.end(), name) !=
        allowed_tools.end();
}

bool CanonicalToolInvocation::from_parts(
        const std::string & name,
        const json & arguments,
        CanonicalToolInvocation & out,
        std::string & error) {
    if (name.empty()) {
        error = "tool name must not be empty";
        return false;
    }
    if (!arguments.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    out.name = name;
    out.arguments = arguments;
    // nlohmann::json objects are key ordered, making dump() a stable identity
    // independent of the provider's input key order.
    out.arguments_json = arguments.dump();
    error.clear();
    return true;
}

bool CanonicalToolInvocation::from_tool_call(
        const ToolCall & call,
        CanonicalToolInvocation & out,
        std::string & error) {
    try {
        const json arguments = call.arguments.empty()
            ? json::object()
            : json::parse(call.arguments);
        return from_parts(call.name, arguments, out, error);
    } catch (const std::exception & exception) {
        error = std::string("authoritative arguments are invalid JSON: ") +
            exception.what();
        return false;
    }
}

ToolSpeculationAttempt::ToolSpeculationAttempt(
        const ToolSpeculationConfig & config,
        std::string request_id,
        ToolSpeculationTransport transport)
    : config_(config),
      request_id_(std::move(request_id)),
      transport_(std::move(transport)) {}

ToolSpeculationAttempt::~ToolSpeculationAttempt() {
    if (started_ && !resolved_) {
        json ignored;
        cancel_service("attempt_destroyed", ignored);
    }
}

std::unique_ptr<ToolSpeculationAttempt> ToolSpeculationAttempt::begin(
        const ToolSpeculationConfig & config,
        const std::string & request_id,
        const json & messages,
        const json & tools,
        const json & tool_choice,
        ToolSpeculationTransport transport) {
    if (!config.enabled() || !transport) return nullptr;
    const json allowed = allowed_tool_definitions(tools, config);
    if (allowed.empty()) return nullptr;
    const json choice = normalized_tool_choice(tool_choice);
    if (choice.is_string() && choice.get<std::string>() == "none") {
        return nullptr;
    }
    if (choice.is_object() && choice.contains("name") &&
        choice["name"].is_string() &&
        !declared_tool(allowed, choice["name"].get<std::string>())) {
        return nullptr;
    }

    auto attempt = std::unique_ptr<ToolSpeculationAttempt>(
        new ToolSpeculationAttempt(config, request_id, std::move(transport)));
    attempt->start(messages, allowed, tool_choice);
    return attempt;
}

void ToolSpeculationAttempt::start(
        const json & messages,
        const json & tools,
        const json & tool_choice) {
    json request = {
        {"protocol", kToolSpeculationProtocol},
        {"operation", "start"},
        {"request_id", request_id_},
        {"messages", messages},
        {"tools", tools},
        {"min_confidence", config_.min_confidence},
    };
    const json normalized_choice = normalized_tool_choice(tool_choice);
    if (!normalized_choice.is_null()) {
        request["tool_choice"] = normalized_choice;
    }

    json response;
    std::string transport_error;
    const auto started_at = std::chrono::steady_clock::now();
    bool ok = false;
    try {
        ok = transport_(request, config_.start_timeout_ms,
                        response, transport_error);
    } catch (const std::exception & exception) {
        transport_error = exception.what();
    } catch (...) {
        transport_error = "unknown transport exception";
    }
    predictor_wall_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started_at).count();
    if (!ok) {
        start_error_ = transport_failure(
            "speculation service unavailable", transport_error);
        return;
    }
    if (!response.is_object()) {
        start_error_ = "start response must be an object";
        return;
    }

    ticket_ = response.contains("ticket") && response["ticket"].is_string()
        ? response["ticket"].get<std::string>() : std::string();
    if (!response.contains("call") || !response["call"].is_object()) {
        start_error_ = "start response call must be an object";
    } else if (!response.contains("confidence") ||
               !response["confidence"].is_number()) {
        start_error_ = "start response confidence must be a number";
    } else {
        try {
            const json & call = response["call"];
            const std::string name =
                call.contains("name") && call["name"].is_string()
                    ? call["name"].get<std::string>() : std::string();
            const json arguments = call.contains("arguments")
                ? call["arguments"] : json();
            confidence_ = response["confidence"].get<double>();
            std::string canonical_error;
            if (ticket_.empty()) {
                start_error_ = "start response ticket must not be empty";
            } else if (!std::isfinite(confidence_) || confidence_ < 0.0 ||
                       confidence_ > 1.0) {
                start_error_ = "prediction confidence must be between 0 and 1";
            } else if (confidence_ < config_.min_confidence) {
                start_error_ = "prediction below minimum confidence";
            } else if (!config_.allows(name) || !declared_tool(tools, name)) {
                start_error_ =
                    "predicted tool is not allowlisted for this request";
            } else if (!CanonicalToolInvocation::from_parts(
                           name, arguments, prediction_, canonical_error)) {
                start_error_ = canonical_error;
            }
        } catch (const std::exception & exception) {
            start_error_ = std::string("invalid start response: ") +
                exception.what();
        }
    }

    if (!start_error_.empty()) {
        if (!ticket_.empty()) {
            json ignored;
            cancel_service("invalid_prediction", ignored);
        }
        return;
    }
    started_ = true;
}

json ToolSpeculationAttempt::base_metadata() const {
    json metadata = {
        {"protocol", kToolSpeculationProtocol},
        {"prediction_source", "external"},
        {"predictor_wall_ms", predictor_wall_ms_},
    };
    if (!prediction_.name.empty()) {
        metadata["predicted_call"] = {
            {"name", prediction_.name},
            {"arguments", prediction_.arguments},
        };
        metadata["confidence"] = confidence_;
    }
    return metadata;
}

void ToolSpeculationAttempt::cancel_service(
        const std::string & reason,
        json & metadata) {
    if (ticket_.empty()) return;
    const json request = {
        {"protocol", kToolSpeculationProtocol},
        {"operation", "cancel"},
        {"request_id", request_id_},
        {"ticket", ticket_},
        {"reason", reason},
    };
    json response;
    std::string error;
    try {
        if (!transport_(request, kCancelTimeoutMs, response, error)) {
            metadata["cancel_error"] = transport_failure(
                "cancel request failed", error);
        }
    } catch (const std::exception & exception) {
        metadata["cancel_error"] = exception.what();
    } catch (...) {
        metadata["cancel_error"] = "unknown transport exception";
    }
    ticket_.clear();
    started_ = false;
}

json ToolSpeculationAttempt::finish(
        const std::vector<ToolCall> & authoritative_calls) {
    if (resolved_) {
        json metadata = base_metadata();
        metadata["status"] = "failed";
        metadata["reason"] = "already_resolved";
        return metadata;
    }
    resolved_ = true;
    json metadata = base_metadata();
    if (!started_) {
        metadata["status"] = "deferred";
        metadata["reason"] = start_error_.empty()
            ? "prediction_unavailable" : start_error_;
        return metadata;
    }
    if (authoritative_calls.size() != 1) {
        cancel_service("authoritative_call_count", metadata);
        metadata["status"] = "miss";
        metadata["reason"] = "authoritative_call_count";
        return metadata;
    }

    CanonicalToolInvocation authoritative;
    std::string canonical_error;
    if (!CanonicalToolInvocation::from_tool_call(
            authoritative_calls[0], authoritative, canonical_error)) {
        cancel_service("invalid_authoritative_call", metadata);
        metadata["status"] = "miss";
        metadata["reason"] = "invalid_authoritative_call";
        return metadata;
    }
    if (!(authoritative == prediction_)) {
        cancel_service("invocation_mismatch", metadata);
        metadata["status"] = "miss";
        metadata["reason"] = "invocation_mismatch";
        return metadata;
    }

    const json request = {
        {"protocol", kToolSpeculationProtocol},
        {"operation", "commit"},
        {"request_id", request_id_},
        {"ticket", ticket_},
    };
    json response;
    std::string error;
    const auto wait_started = std::chrono::steady_clock::now();
    bool ok = false;
    try {
        ok = transport_(request, config_.finish_timeout_ms, response, error);
    } catch (const std::exception & exception) {
        error = exception.what();
    } catch (...) {
        error = "unknown transport exception";
    }
    const double wait_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wait_started).count();
    ticket_.clear();
    started_ = false;
    metadata["commit_wait_ms"] = wait_ms;
    if (!ok) {
        metadata["status"] = "failed";
        metadata["reason"] = "commit_request_failed";
        metadata["detail"] = error;
        return metadata;
    }
    if (!response.is_object() || !response.contains("ok") ||
        !response["ok"].is_boolean() || !response["ok"].get<bool>() ||
        !response.contains("result")) {
        metadata["status"] = "failed";
        metadata["reason"] = "invalid_commit_response";
        return metadata;
    }
    metadata["status"] = "hit";
    metadata["call_id"] = authoritative_calls[0].id;
    metadata["result"] = response["result"];
    return metadata;
}

json ToolSpeculationAttempt::cancel(const std::string & reason) {
    if (resolved_) {
        json metadata = base_metadata();
        metadata["status"] = "failed";
        metadata["reason"] = "already_resolved";
        return metadata;
    }
    resolved_ = true;
    json metadata = base_metadata();
    cancel_service(reason, metadata);
    metadata["status"] = "cancelled";
    metadata["reason"] = reason;
    return metadata;
}

std::string render_tool_speculation_sse(
        ApiFormat api_format,
        const std::string & request_id,
        const std::string & model,
        const json & metadata) {
    switch (api_format) {
    case ApiFormat::OPENAI_CHAT:
        return "data: " + json({
            {"id", request_id},
            {"object", "chat.completion.chunk"},
            {"model", model},
            {"choices", json::array()},
            {"dflash_tool_speculation", metadata},
        }).dump() + "\n\n";
    case ApiFormat::ANTHROPIC:
        return "event: dflash_tool_speculation\ndata: " + json({
            {"type", "dflash_tool_speculation"},
            {"dflash_tool_speculation", metadata},
        }).dump() + "\n\n";
    case ApiFormat::RESPONSES:
        return "event: response.dflash_tool_speculation\ndata: " + json({
            {"type", "response.dflash_tool_speculation"},
            {"response_id", request_id},
            {"dflash_tool_speculation", metadata},
        }).dump() + "\n\n";
    default:
        return "data: " + json({
            {"dflash_tool_speculation", metadata},
        }).dump() + "\n\n";
    }
}

}  // namespace dflash::common
