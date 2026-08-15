#include "CppUnitTestFramework.hpp"

#include "server/tool_speculation.h"

#include <string>
#include <vector>

using namespace dflash::common;

namespace {

struct ToolSpeculationFixture {};

json function_tool(const std::string & name) {
    return {
        {"type", "function"},
        {"function", {
            {"name", name},
            {"description", "test tool"},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"city", {{"type", "string"}}},
                    {"unit", {{"type", "string"}}},
                }},
            }},
        }},
    };
}

ToolSpeculationConfig config() {
    ToolSpeculationConfig value;
    value.endpoint = "http://speculator.test/v1";
    value.allowed_tools = {"weather"};
    value.min_confidence = 0.75;
    return value;
}

struct FakeService {
    json start_response = {
        {"ticket", "ticket-1"},
        {"call", {
            {"name", "weather"},
            {"arguments", {{"city", "Rome"}, {"unit", "celsius"}}},
        }},
        {"confidence", 0.9},
    };
    json commit_response = {
        {"ok", true},
        {"result", {{"temperature", 24}}},
    };
    std::string failing_operation;
    std::vector<json> requests;

    ToolSpeculationTransport transport() {
        return [this](const json & request, int, json & response,
                      std::string & error) {
            requests.push_back(request);
            const std::string operation = request.value("operation", "");
            if (operation == failing_operation) {
                error = "injected failure";
                return false;
            }
            if (operation == "start") response = start_response;
            else if (operation == "commit") response = commit_response;
            else if (operation == "cancel") response = {{"ok", true}};
            else {
                error = "unexpected operation";
                return false;
            }
            error.clear();
            return true;
        };
    }
};

std::unique_ptr<ToolSpeculationAttempt> begin(FakeService & service) {
    return ToolSpeculationAttempt::begin(
        config(), "request-1",
        json::array({{{"role", "user"}, {"content", "Weather in Rome"}}}),
        json::array({function_tool("weather"), function_tool("delete_file")}),
        "auto", service.transport());
}

ToolCall weather_call(
        const std::string & arguments =
            R"({"unit":"celsius","city":"Rome"})") {
    return {"call-1", "weather", arguments};
}

}  // namespace

TEST_CASE(ToolSpeculationFixture, provider_contract_is_model_agnostic) {
    FakeService service;
    auto attempt = begin(service);
    CHECK_NOT_NULL(attempt.get());
    CHECK_EQUAL(size_t{1}, service.requests.size());

    const json & request = service.requests.front();
    CHECK(request["protocol"] == kToolSpeculationProtocol);
    CHECK(request["operation"] == "start");
    CHECK(!request.contains("model"));
    CHECK(!request.contains("token_ids"));
    CHECK_EQUAL(size_t{1}, request["tools"].size());
    CHECK(request["tools"][0]["name"] == "weather");
    CHECK(!request["tools"][0].contains("function"));

    const json metadata = attempt->finish({weather_call()});
    CHECK(metadata["status"] == "hit");
    CHECK(metadata["result"]["temperature"] == 24);
    CHECK(metadata["prediction_source"] == "external");
    CHECK_EQUAL(size_t{2}, service.requests.size());
    CHECK(service.requests.back()["operation"] == "commit");
}

TEST_CASE(ToolSpeculationFixture, mismatch_cancels_and_hides_private_result) {
    FakeService service;
    auto attempt = begin(service);
    const ToolCall actual{
        "call-1", "weather", R"({"city":"Milan","unit":"celsius"})"};

    const json metadata = attempt->finish({actual});
    CHECK(metadata["status"] == "miss");
    CHECK(metadata["reason"] == "invocation_mismatch");
    CHECK(!metadata.contains("result"));
    CHECK_EQUAL(size_t{2}, service.requests.size());
    CHECK(service.requests.back()["operation"] == "cancel");
}

TEST_CASE(ToolSpeculationFixture, undeclared_prediction_is_cancelled) {
    FakeService service;
    service.start_response["call"]["name"] = "delete_file";
    auto attempt = begin(service);

    CHECK_EQUAL(size_t{2}, service.requests.size());
    CHECK(service.requests.back()["operation"] == "cancel");
    const json metadata = attempt->finish({weather_call()});
    CHECK(metadata["status"] == "deferred");
    CHECK(metadata["reason"].get<std::string>().find("allowlisted") !=
          std::string::npos);
    CHECK(!metadata.contains("result"));
}

TEST_CASE(ToolSpeculationFixture, low_confidence_prediction_is_cancelled) {
    FakeService service;
    service.start_response["confidence"] = 0.74;
    auto attempt = begin(service);

    CHECK_EQUAL(size_t{2}, service.requests.size());
    CHECK(service.requests.back()["operation"] == "cancel");
    const json metadata = attempt->finish({weather_call()});
    CHECK(metadata["status"] == "deferred");
    CHECK(metadata["reason"] == "prediction below minimum confidence");
}

TEST_CASE(ToolSpeculationFixture, provider_failure_does_not_touch_generation) {
    FakeService service;
    service.failing_operation = "start";
    auto attempt = begin(service);

    const json metadata = attempt->finish({weather_call()});
    CHECK(metadata["status"] == "deferred");
    CHECK(metadata["reason"].get<std::string>().find("unavailable") !=
          std::string::npos);
    CHECK(!metadata.contains("result"));
}

TEST_CASE(ToolSpeculationFixture, requests_without_allowlisted_tools_are_skipped) {
    FakeService service;
    auto attempt = ToolSpeculationAttempt::begin(
        config(), "request-1", json::array(),
        json::array({function_tool("delete_file")}), "auto",
        service.transport());

    CHECK_NULL(attempt.get());
    CHECK(service.requests.empty());

    attempt = ToolSpeculationAttempt::begin(
        config(), "request-2", json::array(),
        json::array({function_tool("weather")}), "none",
        service.transport());
    CHECK_NULL(attempt.get());
    CHECK(service.requests.empty());
}

TEST_CASE(ToolSpeculationFixture, multiple_authoritative_calls_are_a_miss) {
    FakeService service;
    auto attempt = begin(service);
    const json metadata = attempt->finish({weather_call(), weather_call()});

    CHECK(metadata["status"] == "miss");
    CHECK(metadata["reason"] == "authoritative_call_count");
    CHECK(!metadata.contains("result"));
    CHECK(service.requests.back()["operation"] == "cancel");
}

TEST_CASE(ToolSpeculationFixture, streaming_metadata_preserves_api_shape) {
    const json metadata = {{"status", "hit"}};
    const std::string openai = render_tool_speculation_sse(
        ApiFormat::OPENAI_CHAT, "request-1", "target", metadata);
    const std::string anthropic = render_tool_speculation_sse(
        ApiFormat::ANTHROPIC, "request-1", "target", metadata);
    const std::string responses = render_tool_speculation_sse(
        ApiFormat::RESPONSES, "request-1", "target", metadata);

    CHECK(openai.find("chat.completion.chunk") != std::string::npos);
    CHECK(anthropic.find("event: dflash_tool_speculation") == 0);
    CHECK(responses.find("event: response.dflash_tool_speculation") == 0);
}
