#include "CppUnitTestFramework.hpp"

#include "common/qwen3_tool_predictor_ipc.h"
#include "server/semantic_tool_hint.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {
struct SemanticToolHintFixture {};
}

using namespace dflash::common;

static json weather_tools() {
    return json::array({{
        {"type", "function"},
        {"function", {
            {"name", "get_weather"},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"city", {{"type", "string"}}},
                    {"unit", {{"type", "string"}}},
                }},
            }},
        }},
    }});
}

TEST_CASE(SemanticToolHintFixture, parses_qwen_openai_tool_call_semantics) {
    const json response = {
        {"choices", json::array({{
            {"message", {
                {"role", "assistant"},
                {"content", ""},
                {"tool_calls", json::array({{
                    {"type", "function"},
                    {"function", {
                        {"name", "get_weather"},
                        {"arguments", "{\"city\":\"Rome\",\"unit\":\"celsius\"}"},
                    }},
                }})},
            }},
        }})},
    };
    SemanticToolCall call;
    std::string error;
    CHECK(parse_semantic_tool_prediction(
        response, weather_tools(), call, error));
    CHECK(error.empty());
    CHECK(call.name == "get_weather");
    CHECK(call.arguments.dump() ==
          "{\"city\":\"Rome\",\"unit\":\"celsius\"}");
}

TEST_CASE(SemanticToolHintFixture, rejects_unknown_predicted_function) {
    const json response = {
        {"choices", json::array({{
            {"message", {
                {"tool_calls", json::array({{
                    {"function", {
                        {"name", "delete_everything"},
                        {"arguments", "{}"},
                    }},
                }})},
            }},
        }})},
    };
    SemanticToolCall call;
    std::string error;
    CHECK(!parse_semantic_tool_prediction(
        response, weather_tools(), call, error));
    CHECK(error == "predictor_selected_unknown_function");
}

TEST_CASE(SemanticToolHintFixture, materializes_declared_optional_defaults) {
    json tools = weather_tools();
    tools[0]["function"]["parameters"]["properties"]["unit"]["default"] =
        "celsius";
    SemanticToolCall call;
    call.name = "get_weather";
    call.arguments = ordered_json::parse(R"({"city":"Rome"})");
    std::string error;

    CHECK(materialize_declared_tool_defaults(tools, call, error));
    CHECK(error.empty());
    CHECK(call.arguments.dump() ==
          R"({"city":"Rome","unit":"celsius"})");
}

TEST_CASE(SemanticToolHintFixture, explicit_prediction_beats_schema_default) {
    json tools = weather_tools();
    tools[0]["function"]["parameters"]["properties"]["unit"]["default"] =
        "celsius";
    SemanticToolCall call;
    call.name = "get_weather";
    call.arguments = ordered_json::parse(
        R"({"city":"Rome","unit":"fahrenheit"})");
    std::string error;

    CHECK(materialize_declared_tool_defaults(tools, call, error));
    CHECK(call.arguments["unit"] == "fahrenheit");
}

TEST_CASE(SemanticToolHintFixture, predictor_request_forwards_only_semantics) {
    const json target = {
        {"model", "deepseek-v4-flash"},
        {"messages", json::array({{{"role", "user"}, {"content", "weather"}}})},
        {"tools", weather_tools()},
        {"tool_choice", "required"},
        {"tool_speculation", {{"name", "unsafe"}}},
        {"prefix_cache", {{"scope", "full"}}},
    };
    const json request = build_semantic_tool_predictor_request(
        target, "Qwen3-0.6B", 32);
    CHECK(request["model"] == "Qwen3-0.6B");
    CHECK(request["max_tokens"] == 32);
    CHECK(request["tool_choice"] == "required");
    CHECK(!request.contains("tool_speculation"));
    CHECK(!request.contains("prefix_cache"));
}

TEST_CASE(SemanticToolHintFixture, native_predictor_config_is_independent_of_http) {
    SemanticToolPredictorConfig config;
    config.native_model_path = "/models/qwen3-0.6b.gguf";
    config.native_ipc_bin = "/opt/lucebox/backend_ipc_daemon";
    CHECK(config.native_enabled());
    CHECK(!config.http_enabled());
    CHECK(config.enabled());
    CHECK(config.native_runs_before_model());
    CHECK(std::string(native_tool_predictor_schedule_name(
              config.native_schedule)) == "before-model");
}

TEST_CASE(SemanticToolHintFixture, native_predictor_overlap_is_explicit) {
    NativeToolPredictorSchedule schedule =
        NativeToolPredictorSchedule::BeforeModel;
    CHECK(parse_native_tool_predictor_schedule("overlap", schedule));
    CHECK(schedule == NativeToolPredictorSchedule::Overlap);
    CHECK(std::string(native_tool_predictor_schedule_name(schedule)) ==
          "overlap");
    CHECK(!parse_native_tool_predictor_schedule("automatic", schedule));
}

TEST_CASE(SemanticToolHintFixture, native_prompt_uses_qwen_tool_contract) {
    const json request = {
        {"messages", json::array({{
            {"role", "user"},
            {"content", "What is the weather in Rome?"},
        }})},
        {"tools", weather_tools()},
        {"tool_choice", "required"},
    };
    std::string error;
    const std::string prompt =
        build_native_semantic_tool_predictor_prompt(request, error);
    CHECK(error.empty());
    CHECK(prompt.find("You must call exactly one available function.") !=
          std::string::npos);
    CHECK(prompt.find("get_weather") != std::string::npos);
    CHECK(prompt.find("What is the weather in Rome?") != std::string::npos);
    CHECK(prompt.find("{\"name\": <function-name>, \"arguments\":") !=
          std::string::npos);
    CHECK(prompt.find("<function=example_function_name>") ==
          std::string::npos);
    CHECK(prompt.find("<think>\n\n</think>") != std::string::npos);
}

TEST_CASE(SemanticToolHintFixture, parses_native_qwen_xml_semantics) {
    const std::string generated =
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\nRome\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n"
        "</function>\n"
        "</tool_call>";
    SemanticToolCall call;
    std::string error;
    CHECK(parse_native_semantic_tool_prediction(
        generated, weather_tools(), call, error));
    CHECK(error.empty());
    CHECK(call.name == "get_weather");
    CHECK(call.arguments.dump() ==
          "{\"city\":\"Rome\",\"unit\":\"celsius\"}");
}

TEST_CASE(SemanticToolHintFixture, repairs_qwen_missing_outer_call_brace) {
    const json tools = json::array({{
        {"type", "function"},
        {"function", {
            {"name", "get_stock_quote"},
            {"parameters", {
                {"type", "object"},
                {"properties", {{"symbol", {{"type", "string"}}}}},
                {"required", json::array({"symbol"})},
            }},
        }},
    }});
    const std::string generated =
        "<tool_call>\n"
        "  \"name\": \"get_stock_quote\",\n"
        "  \"arguments\": {\"symbol\": \"NVDA\"}\n"
        "\"}<|im_end|>";
    SemanticToolCall call;
    std::string error;
    CHECK(parse_native_semantic_tool_prediction(
        generated, tools, call, error));
    CHECK(error.empty());
    CHECK(call.name == "get_stock_quote");
    CHECK(call.arguments.dump() == "{\"symbol\":\"NVDA\"}");
}

TEST_CASE(SemanticToolHintFixture, maps_bare_arguments_only_for_one_tool) {
    const json one_tool = json::array({{
        {"name", "benchmark_cpu_sparse"},
        {"parameters", {
            {"type", "object"},
            {"properties", {{"iterations", {{"type", "integer"}}}}},
            {"required", json::array({"iterations"})},
        }},
    }});
    SemanticToolCall call;
    std::string error;
    CHECK(parse_native_semantic_tool_prediction(
        "<tool_call>\n{\"iterations\":172452}\n</tool_call>",
        one_tool, call, error));
    CHECK(error.empty());
    CHECK(call.name == "benchmark_cpu_sparse");
    CHECK(call.arguments.dump() == "{\"iterations\":172452}");

    json two_tools = one_tool;
    two_tools.push_back({
        {"name", "other"},
        {"parameters", {{"type", "object"}}},
    });
    CHECK(!parse_native_semantic_tool_prediction(
        "{\"iterations\":172452}", two_tools, call, error));
}

TEST_CASE(SemanticToolHintFixture, native_parser_rejects_multiple_calls) {
    const std::string call =
        "<function=get_weather><parameter=city>Rome</parameter>"
        "<parameter=unit>celsius</parameter></function>";
    SemanticToolCall prediction;
    std::string error;
    CHECK(!parse_native_semantic_tool_prediction(
        call + call, weather_tools(), prediction, error));
    CHECK(error == "native_predictor_response_has_multiple_calls");
}

TEST_CASE(SemanticToolHintFixture, native_ipc_response_obeys_hard_deadline) {
#if !defined(_WIN32)
    int descriptors[2] = {-1, -1};
    CHECK(::pipe(descriptors) == 0);
    const auto started = std::chrono::steady_clock::now();
    std::vector<int32_t> output;
    std::string error;
    CHECK(!read_qwen3_tool_predictor_response(
        descriptors[0], 8, 25, output, error));
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ::close(descriptors[0]);
    ::close(descriptors[1]);
    CHECK(error == "native_predictor_timeout");
    CHECK(output.empty());
    CHECK(elapsed_ms < 500.0);
#else
    CHECK(true);
#endif
}

TEST_CASE(SemanticToolHintFixture, native_ipc_response_reads_complete_payload) {
#if !defined(_WIN32)
    int descriptors[2] = {-1, -1};
    CHECK(::pipe(descriptors) == 0);
    const int32_t response[] = {0, 3, 17, 18, 19};
    CHECK(::write(descriptors[1], response, sizeof(response)) ==
          static_cast<ssize_t>(sizeof(response)));
    ::close(descriptors[1]);
    std::vector<int32_t> output;
    std::string error;
    CHECK(read_qwen3_tool_predictor_response(
        descriptors[0], 8, 100, output, error));
    ::close(descriptors[0]);
    CHECK(error.empty());
    CHECK(output == std::vector<int32_t>({17, 18, 19}));
#else
    CHECK(true);
#endif
}
