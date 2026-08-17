#include "server/native_semantic_tool_predictor.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

struct Case {
    const char * id;
    const char * prompt;
    const char * expected_name;
    ordered_json expected_arguments;
};

json production_tools() {
    return json::parse(R"json(
[
  {"type":"function","function":{"name":"get_weather","description":"Get current weather for one city.","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city","unit"],"additionalProperties":false}}},
  {"type":"function","function":{"name":"get_stock_quote","description":"Get the latest market quote for a ticker symbol.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"],"additionalProperties":false}}},
  {"type":"function","function":{"name":"search_documents","description":"Search indexed documents.","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":20}},"required":["query","limit"],"additionalProperties":false}}},
  {"type":"function","function":{"name":"calculate","description":"Evaluate one arithmetic expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"],"additionalProperties":false}}},
  {"type":"function","function":{"name":"lookup_order","description":"Look up an order by its identifier.","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"],"additionalProperties":false}}},
  {"type":"function","function":{"name":"translate_text","description":"Translate text to a target language.","parameters":{"type":"object","properties":{"text":{"type":"string"},"target_language":{"type":"string"}},"required":["text","target_language"],"additionalProperties":false}}},
  {"type":"function","function":{"name":"plan_route","description":"Plan a route between two places.","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"mode":{"type":"string","enum":["car","walk","transit"]}},"required":["origin","destination","mode"],"additionalProperties":false}}},
  {"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file from the workspace.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}
]
)json");
}

json make_request(const std::string & prompt, const json & tools,
                  int32_t max_tokens) {
    return {
        {"model", "native-qwen3"},
        {"messages", json::array({{
            {"role", "user"},
            {"content", prompt},
        }})},
        {"tools", tools},
        {"tool_choice", "required"},
        {"temperature", 0},
        {"max_tokens", max_tokens},
    };
}

std::vector<Case> production_cases() {
    return {
        {"weather_rome", "What is the weather in Rome? Use Celsius.",
         "get_weather", {{"city", "Rome"}, {"unit", "celsius"}}},
        {"weather_boston", "Check Boston weather in Fahrenheit.",
         "get_weather", {{"city", "Boston"}, {"unit", "fahrenheit"}}},
        {"stock_nvda", "Get the latest quote for NVDA.",
         "get_stock_quote", {{"symbol", "NVDA"}}},
        {"stock_amd", "Look up AMD's current stock quote.",
         "get_stock_quote", {{"symbol", "AMD"}}},
        {"search_rocm",
         "Search documents for 'ROCm graph replay' and return at most 5 results.",
         "search_documents", {{"query", "ROCm graph replay"}, {"limit", 5}}},
        {"search_tool",
         "Find the top 3 documents about speculative tool execution.",
         "search_documents",
         {{"query", "speculative tool execution"}, {"limit", 3}}},
        {"calculate", "Calculate (73.5 * 4) / 7.",
         "calculate", {{"expression", "(73.5 * 4) / 7"}}},
        {"order", "Look up order LBX-2048-A.",
         "lookup_order", {{"order_id", "LBX-2048-A"}}},
        {"translate", "Translate 'the server is ready' to Italian.",
         "translate_text",
         {{"text", "the server is ready"}, {"target_language", "Italian"}}},
        {"route",
         "Plan a walking route from Termini Station to the Colosseum.",
         "plan_route",
         {{"origin", "Termini Station"},
          {"destination", "the Colosseum"},
          {"mode", "walk"}}},
        {"read_file", "Read the file docs/production.md.",
         "read_file", {{"path", "docs/production.md"}}},
        {"punctuation",
         "Search for the exact phrase 'R9700 + Strix: 0731/DS4' with limit 4.",
         "search_documents",
         {{"query", "R9700 + Strix: 0731/DS4"}, {"limit", 4}}},
    };
}

bool arguments_equal(const ordered_json & left, const ordered_json & right) {
    // Object key order is irrelevant to tool-call semantics. Convert both
    // ordered objects to the canonical map-backed representation first.
    return json::parse(left.dump()) == json::parse(right.dump());
}

void print_prediction(const char * id, const SemanticToolPrediction & prediction,
                      const char * expected_name,
                      const ordered_json & expected_arguments,
                      const std::string & generated) {
    const json output = {
        {"id", id},
        {"ok", prediction.ok},
        {"error", prediction.error},
        {"wall_ms", prediction.wall_ms},
        {"name", prediction.call.name},
        {"arguments", prediction.call.arguments},
        {"generated", generated},
        {"name_match", prediction.ok && prediction.call.name == expected_name},
        {"exact_match", prediction.ok && prediction.call.name == expected_name &&
                                      arguments_equal(prediction.call.arguments,
                                                      expected_arguments)},
    };
    std::printf("%s\n", output.dump().c_str());
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <qwen3.gguf> <backend_ipc_daemon> <gpu> [prompt]\n",
            argv[0]);
        return 2;
    }

    SemanticToolPredictorConfig config;
    config.native_model_path = argv[1];
    config.native_ipc_bin = argv[2];
    config.native_gpu = std::atoi(argv[3]);
    config.native_max_ctx = 4096;
    config.max_tokens = 96;

    std::string error;
    auto predictor = NativeSemanticToolPredictor::create(config, error);
    if (!predictor) {
        std::fprintf(stderr, "predictor start failed: %s\n", error.c_str());
        return 1;
    }

    const json tools = production_tools();
    if (argc > 4) {
        const std::string prompt = argv[4];
        std::string generated;
        const auto prediction = predictor->predict(
            make_request(prompt, tools, config.max_tokens), tools, &generated);
        print_prediction("custom", prediction, "", ordered_json::object(),
                         generated);
        return prediction.ok ? 0 : 1;
    }

    const std::vector<Case> cases = production_cases();
    size_t valid = 0;
    size_t name_matches = 0;
    size_t exact_matches = 0;
    std::vector<double> walls;
    for (const Case & test_case : cases) {
        std::string generated;
        const auto prediction = predictor->predict(
            make_request(test_case.prompt, tools, config.max_tokens), tools,
            &generated);
        print_prediction(test_case.id, prediction, test_case.expected_name,
                         test_case.expected_arguments, generated);
        valid += prediction.ok ? 1 : 0;
        name_matches += prediction.ok &&
                        prediction.call.name == test_case.expected_name ? 1 : 0;
        exact_matches += prediction.ok &&
                         prediction.call.name == test_case.expected_name &&
                         arguments_equal(prediction.call.arguments,
                                         test_case.expected_arguments)
                             ? 1 : 0;
        walls.push_back(prediction.wall_ms);
    }
    std::sort(walls.begin(), walls.end());
    const double wall_p50 = walls.empty()
        ? 0.0 : 0.5 * (walls[(walls.size() - 1) / 2] + walls[walls.size() / 2]);
    const json summary = {
        {"requests", cases.size()},
        {"valid", valid},
        {"name_matches", name_matches},
        {"exact_matches", exact_matches},
        {"name_accuracy", cases.empty() ? 0.0
                                         : static_cast<double>(name_matches) /
                                               static_cast<double>(cases.size())},
        {"exact_accuracy", cases.empty() ? 0.0
                                          : static_cast<double>(exact_matches) /
                                                static_cast<double>(cases.size())},
        {"wall_p50_ms", wall_p50},
    };
    std::printf("%s\n", summary.dump().c_str());
    return valid == cases.size() && name_matches == cases.size() ? 0 : 1;
}
