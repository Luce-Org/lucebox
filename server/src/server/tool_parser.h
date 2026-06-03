// Tool call parser — extracts structured tool calls from generated text.
//
// Supports 6 detection patterns:
//   1. <tool_call><function=name>...</function></tool_call>  (Qwen XML)
//   2. <function=name>...</function>                          (bare function XML)
//   3. <function=name(k="v")></function>                      (function signature)
//   4. <tool_code>{...JSON...}</tool_code>                    (tool_code wrapper)
//   5. call:<ns>?<verb>{relaxed-JSON-args}                    (gemma plain-text)
//   6. Bare JSON objects  {"name":..., "arguments":...}       (raw JSON)

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <utility>

namespace dflash::common {

using json = nlohmann::json;

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON string
};

struct ToolParseResult {
    std::string cleaned_text;
    std::vector<ToolCall> tool_calls;
};

// Parse tool calls from generated text. `tools` is the tool definitions
// (used for type coercion and allow-list filtering). May be null/empty.
ToolParseResult parse_tool_calls(const std::string & text,
                                 const json & tools = json());

}  // namespace dflash::common
