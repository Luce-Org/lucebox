// Tool call parser — extracts structured tool calls from generated text.
//
// Supports 11 detection patterns:
//   1. <tool_call><function=name>...</function></tool_call>  (Qwen XML)
//   2. <function=name>...</function>                          (bare function XML)
//   3. <function=name(k="v")></function>                      (function signature)
//   4. <name><parameter=k>v</parameter></function>            (bare tool tag)
//   5. <parameter name="name">...</function>                  (attribute XML)
//   6. <funcname>name<parameter=k>v</parameter></function>    (malformed XML)
//   7. <function name><parameter=k>v</parameter></function>   (malformed XML)
//   8. <tool_code>{...JSON...}</tool_code>                    (tool_code wrapper)
//   9. call:<ns>?<verb>{relaxed-JSON-args}                    (gemma plain-text)
//  10. Bare JSON objects  {"name":..., "arguments":...}       (raw JSON)
//  11. Whole-response JSON args for one declared tool          {"arg":...}

#pragma once

#include <nlohmann/json.hpp>
#include <cstddef>
#include <string>
#include <vector>
#include <utility>

namespace luce::common {

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

// Find the first supported tool-call opener in streamed text. In addition to
// the built-in compatibility forms, this recognizes `<DECLARED_TOOL>` so the
// streaming emitter and final parser agree on bare tool-name syntax.
bool find_tool_syntax_start(const std::string & text, const json & tools,
                            size_t & pos);

// The literal openers find_tool_syntax_start recognizes (declared tools'
// `<NAME>` included): text starting with one of them is a tool call.
std::vector<std::string> tool_call_openers(const json & tools);

// Number of trailing bytes the streaming emitter must retain to recognize an
// opener split across token boundaries.
size_t tool_syntax_holdback(const json & tools);

// A fresh tool-call ID ("call_" + 24 hex digits).
std::string new_tool_call_id();

// The parameter schemas (`properties`) of declared tool `name`; {} if none.
json tool_parameter_properties(const json & tools, const std::string & name);

// Whether a tool-call value quotes a call that is still open. A value is raw
// text in the XML-like formats and may hold a whole call (a command that
// writes a test of a tool-call parser): a close tag inside that quoted call is
// text. Counts `open` and `close` tags from `from` up to a point that only
// moves forward, so a scan over a value's candidate ends stays one pass.
class QuotedCall {
public:
    QuotedCall(const std::string & text, size_t from, std::string open, std::string close);
    // True when text[from, to) holds more `open` than `close` tags. `to` must
    // not go back from one call to the next.
    bool open_before(size_t to);

private:
    const std::string & text_;
    std::string open_, close_;
    size_t next_open_, next_close_;
    long depth_ = 0;
};

// Parse tool calls from generated text. `tools` is the tool definitions
// (used for type coercion and allow-list filtering). May be null/empty.
ToolParseResult parse_tool_calls(const std::string & text,
                                 const json & tools = json());

}  // namespace luce::common
