// Chat template renderer implementation.

#include "chat_template.h"
#include "common/model_capabilities.h"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace luce::common {

// Qwen3.5 tool preamble — matches the official Jinja template exactly.
static const char QWEN3_TOOL_PREAMBLE[] =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";

static const char QWEN3_TOOL_SUFFIX[] =
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

// Appends "<available_tools>\n{tool}\n...</available_tools>\n\n" to result.
// Each tool is pretty-printed as compact JSON; falls back to the raw
// tools_json string if parsing fails.
static void append_available_tools(std::string & result,
                                   const std::string & tools_json) {
    result += "<available_tools>\n";
    try {
        const nlohmann::json tools = nlohmann::json::parse(tools_json);
        for (const auto & t : tools) {
            result += t.dump();
            result += "\n";
        }
    } catch (const std::exception &) {
        result += tools_json;
        result += "\n";
    }
    result += "</available_tools>\n\n";
}

std::string qwen4exp_template_effort(const std::string & effort) {
    if (effort.empty() || effort == "low" || effort == "medium") return effort;
    return "xhigh";
}

// DeepSeek V4.1 tool block, as encoding/encoding.py in the model repo renders
// it: leading-space DSML tag names and one schema per line.
static const char DS41_TOOLS_HEADER[] =
    "## Tools\n\n"
    "You have access to a set of tools to help answer the user's question. "
    "You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n"
    "<｜DSML｜ calls>\n"
    "<｜DSML｜ invoke name=\"$TOOL_NAME\">\n"
    "<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n"
    "...\n"
    "</｜DSML｜ invoke>\n"
    "<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls>\n\n"
    "String parameters should be specified as is and set `string=\"true\"`. "
    "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
    "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning "
    "inside <think>...</think> BEFORE any tool calls or final response.\n\n"
    "Otherwise, output directly after </think> with tool calls or final response.\n\n"
    "### Available Tool Schemas\n\n";

// Python json.dumps(value, ensure_ascii=False) spelling: ", " and ": "
// separators, keys in their original order.
template <typename Json>
static void append_python_json(std::string & out, const Json & value) {
    if (value.is_object()) {
        out += '{';
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            out += nlohmann::ordered_json(it.key()).dump();
            out += ": ";
            append_python_json(out, it.value());
        }
        out += '}';
    } else if (value.is_array()) {
        out += '[';
        bool first = true;
        for (const auto & item : value) {
            if (!first) out += ", ";
            first = false;
            append_python_json(out, item);
        }
        out += ']';
    } else {
        out += value.dump();
    }
}

static void append_ds41_tools(std::string & result, const std::string & tools_json) {
    result += DS41_TOOLS_HEADER;
    try {
        const auto tools = nlohmann::ordered_json::parse(tools_json);
        bool first = true;
        for (const auto & tool : tools) {
            if (!first) result += "\n";
            first = false;
            append_python_json(result, tool.contains("function") ? tool["function"] : tool);
        }
    } catch (const std::exception &) {
        result += tools_json;
    }
    result += "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
}

// V4.1 reasoning budget: 1-100, with the named levels of the model's encoder.
static int ds41_reasoning_budget(const std::string & effort) {
    if (effort == "low") return 50;
    if (effort == "medium") return 62;
    if (effort == "max") return 100;
    if (!effort.empty() && effort.find_first_not_of("0123456789") == std::string::npos && effort.size() <= 3) {
        const int budget = std::stoi(effort);
        if (budget >= 1 && budget <= 100) return budget;
    }
    return 75;
}

// The XML-like formats deliberately leave string arguments unescaped. JSON
// values retain their types, including nested objects, arrays, booleans and null.
static void append_tool_argument(std::string & out, const nlohmann::json & value) {
    if (value.is_string()) out += value.get_ref<const std::string &>();
    else append_python_json(out, value);
}

static void append_xml_tool_calls(std::string & out, const ChatMessage & msg,
                                  ChatFormat format) {
    if (msg.tool_calls_replayed) return;
    for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
        const auto & call = msg.tool_calls[i];
        if (i != 0 || !msg.content.empty()) {
            out += format == ChatFormat::QWEN3 && i == 0 ? "\n\n" : "\n";
        }
        if (format == ChatFormat::QWEN3) {
            out += "<tool_call>\n<function=";
            out += call.name;
            out += ">\n";
            for (auto it = call.arguments.begin(); it != call.arguments.end(); ++it) {
                out += "<parameter=";
                out += it.key();
                out += ">\n";
                append_tool_argument(out, it.value());
                out += "\n</parameter>\n";
            }
            out += "</function>\n</tool_call>";
        } else {
            // Ling and Laguna's shipped templates put the first key directly
            // after the name, unlike the illustrative preamble.
            out += "<tool_call>";
            out += call.name;
            for (auto it = call.arguments.begin(); it != call.arguments.end(); ++it) {
                out += "<arg_key>";
                out += it.key();
                out += "</arg_key>\n<arg_value>";
                append_tool_argument(out, it.value());
                out += "</arg_value>";
                if (format == ChatFormat::LAGUNA) out += '\n';
            }
            if (format == ChatFormat::BAILINGMOE3) out += '\n';
            out += "</tool_call>";
        }
    }
}

static void append_dsml_tool_calls(std::string & out, const ChatMessage & msg,
                                   bool v41) {
    if (msg.tool_calls_replayed || msg.tool_calls.empty()) return;
    out += v41 ? "\n\n<｜DSML｜ calls>\n" : "\n\n<｜DSML｜tool_calls>\n";
    for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
        if (i != 0) out += '\n';
        const auto & call = msg.tool_calls[i];
        out += v41 ? "<｜DSML｜ invoke name=\"" : "<｜DSML｜invoke name=\"";
        out += call.name;
        out += "\">\n";
        bool first = true;
        for (auto it = call.arguments.begin(); it != call.arguments.end(); ++it) {
            if (!first) out += '\n';
            first = false;
            out += v41 ? "<｜DSML｜ parameter name=\"" : "<｜DSML｜parameter name=\"";
            out += it.key();
            out += it.value().is_string() ? "\" string=\"true\">" : "\" string=\"false\">";
            append_tool_argument(out, it.value());
            out += v41 ? "</｜DSML｜ parameter>" : "</｜DSML｜parameter>";
        }
        out += v41 ? "\n</｜DSML｜ invoke>" : "\n</｜DSML｜invoke>";
    }
    out += v41 ? "\n</｜DSML｜ calls>" : "\n</｜DSML｜tool_calls>";
}

// DeepSeek results carry no IDs in the prompt. Match the reference encoder:
// sort only tool-result slots within each user/tool run, leaving user text
// in place. Return an empty permutation for already ordered histories.
static std::vector<size_t> dsml_message_order(const std::vector<ChatMessage> & messages) {
    std::vector<size_t> order;
    const std::vector<ChatToolCall> * calls = nullptr;
    const auto is_result = [](const ChatMessage & msg) {
        return msg.role == "tool" || msg.role == "function";
    };
    for (size_t begin = 0; begin < messages.size();) {
        const auto & msg = messages[begin];
        if (msg.role == "assistant" && !msg.tool_calls.empty()) calls = &msg.tool_calls;
        if (!calls || (msg.role != "user" && !is_result(msg))) {
            ++begin;
            continue;
        }
        const auto rank = [&](size_t index) {
            const auto & id = messages[index].tool_call_id;
            if (!id.empty()) {
                for (size_t i = 0; i < calls->size(); ++i) {
                    if ((*calls)[i].id == id) return i;
                }
            }
            return size_t{0};  // Reference fallback for absent/unknown IDs.
        };
        size_t end = begin;
        size_t previous = 0;
        bool sorted = true;
        for (; end < messages.size(); ++end) {
            if (messages[end].role != "user" && !is_result(messages[end])) break;
            if (!is_result(messages[end])) continue;
            const size_t current = rank(end);
            if (current < previous) sorted = false;
            previous = current;
        }
        if (!sorted) {
            if (order.empty()) {
                order.resize(messages.size());
                std::iota(order.begin(), order.end(), size_t{0});
            }
            // Sorting (rank, original index) keeps equal-rank results stable
            // without copying message bodies or allocating a sort buffer.
            std::vector<std::pair<size_t, size_t>> results;
            results.reserve(end - begin);
            for (size_t i = begin; i < end; ++i) {
                if (is_result(messages[i])) results.emplace_back(rank(i), i);
            }
            std::sort(results.begin(), results.end());
            size_t next = 0;
            for (size_t i = begin; i < end; ++i) {
                if (is_result(messages[i])) order[i] = results[next++].second;
            }
        }
        begin = end;
    }
    return order;
}

// Google's canonical template (google/gemma-4-26B-A4B-it/chat_template.jinja)
// uses special string delimiters and bare keys recursively for calls and
// responses, not JSON or Python repr().
static void append_gemma_string(std::string & out, const std::string & value,
                                bool uppercase = false) {
    out += "<|\"|>";
    if (uppercase) {
        for (char c : value) out += c >= 'a' && c <= 'z' ? char(c - 'a' + 'A') : c;
    } else {
        out += value;
    }
    out += "<|\"|>";
}

static void append_gemma_value(std::string & out, const nlohmann::json & value) {
    if (value.is_string()) {
        append_gemma_string(out, value.get_ref<const std::string &>());
    } else if (value.is_object()) {
        out += '{';
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first) out += ',';
            first = false;
            out += it.key();
            out += ':';
            append_gemma_value(out, it.value());
        }
        out += '}';
    } else if (value.is_array()) {
        out += '[';
        bool first = true;
        for (const auto & item : value) {
            if (!first) out += ',';
            first = false;
            append_gemma_value(out, item);
        }
        out += ']';
    } else {
        out += value.dump();
    }
}

// Schema type names are uppercase; literal enum/default strings are not.
// Property maps are separate from schemas so a property named "type" remains
// a normal property. Recurse into object and array schemas without flattening.
static void append_gemma_schema(std::string & out, const nlohmann::json & schema,
                                 bool property_map = false) {
    if (!schema.is_object()) {
        append_gemma_value(out, schema);
        return;
    }
    out += '{';
    bool first = true;
    for (auto it = schema.begin(); it != schema.end(); ++it) {
        if (!first) out += ',';
        first = false;
        out += it.key();
        out += ':';
        const auto & value = it.value();
        if (property_map) {
            append_gemma_schema(out, value);
        } else if (it.key() == "type" && value.is_string()) {
            append_gemma_string(out, value.get_ref<const std::string &>(), true);
        } else if (it.key() == "type" && value.is_array()) {
            out += '[';
            for (size_t i = 0; i < value.size(); ++i) {
                if (i != 0) out += ',';
                if (value[i].is_string()) {
                    append_gemma_string(out, value[i].get_ref<const std::string &>(), true);
                } else {
                    append_gemma_value(out, value[i]);
                }
            }
            out += ']';
        } else if (it.key() == "properties" || it.key() == "$defs") {
            append_gemma_schema(out, value, true);
        } else if (it.key() == "items" || it.key() == "additionalProperties") {
            append_gemma_schema(out, value);
        } else if ((it.key() == "anyOf" || it.key() == "oneOf" || it.key() == "allOf") &&
                   value.is_array()) {
            out += '[';
            for (size_t i = 0; i < value.size(); ++i) {
                if (i != 0) out += ',';
                append_gemma_schema(out, value[i]);
            }
            out += ']';
        } else {
            append_gemma_value(out, value);
        }
    }
    out += '}';
}

static void append_gemma_tools(std::string & out, const std::string & tools_json) {
    const auto tools = nlohmann::json::parse(tools_json);
    for (const auto & tool : tools) {
        const auto & function = tool.contains("function") ? tool["function"] : tool;
        out += "<|tool>declaration:";
        out += function.at("name").get_ref<const std::string &>();
        out += "{description:";
        const auto description = function.find("description");
        if (description != function.end() && description->is_string()) {
            append_gemma_string(out, description->get_ref<const std::string &>());
        } else {
            out += "<|\"|><|\"|>";
        }
        const auto parameters = function.find("parameters");
        const auto input_schema = function.find("input_schema");
        if (parameters != function.end() && parameters->is_object()) {
            out += ",parameters:";
            append_gemma_schema(out, *parameters);
        } else if (input_schema != function.end() && input_schema->is_object()) {
            out += ",parameters:";
            append_gemma_schema(out, *input_schema);
        }
        if (function.contains("response")) {
            out += ",response:";
            append_gemma_schema(out, function["response"]);
        }
        out += "}<tool|>";
    }
}

static void append_gemma_tool_calls(std::string & out, const ChatMessage & msg) {
    if (msg.tool_calls_replayed) return;
    for (const auto & call : msg.tool_calls) {
        out += "<|tool_call>call:";
        out += call.name;
        append_gemma_value(out, call.arguments);
        out += "<tool_call|>";
    }
}

ChatFormat chat_format_for_arch(const std::string & arch) {
    if (arch == "deepseek41") return ChatFormat::DEEPSEEK41;
    if (arch_is_deepseek4_family(arch)) return ChatFormat::DEEPSEEK4;
    if (arch == "laguna") return ChatFormat::LAGUNA;
    if (arch == "gemma4") return ChatFormat::GEMMA4;
    if (arch == "bailingmoe3") return ChatFormat::BAILINGMOE3;
    // qwen35, qwen36, qwen3, qwen4exp use the Qwen3/ChatML format
    return ChatFormat::QWEN3;
}

std::string render_chat_template(
    const std::vector<ChatMessage> & messages,
    ChatFormat format,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::string & tools_json,
    const std::string & reasoning_effort)
{
    std::string result;
    bool has_tools = !tools_json.empty() && tools_json != "[]" && tools_json != "null";

    switch (format) {
    case ChatFormat::BAILINGMOE3: {
        // AntLing's shipped Bailing V3 template. The system turn is always
        // present because it carries the model's thinking-mode directive.
        const bool has_system =
            !messages.empty() && messages[0].role == "system";
        const std::string system_content = has_system
            ? messages[0].content : std::string();
        const char * thinking_option = enable_thinking ? "on" : "off";
        const std::string thinking_directive =
            std::string("detailed thinking ") + thinking_option;
        const bool system_sets_requested_thinking =
            system_content.find(thinking_directive) != std::string::npos;

        result += "<role>SYSTEM</role>";
        if (has_tools) {
            if (!system_content.empty()) {
                result += system_content;
                result += '\n';
            }
            result +=
                "# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>";
            try {
                const nlohmann::json tools = nlohmann::json::parse(tools_json);
                for (const auto & tool : tools) {
                    result += '\n';
                    result += tool.dump();
                }
            } catch (const std::exception &) {
                result += '\n';
                result += tools_json;
            }
            result +=
                "\n</tools>\n\n"
                "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n"
                "If you need to use a function, for each function call, output the function name and arguments within the following XML format:\n"
                "<tool_call>{function-name}\n"
                "<arg_key>{arg-key-1}</arg_key>\n"
                "<arg_value>{arg-value-1}</arg_value>\n"
                "<arg_key>{arg-key-2}</arg_key>\n"
                "<arg_value>{arg-value-2}</arg_value>\n"
                "...\n"
                "</tool_call>\n";
            if (!system_sets_requested_thinking) {
                result += thinking_directive;
            }
            result += "<|role_end|>";
        } else if (has_system) {
            result += system_content;
            if (!system_sets_requested_thinking) {
                result += '\n';
                result += thinking_directive;
            }
            result += "<|role_end|>";
        } else {
            result += thinking_directive;
            result += "<|role_end|>";
        }

        bool in_tool_response = false;
        for (size_t i = has_system ? 1 : 0; i < messages.size(); ++i) {
            const ChatMessage & msg = messages[i];
            if (msg.role == "user") {
                result += "<role>HUMAN</role>";
                result += msg.content;
                result += "<|role_end|>";
            } else if (msg.role == "system") {
                result += "<role>SYSTEM</role>";
                result += msg.content;
                result += "<|role_end|>";
            } else if (msg.role == "assistant") {
                result += "<role>ASSISTANT</role>\n";
                if (msg.content.find("<think>") == std::string::npos) {
                    result += "<think></think>";
                }
                result += msg.content;
                append_xml_tool_calls(result, msg, format);
                result += "<|role_end|>";
            } else if (msg.role == "tool") {
                if (!in_tool_response) {
                    result += "<role>OBSERVATION</role>";
                    in_tool_response = true;
                }
                result += "\n<tool_response>\n";
                result += msg.content;
                result += "\n</tool_response>";
                const bool next_is_tool =
                    i + 1 < messages.size() && messages[i + 1].role == "tool";
                if (!next_is_tool) {
                    result += "<|role_end|>";
                    in_tool_response = false;
                }
            }
        }

        if (add_generation_prompt) {
            result += "<role>ASSISTANT</role>\n<think>";
            if (!enable_thinking) result += "</think>";
        }
        break;
    }

    case ChatFormat::QWEN3: {
        // Qwen3/3.5 ChatML format:
        //   <|im_start|>system\n[tool preamble +] content<|im_end|>\n
        //   <|im_start|>user\nHello<|im_end|>\n
        //   <|im_start|>assistant\n...

        // Determine if the first message is a system message.
        size_t start_idx = 0;
        std::string system_content;
        if (!messages.empty() && messages[0].role == "system") {
            system_content = messages[0].content;
            start_idx = 1;
        }

        // Emit system message with tool preamble if tools are present.
        if (has_tools) {
            result += "<|im_start|>system\n";
            result += QWEN3_TOOL_PREAMBLE;
            result += '\n';
            result += tools_json;
            result += QWEN3_TOOL_SUFFIX;
            if (!system_content.empty()) {
                result += "\n\n";
                result += system_content;
            }
            result += "<|im_end|>\n";
        } else if (!system_content.empty()) {
            result += "<|im_start|>system\n";
            result += system_content;
            result += "<|im_end|>\n";
        }

        // Render remaining messages.
        bool in_tool_response = false;
        for (size_t i = start_idx; i < messages.size(); i++) {
            const auto & msg = messages[i];

            if (msg.role == "tool") {
                // Qwen3.5 template: tool responses are grouped inside a user
                // message wrapped in <tool_response> tags.
                if (!in_tool_response) {
                    result += "<|im_start|>user";
                    in_tool_response = true;
                }
                result += "\n<tool_response>\n";
                result += msg.content;
                result += "\n</tool_response>";
                // Close user block if next message is not a tool message.
                bool next_is_tool = (i + 1 < messages.size() &&
                                     messages[i + 1].role == "tool");
                if (!next_is_tool) {
                    result += "<|im_end|>\n";
                    in_tool_response = false;
                }
            } else {
                result += "<|im_start|>";
                result += msg.role;
                result += '\n';
                result += msg.content;
                if (msg.role == "assistant") append_xml_tool_calls(result, msg, format);
                result += "<|im_end|>\n";
            }
        }

        if (add_generation_prompt) {
            result += "<|im_start|>assistant\n";
            if (!enable_thinking) {
                // Qwen3 thinking disabled: inject closed think block so the
                // model skips reasoning and generates the answer directly.
                result += "<think>\n\n</think>\n\n";
            } else {
                // Qwen3.6 enable_thinking: pre-open the thinking block so the
                // model actually enters reasoning mode. Verified against the
                // official Qwen3.6 chat_template.jinja:
                //   enable_thinking=true  → suffix `assistant\n<think>\n`
                //   enable_thinking=false → suffix `assistant\n<think>\n\n</think>\n\n`
                // Without this prefix, Qwen3.6 stays in non-thinking mode
                // even when the client opts in, defeating the thinking-budget
                // mechanism entirely.
                result += "<think>\n";
            }
        }
        break;
    }

    case ChatFormat::LAGUNA: {
        // Laguna XS.2 format (verified against
        // poolside/Laguna-XS.2/chat_template.jinja, 2026-05-25):
        //
        //   〈|EOS|〉<system>
        //   {system_content_or_default}
        //   </system>
        //   <user>
        //   {user_content}
        //   </user>
        //   <assistant>
        //   <think>      ← if enable_thinking (gen prompt)
        //   </think>     ← if NOT enable_thinking (gen prompt — empty
        //                    think block; model continues with answer)
        //
        // Tokens 18/19/23/24/25/26 are special (<think>, </think>,
        // <assistant>, </assistant>, <tool_call>, </tool_call>);
        // <user> / <system> / </user> / </system> are not added-tokens
        // and tokenize as regular bytes — which is what the upstream
        // template expects.
        //
        // The DeepSeek-style template that used to live here
        // (<｜begin▁of▁sentence｜> / <｜User｜> / <｜Assistant｜>)
        // was a copy-paste error; the tokens don't exist in the laguna
        // vocab, so the model saw replacement-character garbage in its
        // prompt and degenerated into echoing the user message back
        // with `<���Assistant���>` artifacts.
        static const std::string DEFAULT_SYSTEM =
            "You are a helpful, conversationally-fluent assistant "
            "made by Poolside. You are here to be helpful to users "
            "through natural language conversations.";
        result = "〈|EOS|〉";

        const bool has_system =
            !messages.empty() && messages[0].role == "system";
        const std::string system_content = has_system
            ? messages[0].content
            : DEFAULT_SYSTEM;
        // System always emitted (default or supplied) when there's any
        // content or tools — matches the template's
        // `if (system_message and system_message.strip()) or tools`.
        if (!system_content.empty() || has_tools) {
            result += "<system>\n";
            if (!system_content.empty()) result += system_content;
            if (has_tools) {
                // Tools block per the upstream template: each tool schema as
                // raw JSON inside <available_tools>, then the calling
                // instruction with the <tool_call>/<arg_key>/<arg_value>
                // example (thinking and non-thinking variants).
                result += "\n\n### Tools\n\n"
                          "You may call functions to assist with the user query.\n"
                          "All available function signatures are listed below:\n";
                append_available_tools(result, tools_json);
                if (enable_thinking) {
                    result += "Wrap your thinking in '<think>', '</think>' tags, "
                              "followed by a function call. For each function call, "
                              "return an unescaped XML-like object with function name "
                              "and arguments within '<tool_call>' and '</tool_call>' "
                              "tags, like here:\n"
                              "<think> your thoughts here </think>\n"
                              "<tool_call>function-name\n"
                              "<arg_key>argument-key</arg_key>\n"
                              "<arg_value>value-of-argument-key</arg_value>\n"
                              "</tool_call>";
                } else {
                    result += "For each function call, return an unescaped XML-like "
                              "object with function name and arguments within "
                              "'<tool_call>' and '</tool_call>' tags, like here:\n"
                              "<tool_call>function-name\n"
                              "<arg_key>argument-key</arg_key>\n"
                              "<arg_value>value-of-argument-key</arg_value>\n"
                              "</tool_call>";
                }
            }
            result += "\n</system>\n";
        }

        const size_t start_idx = has_system ? 1 : 0;
        for (size_t i = start_idx; i < messages.size(); i++) {
            const auto & msg = messages[i];
            if (msg.role == "user") {
                result += "<user>\n";
                result += msg.content;
                result += "\n</user>\n";
            } else if (msg.role == "assistant") {
                // Past assistant turns: wrap in <assistant>...</assistant>.
                // Reasoning content is extracted by the upstream template;
                // for our minimal renderer the content is rendered as-is
                // (including any embedded <think>...</think> blocks).
                result += "<assistant>\n";
                result += msg.content;
                append_xml_tool_calls(result, msg, format);
                result += "\n</assistant>\n";
            } else if (msg.role == "tool") {
                result += "<tool_response>\n";
                result += msg.content;
                result += "\n</tool_response>\n";
            } else if (msg.role == "system") {
                // Additional system messages beyond the first
                result += "<system>\n";
                result += msg.content;
                result += "\n</system>\n";
            }
        }
        if (add_generation_prompt) {
            result += "<assistant>\n";
            if (enable_thinking) {
                result += "<think>";
            } else {
                // Empty think block — model jumps straight to answer.
                result += "</think>";
            }
        }
        break;
    }

    case ChatFormat::GEMMA4: {
        // Gemma4 format (see the chat template embedded in the GGUF
        // metadata of google/gemma-4-26B-A4B-it):
        //
        //   <bos>
        //   <|turn>system
        //   [<|think|>\n      ← if enable_thinking]
        //   {system content}
        //   <turn|>
        //   <|turn>user
        //   {msg}<turn|>
        //   <|turn>model
        //   [<|channel>thought\n<channel|>  ← if NOT enable_thinking]
        //
        // The trailing channel-thought guard is the same trick Qwen3
        // uses (`<think>\n\n</think>\n\n`): when thinking is disabled
        // we pre-fill an empty thought channel so the model SKIPS
        // emitting its own. Without this, Gemma4 self-emits
        // `<|channel>thought\n…<channel|>` which then partially leaks
        // into the visible content because the channel tokens were
        // never opened from the prompt side.
        //
        // The `<|think|>` opener at the start of the system turn is
        // the inverse: it signals "this conversation is in thinking
        // mode" so the model's channel sequence routes to reasoning.
        const bool has_system = !messages.empty() && messages[0].role == "system";
        const bool emit_system_turn = enable_thinking || has_system || has_tools;
        result = "<bos>";

        size_t start_idx = 0;
        std::string system_content;
        if (has_system) {
            system_content = messages[0].content;
            start_idx = 1;
        }

        // System turn — emitted when there's actual system content OR
        // we need somewhere to put the `<|think|>` opener.
        if (emit_system_turn) {
            result += "<|turn>system\n";
            if (enable_thinking) {
                // Per the GGUF chat template: "Inject Thinking token at
                // the very top of the FIRST system turn".
                result += "<|think|>\n";
            }
            if (!system_content.empty()) {
                result += system_content;
            }
            if (has_tools) append_gemma_tools(result, tools_json);
            result += "<turn|>\n";
        }

        // Calls and responses are part of the SAME model turn. A following
        // assistant continues after the responses, without a second header.
        bool model_turn_open = false;
        bool response_prefix_open = false;
        for (size_t i = start_idx; i < messages.size(); ++i) {
            const auto & msg = messages[i];
            if (msg.role == "tool") {
                if (!model_turn_open) {
                    result += "<|turn>model\n";
                    model_turn_open = true;
                }
                if (!response_prefix_open) result += "<|tool_response>";
                result += "response:";
                if (msg.tool_name.empty()) result += "unknown";
                else result += msg.tool_name;
                result += "{value:";
                append_gemma_string(result, msg.content);
                result += "}<tool_response|>";
                response_prefix_open = false;
                const bool continues = i + 1 == messages.size() ||
                    messages[i + 1].role == "tool" || messages[i + 1].role == "assistant";
                if (!continues) {
                    result += "<turn|>\n";
                    model_turn_open = false;
                }
                continue;
            }

            if (msg.role != "assistant" || !model_turn_open) {
                result += "<|turn>";
                if (msg.role == "assistant") result += "model";
                else result += msg.role;
                result += '\n';
            }
            result += msg.content;
            if (msg.role == "assistant" && !msg.tool_calls.empty()) {
                append_gemma_tool_calls(result, msg);
                model_turn_open = true;
                // A raw cache hit can already contain the handoff token.
                static constexpr char response_prefix[] = "<|tool_response>";
                constexpr size_t response_prefix_size = sizeof(response_prefix) - 1;
                response_prefix_open = result.size() >= response_prefix_size &&
                    result.compare(result.size() - response_prefix_size,
                                   response_prefix_size, response_prefix) == 0;
                if (!response_prefix_open) result += response_prefix;
                response_prefix_open = true;
            } else {
                result += "<turn|>\n";
                model_turn_open = false;
                response_prefix_open = false;
            }
        }
        if (add_generation_prompt) {
            if (!model_turn_open) {
                result += "<|turn>model\n";
                if (!enable_thinking) result += "<|channel>thought\n<channel|>";
            } else if (!response_prefix_open && enable_thinking) {
                result += "<|channel>thought\n";
            }
        }
        break;
    }

    case ChatFormat::DEEPSEEK4: {
        // DeepSeek V4 Flash DSML renderer, matching the ds4 reference server:
        //   <｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜></think>
        // Completed assistant turns are terminated with <｜end▁of▁sentence｜>.
        std::string system_content;
        for (const auto & msg : messages) {
            if (msg.role != "system") continue;
            if (!system_content.empty()) system_content += "\n\n";
            system_content += msg.content;
        }

        result = "<｜begin▁of▁sentence｜>";
        if (enable_thinking && reasoning_effort == "high") {
            result += "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n";
            result += "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n";
            result += "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
        } else if (enable_thinking && reasoning_effort == "max") {
            result += "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n";
            result += "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n";
            result += "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";
        }
        if (has_tools) {
            result += "## Tools\n\n"
                      "You have access to a set of tools to help answer the user question. "
                      "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
                      "<｜DSML｜tool_calls>\n"
                      "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
                      "<｜DSML｜parameter name=\"$PARAM_STRING\" string=\"true\">string_value</｜DSML｜parameter>\n"
                      "<｜DSML｜parameter name=\"$PARAM_JSON\" string=\"false\">[1, 2, 3]</｜DSML｜parameter>\n"
                      "</｜DSML｜invoke>\n"
                      "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
                      "...\n"
                      "</｜DSML｜invoke>\n"
                      "</｜DSML｜tool_calls>\n\n"
                      "String parameters should be specified as is and set `string=\"true\"`. "
                      "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
                      "### Available Tool Schemas\n\n";
            append_available_tools(result, tools_json);
            result += "\nYou MUST strictly follow the defined tool schemas to invoke tool calls.\n\n";
        }
        result += system_content;

        bool pending_assistant = false;
        bool pending_tool_result = false;
        const auto order = dsml_message_order(messages);
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto & msg = messages[order.empty() ? i : order[i]];
            if (msg.role == "system") {
                continue;
            } else if (msg.role == "user") {
                result += "<｜User｜>";
                result += msg.content;
                pending_assistant = true;
                pending_tool_result = false;
            } else if (msg.role == "tool" || msg.role == "function") {
                result += pending_tool_result ? "\n\n" : "<｜User｜>";
                result += "<tool_result>";
                result += msg.content;
                result += "</tool_result>";
                pending_assistant = true;
                pending_tool_result = true;
            } else if (msg.role == "assistant") {
                if (pending_assistant) {
                    result += "<｜Assistant｜>";
                    result += enable_thinking ? "<think>" : "</think>";
                }
                // Cold API history contains visible prose, not raw reasoning.
                if (enable_thinking && !msg.tool_calls_replayed && !msg.tool_calls.empty()) {
                    result += "</think>";
                }
                result += msg.content;
                append_dsml_tool_calls(result, msg, false);
                // The official V4 encoder ends tool-call turns with EOS too.
                result += "<｜end▁of▁sentence｜>";
                pending_assistant = false;
                pending_tool_result = false;
            }
        }

        if (add_generation_prompt) {
            result += "<｜Assistant｜>";
            result += enable_thinking ? "<think>" : "</think>";
        }
        break;
    }

    case ChatFormat::DEEPSEEK41: {
        // DeepSeek V4.1 Flash, following encoding/encoding.py in the model repo:
        //   <｜begin▁of▁sentence｜><｜System｜>{effort}{system}\n\n{tools}<｜User｜>{user}<｜Assistant｜></think>
        // V4.1 differs from V4 in the leading-space DSML tag names (<｜DSML｜ calls>),
        // a numeric reasoning effort, and <｜System｜> on system turns, including
        // mid-conversation ones. Tool results join the user turn as
        // <tool_result> blocks separated by blank lines. Every assistant header is
        // rendered the same way in every turn, so a completed turn keeps its prefix.
        const size_t first = !messages.empty() && messages[0].role == "system" ? 1 : 0;
        result = "<｜begin▁of▁sentence｜>";
        if (enable_thinking || first == 1 || has_tools) result += "<｜System｜>";
        if (enable_thinking) {
            result += "Reasoning Effort: " + std::to_string(ds41_reasoning_budget(reasoning_effort)) +
                      " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
        }
        if (first == 1) result += messages[0].content;
        if (has_tools) {
            result += "\n\n";
            append_ds41_tools(result, tools_json);
        }

        bool in_user_turn = false;
        bool pending_assistant = false;
        const auto order = dsml_message_order(messages);
        for (size_t i = first; i < messages.size(); ++i) {
            const auto & msg = messages[order.empty() ? i : order[i]];
            if (msg.role == "system") {
                result += "<｜System｜>";
                result += msg.content;
                in_user_turn = false;
                pending_assistant = true;
            } else if (msg.role == "user" || msg.role == "tool" || msg.role == "function") {
                result += in_user_turn ? "\n\n" : "<｜User｜>";
                if (msg.role == "user") {
                    result += msg.content;
                } else {
                    result += "<tool_result>";
                    result += msg.content;
                    result += "</tool_result>";
                }
                in_user_turn = true;
                pending_assistant = true;
            } else if (msg.role == "assistant") {
                if (pending_assistant) {
                    result += "<｜Assistant｜>";
                    result += enable_thinking ? "<think>" : "</think>";
                }
                if (enable_thinking && !msg.tool_calls_replayed && !msg.tool_calls.empty()) {
                    result += "</think>";
                }
                result += msg.content;
                append_dsml_tool_calls(result, msg, true);
                // V4.1 retains EOS after its distinct, leading-space DSML tags.
                result += "<｜end▁of▁sentence｜>";
                in_user_turn = false;
                pending_assistant = false;
            }
        }

        if (add_generation_prompt) {
            result += "<｜Assistant｜>";
            result += enable_thinking ? "<think>" : "</think>";
        }
        break;
    }
    }

    return result;
}

// ─── Jinja path ─────────────────────────────────────────────────────────
//
// Render via a Jinja chat template (e.g. froggeric Qwen3.6 template). Each
// thread caches the most-recently-parsed program for its template source,
// so steady-state cost is just the runtime execute (parse happens once per
// process per template).

namespace {

struct JinjaCache {
    std::string                       src;
    std::shared_ptr<jinja::program>   prog;
};

static thread_local JinjaCache tls_jinja_cache;

static std::shared_ptr<jinja::program> get_or_parse(const std::string & template_src) {
    if (tls_jinja_cache.prog && tls_jinja_cache.src == template_src) {
        return tls_jinja_cache.prog;
    }
    jinja::lexer lex;
    jinja::lexer_result lex_res;
    try {
        lex_res = lex.tokenize(template_src);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("jinja lexer: ") + e.what());
    }
    auto prog = std::make_shared<jinja::program>(jinja::parse_from_tokens(lex_res));
    tls_jinja_cache.src  = template_src;
    tls_jinja_cache.prog = prog;
    return prog;
}

}  // namespace

static std::string render_jinja(
    const std::string & template_src,
    const std::vector<ChatMessage> & messages,
    const std::string & bos_token,
    const std::string & eos_token,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::string & tools_json,
    const std::string & reasoning_effort,
    int preserve_thinking,
    bool late_system_as_user)
{
    if (template_src.empty()) {
        throw std::runtime_error("render_chat_template_jinja: template_src is empty");
    }

    auto prog = get_or_parse(template_src);

    // Build the JSON input that mirrors llama.cpp's
    // common_chat_template_direct_apply_impl. Field names must match the
    // names the Jinja templates expect (messages, tools, bos_token,
    // eos_token, add_generation_prompt, enable_thinking).
    nlohmann::ordered_json messages_j = nlohmann::ordered_json::array();
    for (const auto & m : messages) {
        // Arbitrary templates may require tool_calls to consume following
        // results (Gemma does). Raw replay cannot safely replace that structure.
        if (m.tool_calls_replayed) {
            throw std::logic_error(
                "render_chat_template_jinja: raw tool replay is unsupported; "
                "normalize with tool-memory replay disabled");
        }
        nlohmann::ordered_json mj;
        const bool late_system = late_system_as_user && !messages_j.empty() &&
            (m.role == "system" || m.role == "developer");
        mj["role"]    = late_system ? std::string("user") : m.role;
        mj["content"] = m.content;
        if (!m.tool_call_id.empty()) {
            mj["tool_call_id"] = m.tool_call_id;
        }
        if (!m.reasoning_content.empty()) {
            mj["reasoning_content"] = m.reasoning_content;
        }
        if (!m.tool_name.empty()) {
            mj["name"] = m.tool_name;
        }
        if (!m.tool_calls.empty()) {
            auto & calls = mj["tool_calls"] = nlohmann::ordered_json::array();
            for (const auto & call : m.tool_calls) {
                calls.push_back({
                    {"id", call.id},
                    {"type", "function"},
                    {"function", {{"name", call.name}, {"arguments", call.arguments}}},
                });
            }
        }
        messages_j.push_back(std::move(mj));
    }

    nlohmann::ordered_json inputs;
    inputs["messages"]              = std::move(messages_j);
    inputs["bos_token"]             = bos_token;
    inputs["eos_token"]             = eos_token;
    inputs["add_generation_prompt"] = add_generation_prompt;
    inputs["enable_thinking"]       = enable_thinking;
    if (!reasoning_effort.empty()) inputs["reasoning_effort"] = reasoning_effort;
    // -1 = unset: leave `preserve_thinking` undefined so the template's own
    // default (official qwen4exp template defaults to true) applies.
    if (preserve_thinking >= 0) inputs["preserve_thinking"] = (preserve_thinking != 0);

    bool has_tools = !tools_json.empty() && tools_json != "[]" && tools_json != "null";
    if (has_tools) {
        try {
            inputs["tools"] = nlohmann::ordered_json::parse(tools_json);
            // Chat Completions wraps functions, Responses flattens them, and
            // Anthropic names the parameter schema input_schema. Templates
            // receive the same OpenAI-shaped function definition for all three.
            for (auto & tool : inputs["tools"]) {
                if (!tool.is_object()) continue;
                if (!tool.contains("function") && tool.contains("name")) {
                    auto function = std::move(tool);
                    function.erase("type");
                    tool = nlohmann::ordered_json::object();
                    tool["type"] = "function";
                    tool["function"] = std::move(function);
                }
                if (!tool.contains("function") || !tool["function"].is_object()) continue;
                auto & function = tool["function"];
                const auto input_schema = function.find("input_schema");
                if (input_schema != function.end()) {
                    if (!function.contains("parameters")) {
                        auto schema = std::move(*input_schema);
                        function.erase("input_schema");
                        function["parameters"] = std::move(schema);
                    } else {
                        function.erase("input_schema");
                    }
                }
            }
        } catch (const std::exception & e) {
            throw std::runtime_error(
                std::string("render_chat_template_jinja: failed to parse tools JSON: ") + e.what());
        }
    }

    jinja::context ctx(template_src);
    try {
        jinja::global_from_json(ctx, inputs, /*mark_input=*/false);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("jinja global_from_json: ") + e.what());
    }

    try {
        jinja::runtime rt(ctx);
        jinja::value results = rt.execute(*prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        return parts->as_string().str();
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("jinja runtime: ") + e.what());
    }
}

std::string render_chat_template_jinja(
    const std::string & template_src,
    const std::vector<ChatMessage> & messages,
    const std::string & bos_token,
    const std::string & eos_token,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::string & tools_json,
    const std::string & reasoning_effort,
    int preserve_thinking)
{
    try {
        return render_jinja(template_src, messages, bos_token, eos_token, add_generation_prompt,
                            enable_thinking, tools_json, reasoning_effort, preserve_thinking, false);
    } catch (const std::runtime_error &) {
        // Some templates take a system message only first (Qwen's raise).
        // Claude Code sends its environment as a system message after the
        // user's first one; for such a template it reads as a user turn
        // there, in place, so the prompt before it stays the same and cached.
        const bool late_system = messages.size() > 1 &&
            std::any_of(messages.begin() + 1, messages.end(), [](const ChatMessage & m) {
                return m.role == "system" || m.role == "developer";
            });
        if (!late_system) throw;
        return render_jinja(template_src, messages, bos_token, eos_token, add_generation_prompt,
                            enable_thinking, tools_json, reasoning_effort, preserve_thinking, true);
    }
}

}  // namespace luce::common
