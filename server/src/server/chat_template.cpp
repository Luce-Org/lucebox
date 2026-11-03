// Chat template renderer implementation.

#include "chat_template.h"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace luce::common {

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

nlohmann::ordered_json openai_tool_definitions(nlohmann::ordered_json tools) {
    if (!tools.is_array()) return tools;
    for (auto & tool : tools) {
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
    return tools;
}

std::string qwen4exp_template_effort(const std::string & effort) {
    if (effort.empty() || effort == "low" || effort == "medium") return effort;
    return "xhigh";
}

// Python's repr(float), which json.dumps writes: the shortest digits that
// read back, in positional notation for decimal exponents -4 < e <= 16, else
// as d.ddde+XX; whole numbers keep ".0".
static std::string python_float_repr(double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value < 0 ? "-Infinity" : "Infinity";
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
    const std::string sci(buf, res.ptr);  // e.g. "-1.25e+15"
    const size_t e = sci.find('e');
    std::string mantissa = sci.substr(0, e);
    const int exponent = std::stoi(sci.substr(e + 1));
    std::string sign;
    if (!mantissa.empty() && mantissa[0] == '-') {
        sign = "-";
        mantissa.erase(0, 1);
    }
    std::string digits;
    for (char c : mantissa) {
        if (c != '.') digits += c;
    }
    const int decpt = exponent + 1;  // digits[0] sits just before this position
    if (decpt <= -4 || decpt > 16) {
        std::string out = sign + digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        const int shown = decpt - 1;
        out += shown < 0 ? "e-" : "e+";
        const std::string magnitude = std::to_string(shown < 0 ? -shown : shown);
        out += (magnitude.size() < 2 ? "0" : "") + magnitude;
        return out;
    }
    std::string out = sign;
    if (decpt <= 0) {
        out += "0." + std::string((size_t) -decpt, '0') + digits;
    } else if ((size_t) decpt >= digits.size()) {
        out += digits + std::string((size_t) decpt - digits.size(), '0') + ".0";
    } else {
        out += digits.substr(0, (size_t) decpt) + "." + digits.substr((size_t) decpt);
    }
    return out;
}

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
    } else if (value.is_number_float()) {
        out += python_float_repr(value.template get<double>());
    } else {
        out += value.dump();
    }
}

std::string python_json_dumps(const nlohmann::ordered_json & value) {
    std::string out;
    append_python_json(out, value);
    return out;
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
        // The built-in renderers have always listed arguments by name.
        const nlohmann::json arguments = call.arguments;
        if (i != 0 || !msg.content.empty()) out += '\n';
        // Ling and Laguna's shipped templates put the first key directly
        // after the name, unlike the illustrative preamble.
        out += "<tool_call>";
        out += call.name;
        for (auto it = arguments.begin(); it != arguments.end(); ++it) {
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
        append_gemma_value(out, nlohmann::json(call.arguments));
        out += "<tool_call|>";
    }
}

std::optional<ChatFormat> chat_format_for_arch(const std::string & arch) {
    if (arch == "laguna") return ChatFormat::LAGUNA;
    if (arch == "gemma4") return ChatFormat::GEMMA4;
    if (arch == "bailingmoe3") return ChatFormat::BAILINGMOE3;
    // The Qwen and DeepSeek families render with their chat codec (or a --chat-template-file).
    return std::nullopt;
}

std::string render_chat_template(
    const std::vector<ChatMessage> & messages,
    ChatFormat format,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::string & tools_json)
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
            inputs["tools"] = openai_tool_definitions(nlohmann::ordered_json::parse(tools_json));
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
