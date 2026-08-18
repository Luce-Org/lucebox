#include "semantic_tool_hint.h"

#include "tool_parser.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <utility>

namespace dflash::common {

namespace {

std::string string_member(const json & object, const char * key) {
    if (!object.is_object()) return {};
    const auto member = object.find(key);
    return member != object.end() && member->is_string()
        ? member->get<std::string>() : std::string{};
}

bool request_has_function(const json & tools, const std::string & name) {
    if (!tools.is_array() || name.empty()) return false;
    for (const auto & tool : tools) {
        if (!tool.is_object()) continue;
        if (string_member(tool, "name") == name) return true;
        const auto function = tool.find("function");
        if (function != tool.end() && function->is_object() &&
            string_member(*function, "name") == name) {
            return true;
        }
    }
    return false;
}

std::string sole_request_function(const json & tools) {
    if (!tools.is_array()) return {};
    std::string sole;
    for (const auto & tool : tools) {
        if (!tool.is_object()) continue;
        std::string name = string_member(tool, "name");
        const auto function = tool.find("function");
        if (name.empty() && function != tool.end() && function->is_object()) {
            name = string_member(*function, "name");
        }
        if (name.empty()) continue;
        if (!sole.empty() && sole != name) return {};
        sole = std::move(name);
    }
    return sole;
}

bool parse_arguments(const json & value, ordered_json & out) {
    try {
        if (value.is_string()) {
            out = ordered_json::parse(value.get<std::string>());
        } else if (value.is_object()) {
            out = ordered_json::parse(value.dump());
        } else {
            return false;
        }
    } catch (...) {
        return false;
    }
    return out.is_object();
}

bool parse_call_object(const json & value, SemanticToolCall & out) {
    if (!value.is_object()) return false;
    std::string name = string_member(value, "name");
    if (name.empty()) name = string_member(value, "function");
    if (name.empty()) return false;

    const json * arguments = nullptr;
    for (const char * key : {"arguments", "parameters", "params"}) {
        const auto it = value.find(key);
        if (it != value.end()) {
            arguments = &*it;
            break;
        }
    }
    ordered_json parsed;
    if (!arguments || !parse_arguments(*arguments, parsed)) return false;
    out.name = name;
    out.arguments = std::move(parsed);
    return true;
}

std::string trim_copy(std::string value) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch); };
    value.erase(value.begin(), std::find_if_not(
        value.begin(), value.end(), is_space));
    value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(),
                value.end());
    return value;
}

bool parse_content_call(const std::string & content, SemanticToolCall & out) {
    const json value = json::parse(trim_copy(content), nullptr, false);
    return !value.is_discarded() && parse_call_object(value, out);
}

bool parse_qwen_tagged_call_repair(
        const std::string & generated_text,
        SemanticToolCall & out) {
    const std::string open = "<tool_call>";
    const size_t open_pos = generated_text.find(open);
    if (open_pos == std::string::npos) return false;
    const size_t content_pos = open_pos + open.size();
    size_t end_pos = generated_text.find("</tool_call>", content_pos);
    if (end_pos == std::string::npos) {
        end_pos = generated_text.find("<|im_end|>", content_pos);
    }
    if (end_pos == std::string::npos) end_pos = generated_text.size();

    std::string payload = trim_copy(
        generated_text.substr(content_pos, end_pos - content_pos));
    if (payload.empty()) return false;

    // Qwen3-0.6B Q8 occasionally drops only the outer opening brace and
    // emits a stray quote before the matching closing brace, while keeping
    // the name and argument object strict JSON. Repair only that narrow
    // envelope error; all semantic fields still pass the normal schema gate.
    if (payload.front() != '{') payload.insert(payload.begin(), '{');
    if (payload.size() >= 3 && payload.back() == '}') {
        size_t quote = payload.size() - 2;
        while (quote > 0 && std::isspace(
                   static_cast<unsigned char>(payload[quote]))) {
            --quote;
        }
        if (payload[quote] == '"') payload.erase(quote, 1);
    }
    const json value = json::parse(payload, nullptr, false);
    return !value.is_discarded() && parse_call_object(value, out);
}

bool parse_qwen_bare_single_tool_arguments(
        const std::string & generated_text,
        const json & request_tools,
        SemanticToolCall & out) {
    const std::string name = sole_request_function(request_tools);
    if (name.empty()) return false;

    size_t begin = generated_text.find("<tool_call>");
    begin = begin == std::string::npos
        ? 0 : begin + std::string("<tool_call>").size();
    begin = generated_text.find('{', begin);
    if (begin == std::string::npos) return false;
    size_t end = generated_text.rfind('}');
    if (end == std::string::npos || end < begin) return false;

    const json arguments = json::parse(
        generated_text.begin() + static_cast<std::ptrdiff_t>(begin),
        generated_text.begin() + static_cast<std::ptrdiff_t>(end + 1),
        nullptr, false);
    if (arguments.is_discarded() || !arguments.is_object()) return false;
    out.name = name;
    out.arguments = ordered_json::parse(arguments.dump());
    return true;
}

bool semantic_deadline_expired(
        const std::chrono::steady_clock::time_point * deadline) {
    return deadline && std::chrono::steady_clock::now() >= *deadline;
}

constexpr size_t kMaxNativePredictorRequestBytes = 256U * 1024U;
constexpr size_t kMaxNativePredictorJsonDepth = 64U;

bool semantic_take_budget(size_t bytes, size_t & remaining) {
    if (bytes > remaining) return false;
    remaining -= bytes;
    return true;
}

bool semantic_string_within_budget(
        const std::string & value,
        const std::chrono::steady_clock::time_point * deadline,
        size_t & remaining,
        size_t & operations,
        bool & timed_out) {
    if (!semantic_take_budget(2, remaining)) return false;
    for (const unsigned char character : value) {
        if ((operations++ & 255U) == 0U &&
            semantic_deadline_expired(deadline)) {
            timed_out = true;
            return false;
        }
        const size_t encoded_bytes = character < 0x20U
            ? 6U : (character == '"' || character == '\\' ? 2U : 1U);
        if (!semantic_take_budget(encoded_bytes, remaining)) return false;
    }
    return true;
}

bool semantic_json_within_budget(
        const json & value,
        const std::chrono::steady_clock::time_point * deadline,
        size_t & remaining,
        size_t depth,
        size_t & operations,
        bool & timed_out) {
    if (depth > kMaxNativePredictorJsonDepth) return false;
    if ((operations++ & 255U) == 0U &&
        semantic_deadline_expired(deadline)) {
        timed_out = true;
        return false;
    }
    if (value.is_null()) return semantic_take_budget(4, remaining);
    if (value.is_boolean()) return semantic_take_budget(5, remaining);
    if (value.is_number()) return semantic_take_budget(64, remaining);
    if (value.is_string()) {
        return semantic_string_within_budget(
            value.get_ref<const json::string_t &>(), deadline,
            remaining, operations, timed_out);
    }
    if (value.is_array()) {
        if (!semantic_take_budget(2, remaining)) return false;
        bool first = true;
        for (const auto & element : value) {
            if (!first && !semantic_take_budget(1, remaining)) return false;
            first = false;
            if (!semantic_json_within_budget(
                    element, deadline, remaining, depth + 1,
                    operations, timed_out)) {
                return false;
            }
        }
        return true;
    }
    if (value.is_object()) {
        if (!semantic_take_budget(2, remaining)) return false;
        bool first = true;
        for (const auto & item : value.items()) {
            if (!first && !semantic_take_budget(1, remaining)) return false;
            first = false;
            if (!semantic_string_within_budget(
                    item.key(), deadline, remaining, operations, timed_out) ||
                !semantic_take_budget(1, remaining) ||
                !semantic_json_within_budget(
                    item.value(), deadline, remaining, depth + 1,
                    operations, timed_out)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool semantic_message_content(
        const json & message,
        const std::chrono::steady_clock::time_point * deadline,
        std::string & text) {
    text.clear();
    const auto content = message.find("content");
    if (content == message.end() || content->is_null()) return true;
    if (content->is_string()) {
        text = content->get<std::string>();
        return !semantic_deadline_expired(deadline);
    }
    if (!content->is_array()) {
        text = content->dump();
        return !semantic_deadline_expired(deadline);
    }

    for (const auto & part : *content) {
        if (semantic_deadline_expired(deadline)) return false;
        if (part.is_string()) {
            text += part.get<std::string>();
            continue;
        }
        if (!part.is_object()) continue;
        const std::string type = string_member(part, "type");
        if (type == "text" || type == "input_text" ||
            type == "output_text") {
            text += string_member(part, "text");
        }
    }
    return !semantic_deadline_expired(deadline);
}

std::string forced_tool_name(const json & choice) {
    if (!choice.is_object()) return {};
    const auto function = choice.find("function");
    if (function != choice.end() && function->is_object()) {
        return string_member(*function, "name");
    }
    return string_member(choice, "name");
}

json canonical_semantic_tools(const json & tools) {
    json canonical = json::array();
    if (!tools.is_array()) return canonical;
    for (const auto & tool : tools) {
        if (!tool.is_object()) continue;
        const json * source = &tool;
        const auto wrapped = tool.find("function");
        if (wrapped != tool.end() && wrapped->is_object()) source = &*wrapped;
        const std::string name = string_member(*source, "name");
        if (name.empty()) continue;

        json function = {{"name", name}};
        const auto description = source->find("description");
        if (description != source->end() && description->is_string()) {
            function["description"] = *description;
        }
        for (const char * schema_key : {"parameters", "input_schema"}) {
            const auto schema = source->find(schema_key);
            if (schema != source->end() && schema->is_object()) {
                function["parameters"] = *schema;
                break;
            }
        }
        const auto strict = source->find("strict");
        if (strict != source->end() && strict->is_boolean()) {
            function["strict"] = *strict;
        }
        canonical.push_back({
            {"type", "function"},
            {"function", std::move(function)},
        });
    }
    return canonical;
}

}  // namespace

bool parse_semantic_tool_prediction(
        const json & response,
        const json & request_tools,
        SemanticToolCall & out,
        std::string & error) {
    error.clear();
    const auto choices = response.find("choices");
    if (choices == response.end() || !choices->is_array() ||
        choices->size() != 1 || !(*choices)[0].is_object()) {
        error = "predictor_response_missing_single_choice";
        return false;
    }
    const auto message = (*choices)[0].find("message");
    if (message == (*choices)[0].end() || !message->is_object()) {
        error = "predictor_response_missing_message";
        return false;
    }

    bool parsed = false;
    const auto calls = message->find("tool_calls");
    if (calls != message->end()) {
        if (!calls->is_array() || calls->size() != 1 ||
            !(*calls)[0].is_object()) {
            error = "predictor_response_requires_single_tool_call";
            return false;
        }
        const auto function = (*calls)[0].find("function");
        if (function != (*calls)[0].end()) {
            parsed = parse_call_object(*function, out);
        }
    } else {
        const auto content = message->find("content");
        if (content != message->end() && content->is_string()) {
            parsed = parse_content_call(content->get<std::string>(), out);
        }
    }
    if (!parsed) {
        error = "predictor_response_has_no_valid_call";
        return false;
    }
    if (!request_has_function(request_tools, out.name)) {
        error = "predictor_selected_unknown_function";
        return false;
    }
    return true;
}

bool materialize_declared_tool_defaults(
        const json & request_tools,
        SemanticToolCall & call,
        std::string & error) {
    error.clear();
    if (!call.arguments.is_object()) {
        error = "predictor_arguments_not_object";
        return false;
    }
    if (!request_tools.is_array()) {
        error = "predictor_tools_not_array";
        return false;
    }

    const json * function = nullptr;
    for (const auto & tool : request_tools) {
        if (!tool.is_object()) continue;
        const json & candidate = tool.contains("function") &&
                tool["function"].is_object()
            ? tool["function"] : tool;
        if (string_member(candidate, "name") == call.name) {
            function = &candidate;
            break;
        }
    }
    if (!function) {
        error = "predictor_selected_unknown_function";
        return false;
    }

    const json * parameters = nullptr;
    for (const char * key : {"parameters", "input_schema"}) {
        const auto found = function->find(key);
        if (found != function->end() && found->is_object()) {
            parameters = &*found;
            break;
        }
    }
    if (!parameters) return true;
    const auto properties = parameters->find("properties");
    if (properties == parameters->end() || !properties->is_object()) {
        return true;
    }
    for (const auto & property : properties->items()) {
        if (call.arguments.contains(property.key()) ||
            !property.value().is_object() ||
            !property.value().contains("default")) {
            continue;
        }
        call.arguments[property.key()] = property.value()["default"];
    }
    return true;
}

std::optional<SemanticToolPredictorRequest>
build_semantic_tool_predictor_request(
        const json & messages,
        const json & tools,
        const json & tool_choice,
        const std::string & sidecar_model,
        int max_tokens,
        std::string & error) {
    error.clear();
    size_t request_budget = kMaxNativePredictorRequestBytes;
    size_t budget_operations = 0;
    bool budget_timed_out = false;
    const json default_choice = "auto";
    const json & effective_choice = tool_choice.is_null()
        ? default_choice : tool_choice;
    if (!semantic_take_budget(512, request_budget) ||
        !semantic_string_within_budget(
            sidecar_model, nullptr, request_budget,
            budget_operations, budget_timed_out) ||
        !semantic_json_within_budget(
            messages, nullptr, request_budget, 0,
            budget_operations, budget_timed_out) ||
        !semantic_json_within_budget(
            tools, nullptr, request_budget, 0,
            budget_operations, budget_timed_out) ||
        !semantic_json_within_budget(
            effective_choice, nullptr, request_budget, 0,
            budget_operations, budget_timed_out)) {
        error = "predictor_request_too_large";
        return std::nullopt;
    }
    json canonical_tools = canonical_semantic_tools(tools);
    if (canonical_tools.empty()) {
        error = "predictor_request_has_no_valid_tools";
        return std::nullopt;
    }
    json request = {
        {"model", sidecar_model},
        {"stream", false},
        {"temperature", 0},
        {"max_tokens", max_tokens},
        {"messages", messages},
        {"tools", std::move(canonical_tools)},
        {"tool_choice", effective_choice},
    };
    // Tool-schema normalization can add OpenAI wrapper objects. Validate the
    // canonical result once before granting the bounded-request type; native
    // rendering then relies on that type instead of scanning the same payload
    // for a third time.
    size_t canonical_budget = kMaxNativePredictorRequestBytes;
    size_t canonical_operations = 0;
    bool canonical_timed_out = false;
    if (!semantic_json_within_budget(
            request, nullptr, canonical_budget, 0,
            canonical_operations, canonical_timed_out)) {
        error = "predictor_request_too_large";
        return std::nullopt;
    }
    return SemanticToolPredictorRequest(std::move(request));
}

std::string build_native_semantic_tool_predictor_prompt(
        const SemanticToolPredictorRequest & bounded_request,
        std::string & error,
        const std::chrono::steady_clock::time_point * deadline) {
    error.clear();
    if (semantic_deadline_expired(deadline)) {
        error = "native_predictor_timeout";
        return {};
    }
    const json & predictor_request = bounded_request.payload();
    const json choice = predictor_request.value("tool_choice", json("auto"));
    if ((choice.is_string() && choice.get<std::string>() == "none") ||
        (choice.is_object() && string_member(choice, "type") == "none")) {
        error = "native_predictor_tool_choice_none";
        return {};
    }
    const auto messages = predictor_request.find("messages");
    if (messages == predictor_request.end() || !messages->is_array() ||
        messages->empty()) {
        error = "native_predictor_missing_messages";
        return {};
    }
    const json tools = predictor_request.value("tools", json::array());
    if (!tools.is_array() || tools.empty()) {
        error = "native_predictor_missing_tools";
        return {};
    }

    struct PredictorMessage {
        std::string role;
        std::string content;
        json tool_calls;
    };
    std::vector<PredictorMessage> chat;
    chat.reserve(messages->size());
    for (const auto & message : *messages) {
        if (semantic_deadline_expired(deadline)) {
            error = "native_predictor_timeout";
            return {};
        }
        if (!message.is_object()) continue;
        std::string role = string_member(message, "role");
        if (role.empty()) role = "user";
        if (role == "developer") role = "system";
        std::string content;
        if (!semantic_message_content(message, deadline, content)) {
            error = "native_predictor_timeout";
            return {};
        }
        chat.push_back({
            std::move(role), std::move(content),
            message.value("tool_calls", json::array()),
        });
    }
    if (chat.empty()) {
        error = "native_predictor_empty_messages";
        return {};
    }

    std::string constraint;
    if (choice.is_string() && choice.get<std::string>() == "required") {
        constraint = "You must call exactly one available function.";
    } else if (const std::string name = forced_tool_name(choice);
               !name.empty()) {
        constraint = "You must call the function " + name + ".";
    }
    // Render the exact tokenizer.chat_template contract embedded in the
    // Qwen3-0.6B GGUF. PFlash's generic Qwen3.5 renderer uses parameter XML,
    // while this model was trained to emit one JSON object inside
    // <tool_call>; using the wrong contract destroys multi-tool accuracy.
    size_t begin = 0;
    std::string system_content;
    if (!chat.empty() && chat.front().role == "system") {
        system_content = chat.front().content;
        begin = 1;
    }
    if (!constraint.empty()) {
        if (!system_content.empty()) system_content += "\n\n";
        system_content += constraint;
    }

    std::string rendered = "<|im_start|>system\n";
    if (!system_content.empty()) {
        rendered += system_content;
        rendered += "\n\n";
    }
    rendered +=
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>";
    for (const auto & tool : tools) {
        if (semantic_deadline_expired(deadline)) {
            error = "native_predictor_timeout";
            return {};
        }
        rendered += tool.dump();
    }
    rendered +=
        "\n</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call><|im_end|>\n";

    bool in_tool_response = false;
    for (size_t index = begin; index < chat.size(); ++index) {
        if (semantic_deadline_expired(deadline)) {
            error = "native_predictor_timeout";
            return {};
        }
        const auto & message = chat[index];
        if (message.role == "tool") {
            if (!in_tool_response) {
                rendered += "<|im_start|>user";
                in_tool_response = true;
            }
            rendered += "\n<tool_response>\n" + message.content +
                        "\n</tool_response>";
            const bool next_is_tool = index + 1 < chat.size() &&
                                      chat[index + 1].role == "tool";
            if (!next_is_tool) {
                rendered += "<|im_end|>\n";
                in_tool_response = false;
            }
            continue;
        }

        rendered += "<|im_start|>" + message.role + "\n" + message.content;
        if (message.role == "assistant" && message.tool_calls.is_array()) {
            for (const auto & raw_call : message.tool_calls) {
                if (semantic_deadline_expired(deadline)) {
                    error = "native_predictor_timeout";
                    return {};
                }
                if (!raw_call.is_object()) continue;
                const json & call = raw_call.contains("function") &&
                                         raw_call["function"].is_object()
                    ? raw_call["function"] : raw_call;
                const std::string name = string_member(call, "name");
                if (name.empty() || !call.contains("arguments")) continue;
                if (!message.content.empty()) rendered += "\n";
                rendered += "<tool_call>\n{\"name\": \"" + name +
                            "\", \"arguments\": ";
                rendered += call["arguments"].is_string()
                    ? call["arguments"].get<std::string>()
                    : call["arguments"].dump();
                rendered += "}\n</tool_call>";
            }
        }
        rendered += "<|im_end|>\n";
    }
    rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    if (semantic_deadline_expired(deadline)) {
        error = "native_predictor_timeout";
        return {};
    }
    return rendered;
}

bool parse_native_semantic_tool_prediction(
        const std::string & generated_text,
        const json & request_tools,
        SemanticToolCall & out,
        std::string & error) {
    error.clear();
    const ToolParseResult parsed = parse_tool_calls(
        generated_text, request_tools);
    if (parsed.tool_calls.size() == 1) {
        try {
            ordered_json arguments = ordered_json::parse(
                parsed.tool_calls.front().arguments);
            if (!arguments.is_object()) {
                error = "native_predictor_arguments_not_object";
                return false;
            }
            out.name = parsed.tool_calls.front().name;
            out.arguments = std::move(arguments);
        } catch (...) {
            error = "native_predictor_arguments_invalid_json";
            return false;
        }
    } else if (parsed.tool_calls.empty()) {
        if (!parse_qwen_tagged_call_repair(generated_text, out) &&
            !parse_qwen_bare_single_tool_arguments(
                generated_text, request_tools, out)) {
            error = "native_predictor_response_has_no_valid_call";
            return false;
        }
    } else {
        error = "native_predictor_response_has_multiple_calls";
        return false;
    }
    if (!request_has_function(request_tools, out.name)) {
        error = "predictor_selected_unknown_function";
        return false;
    }
    return true;
}

}  // namespace dflash::common
