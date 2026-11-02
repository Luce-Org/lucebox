// The client APIs' side of a request: their message formats normalized into
// chat messages, and a generated reply built into each API's response.

// On Windows, winsock2.h must be included BEFORE windows.h (which comes
// transitively via http_server.h).
#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "api_adapters.h"

#include "prompt_normalize.h"

#include <string_view>
#include <unordered_map>

namespace luce::common {

// Normalize Anthropic's `system` field (top-level on /v1/messages and
// /v1/messages/count_tokens) into a leading `{role:"system", content:...}`
// entry on `messages`. Accepts either a flat string or an array of typed
// blocks (`[{type:"text", text:"..."}]`), and strips any
// `x-anthropic-billing-header:`-prefixed block injected by Claude Code so
// it never reaches the model or the token counter.
//
// Side-effect: prepends a system message to `messages` when the body has
// a non-empty `system` field after billing-header filtering. No-op
// otherwise. Both endpoints call this with identical semantics — having
// one helper guarantees token counting and generation can't drift.
void normalize_anthropic_system(const json & body, json & messages) {
    if (!body.contains("system")) return;
    // Delegate strip to the pure fn; insert as system message.
    std::string text = luce::common::normalize_system_for_cache(body["system"]);
    if (!text.empty()) {
        json sys_msg = {{"role", "system"}, {"content", text}};
        messages.insert(messages.begin(), sys_msg);
    }
}

void keep_anthropic_input_order(const nlohmann::ordered_json & ordered_messages, json & messages) {
    if (!ordered_messages.is_array() || !messages.is_array() ||
        messages.size() < ordered_messages.size()) {
        return;
    }
    const size_t offset = messages.size() - ordered_messages.size();
    for (size_t i = 0; i < ordered_messages.size(); ++i) {
        const auto & ordered = ordered_messages[i];
        json & message = messages[offset + i];
        if (!ordered.is_object() || !message.is_object()) continue;
        const auto ordered_content = ordered.find("content");
        const auto content = message.find("content");
        if (ordered_content == ordered.end() || content == message.end() ||
            !ordered_content->is_array() || !content->is_array() ||
            ordered_content->size() != content->size()) {
            continue;
        }
        for (size_t p = 0; p < content->size(); ++p) {
            const auto & ordered_part = (*ordered_content)[p];
            json & part = (*content)[p];
            if (!ordered_part.is_object() || !part.is_object() ||
                ordered_part.value("type", "") != "tool_use" || part.value("type", "") != "tool_use") {
                continue;
            }
            const auto input = ordered_part.find("input");
            if (input != ordered_part.end() && input->is_object()) part["input"] = input->dump();
        }
    }
}

namespace {

// Arguments sent as a JSON string (Chat Completions, Responses) keep the
// client's key order; an object (Anthropic) arrives already parsed.
nlohmann::ordered_json normalize_tool_arguments(const json * arguments) {
    if (!arguments) return nlohmann::ordered_json::object();
    if (arguments->is_object()) return *arguments;
    if (arguments->is_string()) {
        const auto & text = arguments->get_ref<const std::string &>();
        if (text.empty()) return nlohmann::ordered_json::object();
        auto parsed = nlohmann::ordered_json::parse(text, nullptr, false);
        if (parsed.is_object()) return parsed;
    }
    // Retain malformed/non-object arguments rather than silently losing them.
    return nlohmann::ordered_json{{"arguments", *arguments}};
}

// Text of an Anthropic tool_result block: a string, or the text blocks of an
// array (images and documents have no text form here).
std::string anthropic_tool_result_text(const json & block) {
    if (!block.contains("content")) return {};
    const auto & content = block["content"];
    if (content.is_string()) return content.get<std::string>();
    std::string text;
    if (content.is_array()) {
        for (const auto & part : content) {
            if (!part.is_object() || part.value("type", "") != "text") continue;
            if (!text.empty()) text += "\n";
            text += part.value("text", "");
        }
    }
    return text;
}

}  // namespace

std::vector<ChatMessage> normalize_chat_messages(
    const json & messages,
    ApiFormat format,
    ToolMemory & tool_memory,
    bool replay_tool_memory,
    const char * text_part_separator) {
    std::vector<ChatMessage> chat_msgs;
    std::vector<std::string> system_parts;
    ChatMessage response_turn;
    std::vector<const json *> response_arguments;

    auto append_call = [](ChatMessage & message, std::vector<const json *> & arguments,
                          const json & function, std::string id, const char * argument_key) {
        message.tool_calls.push_back({std::move(id), function.value("name", ""), {}});
        const auto supplied = function.find(argument_key);
        arguments.push_back(supplied == function.end() ? nullptr : &*supplied);
    };
    auto finish_calls = [&](ChatMessage & message, const std::vector<const json *> & arguments) {
        if (message.tool_calls.empty()) return;
        std::string raw;
        if (replay_tool_memory && !tool_memory.disabled()) {
            std::vector<std::string> ids;
            ids.reserve(message.tool_calls.size());
            for (const auto & call : message.tool_calls) ids.push_back(call.id);
            // Empty IDs remain in the lookup: partial/id-less turns cannot replay.
            raw = tool_memory.lookup(ids);
        }
        if (!raw.empty()) {
            message.content = std::move(raw);
            message.tool_calls_replayed = true;
        } else {
            for (size_t i = 0; i < message.tool_calls.size(); ++i) {
                message.tool_calls[i].arguments = normalize_tool_arguments(arguments[i]);
            }
        }
    };
    auto flush_response_turn = [&]() {
        if (response_turn.role.empty()) return;
        finish_calls(response_turn, response_arguments);
        chat_msgs.push_back(std::move(response_turn));
        response_turn = {};
        response_arguments.clear();
    };

    if (messages.is_array()) {
        for (const auto & m : messages) {
            if (format == ApiFormat::RESPONSES && m.is_object()) {
                const std::string item_type = m.value("type", "message");
                if (item_type == "function_call") {
                    response_turn.role = "assistant";
                    append_call(response_turn, response_arguments, m,
                                m.value("call_id", m.value("id", "")), "arguments");
                    continue;
                }
                if (item_type != "message" || m.value("role", "user") != "assistant") {
                    flush_response_turn();
                }
                if (item_type == "function_call_output") {
                    std::string output;
                    if (m.contains("output") && m["output"].is_string()) {
                        output = m["output"].get<std::string>();
                    } else if (m.contains("output")) {
                        output = m["output"].dump();
                    }
                    chat_msgs.push_back({"tool", std::move(output),
                                         m.value("call_id", m.value("id", ""))});
                    continue;
                }
                // Other items (reasoning, item references, built-in tool
                // calls) have no form in the chat template.
                if (item_type != "message") continue;
            }

            // Responses splits one assistant turn into message/function_call
            // items. Keep adjacent items together for both raw and cold replay.
            const std::string role = m.value("role", "user");
            const bool response_assistant = format == ApiFormat::RESPONSES && role == "assistant";
            ChatMessage local_message;
            std::vector<const json *> local_arguments;
            ChatMessage & cm = response_assistant ? response_turn : local_message;
            auto & arguments = response_assistant ? response_arguments : local_arguments;
            cm.role = role;

            // Carry a replayed assistant turn's prior <think> text through to
            // the Jinja renderer as message.reasoning_content (OpenAI/DeepSeek
            // dialect; `reasoning` is the OpenRouter/Anthropic-gateway flat
            // alias — same field the response side emits, see
            // format_response_message's reasoning_content/reasoning/
            // reasoning_details trio below).
            if (cm.role == "assistant") {
                if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                    cm.reasoning_content = m["reasoning_content"].get<std::string>();
                    cm.reasoning_sent = true;
                } else if (m.contains("reasoning") && m["reasoning"].is_string()) {
                    cm.reasoning_content = m["reasoning"].get<std::string>();
                    cm.reasoning_sent = true;
                }
            }

            const bool anthropic_blocks = format == ApiFormat::ANTHROPIC &&
                m.contains("content") && m["content"].is_array();

            // Anthropic tool results precede any user text in the same turn.
            if (anthropic_blocks && cm.role == "user") {
                bool has_results = false;
                std::string text;
                for (const auto & part : m["content"]) {
                    if (!part.is_object()) continue;
                    const std::string ptype = part.value("type", "");
                    if (ptype == "tool_result") {
                        chat_msgs.push_back({"tool", anthropic_tool_result_text(part),
                                             part.value("tool_use_id", "")});
                        has_results = true;
                    } else if (ptype == "text") {
                        text += part.value("text", "");
                    }
                }
                if (has_results) {
                    if (!text.empty()) chat_msgs.push_back({"user", std::move(text)});
                    continue;
                }
            }

            if (cm.role == "assistant" && m.contains("tool_calls") &&
                m["tool_calls"].is_array()) {
                cm.tool_calls.reserve(cm.tool_calls.size() + m["tool_calls"].size());
                arguments.reserve(arguments.size() + m["tool_calls"].size());
                for (const auto & tc : m["tool_calls"]) {
                    if (!tc.is_object()) continue;
                    const auto function = tc.find("function");
                    if (function == tc.end() || !function->is_object()) continue;
                    append_call(cm, arguments, *function, tc.value("id", ""), "arguments");
                }
            }
            if (anthropic_blocks && cm.role == "assistant") {
                // Thinking blocks are the turn's reasoning, as the response
                // sent it: an edit to them counts, as one to content does.
                bool thinking_sent = false;
                std::string thinking;
                for (const auto & part : m["content"]) {
                    if (!part.is_object()) continue;
                    const std::string ptype = part.value("type", "");
                    if (ptype == "thinking" && part.contains("thinking") && part["thinking"].is_string()) {
                        thinking += part["thinking"].get<std::string>();
                        thinking_sent = true;
                    } else if (ptype == "redacted_thinking") {
                        // Reasoning the client sent but nobody can read: not the remembered one.
                        thinking_sent = true;
                    } else if (ptype == "tool_use") {
                        append_call(cm, arguments, part, part.value("id", ""), "input");
                    }
                }
                if (thinking_sent && !cm.reasoning_sent) {
                    cm.reasoning_content = std::move(thinking);
                    cm.reasoning_sent = true;
                }
            }

            if (!response_assistant) finish_calls(cm, arguments);
            if (!cm.tool_calls_replayed) {
                std::string text;
                if (m.contains("content") && m["content"].is_string()) {
                    text = m["content"].get<std::string>();
                } else if (m.contains("content") && m["content"].is_array()) {
                    // Parts joined as the model's template joins them.
                    bool first = true;
                    for (const auto & part : m["content"]) {
                        if (!part.is_object()) continue;
                        const std::string ptype = part.value("type", "");
                        if (ptype == "text" || ptype == "input_text" || ptype == "output_text") {
                            if (!first) text += text_part_separator;
                            text += part.value("text", "");
                            first = false;
                        }
                    }
                }
                if (!text.empty()) {
                    if (cm.content.empty()) {
                        cm.content = std::move(text);
                    } else {
                        cm.content += "\n\n";
                        cm.content += text;
                    }
                }
            }
            if (cm.role == "tool" || cm.role == "function") {
                cm.tool_call_id = m.value("tool_call_id", "");
                cm.tool_name = m.value("name", "");
            }
            if (response_assistant) continue;

            if (format == ApiFormat::RESPONSES && (cm.role == "system" || cm.role == "developer")) {
                system_parts.push_back(std::move(cm.content));
            } else {
                chat_msgs.push_back(std::move(cm));
            }
        }
        flush_response_turn();
    } else if (messages.is_string()) {
        chat_msgs.push_back({"user", messages.get<std::string>()});
    }

    if (!system_parts.empty()) {
        std::string merged_system;
        for (size_t i = 0; i < system_parts.size(); i++) {
            if (i) merged_system += "\n\n";
            merged_system += system_parts[i];
        }
        chat_msgs.insert(chat_msgs.begin(), {"system", std::move(merged_system)});
    }

    // Resolve names only after the message vector is stable; views avoid
    // copying IDs/names while associating out-of-order tool results.
    std::unordered_map<std::string_view, std::string_view> tool_names;
    for (auto & message : chat_msgs) {
        for (const auto & call : message.tool_calls) {
            if (!call.id.empty()) tool_names[call.id] = call.name;
        }
        if (message.tool_name.empty() && !message.tool_call_id.empty()) {
            const auto name = tool_names.find(message.tool_call_id);
            if (name != tool_names.end()) message.tool_name = name->second;
        }
    }
    return chat_msgs;
}

TokenDelivery classify_generated_token(
        Tokenizer & tokenizer, SseEmitter & emitter,
        int32_t token, std::string & text) {
    if (token == tokenizer.eos_id() || token == tokenizer.eos_chat_id()) {
        return TokenDelivery::kSkip;
    }

    const std::string & raw = tokenizer.raw_token(token);
    if (emitter.suppress_undeclared_tool_protocol_token(raw)) {
        return TokenDelivery::kSkip;
    }

    // Gemma4 thinking channel (<|channel> / <channel|>) and Qwen3.6
    // thinking markers share one mapped dialect. The Qwen markers
    // <think> (id 248068) and </think> (id 248069) are SINGLE special
    // tokens in the added_tokens vocab: without this mapping they would
    // hit the generic "<...>" strip below and be silently dropped, the
    // emitter would never see the reasoning→content transition, and the
    // whole answer would land in reasoning_content with empty content.
    if (raw == "<|channel>" || raw == "<think>") {
        text = "<think>";
        return TokenDelivery::kThinkTag;
    }
    if (raw == "<channel|>" || raw == "</think>") {
        text = "</think>\n";
        return TokenDelivery::kThinkTag;
    }

    // Bailing V3 stores its tool-call XML delimiters as added special
    // tokens. They are part of the model's public output protocol, not chat
    // control markers: keep them so the shared parser can reconstruct an
    // OpenAI `tool_calls` object. Stripping them leaves only three unrelated
    // text lines (function name, argument name, value).
    if (raw == "<tool_call>" || raw == "</tool_call>" ||
        raw == "<arg_key>" || raw == "</arg_key>" ||
        raw == "<arg_value>" || raw == "</arg_value>") {
        text = raw;
        return TokenDelivery::kText;
    }

    // Other special tokens are internal control markers. Byte-fallback
    // tokens such as <0xAB> are text and must still reach the emitter.
    if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') {
        return TokenDelivery::kSkip;
    }
    if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>' &&
        !(raw.size() == 6 && raw[1] == '0' && raw[2] == 'x')) {
        return TokenDelivery::kSkip;
    }

    text = tokenizer.token_text(token);
    return TokenDelivery::kText;
}

CompletionTokenCounts feed_non_streaming_tokens(
        const std::vector<int32_t> & tokens, Tokenizer & tokenizer,
        SseEmitter & emitter, int generation_cap) {
    // The live token callback already classified these tokens on this
    // emitter; a reply cut off inside a suppressed block left it open.
    emitter.restart_tool_protocol_suppression();
    for (int32_t token : tokens) {
        std::string text;
        const TokenDelivery delivery =
            classify_generated_token(tokenizer, emitter, token, text);
        if (delivery == TokenDelivery::kSkip) continue;

        emitter.emit_token(text);
        // Matching the streaming path, only ordinary text is checked
        // against stop sequences — think-tag markers never count.
        if (delivery == TokenDelivery::kText && emitter.stop_hit()) break;
    }

    CompletionTokenCounts counts;
    counts.total = (int) tokens.size();
    const bool is_eos = !tokens.empty() &&
        (tokens.back() == tokenizer.eos_id() ||
         tokens.back() == tokenizer.eos_chat_id());
    emitter.emit_finish(counts.total, nullptr, generation_cap, is_eos);

    // Split reasoning vs content at the emitter's REASONING→CONTENT
    // transition; see first_content_token_index() in sse_emitter.h.
    // Skipped specials mean `emitted` can be below counts.total — the
    // remainder is unattributed (e.g. TOOL_BUFFER).
    const int first_content = emitter.first_content_token_index();
    const int emitted = emitter.emit_token_count();
    counts.reasoning = first_content < 0 ? emitted : first_content;
    counts.content = first_content < 0 ? 0 : emitted - first_content;
    return counts;
}

namespace {

json build_openai_completion_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings,
        const CompletionTokenCounts & counts, const SseEmitter & emitter,
        const Tokenizer * tokenizer = nullptr) {
    json message = {
        {"role", "assistant"},
        {"content", emitter.accumulated_text()},
    };
    if (!emitter.reasoning_text().empty()) {
        // Multi-dialect reasoning emission — same text, three keys. See
        // docs/specs/thinking-budget.md "Response shape — multi-dialect
        // aliasing".
        //   reasoning_content : DeepSeek R1 / dflash primary
        //   reasoning         : OpenRouter / Anthropic-gateway flat
        //   reasoning_details : typed-block list; single block.
        const std::string & reasoning = emitter.reasoning_text();
        message["reasoning_content"] = reasoning;
        message["reasoning"] = reasoning;
        message["reasoning_details"] = json::array({
            {{"type", "reasoning.text"}, {"text", reasoning}},
        });
    }
    if (!emitter.tool_calls().empty()) {
        json tool_calls = json::array();
        for (const auto & tool_call : emitter.tool_calls()) {
            tool_calls.push_back({
                {"id", tool_call.id},
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", tool_call.arguments},
                }},
            });
        }
        message["tool_calls"] = tool_calls;
    }
    // The emitter only knows "stop" / "tool_calls"; it cannot see that the
    // daemon hit the n_gen cap. Derive "length" from the committed-token
    // count — OpenAI-compatible clients (open-webui, Cline) gate retry
    // logic on finish_reason == "length".
    const bool is_eos = tokenizer && !result.tokens.empty() &&
        (result.tokens.back() == tokenizer->eos_id() ||
         result.tokens.back() == tokenizer->eos_chat_id());
    std::string finish_reason = emitter.finish_reason();
    if (finish_reason == "stop" && !is_eos && counts.total >= generation_cap) {
        finish_reason = "length";
    }
    // Degenerate decode (post-close repetition-loop watchdog) also reports
    // "length": OpenAI/Anthropic/Gemini all collapse budget-class events
    // into one closed enum, with richer signal in sidecar fields below.
    if (result.degenerate_decode_close) finish_reason = "length";

    json choice = {
        {"index", 0},
        {"message", message},
        {"finish_reason", finish_reason},
    };
    if (req.thinking_opt_in) {
        // finish_details mirrors ds4_eval.c's eval_think_close_info.
        // close_kind is "natural" when the model closed its own thinking
        // block, "hard" when the BudgetHook force-closed it at the budget
        // boundary. See docs/specs/thinking-budget.md "v2 design".
        choice["finish_details"] = {
            {"close_kind", result.budget_forced_close ? "hard" : "natural"},
            {"thinking_tokens", counts.reasoning},
            {"content_tokens", counts.content},
            {"total_tokens", counts.total},
        };
        // Honest signaling: the answer after an aborted repetition loop is
        // unreliable; surface that without breaking the finish_reason enum.
        if (result.degenerate_decode_close) {
            choice["finish_details"]["degenerate_decode"] = true;
        }
    }

    // usage.completion_tokens_details.reasoning_tokens — OpenAI o1/o3
    // standard location; kept in sync with finish_details.thinking_tokens.
    // usage.timings — per-request prefill/decode wall clock, additive to
    // the OpenAI shape (ignored by clients that don't recognize it).
    // spec_decode_ran disambiguates a real zero-acceptance speculative run
    // from an autoregressive fallback. See docs/specs/thinking-budget.md §6.3.
    const int prompt_tokens = usage_prompt_tokens(req);
    const json usage = {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", counts.total},
        {"total_tokens", prompt_tokens + counts.total},
        {"completion_tokens_details", {
            {"reasoning_tokens", counts.reasoning},
        }},
        {"prompt_tokens_details", {
            {"cached_tokens", timings.cached_prefix_tokens},
        }},
        {"timings", build_timings_json(timings, counts.total)},
        {"accept_rate", result.accept_rate},
        {"spec_decode_ran", result.spec_decode_ran},
    };
    return {
        {"id", req.response_id},
        {"object", "chat.completion"},
        {"created", std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()},
        {"model", req.model},
        {"choices", json::array({choice})},
        {"usage", usage},
    };
}

json build_anthropic_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings,
        const CompletionTokenCounts & counts, const SseEmitter & emitter,
        const Tokenizer * tokenizer = nullptr) {
    json content = json::array();
    if (!emitter.reasoning_text().empty()) {
        content.push_back({
            {"type", "thinking"},
            {"thinking", emitter.reasoning_text()},
        });
    }
    // Tool-only replies intentionally omit an empty text block; Anthropic
    // clients expect the tool_use blocks to stand alone.
    if (!emitter.accumulated_text().empty()) {
        content.push_back({
            {"type", "text"},
            {"text", emitter.accumulated_text()},
        });
    }
    for (const auto & tool_call : emitter.tool_calls()) {
        json input = json::object();
        if (!tool_call.arguments.empty()) {
            json parsed = json::parse(tool_call.arguments, nullptr, false);
            if (!parsed.is_discarded() && parsed.is_object()) {
                input = std::move(parsed);
            } else {
                input = {{"_raw", tool_call.arguments}};
            }
        }
        content.push_back({
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name},
            {"input", std::move(input)},
        });
    }

    // stop_reason is Anthropic's analog of finish_reason, with the same
    // length-vs-EOS distinction — Cline / Anthropic SDK clients gate
    // retry logic on stop_reason == "max_tokens".
    const bool is_eos = tokenizer && !result.tokens.empty() &&
        (result.tokens.back() == tokenizer->eos_id() ||
         result.tokens.back() == tokenizer->eos_chat_id());
    // A degenerate decode (the repetition watchdog) is "max_tokens", as
    // OpenAI's "length": the reply is unreliable.
    std::string stop_reason;
    if (result.degenerate_decode_close) {
        stop_reason = "max_tokens";
    } else if (emitter.finish_reason() == "tool_calls") {
        stop_reason = "tool_use";
    } else if (!is_eos && counts.total >= generation_cap) {
        stop_reason = "max_tokens";
    } else {
        stop_reason = "end_turn";
    }

    const json usage = {
        {"input_tokens", usage_prompt_tokens(req)},
        {"output_tokens", counts.total},
        {"timings", build_timings_json(timings, counts.total)},
        {"accept_rate", result.accept_rate},
        {"spec_decode_ran", result.spec_decode_ran},
    };
    return {
        {"id", req.response_id},
        {"type", "message"},
        {"role", "assistant"},
        {"model", req.model},
        {"content", content},
        {"stop_reason", stop_reason},
        {"usage", usage},
    };
}

json build_responses_api_response(
        const ParsedRequest & req, const GenerateResult & result,
        const GenTimings & timings, const CompletionTokenCounts & counts,
        const SseEmitter & emitter) {
    // The same items as the stream's response.completed: the assistant
    // message first, then one function_call item per call.
    const json output = responses_output_items(
        req.response_id + "_msg", emitter.accumulated_text(), emitter.tool_calls());

    const int prompt_tokens = usage_prompt_tokens(req);
    const json usage = {
        {"input_tokens", prompt_tokens},
        {"output_tokens", counts.total},
        {"total_tokens", prompt_tokens + counts.total},
        {"input_tokens_details", {
            {"cached_tokens", timings.cached_prefix_tokens},
        }},
        {"timings", build_timings_json(timings, counts.total)},
        {"accept_rate", result.accept_rate},
        {"spec_decode_ran", result.spec_decode_ran},
    };
    json response = {
        {"id", req.response_id},
        {"object", "response"},
        {"model", req.model},
        {"output", output},
        {"usage", usage},
    };
    set_responses_status(response, emitter.hit_output_cap() || result.degenerate_decode_close);
    return response;
}

}  // namespace

json build_non_streaming_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings,
        const CompletionTokenCounts & counts, SseEmitter & emitter,
        const Tokenizer * tokenizer) {
    switch (req.format) {
    case ApiFormat::OPENAI_CHAT:
        return build_openai_completion_response(
            req, result, generation_cap, timings, counts, emitter, tokenizer);
    case ApiFormat::ANTHROPIC:
        return build_anthropic_response(
            req, result, generation_cap, timings, counts, emitter, tokenizer);
    case ApiFormat::RESPONSES:
        return build_responses_api_response(
            req, result, timings, counts, emitter);
    default:
        return {{"text", emitter.accumulated_text()}};
    }
}

json build_non_streaming_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings, Tokenizer & tokenizer,
        SseEmitter & emitter) {
    const CompletionTokenCounts counts = feed_non_streaming_tokens(
        result.tokens, tokenizer, emitter, generation_cap);
    return build_non_streaming_response(
        req, result, generation_cap, timings, counts, emitter, &tokenizer);
}

}  // namespace luce::common
