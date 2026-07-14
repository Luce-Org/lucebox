// SSE emitter implementation — streaming state machine for all 3 API formats.

#include "sse_emitter.h"
#include "utf8_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace dflash::common {

static const char THINK_OPEN[]  = "<think>";
static const char THINK_CLOSE[] = "</think>";
static const char TOOL_OPEN[]   = "<tool_call>";
static const char FUNCTION_OPEN[] = "<function=";
static const char TOOL_CODE_OPEN[] = "<tool_code>";
// Laguna emits `NAME<arg_key>K</arg_key><arg_value>V</arg_value>` with the
// <tool_call> wrapper as special tokens that detokenization strips, so the
// visible trigger is <arg_key> and the tool name sits immediately before it.
static const char ARG_KEY_OPEN[] = "<arg_key>";
static constexpr size_t THINK_OPEN_LEN  = 7;
static constexpr size_t THINK_CLOSE_LEN = 8;

static bool has_request_tools(const json & tools) {
    return tools.is_array() && !tools.empty();
}

static bool has_single_request_tool(const json & tools) {
    return tools.is_array() && tools.size() == 1 && tools[0].is_object();
}

static bool starts_with_potential_bare_json_tool(const std::string & text,
                                                 const json & tools) {
    if (!has_single_request_tool(tools)) return false;
    size_t first = text.find_first_not_of(" \t\n\r");
    return first != std::string::npos && text[first] == '{';
}

static bool find_tool_start(const std::string & text, size_t & pos) {
    size_t idx = text.find('<');
    while (idx != std::string::npos) {
        if (text.compare(idx, sizeof(TOOL_OPEN) - 1, TOOL_OPEN) == 0 ||
            text.compare(idx, sizeof(FUNCTION_OPEN) - 1, FUNCTION_OPEN) == 0 ||
            text.compare(idx, sizeof(TOOL_CODE_OPEN) - 1, TOOL_CODE_OPEN) == 0) {
            pos = idx;
            return true;
        }
        if (text.compare(idx, sizeof(ARG_KEY_OPEN) - 1, ARG_KEY_OPEN) == 0) {
            // Rewind over the tool name so it lands in the tool buffer
            // (the parser extracts it from just before <arg_key>).
            size_t start = idx;
            auto is_ident = [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '-';
            };
            while (start > 0 && is_ident(text[start - 1])) start--;
            if (start < idx) {  // require a non-empty name
                pos = start;
                return true;
            }
        }
        idx = text.find('<', idx + 1);
    }
    return false;
}

static std::string gen_item_id() {
    static std::atomic<uint64_t> ctr{0};
    char buf[32];
    std::snprintf(buf, sizeof(buf), "item_%016llx", (unsigned long long)ctr.fetch_add(1));
    return buf;
}

static int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Round `x` to 1 decimal place. JSON serialization of doubles can emit
// 17 significant digits which is noisy in client logs and bench output;
// caller-side rounding keeps the wire format stable across runs.
static double round1(double x) {
    return std::round(x * 10.0) / 10.0;
}

json build_timings_json(const GenTimings & t, int completion_tokens) {
    const double prefill_ms = round1(t.prefill_s * 1000.0);
    const double decode_ms  = round1(t.decode_s  * 1000.0);
    const double tps = t.decode_s > 0.0
        ? round1((double)completion_tokens / t.decode_s) : 0.0;
    return json{
        {"prefill_ms",            prefill_ms},
        {"decode_ms",             decode_ms},
        {"decode_tokens_per_sec", tps}
    };
}

// ─── Constructor ────────────────────────────────────────────────────────

SseEmitter::SseEmitter(ApiFormat format,
                       const std::string & request_id,
                       const std::string & model_name,
                       int prompt_tokens,
                       const json & tools,
                       ToolMemory * tool_memory,
                       const std::vector<std::string> & stop_sequences,
                       bool started_in_thinking)
    : format_(format)
    , request_id_(request_id)
    , model_name_(model_name)
    , prompt_tokens_(prompt_tokens)
    , tools_(tools)
    , tool_memory_(tool_memory)
    , mode_(started_in_thinking ? StreamMode::REASONING : StreamMode::CONTENT)
    , active_kind_(started_in_thinking ? "thinking" : "text")
    , stop_sequences_(stop_sequences)
    , created_at_(unix_timestamp())
    , msg_item_id_(gen_item_id())
{
    // Compute stop holdback: max length of any stop sequence
    for (const auto & s : stop_sequences_) {
        if (s.size() > stop_holdback_) stop_holdback_ = s.size();
    }
}

// ─── SSE formatting helpers ─────────────────────────────────────────────

std::string SseEmitter::sse_data(const std::string & json_str) {
    return "data: " + json_str + "\n\n";
}

std::string SseEmitter::sse_event(const std::string & type, const std::string & json_str) {
    return "event: " + type + "\ndata: " + json_str + "\n\n";
}

std::string SseEmitter::format_openai_delta(const json & delta, const char * finish) {
    json chunk = {
        {"id", request_id_},
        {"object", "chat.completion.chunk"},
        {"created", created_at_},
        {"model", model_name_},
        {"choices", json::array({{
            {"index", 0},
            {"delta", delta},
            {"finish_reason", finish ? json(finish) : json(nullptr)}
        }})}
    };
    return sse_data(chunk.dump());
}

std::string SseEmitter::format_anthropic_event(const std::string & event_type,
                                                const json & data) {
    json d = data;
    d["type"] = event_type;
    return sse_event(event_type, d.dump());
}

std::string SseEmitter::format_responses_event(const std::string & event_type,
                                                const json & data) {
    json d = data;
    d["type"] = event_type;
    return sse_event(event_type, d.dump());
}

// ─── emit_start ─────────────────────────────────────────────────────────

std::vector<std::string> SseEmitter::emit_start() {
    std::vector<std::string> out;

    switch (format_) {
    case ApiFormat::OPENAI_CHAT:
        // Role delta
        out.push_back(format_openai_delta({{"role", "assistant"}}));
        break;

    case ApiFormat::ANTHROPIC: {
        // message_start
        json msg_start = {
            {"type", "message_start"},
            {"message", {
                {"id", request_id_}, {"type", "message"},
                {"role", "assistant"}, {"model", model_name_},
                {"content", json::array()},
                {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                {"usage", {{"input_tokens", prompt_tokens_}, {"output_tokens", 0}}}
            }}
        };
        out.push_back(sse_event("message_start", msg_start.dump()));

        // First content block
        json block;
        if (active_kind_ == "thinking") {
            block = {{"type", "thinking"}, {"thinking", ""}};
        } else {
            block = {{"type", "text"}, {"text", ""}};
        }
        json block_start = {
            {"type", "content_block_start"},
            {"index", block_index_},
            {"content_block", block}
        };
        out.push_back(sse_event("content_block_start", block_start.dump()));
        break;
    }

    case ApiFormat::RESPONSES: {
        // response.created
        json shell = {
            {"id", request_id_}, {"object", "response"},
            {"created_at", created_at_}, {"status", "in_progress"},
            {"model", model_name_}
        };
        out.push_back(format_responses_event("response.created", {{"response", shell}}));

        // output_item.added
        out.push_back(format_responses_event("response.output_item.added", {
            {"output_index", 0},
            {"item", {{"type", "message"}, {"id", msg_item_id_},
                      {"status", "in_progress"}, {"role", "assistant"},
                      {"content", json::array()}}}
        }));

        // content_part.added
        out.push_back(format_responses_event("response.content_part.added", {
            {"item_id", msg_item_id_}, {"output_index", 0},
            {"content_index", 0},
            {"part", {{"type", "output_text"}, {"text", ""}, {"annotations", json::array()}}}
        }));
        break;
    }

    default:
        break;
    }

    return out;
}

// ─── emit_token ─────────────────────────────────────────────────────────

std::vector<std::string> SseEmitter::emit_token(const std::string & raw_piece) {
    if (stop_hit_) return {};  // already stopped

    // Track the first emit_token call whose mode-on-entry is CONTENT —
    // that's the first token attributed to the visible reply. Mode-on-
    // entry matters because a token whose text *contains* `</think>`
    // arrives while mode is still REASONING; the transition fires
    // mid-emit. The token AFTER that transition is the first content
    // token. Captured here so http_server can compute the natural-close
    // split without a parallel bump_count loop.
    //
    // Exception: a leading `<think>` opener (Qwen3.6's thinking-enabled
    // first token, or the synthesized `<|channel>` → `<think>` map for
    // gemma4) arrives while mode is still CONTENT — the emitter's
    // default — but the piece immediately transitions to REASONING.
    // Capturing fci=0 in that case would misreport thinking_tokens as 0
    // for any streamed-thinking response. Detect the `<think>` opener
    // here (lookahead in the unsanitized piece, before the state
    // machine runs) and skip the capture so it can fire on a later
    // CONTENT-mode token after the natural </think> close.
    const bool entry_is_think_opener =
        mode_ == StreamMode::CONTENT &&
        raw_piece.find(THINK_OPEN) != std::string::npos;
    if (first_content_token_index_ < 0 && mode_ == StreamMode::CONTENT &&
        !entry_is_think_opener) {
        first_content_token_index_ = emit_token_count_;
    }
    emit_token_count_++;

    // Sanitize input to prevent json::dump() from throwing on invalid UTF-8.
    std::string piece = utf8_sanitize(raw_piece);
    std::vector<std::string> out;
    accumulated_raw_ += piece;
    window_ += piece;

    // Stop sequence detection (not in tool_buffer mode, matching Python logic).
    if (!stop_sequences_.empty() && mode_ != StreamMode::TOOL_BUFFER) {
        size_t best = std::string::npos;
        for (const auto & seq : stop_sequences_) {
            size_t pos = window_.find(seq);
            if (pos != std::string::npos && (best == std::string::npos || pos < best)) {
                best = pos;
            }
        }
        if (best != std::string::npos) {
            // Emit everything before the stop sequence
            std::string pre = window_.substr(0, best);
            if (!pre.empty()) {
                if (mode_ == StreamMode::REASONING) {
                    reasoning_text_ += pre;
                    switch (format_) {
                    case ApiFormat::OPENAI_CHAT:
                        out.push_back(format_openai_delta({{"reasoning_content", pre}}));
                        break;
                    case ApiFormat::ANTHROPIC:
                        out.push_back(sse_event("content_block_delta",
                            json({{"type", "content_block_delta"}, {"index", block_index_},
                                  {"delta", {{"type", "thinking_delta"}, {"thinking", pre}}}}).dump()));
                        break;
                    default: break;
                    }
                } else {
                    accumulated_content_ += pre;
                    emit_content_delta(out, pre);
                }
            }
            window_.clear();
            stop_hit_ = true;
            return out;
        }
    }

    // State machine loop — processes the window
    while (true) {
        if (mode_ == StreamMode::TOOL_BUFFER) {
            tool_buffer_ += window_;
            window_.clear();
            break;
        }

        if (mode_ == StreamMode::REASONING) {
            // Strip leading <think> tag from reasoning (ds4 pattern).
            // The model may emit whitespace before <think>, so we skip leading
            // whitespace first, then check for the tag.
            if (!checked_think_prefix_) {
                // Skip leading whitespace to find potential <think> tag
                size_t ws = 0;
                while (ws < window_.size() && (window_[ws] == '\n' || window_[ws] == ' ' || window_[ws] == '\r'))
                    ws++;
                if (ws + THINK_OPEN_LEN > window_.size()) break;  // wait for more
                if (window_.compare(ws, THINK_OPEN_LEN, THINK_OPEN) == 0) {
                    window_ = window_.substr(ws + THINK_OPEN_LEN);
                }
                checked_think_prefix_ = true;
            }

            size_t idx = window_.find(THINK_CLOSE);
            if (idx != std::string::npos) {
                std::string pre = window_.substr(0, idx);
                if (!pre.empty()) {
                    reasoning_text_ += pre;
                    // Emit reasoning delta
                    switch (format_) {
                    case ApiFormat::OPENAI_CHAT:
                        out.push_back(format_openai_delta({{"reasoning_content", pre}}));
                        break;
                    case ApiFormat::ANTHROPIC: {
                        if (active_kind_ != "thinking") {
                            out.push_back(sse_event("content_block_stop",
                                json({{"type", "content_block_stop"}, {"index", block_index_}}).dump()));
                            block_index_++;
                            active_kind_ = "thinking";
                            json new_block = {{"type", "thinking"}, {"thinking", ""}};
                            out.push_back(sse_event("content_block_start",
                                json({{"type", "content_block_start"}, {"index", block_index_},
                                      {"content_block", new_block}}).dump()));
                        }
                        out.push_back(sse_event("content_block_delta",
                            json({{"type", "content_block_delta"}, {"index", block_index_},
                                  {"delta", {{"type", "thinking_delta"}, {"thinking", pre}}}}).dump()));
                        break;
                    }
                    case ApiFormat::RESPONSES:
                        // Responses API doesn't stream reasoning — it's stripped
                        break;
                    default: break;
                    }
                }
                window_ = window_.substr(idx + THINK_CLOSE_LEN);
                mode_ = StreamMode::CONTENT;
                continue;
            }
            // No close tag yet — emit safe prefix if window is large enough
            if (window_.size() > std::max(BASE_HOLDBACK, stop_holdback_)) {
                size_t cut = utf8_safe_len(window_, window_.size() - std::max(BASE_HOLDBACK, stop_holdback_));
                if (cut == 0) break;  // not enough complete chars yet
                std::string safe = window_.substr(0, cut);
                reasoning_text_ += safe;
                switch (format_) {
                case ApiFormat::OPENAI_CHAT:
                    out.push_back(format_openai_delta({{"reasoning_content", safe}}));
                    break;
                case ApiFormat::ANTHROPIC:
                    if (active_kind_ != "thinking") {
                        out.push_back(sse_event("content_block_stop",
                            json({{"type", "content_block_stop"}, {"index", block_index_}}).dump()));
                        block_index_++;
                        active_kind_ = "thinking";
                        json new_block = {{"type", "thinking"}, {"thinking", ""}};
                        out.push_back(sse_event("content_block_start",
                            json({{"type", "content_block_start"}, {"index", block_index_},
                                  {"content_block", new_block}}).dump()));
                    }
                    out.push_back(sse_event("content_block_delta",
                        json({{"type", "content_block_delta"}, {"index", block_index_},
                              {"delta", {{"type", "thinking_delta"}, {"thinking", safe}}}}).dump()));
                    break;
                case ApiFormat::RESPONSES:
                    break;
                default: break;
                }
                window_ = window_.substr(cut);
            }
            break;
        }

        // mode_ == StreamMode::CONTENT
        // Look for <think>, </think>, or supported tool-call starts.
        size_t think_idx = window_.find(THINK_OPEN);
        size_t think_close_idx = window_.find(THINK_CLOSE);
        size_t tool_idx = std::string::npos;
        bool tool_hit = has_request_tools(tools_) &&
                        find_tool_start(window_, tool_idx);

        struct Hit { size_t pos; int type; };  // type: 0=think, 1=think_close, 2=tool-ish
        std::vector<Hit> hits;
        if (think_idx != std::string::npos)       hits.push_back({think_idx, 0});
        if (think_close_idx != std::string::npos) hits.push_back({think_close_idx, 1});
        if (tool_hit)                             hits.push_back({tool_idx, 2});

        if (!hits.empty()) {
            std::sort(hits.begin(), hits.end(),
                      [](const Hit & a, const Hit & b) { return a.pos < b.pos; });
            auto & h = hits[0];
            std::string pre = window_.substr(0, h.pos);
            if (!pre.empty()) {
                accumulated_content_ += pre;
                emit_content_delta(out, pre);
            }

            if (h.type == 0) {
                // <think>
                window_ = window_.substr(h.pos + THINK_OPEN_LEN);
                mode_ = StreamMode::REASONING;
            } else if (h.type == 1) {
                // </think> in content — just skip it
                window_ = window_.substr(h.pos + THINK_CLOSE_LEN);
            } else {
                // Tool-call syntax. Keep the full tag/function text buffered
                // until finish so the parser can validate it.
                tool_buffer_ = window_.substr(h.pos);
                window_.clear();
                mode_ = StreamMode::TOOL_BUFFER;
            }
            continue;
        }

        if (accumulated_content_.find_first_not_of(" \t\n\r") == std::string::npos &&
            starts_with_potential_bare_json_tool(window_, tools_)) {
            tool_buffer_ = window_;
            tool_buffer_fallback_to_content_ = true;
            window_.clear();
            mode_ = StreamMode::TOOL_BUFFER;
            continue;
        }

        // No tags found — emit safe prefix
        if (window_.size() > std::max(BASE_HOLDBACK, stop_holdback_)) {
            size_t cut = utf8_safe_len(window_, window_.size() - std::max(BASE_HOLDBACK, stop_holdback_));
            // When tools are declared, a trailing identifier run may be a
            // Laguna tool name whose <arg_key> has not streamed in yet (the
            // <tool_call> wrapper is a stripped special token). Hold it back
            // (bounded to 64 chars, the OpenAI function-name limit) so the
            // name is still in the window when the trigger fires.
            if (cut > 0 && has_request_tools(tools_)) {
                auto is_ident = [](char c) {
                    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '_' || c == '-';
                };
                size_t name_hold = cut;
                while (name_hold > 0 && cut - name_hold < 64 &&
                       is_ident(window_[name_hold - 1])) name_hold--;
                if (cut - name_hold < 64) cut = name_hold;
            }
            if (cut > 0) {
                std::string safe = window_.substr(0, cut);
                accumulated_content_ += safe;
                emit_content_delta(out, safe);
                window_ = window_.substr(cut);
            }
        }
        break;
    }

    return out;
}

// ─── Content delta emission (format-specific) ───────────────────────────

void SseEmitter::emit_content_delta(std::vector<std::string> & out,
                                    const std::string & text) {
    if (text.empty()) return;

    switch (format_) {
    case ApiFormat::OPENAI_CHAT:
        out.push_back(format_openai_delta({{"content", text}}));
        break;

    case ApiFormat::ANTHROPIC:
        if (active_kind_ != "text") {
            out.push_back(sse_event("content_block_stop",
                json({{"type", "content_block_stop"}, {"index", block_index_}}).dump()));
            block_index_++;
            active_kind_ = "text";
            json new_block = {{"type", "text"}, {"text", ""}};
            out.push_back(sse_event("content_block_start",
                json({{"type", "content_block_start"}, {"index", block_index_},
                      {"content_block", new_block}}).dump()));
        }
        out.push_back(sse_event("content_block_delta",
            json({{"type", "content_block_delta"}, {"index", block_index_},
                  {"delta", {{"type", "text_delta"}, {"text", text}}}}).dump()));
        break;

    case ApiFormat::RESPONSES:
        out.push_back(format_responses_event("response.output_text.delta", {
            {"item_id", msg_item_id_}, {"output_index", 0},
            {"content_index", 0}, {"delta", text}
        }));
        break;

    default:
        break;
    }
}

// ─── emit_finish ────────────────────────────────────────────────────────

std::vector<std::string> SseEmitter::emit_finish(int completion_tokens,
                                                 const GenTimings * timings) {
    std::vector<std::string> out;

    // Flush remaining window
    if (mode_ == StreamMode::REASONING && !window_.empty()) {
        reasoning_text_ += window_;
        switch (format_) {
        case ApiFormat::OPENAI_CHAT:
            out.push_back(format_openai_delta({{"reasoning_content", window_}}));
            break;
        case ApiFormat::ANTHROPIC:
            out.push_back(sse_event("content_block_delta",
                json({{"type", "content_block_delta"}, {"index", block_index_},
                      {"delta", {{"type", "thinking_delta"}, {"thinking", window_}}}}).dump()));
            break;
        default: break;
        }
    } else if (mode_ == StreamMode::CONTENT && !window_.empty()) {
        // Zero-argument Laguna calls in the stripped form are just the bare
        // declared tool name at end of output (the <tool_call> wrapper is
        // special tokens the detokenizer removed, and with no <arg_key> there
        // is no trigger). If the output ENDS on exactly a declared tool name,
        // treat it as a zero-arg call instead of trailing prose.
        bool zero_arg_call = false;
        if (has_request_tools(tools_)) {
            auto is_ident = [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '-';
            };
            size_t name_start = window_.size();
            while (name_start > 0 && is_ident(window_[name_start - 1])) name_start--;
            const std::string tail = window_.substr(name_start);
            if (!tail.empty()) {
                for (const auto & t : tools_) {
                    if (t.contains("function") && t["function"].value("name", "") == tail) {
                        std::string pre = window_.substr(0, name_start);
                        if (!pre.empty()) {
                            accumulated_content_ += pre;
                            emit_content_delta(out, pre);
                        }
                        // Re-wrap in the canonical form the parser's wrapped
                        // path accepts (it emits name-only bodies as
                        // zero-argument calls).
                        tool_buffer_ = "<tool_call>" + tail + "</tool_call>";
                        mode_ = StreamMode::TOOL_BUFFER;
                        zero_arg_call = true;
                        break;
                    }
                }
            }
        }
        if (!zero_arg_call) {
            accumulated_content_ += window_;
            emit_content_delta(out, window_);
        }
    } else if (mode_ == StreamMode::TOOL_BUFFER) {
        tool_buffer_ += window_;
    }
    window_.clear();

    // Parse tool calls from buffer
    std::string fr = "stop";
    if (mode_ == StreamMode::TOOL_BUFFER && !tool_buffer_.empty()) {
        auto parsed = parse_tool_calls(tool_buffer_, tools_);
        tool_calls_ = std::move(parsed.tool_calls);

        if (!tool_calls_.empty()) {
            // Remember for tool memory
            if (tool_memory_) {
                std::vector<std::string> ids;
                for (const auto & tc : tool_calls_) ids.push_back(tc.id);
                tool_memory_->remember(ids, accumulated_raw_);
            }

            // Emit any cleaned text from the tool buffer
            if (!parsed.cleaned_text.empty()) {
                accumulated_content_ += parsed.cleaned_text;
                emit_content_delta(out, parsed.cleaned_text);
            }

            fr = "tool_calls";

            // Format-specific tool call events
            switch (format_) {
            case ApiFormat::OPENAI_CHAT: {
                json tc_list = json::array();
                for (size_t i = 0; i < tool_calls_.size(); i++) {
                    tc_list.push_back({
                        {"index", (int)i},
                        {"id", tool_calls_[i].id},
                        {"type", "function"},
                        {"function", {
                            {"name", tool_calls_[i].name},
                            {"arguments", tool_calls_[i].arguments}
                        }}
                    });
                }
                out.push_back(format_openai_delta({{"tool_calls", tc_list}}));
                break;
            }
            case ApiFormat::ANTHROPIC: {
                // Anthropic tool_use is emitted as separate content blocks.
                // Lifecycle per tool: close the open text/thinking block via
                // content_block_stop, then for each tool_call emit
                // content_block_start { type: tool_use, id, name, input: {} },
                // a content_block_delta { type: input_json_delta, partial_json }
                // carrying the JSON arguments, finally content_block_stop.
                //
                // The initial text/thinking block at index_=0 was opened by
                // emit_start(); we close it now and bump block_index_ for
                // each tool_use block we emit.
                if (!active_kind_.empty()) {
                    out.push_back(sse_event("content_block_stop",
                        json({{"type", "content_block_stop"}, {"index", block_index_}}).dump()));
                    active_kind_.clear();
                }
                for (const auto & tc : tool_calls_) {
                    block_index_++;
                    json tu_block = {
                        {"type",  "tool_use"},
                        {"id",    tc.id},
                        {"name",  tc.name},
                        {"input", json::object()}
                    };
                    out.push_back(sse_event("content_block_start",
                        json({{"type", "content_block_start"},
                              {"index", block_index_},
                              {"content_block", tu_block}}).dump()));
                    // Single-shot delta with the full arguments JSON. Clients
                    // parse this incrementally; emitting it whole is spec-legal
                    // and avoids partial-JSON splitting heuristics.
                    if (!tc.arguments.empty()) {
                        out.push_back(sse_event("content_block_delta",
                            json({{"type",  "content_block_delta"},
                                  {"index", block_index_},
                                  {"delta", {{"type",         "input_json_delta"},
                                             {"partial_json", tc.arguments}}}}).dump()));
                    }
                    out.push_back(sse_event("content_block_stop",
                        json({{"type", "content_block_stop"},
                              {"index", block_index_}}).dump()));
                }
                break;
            }
            case ApiFormat::RESPONSES:
                for (const auto & tc : tool_calls_) {
                    out.push_back(format_responses_event(
                        "response.function_call_arguments.delta", {
                            {"item_id", tc.id}, {"output_index", 0},
                            {"delta", tc.arguments}
                        }));
                    out.push_back(format_responses_event(
                        "response.function_call_arguments.done", {
                            {"item_id", tc.id}, {"output_index", 0},
                            {"arguments", tc.arguments}, {"name", tc.name}
                        }));
                }
                break;
            default: break;
            }
        } else if (tool_buffer_fallback_to_content_) {
            accumulated_content_ += tool_buffer_;
            emit_content_delta(out, tool_buffer_);
        } else {
            // Tool syntax was detected but no valid call parsed. Do not leak
            // malformed/incomplete XML back to the user as assistant text.
            std::fprintf(stderr,
                "[server] tool_call parse failed; suppressing buffered tool text "
                "request_id=%s format=%d bytes=%zu\n",
                request_id_.c_str(), (int)format_, tool_buffer_.size());
        }
    }

    // Format-specific final events
    switch (format_) {
    case ApiFormat::OPENAI_CHAT: {
        // Finish reason chunk
        out.push_back(format_openai_delta(json::object(), fr.c_str()));
        // Usage chunk — includes timings sub-object when caller supplied
        // generation wall-clock breakdown (see spec §6.3).
        json usage_body = {
            {"prompt_tokens", prompt_tokens_},
            {"completion_tokens", completion_tokens},
            {"total_tokens", prompt_tokens_ + completion_tokens}
        };
        if (timings) {
            usage_body["timings"] = build_timings_json(*timings, completion_tokens);
        }
        json usage = {
            {"id", request_id_}, {"object", "chat.completion.chunk"},
            {"created", created_at_}, {"model", model_name_},
            {"choices", json::array()},
            {"usage", usage_body}
        };
        out.push_back(sse_data(usage.dump()));
        out.push_back(sse_data("[DONE]"));
        break;
    }

    case ApiFormat::ANTHROPIC: {
        // content_block_stop only fires if a block is still open. With the
        // tool_use emission added above, the last text/thinking/tool_use
        // block may already be closed — in that case active_kind_ is empty
        // and we skip the redundant close (idempotent, but some Anthropic
        // SDK clients raise parse errors on duplicate stops at the same
        // index).
        if (!active_kind_.empty()) {
            out.push_back(sse_event("content_block_stop",
                json({{"type", "content_block_stop"}, {"index", block_index_}}).dump()));
            active_kind_.clear();
        }
        // stop_reason reflects the model's actual finish: "tool_use" when
        // any tool calls were emitted (downstream SDKs pivot on this to feed
        // tool_result back), else "end_turn". Stop-sequence hits also report
        // "end_turn" (Anthropic has no dedicated reason for that case).
        const char * stop_reason = tool_calls_.empty() ? "end_turn" : "tool_use";
        json anth_usage = {{"output_tokens", completion_tokens}};
        if (timings) {
            anth_usage["timings"] = build_timings_json(*timings, completion_tokens);
        }
        json msg_delta = {
            {"type", "message_delta"},
            {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
            {"usage", anth_usage}
        };
        out.push_back(sse_event("message_delta", msg_delta.dump()));
        // message_stop
        out.push_back(sse_event("message_stop",
            json({{"type", "message_stop"}}).dump()));
        break;
    }

    case ApiFormat::RESPONSES: {
        // output_text.done
        out.push_back(format_responses_event("response.output_text.done", {
            {"item_id", msg_item_id_}, {"output_index", 0},
            {"content_index", 0}, {"text", accumulated_content_}
        }));
        // content_part.done
        out.push_back(format_responses_event("response.content_part.done", {
            {"item_id", msg_item_id_}, {"output_index", 0},
            {"content_index", 0},
            {"part", {{"type", "output_text"}, {"text", accumulated_content_},
                      {"annotations", json::array()}}}
        }));

        // Build final output items
        json final_output = json::array();
        if (!tool_calls_.empty()) {
            for (const auto & tc : tool_calls_) {
                final_output.push_back({
                    {"type", "function_call"}, {"id", tc.id},
                    {"status", "completed"}, {"call_id", tc.id},
                    {"name", tc.name}, {"arguments", tc.arguments}
                });
            }
        } else {
            final_output.push_back({
                {"type", "message"}, {"id", msg_item_id_},
                {"status", "completed"}, {"role", "assistant"},
                {"content", json::array({{
                    {"type", "output_text"}, {"text", accumulated_content_},
                    {"annotations", json::array()}
                }})}
            });
        }

        // output_item.done for each item
        for (size_t i = 0; i < final_output.size(); i++) {
            out.push_back(format_responses_event("response.output_item.done", {
                {"output_index", (int)i},
                {"item", final_output[i]}
            }));
        }

        // response.completed
        json resp_usage = {
            {"input_tokens", prompt_tokens_},
            {"output_tokens", completion_tokens},
            {"total_tokens", prompt_tokens_ + completion_tokens}
        };
        if (timings) {
            resp_usage["timings"] = build_timings_json(*timings, completion_tokens);
        }
        json shell = {
            {"id", request_id_}, {"object", "response"},
            {"created_at", created_at_}, {"status", "completed"},
            {"model", model_name_},
            {"output", final_output},
            {"output_text", accumulated_content_},
            {"usage", resp_usage}
        };
        out.push_back(format_responses_event("response.completed", {{"response", shell}}));
        break;
    }

    default:
        out.push_back(sse_data("[DONE]"));
        break;
    }

    return out;
}

std::string SseEmitter::finish_reason() const {
    if (!tool_calls_.empty()) return "tool_calls";
    return "stop";
}

}  // namespace dflash::common
