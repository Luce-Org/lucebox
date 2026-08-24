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
static constexpr size_t THINK_OPEN_LEN  = 7;
static constexpr size_t THINK_CLOSE_LEN = 8;

static bool has_request_tools(const json & tools) {
    return tools.is_array() && !tools.empty();
}

static bool starts_with_potential_bare_json_tool(const std::string & text,
                                                 const json & tools) {
    if (!has_request_tools(tools)) return false;
    size_t first = text.find_first_not_of(" \t\n\r");
    return first != std::string::npos && text[first] == '{';
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
        {"decode_tokens_per_sec", tps},
        {"cache_hit",             t.cache_hit},
        {"cached_prefix_tokens",  t.cached_prefix_tokens},
        {"prefilled_tokens",      t.prefilled_tokens},
        {"effective_prompt_tokens", t.effective_prompt_tokens},
        {"agent_turn_cache_hit",  t.agent_turn_cache_hit}
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
    // First re-join any incomplete multi-byte tail held back from the previous
    // piece, then hold back a new incomplete tail (if any) so codepoints split
    // across tokens are emitted intact instead of as U+FFFD pairs.
    std::string joined = utf8_tail_ + raw_piece;
    utf8_tail_.clear();
    {
        size_t i = joined.size();
        int cont = 0;
        while (i > 0 && cont < 3 &&
               (static_cast<unsigned char>(joined[i - 1]) & 0xC0) == 0x80) {
            --i; ++cont;
        }
        if (i > 0) {
            const unsigned char lead = static_cast<unsigned char>(joined[i - 1]);
            int need = 0;
            if ((lead & 0xE0) == 0xC0) need = 2;
            else if ((lead & 0xF0) == 0xE0) need = 3;
            else if ((lead & 0xF8) == 0xF0) need = 4;
            if (need > 0 && joined.size() - (i - 1) < static_cast<size_t>(need)) {
                utf8_tail_ = joined.substr(i - 1);
                joined.resize(i - 1);
            }
        }
    }
    std::string piece = utf8_sanitize(joined);
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
            if (tool_from_reasoning_ && first_content_token_index_ < 0) {
                const std::string full = tool_buffer_ + window_;
                const size_t fc_close = full.find("</function_calls>");
                if (fc_close != std::string::npos) {
                    const size_t search_start = fc_close + std::strlen("</function_calls>");
                    const size_t think_close = full.find(THINK_CLOSE, search_start);
                    if (think_close != std::string::npos) {
                        const size_t after_think = think_close + THINK_CLOSE_LEN;
                        if (after_think < full.size() &&
                            full.find_first_not_of(" \t\r\n", after_think) != std::string::npos) {
                            // The current token already carries content after </think>
                            first_content_token_index_ = emit_token_count_ - 1;
                        } else {
                            // First real content token starts on the next token
                            first_content_token_index_ = emit_token_count_;
                        }
                    }
                }
            }
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
            size_t tool_idx = std::string::npos;
            bool tool_hit = has_request_tools(tools_) &&
                            find_tool_syntax_start(window_, tools_, tool_idx);

            if (idx != std::string::npos && (tool_idx == std::string::npos || idx < tool_idx)) {
                std::string pre = window_.substr(0, idx);
                if (!pre.empty()) {
                    reasoning_text_ += pre;
                    emit_reasoning_delta(out, pre);
                }
                window_ = window_.substr(idx + THINK_CLOSE_LEN);
                mode_ = StreamMode::CONTENT;
                continue;
            }
            if (tool_hit) {
                std::string pre = window_.substr(0, tool_idx);
                if (!pre.empty()) {
                    reasoning_text_ += pre;
                    emit_reasoning_delta(out, pre);
                }
                tool_buffer_ = window_.substr(tool_idx);
                tool_from_reasoning_ = true;
                window_.clear();
                mode_ = StreamMode::TOOL_BUFFER;
                continue;
            }
            // No close tag yet — emit safe prefix if window is large enough
            const size_t holdback = std::max(tool_syntax_holdback(tools_), stop_holdback_);
            if (window_.size() > holdback) {
                size_t cut = utf8_safe_len(window_, window_.size() - holdback);
                if (cut == 0) break;  // not enough complete chars yet
                std::string safe = window_.substr(0, cut);
                reasoning_text_ += safe;
                emit_reasoning_delta(out, safe);
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
                        find_tool_syntax_start(window_, tools_, tool_idx);

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
                tool_from_reasoning_ = false;
                window_.clear();
                mode_ = StreamMode::TOOL_BUFFER;
            }
            continue;
        }

        if (accumulated_content_.find_first_not_of(" \t\n\r") == std::string::npos &&
            starts_with_potential_bare_json_tool(window_, tools_)) {
            tool_buffer_ = window_;
            tool_from_reasoning_ = false;
            tool_buffer_fallback_to_content_ = true;
            window_.clear();
            mode_ = StreamMode::TOOL_BUFFER;
            continue;
        }

        // No tags found — emit safe prefix
        const size_t holdback = std::max(tool_syntax_holdback(tools_),
                                         stop_holdback_);
        if (window_.size() > holdback) {
            size_t cut = utf8_safe_len(window_, window_.size() - holdback);
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

// ─── Reasoning delta emission (format-specific) ─────────────────────────

void SseEmitter::emit_reasoning_delta(std::vector<std::string> & out,
                                      const std::string & text) {
    if (text.empty()) return;

    switch (format_) {
    case ApiFormat::OPENAI_CHAT:
        out.push_back(format_openai_delta({{"reasoning_content", text}}));
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
                  {"delta", {{"type", "thinking_delta"}, {"thinking", text}}}}).dump()));
        break;

    case ApiFormat::RESPONSES:
        break;

    default:
        break;
    }
}

// ─── find_top_level_think_close ─────────────────────────────────────────

static size_t find_top_level_think_close(const std::string & buf, const json & tools) {
    bool in_single_quote = false;
    bool in_double_quote = false;
    std::vector<std::string> open_tags;

    auto is_word_char = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_';
    };

    auto is_tool_tag = [&](const std::string & name) {
        if (name == "tool_call" || name == "function_call" ||
            name == "function_calls" || name == "function" ||
            name == "invoke" || name == "parameter" || name == "param" ||
            name == "arguments" || name == "params" || name == "tool_code" ||
            name == "funcname") {
            return true;
        }
        if (tools.is_array()) {
            for (const auto & t : tools) {
                const auto & fn = t.contains("function") ? t["function"] : t;
                if (fn.is_object() && fn.value("name", "") == name) {
                    return true;
                }
            }
        }
        return false;
    };

    for (size_t i = 0; i < buf.size(); ) {
        if (buf[i] == '\\' && i + 1 < buf.size()) {
            i += 2;
            continue;
        }
        if (buf[i] == '\'' && !in_double_quote) {
            const bool apostrophe_in_word =
                i > 0 && i + 1 < buf.size() &&
                is_word_char(buf[i - 1]) && is_word_char(buf[i + 1]);
            if (!apostrophe_in_word) {
                in_single_quote = !in_single_quote;
            }
            i++;
            continue;
        }
        if (buf[i] == '"' && !in_single_quote) {
            in_double_quote = !in_double_quote;
            i++;
            continue;
        }

        if (!in_single_quote && !in_double_quote && buf[i] == '<') {
            // Check for </think>
            if (buf.compare(i, THINK_CLOSE_LEN, THINK_CLOSE) == 0) {
                // If not inside any active unclosed tool tags, this is top-level
                if (open_tags.empty()) {
                    return i;
                }
            }

            // Check for closing tag </tag_name>
            if (i + 1 < buf.size() && buf[i + 1] == '/') {
                size_t close_bracket = buf.find('>', i + 2);
                if (close_bracket != std::string::npos) {
                    std::string raw_name = buf.substr(i + 2, close_bracket - (i + 2));
                    size_t first = raw_name.find_first_not_of(" \t\r\n");
                    size_t last = raw_name.find_last_not_of(" \t\r\n");
                    std::string tag_name = (first == std::string::npos) ? "" : raw_name.substr(first, last - first + 1);
                    for (int idx = (int)open_tags.size() - 1; idx >= 0; --idx) {
                        if (open_tags[idx] == tag_name) {
                            open_tags.resize(idx);
                            break;
                        }
                    }
                    i = close_bracket + 1;
                    continue;
                }
            }

            // Check for opening tag <tag_name ...>
            size_t close_bracket = buf.find('>', i + 1);
            if (close_bracket != std::string::npos) {
                std::string tag_content = buf.substr(i + 1, close_bracket - (i + 1));
                size_t ws_pos = tag_content.find_first_of(" \t\r\n=");
                std::string tag_name = (ws_pos == std::string::npos) ? tag_content : tag_content.substr(0, ws_pos);
                if (is_tool_tag(tag_name)) {
                    if (!tag_content.empty() && tag_content.back() != '/') {
                        open_tags.push_back(tag_name);
                    }
                }
                i = close_bracket + 1;
                continue;
            }
        }
        i++;
    }

    return std::string::npos;
}

// ─── emit_finish ────────────────────────────────────────────────────────

std::vector<std::string> SseEmitter::emit_finish(int completion_tokens,
                                                 const GenTimings * timings,
                                                 int generation_cap,
                                                 bool ended_on_eos) {
    std::vector<std::string> out;

    // A tail still pending at end-of-stream is a genuinely truncated
    // codepoint; sanitize it into the window so nothing is silently lost.
    if (!utf8_tail_.empty()) {
        const std::string flushed = utf8_sanitize(utf8_tail_);
        utf8_tail_.clear();
        accumulated_raw_ += flushed;
        window_ += flushed;
    }

    // Flush remaining window
    if (mode_ == StreamMode::REASONING && !window_.empty()) {
        reasoning_text_ += window_;
        emit_reasoning_delta(out, window_);
    } else if (mode_ == StreamMode::CONTENT && !window_.empty()) {
        // Zero-argument Laguna calls in the stripped form are just the bare
        // declared tool name (the <tool_call> wrapper is special tokens the
        // detokenizer removed, and with no <arg_key> there is no trigger).
        // Only accept a tool-only content section: matching the final word of
        // ordinary prose would turn text such as "I cannot read" into a call.
        bool zero_arg_call = false;
        const char * ws = " \t\r\n";
        const bool prior_content_is_whitespace =
            accumulated_content_.find_first_not_of(ws) == std::string::npos;
        const size_t name_start = window_.find_first_not_of(ws);
        if (has_request_tools(tools_) && prior_content_is_whitespace &&
            name_start != std::string::npos) {
            const size_t name_end = window_.find_last_not_of(ws);
            const std::string candidate =
                window_.substr(name_start, name_end - name_start + 1);
            if (!candidate.empty()) {
                for (const auto & t : tools_) {
                    const auto & fn = t.contains("function")
                        ? t["function"] : t;
                    if (fn.is_object() &&
                        fn.value("name", "") == candidate) {
                        // Re-wrap in the canonical form the parser's wrapped
                        // path accepts (it emits name-only bodies as
                        // zero-argument calls).
                        tool_buffer_ =
                            "<tool_call>" + candidate + "</tool_call>";
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
                size_t think_close = parsed.cleaned_text.find(THINK_CLOSE);
                if (think_close != std::string::npos) {
                    std::string reasoning = parsed.cleaned_text.substr(0, think_close);
                    std::string content = parsed.cleaned_text.substr(think_close + THINK_CLOSE_LEN);
                    if (first_content_token_index_ == -1) {
                        first_content_token_index_ = content.empty() ? emit_token_count_ : std::max(0, emit_token_count_ - 1);
                    }
                    if (!reasoning.empty()) {
                        reasoning_text_ += reasoning;
                        emit_reasoning_delta(out, reasoning);
                    }
                    if (!content.empty()) {
                        accumulated_content_ += content;
                        emit_content_delta(out, content);
                    }
                } else if (tool_from_reasoning_) {
                    reasoning_text_ += parsed.cleaned_text;
                    emit_reasoning_delta(out, parsed.cleaned_text);
                } else {
                    accumulated_content_ += parsed.cleaned_text;
                    emit_content_delta(out, parsed.cleaned_text);
                }
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
            // malformed/incomplete XML back to the user or reasoning channel.
            std::fprintf(stderr,
                "[server] tool_call parse failed; suppressing buffered tool text "
                "request_id=%s format=%d bytes=%zu\n",
                request_id_.c_str(), (int)format_, tool_buffer_.size());

            if (tool_from_reasoning_) {
                size_t think_close = find_top_level_think_close(tool_buffer_, tools_);
                if (think_close != std::string::npos) {
                    std::string content = tool_buffer_.substr(think_close + THINK_CLOSE_LEN);
                    if (!content.empty()) {
                        if (first_content_token_index_ == -1) {
                            first_content_token_index_ = std::max(0, emit_token_count_ - 1);
                        }
                        accumulated_content_ += content;
                        emit_content_delta(out, content);
                    }
                }
            }
        }
    }

    if (fr == "stop" && !stop_hit_ && !ended_on_eos &&
        generation_cap >= 0 && completion_tokens >= generation_cap) {
        fr = "length";
    }
    finish_reason_ = fr;

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
        // tool_result back), else "end_turn" or "max_tokens".
        const char * stop_reason = tool_calls_.empty()
            ? (fr == "length" ? "max_tokens" : "end_turn")
            : "tool_use";
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
    return finish_reason_;
}

}  // namespace dflash::common
