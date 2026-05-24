// SSE stream emitter — format-specific SSE event sequences.
//
// Encapsulates the streaming state machine (reasoning/content/tool_buffer modes)
// and emits correctly formatted SSE events for OpenAI, Anthropic, and Responses APIs.

#pragma once

#include "tool_parser.h"
#include "tool_memory.h"
#include "reasoning.h"
#include "api_types.h"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dflash::common {

using json = nlohmann::json;

// Callback to send an SSE chunk. Returns false if client disconnected.
using SseSendFn = std::function<bool(const std::string & data)>;

// Callback to send an SSE event with event type (for Anthropic/Responses).
// Format: "event: {type}\ndata: {data}\n\n"
using SseEventFn = std::function<bool(const std::string & event_type,
                                      const std::string & data)>;

// Stream state machine modes
enum class StreamMode { REASONING, CONTENT, TOOL_BUFFER };

// Manages SSE streaming for a single request.
class SseEmitter {
public:
    SseEmitter(ApiFormat format,
               const std::string & request_id,
               const std::string & model_name,
               int prompt_tokens,
               const json & tools,
               ToolMemory * tool_memory,
               bool started_in_thinking,
               const std::vector<std::string> & stop_sequences = {});

    // Emit the initial SSE events (role delta, message_start, etc.)
    // Returns the formatted SSE strings to send.
    std::vector<std::string> emit_start();

    // Process a text token and return SSE chunks to send.
    std::vector<std::string> emit_token(const std::string & piece);

    // Flush remaining buffered content and emit final events.
    // `completion_tokens` is the total token count.
    std::vector<std::string> emit_finish(int completion_tokens);

    // Get the finish_reason for non-streaming responses.
    std::string finish_reason() const;

    // Check if a stop sequence was hit (signals caller to stop generation).
    bool stop_hit() const { return stop_hit_; }

    // Get accumulated content (for non-streaming).
    const std::string & accumulated_text() const { return accumulated_content_; }

    // Get the parsed tool calls (after emit_finish).
    const std::vector<ToolCall> & tool_calls() const { return tool_calls_; }

    // Get the reasoning text (after emit_finish).
    const std::string & reasoning_text() const { return reasoning_text_; }

    // Current stream mode — callers tracking per-mode token counts use
    // this to attribute a token to either REASONING or CONTENT. Sampled
    // before each emit_token() call so tokens that span a </think>
    // transition are attributed to the mode they entered with.
    StreamMode mode() const { return mode_; }

private:
    // Format helpers
    std::string format_openai_delta(const json & delta, const char * finish = nullptr);
    std::string format_anthropic_event(const std::string & event_type, const json & data);
    std::string format_responses_event(const std::string & event_type, const json & data);

    // Emit a content delta (format-specific).
    void emit_content_delta(std::vector<std::string> & out, const std::string & text);

    // SSE data line
    static std::string sse_data(const std::string & json_str);
    static std::string sse_event(const std::string & type, const std::string & json_str);

    ApiFormat    format_;
    std::string  request_id_;
    std::string  model_name_;
    int          prompt_tokens_;
    json         tools_;
    ToolMemory * tool_memory_;

    StreamMode   mode_;
    std::string  window_;           // holdback buffer
    std::string  tool_buffer_;      // accumulated tool text
    std::string  accumulated_content_;
    std::string  accumulated_raw_;  // all raw text for tool memory
    std::string  reasoning_text_;
    std::vector<ToolCall> tool_calls_;

    // Anthropic block tracking
    int          block_index_ = 0;
    std::string  active_kind_;  // "thinking" or "text"

    // Strip leading <think> tag from reasoning (ds4 pattern).
    bool         checked_think_prefix_ = false;

    // Stop sequences support
    std::vector<std::string> stop_sequences_;
    size_t       stop_holdback_ = 0;  // max length of any stop sequence
    bool         stop_hit_ = false;

    int64_t      created_at_;

    // Responses API IDs
    std::string  msg_item_id_;

    static constexpr size_t BASE_HOLDBACK = 12;  // max(len("<tool_call>"), len("</think>"), len("<think>"))
};

}  // namespace dflash::common
