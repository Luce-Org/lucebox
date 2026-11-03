// The client APIs' side of a request (OpenAI Chat Completions, Anthropic
// Messages, OpenAI Responses): their messages normalized into the chat
// messages the renderers take, and a generated reply built into each API's
// non-streaming response.
#pragma once

#include "http_server.h"

#include <string>
#include <vector>

namespace luce::common {

// Anthropic's top-level `system` (a string or text blocks, Claude Code's
// billing header dropped) as a leading system message on `messages`.
void normalize_anthropic_system(const json & body, json & messages);

// Anthropic tool_use `input` objects keep the client's key order: `json`
// sorts keys, so each becomes its JSON text, read from `ordered_messages`
// (the body's messages parsed in order). `messages` may hold a leading system
// message that the body's messages do not.
void keep_anthropic_input_order(const nlohmann::ordered_json & ordered_messages, json & messages);

// The request's messages, in any API's format, as chat messages. A tool-call
// turn found in `tool_memory` (and `replay_tool_memory`) comes back as the
// text the model generated. `text_part_separator` joins a message's text
// parts, as the model's template joins them.
std::vector<ChatMessage> normalize_chat_messages(
    const json & messages,
    ApiFormat format,
    ToolMemory & tool_memory,
    bool replay_tool_memory = true,
    const char * text_part_separator = "");

struct CompletionTokenCounts {
    int total = 0;
    int reasoning = 0;
    int content = 0;
};

enum class TokenDelivery {
    kSkip,      // EOS or internal control marker — never reaches the emitter
    kThinkTag,  // thinking-boundary marker mapped to its <think> text form
    kText,      // ordinary text (token_text may still be empty)
};

// Classifies one generated token for delivery to the SseEmitter, filling
// `text` with the deliverable form for kThinkTag / kText. Shared by the
// streaming on_token callback and the non-streaming replay: the two paths
// MUST classify identically, or the reasoning/content split diverges
// between streamed and non-streamed responses.

// Classifies one generated token for delivery to the SseEmitter, filling
// `text` with the deliverable form for kThinkTag / kText. The streaming and
// non-streaming paths both call it, so they split reasoning and content alike.
TokenDelivery classify_generated_token(
        Tokenizer & tokenizer, SseEmitter & emitter,
        int32_t token, std::string & text);

// Replays a finished generation through `emitter` (the non-streaming path)
// and counts its reasoning and content tokens.
CompletionTokenCounts feed_non_streaming_tokens(
        const std::vector<int32_t> & tokens, Tokenizer & tokenizer,
        SseEmitter & emitter, int generation_cap);

// The non-streaming response body in the request's API format.
json build_non_streaming_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings,
        const CompletionTokenCounts & counts, SseEmitter & emitter,
        const Tokenizer * tokenizer = nullptr);
json build_non_streaming_response(
        const ParsedRequest & req, const GenerateResult & result,
        int generation_cap, const GenTimings & timings, Tokenizer & tokenizer,
        SseEmitter & emitter);

}  // namespace luce::common
