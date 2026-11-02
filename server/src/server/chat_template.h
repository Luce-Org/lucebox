// Chat template renderer for luce::common native server.
//
// Renders chat messages (system/user/assistant/tool) into the model-specific
// token format. Hard-coded for supported architectures:
//   - BailingMoE3: <role>SYSTEM/HUMAN/ASSISTANT</role>...<|role_end|>
//   - Laguna: XML-style role blocks
//   - Gemma4: <|turn>role\ncontent<turn|>\n
//
// The Qwen and DeepSeek families have no renderer here: they render with
// their chat codec (chat_codec.h) or a --chat-template-file.

#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace luce::common {

// Model-independent assistant call. Arguments are decoded once on a memory
// miss, in the client's key order; renderers choose the model's native syntax.
struct ChatToolCall {
    std::string id;
    std::string name;
    nlohmann::ordered_json arguments;
};

// A single message in a chat conversation.
struct ChatMessage {
    std::string role;       // "system", "user", "assistant", "tool"
    std::string content;    // message text
    // Optional tool_call_id for tool result messages.
    std::string tool_call_id;
    // Optional prior <think> text for assistant messages (OpenAI-compatible
    // `reasoning_content` / `reasoning` fields on request history). Jinja
    // templates get it as `message.reasoning_content` and decide whether to
    // replay it inside <think>...</think> (e.g. qwen4exp); the chat codecs
    // replay it where their official templates do.
    std::string reasoning_content;
    // The client sent a reasoning field (even an empty one): it is part of the
    // turn as sent, so turn memory must match it.
    bool reasoning_sent = false;
    std::vector<ChatToolCall> tool_calls;
    // On a memory hit, content is the original raw turn. Calls retain only
    // their IDs/names for result association; do not render them a second time.
    bool tool_calls_replayed = false;
    std::string tool_name;  // Result's function name, resolved from tool_call_id.
};

// Chat template format.
enum class ChatFormat {
    BAILINGMOE3, // <role>SYSTEM/HUMAN/ASSISTANT</role>...<|role_end|>
    LAGUNA,    // <|begin_of_sentence|><|User|>...<|Assistant|>
    GEMMA4,    // <bos><|turn>role\n...<turn|>\n
};

// Render chat messages into the model-specific prompt string.
// The result is plain text ready to be tokenized.
//
// If `add_generation_prompt` is true, appends the assistant turn prefix
// at the end (so the model starts generating as assistant).
//
// `enable_thinking` selects the template's thinking mode.
//
// `tools_json` is an optional JSON string containing the tool definitions
// array. Each renderer injects the corresponding native tool preamble.
std::string render_chat_template(
    const std::vector<ChatMessage> & messages,
    ChatFormat format,
    bool add_generation_prompt = true,
    bool enable_thinking = false,
    const std::string & tools_json = "");

// The built-in renderer of an architecture; none for those that render only
// with their chat codec (the Qwen and DeepSeek families).
std::optional<ChatFormat> chat_format_for_arch(const std::string & arch);

// Render chat messages via a Jinja chat template (e.g. froggeric Qwen3.6
// template, or any of the llama.cpp models/templates/*.jinja files).
//
// Mirrors llama.cpp's common_chat_template_direct_apply: parses the template
// once per thread, converts inputs to jinja values, runs the program, returns
// the rendered prompt string.
//
// `template_src`  literal Jinja source (read from --chat-template-file)
// `bos_token`,
// `eos_token`    passed through to the template (Qwen3.6 templates may use
//                {{bos_token}} / {{eos_token}}). Use empty strings if unknown.
// `tools_json`   optional JSON array of tool definitions, normalized from
//                Chat Completions, Responses, or Anthropic to tools[].function.
// `reasoning_effort` optional; injected as `reasoning_effort` when non-empty.
// `preserve_thinking` tri-state: -1 leaves the template variable undefined
//                (so the template's own default — typically true — applies);
//                0/1 inject it as the `preserve_thinking` boolean. Official
//                templates (e.g. qwen4exp) use this to decide whether earlier
//                assistant turns replay their recorded <think> block
//                (message.reasoning_content) or render with it stripped.
//
// Messages must contain structured tool calls, not raw tool-memory replay.
// function.arguments is exposed as an object, and result messages retain
// tool_call_id and name. Raw-replayed messages throw std::logic_error.
//
// Internally caches the most recently parsed program per thread (avoids
// re-parsing the template on every request). Throws std::runtime_error on
// lexer/parser/runtime failure (caller should surface a 500 response).
std::string render_chat_template_jinja(
    const std::string & template_src,
    const std::vector<ChatMessage> & messages,
    const std::string & bos_token,
    const std::string & eos_token,
    bool add_generation_prompt = true,
    bool enable_thinking = false,
    const std::string & tools_json = "",
    const std::string & reasoning_effort = "",
    int preserve_thinking = -1);

// Tool definitions as official templates expect them: Chat Completions'
// {type, function: {name, description, parameters}}, from any API's shape
// (Responses flattens the function; Anthropic names the schema input_schema).
// Key order is kept.
nlohmann::ordered_json openai_tool_definitions(nlohmann::ordered_json tools);

// Python json.dumps(value, ensure_ascii=False): the Hugging Face templates'
// |tojson. ", " and ": " separators, keys in the value's order.
std::string python_json_dumps(const nlohmann::ordered_json & value);

// Qwen3.8-Flash-Next's template knows the efforts low, medium and xhigh (its default); Lucebox's high, x-high and
// max map to xhigh. Empty stays empty so the template applies its own default.
std::string qwen4exp_template_effort(const std::string & effort);

}  // namespace luce::common
