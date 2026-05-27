// Chat template renderer for dflash::common native server.
//
// Renders chat messages (system/user/assistant/tool) into the model-specific
// token format. Hard-coded for supported architectures:
//   - Qwen3/3.5: <|im_start|>role\ncontent<|im_end|>\n
//   - Laguna: XML-style <|begin_of_sentence|><|User|>...<|Assistant|>

#pragma once

#include <string>
#include <vector>

namespace dflash::common {

// A single message in a chat conversation.
struct ChatMessage {
    std::string role;       // "system", "user", "assistant", "tool"
    std::string content;    // message text
    // Optional tool_call_id for tool result messages.
    std::string tool_call_id;
};

// Chat template format.
enum class ChatFormat {
    QWEN3,     // <|im_start|>role\n...<|im_end|>\n
    LAGUNA,    // <|begin_of_sentence|><|User|>...<|Assistant|>
    GEMMA4,    // <bos><|turn>role\n...<turn|>\n
};

// Render chat messages into the model-specific prompt string.
// The result is plain text ready to be tokenized.
//
// If `add_generation_prompt` is true, appends the assistant turn prefix
// at the end (so the model starts generating as assistant).
//
// `enable_thinking` controls Qwen3/3.5 think mode:
//   true  → assistant starts open-ended (model will produce <think>...</think>)
//   false → assistant starts with <think>\n\n</think>\n\n (skip thinking)
//
// `tools_json` is an optional JSON string containing the tool definitions
// array. When non-empty, the Qwen3/3.5 template injects a tool preamble
// into the system message instructing the model how to emit <tool_call> tags.
std::string render_chat_template(
    const std::vector<ChatMessage> & messages,
    ChatFormat format,
    bool add_generation_prompt = true,
    bool enable_thinking = false,
    const std::string & tools_json = "");

// Detect the appropriate chat format for an architecture.
ChatFormat chat_format_for_arch(const std::string & arch);

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
// `tools_json`   optional JSON array of tool definitions; when non-empty it
//                is parsed and injected as `tools` into the template context.
// `arch_hint`    model architecture (controls arch-specific post-processing;
//                the closed-think prefill injection is Qwen3/3.5/3.6 only).
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
    ChatFormat arch_hint = ChatFormat::QWEN3);

}  // namespace dflash::common
