// Chat template renderer for luce::common native server.
//
// Renders chat messages (system/user/assistant/tool) into the model-specific
// token format. Hard-coded for supported architectures:
//   - Qwen3/3.5: <|im_start|>role\ncontent<|im_end|>\n
//   - BailingMoE3: <role>SYSTEM/HUMAN/ASSISTANT</role>...<|role_end|>
//   - Laguna: XML-style role blocks

#pragma once

#include <string>
#include <vector>

namespace luce::common {

// A single message in a chat conversation.
struct ChatMessage {
    std::string role;       // "system", "user", "assistant", "tool"
    std::string content;    // message text
    // Optional tool_call_id for tool result messages.
    std::string tool_call_id;
    // Optional prior <think> text for assistant messages (OpenAI-compatible
    // `reasoning_content` / `reasoning` fields on request history). Only the
    // Jinja renderer consumes this — exposed to templates as
    // `message.reasoning_content` so official templates (e.g. qwen4exp) can
    // decide whether to replay it inside <think>...</think>.
    std::string reasoning_content;
};

// Chat template format.
enum class ChatFormat {
    QWEN3,     // <|im_start|>role\n...<|im_end|>\n
    BAILINGMOE3, // <role>SYSTEM/HUMAN/ASSISTANT</role>...<|role_end|>
    LAGUNA,    // <|begin_of_sentence|><|User|>...<|Assistant|>
    GEMMA4,    // <bos><|turn>role\n...<turn|>\n
    DEEPSEEK4, // <｜begin▁of▁sentence｜>...<｜User｜>...<｜Assistant｜>
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
//
// `reasoning_effort` is the normalized model-facing effort. DeepSeek V4 uses
// low, high, and max; high and max prepend the official encoding prefixes.
std::string render_chat_template(
    const std::vector<ChatMessage> & messages,
    ChatFormat format,
    bool add_generation_prompt = true,
    bool enable_thinking = false,
    const std::string & tools_json = "",
    const std::string & reasoning_effort = "");

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
// `reasoning_effort` optional; injected as `reasoning_effort` when non-empty.
// `preserve_thinking` tri-state: -1 leaves the template variable undefined
//                (so the template's own default — typically true — applies);
//                0/1 inject it as the `preserve_thinking` boolean. Official
//                templates (e.g. qwen4exp) use this to decide whether earlier
//                assistant turns replay their recorded <think> block
//                (message.reasoning_content) or render with it stripped.
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

// Qwen3.8-Flash-Next's template knows the efforts low, medium and xhigh (its default); Lucebox's high, x-high and
// max map to xhigh. Empty stays empty so the template applies its own default.
std::string qwen4exp_template_effort(const std::string & effort);

}  // namespace luce::common
