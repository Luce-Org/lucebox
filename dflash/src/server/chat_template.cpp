// Chat template renderer implementation.

#include "chat_template.h"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <stdexcept>

namespace dflash::common {

// Qwen3.5 tool preamble — matches the official Jinja template exactly.
static const char QWEN3_TOOL_PREAMBLE[] =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";

static const char QWEN3_TOOL_SUFFIX[] =
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

ChatFormat chat_format_for_arch(const std::string & arch) {
    if (arch == "laguna") return ChatFormat::LAGUNA;
    if (arch == "gemma4") return ChatFormat::GEMMA4;
    // qwen35, qwen3 use the Qwen3/ChatML format
    return ChatFormat::QWEN3;
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
    case ChatFormat::QWEN3: {
        // Qwen3/3.5 ChatML format:
        //   <|im_start|>system\n[tool preamble +] content<|im_end|>\n
        //   <|im_start|>user\nHello<|im_end|>\n
        //   <|im_start|>assistant\n...

        // Determine if the first message is a system message.
        size_t start_idx = 0;
        std::string system_content;
        if (!messages.empty() && messages[0].role == "system") {
            system_content = messages[0].content;
            start_idx = 1;
        }

        // Emit system message with tool preamble if tools are present.
        if (has_tools) {
            result += "<|im_start|>system\n";
            result += QWEN3_TOOL_PREAMBLE;
            result += '\n';
            result += tools_json;
            result += QWEN3_TOOL_SUFFIX;
            if (!system_content.empty()) {
                result += "\n\n";
                result += system_content;
            }
            result += "<|im_end|>\n";
        } else if (!system_content.empty()) {
            result += "<|im_start|>system\n";
            result += system_content;
            result += "<|im_end|>\n";
        }

        // Render remaining messages.
        bool in_tool_response = false;
        for (size_t i = start_idx; i < messages.size(); i++) {
            const auto & msg = messages[i];

            if (msg.role == "tool") {
                // Qwen3.5 template: tool responses are grouped inside a user
                // message wrapped in <tool_response> tags.
                if (!in_tool_response) {
                    result += "<|im_start|>user";
                    in_tool_response = true;
                }
                result += "\n<tool_response>\n";
                result += msg.content;
                result += "\n</tool_response>";
                // Close user block if next message is not a tool message.
                bool next_is_tool = (i + 1 < messages.size() &&
                                     messages[i + 1].role == "tool");
                if (!next_is_tool) {
                    result += "<|im_end|>\n";
                    in_tool_response = false;
                }
            } else {
                result += "<|im_start|>";
                result += msg.role;
                result += '\n';
                result += msg.content;
                result += "<|im_end|>\n";
            }
        }

        if (add_generation_prompt) {
            result += "<|im_start|>assistant\n";
            if (!enable_thinking) {
                // Qwen3 thinking disabled: inject closed think block so the
                // model skips reasoning and generates the answer directly.
                result += "<think>\n\n</think>\n\n";
            } else {
                // Qwen3.6 enable_thinking: pre-open the thinking block so the
                // model actually enters reasoning mode. Verified against the
                // official Qwen3.6 chat_template.jinja:
                //   enable_thinking=true  → suffix `assistant\n<think>\n`
                //   enable_thinking=false → suffix `assistant\n<think>\n\n</think>\n\n`
                // Without this prefix, Qwen3.6 stays in non-thinking mode
                // even when the client opts in, defeating the thinking-budget
                // mechanism entirely (phase-2 reprompt never fires because
                // started_in_thinking=false).
                result += "<think>\n";
            }
        }
        break;
    }

    case ChatFormat::LAGUNA: {
        // Laguna/DeepSeek format:
        //   <｜begin▁of▁sentence｜>system content
        //   <｜User｜>user content<｜Assistant｜>
        result = "<｜begin▁of▁sentence｜>";
        for (const auto & msg : messages) {
            if (msg.role == "system") {
                result += msg.content;
            } else if (msg.role == "user") {
                result += "<｜User｜>";
                result += msg.content;
            } else if (msg.role == "assistant") {
                result += "<｜Assistant｜>";
                result += msg.content;
            } else if (msg.role == "tool") {
                // Tool results inline as user content
                result += "<｜User｜>[tool_result:" + msg.tool_call_id + "]\n";
                result += msg.content;
            }
        }
        if (add_generation_prompt) {
            result += "<｜Assistant｜>";
        }
        break;
    }

    case ChatFormat::GEMMA4: {
        // Gemma4 format:
        //   <bos><|turn>user\n{msg}<turn|>\n<|turn>model\n
        // System messages are prepended to the first user message.
        result = "<bos>";
        std::string system_content;
        size_t start_idx = 0;
        if (!messages.empty() && messages[0].role == "system") {
            system_content = messages[0].content;
            start_idx = 1;
        }

        for (size_t i = start_idx; i < messages.size(); i++) {
            const auto & msg = messages[i];
            std::string role = msg.role;
            if (role == "assistant") role = "model";

            result += "<|turn>";
            result += role;
            result += '\n';
            // Inject system content at the start of the first user message.
            if (i == start_idx && !system_content.empty() && msg.role == "user") {
                result += system_content;
                result += "\n\n";
            }
            result += msg.content;
            result += "<turn|>\n";
        }
        if (add_generation_prompt) {
            result += "<|turn>model\n";
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

std::string render_chat_template_jinja(
    const std::string & template_src,
    const std::vector<ChatMessage> & messages,
    const std::string & bos_token,
    const std::string & eos_token,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::string & tools_json)
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
        nlohmann::ordered_json mj;
        mj["role"]    = m.role;
        mj["content"] = m.content;
        if (!m.tool_call_id.empty()) {
            mj["tool_call_id"] = m.tool_call_id;
        }
        messages_j.push_back(std::move(mj));
    }

    nlohmann::ordered_json inputs;
    inputs["messages"]              = std::move(messages_j);
    inputs["bos_token"]             = bos_token;
    inputs["eos_token"]             = eos_token;
    inputs["add_generation_prompt"] = add_generation_prompt;
    inputs["enable_thinking"]       = enable_thinking;

    bool has_tools = !tools_json.empty() && tools_json != "[]" && tools_json != "null";
    if (has_tools) {
        try {
            inputs["tools"] = nlohmann::ordered_json::parse(tools_json);
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

}  // namespace dflash::common
