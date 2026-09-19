#include "server/tool_memory.h"
#include "server/api_types.h"
#include "server/tool_parser.h"
#include "server/chat_template.h"
#include "support/tool_schemas.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

namespace dflash::common {
std::vector<ChatMessage> normalize_chat_messages(
    const json & messages,
    ApiFormat format,
    ToolMemory & tool_memory);
}

// ═══════════════════════════════════════════════════════════════════════
// Jinja chat template
// ═══════════════════════════════════════════════════════════════════════

// Minimal Jinja template: just join roles + contents. Used to verify the
// runtime + global_from_json plumbing without depending on any external
// .jinja file at test time.
static const char MINI_JINJA_TEMPLATE[] =
    "{%- for m in messages -%}"
    "<|{{ m.role }}|>{{ m.content }}\n"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "<|assistant|>"
    "{%- endif -%}";

TEST_CASE(ServerUnitFixture, test_deepseek4_render_system_only_gen_prompt) {
    std::vector<ChatMessage> msgs = {
        {"system", "sys only", ""},
    };
    const std::string out = render_chat_template(
        msgs, ChatFormat::DEEPSEEK4,
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/false,
        /*tools_json=*/"");
    const std::string expected =
        "<｜begin▁of▁sentence｜>sys only<｜Assistant｜></think>";
    TEST_ASSERT(out == expected);
}

TEST_CASE(ServerUnitFixture, test_deepseek4_render_empty_chat_gen_prompt) {
    std::vector<ChatMessage> msgs;
    const std::string out = render_chat_template(
        msgs, ChatFormat::DEEPSEEK4,
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/false,
        /*tools_json=*/"");
    const std::string expected =
        "<｜begin▁of▁sentence｜><｜Assistant｜></think>";
    TEST_ASSERT(out == expected);
}

TEST_CASE(ServerUnitFixture, test_deepseek4_render_reasoning_effort_prefixes) {
    std::vector<ChatMessage> msgs = {
        {"system", "system message", ""},
        {"user", "hard problem", ""},
    };
    const std::string bos = "<｜begin▁of▁sentence｜>";
    const std::string high_prefix =
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively "
        "decompose the problem to resolve the root cause, rigorously "
        "stress-testing your logic against all potential paths, edge cases, "
        "and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting "
        "every intermediate step, considered alternative, and rejected "
        "hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
    const std::string max_prefix =
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
        "uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely "
        "nothing to chance: exhaustively decompose the problem into its most "
        "fundamental components, trace every causal chain to its root, and "
        "resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the "
        "solution from multiple angles and are certain that no assumption "
        "remains unchecked and no error remains undiscovered.\n\n";
    const auto ends_with = [](const std::string & text,
                              const std::string & suffix) {
        return text.size() >= suffix.size() &&
            text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    const std::string high = render_chat_template(
        msgs, ChatFormat::DEEPSEEK4, true, true, "", "high");
    TEST_ASSERT(high.rfind(bos + high_prefix + "system message", 0) == 0);
    TEST_ASSERT(high.find(max_prefix) == std::string::npos);
    TEST_ASSERT(ends_with(high, "<｜Assistant｜><think>"));

    const std::string max = render_chat_template(
        msgs, ChatFormat::DEEPSEEK4, true, true, "", "max");
    TEST_ASSERT(max.rfind(bos + max_prefix + "system message", 0) == 0);
    TEST_ASSERT(max.find(high_prefix) == std::string::npos);
    TEST_ASSERT(ends_with(max, "<｜Assistant｜><think>"));

    const std::string low = render_chat_template(
        msgs, ChatFormat::DEEPSEEK4, true, true, "", "low");
    TEST_ASSERT(low.find(high_prefix) == std::string::npos);
    TEST_ASSERT(low.find(max_prefix) == std::string::npos);
    TEST_ASSERT(low.rfind(bos + "system message", 0) == 0);

    const std::string disabled = render_chat_template(
        msgs, ChatFormat::DEEPSEEK4, true, false, "", "max");
    TEST_ASSERT(disabled.find(max_prefix) == std::string::npos);
    TEST_ASSERT(ends_with(disabled, "<｜Assistant｜></think>"));

    const std::string completed_turn = render_chat_template(
        msgs, ChatFormat::DEEPSEEK4, false, true, "", "high");
    TEST_ASSERT(completed_turn.rfind(bos + high_prefix, 0) == 0);
    TEST_ASSERT(!ends_with(completed_turn, "<｜Assistant｜><think>"));
}

TEST_CASE(ServerUnitFixture, test_jinja_render_basic) {
    std::vector<ChatMessage> msgs = {
        {"system", "you are helpful", ""},
        {"user",   "hi",              ""},
    };
    std::string out = render_chat_template_jinja(
        MINI_JINJA_TEMPLATE, msgs,
        /*bos=*/"<s>", /*eos=*/"</s>",
        /*add_gen=*/true, /*think=*/false,
        /*tools=*/"");
    TEST_ASSERT(out.find("<|system|>you are helpful") != std::string::npos);
    TEST_ASSERT(out.find("<|user|>hi")               != std::string::npos);
    TEST_ASSERT(out.find("<|assistant|>")            != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_jinja_render_no_gen_prompt) {
    std::vector<ChatMessage> msgs = {{"user", "ping", ""}};
    std::string out = render_chat_template_jinja(
        MINI_JINJA_TEMPLATE, msgs, "", "",
        /*add_gen=*/false, /*think=*/false, "");
    TEST_ASSERT(out.find("<|user|>ping") != std::string::npos);
    TEST_ASSERT(out.find("<|assistant|>") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_jinja_render_tools_injected) {
    // Template references `tools` to confirm it was passed in.
    static const char TPL[] =
        "{%- if tools -%}TOOLS_PRESENT:{{ tools[0].name }}{%- endif -%}"
        "{%- for m in messages -%}<|{{ m.role }}|>{{ m.content }}{%- endfor -%}";
    std::vector<ChatMessage> msgs = {{"user", "?", ""}};
    std::string tools = R"([{"name":"my_tool","description":"test"}])";
    std::string out = render_chat_template_jinja(
        TPL, msgs, "", "", false, false, tools);
    TEST_ASSERT(out.find("TOOLS_PRESENT:my_tool") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_jinja_render_empty_tools_skipped) {
    // tools_json == "[]" must NOT define `tools` in the template context.
    static const char TPL[] =
        "{%- if tools -%}TOOLS_PRESENT{%- else -%}NO_TOOLS{%- endif -%}";
    std::vector<ChatMessage> msgs = {{"user", "?", ""}};
    std::string out = render_chat_template_jinja(
        TPL, msgs, "", "", false, false, "[]");
    TEST_ASSERT(out.find("NO_TOOLS")        != std::string::npos);
    TEST_ASSERT(out.find("TOOLS_PRESENT")   == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_jinja_render_bos_eos_threaded) {
    // {{ bos_token }} and {{ eos_token }} must reach the template.
    static const char TPL[] = "{{ bos_token }}HI{{ eos_token }}";
    std::vector<ChatMessage> msgs;
    std::string out = render_chat_template_jinja(
        TPL, msgs, "<BOS>", "<EOS>", false, false, "");
    TEST_ASSERT(out == "<BOS>HI<EOS>");
}

TEST_CASE(ServerUnitFixture, test_jinja_render_empty_template_throws) {
    std::vector<ChatMessage> msgs = {{"user", "x", ""}};
    bool threw = false;
    try {
        (void)render_chat_template_jinja("", msgs, "", "", true, false, "");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    TEST_ASSERT(threw);
}

TEST_CASE(ServerUnitFixture, test_jinja_render_bad_tools_json_throws) {
    static const char TPL[] = "{%- for m in messages -%}{{ m.role }}{%- endfor -%}";
    std::vector<ChatMessage> msgs = {{"user", "x", ""}};
    bool threw = false;
    try {
        (void)render_chat_template_jinja(
            TPL, msgs, "", "", true, false, "{not valid json");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    TEST_ASSERT(threw);
}

TEST_CASE(ServerUnitFixture, test_normalize_responses_tool_followup_messages) {
    ToolMemory tool_memory;
    const std::string call_id = "call_exec_001";
    const std::string second_call_id = "call_read_002";
    const std::string raw_tool_call =
        "\n\n<function=exec_command>\n"
        "<parameter=cmd>\n"
        "git fetch origin && git status\n"
        "</parameter>\n"
        "</function>\n"
        "<function=read_file>\n"
        "<parameter=path>\n"
        "src/main.cpp\n"
        "</parameter>\n"
        "</function>\n";
    tool_memory.remember({call_id, second_call_id}, raw_tool_call);

    json messages = json::array({
        {
            {"role", "developer"},
            {"content", json::array({{
                {"type", "input_text"},
                {"text", "Developer rules"}
            }})}
        },
        {
            {"role", "user"},
            {"content", json::array({{
                {"type", "input_text"},
                {"text", "fetch latest code"}
            }})}
        },
        {
            {"type", "function_call"},
            {"call_id", call_id},
            {"name", "exec_command"},
            {"arguments", R"({"cmd":"git fetch origin && git status"})"}
        },
        {
            {"type", "function_call"},
            {"call_id", second_call_id},
            {"name", "read_file"},
            {"arguments", R"({"path":"src/main.cpp"})"}
        },
        {
            {"type", "function_call_output"},
            {"call_id", call_id},
            {"output", "Process exited with code 0"}
        },
        {
            {"type", "function_call_output"},
            {"call_id", second_call_id},
            {"output", "int main() {}"}
        }
    });

    auto chat_msgs = normalize_chat_messages(messages, ApiFormat::RESPONSES, tool_memory);
    TEST_ASSERT(chat_msgs.size() == 5);
    if (chat_msgs.size() == 5) {
        TEST_ASSERT(chat_msgs[0].role == "system");
        TEST_ASSERT(chat_msgs[0].content == "Developer rules");
        TEST_ASSERT(chat_msgs[1].role == "user");
        TEST_ASSERT(chat_msgs[1].content == "fetch latest code");
        TEST_ASSERT(chat_msgs[2].role == "assistant");
        TEST_ASSERT(chat_msgs[2].content == raw_tool_call);
        TEST_ASSERT(chat_msgs[3].role == "tool");
        TEST_ASSERT(chat_msgs[3].tool_call_id == call_id);
        TEST_ASSERT(chat_msgs[3].content == "Process exited with code 0");
        TEST_ASSERT(chat_msgs[4].role == "tool");
        TEST_ASSERT(chat_msgs[4].tool_call_id == second_call_id);
        TEST_ASSERT(chat_msgs[4].content == "int main() {}");
    }
}

TEST_CASE(ServerUnitFixture, test_render_deepseek4_chat_template_dsml_tools) {
    std::vector<ChatMessage> msgs = {
        {"system", "You are an assistant."},
        {"user", "Hello"}
    };
    std::string tools_json = R"([{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}])";
    std::string rendered = render_chat_template(msgs, ChatFormat::DEEPSEEK4, true, false, tools_json);
    TEST_ASSERT(rendered.find("<｜DSML｜tool_calls>") != std::string::npos);
    TEST_ASSERT(rendered.find("<｜DSML｜invoke name=\"$TOOL_NAME\">") != std::string::npos);
    TEST_ASSERT(rendered.find("\"name\":\"read\"") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_bailingmoe3_render_official_role_format) {
    TEST_ASSERT(chat_format_for_arch("bailingmoe3") == ChatFormat::BAILINGMOE3);
    const std::vector<ChatMessage> msgs = {{"user", "Hello", ""}};
    const std::string out = render_chat_template(
        msgs, ChatFormat::BAILINGMOE3,
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/false,
        /*tools_json=*/"");
    TEST_ASSERT(out ==
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        "<role>HUMAN</role>Hello<|role_end|>"
        "<role>ASSISTANT</role>\n<think></think>");
}

TEST_CASE(ServerUnitFixture, test_bailingmoe3_render_thinking_and_tools) {
    const std::vector<ChatMessage> msgs = {
        {"system", "Be concise.", ""},
        {"user", "Check Rome", ""},
    };
    const std::string tools =
        R"([{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}])";
    const std::string out = render_chat_template(
        msgs, ChatFormat::BAILINGMOE3,
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/true,
        tools);
    TEST_ASSERT(out.find("<role>SYSTEM</role>Be concise.\n# Tools") == 0);
    TEST_ASSERT(out.find("<tools>\n{\"function\":") != std::string::npos);
    TEST_ASSERT(out.find("detailed thinking on<|role_end|>") != std::string::npos);
    TEST_ASSERT(out.find("<role>HUMAN</role>Check Rome<|role_end|>") != std::string::npos);
    const std::string suffix = "<role>ASSISTANT</role>\n<think>";
    TEST_ASSERT(out.size() >= suffix.size());
    TEST_ASSERT(out.compare(out.size() - suffix.size(), suffix.size(), suffix) == 0);
}

TEST_CASE(ServerUnitFixture, test_bailingmoe3_request_overrides_system_thinking) {
    const std::vector<ChatMessage> thinking_on = {
        {"system", "Keep this note. detailed thinking on", ""},
        {"user", "Hello", ""},
    };
    const std::string disabled = render_chat_template(
        thinking_on, ChatFormat::BAILINGMOE3,
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/false,
        /*tools_json=*/"");
    const size_t prior_on = disabled.find("detailed thinking on");
    const size_t requested_off = disabled.rfind("detailed thinking off");
    TEST_ASSERT(prior_on != std::string::npos);
    TEST_ASSERT(requested_off != std::string::npos && requested_off > prior_on);

    const std::vector<ChatMessage> thinking_off = {
        {"system", "Keep this note. detailed thinking off", ""},
        {"user", "Hello", ""},
    };
    const std::string tools =
        R"([{"type":"function","function":{"name":"weather"}}])";
    const std::string enabled = render_chat_template(
        thinking_off, ChatFormat::BAILINGMOE3,
        /*add_generation_prompt=*/true,
        /*enable_thinking=*/true,
        tools);
    const size_t prior_off = enabled.find("detailed thinking off");
    const size_t requested_on = enabled.rfind("detailed thinking on");
    TEST_ASSERT(prior_off != std::string::npos);
    TEST_ASSERT(requested_on != std::string::npos && requested_on > prior_off);
}
