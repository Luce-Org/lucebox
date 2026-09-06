#include "server/sse_emitter.h"
#include "support/streaming.h"
#include "support/tool_schemas.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cstdio>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

// ═══════════════════════════════════════════════════════════════════════
// SSE Emitter tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_emitter_reasoning_split_openai) {
    // Feed reasoning + content through emitter, verify split.
    // Model emits the opening <think> as its first token (Qwen3.6 path
    // — the streaming on_token lambda maps the special <think> id to
    // emit_token("<think>")).
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();

    // Open reasoning, feed reasoning tokens
    em.emit_token("<think>");
    em.emit_token("Let me think about this...");
    // Close thinking and start content
    em.emit_token("</think>");
    em.emit_token("The answer is 42.");

    em.emit_finish(10);

    TEST_ASSERT(!em.reasoning_text().empty());
    TEST_ASSERT(em.reasoning_text().find("<think>") == std::string::npos);
    TEST_ASSERT(em.reasoning_text().find("</think>") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("42") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("</think>") == std::string::npos);
}

// SseEmitter::emit_token_count() / accumulated text accessors drive
// http_server's finish_details accounting on the natural-close path
// (model self-closes </think> mid-stream). Each test feeds tokens
// one-per-call so the emit_token index is straightforward to reason
// about.
TEST_CASE(ServerUnitFixture, test_emitter_first_content_index_natural_close) {
    // Emit reasoning tokens (with explicit <think> open + </think>
    // close), then content tokens. The emit_token_count() reflects
    // all delivered tokens; the reasoning/content split is also
    // recoverable from accumulated_text / reasoning_text.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    TEST_ASSERT(em.emit_token_count() == 0);

    em.emit_token("<think>");
    em.emit_token("reasoning1");
    em.emit_token("reasoning2");
    em.emit_token("end</think>");
    em.emit_token("content1");
    em.emit_token("content2");
    em.emit_finish(6);

    TEST_ASSERT(em.emit_token_count() == 6);
    // Reasoning + content text both populated.
    TEST_ASSERT(!em.reasoning_text().empty());
    TEST_ASSERT(em.accumulated_text().find("content1") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("content2") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_first_content_index_never_closed) {
    // Model opens <think> then emits reasoning only — never closes
    // </think>. All produced text lands in reasoning_text; visible
    // accumulated_text stays empty.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();

    em.emit_token("<think>");
    em.emit_token("reasoning never closes");
    em.emit_token("still thinking");
    em.emit_finish(3);

    TEST_ASSERT(em.emit_token_count() == 3);
    TEST_ASSERT(em.reasoning_text().find("reasoning") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().empty());
}

TEST_CASE(ServerUnitFixture, test_emitter_first_content_index_content_only) {
    // Non-thinking request: emitter starts in CONTENT mode, so the
    // very first emit_token lands at index 0.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("immediate content");
    em.emit_finish(1);

    TEST_ASSERT(em.first_content_token_index() == 0);
    TEST_ASSERT(em.emit_token_count() == 1);
}

TEST_CASE(ServerUnitFixture, test_emitter_first_content_index_qwen36_streaming_thinking) {
    // Regression: when the chat template emits a leading `<think>` token
    // (Qwen3.6 thinking-enabled path, or gemma4 `<|channel>` → `<think>`
    // map) the emitter starts in CONTENT mode by default and naively
    // captured first_content_token_index_=0 on the first emit_token
    // call, before the state machine transitioned to REASONING. Result:
    // finish_details.thinking_tokens misreported as 0 for any streamed-
    // thinking response. Fix: detect the `<think>` opener up-front and
    // defer the fci capture until a true CONTENT-mode token arrives.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();

    // Mirror http_server's on_token mapping: the special <think> id is
    // forwarded as a standalone "<think>" piece, followed by reasoning
    // text, the closing "</think>" piece, then the answer.
    em.emit_token("<think>");
    em.emit_token("reasoning step 1");
    em.emit_token("reasoning step 2");
    em.emit_token("</think>\n");
    em.emit_token("answer text");
    em.emit_finish(5);

    // fci must point at the first true content token, NOT 0.
    TEST_ASSERT(em.first_content_token_index() > 0);
    // Reasoning text populated, leading <think> stripped.
    TEST_ASSERT(!em.reasoning_text().empty());
    TEST_ASSERT(em.reasoning_text().find("<think>") == std::string::npos);
    // Content text populated.
    TEST_ASSERT(em.accumulated_text().find("answer") != std::string::npos);
    // emit_token_count - fci should be the content-suffix size
    // (>0 means at least one content-mode token was attributed).
    TEST_ASSERT(em.emit_token_count() - em.first_content_token_index() > 0);
}

TEST_CASE(ServerUnitFixture, test_emitter_reasoning_strips_leading_think_tag) {
    // Model emits leading whitespace + <think> as one token, then
    // continues thinking. The leading-<think>-with-whitespace-prefix
    // strip ensures the reasoning text doesn't contain the open tag.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();

    // Model emits \n<think>\n before actual reasoning
    em.emit_token("\n<think>\nActual reasoning here");
    em.emit_token("</think>");
    em.emit_token("Content");

    em.emit_finish(10);

    // Leading <think> should be stripped from reasoning
    TEST_ASSERT(em.reasoning_text().find("<think>") == std::string::npos);
    TEST_ASSERT(em.reasoning_text().find("Actual reasoning") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_content_only_no_thinking) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("Hello, world!");
    em.emit_finish(5);

    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
    TEST_ASSERT(em.reasoning_text().empty());
}

TEST_CASE(ServerUnitFixture, test_emitter_tool_buffer_detection) {
    // When the emitter sees <tool_call>, it should buffer and parse tools.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, weather_tools());
    em.emit_start();
    em.emit_token("<tool_call>\n"
                  "<function=get_weather>\n"
                  "<parameter=location>NYC</parameter>\n"
                  "</function>\n"
                  "</tool_call>");
    em.emit_finish(20);

    TEST_ASSERT(!em.tool_calls().empty());
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "get_weather");
    }
    // Tool call text should not leak into accumulated content
    TEST_ASSERT(em.accumulated_text().find("<tool_call>") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_function_call_tool_buffer_detection) {
    // When the emitter sees <function_call>, it should buffer and parse tools.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, bash_tools());
    em.emit_start();
    em.emit_token("<function_call>\n"
                  "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}\n"
                  "</function_call>");
    em.emit_finish(20);

    TEST_ASSERT(!em.tool_calls().empty());
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "bash");
    }
    // Tool call text should not leak into accumulated content
    TEST_ASSERT(em.accumulated_text().find("<function_call>") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("bash") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_bare_function_json_tool_buffer_detection) {
    // When the emitter sees <function>, it should buffer and parse tools.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, bash_tools());
    em.emit_start();
    em.emit_token("<function>\n"
                  "{\n"
                  "  \"name\": \"bash\",\n"
                  "  \"parameters\": {\n"
                  "    \"command\": \"ls -la \\\"/home/dpavlin/aimax project\\\"\"\n"
                  "  }\n"
                  "}\n"
                  "</function>");
    em.emit_finish(20);

    TEST_ASSERT(!em.tool_calls().empty());
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "bash");
    }
    // Tool call text should not leak into accumulated content
    TEST_ASSERT(em.accumulated_text().find("<function>") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("bash") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_named_json_with_multiple_tools) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_and_bash_tools());
    em.emit_start();
    em.emit_token("{\"function\":\"bash\",");
    em.emit_token("\"parameters\":{\"command\":\"pwd\"}}");
    const auto finish = em.emit_finish(20);

    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "bash");
        const auto args = json::parse(em.tool_calls()[0].arguments);
        TEST_ASSERT(args["command"] == "pwd");
    }
    TEST_ASSERT(em.accumulated_text().empty());
    const std::string wire = concat(finish);
    TEST_ASSERT(wire.find("bash") != std::string::npos);
    TEST_ASSERT(wire.find("tool_calls") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_multi_tool_json_content_is_preserved) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_and_bash_tools());
    em.emit_start();
    em.emit_token("{\"status\":\"ok\"}");
    const auto finish = em.emit_finish(20);

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text() == "{\"status\":\"ok\"}");
    TEST_ASSERT(concat(finish).find("status") != std::string::npos);
}



TEST_CASE(ServerUnitFixture, test_emitter_anthropic_tool_use_blocks) {
    // The Anthropic streaming tool-use branch used to be a no-op; the model
    // would emit a <tool_call>...</tool_call> block, the parser would detect
    // it, but no tool_use SSE event was sent. Verify the lifecycle now:
    //   message_start, content_block_start (text), content_block_stop (text),
    //   content_block_start (tool_use), content_block_delta (input_json_delta),
    //   content_block_stop, message_delta(stop_reason="tool_use"), message_stop
    json tools = json::array();
    tools.push_back({
        {"name", "get_weather"},
        {"description", "weather"},
        {"input_schema", {{"type", "object"},
                          {"properties", {{"city", {{"type", "string"}}}}}}}
    });
    SseEmitter em(ApiFormat::ANTHROPIC, "req_id", "test-model", 10,
                  tools, nullptr);
    (void)em.emit_start();
    // Feed Qwen3 XML tool call in chunks so the holdback buffer flushes;
    // parser will detect <tool_call><function=NAME>...</tool_call>.
    em.emit_token("<tool_call>\n<function=get_weather>\n");
    em.emit_token("<parameter=city>\nTokyo\n</parameter>\n");
    em.emit_token("</function>\n</tool_call>");
    auto finish = em.emit_finish(20);
    std::string s = concat(finish);

    TEST_ASSERT(s.find("\"type\":\"tool_use\"")          != std::string::npos);
    TEST_ASSERT(s.find("\"name\":\"get_weather\"")     != std::string::npos);
    TEST_ASSERT(s.find("\"type\":\"input_json_delta\"") != std::string::npos);
    TEST_ASSERT(s.find("Tokyo")                          != std::string::npos);
    TEST_ASSERT(s.find("\"stop_reason\":\"tool_use\"")  != std::string::npos);
    TEST_ASSERT(s.find("message_stop")                   != std::string::npos);
    // Regression guard: at minimum text-block-stop + tool_use-block-stop.
    size_t n_stop = 0; size_t pos = 0;
    while ((pos = s.find("content_block_stop", pos)) != std::string::npos) {
        n_stop++; pos++;
    }
    TEST_ASSERT(n_stop >= 2);
}

TEST_CASE(ServerUnitFixture, test_emitter_single_tool_bare_json_args) {
    auto em = make_emitter(ApiFormat::ANTHROPIC, shell_tools());
    em.emit_start();
    em.emit_token("{\n");
    em.emit_token("  \"command\": \"git branch --show-current\"\n");
    em.emit_token("}");
    em.emit_finish(16);

    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "shell");
        auto args = json::parse(em.tool_calls()[0].arguments);
        TEST_ASSERT(args["command"] == "git branch --show-current");
    }
    TEST_ASSERT(em.accumulated_text().empty());
}

TEST_CASE(ServerUnitFixture, test_emitter_bare_json_args_do_not_trigger_after_content) {
    auto em = make_emitter(ApiFormat::ANTHROPIC, shell_tools());
    em.emit_start();
    em.emit_token("This answer already emitted visible prose before JSON appears.");
    em.emit_token("                    ");
    em.emit_token("{\"command\":\"git status\"}");
    em.emit_finish(16);

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text().find("visible prose") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("\"command\":\"git status\"") !=
                std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_bare_function_tool_buffer_detection) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, weather_tools());
    em.emit_start();
    em.emit_token("<function=terminal>\n"
                  "<parameter=command>\n"
                  "ls -la /tmp/lop/\n"
                  "</parameter>\n"
                  "</function>");
    em.emit_finish(20);

    TEST_ASSERT(!em.tool_calls().empty());
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "terminal");
        auto args = json::parse(em.tool_calls()[0].arguments);
        TEST_ASSERT(args["command"] == "ls -la /tmp/lop/");
    }
    TEST_ASSERT(em.accumulated_text().find("<function=terminal>") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_attribute_style_tool_buffer_detection) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, bash_tools());
    em.emit_start();
    em.emit_token("The branch already exists. Let me check the current state:\n\n"
                  "<param");
    em.emit_token("eter name=\"bash\"><parameter name=\"command\">"
                  "git status -sb && git branch\n");
    em.emit_token(" --show-current</parameter>\n</function>");
    auto finish = em.emit_finish(20);
    const std::string wire = concat(finish);

    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "bash");
        auto args = json::parse(em.tool_calls()[0].arguments);
        TEST_ASSERT(args["command"] ==
                    "git status -sb && git branch\n --show-current");
    }
    TEST_ASSERT(em.accumulated_text() ==
                "The branch already exists. Let me check the current state:\n\n");
    TEST_ASSERT(em.accumulated_text().find("<parameter") == std::string::npos);
    TEST_ASSERT(wire.find("\"finish_reason\":\"tool_calls\"") !=
                std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_funcname_tool_buffer_detection) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools());
    em.emit_start();
    em.emit_token("\n\n<func");
    em.emit_token("name>read\n<parameter=limit>\n50\n</parameter>\n"
                  "<parameter=offset>\n1\n</parameter>\n");
    em.emit_token("<parameter=path>\n"
                  "/tmp/tool-input.md\n"
                  "</parameter>\n</function>\n");
    auto finish = em.emit_finish(88);
    const std::string wire = concat(finish);

    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "read");
        auto args = json::parse(em.tool_calls()[0].arguments);
        TEST_ASSERT(args["limit"] == 50);
        TEST_ASSERT(args["offset"] == 1);
        TEST_ASSERT(args["path"] == "/tmp/tool-input.md");
    }
    TEST_ASSERT(em.accumulated_text().find("<funcname>") == std::string::npos);
    TEST_ASSERT(em.accumulated_text() == "\n\n");
    TEST_ASSERT(wire.find("\"finish_reason\":\"tool_calls\"") !=
                std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_space_function_tool_buffer_detection) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools());
    em.emit_start();
    em.emit_token("Let me read it.\n\n<funct");
    em.emit_token("ion read>\n<parameter=path>\n");
    em.emit_token("/tmp/tool-input.md\n"
                  "</parameter>\n</function>");
    const std::string wire = concat(em.emit_finish(42));

    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "read");
        TEST_ASSERT(json::parse(em.tool_calls()[0].arguments)["path"] ==
                    "/tmp/tool-input.md");
    }
    TEST_ASSERT(em.accumulated_text() == "Let me read it.\n\n");
    TEST_ASSERT(em.accumulated_text().find("<function read>") ==
                std::string::npos);
    TEST_ASSERT(wire.find("\"finish_reason\":\"tool_calls\"") !=
                std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_repeated_bare_edit_calls) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, edit_tools());
    em.emit_start();
    em.emit_token("Applying both updates.\n\n<ed");
    em.emit_token("it>\n<parameter=edits>\n"
                  "[{\"path\":\"/workspace/first.conf\","
                  "\"oldText\":\"auto\",\"newText\":\"enabled\"}]\n"
                  "</parameter>\n</function>\n\n<edit>\n");
    em.emit_token("<parameter=edits>\n"
                  "[{\"path\":\"/workspace/second.conf\","
                  "\"oldText\":\"auto\",\"newText\":\"enabled\"}]\n"
                  "</parameter>\n</function>\n\n</edit>");
    const std::string wire = concat(em.emit_finish(96));

    TEST_ASSERT(em.tool_calls().size() == 2);
    if (em.tool_calls().size() == 2) {
        const auto first = json::parse(em.tool_calls()[0].arguments);
        const auto second = json::parse(em.tool_calls()[1].arguments);
        TEST_ASSERT(first["edits"][0]["path"] == "/workspace/first.conf");
        TEST_ASSERT(second["edits"][0]["path"] == "/workspace/second.conf");
    }
    TEST_ASSERT(em.accumulated_text() == "Applying both updates.\n\n");
    TEST_ASSERT(em.accumulated_text().find("<edit>") == std::string::npos);
    TEST_ASSERT(wire.find("\"finish_reason\":\"tool_calls\"") !=
                std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_does_not_leak_malformed_tool_xml) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, weather_tools());
    em.emit_start();
    em.emit_token("Let me list files.\n\n");
    em.emit_token("<tool_call>\n"
                  "<function=terminal>\n"
                  "<parameter=command>\n"
                  "ls -la /tmp/lop/\n"
                  "</parameter>");
    em.emit_finish(20);

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text().find("Let me list files.") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("<tool_call>") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("<function=terminal>") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_parses_tool_call_missing_outer_close) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, weather_tools());
    em.emit_start();
    em.emit_token("<tool_call>\n"
                  "<function=terminal>\n"
                  "<parameter=command>\n"
                  "ls -la /tmp/lop/\n"
                  "</parameter>\n"
                  "</function>");
    em.emit_finish(20);

    TEST_ASSERT(!em.tool_calls().empty());
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "terminal");
        auto args = json::parse(em.tool_calls()[0].arguments);
        TEST_ASSERT(args["command"] == "ls -la /tmp/lop/");
    }
    TEST_ASSERT(em.accumulated_text().find("<tool_call>") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("<function=terminal>") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_no_tools_keeps_tool_like_text) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("<function=terminal>\n"
                  "<parameter=command>ls</parameter>\n"
                  "</function>");
    em.emit_finish(20);

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text().find("<function=terminal>") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_undeclared_file_tag_stays_content) {
    const std::string text =
        "Let me check if the file exists first.\n\n"
        "<file>\n"
        "<parameter=path>\n"
        "~/.pi/agent/models.json\n"
        "</parameter>\n"
        "</function>";

    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools());
    em.emit_start();
    em.emit_token(text);
    em.emit_finish(20);

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text() == text);
}

TEST_CASE(ServerUnitFixture, test_emitter_anthropic_structure) {
    // Verify Anthropic format emits proper event sequence.
    auto em = make_emitter(ApiFormat::ANTHROPIC);
    auto start = em.emit_start();
    std::string start_str = concat(start);

    // Should have message_start event
    TEST_ASSERT(start_str.find("message_start") != std::string::npos);
    TEST_ASSERT(start_str.find("content_block_start") != std::string::npos);

    auto chunks = em.emit_token("Hello");
    auto chunks2 = em.emit_token(" world! This is enough text to flush the holdback buffer.");
    std::string chunk_str = concat(chunks) + concat(chunks2);
    // At least one emission should contain content_block_delta
    TEST_ASSERT(chunk_str.find("content_block_delta") != std::string::npos);

    // Feed enough to flush holdback
    em.emit_token(" world! This is a longer sentence to exceed holdback.");
    auto finish = em.emit_finish(10);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("content_block_stop") != std::string::npos);
    TEST_ASSERT(finish_str.find("message_stop") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_responses_structure) {
    auto em = make_emitter(ApiFormat::RESPONSES);
    auto start = em.emit_start();
    std::string start_str = concat(start);

    TEST_ASSERT(start_str.find("response.created") != std::string::npos);
    TEST_ASSERT(start_str.find("response.output_item.added") != std::string::npos);

    em.emit_token("Hi there! How are you doing today?");
    auto finish = em.emit_finish(10);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("response.completed") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_responses_bare_function_tool_call) {
    json tools = json::array({{
        {"type", "function"},
        {"name", "exec_command"},
        {"description", "Run a command"},
        {"parameters", {
            {"type", "object"},
            {"properties", {{"cmd", {{"type", "string"}}}}},
            {"required", json::array({"cmd"})}
        }}
    }});
    SseEmitter em(ApiFormat::RESPONSES, "resp_test_001", "test-model", 10,
                  tools, nullptr);
    em.emit_start();
    em.emit_token("\n\n<function=exec_command>\n<parameter=cmd>\ngit pull\n");
    em.emit_token("</parameter>\n</function>\n");
    auto finish = em.emit_finish(8);
    std::string finish_str = concat(finish);

    TEST_ASSERT(!em.tool_calls().empty());
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "exec_command");
        auto args = json::parse(em.tool_calls()[0].arguments);
        TEST_ASSERT(args["cmd"] == "git pull");
    }
    TEST_ASSERT(finish_str.find("\"type\":\"function_call\"") != std::string::npos);
    TEST_ASSERT(finish_str.find("response.function_call_arguments.done") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_openai_has_done) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("Hello");
    auto finish = em.emit_finish(3);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("[DONE]") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_nonstreaming_accumulates) {
    // Non-streaming: tokens fed through emitter, accumulated_text() has all content.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_token("Hello ");
    em.emit_token("world");
    em.emit_finish(5);

    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("world") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_anthropic_thinking_blocks) {
    auto em = make_emitter(ApiFormat::ANTHROPIC);
    auto start = em.emit_start();
    std::string start_str = concat(start);

    // Model opens <think>, emits reasoning, closes, emits content.
    auto t1 = em.emit_token("<think>");
    auto t2 = em.emit_token("Reasoning about the problem at length here...");
    auto t3 = em.emit_token("</think>");
    auto t4 = em.emit_token("The answer is clear now.");
    auto finish = em.emit_finish(20);
    std::string all = start_str + concat(t1) + concat(t2) + concat(t3) +
                      concat(t4) + concat(finish);

    // Should have both thinking and text blocks somewhere in the stream
    TEST_ASSERT(all.find("thinking") != std::string::npos);
    TEST_ASSERT(!em.reasoning_text().empty());
    TEST_ASSERT(!em.accumulated_text().empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Stop sequences tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_stop_sequence_basic) {
    // Stop sequence should truncate content at the match point.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"STOP"});
    em.emit_token("Hello ");
    em.emit_token("world ");
    em.emit_token("STOP");
    em.emit_token(" more text");  // should be ignored

    TEST_ASSERT(em.stop_hit());
    em.emit_finish(5);
    // Content should NOT contain "STOP" or "more text"
    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("STOP") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("more") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_mid_token) {
    // Stop sequence may span multiple tokens due to holdback buffering.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"END"});
    em.emit_token("Go ");
    em.emit_token("to the E");
    em.emit_token("ND now");

    TEST_ASSERT(em.stop_hit());
    em.emit_finish(5);
    TEST_ASSERT(em.accumulated_text().find("Go") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("END") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("now") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_multiple) {
    // Multiple stop sequences — earliest match wins.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"AAA", "BB"});
    em.emit_token("xBBy");

    TEST_ASSERT(em.stop_hit());
    em.emit_finish(2);
    TEST_ASSERT(em.accumulated_text() == "x");
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_no_match) {
    // No stop sequence hit — normal operation.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"NOMATCH"});
    em.emit_token("Hello world this is a long text");
    em.emit_finish(10);

    TEST_ASSERT(!em.stop_hit());
    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_empty_list) {
    // Empty stop list — no effect.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{});
    em.emit_token("Hello STOP world");
    em.emit_finish(5);

    TEST_ASSERT(!em.stop_hit());
    TEST_ASSERT(em.accumulated_text().find("STOP") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_finish_reason) {
    // finish_reason should be "stop" when stop sequence hit.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"END"});
    em.emit_token("content END more");

    TEST_ASSERT(em.stop_hit());
    em.emit_finish(3);
    TEST_ASSERT(em.finish_reason() == "stop");
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_streaming_output) {
    // Streaming: verify the [DONE] is still emitted after stop.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"HALT"});
    auto start = em.emit_start();
    em.emit_token("some text HALT rest");

    TEST_ASSERT(em.stop_hit());
    auto finish = em.emit_finish(5);
    std::string all = concat(finish);
    TEST_ASSERT(all.find("[DONE]") != std::string::npos);
    TEST_ASSERT(all.find("\"finish_reason\":\"stop\"") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_anthropic_format) {
    // Anthropic format should emit end_turn stop_reason.
    auto em = make_emitter_with_stops(ApiFormat::ANTHROPIC, {"DONE"});
    em.emit_start();
    em.emit_token("This is content DONE rest");

    TEST_ASSERT(em.stop_hit());
    auto finish = em.emit_finish(5);
    std::string all = concat(finish);
    TEST_ASSERT(all.find("end_turn") != std::string::npos);
    TEST_ASSERT(all.find("message_stop") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_in_reasoning_mode) {
    // Stop sequence in reasoning mode should still stop. Model opens
    // <think> first to enter REASONING.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT, {"CUTOFF"});
    em.emit_token("<think>");
    em.emit_token("Thinking deeply about this CUTOFF answer");

    TEST_ASSERT(em.stop_hit());
    em.emit_finish(5);
    TEST_ASSERT(em.reasoning_text().find("Thinking") != std::string::npos);
    TEST_ASSERT(em.reasoning_text().find("CUTOFF") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_stop_sequence_holdback_extends) {
    // With a long stop sequence, holdback buffer should extend to prevent
    // emitting text that's part of a stop sequence.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,
                                       {"LONGSTOPSEQUENCE"});
    // Feed text token by token — the holdback should prevent premature emission
    em.emit_token("prefix ");
    em.emit_token("LONG");
    em.emit_token("STOP");
    em.emit_token("SEQUENCE");
    em.emit_token(" suffix");

    TEST_ASSERT(em.stop_hit());
    em.emit_finish(10);
    TEST_ASSERT(em.accumulated_text().find("prefix") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("LONGSTOPSEQUENCE") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("suffix") == std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════
// usage.timings — per-request prefill / decode wall-clock breakdown
// surfaced under usage.timings (spec §6.3). Tests cover all three
// response shapes plus the zero-decode_s div-by-zero guard.
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_usage_timings_openai_chat_streaming) {
    // OpenAI Chat streaming: the terminal usage chunk (just before
    // data: [DONE]) carries `timings.{prefill_ms, decode_ms,
    // decode_tokens_per_sec}` when timings are passed to emit_finish.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("Hello world");

    GenTimings t{0.2345, 2.4567};  // 234.5 ms / 2456.7 ms
    auto finish = em.emit_finish(/*completion_tokens*/ 100, &t);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("\"timings\"") != std::string::npos);
    TEST_ASSERT(finish_str.find("\"prefill_ms\":234.5") != std::string::npos);
    TEST_ASSERT(finish_str.find("\"decode_ms\":2456.7") != std::string::npos);
    // 100 / 2.4567 = 40.7048... → rounds to 40.7
    TEST_ASSERT(finish_str.find("\"decode_tokens_per_sec\":40.7") != std::string::npos);
    TEST_ASSERT(finish_str.find("[DONE]") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_usage_timings_anthropic_streaming) {
    // Anthropic streaming: message_delta.usage gains a `timings`
    // sibling alongside `output_tokens`.
    auto em = make_emitter(ApiFormat::ANTHROPIC);
    em.emit_start();
    em.emit_token("ok");
    GenTimings t{0.05, 0.5};  // 50.0 ms / 500.0 ms
    auto finish = em.emit_finish(/*completion_tokens*/ 10, &t);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("\"timings\"") != std::string::npos);
    TEST_ASSERT(finish_str.find("\"prefill_ms\":50.0") != std::string::npos);
    TEST_ASSERT(finish_str.find("\"decode_ms\":500.0") != std::string::npos);
    // 10 / 0.5 = 20.0
    TEST_ASSERT(finish_str.find("\"decode_tokens_per_sec\":20.0") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_usage_timings_responses_streaming) {
    // Responses streaming: response.completed.usage gains `timings`.
    auto em = make_emitter(ApiFormat::RESPONSES);
    em.emit_start();
    em.emit_token("done");
    GenTimings t{0.1, 1.0};
    auto finish = em.emit_finish(/*completion_tokens*/ 25, &t);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("\"timings\"") != std::string::npos);
    TEST_ASSERT(finish_str.find("\"prefill_ms\":100.0") != std::string::npos);
    TEST_ASSERT(finish_str.find("\"decode_ms\":1000.0") != std::string::npos);
    // 25 / 1.0 = 25.0
    TEST_ASSERT(finish_str.find("\"decode_tokens_per_sec\":25.0") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_usage_timings_zero_decode_no_div_by_zero) {
    // decode_s == 0 (prefill-only / no tokens generated path): emit
    // decode_tokens_per_sec = 0.0 without div-by-zero.
    GenTimings t{0.123, 0.0};
    json j = build_timings_json(t, /*completion_tokens*/ 42);
    TEST_ASSERT(j["prefill_ms"].get<double>() == 123.0);
    TEST_ASSERT(j["decode_ms"].get<double>() == 0.0);
    TEST_ASSERT(j["decode_tokens_per_sec"].get<double>() == 0.0);

    // Also exercise via OpenAI streaming path — finite JSON output, no NaN/Inf.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    auto finish = em.emit_finish(/*completion_tokens*/ 0, &t);
    std::string finish_str = concat(finish);
    TEST_ASSERT(finish_str.find("\"decode_tokens_per_sec\":0.0") != std::string::npos);
    // No NaN / Inf serialization leak.
    TEST_ASSERT(finish_str.find("inf") == std::string::npos);
    TEST_ASSERT(finish_str.find("nan") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_usage_timings_reports_prefix_cache_work) {
    GenTimings t{0.012, 0.25, true, 8192, 64, 8256, true};
    json j = build_timings_json(t, /*completion_tokens=*/10);
    TEST_ASSERT(j["cache_hit"].get<bool>());
    TEST_ASSERT(j["cached_prefix_tokens"].get<int>() == 8192);
    TEST_ASSERT(j["prefilled_tokens"].get<int>() == 64);
    TEST_ASSERT(j["effective_prompt_tokens"].get<int>() == 8256);
    TEST_ASSERT(j["agent_turn_cache_hit"].get<bool>());
}

TEST_CASE(ServerUnitFixture, test_usage_timings_omitted_when_null) {
    // Backward compat: emit_finish(n) (no timings) emits the legacy
    // usage block — no `timings` key. Guards the SDK-facing default
    // for callers that don't yet wire timings through.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("x");
    auto finish = em.emit_finish(3);  // no timings arg
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("\"timings\"") == std::string::npos);
    TEST_ASSERT(finish_str.find("[DONE]") != std::string::npos);
}

namespace {
struct StderrCapture {
    int old_stderr = -1;
    std::FILE * file = nullptr;

    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        if (file == nullptr) return;

        old_stderr = dup(STDERR_FILENO);
        if (old_stderr == -1 || dup2(fileno(file), STDERR_FILENO) == -1) {
            if (old_stderr != -1) close(old_stderr);
            old_stderr = -1;
            std::fclose(file);
            file = nullptr;
        }
    }

    std::string str() {
        restore();
        if (file == nullptr) return "";

        std::rewind(file);
        std::string out;
        char buf[1024];
        size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof(buf), file)) > 0) {
            out.append(buf, n);
        }
        std::fclose(file);
        file = nullptr;
        return out;
    }

    void restore() {
        if (old_stderr != -1) {
            std::fflush(stderr);
            dup2(old_stderr, STDERR_FILENO);
            close(old_stderr);
            old_stderr = -1;
        }
    }

    ~StderrCapture() {
        restore();
        if (file != nullptr) std::fclose(file);
    }
};
}  // namespace

TEST_CASE(ServerUnitFixture, test_emitter_suppresses_malformed_multiline_tool_buffer) {
    StderrCapture capture;

    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), false);
    em.emit_start();
    em.emit_token("<function_call>\n  <invoke name=\"read\">\n    malformed prose body with\nnew lines and \t tabs\n");
    em.emit_finish(10);

    std::string captured = capture.str();

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text().empty());
    TEST_ASSERT(captured.find("suppressing buffered tool text") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_function_calls_inside_reasoning) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), false);
    auto c1 = em.emit_token("<think>Analyzing build files.\n");
    auto c2 = em.emit_token("<function_calls>\n  <invoke name=\"read\">\n    <param name=\"path\">CMakeLists.txt</param>\n  </invoke>\n</function_calls>\n</think>");
    auto fin = em.emit_finish(2);

    std::string all = concat(c1) + concat(c2) + concat(fin);
    TEST_ASSERT(em.tool_calls().size() == 1);
    TEST_ASSERT(em.reasoning_text().find("Analyzing build files.") != std::string::npos);
    TEST_ASSERT(all.find("\"finish_reason\":\"tool_calls\"") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_function_calls_unclosed_think_flushes_reasoning) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), false);
    auto c1 = em.emit_token("<think>Analyzing build files without closing tag.\n");
    auto c2 = em.emit_token("<function_calls>\n  <invoke name=\"read\">\n    <param name=\"path\">CMakeLists.txt</param>\n  </invoke>\n</function_calls>");
    auto fin = em.emit_finish(2);

    std::string all = concat(c1) + concat(c2) + concat(fin);
    TEST_ASSERT(em.tool_calls().size() == 1);
    TEST_ASSERT(em.reasoning_text().find("Analyzing build files without closing tag.") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("Analyzing build files") == std::string::npos);
    TEST_ASSERT(all.find("\"finish_reason\":\"tool_calls\"") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_function_calls_content_tokens_accounting) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), false);
    // Token 0: reasoning
    em.emit_token("<think>Analyzing build configuration.\n");
    // Token 1: function_calls
    em.emit_token("<function_calls>\n  <invoke name=\"read\">\n    <param name=\"path\">CMakeLists.txt</param>\n  </invoke>\n</function_calls>\n");
    // Token 2: close think
    em.emit_token("</think>\n");
    // Token 3: content
    em.emit_token("Here is the build summary.");
    em.emit_finish(4);

    TEST_ASSERT(em.tool_calls().size() == 1);
    TEST_ASSERT(em.first_content_token_index() == 3);
    TEST_ASSERT(em.emit_token_count() == 4);
    TEST_ASSERT(em.emit_token_count() - em.first_content_token_index() == 1);
}

TEST_CASE(ServerUnitFixture, test_emitter_function_calls_param_with_literal_think_close) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), false);
    // Token 0: reasoning
    em.emit_token("<think>Searching for tag.\n");
    // Token 1: parameter with literal </think> inside
    em.emit_token("<function_calls>\n  <invoke name=\"read\">\n    <param name=\"path\">test_</think>.cpp</param>\n  </invoke>\n</function_calls>\n");
    // Token 2: real close think + trailing content in same token
    em.emit_token("</think> Found file.");
    em.emit_finish(3);

    TEST_ASSERT(em.tool_calls().size() == 1);
    TEST_ASSERT(em.first_content_token_index() == 2);
    TEST_ASSERT(em.emit_token_count() == 3);
    TEST_ASSERT(em.emit_token_count() - em.first_content_token_index() == 1);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_length_finish_reason_at_cap) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, json::array(), false);
    em.emit_start();
    em.emit_token("hello world");
    auto chunks = em.emit_finish(10, nullptr, 10);
    TEST_ASSERT(em.finish_reason() == "length");
    bool found_length = false;
    for (const auto & chunk : chunks) {
        if (chunk.find("\"finish_reason\":\"length\"") != std::string::npos) {
            found_length = true;
            break;
        }
    }
    TEST_ASSERT(found_length);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_length_finish_reason_at_zero_cap) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, json::array(), false);
    em.emit_start();
    auto chunks = em.emit_finish(0, nullptr, 0);
    TEST_ASSERT(em.finish_reason() == "length");
    bool found_length = false;
    for (const auto & chunk : chunks) {
        if (chunk.find("\"finish_reason\":\"length\"") != std::string::npos) {
            found_length = true;
            break;
        }
    }
    TEST_ASSERT(found_length);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_stop_sequence_beats_length_at_cap) {
    std::vector<std::string> stops = {"END"};
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT, stops);
    em.emit_start();
    em.emit_token("finished END");
    auto chunks = em.emit_finish(10, nullptr, 10);
    TEST_ASSERT(em.stop_hit());
    TEST_ASSERT(em.finish_reason() == "stop");
    bool found_stop = false;
    for (const auto & chunk : chunks) {
        if (chunk.find("\"finish_reason\":\"stop\"") != std::string::npos) {
            found_stop = true;
            break;
        }
    }
    TEST_ASSERT(found_stop);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_anthropic_length_finish_reason_at_cap) {
    auto em = make_emitter(ApiFormat::ANTHROPIC, json::array(), false);
    em.emit_start();
    em.emit_token("hello world");
    auto chunks = em.emit_finish(10, nullptr, 10);
    TEST_ASSERT(em.finish_reason() == "length");
    std::string text = concat(chunks);
    TEST_ASSERT(text.find("\"stop_reason\":\"max_tokens\"") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_anthropic_stop_sequence_beats_length_at_cap) {
    std::vector<std::string> stops = {"END"};
    auto em = make_emitter_with_stops(ApiFormat::ANTHROPIC, stops);
    em.emit_start();
    em.emit_token("finished END");
    auto chunks = em.emit_finish(10, nullptr, 10);
    TEST_ASSERT(em.stop_hit());
    TEST_ASSERT(em.finish_reason() == "stop");
    std::string text = concat(chunks);
    TEST_ASSERT(text.find("\"stop_reason\":\"end_turn\"") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_json_syntax_error_openai_delta) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), false);
    em.emit_start();
    em.emit_token("<function_call>{\"name\": \"read\", \"arguments\": {\"offset\": 5o1}}</function_call>");
    auto chunks = em.emit_finish(10, nullptr, -1);
    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "read");
        TEST_ASSERT(em.tool_calls()[0].arguments == "{\"offset\": 5o1}");
    }
    TEST_ASSERT(em.finish_reason() == "tool_calls");
    std::string text = concat(chunks);
    TEST_ASSERT(text.find("5o1") != std::string::npos);
    TEST_ASSERT(text.find("\"finish_reason\":\"tool_calls\"") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_json_syntax_error_anthropic) {
    auto em = make_emitter(ApiFormat::ANTHROPIC, read_tools(), false);
    em.emit_start();
    em.emit_token("<function_call>{\"name\": \"read\", \"arguments\": {\"offset\": 5o1}}</function_call>");
    auto chunks = em.emit_finish(10, nullptr, -1);
    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "read");
        TEST_ASSERT(em.tool_calls()[0].arguments == "{\"offset\": 5o1}");
    }
    std::string text = concat(chunks);
    TEST_ASSERT(text.find("\"type\":\"tool_use\"") != std::string::npos);
    TEST_ASSERT(text.find("5o1") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_malformed_tool_in_think_recovers_answer) {
    auto em = make_emitter(ApiFormat::ANTHROPIC, read_tools(), true);
    em.emit_start();
    auto c1 = em.emit_token("<think>Let me see <tool_call><bad_tool></tool_call></think>Here is the final answer.");
    auto c2 = em.emit_finish(10, nullptr, -1);
    std::string text = concat(c1) + concat(c2);
    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.reasoning_text().find("<bad_tool>") == std::string::npos);
    TEST_ASSERT(em.reasoning_text().find("Let me see ") != std::string::npos);
    TEST_ASSERT(em.accumulated_text() == "Here is the final answer.");
    TEST_ASSERT(text.find("\"type\":\"thinking_delta\"") != std::string::npos);
    TEST_ASSERT(text.find("\"type\":\"text_delta\"") != std::string::npos);
    TEST_ASSERT(text.find("Here is the final answer.") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_malformed_tool_apostrophe_recovers_answer) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), true);
    em.emit_start();
    em.emit_token(
        "<think>Inspect <tool_call>it's malformed</tool_call></think>Recovered answer.");
    em.emit_finish(10, nullptr, -1);

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.reasoning_text().find("it's malformed") == std::string::npos);
    TEST_ASSERT(em.accumulated_text() == "Recovered answer.");
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_malformed_tool_in_think_without_think_close_suppressed) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), true);
    em.emit_start();
    em.emit_token("<think>Let me see <tool_call><bad_tool>");
    auto chunks = em.emit_finish(10, nullptr, -1);
    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text().empty());
    TEST_ASSERT(em.reasoning_text().find("<bad_tool>") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_malformed_tool_in_think_literal_think_close_in_args_does_not_leak) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), true);
    em.emit_start();
    em.emit_token("<think>Let me see <tool_call><function=bash><parameter=cmd>grep '</think>' file.txt");
    auto chunks = em.emit_finish(10, nullptr, -1);
    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text().empty());
    TEST_ASSERT(em.reasoning_text().find("grep") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_malformed_tool_in_think_envelope_with_internal_think_close_recovers_answer) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), true);
    em.emit_start();
    em.emit_token("<think>Let me see <tool_call><function=bash><parameter=cmd>grep '</think>' file.txt</parameter></function></tool_call></think>Real answer here.");
    auto chunks = em.emit_finish(10, nullptr, -1);
    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text() == "Real answer here.");
    TEST_ASSERT(em.reasoning_text().find("grep") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_long_tool_name_split_in_reasoning_holds_back) {
    json tools = json::array({
        {
            {"type", "function"},
            {"function", {
                {"name", "fetch_authenticated_user_profile_data"},
                {"description", "fetch user profile"},
                {"parameters", {{"type", "object"}, {"properties", {{"id", {{"type", "string"}}}}}}}
            }}
        }
    });

    auto em = make_emitter(ApiFormat::OPENAI_CHAT, tools, true);
    em.emit_start();
    em.emit_token("<think>I should fetch the user data. ");
    em.emit_token("<fetch_authenticated_");
    em.emit_token("user_profile_data>\n<parameter=id>\n123\n</parameter>\n</fetch_authenticated_user_profile_data></think>");
    auto chunks = em.emit_finish(10, nullptr, -1);

    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "fetch_authenticated_user_profile_data");
        auto args = json::parse(em.tool_calls()[0].arguments);
        TEST_ASSERT(args["id"] == "123");
    }
    TEST_ASSERT(em.reasoning_text().find("fetch_authenticated") == std::string::npos);
    TEST_ASSERT(em.finish_reason() == "tool_calls");
}

TEST_CASE(ServerUnitFixture, test_emitter_prose_ending_in_tool_name_stays_content) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools());
    em.emit_start();
    em.emit_token("I cannot read");
    em.emit_finish(3);

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text() == "I cannot read");
}

TEST_CASE(ServerUnitFixture, test_emitter_tool_only_bare_name_remains_zero_arg_call) {
    json tools = json::array({
        {
            {"type", "function"},
            {"function", {
                {"name", "ping"},
                {"parameters", {{"type", "object"}, {"properties", json::object()}}}
            }}
        }
    });
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, tools);
    em.emit_start();
    em.emit_token("ping");
    em.emit_finish(1);

    TEST_ASSERT(em.tool_calls().size() == 1);
    if (!em.tool_calls().empty()) {
        TEST_ASSERT(em.tool_calls()[0].name == "ping");
        TEST_ASSERT(em.tool_calls()[0].arguments == "{}");
    }
    TEST_ASSERT(em.accumulated_text().empty());
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_malformed_tool_in_think_answer_containing_close_tag_recovers_answer) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), true);
    em.emit_start();
    em.emit_token("<think><tool_call><bad_param></tool_call></think>The output is </parameter> end.");
    auto chunks = em.emit_finish(10, nullptr, -1);
    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text() == "The output is </parameter> end.");
}


TEST_CASE(ServerUnitFixture, test_build_response_suppresses_length_finish_reason_on_eos) {
    // EOS wins even when it lands exactly on the configured token cap.
    auto em_openai = make_emitter(ApiFormat::OPENAI_CHAT);
    em_openai.emit_start();
    em_openai.emit_token("Hello world");
    auto chunks_openai = em_openai.emit_finish(3, nullptr, 3, true);
    TEST_ASSERT(em_openai.finish_reason() == "stop");

    auto em_anthropic = make_emitter(ApiFormat::ANTHROPIC);
    em_anthropic.emit_start();
    em_anthropic.emit_token("Hello world");
    auto chunks_anthropic = em_anthropic.emit_finish(3, nullptr, 3, true);
    TEST_ASSERT(em_anthropic.finish_reason() == "stop");
}

TEST_CASE(ServerUnitFixture, test_emitter_suppresses_undeclared_bailing_tool_block) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();

    const auto deliver = [&](const std::string & raw_token) {
        if (!em.suppress_undeclared_tool_protocol_token(raw_token)) {
            em.emit_token(raw_token);
        }
    };
    deliver("<tool_call>");
    deliver("weather");
    deliver("<arg_key>");
    deliver("city");
    deliver("</arg_key>");
    deliver("<arg_value>");
    deliver("Rome");
    deliver("</arg_value>");
    deliver("</tool_call>");
    deliver("Visible answer");
    em.emit_finish(10);

    TEST_ASSERT(em.accumulated_text() == "Visible answer");
    TEST_ASSERT(em.accumulated_text().find("weather") == std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("Rome") == std::string::npos);
}
