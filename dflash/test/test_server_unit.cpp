// Unit tests for server components — no GPU, no model files required.
//
// Tests: SseEmitter, ToolParser, Reasoning, PrefixCache (hash/boundary),
//        UTF-8 utilities.
//
// Ported from ds4_server.c's ds4_server_unit_tests_run() pattern.
// Build: cmake --build . --target test_server_unit
// Run:   ./test_server_unit

#include "server/sse_emitter.h"
#include "server/tool_parser.h"
#include "server/reasoning.h"
#include "server/prefix_cache.h"
#include "server/disk_prefix_cache.h"
#include "server/utf8_utils.h"
#include "server/api_types.h"
#include "server/http_server.h"
#include "server/chat_template.h"
#include "common/sampler.h"
#include "common/backend_ipc.h"
#include "placement/pflash_placement.h"
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

namespace dflash::common {
std::vector<ChatMessage> normalize_chat_messages(
    const json & messages,
    ApiFormat format,
    ToolMemory & tool_memory);
}

// ─── Test framework (ds4 style) ────────────────────────────────────────

static int test_failures = 0;
static int test_count = 0;
static const char * current_test = nullptr;

#define TEST_ASSERT(expr) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define TEST_ASSERT_MSG(expr, msg) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s — %s\n", __FILE__, __LINE__, #expr, msg); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    current_test = #fn; \
    std::fprintf(stderr, "  %s ...", #fn); \
    int before = test_failures; \
    fn(); \
    if (test_failures == before) std::fprintf(stderr, " ok\n"); \
    else std::fprintf(stderr, "\n"); \
} while (0)

// ─── Helper: create an SseEmitter with minimal config ──────────────────

static json weather_tools() {
    return json::array({
        {{"type", "function"},
         {"function", {
             {"name", "get_weather"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"location", {{"type", "string"}}},
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}},
        {{"type", "function"},
         {"function", {
             {"name", "terminal"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
}

static SseEmitter make_emitter(ApiFormat fmt, json tools = json::array()) {
    return SseEmitter(fmt, "test_id_001", "test-model", 10,
                      tools, nullptr);
}

// Concatenate all SSE chunks into a single string.
static std::string concat(const std::vector<std::string> & chunks) {
    std::string out;
    for (const auto & c : chunks) out += c;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════
// UTF-8 utility tests
// ═══════════════════════════════════════════════════════════════════════

static void test_utf8_safe_len_ascii() {
    std::string s = "Hello, world!";
    TEST_ASSERT(utf8_safe_len(s, s.size()) == s.size());
    TEST_ASSERT(utf8_safe_len(s, 5) == 5);
    TEST_ASSERT(utf8_safe_len(s, 0) == 0);
}

static void test_utf8_safe_len_partial_2byte() {
    // é = 0xC3 0xA9
    std::string s = "caf\xC3\xA9!";  // "café!"
    TEST_ASSERT(utf8_safe_len(s, 5) == 5);  // after é, ok
    TEST_ASSERT(utf8_safe_len(s, 4) == 3);  // mid-é, snap back to before é
}

static void test_utf8_safe_len_partial_3byte() {
    // ん = 0xE3 0x82 0x93
    std::string s = "A\xE3\x82\x93Z";  // "AんZ"
    TEST_ASSERT(utf8_safe_len(s, 4) == 4);  // after ん
    TEST_ASSERT(utf8_safe_len(s, 3) == 1);  // mid-ん, snap back to A
    TEST_ASSERT(utf8_safe_len(s, 2) == 1);  // mid-ん, snap back to A
}

static void test_utf8_safe_len_partial_4byte() {
    // 🚩 = 0xF0 0x9F 0x9A 0xA9
    std::string s = "A \xF0\x9F\x9A\xA9 done";
    TEST_ASSERT(utf8_safe_len(s, 6) == 6);  // after 🚩
    // Mid-emoji should snap back to position 2 (before 🚩)
    TEST_ASSERT(utf8_safe_len(s, 5) == 2);
    TEST_ASSERT(utf8_safe_len(s, 4) == 2);
    TEST_ASSERT(utf8_safe_len(s, 3) == 2);
}

static void test_utf8_sanitize_valid() {
    std::string s = "Hello, world! 🎉";
    TEST_ASSERT(utf8_sanitize(s) == s);
}

static void test_utf8_sanitize_replaces_invalid() {
    // Lone continuation byte
    std::string s = "A\x80Z";
    std::string out = utf8_sanitize(s);
    TEST_ASSERT(out == "A\xEF\xBF\xBDZ");

    // Truncated 4-byte sequence
    std::string s2 = "X\xF0\x9F";
    std::string out2 = utf8_sanitize(s2);
    // Each invalid byte becomes U+FFFD
    TEST_ASSERT(out2.find("X") == 0);
    TEST_ASSERT(out2.size() > 1);  // has replacement(s)
}

static void test_utf8_sanitize_empty() {
    TEST_ASSERT(utf8_sanitize("") == "");
}

// ═══════════════════════════════════════════════════════════════════════
// Reasoning parser tests
// ═══════════════════════════════════════════════════════════════════════

static void test_reasoning_basic() {
    auto r = parse_reasoning("<think>I need to think</think>The answer is 42");
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "I need to think");
    TEST_ASSERT(r.content == "The answer is 42");
}

static void test_reasoning_no_tags() {
    auto r = parse_reasoning("Just plain text");
    TEST_ASSERT(!r.has_reasoning);
    TEST_ASSERT(r.content == "Just plain text");
}

static void test_reasoning_started_in_thinking() {
    auto r = parse_reasoning("thinking body</think>content here",
                             true, true);
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "thinking body");
    TEST_ASSERT(r.content == "content here");
}

static void test_reasoning_unclosed_think() {
    auto r = parse_reasoning("<think>still thinking no close",
                             true, false);
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "still thinking no close");
    TEST_ASSERT(r.content.empty());
}

static void test_reasoning_empty_thinking() {
    auto r = parse_reasoning("<think></think>answer");
    TEST_ASSERT(!r.has_reasoning);  // empty reasoning
    TEST_ASSERT(r.content == "answer");
}

static void test_reasoning_whitespace_in_think() {
    auto r = parse_reasoning("<think>\n  reasoning \n</think>\ncontent");
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "reasoning");
    TEST_ASSERT(r.content == "content");
}

static void test_reasoning_disabled() {
    // When thinking disabled but tags present, the parser still finds them
    // (the caller decides whether to use the reasoning field).
    auto r = parse_reasoning("<think>ignored</think>content",
                             false, false);
    // Tags are still parsed — has_reasoning is true because reasoning text is non-empty
    TEST_ASSERT(r.content == "content");
}

// ═══════════════════════════════════════════════════════════════════════
// Tool parser tests
// ═══════════════════════════════════════════════════════════════════════

static void test_parse_tool_call_xml() {
    std::string text =
        "Some text\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=location>San Francisco</parameter>\n"
        "<parameter=unit>celsius</parameter>\n"
        "</function>\n"
        "</tool_call>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args.contains("location"));
        TEST_ASSERT(args["location"] == "San Francisco");
        TEST_ASSERT(args.contains("unit"));
        TEST_ASSERT(args["unit"] == "celsius");
    }
    TEST_ASSERT(result.cleaned_text.find("<tool_call>") == std::string::npos);
}

static void test_parse_bare_function_xml() {
    std::string text =
        "<function=list_files>\n"
        "<parameter=path>/home</parameter>\n"
        "</function>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "list_files");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home");
    }
}

static void test_parse_json_tool_call() {
    std::string text =
        "{\"name\": \"search\", \"arguments\": {\"query\": \"hello world\"}}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "search");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["query"] == "hello world");
    }
}

static void test_parse_no_tools() {
    std::string text = "Just plain text without any tool calls.";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(!result.cleaned_text.empty());
}

static void test_parse_tool_code_wrapper() {
    std::string text =
        "<tool_code>\n"
        "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}\n"
        "</tool_code>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
    }
}

static void test_parse_tool_allowed_filter() {
    std::string text =
        "<function=blocked_tool>\n"
        "<parameter=x>1</parameter>\n"
        "</function>";
    json tools = json::array({
        {{"type", "function"}, {"function", {{"name", "allowed_tool"}}}}
    });
    auto result = parse_tool_calls(text, tools);
    // Tool not in allow-list should be filtered
    TEST_ASSERT(result.tool_calls.empty());
}

// ═══════════════════════════════════════════════════════════════════════
// SSE Emitter tests
// ═══════════════════════════════════════════════════════════════════════

static void test_emitter_reasoning_split_openai() {
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
static void test_emitter_first_content_index_natural_close() {
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

static void test_emitter_first_content_index_never_closed() {
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

static void test_emitter_first_content_index_content_only() {
    // Non-thinking request: emitter starts in CONTENT mode, so the
    // very first emit_token lands at index 0.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("immediate content");
    em.emit_finish(1);

    TEST_ASSERT(em.first_content_token_index() == 0);
    TEST_ASSERT(em.emit_token_count() == 1);
}

static void test_emitter_first_content_index_qwen36_streaming_thinking() {
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

static void test_emitter_reasoning_strips_leading_think_tag() {
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

static void test_emitter_content_only_no_thinking() {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("Hello, world!");
    em.emit_finish(5);

    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
    TEST_ASSERT(em.reasoning_text().empty());
}

static void test_emitter_tool_buffer_detection() {
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

static void test_emitter_anthropic_tool_use_blocks() {
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

static void test_emitter_bare_function_tool_buffer_detection() {
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

static void test_emitter_does_not_leak_malformed_tool_xml() {
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

static void test_emitter_parses_tool_call_missing_outer_close() {
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

static void test_emitter_no_tools_keeps_tool_like_text() {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("<function=terminal>\n"
                  "<parameter=command>ls</parameter>\n"
                  "</function>");
    em.emit_finish(20);

    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text().find("<function=terminal>") != std::string::npos);
}

static void test_emitter_anthropic_structure() {
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

static void test_emitter_responses_structure() {
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

static void test_emitter_responses_bare_function_tool_call() {
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

static void test_emitter_streaming_openai_has_done() {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_start();
    em.emit_token("Hello");
    auto finish = em.emit_finish(3);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("[DONE]") != std::string::npos);
}

static void test_emitter_nonstreaming_accumulates() {
    // Non-streaming: tokens fed through emitter, accumulated_text() has all content.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT);
    em.emit_token("Hello ");
    em.emit_token("world");
    em.emit_finish(5);

    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("world") != std::string::npos);
}

static void test_emitter_anthropic_thinking_blocks() {
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

static SseEmitter make_emitter_with_stops(ApiFormat fmt,
                                           const std::vector<std::string> & stops) {
    return SseEmitter(fmt, "test_id_001", "test-model", 10,
                      json::array(), nullptr, stops);
}

static void test_stop_sequence_basic() {
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

static void test_stop_sequence_mid_token() {
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

static void test_stop_sequence_multiple() {
    // Multiple stop sequences — earliest match wins.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"AAA", "BB"});
    em.emit_token("xBBy");

    TEST_ASSERT(em.stop_hit());
    em.emit_finish(2);
    TEST_ASSERT(em.accumulated_text() == "x");
}

static void test_stop_sequence_no_match() {
    // No stop sequence hit — normal operation.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"NOMATCH"});
    em.emit_token("Hello world this is a long text");
    em.emit_finish(10);

    TEST_ASSERT(!em.stop_hit());
    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
}

static void test_stop_sequence_empty_list() {
    // Empty stop list — no effect.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{});
    em.emit_token("Hello STOP world");
    em.emit_finish(5);

    TEST_ASSERT(!em.stop_hit());
    TEST_ASSERT(em.accumulated_text().find("STOP") != std::string::npos);
}

static void test_stop_sequence_finish_reason() {
    // finish_reason should be "stop" when stop sequence hit.
    auto em = make_emitter_with_stops(ApiFormat::OPENAI_CHAT,{"END"});
    em.emit_token("content END more");

    TEST_ASSERT(em.stop_hit());
    em.emit_finish(3);
    TEST_ASSERT(em.finish_reason() == "stop");
}

static void test_stop_sequence_streaming_output() {
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

static void test_stop_sequence_anthropic_format() {
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

static void test_stop_sequence_in_reasoning_mode() {
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

static void test_stop_sequence_holdback_extends() {
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
// Prefix cache hash tests (model-free)
// ═══════════════════════════════════════════════════════════════════════

static void test_hash_prefix_deterministic() {
    std::vector<int32_t> ids = {100, 200, 300, 400, 500};
    auto h1 = hash_prefix(ids.data(), (int)ids.size());
    auto h2 = hash_prefix(ids.data(), (int)ids.size());
    TEST_ASSERT(h1 == h2);
}

static void test_hash_prefix_different_inputs() {
    std::vector<int32_t> ids1 = {100, 200, 300};
    std::vector<int32_t> ids2 = {100, 200, 301};
    auto h1 = hash_prefix(ids1.data(), (int)ids1.size());
    auto h2 = hash_prefix(ids2.data(), (int)ids2.size());
    TEST_ASSERT(h1 != h2);
}

static void test_hash_prefix_different_lengths() {
    std::vector<int32_t> ids1 = {100, 200, 300};
    std::vector<int32_t> ids2 = {100, 200, 300, 400};
    auto h1 = hash_prefix(ids1.data(), (int)ids1.size());
    auto h2 = hash_prefix(ids2.data(), (int)ids2.size());
    TEST_ASSERT(h1 != h2);
}

static void test_hash_prefix_empty() {
    auto h = hash_prefix(nullptr, 0);
    // Should not crash, just return a hash of empty input
    TEST_ASSERT(h.size() == 16);
}

static void test_find_boundaries_empty() {
    ChatMarkers markers;
    markers.family = "qwen";
    std::vector<int32_t> ids;
    auto bounds = find_all_boundaries(ids, markers);
    TEST_ASSERT(bounds.empty());
}

// ═══════════════════════════════════════════════════════════════════════
// PFlash config tests (model-free)
// ═══════════════════════════════════════════════════════════════════════

static void test_pflash_config_defaults() {
    ServerConfig cfg;
    TEST_ASSERT(cfg.pflash_mode == ServerConfig::PflashMode::OFF);
    TEST_ASSERT(cfg.pflash_threshold == 32000);
    TEST_ASSERT(cfg.pflash_keep_ratio > 0.04f && cfg.pflash_keep_ratio < 0.06f);
    TEST_ASSERT(cfg.pflash_drafter_path.empty());
    TEST_ASSERT(!cfg.pflash_skip_park);
}

static void test_pflash_config_modes() {
    ServerConfig cfg;
    cfg.pflash_mode = ServerConfig::PflashMode::AUTO;
    TEST_ASSERT(cfg.pflash_mode != ServerConfig::PflashMode::OFF);

    cfg.pflash_mode = ServerConfig::PflashMode::ALWAYS;
    TEST_ASSERT(cfg.pflash_mode != ServerConfig::PflashMode::OFF);
    TEST_ASSERT(cfg.pflash_mode != ServerConfig::PflashMode::AUTO);
}

static void test_pflash_compress_request_struct() {
    ModelBackend::CompressRequest req;
    req.input_ids = {1, 2, 3, 4, 5};
    req.keep_ratio = 0.05f;
    req.drafter_path = "/path/to/drafter.gguf";
    req.skip_park = true;

    TEST_ASSERT(req.input_ids.size() == 5);
    TEST_ASSERT(req.keep_ratio > 0.0f);
    TEST_ASSERT(!req.drafter_path.empty());
    TEST_ASSERT(req.skip_park);
}

static void test_pflash_compress_result_defaults() {
    ModelBackend::CompressResult result;
    TEST_ASSERT(!result.ok);
    TEST_ASSERT(result.compressed_ids.empty());
}

static void test_pflash_threshold_auto_mode() {
    // Simulate the threshold check logic from http_server.cpp
    ServerConfig cfg;
    cfg.pflash_mode = ServerConfig::PflashMode::AUTO;
    cfg.pflash_threshold = 1000;

    // Below threshold: don't compress
    int n_prompt = 500;
    bool should = (cfg.pflash_mode == ServerConfig::PflashMode::ALWAYS) ||
                  (cfg.pflash_mode == ServerConfig::PflashMode::AUTO && n_prompt >= cfg.pflash_threshold);
    TEST_ASSERT(!should);

    // Above threshold: compress
    n_prompt = 2000;
    should = (cfg.pflash_mode == ServerConfig::PflashMode::ALWAYS) ||
             (cfg.pflash_mode == ServerConfig::PflashMode::AUTO && n_prompt >= cfg.pflash_threshold);
    TEST_ASSERT(should);
}

static void test_pflash_threshold_always_mode() {
    ServerConfig cfg;
    cfg.pflash_mode = ServerConfig::PflashMode::ALWAYS;

    // Even small prompts should compress in ALWAYS mode
    int n_prompt = 10;
    bool should = (cfg.pflash_mode == ServerConfig::PflashMode::ALWAYS) ||
                  (cfg.pflash_mode == ServerConfig::PflashMode::AUTO && n_prompt >= cfg.pflash_threshold);
    TEST_ASSERT(should);
}

static void test_pflash_placement_same_backend_local() {
    DevicePlacement target;
    target.backend = compiled_placement_backend();
    target.gpu = 0;
    DevicePlacement drafter;
    drafter.backend = target.backend;
    drafter.gpu = 2;
    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";

    auto placement = resolve_pflash_drafter_placement(
        target, drafter, remote, /*pflash_enabled=*/true);
    TEST_ASSERT(placement.target_backend == target.backend);
    TEST_ASSERT(placement.drafter_backend == target.backend);
    TEST_ASSERT(placement.drafter_gpu == 2);
    TEST_ASSERT(!placement.remote_drafter);
    TEST_ASSERT(!placement.remote.enabled());
}

static void test_pflash_placement_mixed_backend_remote() {
    DevicePlacement target;
    target.backend = PlacementBackend::Cuda;
    target.gpu = 0;
    DevicePlacement drafter;
    drafter.backend = PlacementBackend::Hip;
    drafter.gpu = 1;
    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";
    remote.work_dir = "/tmp/pflash-ipc";

    auto placement = resolve_pflash_drafter_placement(
        target, drafter, remote, /*pflash_enabled=*/true);
    TEST_ASSERT(placement.target_backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.drafter_backend == PlacementBackend::Hip);
    TEST_ASSERT(placement.drafter_gpu == 1);
    TEST_ASSERT(placement.remote_drafter);
    TEST_ASSERT(placement.remote.enabled());
    TEST_ASSERT(placement.remote.work_dir == "/tmp/pflash-ipc");
}

static void test_pflash_placement_auto_draft_follows_target() {
    DevicePlacement target;
    target.backend = PlacementBackend::Hip;
    target.gpu = 0;
    DevicePlacement drafter;
    drafter.backend = PlacementBackend::Auto;
    drafter.gpu = 3;
    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";

    auto placement = resolve_pflash_drafter_placement(
        target, drafter, remote, /*pflash_enabled=*/true);
    TEST_ASSERT(placement.target_backend == PlacementBackend::Hip);
    TEST_ASSERT(placement.drafter_backend == PlacementBackend::Hip);
    TEST_ASSERT(placement.drafter_gpu == 3);
    TEST_ASSERT(!placement.remote_drafter);
}

static void test_pflash_placement_disabled_never_remote() {
    DevicePlacement target;
    target.backend = PlacementBackend::Cuda;
    DevicePlacement drafter;
    drafter.backend = PlacementBackend::Hip;
    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";

    auto placement = resolve_pflash_drafter_placement(
        target, drafter, remote, /*pflash_enabled=*/false);
    TEST_ASSERT(placement.target_backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.drafter_backend == PlacementBackend::Hip);
    TEST_ASSERT(!placement.remote_drafter);
    TEST_ASSERT(!placement.remote.enabled());
}

static void test_pflash_placement_usage_gate() {
    TEST_ASSERT(!pflash_drafter_placement_used(
        /*pflash_enabled=*/false, /*has_decode_draft=*/false));
    TEST_ASSERT(pflash_drafter_placement_used(
        /*pflash_enabled=*/false, /*has_decode_draft=*/true));
    TEST_ASSERT(pflash_drafter_placement_used(
        /*pflash_enabled=*/true, /*has_decode_draft=*/false));
    TEST_ASSERT(pflash_drafter_placement_used(
        /*pflash_enabled=*/true, /*has_decode_draft=*/true));
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

static void test_jinja_render_basic() {
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

static void test_jinja_render_no_gen_prompt() {
    std::vector<ChatMessage> msgs = {{"user", "ping", ""}};
    std::string out = render_chat_template_jinja(
        MINI_JINJA_TEMPLATE, msgs, "", "",
        /*add_gen=*/false, /*think=*/false, "");
    TEST_ASSERT(out.find("<|user|>ping") != std::string::npos);
    TEST_ASSERT(out.find("<|assistant|>") == std::string::npos);
}

static void test_jinja_render_tools_injected() {
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

static void test_jinja_render_empty_tools_skipped() {
    // tools_json == "[]" must NOT define `tools` in the template context.
    static const char TPL[] =
        "{%- if tools -%}TOOLS_PRESENT{%- else -%}NO_TOOLS{%- endif -%}";
    std::vector<ChatMessage> msgs = {{"user", "?", ""}};
    std::string out = render_chat_template_jinja(
        TPL, msgs, "", "", false, false, "[]");
    TEST_ASSERT(out.find("NO_TOOLS")        != std::string::npos);
    TEST_ASSERT(out.find("TOOLS_PRESENT")   == std::string::npos);
}

static void test_jinja_render_bos_eos_threaded() {
    // {{ bos_token }} and {{ eos_token }} must reach the template.
    static const char TPL[] = "{{ bos_token }}HI{{ eos_token }}";
    std::vector<ChatMessage> msgs;
    std::string out = render_chat_template_jinja(
        TPL, msgs, "<BOS>", "<EOS>", false, false, "");
    TEST_ASSERT(out == "<BOS>HI<EOS>");
}

static void test_jinja_render_empty_template_throws() {
    std::vector<ChatMessage> msgs = {{"user", "x", ""}};
    bool threw = false;
    try {
        (void)render_chat_template_jinja("", msgs, "", "", true, false, "");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    TEST_ASSERT(threw);
}

static void test_jinja_render_bad_tools_json_throws() {
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

static void test_normalize_responses_tool_followup_messages() {
    ToolMemory tool_memory;
    const std::string call_id = "call_exec_001";
    const std::string raw_tool_call =
        "\n\n<function=exec_command>\n"
        "<parameter=cmd>\n"
        "git fetch origin && git status\n"
        "</parameter>\n"
        "</function>\n";
    tool_memory.remember({call_id}, raw_tool_call);

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
            {"type", "function_call_output"},
            {"call_id", call_id},
            {"output", "Process exited with code 0"}
        }
    });

    auto chat_msgs = normalize_chat_messages(messages, ApiFormat::RESPONSES, tool_memory);
    TEST_ASSERT(chat_msgs.size() == 4);
    if (chat_msgs.size() == 4) {
        TEST_ASSERT(chat_msgs[0].role == "system");
        TEST_ASSERT(chat_msgs[0].content == "Developer rules");
        TEST_ASSERT(chat_msgs[1].role == "user");
        TEST_ASSERT(chat_msgs[1].content == "fetch latest code");
        TEST_ASSERT(chat_msgs[2].role == "assistant");
        TEST_ASSERT(chat_msgs[2].content == raw_tool_call);
        TEST_ASSERT(chat_msgs[3].role == "tool");
        TEST_ASSERT(chat_msgs[3].tool_call_id == call_id);
        TEST_ASSERT(chat_msgs[3].content == "Process exited with code 0");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Disk Prefix Cache Tests
// ═══════════════════════════════════════════════════════════════════════

// Minimal mock backend for testing (no GPU needed).
struct MockBackend : ModelBackend {
    void print_ready_banner() const override {}
    bool park(const std::string &) override { return true; }
    bool unpark(const std::string &) override { return true; }
    bool is_target_parked() const override { return false; }
    GenerateResult generate(const GenerateRequest &, const DaemonIO &) override { return {}; }
    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int  snapshot_cur_pos(int) const override { return 0; }
    GenerateResult restore_and_generate(int, const GenerateRequest &, const DaemonIO &) override { return {}; }
    bool handle_compress(const std::string &, const DaemonIO &) override { return false; }
    void free_drafter() override {}
    void shutdown() override {}
};

// Helper: recursively remove a directory.
static void rm_rf(const std::string & path) {
    DIR * dir = opendir(path.c_str());
    if (!dir) { unlink(path.c_str()); return; }
    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (std::strcmp(ent->d_name, ".") == 0 || std::strcmp(ent->d_name, "..") == 0) continue;
        std::string child = path + "/" + ent->d_name;
        struct stat st;
        if (stat(child.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            rm_rf(child);
        } else {
            unlink(child.c_str());
        }
    }
    closedir(dir);
    rmdir(path.c_str());
}

static void test_disk_cache_config_defaults() {
    DiskCacheConfig cfg;
    TEST_ASSERT(cfg.cache_dir.empty());
    TEST_ASSERT(cfg.budget_bytes == (size_t)4 * 1024 * 1024 * 1024);
    TEST_ASSERT(cfg.min_tokens == 512);
    TEST_ASSERT(cfg.continued_interval == 10240);
    TEST_ASSERT(cfg.cold_max_tokens == 10240);
}

static void test_disk_cache_disabled_when_no_dir() {
    MockBackend backend;
    DiskCacheConfig cfg;
    cfg.cache_dir = "";
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.disabled());
    // Operations should be no-ops.
    std::vector<int32_t> ids = {1, 2, 3, 4, 5};
    TEST_ASSERT(!cache.lookup(ids, 0));
    TEST_ASSERT(!cache.save(0, ids));
}

static void test_disk_cache_init_creates_directory() {
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_disk_cache_init";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(!cache.disabled());
    TEST_ASSERT(cache.init());

    // Directory should exist.
    struct stat st;
    TEST_ASSERT(stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode));

    rm_rf(dir);
}

static void test_disk_cache_header_size() {
    // The header should be exactly 80 bytes.
    TEST_ASSERT(DISK_CACHE_HEADER_SIZE == 80);
    TEST_ASSERT(DISK_CACHE_VERSION == 1);
}

static void test_disk_cache_header_round_trip() {
    // Write and read a header to verify serialization.
    std::string path = "/tmp/dflash_test_header_rt.dkv";
    unlink(path.c_str());

    DiskCacheHeader hdr{};
    std::memcpy(hdr.magic, "DKVC", 4);
    hdr.version = DISK_CACHE_VERSION;
    std::memset(hdr.layout_id, 0xAB, 16);
    hdr.cur_pos = 1234;
    hdr.n_tensors = 42;
    hdr.token_count = 567;
    std::memset(hdr.token_hash, 0xCD, 16);
    hdr.payload_bytes = 9999999;
    hdr.created_at = 1700000000;
    hdr.last_used = 1700000100;
    hdr.last_tok = 151643;

    // Use DiskPrefixCache's static write/read_header (they are private, so
    // we test indirectly through file I/O matching the on-disk format).
    FILE * f = std::fopen(path.c_str(), "wb");
    TEST_ASSERT(f != nullptr);
    // Write field-by-field matching disk_prefix_cache.cpp's write_header.
    std::fwrite(hdr.magic, 4, 1, f);
    uint32_t v;
    v = hdr.version; std::fwrite(&v, 4, 1, f);
    std::fwrite(hdr.layout_id, 16, 1, f);
    v = hdr.cur_pos; std::fwrite(&v, 4, 1, f);
    v = hdr.n_tensors; std::fwrite(&v, 4, 1, f);
    v = hdr.token_count; std::fwrite(&v, 4, 1, f);
    std::fwrite(hdr.token_hash, 16, 1, f);
    uint64_t u64 = hdr.payload_bytes; std::fwrite(&u64, 8, 1, f);
    u64 = hdr.created_at; std::fwrite(&u64, 8, 1, f);
    u64 = hdr.last_used; std::fwrite(&u64, 8, 1, f);
    int32_t i32 = hdr.last_tok; std::fwrite(&i32, 4, 1, f);
    std::fclose(f);

    // Verify file size is DISK_CACHE_HEADER_SIZE.
    struct stat st;
    stat(path.c_str(), &st);
    TEST_ASSERT((size_t)st.st_size == DISK_CACHE_HEADER_SIZE);

    // Read back and verify.
    f = std::fopen(path.c_str(), "rb");
    TEST_ASSERT(f != nullptr);
    char magic[4]; std::fread(magic, 4, 1, f);
    TEST_ASSERT(std::memcmp(magic, "DKVC", 4) == 0);
    uint32_t rv; std::fread(&rv, 4, 1, f);
    TEST_ASSERT(rv == DISK_CACHE_VERSION);
    uint8_t lid[16]; std::fread(lid, 16, 1, f);
    TEST_ASSERT(lid[0] == 0xAB && lid[15] == 0xAB);
    std::fread(&rv, 4, 1, f); TEST_ASSERT(rv == 1234);  // cur_pos
    std::fread(&rv, 4, 1, f); TEST_ASSERT(rv == 42);    // n_tensors
    std::fread(&rv, 4, 1, f); TEST_ASSERT(rv == 567);   // token_count
    uint8_t th[16]; std::fread(th, 16, 1, f);
    TEST_ASSERT(th[0] == 0xCD && th[15] == 0xCD);
    uint64_t ru64; std::fread(&ru64, 8, 1, f); TEST_ASSERT(ru64 == 9999999);  // payload
    std::fread(&ru64, 8, 1, f); TEST_ASSERT(ru64 == 1700000000);  // created_at
    std::fread(&ru64, 8, 1, f); TEST_ASSERT(ru64 == 1700000100);  // last_used
    int32_t ri32; std::fread(&ri32, 4, 1, f); TEST_ASSERT(ri32 == 151643);  // last_tok
    std::fclose(f);

    unlink(path.c_str());
}

static void test_disk_cache_continued_boundary() {
    // Test maybe_store_continued logic: saves at interval boundaries.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_continued";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.min_tokens = 100;
    cfg.continued_interval = 1000;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    // Without layout known, save should fail gracefully.
    std::vector<int32_t> tokens(1500, 42);
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 1000));

    // Reset continued tracking.
    cache.reset_continued();

    // Below interval, no save (even if tokens available).
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 500));

    // At exactly 1000 tokens — would save if layout were known.
    // But backend mock can't provide snapshots, so it fails gracefully.
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 1000));

    rm_rf(dir);
}

static void test_disk_cache_continued_interval_logic() {
    // Verify the continued boundary math independently.
    // Target = (cur_pos / interval) * interval
    // Only fires when target > last_store_pos AND target >= min_tokens.
    int interval = 10240;
    int min_tokens = 512;

    // cur_pos=10239: target = 10239/10240 * 10240 = 0. No save.
    int target = (10239 / interval) * interval;
    TEST_ASSERT(target == 0);

    // cur_pos=10240: target = 10240. Save.
    target = (10240 / interval) * interval;
    TEST_ASSERT(target == 10240);

    // cur_pos=20479: target = 10240. But if last_store=10240, no save.
    target = (20479 / interval) * interval;
    TEST_ASSERT(target == 10240);

    // cur_pos=20480: target = 20480. Save.
    target = (20480 / interval) * interval;
    TEST_ASSERT(target == 20480);

    // Verify min_tokens gate.
    int small_interval = 100;
    target = (150 / small_interval) * small_interval;
    TEST_ASSERT(target == 100);
    // target=100 < min_tokens=512, so the continued save should NOT fire.
    TEST_ASSERT(target < min_tokens);
    (void)min_tokens;
}

static void test_disk_cache_cold_prefix_short_prompt() {
    // Cold prefix should not trigger for short prompts.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_cold_short";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 10240;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    // Prompt shorter than cold_max_tokens.
    std::vector<int32_t> prompt(5000, 1);
    std::vector<int> boundaries = {1000, 2000, 3000, 4000};
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, boundaries) == 0);

    rm_rf(dir);
}

static void test_disk_cache_cold_prefix_no_boundaries() {
    // Cold prefix should not trigger if no boundaries provided.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_cold_nobound";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 5000;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    std::vector<int32_t> prompt(10000, 1);
    std::vector<int> empty_boundaries;
    TEST_ASSERT(cache.cold_prefix_boundary(prompt, empty_boundaries) == 0);

    rm_rf(dir);
}

static void test_disk_cache_cold_prefix_finds_boundary() {
    // Cold prefix should find the last boundary <= cold_max_tokens.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_cold_finds";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.cold_max_tokens = 5000;
    cfg.min_tokens = 512;
    DiskPrefixCache cache(cfg, backend);
    cache.init();
    // Manually mark layout as known (hack for testing without real snapshots).
    // Since cold_prefix_boundary checks layout_known_, and we can't easily
    // set it without a real snapshot, the function will return 0.
    // This tests that short prompts / bad boundaries correctly return 0.
    std::vector<int32_t> prompt(10000, 1);
    std::vector<int> boundaries = {1000, 2000, 3000, 4000, 6000, 8000};
    // Without layout_known_, returns 0.
    int result = cache.cold_prefix_boundary(prompt, boundaries);
    TEST_ASSERT(result == 0);  // layout not known yet

    rm_rf(dir);
}

static void test_disk_cache_budget_enforcement_scoring() {
    // Test that eviction scoring prefers lower-value entries.
    // score = (hits+1) * token_count / file_size
    // Entry with fewer tokens + fewer hits should have lower score.

    // Simulate: entry A: 100 tokens, 0 hits, 1MB → score = 1*100/1M = 0.0001
    //           entry B: 10000 tokens, 5 hits, 1MB → score = 6*10000/1M = 0.06
    // Entry A should be evicted first.
    double score_a = (0.0 + 1.0) * 100.0 / (1024.0 * 1024.0);
    double score_b = (5.0 + 1.0) * 10000.0 / (1024.0 * 1024.0);
    TEST_ASSERT(score_a < score_b);

    // With time decay: entry B with 24h old hits (4 half-lives = 0.0625 remaining)
    double decay_24h = std::exp(-86400.0 * 3.2e-5);  // ~0.064
    double score_b_decayed = (5.0 * decay_24h + 1.0) * 10000.0 / (1024.0 * 1024.0);
    // Should still be higher than A since (5*0.064+1)=1.32 > 1.0
    TEST_ASSERT(score_b_decayed > score_a);

    // With 7 days old (massive decay), hits are nearly zero:
    double decay_7d = std::exp(-604800.0 * 3.2e-5);  // ~5e-9
    double score_b_ancient = (5.0 * decay_7d + 1.0) * 10000.0 / (1024.0 * 1024.0);
    // (5*~0 + 1)*10000/1M ≈ 0.01 — still > score_a since more tokens
    TEST_ASSERT(score_b_ancient > score_a);
}

static void test_disk_cache_lookup_miss_no_layout() {
    // Lookup with no layout known should return false.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_lookup_miss";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8};
    TEST_ASSERT(!cache.lookup(ids, 0));

    rm_rf(dir);
}

static void test_disk_cache_save_below_min_tokens() {
    // Save with fewer tokens than min_tokens should be rejected.
    MockBackend backend;
    std::string dir = "/tmp/dflash_test_save_below";
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    cfg.min_tokens = 100;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    std::vector<int32_t> ids(50, 1);  // only 50 tokens
    TEST_ASSERT(!cache.save(0, ids));

    rm_rf(dir);
}

static void test_backend_ipc_rejects_file_work_dir() {
    const std::string file_path = "/tmp/dflash_test_backend_ipc_work_dir_file";
    unlink(file_path.c_str());
    int fd = open(file_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    TEST_ASSERT(fd >= 0);
    if (fd >= 0) {
        const char payload[] = "not a dir";
        (void)write(fd, payload, sizeof(payload) - 1);
        close(fd);
    }

    BackendIpcLaunchConfig cfg;
    cfg.bin = "/bin/true";
    cfg.payload_path = "/tmp/dflash_test_backend_ipc_payload";
    cfg.work_dir = file_path;

    BackendIpcProcess proc;
    TEST_ASSERT(!proc.start(cfg));
    TEST_ASSERT(!proc.active());
    unlink(file_path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════
// Sampler tests (model-independent, CPU-only)
// ═══════════════════════════════════════════════════════════════════════

static void test_sampler_cfg_defaults() {
    SamplerCfg cfg;
    TEST_ASSERT(cfg.temp == 0.0f);
    TEST_ASSERT(cfg.top_p == 1.0f);
    TEST_ASSERT(cfg.top_k == 0);
    TEST_ASSERT(cfg.rep_pen == 1.0f);
    TEST_ASSERT(cfg.rep_window == 256);
    TEST_ASSERT(cfg.seed == 0);
    TEST_ASSERT(cfg.freq_pen == 0.0f);
    TEST_ASSERT(cfg.pres_pen == 0.0f);
}

static void test_sampler_greedy_argmax() {
    // With temp=0 logic, caller uses argmax. But sample_logits with very
    // low temp should still pick the highest logit token reliably.
    float logits[] = {1.0f, 5.0f, 2.0f, 3.0f, 0.5f};
    SamplerCfg cfg;
    cfg.temp = 0.001f;  // near-zero temp → essentially greedy
    std::vector<int32_t> history;
    std::mt19937_64 rng(42);

    int tok = sample_logits(logits, 5, cfg, history, rng);
    TEST_ASSERT(tok == 1);  // token 1 has logit 5.0 (highest)
}

static void test_sampler_temperature_affects_distribution() {
    // High temperature should spread probability; verify by sampling many
    // times and checking that non-top tokens appear.
    float logits[] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    SamplerCfg cfg;
    cfg.temp = 2.0f;  // high temp → more uniform
    std::vector<int32_t> history;
    std::mt19937_64 rng(123);

    int counts[5] = {};
    for (int i = 0; i < 1000; i++) {
        int tok = sample_logits(logits, 5, cfg, history, rng);
        TEST_ASSERT(tok >= 0 && tok < 5);
        counts[tok]++;
    }
    // With high temp, non-max tokens should appear frequently
    TEST_ASSERT(counts[0] > 50);  // token 0 should appear sometimes
    TEST_ASSERT(counts[1] > 100); // token 1 still most likely
}

static void test_sampler_top_p_truncation() {
    // With very low top_p, only the top token(s) should be selected.
    float logits[] = {0.0f, 10.0f, 0.0f, 0.0f, 0.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.top_p = 0.01f;  // very restrictive → only the top token
    std::vector<int32_t> history;
    std::mt19937_64 rng(42);

    for (int i = 0; i < 100; i++) {
        int tok = sample_logits(logits, 5, cfg, history, rng);
        TEST_ASSERT(tok == 1);  // only token 1 should survive top_p
    }
}

static void test_sampler_top_k_truncation() {
    // top_k=2 should limit candidates to the top 2.
    float logits[] = {1.0f, 5.0f, 3.0f, 0.0f, 0.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.top_k = 2;
    std::vector<int32_t> history;
    std::mt19937_64 rng(42);

    int counts[5] = {};
    for (int i = 0; i < 500; i++) {
        int tok = sample_logits(logits, 5, cfg, history, rng);
        counts[tok]++;
    }
    // Only tokens 1 (logit=5) and 2 (logit=3) should appear
    TEST_ASSERT(counts[0] == 0);
    TEST_ASSERT(counts[3] == 0);
    TEST_ASSERT(counts[4] == 0);
    TEST_ASSERT(counts[1] > 0);
    TEST_ASSERT(counts[2] > 0);
}

static void test_sampler_repetition_penalty() {
    // Multiplicative rep_pen should reduce probability of repeated tokens.
    float logits[] = {3.0f, 3.0f, 3.0f, 3.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.rep_pen = 2.0f;
    std::vector<int32_t> history = {0, 1};  // tokens 0 and 1 in history
    std::mt19937_64 rng(42);

    int counts[4] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 4, cfg, history, rng);
        counts[tok]++;
    }
    // Tokens 0,1 are penalized → tokens 2,3 should appear more
    TEST_ASSERT(counts[2] + counts[3] > counts[0] + counts[1]);
}

static void test_sampler_frequency_penalty() {
    // freq_pen subtracts freq_pen * count(token) from logits.
    // Token 0 appears 5 times → logit reduced by 5*1.0 = 5.0
    float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.freq_pen = 1.0f;
    std::vector<int32_t> history = {0, 0, 0, 0, 0, 1};  // token 0 x5, token 1 x1
    std::mt19937_64 rng(42);

    int counts[4] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 4, cfg, history, rng);
        counts[tok]++;
    }
    // Token 0 penalized most (5*1.0=5), token 1 penalized some (1*1.0=1).
    // Tokens 2,3 unpenalized → should dominate.
    TEST_ASSERT(counts[2] + counts[3] > counts[0] + counts[1]);
    // Token 0 should appear less than token 1 (penalized more).
    TEST_ASSERT(counts[0] < counts[1]);
}

static void test_sampler_presence_penalty() {
    // pres_pen subtracts pres_pen * 1(appeared) from logits.
    float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.pres_pen = 3.0f;
    std::vector<int32_t> history = {0, 1};  // tokens 0,1 appeared
    std::mt19937_64 rng(42);

    int counts[4] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 4, cfg, history, rng);
        counts[tok]++;
    }
    // Tokens 0,1 penalized (logit 5-3=2), tokens 2,3 unpenalized (logit 5).
    TEST_ASSERT(counts[2] + counts[3] > counts[0] + counts[1]);
}

static void test_sampler_freq_and_pres_combined() {
    // Both penalties applied together.
    float logits[] = {5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.freq_pen = 0.5f;
    cfg.pres_pen = 1.0f;
    // Token 0 appears 4 times: penalty = 0.5*4 + 1.0 = 3.0 → logit=2.0
    // Token 1 appears 1 time:  penalty = 0.5*1 + 1.0 = 1.5 → logit=3.5
    // Token 2 never appeared:  penalty = 0                   → logit=5.0
    std::vector<int32_t> history = {0, 0, 0, 0, 1};
    std::mt19937_64 rng(42);

    int counts[3] = {};
    for (int i = 0; i < 3000; i++) {
        int tok = sample_logits(logits, 3, cfg, history, rng);
        counts[tok]++;
    }
    // Token 2 should appear most, token 0 least.
    TEST_ASSERT(counts[2] > counts[1]);
    TEST_ASSERT(counts[1] > counts[0]);
}

static void test_sampler_negative_frequency_penalty() {
    // Negative freq_pen should encourage repetition.
    float logits[] = {3.0f, 3.0f, 3.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.freq_pen = -2.0f;
    std::vector<int32_t> history = {0, 0, 0};  // token 0 appears 3x
    std::mt19937_64 rng(42);

    int counts[3] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 3, cfg, history, rng);
        counts[tok]++;
    }
    // Token 0 logit boosted by 6.0 (3*2.0) → should dominate.
    TEST_ASSERT(counts[0] > counts[1]);
    TEST_ASSERT(counts[0] > counts[2]);
}

static void test_sampler_seed_reproducibility() {
    // Same seed should produce identical sequences.
    float logits[] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    std::vector<int32_t> history;

    std::mt19937_64 rng1(12345);
    std::mt19937_64 rng2(12345);

    for (int i = 0; i < 50; i++) {
        int t1 = sample_logits(logits, 5, cfg, history, rng1);
        int t2 = sample_logits(logits, 5, cfg, history, rng2);
        TEST_ASSERT(t1 == t2);
    }
}

static void test_sampler_rep_window_limits_scope() {
    // With rep_window=2, only the last 2 history tokens should be penalized.
    float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    cfg.pres_pen = 5.0f;
    cfg.rep_window = 2;
    // History: [0, 1, 2, 3] but window=2 → only tokens 2,3 penalized.
    std::vector<int32_t> history = {0, 1, 2, 3};
    std::mt19937_64 rng(42);

    int counts[4] = {};
    for (int i = 0; i < 2000; i++) {
        int tok = sample_logits(logits, 4, cfg, history, rng);
        counts[tok]++;
    }
    // Tokens 0,1 should appear much more than 2,3 (which are in-window).
    TEST_ASSERT(counts[0] + counts[1] > counts[2] + counts[3]);
}

static void test_parse_sampler_token_basic() {
    std::string line = "gen 128 samp=0.7,0.9,40,1.1,42";
    SamplerCfg cfg;
    TEST_ASSERT(parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 128");
    TEST_ASSERT(std::abs(cfg.temp - 0.7f) < 1e-5f);
    TEST_ASSERT(std::abs(cfg.top_p - 0.9f) < 1e-5f);
    TEST_ASSERT(cfg.top_k == 40);
    TEST_ASSERT(std::abs(cfg.rep_pen - 1.1f) < 1e-5f);
    TEST_ASSERT(cfg.seed == 42);
    TEST_ASSERT(cfg.freq_pen == 0.0f);  // not specified → default
    TEST_ASSERT(cfg.pres_pen == 0.0f);
}

static void test_parse_sampler_token_with_penalties() {
    std::string line = "gen 64 samp=0.5,0.95,20,1.0,0,0.8,1.2";
    SamplerCfg cfg;
    TEST_ASSERT(parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 64");
    TEST_ASSERT(std::abs(cfg.temp - 0.5f) < 1e-5f);
    TEST_ASSERT(std::abs(cfg.top_p - 0.95f) < 1e-5f);
    TEST_ASSERT(cfg.top_k == 20);
    TEST_ASSERT(std::abs(cfg.rep_pen - 1.0f) < 1e-5f);
    TEST_ASSERT(cfg.seed == 0);
    TEST_ASSERT(std::abs(cfg.freq_pen - 0.8f) < 1e-5f);
    TEST_ASSERT(std::abs(cfg.pres_pen - 1.2f) < 1e-5f);
}

static void test_parse_sampler_token_minimal() {
    // Only temp specified.
    std::string line = "gen 32 samp=0.3";
    SamplerCfg cfg;
    TEST_ASSERT(parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 32");
    TEST_ASSERT(std::abs(cfg.temp - 0.3f) < 1e-5f);
    TEST_ASSERT(cfg.top_p == 1.0f);  // default
    TEST_ASSERT(cfg.top_k == 0);
    TEST_ASSERT(cfg.freq_pen == 0.0f);
    TEST_ASSERT(cfg.pres_pen == 0.0f);
}

static void test_parse_sampler_token_no_samp() {
    std::string line = "gen 128";
    SamplerCfg cfg;
    TEST_ASSERT(!parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 128");  // unchanged
}

static void test_sampler_temp_zero_with_penalties_uses_argmax() {
    // temp=0 + penalties should apply penalties then return argmax (deterministic).
    float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
    SamplerCfg cfg;
    cfg.temp = 0.0f;
    cfg.pres_pen = 3.0f;
    std::vector<int32_t> history = {0, 1};  // penalize tokens 0,1
    std::mt19937_64 rng(42);

    // Tokens 0,1 have logit 5-3=2; tokens 2,3 have logit 5 (unpenalized).
    // Argmax should always return 2 or 3 (whichever sorts first = stable).
    int tok = sample_logits(logits, 4, cfg, history, rng);
    TEST_ASSERT(tok == 2 || tok == 3);

    // Must be deterministic: same result every time.
    for (int i = 0; i < 10; i++) {
        int t = sample_logits(logits, 4, cfg, history, rng);
        TEST_ASSERT(t == tok);
    }
}

static void test_sampler_needs_logit_processing() {
    SamplerCfg cfg;
    TEST_ASSERT(!cfg.needs_logit_processing());  // all defaults → no processing

    cfg.temp = 0.5f;
    TEST_ASSERT(cfg.needs_logit_processing());

    cfg.temp = 0.0f;
    cfg.freq_pen = 1.0f;
    TEST_ASSERT(cfg.needs_logit_processing());

    cfg.freq_pen = 0.0f;
    cfg.pres_pen = 0.5f;
    TEST_ASSERT(cfg.needs_logit_processing());

    cfg.pres_pen = 0.0f;
    cfg.rep_pen = 1.5f;
    TEST_ASSERT(cfg.needs_logit_processing());

    cfg.rep_pen = 1.0f;
    TEST_ASSERT(!cfg.needs_logit_processing());
}

// ═══════════════════════════════════════════════════════════════════════
// /props body shape tests (model-free)
//
// Verify build_props_body's new wholesale-sidecar `model_card` + new
// `budget_envelope` section per docs/specs/props-endpoint.md §4.9 / §4.X.
// ═══════════════════════════════════════════════════════════════════════

static ServerConfig make_props_config_with_sidecar(const json & sidecar) {
    ServerConfig cfg;
    cfg.arch                    = "qwen35";
    cfg.model_path              = "/tmp/fake/model.gguf";
    cfg.model_card_source_label = "share/model_cards/qwen3.6-27b.json";
    cfg.model_card_json         = sidecar;
    cfg.default_max_tokens      = 32768;
    cfg.hard_limit_reply_budget = 512;
    cfg.think_max_tokens        = 32256;
    cfg.effort_tiers.low    = 4032;
    cfg.effort_tiers.medium = 16128;
    cfg.effort_tiers.high   = 32256;
    cfg.effort_tiers.x_high = 56832;
    cfg.effort_tiers.max    = 81408;
    return cfg;
}

static void test_props_model_card_wholesale_sidecar() {
    // When a sidecar was loaded, /props.model_card should be the parsed
    // sidecar JSON verbatim — *all* fields from the file, not just the
    // five budget-derived ones from the pre-refactor shape.
    json sidecar = {
        {"name",         "Qwen3.6 27B"},
        {"source",       "https://huggingface.co/Qwen/Qwen3.6-27B"},
        {"verified_at", "2026-05-23"},
        {"max_tokens",   32768},
        {"complex_problem_max_tokens", 81920},
        {"sampling", {
            {"temperature", 1.0},
            {"top_p",       0.95},
            {"top_k",       20},
        }},
        {"reasoning_effort_tiers", {
            {"low",    4032},
            {"medium", 16128},
            {"high",   32256},
            {"x-high", 56832},
            {"max",    81408},
        }},
        {"notes", "test card"},
    };
    ServerConfig cfg = make_props_config_with_sidecar(sidecar);
    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body.contains("model_card"));
    TEST_ASSERT(!body["model_card"].is_null());
    // `source` is the upstream URL, NOT the filepath. The filepath label
    // moved to budget_envelope.model_card_source post-refactor.
    TEST_ASSERT(body["model_card"]["source"].get<std::string>() ==
                "https://huggingface.co/Qwen/Qwen3.6-27B");
    TEST_ASSERT(body["model_card"]["name"].get<std::string>() == "Qwen3.6 27B");
    TEST_ASSERT(body["model_card"]["max_tokens"].get<int>() == 32768);
    TEST_ASSERT(body["model_card"]["complex_problem_max_tokens"].get<int>() == 81920);
    TEST_ASSERT(body["model_card"].contains("sampling"));
    TEST_ASSERT(body["model_card"].contains("reasoning_effort_tiers"));
    TEST_ASSERT(body["model_card"]["notes"].get<std::string>() == "test card");
    // The pre-refactor `think_max_tokens` / `hard_limit_reply_budget`
    // keys are NOT in the wholesale shape — they moved to budget_envelope.
    TEST_ASSERT(!body["model_card"].contains("think_max_tokens"));
    TEST_ASSERT(!body["model_card"].contains("hard_limit_reply_budget"));
}

static void test_props_model_card_null_on_family_fallback() {
    // When family or hard fallback was used (no sidecar), /props.model_card
    // is JSON null. The budget_envelope still carries the resolved values.
    ServerConfig cfg;
    cfg.arch                    = "qwen35";
    cfg.model_card_source_label = "family:qwen35";
    cfg.model_card_json         = nullptr;  // no sidecar parsed
    cfg.default_max_tokens      = 32768;
    cfg.hard_limit_reply_budget = 512;
    cfg.think_max_tokens        = 32256;
    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body.contains("model_card"));
    TEST_ASSERT(body["model_card"].is_null());
    // budget_envelope still present and carries the family-fallback label.
    TEST_ASSERT(body.contains("budget_envelope"));
    TEST_ASSERT(body["budget_envelope"]["model_card_source"].get<std::string>() ==
                "family:qwen35");
    TEST_ASSERT(body["budget_envelope"]["default_max_tokens"].get<int>() == 32768);
}

static void test_props_budget_envelope_shape() {
    // budget_envelope is always present with all five fields and the
    // expected effort_tiers vocabulary (low|medium|high|x-high|max).
    // Values mirror ServerConfig regardless of what the sidecar carried.
    json sidecar = {
        {"name",        "Qwen3.6 27B"},
        {"source",      "https://huggingface.co/Qwen/Qwen3.6-27B"},
        {"verified_at", "2026-05-23"},
        {"max_tokens",  32768},
    };
    ServerConfig cfg = make_props_config_with_sidecar(sidecar);
    // Simulate CLI override: budget_envelope reflects the runtime value,
    // which may diverge from the sidecar (here, 16000 != sidecar 32768).
    cfg.default_max_tokens      = 16000;
    cfg.hard_limit_reply_budget = 512;
    cfg.think_max_tokens        = 15488;
    cfg.effort_tiers.low    = 100;
    cfg.effort_tiers.medium = 200;
    cfg.effort_tiers.high   = 300;
    cfg.effort_tiers.x_high = 400;
    cfg.effort_tiers.max    = 500;

    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body.contains("budget_envelope"));
    const json & be = body["budget_envelope"];
    TEST_ASSERT(be["model_card_source"].get<std::string>() ==
                "share/model_cards/qwen3.6-27b.json");
    TEST_ASSERT(be["default_max_tokens"].get<int>() == 16000);
    TEST_ASSERT(be["hard_limit_reply_budget"].get<int>() == 512);
    TEST_ASSERT(be["think_max_tokens"].get<int>() == 15488);
    TEST_ASSERT(be["effort_tiers"]["low"].get<int>()    == 100);
    TEST_ASSERT(be["effort_tiers"]["medium"].get<int>() == 200);
    TEST_ASSERT(be["effort_tiers"]["high"].get<int>()   == 300);
    TEST_ASSERT(be["effort_tiers"]["x-high"].get<int>() == 400);
    TEST_ASSERT(be["effort_tiers"]["max"].get<int>()    == 500);

    // Sanity: budget_envelope can diverge from model_card.max_tokens
    // (CLI override case). Verifies the two sections aren't a tautology.
    TEST_ASSERT(body["model_card"]["max_tokens"].get<int>() == 32768);
    TEST_ASSERT(be["default_max_tokens"].get<int>() == 16000);

    // Sanity: props_schema bumped to 2 (breaking change).
    TEST_ASSERT(body["server"]["props_schema"].get<int>() == 2);
}

// ─── /props.runtime captures full config (§4.16) ──────────────────────
// Snapshot/bench tooling reads /props.runtime wholesale into
// result.json.server_info; this test pins the field set so additions
// elsewhere don't accidentally drop a knob we depend on for forensics.
static void test_props_runtime_shape() {
    ServerConfig cfg = make_props_config_with_sidecar(json{
        {"name", "Qwen3.6 27B"},
        {"source", "https://huggingface.co/Qwen/Qwen3.6-27B"},
        {"verified_at", "2026-05-23"},
        {"max_tokens", 32768},
    });
    cfg.runtime_backend = "cuda";
    cfg.fa_window       = 2048;
    cfg.kv_cache_k      = "tq3_0";
    cfg.kv_cache_v      = "tq3_0";
    cfg.lazy_draft      = false;
    cfg.target_sharding = false;
    cfg.chunk           = 512;
    cfg.target_device   = "auto:0";
    cfg.draft_device    = "auto:0";

    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body.contains("runtime"));
    const json & rt = body["runtime"];
    TEST_ASSERT(rt["backend"].get<std::string>()         == "cuda");
    TEST_ASSERT(rt["fa_window"].get<int>()               == 2048);
    TEST_ASSERT(rt["kv_cache_k"].get<std::string>()      == "tq3_0");
    TEST_ASSERT(rt["kv_cache_v"].get<std::string>()      == "tq3_0");
    TEST_ASSERT(rt["lazy_draft"].get<bool>()             == false);
    TEST_ASSERT(rt["target_sharding"].get<bool>()        == false);
    TEST_ASSERT(rt["chunk"].get<int>()                   == 512);
    TEST_ASSERT(rt["target_device"].get<std::string>()   == "auto:0");
    TEST_ASSERT(rt["draft_device"].get<std::string>()    == "auto:0");

    // draft_device is null when no draft model is loaded.
    cfg.draft_device.clear();
    body = build_props_body(cfg, pc, tm);
    TEST_ASSERT(body["runtime"]["draft_device"].is_null());
}

// ═══════════════════════════════════════════════════════════════════════
// usage.timings — per-request prefill / decode wall-clock breakdown
// surfaced under usage.timings (spec §6.3). Tests cover all three
// response shapes plus the zero-decode_s div-by-zero guard.
// ═══════════════════════════════════════════════════════════════════════

static void test_usage_timings_openai_chat_streaming() {
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

static void test_usage_timings_anthropic_streaming() {
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

static void test_usage_timings_responses_streaming() {
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

static void test_usage_timings_zero_decode_no_div_by_zero() {
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

static void test_usage_timings_omitted_when_null() {
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

int main() {
    std::fprintf(stderr, "══════════════════════════════════════════\n");
    std::fprintf(stderr, " Server Unit Tests\n");
    std::fprintf(stderr, "══════════════════════════════════════════\n");

    std::fprintf(stderr, "\n── UTF-8 utilities ──\n");
    RUN_TEST(test_utf8_safe_len_ascii);
    RUN_TEST(test_utf8_safe_len_partial_2byte);
    RUN_TEST(test_utf8_safe_len_partial_3byte);
    RUN_TEST(test_utf8_safe_len_partial_4byte);
    RUN_TEST(test_utf8_sanitize_valid);
    RUN_TEST(test_utf8_sanitize_replaces_invalid);
    RUN_TEST(test_utf8_sanitize_empty);

    std::fprintf(stderr, "\n── Reasoning parser ──\n");
    RUN_TEST(test_reasoning_basic);
    RUN_TEST(test_reasoning_no_tags);
    RUN_TEST(test_reasoning_started_in_thinking);
    RUN_TEST(test_reasoning_unclosed_think);
    RUN_TEST(test_reasoning_empty_thinking);
    RUN_TEST(test_reasoning_whitespace_in_think);
    RUN_TEST(test_reasoning_disabled);

    std::fprintf(stderr, "\n── Tool parser ──\n");
    RUN_TEST(test_parse_tool_call_xml);
    RUN_TEST(test_parse_bare_function_xml);
    RUN_TEST(test_parse_json_tool_call);
    RUN_TEST(test_parse_no_tools);
    RUN_TEST(test_parse_tool_code_wrapper);
    RUN_TEST(test_parse_tool_allowed_filter);

    std::fprintf(stderr, "\n── SSE Emitter ──\n");
    RUN_TEST(test_emitter_reasoning_split_openai);
    RUN_TEST(test_emitter_first_content_index_natural_close);
    RUN_TEST(test_emitter_first_content_index_never_closed);
    RUN_TEST(test_emitter_first_content_index_content_only);
    RUN_TEST(test_emitter_first_content_index_qwen36_streaming_thinking);
    RUN_TEST(test_emitter_reasoning_strips_leading_think_tag);
    RUN_TEST(test_emitter_content_only_no_thinking);
    RUN_TEST(test_emitter_tool_buffer_detection);
    RUN_TEST(test_emitter_anthropic_tool_use_blocks);
    RUN_TEST(test_emitter_bare_function_tool_buffer_detection);
    RUN_TEST(test_emitter_does_not_leak_malformed_tool_xml);
    RUN_TEST(test_emitter_parses_tool_call_missing_outer_close);
    RUN_TEST(test_emitter_no_tools_keeps_tool_like_text);
    RUN_TEST(test_emitter_anthropic_structure);
    RUN_TEST(test_emitter_responses_structure);
    RUN_TEST(test_emitter_responses_bare_function_tool_call);
    RUN_TEST(test_emitter_streaming_openai_has_done);
    RUN_TEST(test_emitter_nonstreaming_accumulates);
    RUN_TEST(test_emitter_anthropic_thinking_blocks);

    std::fprintf(stderr, "\n── Stop sequences ──\n");
    RUN_TEST(test_stop_sequence_basic);
    RUN_TEST(test_stop_sequence_mid_token);
    RUN_TEST(test_stop_sequence_multiple);
    RUN_TEST(test_stop_sequence_no_match);
    RUN_TEST(test_stop_sequence_empty_list);
    RUN_TEST(test_stop_sequence_finish_reason);
    RUN_TEST(test_stop_sequence_streaming_output);
    RUN_TEST(test_stop_sequence_anthropic_format);
    RUN_TEST(test_stop_sequence_in_reasoning_mode);
    RUN_TEST(test_stop_sequence_holdback_extends);

    std::fprintf(stderr, "\n── Prefix cache (hash) ──\n");
    RUN_TEST(test_hash_prefix_deterministic);
    RUN_TEST(test_hash_prefix_different_inputs);
    RUN_TEST(test_hash_prefix_different_lengths);
    RUN_TEST(test_hash_prefix_empty);
    RUN_TEST(test_find_boundaries_empty);

    std::fprintf(stderr, "\n── PFlash config ──\n");
    RUN_TEST(test_pflash_config_defaults);
    RUN_TEST(test_pflash_config_modes);
    RUN_TEST(test_pflash_compress_request_struct);
    RUN_TEST(test_pflash_compress_result_defaults);
    RUN_TEST(test_pflash_threshold_auto_mode);
    RUN_TEST(test_pflash_threshold_always_mode);
    RUN_TEST(test_pflash_placement_same_backend_local);
    RUN_TEST(test_pflash_placement_mixed_backend_remote);
    RUN_TEST(test_pflash_placement_auto_draft_follows_target);
    RUN_TEST(test_pflash_placement_disabled_never_remote);
    RUN_TEST(test_pflash_placement_usage_gate);

    std::fprintf(stderr, "\n── Jinja chat template ──\n");
    RUN_TEST(test_jinja_render_basic);
    RUN_TEST(test_jinja_render_no_gen_prompt);
    RUN_TEST(test_jinja_render_tools_injected);
    RUN_TEST(test_jinja_render_empty_tools_skipped);
    RUN_TEST(test_jinja_render_bos_eos_threaded);
    RUN_TEST(test_jinja_render_empty_template_throws);
    RUN_TEST(test_jinja_render_bad_tools_json_throws);
    RUN_TEST(test_normalize_responses_tool_followup_messages);

    std::fprintf(stderr, "\n── Disk prefix cache ──\n");
    RUN_TEST(test_disk_cache_config_defaults);
    RUN_TEST(test_disk_cache_disabled_when_no_dir);
    RUN_TEST(test_disk_cache_init_creates_directory);
    RUN_TEST(test_disk_cache_header_size);
    RUN_TEST(test_disk_cache_header_round_trip);
    RUN_TEST(test_disk_cache_continued_boundary);
    RUN_TEST(test_disk_cache_continued_interval_logic);
    RUN_TEST(test_disk_cache_cold_prefix_short_prompt);
    RUN_TEST(test_disk_cache_cold_prefix_no_boundaries);
    RUN_TEST(test_disk_cache_cold_prefix_finds_boundary);
    RUN_TEST(test_disk_cache_budget_enforcement_scoring);
    RUN_TEST(test_disk_cache_lookup_miss_no_layout);
    RUN_TEST(test_disk_cache_save_below_min_tokens);
    RUN_TEST(test_backend_ipc_rejects_file_work_dir);

    std::fprintf(stderr, "\n── Sampler ──\n");
    RUN_TEST(test_sampler_cfg_defaults);
    RUN_TEST(test_sampler_greedy_argmax);
    RUN_TEST(test_sampler_temperature_affects_distribution);
    RUN_TEST(test_sampler_top_p_truncation);
    RUN_TEST(test_sampler_top_k_truncation);
    RUN_TEST(test_sampler_repetition_penalty);
    RUN_TEST(test_sampler_frequency_penalty);
    RUN_TEST(test_sampler_presence_penalty);
    RUN_TEST(test_sampler_freq_and_pres_combined);
    RUN_TEST(test_sampler_negative_frequency_penalty);
    RUN_TEST(test_sampler_seed_reproducibility);
    RUN_TEST(test_sampler_rep_window_limits_scope);
    RUN_TEST(test_parse_sampler_token_basic);
    RUN_TEST(test_parse_sampler_token_with_penalties);
    RUN_TEST(test_parse_sampler_token_minimal);
    RUN_TEST(test_parse_sampler_token_no_samp);
    RUN_TEST(test_sampler_temp_zero_with_penalties_uses_argmax);
    RUN_TEST(test_sampler_needs_logit_processing);

    std::fprintf(stderr, "\n── /props body shape ──\n");
    RUN_TEST(test_props_model_card_wholesale_sidecar);
    RUN_TEST(test_props_model_card_null_on_family_fallback);
    RUN_TEST(test_props_budget_envelope_shape);
    RUN_TEST(test_props_runtime_shape);

    std::fprintf(stderr, "\n── usage.timings ──\n");
    RUN_TEST(test_usage_timings_openai_chat_streaming);
    RUN_TEST(test_usage_timings_anthropic_streaming);
    RUN_TEST(test_usage_timings_responses_streaming);
    RUN_TEST(test_usage_timings_zero_decode_no_div_by_zero);
    RUN_TEST(test_usage_timings_omitted_when_null);

    std::fprintf(stderr, "\n══════════════════════════════════════════\n");
    std::fprintf(stderr, " Results: %d assertions, %d failures\n",
                 test_count, test_failures);
    std::fprintf(stderr, "══════════════════════════════════════════\n");

    if (test_failures) {
        std::fprintf(stderr, "FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "ALL PASSED\n");
    return 0;
}
