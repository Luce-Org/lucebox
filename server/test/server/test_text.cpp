#include "server/utf8_utils.h"
#include "server/reasoning.h"
#include "support/streaming.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <string>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

// ═══════════════════════════════════════════════════════════════════════
// UTF-8 utility tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_utf8_safe_len_ascii) {
    std::string s = "Hello, world!";
    TEST_ASSERT(utf8_safe_len(s, s.size()) == s.size());
    TEST_ASSERT(utf8_safe_len(s, 5) == 5);
    TEST_ASSERT(utf8_safe_len(s, 0) == 0);
}

TEST_CASE(ServerUnitFixture, test_utf8_safe_len_partial_2byte) {
    // é = 0xC3 0xA9
    std::string s = "caf\xC3\xA9!";  // "café!"
    TEST_ASSERT(utf8_safe_len(s, 5) == 5);  // after é, ok
    TEST_ASSERT(utf8_safe_len(s, 4) == 3);  // mid-é, snap back to before é
}

TEST_CASE(ServerUnitFixture, test_utf8_safe_len_partial_3byte) {
    // ん = 0xE3 0x82 0x93
    std::string s = "A\xE3\x82\x93Z";  // "AんZ"
    TEST_ASSERT(utf8_safe_len(s, 4) == 4);  // after ん
    TEST_ASSERT(utf8_safe_len(s, 3) == 1);  // mid-ん, snap back to A
    TEST_ASSERT(utf8_safe_len(s, 2) == 1);  // mid-ん, snap back to A
}

TEST_CASE(ServerUnitFixture, test_utf8_safe_len_partial_4byte) {
    // 🚩 = 0xF0 0x9F 0x9A 0xA9
    std::string s = "A \xF0\x9F\x9A\xA9 done";
    TEST_ASSERT(utf8_safe_len(s, 6) == 6);  // after 🚩
    // Mid-emoji should snap back to position 2 (before 🚩)
    TEST_ASSERT(utf8_safe_len(s, 5) == 2);
    TEST_ASSERT(utf8_safe_len(s, 4) == 2);
    TEST_ASSERT(utf8_safe_len(s, 3) == 2);
}

TEST_CASE(ServerUnitFixture, test_utf8_sanitize_valid) {
    std::string s = "Hello, world! 🎉";
    TEST_ASSERT(utf8_sanitize(s) == s);
}

TEST_CASE(ServerUnitFixture, test_utf8_sanitize_replaces_invalid) {
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

TEST_CASE(ServerUnitFixture, test_utf8_sanitize_empty) {
    TEST_ASSERT(utf8_sanitize("") == "");
}

// ═══════════════════════════════════════════════════════════════════════
// Reasoning parser tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_reasoning_basic) {
    auto r = parse_reasoning("<think>I need to think</think>The answer is 42");
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "I need to think");
    TEST_ASSERT(r.content == "The answer is 42");
}

TEST_CASE(ServerUnitFixture, test_reasoning_no_tags) {
    auto r = parse_reasoning("Just plain text");
    TEST_ASSERT(!r.has_reasoning);
    TEST_ASSERT(r.content == "Just plain text");
}

TEST_CASE(ServerUnitFixture, test_reasoning_started_in_thinking) {
    auto r = parse_reasoning("thinking body</think>content here",
                             true, true);
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "thinking body");
    TEST_ASSERT(r.content == "content here");
}

TEST_CASE(ServerUnitFixture, test_emitter_started_in_thinking_without_open_tag) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, json::array(), true);
    auto chunks = em.emit_token("Thinking Process: calculate 9 + 6.");
    auto close = em.emit_token("</think>");
    auto answer = em.emit_token("15");
    em.emit_finish(3);

    std::string all = concat(chunks) + concat(close) + concat(answer);
    TEST_ASSERT(em.reasoning_text().find("Thinking Process") != std::string::npos);
    TEST_ASSERT(em.accumulated_text() == "15");
    TEST_ASSERT(em.first_content_token_index() == 2);
    TEST_ASSERT(all.find("reasoning_content") != std::string::npos);
    TEST_ASSERT(all.find("\"content\":\"Thinking Process") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_reasoning_unclosed_think) {
    auto r = parse_reasoning("<think>still thinking no close",
                             true, false);
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "still thinking no close");
    TEST_ASSERT(r.content.empty());
}

TEST_CASE(ServerUnitFixture, test_reasoning_empty_thinking) {
    auto r = parse_reasoning("<think></think>answer");
    TEST_ASSERT(!r.has_reasoning);  // empty reasoning
    TEST_ASSERT(r.content == "answer");
}

TEST_CASE(ServerUnitFixture, test_reasoning_whitespace_in_think) {
    auto r = parse_reasoning("<think>\n  reasoning \n</think>\ncontent");
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "reasoning");
    TEST_ASSERT(r.content == "content");
}

TEST_CASE(ServerUnitFixture, test_reasoning_disabled) {
    // When thinking disabled but tags present, the parser still finds them
    // (the caller decides whether to use the reasoning field).
    auto r = parse_reasoning("<think>ignored</think>content",
                             false, false);
    TEST_ASSERT(r.has_reasoning);
    TEST_ASSERT(r.reasoning == "ignored");
    TEST_ASSERT(r.content == "content");
}

TEST_CASE(ServerUnitFixture, test_escape_for_logging) {
    // Standard escapes
    TEST_ASSERT(escape_for_logging("hello\nworld\r\t'\\") == "hello\\nworld\\r\\t\\'\\\\");
    // NUL byte (escaped as fixed-width \u0000 for consistency)
    TEST_ASSERT(escape_for_logging(std::string("null\0byte", 9)) == "null\\u0000byte");
    // Control byte followed by hex digit must be unambiguous (\u0001f)
    TEST_ASSERT(escape_for_logging(std::string("\x01", 1) + "f") == "\\u0001f");
    TEST_ASSERT(escape_for_logging(std::string("\x1f", 1) + "abc") == "\\u001fabc");
    TEST_ASSERT(escape_for_logging(std::string("\x7f", 1) + "xyz") == "\\u007fxyz");
}
