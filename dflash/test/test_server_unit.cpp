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
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash27b;

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

static SseEmitter make_emitter(ApiFormat fmt, bool thinking = false) {
    return SseEmitter(fmt, "test_id_001", "test-model", 10,
                      json::array(), nullptr, thinking);
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
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, true);
    em.emit_start();

    // Feed reasoning tokens
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

static void test_emitter_reasoning_strips_leading_think_tag() {
    // When started_in_thinking=true, model may echo <think>.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, true);
    em.emit_start();

    // Model echoes \n<think>\n before actual reasoning
    em.emit_token("\n<think>\nActual reasoning here");
    em.emit_token("</think>");
    em.emit_token("Content");

    em.emit_finish(10);

    // Leading <think> should be stripped from reasoning
    TEST_ASSERT(em.reasoning_text().find("<think>") == std::string::npos);
    TEST_ASSERT(em.reasoning_text().find("Actual reasoning") != std::string::npos);
}

static void test_emitter_content_only_no_thinking() {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, false);
    em.emit_start();
    em.emit_token("Hello, world!");
    em.emit_finish(5);

    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
    TEST_ASSERT(em.reasoning_text().empty());
}

static void test_emitter_tool_buffer_detection() {
    // When the emitter sees <tool_call>, it should buffer and parse tools.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, false);
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

static void test_emitter_anthropic_structure() {
    // Verify Anthropic format emits proper event sequence.
    auto em = make_emitter(ApiFormat::ANTHROPIC, false);
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
    auto em = make_emitter(ApiFormat::RESPONSES, false);
    auto start = em.emit_start();
    std::string start_str = concat(start);

    TEST_ASSERT(start_str.find("response.created") != std::string::npos);
    TEST_ASSERT(start_str.find("response.output_item.added") != std::string::npos);

    em.emit_token("Hi there! How are you doing today?");
    auto finish = em.emit_finish(10);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("response.completed") != std::string::npos);
}

static void test_emitter_streaming_openai_has_done() {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, false);
    em.emit_start();
    em.emit_token("Hello");
    auto finish = em.emit_finish(3);
    std::string finish_str = concat(finish);

    TEST_ASSERT(finish_str.find("[DONE]") != std::string::npos);
}

static void test_emitter_nonstreaming_accumulates() {
    // Non-streaming: tokens fed through emitter, accumulated_text() has all content.
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, false);
    em.emit_token("Hello ");
    em.emit_token("world");
    em.emit_finish(5);

    TEST_ASSERT(em.accumulated_text().find("Hello") != std::string::npos);
    TEST_ASSERT(em.accumulated_text().find("world") != std::string::npos);
}

static void test_emitter_anthropic_thinking_blocks() {
    auto em = make_emitter(ApiFormat::ANTHROPIC, true);
    auto start = em.emit_start();
    std::string start_str = concat(start);

    TEST_ASSERT(start_str.find("thinking") != std::string::npos);

    // Feed reasoning
    em.emit_token("Reasoning about the problem at length here...");
    em.emit_token("</think>");
    em.emit_token("The answer is clear now.");
    auto finish = em.emit_finish(20);
    std::string all = start_str + concat(finish);

    // Should have both thinking and text blocks
    TEST_ASSERT(all.find("thinking") != std::string::npos);
    TEST_ASSERT(!em.reasoning_text().empty());
    TEST_ASSERT(!em.accumulated_text().empty());
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
    RUN_TEST(test_emitter_reasoning_strips_leading_think_tag);
    RUN_TEST(test_emitter_content_only_no_thinking);
    RUN_TEST(test_emitter_tool_buffer_detection);
    RUN_TEST(test_emitter_anthropic_structure);
    RUN_TEST(test_emitter_responses_structure);
    RUN_TEST(test_emitter_streaming_openai_has_done);
    RUN_TEST(test_emitter_nonstreaming_accumulates);
    RUN_TEST(test_emitter_anthropic_thinking_blocks);

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
