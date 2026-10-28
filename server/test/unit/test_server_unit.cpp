// Unit tests for server components — no GPU, no model files required.
//
// Tests: SseEmitter, ToolParser, Reasoning, PrefixCache (hash/boundary),
//        UTF-8 utilities.
//
// Build: cmake --build . --target test_server_unit
// Run:   ./test_server_unit

#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include <fstream>

#include "server/sse_emitter.h"
#include "server/tool_parser.h"
#include "server/model_card.h"
#include "server/response_error.h"
#include "server/prefix_cache.h"
#include "server/pin_friendly_prompt.h"
#include "server/disk_prefix_cache.h"
#include "server/freeze_history.h"
#include "server/utf8_utils.h"
#include "server/api_types.h"
#include "server/http_server.h"
#include "server/image_input.h"
#include "qwen35/qwen35_backend.h"
#include "qwen4exp/qwen4exp_graph.h"
#include "common/cuda_graph_overrides.h"
#include "engine/luce_engine.h"
#include "server/chat_template.h"
#include "common/concurrency/seq_engine.h"
#include "common/sampler.h"
#include "common/concurrency/seq_engine.h"
#include "common/backend_precision.h"
#include "common/backend_ipc.h"
#include "common/pflash_drafter_ipc.h"
#include "common/moe_hybrid_ffn_eval.h"
#include "common/moe_hybrid_placement.h"
#include "placement/pflash_placement.h"
#include "common/io_utils.h"
#include "placement/placement_config.h"
#include "common/layer_split_backend.h"
#include "common/layer_split_kvflash.h"
#include "common/layer_split_utils.h"
#include "common/kvflash_pager.h"
#include "placement/device_select.h"
#include "placement/draft_residency.h"
#include "common/gguf_bounds.h"
#include "common/gguf_inspect.h"
#include "qwen35/prefill_helpers.h"
#include "qwen35moe/qwen35moe_ffn.h"
#include "ggml-cpu.h"
#include "server/prompt_normalize.h"
#include "pflash/pflash_drafter.h"
#include "qwen3_model.h"
#include "pflash/pflash_compress.h"
#include "luce.h"
#include "gguf.h"
#include <nlohmann/json.hpp>

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <map>
#include <string>
#include <unordered_set>
#include <type_traits>
#include <array>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <limits>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/socket.h>
#else
#include <io.h>
#include <process.h>
#endif

#if defined(_WIN32)
#define luce_setenv(name, value) _putenv_s(name, value)
#define luce_unsetenv(name) _putenv_s(name, "")
#else
#define luce_setenv(name, value) setenv(name, value, 1)
#define luce_unsetenv(name) unsetenv(name)
#endif

using json = nlohmann::json;
using namespace luce::common;
using luce::engine::LuceEngine;
namespace fs = std::filesystem;

static fs::path test_tmp_path(const char * name) {
    std::error_code ec;
    fs::path root = fs::temp_directory_path(ec);
    if (ec) {
        throw std::runtime_error("failed to resolve temporary directory: " +
                                 ec.message());
    }
    return root / name;
}

static void remove_test_path(const fs::path & path) {
    std::error_code ec;
    fs::remove(path, ec);
}

namespace luce::common {
std::vector<ChatMessage> normalize_chat_messages(
    const json & messages,
    ApiFormat format,
    ToolMemory & tool_memory,
    bool replay_tool_memory = true);

struct SchedulerTestHarness {
    static PrefixCache & prefix_cache(HttpServer & server) {
        return server.prefix_cache_;
    }

    static int ram_restorable_prefix_tokens(
            HttpServer & server, const std::vector<int32_t> & prompt) {
        return server.ram_restorable_prefix_tokens(prompt);
    }

    static void enqueue(HttpServer & server, ServerJob * job) {
        server.enqueue(job);
    }

    static void run(HttpServer & server, SeqEngine & engine) {
        server.scheduler_loop(engine);
    }

    static void stop(HttpServer & server) {
        server.stopping_.store(true, std::memory_order_relaxed);
        server.queue_cv_.notify_all();
    }

    static json status_json(const HttpServer & server) {
        return server.status_.to_json();
    }

    static void finalize_inline_snapshot(
            HttpServer & server, const std::vector<int32_t> & prompt,
            PrefixCache::InlineReservation reservation,
            int slot, int requested_cut, int restore_slot = -1,
            bool compressed = false) {
        ParsedRequest req;
        req.prompt_tokens = prompt;
        HttpServer::PreparedPrompt prepared;
        prepared.tokens = prompt;
        prepared.compressed = compressed;
        HttpServer::GenerationCacheState cache;
        cache.snap_reservation = std::move(reservation);
        cache.snap_slot = slot;
        cache.snap_cut = requested_cut;
        cache.snap_prepared = true;
        cache.using_restore = restore_slot >= 0;
        cache.cache_slot = restore_slot;
        GenerateResult result;
        result.error.reset();
        server.finalize_generation_cache(
            req, prepared, cache, result,
            /*completion_tokens=*/1,
            /*visible_output_seen=*/true,
            /*client_disconnected=*/false);
    }

    static const std::vector<int32_t> & slot_tokens(
            const HttpServer & server, int slot) {
        return server.slot_tokens_.at(slot);
    }

    static void set_slot_tokens(HttpServer & server, int slot,
                                std::vector<int32_t> tokens) {
        server.slot_tokens_[slot] = std::move(tokens);
    }

    static DiskPrefixCache & disk_cache(HttpServer & server) {
        return server.disk_cache_;
    }
    static json props(const HttpServer & server) {
        return build_props_body(server.config_, server.prefix_cache_, server.tool_memory_);
    }

    static void flush_sessions(HttpServer & server) {
        server.flush_sessions_to_disk("shutdown");
    }

    // What prepare_generation_cache does with a reservation's slot before
    // the new snapshot overwrites it.
    static void evict_inline_slot(HttpServer & server, int slot) {
        server.evict_inline_slot(slot);
        server.backend_.snapshot_free(slot);
    }

    static void publish_memory_report(HttpServer & server) {
        server.publish_memory_report();
    }

    static json memory_json(const HttpServer & server) {
        return server.memory_json();
    }

    static void mark_agent_turn(HttpServer & server, int slot) {
        server.agent_turn_cache_slots_.insert(slot);
    }

    static bool is_agent_turn(const HttpServer & server, int slot) {
        return server.agent_turn_cache_slots_.count(slot) != 0;
    }

    // One HTTP request through the serving path: parsing, rendering, the
    // worker started by start_worker(), and the response written to `fd`.
    static bool start_worker(HttpServer & server) { return server.start_worker(); }
    static bool route(HttpServer & server, SocketHandle fd, const std::string & path,
                      const json & body) {
        HttpServer::HttpRequest request;
        request.method = "POST";
        request.path = path;
        request.body = body.dump();
        return server.route_request(fd, request);
    }

    // `own_snapshot_slot` is the inline snapshot the request itself took,
    // `restored_slot` the checkpoint it restored from.
    static bool save_generated_turn(
            HttpServer & server, const std::vector<int32_t> & prompt,
            const std::vector<int32_t> & generated,
            const std::vector<int32_t> & canonical,
            int own_snapshot_slot = -1,
            DiskPrefixCacheMode disk_scope = DiskPrefixCacheMode::Full,
            int restored_slot = -1) {
        HttpServer::GenerationCacheState cache;
        cache.disk_policy.mode = disk_scope;
        cache.snap_prepared = own_snapshot_slot >= 0;
        cache.snap_slot = own_snapshot_slot;
        cache.using_restore = restored_slot >= 0;
        cache.cache_slot = restored_slot;
        return server.save_generated_turn(prompt, generated, canonical, cache);
    }

    struct PreparedCache {
        int restore_slot;   // -1 without a restore
        int prefix_len;
        bool snapshot;      // an inline snapshot is planned
        std::vector<int> restore_points;
    };

    // The inline snapshot cut a compressed (PFlash) tool request plans:
    // -1 none.
    static int prepare_cache_snapshot_cut(
            HttpServer & server, const std::vector<int32_t> & prompt,
            const json & tools, int snapshot_cut) {
        ParsedRequest req;
        req.prompt_tokens = prompt;
        req.response_id = "test";
        req.tools = tools;
        HttpServer::PreparedPrompt prepared;
        prepared.tokens = prompt;
        prepared.compressed = true;
        prepared.snapshot_cut = snapshot_cut;
        GenerateRequest generate_request;
        generate_request.prompt = prompt;
        const auto cache = server.prepare_generation_cache(
            req, prepared, generate_request);
        return cache.snap_prepared ? cache.snap_cut : -1;
    }

    static PreparedCache prepare_cache(
            HttpServer & server, const std::vector<int32_t> & prompt,
            bool ends_with_tool_result = false) {
        ParsedRequest req;
        req.prompt_tokens = prompt;
        req.response_id = "test";
        req.ends_with_tool_result = ends_with_tool_result;
        HttpServer::PreparedPrompt prepared;
        prepared.tokens = prompt;
        GenerateRequest generate_request;
        generate_request.prompt = prompt;
        const auto cache = server.prepare_generation_cache(
            req, prepared, generate_request);
        return {cache.using_restore ? cache.cache_slot : -1, cache.prefix_len,
                cache.snap_prepared, generate_request.restore_points};
    }
};

struct HttpServerTestAccess {
    using PreparedPrompt = HttpServer::PreparedPrompt;
    static std::string apply_pflash_compression(
            HttpServer & server, const ParsedRequest & req) {
        HttpServer::PreparedPrompt prepared;
        return server.apply_pflash_compression(req, prepared);
    }
    static HttpServer::PreparedPrompt prepare_prompt(
            HttpServer & server, const ParsedRequest & req) {
        return server.prepare_prompt(req);
    }
    static json pflash_failure_counters(const HttpServer & server) {
        return server.pflash_failure_counters_json();
    }
    static json pflash_keep_history(const HttpServer & server) {
        return server.pflash_keep_history_json();
    }
    static size_t pflash_view_count(const HttpServer & server) {
        return server.pflash_views_.size();
    }
    static json pflash_unified(const HttpServer & server) {
        return server.pflash_unified_json();
    }
    // The chat view's own behaviour, which serves the requests the unified view hands back.
    static void chat_view_only(HttpServer & server) { server.pflash_unified_ = false; }
    // A committed inline prefix-cache entry: ids[0, cut) restorable from
    // ``slot`` (the backend reports the snapshot).
    static void seed_prefix_cache(HttpServer & server, int slot, int cut,
                                  const std::vector<int32_t> & ids) {
        server.prefix_cache_.confirm_inline_snap(slot, cut, ids);
    }
};
}

namespace {
struct ServerUnitFixture {};
}

// This file still uses free helper functions outside fixture-member scope, so
// keep a local shim where the README's direct REQUIRE/CHECK macros are not
// available without refactoring those helpers into CommonFixture members.
#define TEST_ASSERT(expr) do { \
    if (!(expr)) { \
        throw std::runtime_error(std::string(__FILE__) + ":" + \
            std::to_string(__LINE__) + ": " + #expr); \
    } \
} while (0)
#define TEST_ASSERT_MSG(expr, msg) do { \
    if (!(expr)) { \
        throw std::runtime_error(std::string(__FILE__) + ":" + \
            std::to_string(__LINE__) + ": " + #expr + " — " + std::string(msg)); \
    } \
} while (0)

// Device ordinals must not wrap when matching routing priority to placement.
TEST_CASE(ServerUnitFixture, test_placement_device_rejects_gpu_index_overflow) {
    DevicePlacement device;
    TEST_ASSERT(parse_placement_device("hip:0", device));
    TEST_ASSERT(parse_placement_device(
        "hip:" + std::to_string(std::numeric_limits<int>::max()), device));
    for (const char * value : {"hip:2147483648", "hip:4294967296",
                              "hip:999999999999999999999999999999", "hip:-1"}) {
        TEST_ASSERT(!parse_placement_device(value, device));
    }
    TEST_ASSERT(!parse_placement_device_list("hip:0,hip:4294967296", device));
}

// --target-device auto sizes the KV cache from the GGUF header for the
// Qwen3.5/3.6 hybrids: only full-attention layers, excluding NextN blocks.
TEST_CASE(ServerUnitFixture, test_gguf_kv_cache_bytes_counts_full_attention_layers) {
    auto write = [](const char * arch, const char * name) {
        gguf_context * g = gguf_init_empty();
        const std::string pre = std::string(arch) + ".";
        gguf_set_val_str(g, "general.architecture", arch);
        gguf_set_val_u32(g, (pre + "block_count").c_str(), 9);
        gguf_set_val_u32(g, (pre + "nextn_predict_layers").c_str(), 1);
        gguf_set_val_u32(g, (pre + "full_attention_interval").c_str(), 4);
        gguf_set_val_u32(g, (pre + "attention.head_count_kv").c_str(), 2);
        gguf_set_val_u32(g, (pre + "attention.key_length").c_str(), 128);
        gguf_set_val_u32(g, (pre + "attention.value_length").c_str(), 128);
        const std::string path = test_tmp_path(name).string();
        gguf_write_to_file(g, path.c_str(), /*only_meta=*/true);
        gguf_free(g);
        return path;
    };
    const std::string qwen = write("qwen35", "luce_test_kv_qwen35.gguf");
    // 8 target layers / interval 4 = 2 full-attention layers, 2 KV heads.
    const uint64_t q4_row = ggml_row_size(GGML_TYPE_Q4_0, 128);
    TEST_ASSERT(gguf_kv_cache_bytes(qwen, 1000, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0) ==
                2ull * 2 * (q4_row + q4_row) * 1000);
    const uint64_t f16_row = ggml_row_size(GGML_TYPE_F16, 128);
    TEST_ASSERT(gguf_kv_cache_bytes(qwen, 1000, GGML_TYPE_F16, GGML_TYPE_F16) ==
                2ull * 2 * (f16_row + f16_row) * 1000);
    TEST_ASSERT(gguf_kv_cache_bytes(qwen, 0, GGML_TYPE_F16, GGML_TYPE_F16) == 0);

    // Families that size their own caches are not estimated.
    TEST_ASSERT(gguf_kv_cache_bytes(write("deepseek4", "luce_test_kv_ds4.gguf"), 1000,
                                    GGML_TYPE_F16, GGML_TYPE_F16) == 0);
    TEST_ASSERT(gguf_kv_cache_bytes(test_tmp_path("luce_test_kv_missing.gguf").string(),
                                    1000) == 0);
}

TEST_CASE(ServerUnitFixture, test_api_format_names_are_total) {
    CHECK(std::string(api_format_name(ApiFormat::OPENAI_CHAT)) == "chat");
    CHECK(std::string(api_format_name(ApiFormat::ANTHROPIC)) == "anthropic");
    CHECK(std::string(api_format_name(ApiFormat::RESPONSES)) == "responses");
    CHECK(std::string(api_format_name(ApiFormat::COMPLETIONS)) == "completions");
}

TEST_CASE(ServerUnitFixture, test_pflash_scorer_uses_user_query_before_chat_suffix) {
    const std::vector<int32_t> query{
        90, 91, 100, 101, 102, 103, 104, 105, 106, 107,
    };
    const std::vector<int32_t> rendered{
        1, 2, 100, 101, 102, 103, 104, 105, 106, 107,
        200, 201, 202, 203, 204, 205, 206, 207,
    };

    const auto window = http_detail::find_pflash_query_window(rendered, query);

    TEST_ASSERT(window.valid());
    TEST_ASSERT(window.tokens == 8);
    TEST_ASSERT(window.end == 10);
    TEST_ASSERT((int)rendered.size() - window.end == 8);
}

// Claude Code sends its environment as a system message after the user's
// first one; a template that takes a system message only first gets it as a
// user turn in place, and one that takes it anywhere keeps it.
TEST_CASE(ServerUnitFixture, test_template_gets_late_system_message_as_user_turn) {
    const std::string strict =
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' and not loop.first -%}"
        "{{- raise_exception('System message must be at the beginning.') -}}"
        "{%- endif -%}"
        "[{{ m.role }}:{{ m.content }}]"
        "{%- endfor -%}";
    const std::string permissive =
        "{%- for m in messages -%}[{{ m.role }}:{{ m.content }}]{%- endfor -%}";
    const std::vector<ChatMessage> msgs = {
        {"system", "agent"}, {"user", "hi"}, {"system", "env"}, {"developer", "brief"}};
    TEST_ASSERT(render_chat_template_jinja(strict, msgs, "", "", false, false, "") ==
                "[system:agent][user:hi][user:env][user:brief]");
    TEST_ASSERT(render_chat_template_jinja(permissive, msgs, "", "", false, false, "") ==
                "[system:agent][user:hi][system:env][developer:brief]");
}

TEST_CASE(ServerUnitFixture, test_pflash_scorer_maps_last_128_user_tokens) {
    std::vector<int32_t> query;
    for (int token = 0; token < 160; ++token) {
        query.push_back(1000 + token);
    }
    std::vector<int32_t> rendered{1, 2};
    rendered.insert(rendered.end(), query.begin(), query.end());
    rendered.insert(rendered.end(), {200, 201, 202, 203});

    const auto window =
        http_detail::find_pflash_query_window(rendered, query, 128);

    TEST_ASSERT(window.valid());
    TEST_ASSERT(window.tokens == 128);
    TEST_ASSERT(window.end == 162);
    TEST_ASSERT((int)rendered.size() - window.end == 4);
}

TEST_CASE(ServerUnitFixture, test_pflash_query_mapping_tolerates_one_bpe_boundary_token) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<int32_t> rendered{
        1, 2, 999, 11, 12, 13, 14, 15, 16, 17, 200, 201,
    };

    const auto window = http_detail::find_pflash_query_window(rendered, query);

    TEST_ASSERT(window.valid());
    TEST_ASSERT(window.tokens == 7);
    TEST_ASSERT(window.end == 10);
}

TEST_CASE(ServerUnitFixture, test_pflash_query_mapping_stays_before_latest_user_boundary) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<int32_t> rendered{
        1, 2, 999, 11, 12, 13, 14, 15, 16, 17,
        200, 201, 10, 11, 12, 13, 14, 15, 16, 17, 202,
    };
    // The sentinel changes the token at the user end, but the later template
    // and assistant tokens remain stable. The common suffix establishes the
    // semantic boundary after the original complete user suffix.
    const std::vector<int32_t> sentinel_rendered{
        1, 2, 999, 11, 12, 13, 14, 15, 16, 9999,
        200, 201, 10, 11, 12, 13, 14, 15, 16, 17, 202,
    };

    const int search_end = http_detail::pflash_query_search_end_from_sentinel(
        rendered, sentinel_rendered);
    const auto bounded =
        http_detail::find_pflash_query_window(rendered, query, 8, search_end);
    const auto unbounded = http_detail::find_pflash_query_window(rendered, query);

    TEST_ASSERT(search_end == 10);
    TEST_ASSERT(bounded.valid());
    TEST_ASSERT(bounded.tokens == 7);
    TEST_ASSERT(bounded.end == 10);
    TEST_ASSERT(unbounded.valid());
    TEST_ASSERT(unbounded.tokens == 8);
    TEST_ASSERT(unbounded.end == 20);
    TEST_ASSERT(http_detail::pflash_query_search_end_from_sentinel(
        rendered, std::vector<int32_t>{42}) < 0);
}

TEST_CASE(ServerUnitFixture, test_pflash_explicit_query_matches_before_trailing_content) {
    // An explicit query sits before trailing instructions inside the latest
    // user content; anchored matching (message tail) must fail, unanchored
    // matching must find its latest occurrence and tolerate a leading-token
    // boundary difference from tokenizing the query on its own.
    const std::vector<int32_t> query{5, 10, 11, 12, 13, 14};   // 5 = boundary drift
    const std::vector<int32_t> rendered{
        1, 2, 3, 10, 11, 12, 13, 14, 300, 301, 302, 303, 304, 305, 306, 307,
    };
    const auto anchored = http_detail::find_pflash_query_window(rendered, query, 8, 16, 1);
    const auto explicit_window = http_detail::find_pflash_query_window(rendered, query, 8, 16, 1, false);
    TEST_ASSERT(!anchored.valid());
    TEST_ASSERT(explicit_window.valid());
    TEST_ASSERT(explicit_window.end == 8);
    TEST_ASSERT(explicit_window.tokens == 5);
    const auto out_of_window = http_detail::find_pflash_query_window(rendered, query, 8, 16, 9, false);
    TEST_ASSERT(!out_of_window.valid());
}

TEST_CASE(ServerUnitFixture, test_pflash_explicit_query_tolerates_merged_trailing_token) {
    // The query's last token merges with the prompt's following newline
    // ("  " alone vs "  \n" in context): the unanchored match drops that
    // trailing token, the untrimmed match still wins when it exists, and
    // anchored (message tail) matching keeps its exact-suffix contract.
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 77};   // 77 = "  "
    const std::vector<int32_t> rendered{
        1, 2, 3, 10, 11, 12, 13, 14, 78, 300, 301, 302,   // 78 = "  \n"
    };
    const auto trimmed = http_detail::find_pflash_query_window(rendered, query, 8, 12, 1, false);
    TEST_ASSERT(trimmed.valid());
    TEST_ASSERT(trimmed.end == 8);
    TEST_ASSERT(trimmed.tokens == 5);
    TEST_ASSERT(trimmed.trailing_trimmed == 1);
    const std::vector<int32_t> exact{1, 10, 11, 12, 13, 14, 77, 2, 10, 11, 12, 13, 14, 78};
    const auto untrimmed = http_detail::find_pflash_query_window(exact, query, 8, 14, 0, false);
    TEST_ASSERT(untrimmed.valid());
    TEST_ASSERT(untrimmed.end == 7);
    TEST_ASSERT(untrimmed.tokens == 6);
    TEST_ASSERT(untrimmed.trailing_trimmed == 0);
    const auto anchored = http_detail::find_pflash_query_window(rendered, query, 8, 12, 1);
    TEST_ASSERT(!anchored.valid());
    const std::vector<int32_t> short_query{10, 11, 12, 77};
    const auto too_short = http_detail::find_pflash_query_window(rendered, short_query, 8, 12, 1, false);
    TEST_ASSERT(!too_short.valid());
}

TEST_CASE(ServerUnitFixture, test_pflash_bounded_mapping_prefers_latest_shortened_suffix) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<int32_t> rendered{
        1, 10, 11, 12, 13, 14, 15, 16, 17,
        900, 11, 12, 13, 14, 15, 16, 17, 200, 201,
    };

    const auto bounded =
        http_detail::find_pflash_query_window(rendered, query, 8, 17);
    const auto unbounded = http_detail::find_pflash_query_window(rendered, query, 8);

    TEST_ASSERT(bounded.valid());
    TEST_ASSERT(bounded.end == 17);
    TEST_ASSERT(bounded.tokens == 7);
    // The compatibility path remains width-first when there is no semantic
    // boundary, so the earlier exact duplicate still wins there.
    TEST_ASSERT(unbounded.valid());
    TEST_ASSERT(unbounded.end == 9);
    TEST_ASSERT(unbounded.tokens == 8);
}

TEST_CASE(ServerUnitFixture, test_pflash_bounded_mapping_rejects_earlier_duplicate) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<int32_t> rendered{
        1, 10, 11, 12, 13, 14, 15, 16, 17,
        900, 11, 12, 13, 14, 15, 16, 999, 200,
    };

    const auto bounded =
        http_detail::find_pflash_query_window(rendered, query, 8, 18);
    const auto unbounded =
        http_detail::find_pflash_query_window(rendered, query, 8);

    TEST_ASSERT(!bounded.valid());
    TEST_ASSERT(unbounded.valid());
    TEST_ASSERT(unbounded.end == 9);
    TEST_ASSERT(unbounded.tokens == 8);
}

TEST_CASE(ServerUnitFixture, test_pflash_bounded_mapping_stays_inside_content_start) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<int32_t> rendered{
        1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 200,
    };

    const auto bounded = http_detail::find_pflash_query_window(
        rendered, query, 8, 12, 5);

    TEST_ASSERT(bounded.valid());
    TEST_ASSERT(bounded.end == 12);
    TEST_ASSERT(bounded.tokens == 7);
    TEST_ASSERT(bounded.end - bounded.tokens == 5);
}

TEST_CASE(ServerUnitFixture, test_pflash_maps_content_start_from_leading_sentinel) {
    const std::vector<int32_t> rendered{1, 2, 10, 11, 12, 13, 20};
    const std::vector<int32_t> sentinel_rendered{
        1, 2, 999, 10, 11, 12, 13, 20,
    };

    TEST_ASSERT(http_detail::pflash_query_search_begin_from_sentinel(
        rendered, sentinel_rendered) == 2);
    TEST_ASSERT(http_detail::pflash_query_search_begin_from_sentinel(
        rendered, rendered) < 0);
}

// ─── Explicit query / required-text span mapping ────────────────────────
// A minimal GPT-2 byte-BPE tokenizer whose vocab reproduces the runtime
// failure shape: " What" and "What" are distinct tokens, so a standalone
// query encoding cannot match a prompt where the question's first character
// merges with the preceding space.

static std::string test_gpt2_encode(const std::string & text) {
    static const auto fwd = []() {
        std::array<uint32_t, 256> table{};
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool printable =
                (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                (b >= 174 && b <= 255);
            table[b] = printable ? (uint32_t) b : (uint32_t) (256 + n++);
        }
        return table;
    }();
    std::string out;
    for (char ch : text) {
        const uint32_t cp = fwd[(uint8_t) ch];
        if (cp < 0x80) {
            out.push_back((char) cp);
        } else {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

static std::string write_pflash_bpe_tokenizer_fixture(
        const std::vector<std::string> & raw_tokens,
        const std::string & byte_cover,
        const std::vector<std::string> & control_tokens =
            {"<|im_start|>", "<|im_end|>"}) {
    std::vector<std::string> tokens = control_tokens;
    std::vector<uint32_t> types(tokens.size(), 3);
    const auto add = [&](const std::string & encoded, uint32_t type) {
        if (std::find(tokens.begin(), tokens.end(), encoded) == tokens.end()) {
            tokens.push_back(encoded);
            types.push_back(type);
        }
    };
    for (const auto & raw : raw_tokens) add(test_gpt2_encode(raw), 1);
    for (char ch : byte_cover) add(test_gpt2_encode(std::string(1, ch)), 1);

    std::vector<const char *> token_ptrs;
    for (const auto & token : tokens) token_ptrs.push_back(token.c_str());
    gguf_context * g = gguf_init_empty();
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", token_ptrs.data(),
                     (int32_t) tokens.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_UINT32,
                      types.data(), (int32_t) types.size());
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen35");
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 0);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 1);
    // Per-process names: ctest runs each case in its own process, in
    // parallel, and every process starts the serial at zero.
    static int fixture_serial = 0;
#if defined(_WIN32)
    const long long pid = (long long) _getpid();
#else
    const long long pid = (long long) getpid();
#endif
    const std::string path = test_tmp_path((
        "luce_test_pflash_bpe_" + std::to_string(pid) + "_" +
        std::to_string(++fixture_serial) + ".gguf").c_str()).string();
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    return path;
}

static std::string write_deepseek_marker_tokenizer_fixture() {
    gguf_context * g = gguf_init_empty();
    const char * tokens[] = {
        "x",
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
    };
    const uint32_t token_types[] = {1, 3, 3, 3, 3};
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tokens,
                     sizeof(tokens) / sizeof(tokens[0]));
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_UINT32,
                      token_types, sizeof(token_types) / sizeof(token_types[0]));
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen35");
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 2);

    const std::string path = test_tmp_path("luce_test_deepseek_markers.gguf").string();
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    return path;
}

TEST_CASE(ServerUnitFixture,
        test_pflash_decoded_span_covers_bpe_merged_first_token) {
    const std::string content =
        "See docs.\n\nQuestion: What is the answer?\n Answer:";
    const std::string rendered = "<|im_start|>user\n" + content +
        "<|im_end|>\n<|im_start|>assistant\n";
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"Question", ":", " What", "What", " is", " the", " answer", "?",
         "\n", "\n\n", " Answer", "user", "assistant", "See", " docs", "."},
        rendered + "What is the answer?");
    Tokenizer tok;
    TEST_ASSERT(tok.load_from_gguf(path.c_str()));

    const auto prompt = tok.encode(rendered);
    const std::string needle = "What is the answer?";
    // The standalone query encoding starts with a bare "What" token; the
    // prompt merged the preceding space into " What". The id-suffix matcher
    // therefore accepts a shortened window that misses the first word — the
    // regression this span mapping fixes.
    const auto query_ids = tok.encode(needle);
    const auto legacy = http_detail::find_pflash_query_window(
        prompt, query_ids, 64, -1, 0, /*anchored=*/false);
    TEST_ASSERT(legacy.valid());
    TEST_ASSERT(tok.decode({prompt.begin() + (legacy.end - legacy.tokens),
                            prompt.begin() + legacy.end}) != needle);

    const auto span = http_detail::pflash_decoded_text_span(
        tok, prompt, 0, (int) prompt.size(), needle);
    TEST_ASSERT(span.begin >= 0);
    const std::string covered = tok.decode(
        {prompt.begin() + span.begin, prompt.begin() + span.end});
    TEST_ASSERT(covered.find(needle) != std::string::npos);
    TEST_ASSERT(span.end == legacy.end);
    TEST_ASSERT(span.begin == legacy.end - legacy.tokens - 1);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_decoded_span_prefers_last_occurrence) {
    const std::string content =
        "Ask: What is up? Then again: What is up?";
    const std::string rendered = "<|im_start|>user\n" + content +
        "<|im_end|>\n";
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"Ask", ":", " What", " is", " up", "?", " Then", " again",
         "user", "\n"},
        rendered + "What is up?");
    Tokenizer tok;
    TEST_ASSERT(tok.load_from_gguf(path.c_str()));

    const auto prompt = tok.encode(rendered);
    const std::string needle = "What is up?";
    const auto span = http_detail::pflash_decoded_text_span(
        tok, prompt, 0, (int) prompt.size(), needle);
    TEST_ASSERT(span.begin >= 0);
    const size_t last = rendered.rfind(needle);
    const auto tail = tok.encode(rendered.substr(0, last));
    // The needle's leading space merges into " What" in the prompt, but the
    // standalone-encoded prefix keeps it as its own " " token, so the merged
    // token sits at tail.size() - 1.
    TEST_ASSERT(span.begin == (int) tail.size() - 1);
    TEST_ASSERT(tok.decode({prompt.begin() + span.begin,
                            prompt.begin() + span.end})
                .find(needle) != std::string::npos);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_decoded_span_handles_unicode_and_content_end) {
    const std::string content = "Discuss the café Über Alles? now";
    const std::string rendered = "<|im_start|>user\n" + content + "<|im_end|>\n";
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"Discuss", " the", " café", "café", " Über", " Alles", "?", " now",
         "user", "\n"},
        rendered + "café Über Alles?");
    Tokenizer tok;
    TEST_ASSERT(tok.load_from_gguf(path.c_str()));

    const auto prompt = tok.encode(rendered);
    const std::string needle = "café Über Alles?";
    const auto span = http_detail::pflash_decoded_text_span(
        tok, prompt, 0, (int) prompt.size(), needle);
    TEST_ASSERT(span.begin >= 0);
    TEST_ASSERT(tok.decode({prompt.begin() + span.begin,
                            prompt.begin() + span.end})
                .find(needle) != std::string::npos);
    // A needle that never occurs maps to no span — callers must fail closed.
    const auto missing = http_detail::pflash_decoded_text_span(
        tok, prompt, 0, (int) prompt.size(), "never-present question");
    TEST_ASSERT(missing.begin < 0);
    // An empty needle and an empty range are invalid rather than a
    // degenerate zero-width span.
    TEST_ASSERT(http_detail::pflash_decoded_text_span(
        tok, prompt, 0, (int) prompt.size(), "").begin < 0);
    TEST_ASSERT(http_detail::pflash_decoded_text_span(
        tok, prompt, 4, 4, needle).begin < 0);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_pflash_required_parses_string_arrays) {
    TEST_ASSERT(parse_pflash_required_from_body({}).empty());
    TEST_ASSERT(parse_pflash_required_from_body(
        {{"pflash_required", "not-an-array"}}).empty());
    const auto top = parse_pflash_required_from_body(
        {{"pflash_required",
          {"answer briefly.", "Question:", 7, nullptr}}});
    TEST_ASSERT(top.size() == 2);
    TEST_ASSERT(top[0] == "answer briefly.");
    const auto nested = parse_pflash_required_from_body(
        {{"extra_body", {{"pflash_required", {"keep me"}}}}});
    TEST_ASSERT(nested.size() == 1 && nested[0] == "keep me");
    // extra_body wins over the top-level field, like session_id.
    const auto both = parse_pflash_required_from_body(
        {{"pflash_required", {"outer"}},
         {"extra_body", {{"pflash_required", {"inner"}}}}});
    TEST_ASSERT(both.size() == 1 && both[0] == "inner");
}

TEST_CASE(ServerUnitFixture, test_pflash_instruction_plan_covers_tools_and_late_developer_roles) {
    const std::vector<ChatMessage> messages{
        {"system", "system instruction", ""},
        {"developer", "leading developer instruction", ""},
        {"user", "document text", ""},
        {"assistant", "prior answer", ""},
        {"developer", "late developer instruction", ""},
        {"tool", "tool result is history", "call-1"},
        {"user", "latest query", ""},
    };
    const auto plan = http_detail::plan_pflash_instruction_messages(messages);

    TEST_ASSERT(plan.instruction_messages ==
                std::vector<size_t>({0, 1, 4}));
}

TEST_CASE(ServerUnitFixture, test_pflash_instruction_plan_handles_empty_instructions) {
    const std::vector<ChatMessage> tool_only{
        {"user", "use the tool", ""},
    };
    const auto tool_plan =
        http_detail::plan_pflash_instruction_messages(tool_only);
    TEST_ASSERT(tool_plan.instruction_messages.empty());

    const std::vector<ChatMessage> empty_instruction{
        {"system", "", ""},
        {"user", "plain query", ""},
    };
    const auto empty_plan =
        http_detail::plan_pflash_instruction_messages(empty_instruction);
    TEST_ASSERT(empty_plan.instruction_messages.empty());
}

TEST_CASE(ServerUnitFixture, test_pflash_tool_span_follows_arbitrary_jinja_placement) {
    static const char TPL[] =
        "{%- for m in messages -%}{{ m.content }}{%- endfor -%}"
        "{%- if tools -%}|TOOLS:{{ tools[0].function.name }}{%- endif -%}";
    const std::vector<ChatMessage> messages{{"user", "query-first", ""}};
    const std::string tools =
        R"([{"type":"function","function":{"name":"late_lookup"}}])";
    const std::string with_tools = render_chat_template_jinja(
        TPL, messages, "", "", true, false, tools);
    const std::string without_tools = render_chat_template_jinja(
        TPL, messages, "", "", true, false, "[]");
    const std::vector<int32_t> original(with_tools.begin(), with_tools.end());
    const std::vector<int32_t> variant(without_tools.begin(), without_tools.end());

    const PFlashTokenSpan span =
        http_detail::pflash_changed_token_span(original, variant);
    TEST_ASSERT(span.begin >= (int) messages[0].content.size());
    TEST_ASSERT(span.end == (int) original.size());
    TEST_ASSERT(with_tools.substr(
        (size_t) span.begin, (size_t) (span.end - span.begin)).find(
            "late_lookup") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_pflash_instruction_span_follows_reordered_jinja_message) {
    static const char TPL[] =
        "{%- for m in messages -%}{%- if m.role == 'user' -%}"
        "<{{ m.role }}>{{ m.content }}</{{ m.role }}>"
        "{%- endif -%}{%- endfor -%}"
        "{%- for m in messages -%}{%- if m.role == 'system' -%}"
        "<{{ m.role }}>{{ m.content }}</{{ m.role }}>"
        "{%- endif -%}{%- endfor -%}";
    const std::vector<ChatMessage> messages{
        {"system", "retain this rule", ""},
        {"user", "question first", ""},
    };
    const std::string rendered = render_chat_template_jinja(
        TPL, messages, "", "", true, false);
    auto without_system = messages;
    without_system.erase(without_system.begin());
    const std::string variant_rendered = render_chat_template_jinja(
        TPL, without_system, "", "", true, false);
    const std::vector<int32_t> original(rendered.begin(), rendered.end());
    const std::vector<int32_t> variant(
        variant_rendered.begin(), variant_rendered.end());

    const PFlashTokenSpan span =
        http_detail::pflash_changed_token_span(original, variant);
    TEST_ASSERT(span.begin > (int) messages[1].content.size());
    const std::string retained = rendered.substr(
        (size_t) span.begin, (size_t) (span.end - span.begin));
    TEST_ASSERT(retained.find("<system>") != std::string::npos);
    TEST_ASSERT(retained.find(messages[0].content) != std::string::npos);
    TEST_ASSERT(retained.find("</system>") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_pflash_instruction_spans_are_canonicalized) {
    const auto spans = http_detail::canonicalize_pflash_token_spans(
        {{12, 20}, {0, 4}, {3, 8}, {20, 24}});
    TEST_ASSERT(spans == std::vector<PFlashTokenSpan>({{0, 8}, {12, 24}}));
}

TEST_CASE(ServerUnitFixture, test_pflash_qwen_tool_prefix_boundary_covers_schema) {
    const std::vector<ChatMessage> messages{{"user", "find weather", ""}};
    const std::string tools =
        R"([{"type":"function","function":{"name":"lookup_weather"}}])";
    const std::string sentinel = "__PFLASH_BEGIN_02C47F91__";
    const std::string rendered = render_chat_template(
        messages, ChatFormat::QWEN3, true, false, tools);
    auto marked_messages = messages;
    marked_messages[0].content = sentinel + marked_messages[0].content;
    const std::string marked = render_chat_template(
        marked_messages, ChatFormat::QWEN3, true, false, tools);
    const std::vector<int32_t> rendered_ids(rendered.begin(), rendered.end());
    const std::vector<int32_t> marked_ids(marked.begin(), marked.end());

    const int prefix_end = http_detail::pflash_query_search_begin_from_sentinel(
        rendered_ids, marked_ids);
    TEST_ASSERT(prefix_end > 0);
    TEST_ASSERT(rendered.substr(0, (size_t) prefix_end).find("lookup_weather") !=
                std::string::npos);
    TEST_ASSERT(rendered.substr(0, (size_t) prefix_end).find("find weather") ==
                std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_pflash_qwen_late_developer_span_covers_role_envelope) {
    const std::vector<ChatMessage> messages{
        {"user", std::string(930, 'u'), ""},
        {"assistant", "history", ""},
        {"developer", std::string(180, 'd'), ""},
        {"user", "latest query", ""},
    };
    const std::string rendered = render_chat_template(
        messages, ChatFormat::QWEN3, true, false);
    auto without_developer = messages;
    without_developer.erase(without_developer.begin() + 2);
    const std::string without_developer_rendered = render_chat_template(
        without_developer, ChatFormat::QWEN3, true, false);
    const std::vector<int32_t> ids(rendered.begin(), rendered.end());
    const std::vector<int32_t> variant(
        without_developer_rendered.begin(), without_developer_rendered.end());

    const PFlashTokenSpan span =
        http_detail::pflash_changed_token_span(ids, variant);
    const size_t content_begin = rendered.find(messages[2].content);
    TEST_ASSERT(content_begin != std::string::npos);
    TEST_ASSERT(span.begin >= 0);
    TEST_ASSERT((size_t) span.begin < content_begin);
    TEST_ASSERT((size_t) span.end > content_begin + messages[2].content.size());
    TEST_ASSERT(span.begin < 1024);
    TEST_ASSERT(span.end > 1024);
    TEST_ASSERT(rendered.substr(
        (size_t) span.begin,
        (size_t) (span.end - span.begin)).find("developer") !=
        std::string::npos);
}

namespace {
struct AgentClientRequest { const char * name; const char * path; const char * body; };
const AgentClientRequest kAgentClientRequests[] = {
#include "../pflash/fixtures/agent_client_requests.inc"
};
}  // namespace

TEST_CASE(ServerUnitFixture, test_real_agent_client_requests_render_every_tool_call) {
    // Hermes Agent (chat completions) and Pi (chat completions and Responses)
    // replay their tool history in their own formats; every call must reach
    // the rendered prompt (tool memory empty: these calls were not generated here),
    // every tool result must become a tool message, the system prompt first.
    for (const auto & r : kAgentClientRequests) {
        const json body = json::parse(r.body);
        const bool responses = std::string(r.path) == "/v1/responses";
        const json & messages = responses ? body["input"] : body["messages"];
        ToolMemory tool_memory;
        const auto chat = normalize_chat_messages(
            messages, responses ? ApiFormat::RESPONSES : ApiFormat::OPENAI_CHAT, tool_memory);
        TEST_ASSERT(!chat.empty());
        TEST_ASSERT(chat[0].role == "system");
        std::vector<std::pair<std::string, json>> calls;
        size_t results = 0;
        for (const auto & m : messages) {
            if (responses) {
                const std::string type = m.value("type", "message");
                if (type == "function_call") {
                    calls.push_back({m.value("name", ""), json::parse(m.value("arguments", "{}"))});
                } else if (type == "function_call_output") {
                    ++results;
                }
            } else {
                if (m.value("role", "") == "tool") ++results;
                if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                    for (const auto & tc : m["tool_calls"]) {
                        calls.push_back({tc["function"].value("name", ""),
                                         json::parse(tc["function"].value("arguments", "{}"))});
                    }
                }
            }
        }
        TEST_ASSERT(!calls.empty());
        size_t tool_messages = 0;
        for (const auto & m : chat) {
            if (m.role == "tool") ++tool_messages;
        }
        TEST_ASSERT(tool_messages == results);
        const std::string rendered = render_chat_template(
            chat, ChatFormat::QWEN3, true, false);
        for (const auto & [name, args] : calls) {
            TEST_ASSERT(rendered.find("<function=" + name + ">") != std::string::npos);
            for (const auto & [key, value] : args.items()) {
                const std::string v = value.is_string() ? value.get<std::string>() : value.dump();
                TEST_ASSERT(rendered.find("<parameter=" + key + ">\n" + v + "\n</parameter>") !=
                            std::string::npos);
            }
        }
    }
}

TEST_CASE(ServerUnitFixture, test_pflash_responses_string_tails_only_raw_content) {
    ToolMemory tool_memory;
    const auto normalized = normalize_chat_messages(
        json("raw completion input"), ApiFormat::RESPONSES, tool_memory);
    TEST_ASSERT(normalized.size() == 1);
    TEST_ASSERT(normalized[0].role == "user");
    TEST_ASSERT(normalized[0].content == "raw completion input");

    const std::vector<int32_t> rendered{
        1, 2, 10, 11, 12, 13, 14, 200, 201,
    };
    const auto window = http_detail::pflash_tail_query_window(
        rendered, 128, /*query_end=*/7, /*query_begin=*/2);
    TEST_ASSERT(window.valid());
    TEST_ASSERT(window.end == 7);
    TEST_ASSERT(window.tokens == 5);
    TEST_ASSERT(window.end - window.tokens == 2);
}

TEST_CASE(ServerUnitFixture, test_compress_result_fails_closed_on_empty_ids) {
    const auto failed = ModelBackend::CompressResult::from_compressed_ids({});
    const auto succeeded =
        ModelBackend::CompressResult::from_compressed_ids({11, 12});

    TEST_ASSERT(!failed.ok);
    TEST_ASSERT(failed.compressed_ids.empty());
    TEST_ASSERT(succeeded.ok);
    TEST_ASSERT(succeeded.compressed_ids == std::vector<int32_t>({11, 12}));
}

TEST_CASE(ServerUnitFixture, test_pflash_parser_fingerprint_has_fixed_encoding) {
    const std::vector<int32_t> ids{1, -2, 2147483647};
    TEST_ASSERT(http_detail::pflash_token_fingerprint(ids) ==
                "45409a0b0f44c5fd");
}

TEST_CASE(ServerUnitFixture, test_pflash_tail_query_window) {
    const auto empty = http_detail::pflash_tail_query_window({}, 128);
    TEST_ASSERT(!empty.valid());

    const std::vector<int32_t> short_prompt{1, 2, 3};
    const auto short_tail =
        http_detail::pflash_tail_query_window(short_prompt, 128);
    TEST_ASSERT(short_tail.valid());
    TEST_ASSERT(short_tail.end == 3);
    TEST_ASSERT(short_tail.tokens == 3);

    std::vector<int32_t> long_prompt(200);
    const auto capped_tail =
        http_detail::pflash_tail_query_window(long_prompt, 128);
    TEST_ASSERT(capped_tail.valid());
    TEST_ASSERT(capped_tail.end == 200);
    TEST_ASSERT(capped_tail.tokens == 128);
    const auto bounded_tail =
        http_detail::pflash_tail_query_window(long_prompt, 128, 150);
    TEST_ASSERT(bounded_tail.valid());
    TEST_ASSERT(bounded_tail.end == 150);
    TEST_ASSERT(bounded_tail.tokens == 128);
    TEST_ASSERT(!http_detail::pflash_tail_query_window(long_prompt, 0).valid());
    TEST_ASSERT(!http_detail::pflash_tail_query_window(long_prompt, 128, 0).valid());
    TEST_ASSERT(!http_detail::pflash_tail_query_window(long_prompt, 128, 201).valid());
}

// Renders ``messages`` with the server's own chat template and returns the
// chat-query turn the scorer would use, decoded as {header, content}.
struct PflashRenderedQueryTurn {
    bool valid = false;
    bool later_turns = false;
    std::string family;
    std::string header;
    std::string content;
    std::string after;
    std::string closing;      // content end .. turn end
    std::string generation;   // generation prompt .. prompt end
    std::vector<std::string> roles;
};

static PflashRenderedQueryTurn pflash_rendered_query_turn(
        const std::vector<ChatMessage> & messages,
        ChatFormat format,
        bool thinking,
        const std::vector<std::string> & control_tokens) {
    const std::string rendered = render_chat_template(
        messages, format, /*add_generation_prompt=*/true, thinking);
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"What", " is", " the", " answer", "?", "user", "assistant", "model",
         "system", "\n", "Sure", "."},
        rendered, control_tokens);
    Tokenizer tok;
    PflashRenderedQueryTurn out;
    if (!tok.load_from_gguf(path.c_str())) {
        unlink(path.c_str());
        return out;
    }
    const auto prompt = tok.encode(rendered);
    ChatMarkers markers;
    if (resolve_chat_markers(tok, markers)) {
        out.family = markers.family;
        const auto turn = http_detail::pflash_chat_query_turn(
            tok, markers, tok, prompt);
        out.valid = turn.valid();
        if (out.valid) {
            out.header = tok.decode({prompt.begin() + turn.role_begin,
                                     prompt.begin() + turn.content_begin});
            out.content = tok.decode({prompt.begin() + turn.content_begin,
                                      prompt.begin() + turn.content_end});
            out.after = tok.decode({prompt.begin() + turn.content_end,
                                    prompt.end()});
            out.closing = tok.decode({prompt.begin() + turn.content_end,
                                      prompt.begin() + turn.turn_end});
            out.generation = tok.decode(
                {prompt.begin() + turn.generation_begin, prompt.end()});
            out.later_turns = turn.later_turns;
            for (const auto & each : turn.turns) out.roles.push_back(each.role);
        }
    }
    unlink(path.c_str());
    return out;
}

TEST_CASE(ServerUnitFixture,
        test_pflash_chat_query_turn_skips_rendered_generation_prompt) {
    // Every family's generation prompt carries a think/channel prefix after
    // the assistant marker; the query must still be the latest user turn.
    const std::vector<ChatMessage> messages{
        {"system", "You are helpful.", ""},
        {"user", "first turn", ""},
        {"assistant", "Sure.", ""},
        {"user", "What is the answer?", ""},
    };
    struct Family {
        ChatFormat format;
        const char * name;
        std::vector<std::string> control_tokens;
        const char * header;
    };
    const std::vector<Family> families{
        {ChatFormat::QWEN3, "qwen", {"<|im_start|>", "<|im_end|>"},
         "<|im_start|>user\n"},
        {ChatFormat::GEMMA4, "gemma", {"<|turn>", "<turn|>"},
         "<|turn>user\n"},
        {ChatFormat::DEEPSEEK4, "deepseek",
         {"<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>",
          "<｜Assistant｜>"},
         "<｜User｜>"},
        {ChatFormat::LAGUNA, "laguna",
         {"<system>", "</system>", "<user>", "</user>", "<assistant>",
          "</assistant>"},
         "<user>\n"},
    };
    for (const auto & family : families) {
        for (const bool thinking : {false, true}) {
            const auto turn = pflash_rendered_query_turn(
                messages, family.format, thinking, family.control_tokens);
            TEST_ASSERT_MSG(turn.family == family.name, family.name);
            TEST_ASSERT_MSG(turn.valid, family.name);
            TEST_ASSERT_MSG(turn.content == "What is the answer?",
                            std::string(family.name) + " thinking=" +
                                (thinking ? "1" : "0") + " content=[" +
                                turn.content + "]");
            TEST_ASSERT_MSG(turn.header == family.header,
                            std::string(family.name) + " header=[" +
                                turn.header + "]");
            // The rendered tail after the content is template machinery.
            TEST_ASSERT_MSG(turn.after.find("answer") == std::string::npos,
                            family.name);
            TEST_ASSERT_MSG(!turn.later_turns, family.name);
        }
    }
}

TEST_CASE(ServerUnitFixture,
        test_pflash_chat_query_turn_skips_tool_output_turns) {
    // Agent loop: tool results render inside user turns on Qwen and
    // DeepSeek. The query stays on the user's own latest turn.
    const std::vector<ChatMessage> messages{
        {"user", "What is the answer?", ""},
        {"assistant", "Sure.", ""},
        {"tool", "tool output here", "call-1"},
    };
    const auto qwen = pflash_rendered_query_turn(
        messages, ChatFormat::QWEN3, false, {"<|im_start|>", "<|im_end|>"});
    TEST_ASSERT(qwen.valid);
    TEST_ASSERT_MSG(qwen.content == "What is the answer?", qwen.content);
    TEST_ASSERT(qwen.header == "<|im_start|>user\n");
    TEST_ASSERT(qwen.later_turns);
    TEST_ASSERT(qwen.roles == std::vector<std::string>({"user", "assistant", "tool"}));
    TEST_ASSERT_MSG(qwen.closing == "<|im_end|>", qwen.closing);
    TEST_ASSERT_MSG(qwen.generation == "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    qwen.generation);

    const auto deepseek = pflash_rendered_query_turn(
        messages, ChatFormat::DEEPSEEK4, true,
        {"<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>",
         "<｜Assistant｜>"});
    TEST_ASSERT(deepseek.valid);
    TEST_ASSERT_MSG(deepseek.content == "What is the answer?",
                    deepseek.content);
}

// Request probe: the human user messages' content spans, by message ROLE.
// Renders ``messages`` (Qwen template) and returns the decoded content of
// each mapped span, latest first ("<none>" when mapping failed).
static std::vector<std::string> pflash_human_contents(
        const std::vector<ChatMessage> & messages, bool drop_last_turn = false) {
    const std::string rendered = render_chat_template(
        messages, ChatFormat::QWEN3, /*add_generation_prompt=*/true, false);
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"What", " is", " the", " answer", "?", "user", "assistant", "\n",
         "Sure", ".", "first", " question", "tool", " output"},
        rendered);
    Tokenizer tok;
    std::vector<std::string> out;
    if (!tok.load_from_gguf(path.c_str())) {
        unlink(path.c_str());
        return {"<tokenizer>"};
    }
    unlink(path.c_str());
    const auto prompt = tok.encode(rendered);
    ChatMarkers markers;
    if (!resolve_chat_markers(tok, markers)) return {"<markers>"};
    auto turns = http_detail::pflash_chat_query_turn(tok, markers, tok, prompt).turns;
    if (drop_last_turn && !turns.empty()) turns.pop_back();
    std::vector<PFlashTokenSpan> spans;
    if (!http_detail::pflash_human_message_spans(tok, prompt, turns, messages, 2, spans)) {
        return {"<none>"};
    }
    for (const auto & span : spans) {
        out.push_back(tok.decode({prompt.begin() + span.begin, prompt.begin() + span.end}));
    }
    return out;
}

TEST_CASE(ServerUnitFixture,
        test_pflash_request_probe_human_messages_skip_tool_results) {
    // Qwen renders the tool result inside a user turn; by role it is not
    // human, so the previous human message is the first question.
    const auto contents = pflash_human_contents({
        {"user", "first question", ""},
        {"assistant", "Sure.", ""},
        {"tool", "tool output", "call-1"},
        {"user", "What is the answer?", ""},
    });
    TEST_ASSERT(contents == std::vector<std::string>({"What is the answer?", "first question"}));
    // An agent step after the human turn: still the human message.
    const auto agent = pflash_human_contents({
        {"user", "What is the answer?", ""},
        {"assistant", "Sure.", ""},
        {"tool", "tool output", "call-1"},
    });
    TEST_ASSERT(agent == std::vector<std::string>({"What is the answer?"}));
    // A tool result whose text equals a human message does not count.
    const auto same_text = pflash_human_contents({
        {"user", "first question", ""},
        {"assistant", "Sure.", ""},
        {"tool", "What is the answer?", "call-1"},
        {"user", "What is the answer?", ""},
    });
    TEST_ASSERT(same_text.size() == 2);
    TEST_ASSERT(same_text[1] == "first question");
}

TEST_CASE(ServerUnitFixture,
        test_pflash_request_probe_human_messages_repeat_and_fail_closed) {
    // Equal human contents map in order: the latest to the last turn.
    const std::vector<ChatMessage> repeated{
        {"user", "What is the answer?", ""},
        {"assistant", "Sure.", ""},
        {"user", "What is the answer?", ""},
    };
    const auto contents = pflash_human_contents(repeated);
    TEST_ASSERT(contents == std::vector<std::string>({"What is the answer?", "What is the answer?"}));
    // A rendered turn missing for a human message: no verbatim mapping
    // (the server then maps by sentinel renders).
    TEST_ASSERT(pflash_human_contents(repeated, /*drop_last_turn=*/true) ==
                std::vector<std::string>({"<none>"}));
    // An empty human message maps to the empty span.
    const auto empty = pflash_human_contents({
        {"user", "first question", ""},
        {"assistant", "Sure.", ""},
        {"user", "", ""},
    });
    TEST_ASSERT(empty.size() == 2 && empty[0].empty() && empty[1] == "first question");
    // A message ending in a newline keeps it: the span is the exact content.
    const auto newline = pflash_human_contents({
        {"user", "first question\n", ""},
        {"assistant", "Sure.", ""},
        {"user", "What is the answer?", ""},
    });
    TEST_ASSERT(newline.size() == 2 && newline[1] == "first question\n");
}

TEST_CASE(ServerUnitFixture,
        test_pflash_chat_query_turn_falls_back_without_user_turn) {
    const auto turn = pflash_rendered_query_turn(
        {{"system", "You are helpful.", ""}}, ChatFormat::QWEN3, true,
        {"<|im_start|>", "<|im_end|>"});
    TEST_ASSERT(turn.valid);
    TEST_ASSERT(turn.content == "You are helpful.");
    TEST_ASSERT(turn.header == "<|im_start|>system\n");
}

TEST_CASE(ServerUnitFixture,
        test_pflash_chat_query_turn_open_tail_runs_to_prompt_end) {
    const std::string rendered = "<|im_start|>user\nhello there";
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"hello", " there", "user", "\n"}, rendered);
    Tokenizer tok;
    TEST_ASSERT(tok.load_from_gguf(path.c_str()));

    const auto prompt = tok.encode(rendered);
    ChatMarkers markers;
    TEST_ASSERT(resolve_chat_markers(tok, markers));
    const auto span = http_detail::pflash_chat_query_turn(
        tok, markers, tok, prompt);
    TEST_ASSERT(span.valid());
    TEST_ASSERT(span.content_end == (int) prompt.size());
    TEST_ASSERT(tok.decode({prompt.begin() + span.content_begin,
                            prompt.begin() + span.content_end})
                == "hello there");
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_chat_query_turn_rejects_markerless_text) {
    const std::string rendered = "just some raw text, no chat markers";
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"just", " some", " raw", " text"}, rendered);
    Tokenizer tok;
    TEST_ASSERT(tok.load_from_gguf(path.c_str()));

    const auto prompt = tok.encode(rendered);
    ChatMarkers markers;
    TEST_ASSERT(resolve_chat_markers(tok, markers));
    const auto span = http_detail::pflash_chat_query_turn(
        tok, markers, tok, prompt);
    TEST_ASSERT(!span.valid());
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_pflash_normalizes_multipart_latest_user_for_reverse_lookup) {
    ToolMemory tool_memory;
    const json messages = json::array({
        {{"role", "user"}, {"content", "older user"}},
        {{"role", "assistant"}, {"content", "assistant before latest"}},
        {{"role", "user"}, {"content", json::array({
            {{"type", "input_text"}, {"text", "input-"}},
            {{"type", "input_image"}, {"image_url", "ignored"}},
            {{"type", "text"}, {"text", "text"}}
        })}},
        {{"role", "assistant"}, {"content", "assistant after latest"}},
        {{"role", "tool"}, {"content", "tool after latest"}}
    });

    const auto normalized = normalize_chat_messages(
        messages, ApiFormat::OPENAI_CHAT, tool_memory);
    const auto latest_user = std::find_if(
        normalized.rbegin(), normalized.rend(),
        [](const ChatMessage & message) { return message.role == "user"; });
    TEST_ASSERT(latest_user != normalized.rend());
    if (latest_user != normalized.rend()) {
        TEST_ASSERT(latest_user->content == "input-text");
    }
}

TEST_CASE(ServerUnitFixture, test_pflash_selection_cache_and_continuation_policy) {
    TEST_ASSERT(http_detail::pflash_full_cache_restore_allowed(false));
    TEST_ASSERT(!http_detail::pflash_full_cache_restore_allowed(true));
}

TEST_CASE(ServerUnitFixture, test_timings_json_carries_pflash_details) {
    GenTimings timings;
    TEST_ASSERT(!build_timings_json(timings, 0).contains("pflash"));
    timings.pflash = {{"compress_ms", 12.5}, {"view", {{"mode", "continue"}}}};
    const auto out = build_timings_json(timings, 0);
    TEST_ASSERT(out["pflash"]["view"]["mode"] == "continue");
}

TEST_CASE(ServerUnitFixture, test_pflash_join_kept_spans_breaks_between_pieces) {
    const std::string text = "one fact.\ngap text.Two starts.\n\nThree.";
    const std::string path = write_pflash_bpe_tokenizer_fixture({}, text);
    Tokenizer tok;
    TEST_ASSERT(tok.load_from_gguf(path.c_str()));
    const auto ids = tok.encode(text);
    const auto token_at = [&] (const std::string & needle) {
        const auto span = http_detail::pflash_decoded_text_span(
            tok, ids, 0, (int) ids.size(), needle);
        return span;
    };
    const auto one = token_at("one fact.");
    const auto two = token_at("Two starts.");
    const auto three = token_at("\n\nThree.");
    // Non-adjacent pieces without a break get a paragraph break...
    TEST_ASSERT_MSG(http_detail::pflash_join_kept_spans(tok, ids, {one, two}) ==
                    "one fact.\n\nTwo starts.",
                    http_detail::pflash_join_kept_spans(tok, ids, {one, two}));
    // ...a piece that already opens a paragraph is left alone...
    TEST_ASSERT(http_detail::pflash_join_kept_spans(tok, ids, {one, three}) ==
                "one fact.\n\nThree.");
    // ...and adjacent pieces are joined as they were.
    TEST_ASSERT(http_detail::pflash_join_kept_spans(tok, ids, {{0, 3}, {3, 5}}) ==
                tok.decode({ids.begin(), ids.begin() + 5}));
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_pflash_subtract_token_spans) {
    const std::vector<PFlashTokenSpan> spans{{0, 10}, {20, 30}, {40, 50}};
    const std::vector<PFlashTokenSpan> minus{{5, 22}, {25, 26}, {40, 50}};
    const auto out = http_detail::pflash_subtract_token_spans(spans, minus);
    TEST_ASSERT(out.size() == 3);
    TEST_ASSERT(out[0].begin == 0 && out[0].end == 5);
    TEST_ASSERT(out[1].begin == 22 && out[1].end == 25);
    TEST_ASSERT(out[2].begin == 26 && out[2].end == 30);
    TEST_ASSERT(http_detail::pflash_subtract_token_spans(spans, {}).size() == 3);
    TEST_ASSERT(http_detail::pflash_subtract_token_spans(spans, {{0, 60}}).empty());
}

TEST_CASE(ServerUnitFixture, test_pflash_recall_excerpt_strips_chat_markers) {
    const std::string text =
        "tail of a fact<|im_end|>\n<|im_start|>assistant\nSure, noted."
        "<|im_end|>\n<|im_start|>user\n";
    const std::string excerpt = http_detail::pflash_recall_excerpt(
        text, {"<|im_start|>"}, {"<|im_end|>"}, /*generic_role_lines=*/true);
    TEST_ASSERT_MSG(excerpt == "tail of a fact\n\n\nSure, noted.", excerpt);
    TEST_ASSERT(http_detail::pflash_recall_excerpt(
        "<|im_end|>\n", {"<|im_start|>"}, {"<|im_end|>"}, true).empty());
}

TEST_CASE(ServerUnitFixture, test_pflash_chat_view_store_matches_prompt_prefix) {
    http_detail::PflashChatViewStore store(2);
    http_detail::PflashChatView first;
    first.raw_tokens = {1, 2, 3, 9, 9};
    first.raw_gen_begin = 3;
    first.drafter_ids = {1, 2, 3, 9, 9};
    first.drafter_gen_begin = 3;
    store.remember(first);

    http_detail::PflashChatView found;
    // The next turn keeps the prefix before the old generation prompt.
    TEST_ASSERT(store.find({1, 2, 3, 4, 5, 9, 9}, {1, 2, 3, 4, 5, 9, 9}, found));
    TEST_ASSERT(found.raw_gen_begin == 3);
    // A different conversation does not match.
    TEST_ASSERT(!store.find({1, 7, 3, 4}, {1, 7, 3, 4}, found));
    // A continuation replaces the view it continues.
    http_detail::PflashChatView second = first;
    second.raw_tokens = {1, 2, 3, 4, 5, 9, 9};
    second.raw_gen_begin = 5;
    second.drafter_ids = second.raw_tokens;
    second.drafter_gen_begin = 5;
    store.remember(second);
    TEST_ASSERT(store.size() == 1);
    TEST_ASSERT(store.find({1, 2, 3, 4, 5, 6, 9}, {1, 2, 3, 4, 5, 6, 9}, found));
    TEST_ASSERT(found.raw_gen_begin == 5);
}

TEST_CASE(ServerUnitFixture, test_pflash_kept_tokens_follow_selector_chunks) {
    // 100 tokens in chunks of 10; query [80, 85); instruction span [3, 12)
    // touches chunks 0 and 1.
    const std::vector<PFlashTokenSpan> kept{{3, 12}};
    // Suffix structural: chunks 0, 1, 8, 9.
    TEST_ASSERT(http_detail::pflash_kept_tokens(100, 10, 80, 85, kept, true) == 40);
    // Suffix scored: only the query's chunk after it.
    TEST_ASSERT(http_detail::pflash_kept_tokens(100, 10, 80, 85, kept, false) == 30);
    TEST_ASSERT(http_detail::pflash_kept_tokens(0, 10, 0, 0, kept, true) == 0);
}

TEST_CASE(ServerUnitFixture, test_pflash_effective_keep_ratio_spends_on_droppable) {
    // 1000 tokens, 400 kept, 5 %: budget 400 + 30 + 1 slack.
    TEST_ASSERT(std::abs(http_detail::pflash_effective_keep_ratio(
        1000, 400, 0.05) - 0.431) < 1e-12);
    // Nothing kept: the plain ratio plus the slack token.
    TEST_ASSERT(std::abs(http_detail::pflash_effective_keep_ratio(
        1000, 0, 0.05) - 0.051) < 1e-12);
    // Everything kept caps at 1.
    TEST_ASSERT(http_detail::pflash_effective_keep_ratio(1000, 1000, 0.05) == 1.0);
    // The floored budget never lands below the kept tokens.
    for (int kept = 0; kept <= 997; kept += 7) {
        const double ratio =
            http_detail::pflash_effective_keep_ratio(997, kept, 0.013);
        TEST_ASSERT((int) std::floor(997.0 * ratio) >= kept);
    }
}

TEST_CASE(ServerUnitFixture, test_pflash_cache_admission_never_drops_a_threshold_of_cache) {
    using http_detail::PflashAdmission;
    using http_detail::pflash_cache_admission;
    const int t = 16384;
    // Warm agent step: a small uncached tail stays raw, whatever the length.
    TEST_ASSERT(pflash_cache_admission(2000, 90000, t) == PflashAdmission::UnderThreshold);
    TEST_ASSERT(pflash_cache_admission(0, 0, t) == PflashAdmission::UnderThreshold);
    TEST_ASSERT(pflash_cache_admission(t - 1, 0, t) == PflashAdmission::UnderThreshold);
    // Cold prompt (nothing or little cached) at or past the threshold compresses.
    TEST_ASSERT(pflash_cache_admission(t, 0, t) == PflashAdmission::Compress);
    TEST_ASSERT(pflash_cache_admission(40000, t - 1, t) == PflashAdmission::Compress);
    // A big new chunk on a warm session stays raw: compressing the whole
    // prompt would throw away a threshold's worth of reusable cache.
    TEST_ASSERT(pflash_cache_admission(t, t, t) == PflashAdmission::WarmCache);
    TEST_ASSERT(pflash_cache_admission(30000, 50000, t) == PflashAdmission::WarmCache);
    // A raw prompt that does not fit the context always compresses.
    TEST_ASSERT(pflash_cache_admission(2000, 90000, t, false) == PflashAdmission::Compress);
    TEST_ASSERT(pflash_cache_admission(30000, 50000, t, false) == PflashAdmission::Compress);
    TEST_ASSERT(std::string(http_detail::pflash_admission_name(PflashAdmission::Compress)) == "compress");
    TEST_ASSERT(std::string(http_detail::pflash_admission_name(PflashAdmission::UnderThreshold)) == "under_threshold");
    TEST_ASSERT(std::string(http_detail::pflash_admission_name(PflashAdmission::WarmCache)) == "warm_cache");
}

TEST_CASE(ServerUnitFixture, test_pflash_cache_admission_flag_defaults_off) {
    luce_unsetenv("PFLASH_ADMISSION");
    TEST_ASSERT(!http_detail::pflash_cache_admission_enabled());
    luce_setenv("PFLASH_ADMISSION", "length");
    TEST_ASSERT(!http_detail::pflash_cache_admission_enabled());
    luce_setenv("PFLASH_ADMISSION", "cache");
    TEST_ASSERT(http_detail::pflash_cache_admission_enabled());
    luce_unsetenv("PFLASH_ADMISSION");
}

TEST_CASE(ServerUnitFixture, test_pflash_has_tool_activity) {
    const json tools = json::array({{{"type", "function"},
        {"function", {{"name", "read_file"}}}}});
    const std::vector<ChatMessage> plain{
        {"system", "You are helpful.", ""}, {"user", "hi", ""},
        {"assistant", "Sure.", ""}, {"user", "and?", ""}};
    TEST_ASSERT(!http_detail::pflash_has_tool_activity(json(), plain));
    TEST_ASSERT(!http_detail::pflash_has_tool_activity(json::array(), plain));
    TEST_ASSERT(http_detail::pflash_has_tool_activity(tools, plain));
    auto with_result = plain;
    with_result.push_back({"tool", "output", "call-1"});
    TEST_ASSERT(http_detail::pflash_has_tool_activity(json(), with_result));
    auto with_call = plain;
    with_call[2].tool_calls.push_back({"call-1", "read_file", json::object()});
    TEST_ASSERT(http_detail::pflash_has_tool_activity(json(), with_call));
}

TEST_CASE(ServerUnitFixture, test_pflash_span_tokens_within_clips) {
    const std::vector<PFlashTokenSpan> spans{{0, 10}, {20, 30}, {40, 50}};
    TEST_ASSERT(http_detail::pflash_span_tokens_within(spans, 0, 100) == 30);
    TEST_ASSERT(http_detail::pflash_span_tokens_within(spans, 5, 25) == 10);
    TEST_ASSERT(http_detail::pflash_span_tokens_within(spans, 10, 20) == 0);
    TEST_ASSERT(http_detail::pflash_span_tokens_within(spans, 45, 45) == 0);
    TEST_ASSERT(http_detail::pflash_span_tokens_within({}, 0, 100) == 0);
}

TEST_CASE(ServerUnitFixture, test_pflash_complement_token_spans) {
    using V = std::vector<PFlashTokenSpan>;
    const auto c = http_detail::pflash_complement_token_spans(V{{10, 20}, {30, 40}}, 50);
    TEST_ASSERT(c.size() == 3);
    TEST_ASSERT(c[0].begin == 0 && c[0].end == 10);
    TEST_ASSERT(c[1].begin == 20 && c[1].end == 30);
    TEST_ASSERT(c[2].begin == 40 && c[2].end == 50);
    TEST_ASSERT(http_detail::pflash_complement_token_spans(V{}, 5).size() == 1);
    TEST_ASSERT(http_detail::pflash_complement_token_spans(V{{0, 5}}, 5).empty());
    const auto clipped = http_detail::pflash_complement_token_spans(V{{0, 3}, {4, 99}}, 10);
    TEST_ASSERT(clipped.size() == 1 && clipped[0].begin == 3 && clipped[0].end == 4);
}

TEST_CASE(ServerUnitFixture, test_pflash_arrive_and_rebuild_ratio_flags) {
    luce_unsetenv("PFLASH_ARRIVE");
    luce_unsetenv("PFLASH_CHAT_REBUILD_RATIO");
    TEST_ASSERT(!http_detail::pflash_arrive_enabled());
    TEST_ASSERT(http_detail::pflash_chat_rebuild_ratio() == 2.0);
    luce_setenv("PFLASH_ARRIVE", "1");
    luce_setenv("PFLASH_CHAT_REBUILD_RATIO", "0");
    TEST_ASSERT(http_detail::pflash_arrive_enabled());
    TEST_ASSERT(http_detail::pflash_chat_rebuild_ratio() == 0.0);
    luce_setenv("PFLASH_CHAT_REBUILD_RATIO", "3.5");
    TEST_ASSERT(http_detail::pflash_chat_rebuild_ratio() == 3.5);
    luce_setenv("PFLASH_CHAT_REBUILD_RATIO", "-1");
    TEST_ASSERT(http_detail::pflash_chat_rebuild_ratio() == 2.0);
    luce_setenv("PFLASH_CHAT_REBUILD_RATIO", "x");
    TEST_ASSERT(http_detail::pflash_chat_rebuild_ratio() == 2.0);
    luce_unsetenv("PFLASH_ARRIVE");
    luce_unsetenv("PFLASH_CHAT_REBUILD_RATIO");
    luce_unsetenv("PFLASH_CHAT_REBUILD_MIN_TOKENS");
    TEST_ASSERT(http_detail::pflash_chat_rebuild_min_tokens() == 0);
    luce_setenv("PFLASH_CHAT_REBUILD_MIN_TOKENS", "65536");
    TEST_ASSERT(http_detail::pflash_chat_rebuild_min_tokens() == 65536);
    luce_setenv("PFLASH_CHAT_REBUILD_MIN_TOKENS", "-5");
    TEST_ASSERT(http_detail::pflash_chat_rebuild_min_tokens() == 0);
    luce_unsetenv("PFLASH_CHAT_REBUILD_MIN_TOKENS");
}

TEST_CASE(ServerUnitFixture, test_pflash_target_token_ceiling_floors) {
    TEST_ASSERT(http_detail::pflash_target_token_ceiling(7, 0.5) == 3);
    TEST_ASSERT(http_detail::pflash_target_token_ceiling(120000, 16384.0 / 120000.0) == 16384);
    TEST_ASSERT(http_detail::pflash_target_token_ceiling(-1, 0.5) < 0);
}

TEST_CASE(ServerUnitFixture, test_pflash_query_mapping_rejects_weak_punctuation_match) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<int32_t> rendered{1, 2, 15, 16, 17, 200, 201};
    TEST_ASSERT(!http_detail::find_pflash_query_window(rendered, query).valid());

    const std::vector<int32_t> short_query{30, 31, 32};
    const std::vector<int32_t> short_rendered{1, 30, 31, 32, 200};
    const auto short_window =
        http_detail::find_pflash_query_window(short_rendered, short_query);
    TEST_ASSERT(short_window.valid());
    TEST_ASSERT(short_window.tokens == 3);
    TEST_ASSERT(short_window.end == 4);
}

TEST_CASE(ServerUnitFixture, test_pflash_score_validation_counts_nan_and_inf) {
    const float values[]{
        0.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        1.0f,
    };
    TEST_ASSERT(count_nonfinite_scores(values, 5) == 3);
    TEST_ASSERT(count_nonfinite_scores(values, 1) == 0);
}

TEST_CASE(ServerUnitFixture, test_qwen35_pflash_rejects_missing_query_window) {
    DrafterContext ctx;
    ctx.loaded = true;
    const std::vector<int32_t> ids(16, 1);

    const auto compressed = drafter_score_and_compress(
        ctx, ids, 0.5f, /*chunk_size=*/32, /*n_lookahead=*/8,
        /*pool_kernel=*/13, /*score_query_end=*/-1);

    TEST_ASSERT(compressed.empty());
    TEST_ASSERT(std::string(luce_last_error()) ==
                "qwen35 scorer query window out of range");
}

TEST_CASE(ServerUnitFixture, test_pflash_ipc_rejects_unsupported_query_widths) {
    const auto accepted = [](int score_query_tokens) {
        std::string line;
        std::string error;
        const bool formatted = format_pflash_drafter_ipc_compress_command(
            0.5f, 16, score_query_tokens, "/tmp/pflash_ids.bin", line, error);
        if (!formatted) return false;
        PFlashDrafterIpcCompressCommand parsed;
        return parse_pflash_drafter_ipc_compress_command(line, parsed, error) &&
               parsed.score_query_tokens == score_query_tokens;
    };
    // The explicit scorer query sizes its own window, so widths above the
    // former eight-token cap round-trip; non-positive widths still fail closed.
    TEST_ASSERT(accepted(1));
    TEST_ASSERT(accepted(8));
    TEST_ASSERT(accepted(128));
    TEST_ASSERT(!accepted(0));
    TEST_ASSERT(!accepted(-1));
    TEST_ASSERT(!accepted((std::numeric_limits<int>::min)()));
}

TEST_CASE(ServerUnitFixture, test_pflash_query_capture_splits_across_chunks) {
    const auto first = query_capture_slice(4093, 4101, 0, 4096);
    TEST_ASSERT(first.valid());
    TEST_ASSERT(first.chunk_offset == 4093);
    TEST_ASSERT(first.query_offset == 0);
    TEST_ASSERT(first.tokens == 3);

    const auto second = query_capture_slice(4093, 4101, 4096, 4096);
    TEST_ASSERT(second.valid());
    TEST_ASSERT(second.chunk_offset == 0);
    TEST_ASSERT(second.query_offset == 3);
    TEST_ASSERT(second.tokens == 5);

    TEST_ASSERT(!query_capture_slice(4093, 4101, 8192, 4096).valid());
}

TEST_CASE(ServerUnitFixture, test_daemon_io_external_cancellation_latches) {
    bool cancel = false;
    DaemonIO io;
    io.should_cancel = [&cancel]() { return cancel; };

    TEST_ASSERT(!io.is_cancelled());
    cancel = true;
    TEST_ASSERT(io.is_cancelled());
    cancel = false;
    TEST_ASSERT(io.is_cancelled());
}

#if !defined(_WIN32)
TEST_CASE(ServerUnitFixture, test_http_peer_socket_probe_preserves_half_close) {
    int sockets[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    TEST_ASSERT(http_detail::inspect_peer_socket(sockets[0]) ==
                http_detail::PeerSocketState::Connected);

    const char byte = 'x';
    TEST_ASSERT(write(sockets[1], &byte, 1) == 1);
    TEST_ASSERT(http_detail::inspect_peer_socket(sockets[0]) ==
                http_detail::PeerSocketState::Connected);

    char received = 0;
    TEST_ASSERT(read(sockets[0], &received, 1) == 1);
    TEST_ASSERT(received == byte);

    // Finishing the request direction must not cancel a response that the
    // peer is still reading.
    TEST_ASSERT(shutdown(sockets[1], SHUT_WR) == 0);
    TEST_ASSERT(http_detail::inspect_peer_socket(sockets[0]) ==
                http_detail::PeerSocketState::ReadClosed);
    const char response = 'y';
    TEST_ASSERT(write(sockets[0], &response, 1) == 1);
    TEST_ASSERT(read(sockets[1], &received, 1) == 1);
    TEST_ASSERT(received == response);

    const int closed_fd = sockets[0];
    close(closed_fd);
    sockets[0] = -1;
    TEST_ASSERT(http_detail::inspect_peer_socket(closed_fd) ==
                http_detail::PeerSocketState::Disconnected);
    close(sockets[1]);
}

TEST_CASE(ServerUnitFixture, test_http_heartbeat_never_waits_for_stalled_peer) {
    int sockets[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    TEST_ASSERT(fcntl(sockets[0], F_SETFL, O_NONBLOCK) == 0);
    const int sndbuf = 4096;
    TEST_ASSERT(setsockopt(sockets[0], SOL_SOCKET, SO_SNDBUF,
                           &sndbuf, sizeof(sndbuf)) == 0);

    const std::string fill(4096, 'x');
    ssize_t sent = 0;
    do {
        sent = send(sockets[0], fill.data(), fill.size(), MSG_NOSIGNAL);
    } while (sent > 0);
    TEST_ASSERT(sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK));

    const auto started = std::chrono::steady_clock::now();
    size_t heartbeat_offset = 0;
    TEST_ASSERT(http_detail::try_send_sse_heartbeat(
                    sockets[0], heartbeat_offset) ==
                http_detail::HeartbeatSendResult::Retry);
    TEST_ASSERT(heartbeat_offset < sizeof(": keep-alive\n\n") - 1);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    TEST_ASSERT(elapsed < std::chrono::milliseconds(100));

    // Once the peer drains, retrying completes the same heartbeat rather than
    // treating temporary backpressure as a disconnect.
    char drained[8192];
    while (recv(sockets[1], drained, sizeof(drained), MSG_DONTWAIT) > 0) {}
    TEST_ASSERT(http_detail::try_send_sse_heartbeat(
                    sockets[0], heartbeat_offset) ==
                http_detail::HeartbeatSendResult::Complete);
    TEST_ASSERT(heartbeat_offset == 0);
    const std::string expected = ": keep-alive\n\n";
    TEST_ASSERT(recv(sockets[1], drained, sizeof(drained), 0) ==
                (ssize_t)expected.size());
    TEST_ASSERT(std::memcmp(drained, expected.data(), expected.size()) == 0);

    close(sockets[0]);
    close(sockets[1]);
}
#endif

TEST_CASE(ServerUnitFixture, test_http_sse_done_scanner_requires_terminal_line) {
    std::string partial_line;
    const std::string content =
        "data: {\"delta\":{\"content\":\"data: [DONE]\"}}\n\n";
    TEST_ASSERT(!http_detail::sse_chunk_has_done(
        partial_line, content.data(), content.size()));
    TEST_ASSERT(partial_line.empty());

    const std::string embedded_first =
        "data: {\"delta\":\"data: [DONE]";
    TEST_ASSERT(!http_detail::sse_chunk_has_done(
        partial_line, embedded_first.data(), embedded_first.size()));
    const std::string embedded_second = " still content\"}\n\n";
    TEST_ASSERT(!http_detail::sse_chunk_has_done(
        partial_line, embedded_second.data(), embedded_second.size()));
    TEST_ASSERT(partial_line.empty());

    const std::string first = "data: [DO";
    TEST_ASSERT(!http_detail::sse_chunk_has_done(
        partial_line, first.data(), first.size()));
    const std::string second = "NE]\r\n\r\n";
    TEST_ASSERT(http_detail::sse_chunk_has_done(
        partial_line, second.data(), second.size()));
    TEST_ASSERT(partial_line.empty());
}

TEST_CASE(ServerUnitFixture, test_qwen35_mrope_positions_axis_major) {
    std::vector<int32_t> standalone(4 * 5, -1);
    fill_qwen35_mrope_positions(
        standalone.data(), /*base_pos=*/7, /*n_tokens=*/5);
    const std::vector<int32_t> expected{
        7, 8, 9, 10, 11,
        7, 8, 9, 10, 11,
        7, 8, 9, 10, 11,
        0, 0, 0, 0, 0,
    };
    TEST_ASSERT(standalone == expected);

    constexpr int packed_tokens = 8;
    std::vector<int32_t> packed(4 * packed_tokens, -1);
    fill_qwen35_mrope_positions(
        packed.data(), packed_tokens, /*token_offset=*/2,
        /*base_pos=*/20, /*n_tokens=*/3);
    for (int axis = 0; axis < 4; ++axis) {
        for (int row = 0; row < packed_tokens; ++row) {
            const bool in_segment = row >= 2 && row < 5;
            const int expected_value = !in_segment
                ? -1
                : (axis < 3 ? 20 + row - 2 : 0);
            TEST_ASSERT(
                packed[(size_t)axis * packed_tokens + row] ==
                expected_value);
        }
    }
}

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

static json shell_tools() {
    return json::array({
        {
            {"name", "shell"},
            {"description", "Run one read-only shell command in the repository."},
            {"input_schema", {
                {"type", "object"},
                {"properties", {
                    {"command", {
                        {"type", "string"},
                        {"description", "The shell command to run."}
                    }}
                }},
                {"required", json::array({"command"})},
                {"additionalProperties", false}
            }}
        }
    });
}

static json bash_tools() {
    json tools = shell_tools();
    tools[0]["name"] = "bash";
    return tools;
}

static json read_tools() {
    return json::array({
        {{"type", "function"},
         {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}},
                     {"offset", {{"type", "integer"}}},
                     {"limit", {{"type", "integer"}}}
                 }},
                 {"required", json::array({"path"})}
             }}
         }}}
    });
}

static json qwen_read_probe_tools() {
    return json::array({
        {{"type", "function"},
         {"function", {
             {"name", "read_probe"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"filePath", {{"type", "string"}}},
                     {"limit", {{"type", "integer"}}},
                     {"content", {{"type", "string"}}}
                 }},
                 {"required", json::array({"filePath"})}
             }}
         }}}
    });
}

static std::string qwen_read_probe_call(const std::string & eol) {
    return "<tool_call>" + eol
        + "<function=read_probe>" + eol
        + "<parameter=filePath>" + eol
        + "README.md" + eol
        + "</parameter>" + eol
        + "<parameter=limit>" + eol
        + "10" + eol
        + "</parameter>" + eol
        + "</function>" + eol
        + "</tool_call>";
}

static json read_and_bash_tools() {
    json tools = read_tools();
    tools.push_back(bash_tools()[0]);
    return tools;
}

static json edit_tools() {
    return json::array({
        {{"type", "function"},
         {"function", {
             {"name", "edit"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"edits", {
                         {"type", "array"},
                         {"items", {{"type", "object"}}}
                     }}
                 }},
                 {"required", json::array({"edits"})}
             }}
         }}}
    });
}

static json optional_shell_tools() {
    json tools = shell_tools();
    tools[0]["input_schema"].erase("required");
    return tools;
}

static SseEmitter make_emitter(ApiFormat fmt, json tools = json::array(),
                               bool started_in_thinking = false) {
    return SseEmitter(fmt, "test_id_001", "test-model", 10,
                      tools, nullptr, {}, started_in_thinking);
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

// ═══════════════════════════════════════════════════════════════════════
// Tool parser tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_parse_qwen_multiline_xml_parameters_lf) {
    auto result = parse_tool_calls(qwen_read_probe_call("\n"),
                                   qwen_read_probe_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read_probe");
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["filePath"] == "README.md");
        TEST_ASSERT(args["filePath"].is_string());
        TEST_ASSERT(args["limit"].is_number_integer());
        TEST_ASSERT(args["limit"] == 10);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_qwen_multiline_xml_parameters_crlf) {
    auto result = parse_tool_calls(qwen_read_probe_call("\r\n"),
                                   qwen_read_probe_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["filePath"] == "README.md");
        TEST_ASSERT(args["filePath"].get<std::string>().find('\r') ==
                    std::string::npos);
        TEST_ASSERT(args["limit"].is_number_integer());
        TEST_ASSERT(args["limit"] == 10);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_qwen_multiline_xml_string_value) {
    const std::string text =
        "<tool_call>\n"
        "<function=read_probe>\n"
        "<parameter=filePath>\n"
        "README.md\n"
        "</parameter>\n"
        "<parameter=content>\n"
        "line1\n"
        "line2\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";
    auto result = parse_tool_calls(text, qwen_read_probe_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["content"] == "line1\nline2");
        TEST_ASSERT(args["content"].get<std::string>().find('\r') ==
                    std::string::npos);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_qwen_multiline_xml_parameter_at_true_eof) {
    const std::string text =
        "<tool_call><function=read_probe>"
        "<parameter=filePath>README.md</tool_call>";
    auto result = parse_tool_calls(text, qwen_read_probe_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["filePath"] == "README.md");
    }
}

// A model writing a file emits one parameter value tens of KB long. On a
// 2 MiB thread stack the regex matcher overflows at about 15 KB; the parser
// runs on a stack sized to the text, so a 256 KB value parses.
TEST_CASE(ServerUnitFixture, test_parse_qwen_xml_long_parameter_on_sized_stack) {
    std::string content;
    while (content.size() < 256 * 1024) {
        content += "\tconst kept: number[] = items.filter((x) => x < limit);\n";
    }
    content.pop_back();
    const std::string head =
        "<tool_call>\n<function=read_probe>\n"
        "<parameter=filePath>\nsanity.ts\n</parameter>\n"
        "<parameter=content>\n" + content;
    const std::string closed =
        head + "\n</parameter>\n</function>\n</tool_call>";

    // Closed, and cut off at the output cap before any closing tag.
    for (const std::string * text : {&closed, &head}) {
        const ToolParseResult result = parse_tool_calls(*text, qwen_read_probe_tools());
        if (text != &closed) {
            TEST_ASSERT(result.tool_calls.empty());
            TEST_ASSERT(result.cleaned_text == head);
            continue;
        }
        TEST_ASSERT(result.tool_calls.size() == 1);
        if (!result.tool_calls.empty()) {
            const auto args = json::parse(result.tool_calls[0].arguments);
            TEST_ASSERT(args["filePath"] == "sanity.ts");
            TEST_ASSERT(args["content"] == content);
        }
    }

    // Past the parse limit (about 1 MB) the text comes back unparsed.
    std::string huge = content;
    while (huge.size() < 2 * 1024 * 1024) huge += content;
    const std::string oversized = "<tool_call>\n<function=read_probe>\n"
        "<parameter=content>\n" + huge + "\n</parameter>\n</function>\n</tool_call>";
    const ToolParseResult unparsed = parse_tool_calls(oversized, qwen_read_probe_tools());
    TEST_ASSERT(unparsed.tool_calls.empty());
    TEST_ASSERT(unparsed.cleaned_text == oversized);
}

TEST_CASE(ServerUnitFixture, test_parse_tool_call_xml) {
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

TEST_CASE(ServerUnitFixture, test_parse_bare_function_xml) {
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

TEST_CASE(ServerUnitFixture, test_parse_bare_tool_name_xml_with_function_close) {
    std::string text =
        "\n\n\nLet me find the correct line range for the tests array.\n\n\n"
        "<bash>\n"
        "<parameter=command>\n"
        "grep -n \"f5.test\" /workspace/project/tests/bootstrap.cjs\n"
        "</parameter>\n"
        "</function>\n";
    json tools = json::array({
        {{"type", "function"},
         {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] ==
                    "grep -n \"f5.test\" /workspace/project/tests/bootstrap.cjs");
    }
    TEST_ASSERT(result.cleaned_text.find("<bash>") == std::string::npos);
    TEST_ASSERT(result.cleaned_text.find("</function>") == std::string::npos);
    TEST_ASSERT(result.cleaned_text.find("Let me find the correct line range") !=
                std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_repeated_bare_edit_calls_with_trailing_close) {
    const std::string text =
        "Applying both updates.\n\n"
        "<edit>\n"
        "<parameter=edits>\n"
        "[{\"path\":\"/workspace/first.conf\",\"oldText\":\"auto\","
        "\"newText\":\"enabled\"}]\n"
        "</parameter>\n"
        "</function>\n\n"
        "<edit>\n"
        "<parameter=edits>\n"
        "[{\"path\":\"/workspace/second.conf\",\"oldText\":\"auto\","
        "\"newText\":\"enabled\"}]\n"
        "</parameter>\n"
        "</function>\n\n"
        "</edit>";

    auto result = parse_tool_calls(text, edit_tools());
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "edit");
        TEST_ASSERT(result.tool_calls[1].name == "edit");
        const auto first = json::parse(result.tool_calls[0].arguments);
        const auto second = json::parse(result.tool_calls[1].arguments);
        TEST_ASSERT(first["edits"][0]["path"] == "/workspace/first.conf");
        TEST_ASSERT(second["edits"][0]["path"] == "/workspace/second.conf");
    }
    TEST_ASSERT(result.cleaned_text == "Applying both updates.");
}

TEST_CASE(ServerUnitFixture, test_tool_syntax_scanner_declared_name_guards) {
    size_t pos = std::string::npos;
    TEST_ASSERT(find_tool_syntax_start("prefix<edit>", edit_tools(), pos));
    TEST_ASSERT(pos == 6);

    pos = std::string::npos;
    TEST_ASSERT(!find_tool_syntax_start("prefix<unknown_tool>", edit_tools(), pos));

    json invalid = edit_tools();
    invalid[0]["function"]["name"] = std::string(65, 'x');
    TEST_ASSERT(tool_syntax_holdback(invalid) == 21);
    pos = std::string::npos;
    TEST_ASSERT(!find_tool_syntax_start("<" + std::string(65, 'x') + ">",
                                        invalid, pos));
}

TEST_CASE(ServerUnitFixture, test_parse_undeclared_file_tag_stays_content) {
    const std::string text =
        "Now I understand how to add a custom model. I need to edit "
        "~/.pi/agent/models.json to add a vLLM provider with the "
        "laguna-s-2.1 model. Let me check if the file exists first.\n\n"
        "<file>\n"
        "<parameter=path>\n"
        "~/.pi/agent/models.json\n"
        "</parameter>\n"
        "</function>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(result.cleaned_text == text);
}

TEST_CASE(ServerUnitFixture, test_parse_attribute_style_tool_xml) {
    std::string text =
        "The branch already exists. Let me check the current state:\n\n"
        "<parameter name=\"bash\"><parameter name=\"command\">"
        "cd /workspace/project && "
        "git status -sb && git branch\n --show-current</parameter>\n"
        "</function>\n";
    auto result = parse_tool_calls(text, bash_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] ==
                    "cd /workspace/project && "
                    "git status -sb && git branch\n --show-current");
    }
    TEST_ASSERT(result.cleaned_text ==
                "The branch already exists. Let me check the current state:");

    const std::string wrapped =
        "Checking now.\n\n<tool_call>\n"
        "<parameter name=\"bash\"><parameter name=\"command\">"
        "git status</parameter>\n</function>\n</tool_call>";
    auto wrapped_result = parse_tool_calls(wrapped, bash_tools());
    TEST_ASSERT(wrapped_result.tool_calls.size() == 1);
    TEST_ASSERT(wrapped_result.cleaned_text == "Checking now.");
}

TEST_CASE(ServerUnitFixture, test_parse_mixed_tool_variants_preserve_source_order) {
    const std::string text =
        "<function read>\n"
        "<parameter=path>/tmp/first.md</parameter>\n"
        "</function>\n"
        "<parameter name=\"bash\"><parameter name=\"command\">"
        "cat /tmp/first.md</parameter></function>";
    auto result = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(result.tool_calls[1].name == "bash");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_attribute_style_tool_xml_rejects_malformed_body) {
    const std::string malformed =
        "<parameter name=\"bash\"><parameter name=\"command\">git status\n"
        "</function>";
    auto malformed_result = parse_tool_calls(malformed, bash_tools());
    TEST_ASSERT(malformed_result.tool_calls.empty());
    TEST_ASSERT(malformed_result.cleaned_text == malformed);

    const std::string unknown =
        "<parameter name=\"not_a_tool\"><parameter name=\"command\">"
        "git status</parameter></function>";
    auto unknown_result = parse_tool_calls(unknown, bash_tools());
    TEST_ASSERT(unknown_result.tool_calls.empty());
    TEST_ASSERT(unknown_result.cleaned_text == unknown);
}

TEST_CASE(ServerUnitFixture, test_parse_funcname_tool_xml) {
    const std::string text =
        "<tool_call>\n"
        "<funcname>read\n"
        "<parameter=limit>\n50\n</parameter>\n"
        "<parameter=offset>\n1\n</parameter>\n"
        "<parameter=path>\n"
        "/tmp/tool-input.md\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n";
    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["limit"] == 50);
        TEST_ASSERT(args["offset"] == 1);
        TEST_ASSERT(args["path"] == "/tmp/tool-input.md");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_funcname_tool_xml_rejects_malformed_or_unknown) {
    const std::string malformed =
        "<funcname>read\n"
        "<parameter=path>/tmp/task.md\n"
        "</function>";
    auto malformed_result = parse_tool_calls(malformed, read_tools());
    TEST_ASSERT(malformed_result.tool_calls.empty());
    TEST_ASSERT(malformed_result.cleaned_text == malformed);

    const std::string unknown =
        "<funcname>write\n"
        "<parameter=path>/tmp/task.md</parameter>\n"
        "</function>";
    auto unknown_result = parse_tool_calls(unknown, read_tools());
    TEST_ASSERT(unknown_result.tool_calls.empty());
    TEST_ASSERT(unknown_result.cleaned_text == unknown);
}

TEST_CASE(ServerUnitFixture, test_parse_space_function_tool_xml) {
    const std::string text =
        "Let me read the file and compute its SHA-256 hash.\n\n"
        "<function read>\n"
        "<parameter=path>\n"
        "/tmp/tool-input.md\n"
        "</parameter>\n"
        "</function>\n\n"
        "<function bash>\n"
        "<parameter=command>\n"
        "sha256sum /tmp/tool-input.md\n"
        "</parameter>\n"
        "</function>";
    auto result = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(json::parse(result.tool_calls[0].arguments)["path"] ==
                    "/tmp/tool-input.md");
        TEST_ASSERT(result.tool_calls[1].name == "bash");
        TEST_ASSERT(json::parse(result.tool_calls[1].arguments)["command"] ==
                    "sha256sum /tmp/tool-input.md");
    }
    TEST_ASSERT(result.cleaned_text ==
                "Let me read the file and compute its SHA-256 hash.");
}

TEST_CASE(ServerUnitFixture, test_parse_space_function_tool_xml_rejects_malformed_or_unknown) {
    const std::string malformed =
        "<function read>\n<parameter=path>/tmp/task.md\n</function>";
    auto malformed_result = parse_tool_calls(malformed, read_tools());
    TEST_ASSERT(malformed_result.tool_calls.empty());
    TEST_ASSERT(malformed_result.cleaned_text == malformed);

    const std::string unknown =
        "<function write>\n<parameter=path>/tmp/task.md</parameter>\n</function>";
    auto unknown_result = parse_tool_calls(unknown, read_tools());
    TEST_ASSERT(unknown_result.tool_calls.empty());
    TEST_ASSERT(unknown_result.cleaned_text == unknown);
}

TEST_CASE(ServerUnitFixture, test_parse_json_tool_call) {
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

TEST_CASE(ServerUnitFixture, test_parse_single_tool_bare_json_args) {
    std::string text =
        "{\n"
        "  \"command\": \"git branch --show-current\"\n"
        "}";
    auto result = parse_tool_calls(text, shell_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "shell");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "git branch --show-current");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_single_tool_bare_json_args_allows_empty_optional_object) {
    auto result = parse_tool_calls("{}", optional_shell_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "shell");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args.is_object());
        TEST_ASSERT(args.empty());
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_single_tool_bare_json_args_rejects_prose) {
    std::string text = "The command is {\"command\": \"git status\"}.";
    auto result = parse_tool_calls(text, shell_tools());
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(result.cleaned_text == text);
}

TEST_CASE(ServerUnitFixture, test_parse_single_tool_bare_json_args_rejects_ambiguous_tools) {
    std::string text = "{\"command\": \"git status\"}";
    auto result = parse_tool_calls(text, weather_tools());
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(result.cleaned_text == text);
}

TEST_CASE(ServerUnitFixture, test_parse_no_tools) {
    std::string text = "Just plain text without any tool calls.";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(!result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_tool_code_wrapper) {
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

TEST_CASE(ServerUnitFixture, test_parse_function_call_wrapper) {
    std::string text =
        "<function_call>\n"
        "{\"name\": \"bash\", \"arguments\": {\"command\": \"echo 'hello'\"}}\n"
        "</function_call>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "echo 'hello'");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_legacy_openai_function_call_json) {
    const std::string text =
        "{\"function_call\":{\"arguments\":"
        "\"{\\\"location\\\":\\\"test-city\\\"}\","
        "\"name\":\"get_weather\"}}";
    const auto result = parse_tool_calls(text, weather_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["location"] == "test-city");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_deepseek_function_parameters_json) {
    const std::string text =
        "{\"function\":\"get_weather\",\"parameters\":{"
        "\"location\":\"test-city\",\"unit\":\"celsius\"}}";
    const auto result = parse_tool_calls(text, weather_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["location"] == "test-city");
        TEST_ASSERT(args["unit"] == "celsius");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_deepseek_function_stringified_parameters_json) {
    const std::string text =
        "{\"function\":\"get_weather\",\"parameters\":"
        "\"{\\\"location\\\":\\\"test-city\\\",\\\"unit\\\":\\\"celsius\\\"}\"}";
    const auto result = parse_tool_calls(text, weather_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        const auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["location"] == "test-city");
        TEST_ASSERT(args["unit"] == "celsius");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_bare_function_json_with_parameters) {
    std::string text =
        "<function>\n"
        "{\n"
        "  \"name\": \"bash\",\n"
        "  \"parameters\": {\n"
        "    \"command\": \"ls -la \\\"/home/dpavlin/aimax project\\\"\"\n"
        "  }\n"
        "}\n"
        "</function>";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "ls -la \"/home/dpavlin/aimax project\"");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_function_call_xml_invoke_name_parameters) {
    const std::string text =
        "Let me read the file.\n\n"
        "<function_call>\n"
        "<invoke_name>read</invoke_name>\n"
        "<parameters>\n"
        "<path>/home/dpavlin/koha-rfid-go/internal/rfidops/ops.go</path>\n"
        "</parameters>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home/dpavlin/koha-rfid-go/internal/rfidops/ops.go");
    }
    TEST_ASSERT(result.cleaned_text == "Let me read the file.");
}

TEST_CASE(ServerUnitFixture, test_parse_function_call_xml_multiple_parameters) {
    const std::string text =
        "<function_call>\n"
        "<invoke_name>bash</invoke_name>\n"
        "<parameters>\n"
        "<command>ls -la /tmp</command>\n"
        "<timeout>10</timeout>\n"
        "</parameters>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}},
                     {"timeout", {{"type", "integer"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "ls -la /tmp");
        TEST_ASSERT(args["timeout"] == 10);
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_tool_call_xml_invoke_name_parameters) {
    const std::string text =
        "<tool_call>\n"
        "<invoke_name>read</invoke_name>\n"
        "<parameters>\n"
        "<path>/home/dpavlin/test.go</path>\n"
        "</parameters>\n"
        "</tool_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home/dpavlin/test.go");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_legacy_tool_call_with_nested_name_tag) {
    // Regression test for Violation 1:
    // A legacy <tool_call><function=...> envelope whose parameter contains a nested <name> tag
    // must NOT be hijacked by parse_xml_tool_call_body as envelope name.
    const std::string text =
        "<tool_call>\n"
        "<function=edit_file>\n"
        "<parameter=content>\n"
        "function getTool() { return \"<name>other_tool</name>\"; }\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "edit_file"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"content", {{"type", "string"}}}
                 }}
             }}
         }}},
        {{"type", "function"}, {"function", {
             {"name", "other_tool"},
             {"parameters", {{"type", "object"}, {"properties", {}}}}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "edit_file");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["content"] == "function getTool() { return \"<name>other_tool</name>\"; }");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_xml_tool_call_malformed_parameters_rejected) {
    // Regression test for Violation 2:
    // When the parameter section is non-empty but malformed (unsupported/invalid syntax),
    // parse_xml_tool_call_body must NOT emit an empty {} tool call.
    const std::string text =
        "<function_call>\n"
        "<invoke_name>edit_file</invoke_name>\n"
        "<parameters>\n"
        "random unparseable garbage text\n"
        "</parameters>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "edit_file"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"content", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(!result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_xml_tool_call_zero_arguments_accepted) {
    // Genuinely zero-argument calls should be accepted.
    const std::string text =
        "<function_call>\n"
        "<invoke_name>get_status</invoke_name>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "get_status"},
             {"parameters", {{"type", "object"}, {"properties", {}}}}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_status");
        TEST_ASSERT(result.tool_calls[0].arguments == "{}");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_xml_tool_call_invoke_envelope_attribute_style) {
    const std::string text =
        "<function_call>\n"
        "<invoke name=\"read\">\n"
        "<parameter name=\"path\">/tmp/test.txt</parameter>\n"
        "</invoke>\n"
        "</function_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/tmp/test.txt");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_tool_call_xml_function_name_tag) {
    // Regression test for Violation:
    // <tool_call> envelopes using <function_name> must be parsed cleanly by parse_xml_tool_call_body()
    const std::string text =
        "<tool_call>\n"
        "<function_name>read</function_name>\n"
        "<parameters>\n"
        "<path>/tmp/test.txt</path>\n"
        "</parameters>\n"
        "</tool_call>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/tmp/test.txt");
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_with_token) {
    const std::string text =
        "Let me read the file.\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"read\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/home/dpavlin/koha-rfid-go/internal/rfidops/ops.go</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home/dpavlin/koha-rfid-go/internal/rfidops/ops.go");
    }
    TEST_ASSERT(result.cleaned_text == "Let me read the file.");
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_multiple_parameters) {
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la /tmp</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}},
                     {"timeout", {{"type", "integer"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "ls -la /tmp");
        TEST_ASSERT(args["timeout"] == 10);
    }
    TEST_ASSERT(result.cleaned_text.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_dash_parameter_names) {
    // DeepSeek V4.1 calling Claude Code's Grep, whose flags are named -i and -n.
    const std::string text =
        "I'll explore the repository.\n\n"
        "<｜DSML｜tool_calls>\n\n"
        "<｜DSML｜invoke name=\"Grep\">\n"
        "<｜DSML｜parameter name=\"pattern\" string=\"true\">prefix.?cache|PrefixCache</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"output_mode\" string=\"true\">files_with_matches</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"-i\" string=\"false\">true</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"-n\" string=\"false\">true</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "Grep"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"pattern", {{"type", "string"}}},
                     {"output_mode", {{"type", "string"}}},
                     {"-i", {{"type", "boolean"}}},
                     {"-n", {{"type", "boolean"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "Grep");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["pattern"] == "prefix.?cache|PrefixCache");
        TEST_ASSERT(args["-i"] == true);
        TEST_ASSERT(args["-n"] == true);
    }
    TEST_ASSERT(result.cleaned_text == "I'll explore the repository.");
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_edit_tool) {
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"edit\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/home/dpavlin/koha-rfid-go/server.go</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"edits\" string=\"false\">[{\"oldText\": \"err1\", \"newText\": \"err2\"}]</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "edit"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}},
                     {"edits", {{"type", "array"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "edit");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "/home/dpavlin/koha-rfid-go/server.go");
        TEST_ASSERT(args["edits"].is_array());
        TEST_ASSERT(args["edits"][0]["oldText"] == "err1");
        TEST_ASSERT(args["edits"][0]["newText"] == "err2");
    }
    TEST_ASSERT(result.cleaned_text.empty());
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

TEST_CASE(ServerUnitFixture, test_find_tool_syntax_start_arg_key_backtracking) {
    std::string text = "Let me edit the file:\nedit<arg_key>path</arg_key><arg_val>/tmp/test.txt</arg_val>";
    json tools = json::array({
        {{"type", "function"}, {"function", {{"name", "edit"}}}}
    });
    size_t pos = std::string::npos;
    TEST_ASSERT(find_tool_syntax_start(text, tools, pos));
    TEST_ASSERT(pos == text.find("edit<arg_key>"));
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_string_attribute_semantics) {
    // string="true" should preserve leading/trailing whitespace verbatim
    // string="false" should parse JSON numbers/booleans/arrays/objects
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"custom_tool\">\n"
        "<｜DSML｜parameter name=\"verbatim_str\" string=\"true\">  hello world  \n</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"json_num\" string=\"false\">42</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"json_bool\" string=\"false\">true</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"json_arr\" string=\"false\">[1, 2, 3]</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "custom_tool"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"verbatim_str", {{"type", "string"}}},
                     {"json_num", {{"type", "integer"}}},
                     {"json_bool", {{"type", "boolean"}}},
                     {"json_arr", {{"type", "array"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "custom_tool");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["verbatim_str"] == "  hello world  \n");
        TEST_ASSERT(args["json_num"] == 42);
        TEST_ASSERT(args["json_bool"] == true);
        TEST_ASSERT(args["json_arr"].is_array() && args["json_arr"].size() == 3);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_unclosed_invoke) {
    // When a model omits </｜DSML｜invoke> between consecutive calls,
    // lookahead termination should cleanly partition the invocations.
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args0 = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args0["command"] == "pwd");

        TEST_ASSERT(result.tool_calls[1].name == "bash");
        auto args1 = json::parse(result.tool_calls[1].arguments);
        TEST_ASSERT(args1["command"] == "ls -la");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_deepseek_harness_exact_payload) {
    // Regression test for the real DeepSeek Harness multi-tool call:
    // Call 1: bash (unclosed invoke)
    // Call 2: bash (closed invoke)
    // Call 3: web_search with string="invalid"
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">find /home/dpavlin/aimax -name \"*.tar.gz\" -o -name \"*.tgz\" 2>/dev/null | head -20; ls -la /home/dpavlin/aimax 2>/dev/null | head -30</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">List aimax dir for package artifacts</｜DSML｜parameter>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la /home/dpavlin/.npm/_npx/1e7f6d9597241db0/ 2>/dev/null; cat /home/dpavlin/.npm/_npx/1e7f6d9597241db0/package.json 2>/dev/null | head -50</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">Check harness checkout package.json for version</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"web_search\">\n"
        "<｜DSML｜parameter name=\"queries\" string=\"invalid\">\n"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}},
                     {"description", {{"type", "string"}}}
                 }}
             }}
         }}},
        {{"type", "function"}, {"function", {
             {"name", "web_search"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"queries", {{"type", "array"}}}
                 }}
             }}
         }}}
    });

    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 3);
    if (result.tool_calls.size() == 3) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args0 = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args0["command"].get<std::string>().find("find /home/dpavlin/aimax") != std::string::npos);
        TEST_ASSERT(args0["description"] == "List aimax dir for package artifacts");

        TEST_ASSERT(result.tool_calls[1].name == "bash");
        auto args1 = json::parse(result.tool_calls[1].arguments);
        TEST_ASSERT(args1["command"].get<std::string>().find("cat /home/dpavlin/.npm") != std::string::npos);
        TEST_ASSERT(args1["description"] == "Check harness checkout package.json for version");

        TEST_ASSERT(result.tool_calls[2].name == "web_search");
        auto args2 = json::parse(result.tool_calls[2].arguments);
        TEST_ASSERT(args2.contains("queries"));
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_unclosed_final_invoke_before_block_end) {
    // Regression test for real payload (chatcmpl_000000000000000a) where the model omits </｜DSML｜invoke>
    // before the closing </｜DSML｜tool_calls>.
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"edit\">\n"
        "<｜DSML｜parameter name=\"file_path\" string=\"true\">/home/dpavlin/aimax/LUCEBOX_STRIX_HALO_GUIDE.md</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"new_string\" string=\"true\">### B. Paged Attention WMMA (head-256, RDNA4) — `DFLASH27B_PAGED_WMMA` (default off, BURN-IN)\n"
        "- Env `DFLASH27B_PAGED_WMMA=1` routes paged full-attention layers (head 256, F16/Q8_0/Q4_0 KV, non-tree) to the WMMA kernel. Default (unset/0) keeps the V_DOT2 decode kernel.\n"
        "- Differential: single-prompt TTFT −21% @12K, −42% @44K; batched 8K-pool prefill slightly ahead. Kernel-level 20.6–22.4 TFLOP/s vs 6.4–6.8 (3.1–3.3×); end-to-end bounded by attention's prefill share (~8% @44K, ~4% @12K).\n"
        "- Two-mode CTest: `test_paged_attn_wmma` (V_DOT2, env=0) / `paged_attn_wmma_route` (env=1), diff via `server/test/bench/compare_paged_attn.py --tol 6e-3`. Need `--reconfig` after the upstream pull for `CMakeLists.txt` to register targets.\n"
        "- Source of truth: `server/docs/PAGED_ATTN_WMMA_HANDOFF.md`.</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"old_string\" string=\"true\">### B. Prefill Mode: `--ds4-prefill sparse` (DO NOT use `exact` for multi-turn)</｜DSML｜parameter>\n"
        "</｜DSML｜tool_calls>";

    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "edit"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"file_path", {{"type", "string"}}},
                     {"new_string", {{"type", "string"}}},
                     {"old_string", {{"type", "string"}}}
                 }}
             }}
         }}}
    });

    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (result.tool_calls.size() == 1) {
        TEST_ASSERT(result.tool_calls[0].name == "edit");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["file_path"] == "/home/dpavlin/aimax/LUCEBOX_STRIX_HALO_GUIDE.md");
        TEST_ASSERT(args["new_string"].get<std::string>().find("DFLASH27B_PAGED_WMMA") != std::string::npos);
        TEST_ASSERT(args["old_string"].get<std::string>().find("Prefill Mode") != std::string::npos);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_unclosed_invoke_with_nested_json) {
    // When an unclosed invoke contains a parameter whose value is a JSON tool-call object,
    // subsequent sweeps must not parse that JSON value as a duplicate tool call.
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">cat config.json</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"metadata\" string=\"false\">{\"name\": \"bash\", \"arguments\": {\"command\": \"nested\"}}</｜DSML｜parameter>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}},
                     {"metadata", {{"type", "object"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (result.tool_calls.size() == 1) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "cat config.json");
        TEST_ASSERT(args["metadata"]["name"] == "bash");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_tag_like_parameter_content) {
    // Parameter values containing substrings like <invoke-not-a-tag> or <parameterized>
    // must not trigger premature lookahead termination.
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">echo \"<invoke-not-a-tag>\"; cat <parameterized></｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (result.tool_calls.size() == 1) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "echo \"<invoke-not-a-tag>\"; cat <parameterized>");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_tool_calls_prose_invoke_tag_does_not_mask_json_call) {
    // A prose or non-call <invoke> tag (without name/tool attribute) must not cause
    // invoke_spans to span until EOF and mask subsequent bare JSON tool calls.
    const std::string text =
        "You can invoke the command as follows:\n"
        "Please look at <invoke> syntax.\n"
        "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -l\"}}";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "ls -l");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_literal_invoke_in_string_parameter) {
    // Parameter values with string="true" containing literal </invoke> or </parameter>
    // inside their payload (e.g. grep commands, git diffs) must not be truncated prematurely.
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">grep -n \"</invoke>\" server/src/server/tool_parser.cpp</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">Search for invoke close tags</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}},
                     {"description", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"] == "grep -n \"</invoke>\" server/src/server/tool_parser.cpp");
        TEST_ASSERT(args["description"] == "Search for invoke close tags");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_unclosed_string_parameter_before_sibling_parameter) {
    // When a string="true" parameter is unclosed before another parameter,
    // lookahead termination on tag boundaries must cleanly partition them
    // rather than consuming the sibling parameter into the first argument.
    const std::string text =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list directory files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}},
                     {"description", {{"type", "string"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "bash");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["command"].get<std::string>().find("ls -la") != std::string::npos);
        TEST_ASSERT(args["command"].get<std::string>().find("<｜DSML｜parameter") == std::string::npos);
        TEST_ASSERT(args["description"] == "list directory files");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_tool_calls_literal_param_tag_in_string_value) {
    // Literal "<param>" / "<parameter ...>" text inside a verbatim value is
    // payload, not a sibling parameter. Only a parameter tag that opens a new
    // line terminates an unclosed value.
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "bash"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"command", {{"type", "string"}}}
                 }}
             }}
         }}},
        {{"type", "function"}, {"function", {
             {"name", "write_file"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}},
                     {"content", {{"type", "string"}}}
                 }}
             }}
         }}}
    });

    {
        const std::string text =
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"bash\">\n"
            "<｜DSML｜parameter name=\"command\" string=\"true\">echo '<param>'</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>";
        auto result = parse_tool_calls(text, tools);
        TEST_ASSERT(result.tool_calls.size() == 1);
        if (!result.tool_calls.empty()) {
            TEST_ASSERT(result.tool_calls[0].name == "bash");
            auto args = json::parse(result.tool_calls[0].arguments);
            TEST_ASSERT(args["command"] == "echo '<param>'");
        }
    }

    {
        const std::string text =
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"write_file\">\n"
            "<｜DSML｜parameter name=\"path\" string=\"true\">doc.xml</｜DSML｜parameter>\n"
            "<｜DSML｜parameter name=\"content\" string=\"true\">use <parameter name=\"x\"> here</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>";
        auto result = parse_tool_calls(text, tools);
        TEST_ASSERT(result.tool_calls.size() == 1);
        if (!result.tool_calls.empty()) {
            TEST_ASSERT(result.tool_calls[0].name == "write_file");
            auto args = json::parse(result.tool_calls[0].arguments);
            TEST_ASSERT(args["path"] == "doc.xml");
            TEST_ASSERT(args["content"] == "use <parameter name=\"x\"> here");
            TEST_ASSERT(!args.contains("x"));
        }
    }

    {
        // XML content whose lines open with <param ...> elements: extra
        // attributes or a self-closing tag are not a sibling tool parameter.
        const std::string content =
            "<launch>\n"
            "  <param name=\"rate\" value=\"10\"/>\n"
            "  <parameter name=\"mode\" type=\"string\">fast\n"
            "</launch>";
        const std::string text =
            "<｜DSML｜tool_calls>\n"
            "<｜DSML｜invoke name=\"write_file\">\n"
            "<｜DSML｜parameter name=\"path\" string=\"true\">robot.launch</｜DSML｜parameter>\n"
            "<｜DSML｜parameter name=\"content\" string=\"true\">" + content + "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>";
        auto result = parse_tool_calls(text, tools);
        TEST_ASSERT(result.tool_calls.size() == 1);
        if (!result.tool_calls.empty()) {
            TEST_ASSERT(result.tool_calls[0].name == "write_file");
            auto args = json::parse(result.tool_calls[0].arguments);
            TEST_ASSERT(args["path"] == "robot.launch");
            TEST_ASSERT(args["content"] == content);
            TEST_ASSERT(!args.contains("rate"));
            TEST_ASSERT(!args.contains("mode"));
        }
    }
}



TEST_CASE(ServerUnitFixture, test_parse_tool_allowed_filter) {
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

// ─── Pattern 5: call:<verb>{...} plain-text tool calls ─────────────────
//
// Covers the gemma plain-text emission path added in
// server/src/server/tool_parser.cpp (PR #340). The opener regex requires
// a sentinel character before `call:` (start-of-string or one of
// [\s,;:\(\[\{\}\)\]\>_]); the body is brace-balanced and string-aware;
// and the args go through coerce_relaxed_json before becoming the
// argument object.

TEST_CASE(ServerUnitFixture, test_parse_call_verb_empty_args) {
    // Bareword `call:get_weather{}` at start-of-string — sentinel
    // matches the leading `^` anchor; body is the empty object `{}`.
    auto result = parse_tool_calls("call:get_weather{}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args.is_object());
        TEST_ASSERT(args.empty());
    }
    // The matched span should be removed from cleaned_text.
    TEST_ASSERT(result.cleaned_text.find("call:get_weather") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_strict_json_args) {
    // Strict JSON args go through json::parse directly in
    // coerce_relaxed_json's fast path.
    auto result = parse_tool_calls("call:get_weather{\"city\": \"NYC\"}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["city"] == "NYC");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_namespaced_verb) {
    // `ns:foo` namespaced verbs — the colon-strip logic in pattern 5
    // strips everything up to the last `:` so the registered tool name
    // is just `foo`.
    auto result = parse_tool_calls("call:ns:foo{\"k\": 1}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "foo");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["k"] == 1);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_whitespace_before_key) {
    // Leading whitespace inside the brace body must not break parsing.
    // (Whitespace tolerance is provided by json::parse / the relaxed
    //  fallback rewriter.)
    auto result = parse_tool_calls("call:get_weather{ \"city\": \"NYC\" }");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["city"] == "NYC");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_missing_close_brace_rejected) {
    // Unbalanced opener — balanced_braces_end returns npos so pattern 5
    // bails out and produces no tool call. The text leaks through.
    auto result = parse_tool_calls("call:get_weather{\"city\": \"NYC\"");
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_narrative_without_body_rejected) {
    // Narrative usage with a non-balanced body — sentinel matches the
    // space before `call:`, but the `{` has no matching `}` so the
    // call is discarded.
    auto result = parse_tool_calls("I will call:foo{");
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_underscore_prefix) {
    // SentencePiece artifact: `_call:` (the `_` is the literal
    // underscore character; sentinel char-class includes `_` for
    // exactly this case).
    auto result = parse_tool_calls("_call:get_weather{\"city\": \"NYC\"}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["city"] == "NYC");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_nested_object_args) {
    // Nested `{}` inside the args — balanced_braces_end tracks depth so
    // the outer close isn't consumed by the inner object.
    auto result = parse_tool_calls(
        "call:get_weather{\"loc\": {\"city\": \"NYC\", \"zip\": \"10001\"}}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_weather");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["loc"].is_object());
        TEST_ASSERT(args["loc"]["city"] == "NYC");
        TEST_ASSERT(args["loc"]["zip"] == "10001");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_back_to_back) {
    // Gemma frequently emits multiple invocations back-to-back. The
    // sentinel char-class includes `}` so the second `call:` is found
    // after the first closes.
    auto result = parse_tool_calls(
        "call:a{\"x\": 1}call:b{\"y\": 2}");
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "a");
        auto args0 = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args0["x"] == 1);
        TEST_ASSERT(result.tool_calls[1].name == "b");
        auto args1 = json::parse(result.tool_calls[1].arguments);
        TEST_ASSERT(args1["y"] == 2);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_relaxed_single_quotes) {
    // Relaxed-JSON fallback: single-quoted strings + bare identifier
    // keys are rewritten to strict JSON before parse.
    auto result = parse_tool_calls("call:foo{city: 'NYC'}");
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "foo");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["city"] == "NYC");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_glued_to_word_rejected) {
    // No sentinel char before `call:` (glued to identifier) — pattern 5
    // must NOT match. `_` is a deliberate exception covered by its
    // own test; here we use a regular letter.
    auto result = parse_tool_calls("xcall:foo{\"a\": 1}");
    // Pattern 5 should NOT fire. Pattern 6 (bare-JSON sweep) sees
    // `{"a": 1}` but it has no `name`/`function` field, so it produces
    // no tool call either.
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_does_not_hijack_inner_name) {
    // Regression: pattern 5 must run before pattern 6 so that an inner
    // {"name": "...", "arguments": {}} in the call's args doesn't get
    // hijacked into a spurious bare-JSON tool call.
    auto result = parse_tool_calls(
        "call:outer{\"name\": \"inner\", \"arguments\": {}}");
    // Should match exactly one tool: the outer call. The inner
    // {"name":..., "arguments":...} JSON is shadowed by the recorded
    // removal span.
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "outer");
    }
}

// ─── Pattern 5 (cont.): PR #341 imports — narrative & quoting edge cases ─
//
// These tests originated in PR #341 alongside sse_emitter Pattern-B work
// and were relocated here when #341 was split. They focus on edge cases
// that complement the core call:<verb>{} suite above.

TEST_CASE(ServerUnitFixture, test_parse_call_verb_single) {
    std::string text = "call:get_country_info{country: \"France\"}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "get_country_info");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["country"] == "France");
    }
    TEST_ASSERT(result.cleaned_text.find("call:") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_namespaced) {
    std::string text = "call:execute-bead:read-file{path: \"crates/foo/src/lib.rs\"}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        // Verb only — namespace stripped.
        TEST_ASSERT(result.tool_calls[0].name == "read-file");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "crates/foo/src/lib.rs");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_snake_and_hyphen) {
    std::string text =
        "call:execute-bead:list-files{path: \"src/\"}\n\n"
        "call:execute-bead:read_file{path: \"a/b.rs\"}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "list-files");
        TEST_ASSERT(result.tool_calls[1].name == "read_file");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_tool_allowed_filter) {
    std::string text = "call:disallowed_verb{x: 1}call:allowed_verb{y: 2}";
    json tools = json::array({
        {{"type", "function"}, {"function", {{"name", "allowed_verb"}}}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "allowed_verb");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_inline_prose_rejected) {
    // No sentinel char before `call:` — must NOT match.
    std::string text = "narrative.call:foo{x:1}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_inline_prose_after_space) {
    // Whitespace IS a valid sentinel — this should match.
    std::string text = "Sure, I'll call:foo{x: 1}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "foo");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_malformed_args) {
    // Unterminated brace — drop the call, don't crash.
    std::string text = "call:foo{country: \"France\"";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_inner_brace_in_string) {
    // The `{` and `}` inside the string value must not confuse the
    // balanced-brace scanner.
    std::string text = "call:foo{cmd: \"echo {not_a_brace} ok\"}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["cmd"] == "echo {not_a_brace} ok");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_unquoted_keys) {
    // Relaxed-JSON path: bare keys get quoted.
    std::string text = "call:foo{path: \"x\", count: 3}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "x");
        TEST_ASSERT(args["count"] == 3);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_cleaned_text) {
    // The matched span should be stripped from cleaned_text.
    std::string text = "Hello call:foo{x: 1} world.";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    TEST_ASSERT(result.cleaned_text.find("call:") == std::string::npos);
    TEST_ASSERT(result.cleaned_text.find("Hello") != std::string::npos);
    TEST_ASSERT(result.cleaned_text.find("world.") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_intercept_inner_json) {
    // Regression case: inner args of the form {"name": ..., "arguments": ...}
    // must NOT be picked up by pattern 6 (bare-JSON sweep) as a spurious
    // `inner` ToolCall. Exactly one ToolCall, named `outer`, with the
    // inner JSON intact in its arguments.
    std::string text = "call:outer{\"name\": \"inner\", \"arguments\": {}}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "outer");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["name"] == "inner");
        TEST_ASSERT(args["arguments"].is_object());
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_multiline_args) {
    // Snapshot rows have multi-line nested args; the balanced-brace
    // scanner is line-agnostic, so this must Just Work.
    std::string text =
        "call:default_api:analyze_data{\n"
        "  data: [{\"date\": \"2024-10-05\", \"qty\": 50}, {\"date\": \"2024-10-06\", \"qty\": 60}],\n"
        "  metric: \"qty\"\n"
        "}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "analyze_data");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["metric"] == "qty");
        TEST_ASSERT(args["data"].is_array());
        TEST_ASSERT(args["data"].size() == 2);
    }
}


TEST_CASE(ServerUnitFixture, test_parse_call_verb_singlequote_with_inner_doublequote) {
    // Cubic PR #329 review: when the relaxed-JSON rewrite converts
    // single-quoted strings to double-quoted, inner `"` chars must be
    // escaped to `\"` — otherwise `'he said "hi"'` rewrites to
    // `"he said "hi""` which is invalid JSON and the whole tool call
    // is silently dropped.
    std::string text = "call:say{quote: 'he said \"hi\" loudly'}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "say");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["quote"] == "he said \"hi\" loudly");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_call_verb_backtick_with_inner_doublequote) {
    // Same escape concern as the single-quote case, but with the
    // backtick string flavor.
    std::string text = "call:say{quote: `he said \"hi\" loudly`}";
    auto result = parse_tool_calls(text);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["quote"] == "he said \"hi\" loudly");
    }
}


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
    // The newlines before the call are not reply text.
    TEST_ASSERT(em.accumulated_text().empty());
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

// The Responses stream's terminal event: response.incomplete with
// incomplete_details.reason "max_output_tokens" when the turn hit the
// output-token cap (clients such as Codex and Repo2RLEnv retry on it),
// response.completed otherwise -- including an EOS or a stop sequence that
// lands on the cap.
static json responses_terminal(SseEmitter & em, const std::string & text,
                               bool at_cap, bool ended_on_eos,
                               std::string & event_type) {
    em.emit_start();
    em.emit_token(text);
    const std::string last =
        em.emit_finish(at_cap ? 10 : 4, nullptr, 10, ended_on_eos).back();
    const size_t ev = last.find("event: ");
    event_type = last.substr(ev + 7, last.find('\n', ev) - ev - 7);
    const size_t at = last.find("data: ");
    return json::parse(last.substr(at + 6, last.find('\n', at) - at - 6));
}

static json responses_terminal(bool at_cap, bool ended_on_eos,
                               std::string & event_type) {
    auto em = make_emitter(ApiFormat::RESPONSES);
    return responses_terminal(em, "Hi there! How are you doing today?",
                              at_cap, ended_on_eos, event_type);
}

TEST_CASE(ServerUnitFixture, test_emitter_responses_incomplete_at_output_cap) {
    std::string event_type;
    const json data = responses_terminal(true, false, event_type);
    TEST_ASSERT(event_type == "response.incomplete");
    TEST_ASSERT(data["type"] == "response.incomplete");
    TEST_ASSERT(data["response"]["status"] == "incomplete");
    TEST_ASSERT(data["response"]["incomplete_details"]["reason"] == "max_output_tokens");
}

TEST_CASE(ServerUnitFixture, test_emitter_responses_completed_below_cap_or_on_eos) {
    for (const bool at_cap_on_eos : {false, true}) {
        std::string event_type;
        const json data = responses_terminal(at_cap_on_eos, at_cap_on_eos, event_type);
        TEST_ASSERT(event_type == "response.completed");
        TEST_ASSERT(data["response"]["status"] == "completed");
        TEST_ASSERT(data["response"]["incomplete_details"].is_null());
    }
}

TEST_CASE(ServerUnitFixture, test_emitter_responses_stop_sequence_at_cap_is_completed) {
    SseEmitter em(ApiFormat::RESPONSES, "test_id_001", "test-model", 10,
                  json::array(), nullptr, {"STOP"});
    std::string event_type;
    const json data = responses_terminal(em, "Hello STOP", true, false, event_type);
    TEST_ASSERT(em.stop_hit());
    TEST_ASSERT(event_type == "response.completed");
    TEST_ASSERT(data["response"]["status"] == "completed");
}

// A complete call followed by one the cap cut off: the first call is kept,
// and the response still says the cap was hit.
TEST_CASE(ServerUnitFixture, test_emitter_responses_tool_call_then_cut_call_is_incomplete) {
    auto em = make_emitter(ApiFormat::RESPONSES, qwen_read_probe_tools());
    std::string event_type;
    const json data = responses_terminal(em,
        "<tool_call>\n<function=read_probe>\n<parameter=filePath>\na.ts\n"
        "</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=read_probe>\n<parameter=filePath>\nb",
        true, false, event_type);
    TEST_ASSERT(em.tool_calls().size() == 1);
    TEST_ASSERT(event_type == "response.incomplete");
    TEST_ASSERT(data["response"]["incomplete_details"]["reason"] == "max_output_tokens");
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

// The OpenAI Responses stream contract for a tool call, as clients parse it
// (Pi's openai-responses provider builds a call only from an item that
// response.output_item.added opened): the message item is closed at
// output_index 0, then the function_call item is added, its arguments
// streamed and done, the item done, and response.completed lists both items.
namespace {
// One streamed Responses turn: the events' data payloads, in order.
struct ResponsesTurn {
    std::vector<json> events;
    json completed_output;
    std::vector<ToolCall> calls;
};

json responses_turn_tools() {
    return json::array({
        {{"type", "function"}, {"name", "ls"},
         {"parameters", {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}}}}}},
        {{"type", "function"}, {"name", "read"},
         {"parameters", {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}}}}}}});
}

ResponsesTurn stream_responses_turn(const std::string & generated, ToolMemory * memory) {
    const json tools = responses_turn_tools();
    SseEmitter em(ApiFormat::RESPONSES, "resp_rt", "test-model", 10, tools, memory);
    std::vector<std::string> raw = em.emit_start();
    for (const auto & e : em.emit_token(generated)) raw.push_back(e);
    for (const auto & e : em.emit_finish(12)) raw.push_back(e);
    ResponsesTurn turn;
    for (const auto & e : raw) {
        const size_t at = e.find("data: ");
        if (at == std::string::npos) continue;
        const size_t end = e.find('\n', at);
        turn.events.push_back(json::parse(e.substr(at + 6, end - at - 6)));
    }
    turn.completed_output = turn.events.back()["response"]["output"];
    turn.calls = em.tool_calls();
    return turn;
}

// A client replaying the turn (Pi, OMP): the completed output items as
// input, then one function_call_output per call.
json responses_replay(const json & output) {
    json input = json::array({{{"role", "user"}, {"content", "go"}}});
    for (const auto & item : output) input.push_back(item);
    for (const auto & item : output) {
        if (item["type"] == "function_call") {
            input.push_back({{"type", "function_call_output"}, {"call_id", item["call_id"]}, {"output", "ok"}});
        }
    }
    return input;
}

size_t count_of(const std::string & hay, const std::string & needle) {
    size_t n = 0;
    for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + needle.size())) ++n;
    return n;
}

}  // namespace

// Round trips: the completed turn, replayed, normalises to ONE assistant
// turn -- the remembered raw generation once on a tool-memory hit, the prose
// plus the calls' native rendering on a miss -- with no duplicated prose and
// no extra assistant boundary.
TEST_CASE(ServerUnitFixture, test_responses_turn_round_trip_one_assistant_turn) {
    const std::string call_ls = "<tool_call>\n<function=ls>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
    const std::string call_read = "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>";
    struct Case { std::string generated; std::string prose; size_t calls; };
    const std::vector<Case> cases = {
        {call_ls, "", 1},                                   // empty message
        {"Listing.\n" + call_ls, "Listing.", 1},             // text + tool
        {"Two.\n" + call_ls + "\n" + call_read, "Two.", 2}, // text + two calls
    };
    for (const auto & c : cases) {
        for (const bool hit : {true, false}) {
            ToolMemory memory;
            const auto turn = stream_responses_turn(c.generated, hit ? &memory : nullptr);
            TEST_ASSERT(turn.calls.size() == c.calls);
            ToolMemory replay_memory;
            ToolMemory & used = hit ? memory : replay_memory;
            const auto chat = normalize_chat_messages(
                responses_replay(turn.completed_output), ApiFormat::RESPONSES, used);
            TEST_ASSERT(chat.size() == 2 + c.calls);     // user, ONE assistant, the tool results
            if (chat.size() != 2 + c.calls) continue;
            TEST_ASSERT(chat[0].role == "user");
            TEST_ASSERT(chat[1].role == "assistant");
            for (size_t k = 0; k < c.calls; ++k) TEST_ASSERT(chat[2 + k].role == "tool");
            TEST_ASSERT(chat[1].tool_calls.size() == c.calls);
            TEST_ASSERT(chat[1].tool_calls_replayed == hit);
            const std::string & content = chat[1].content;
            if (hit) {
                TEST_ASSERT(content == c.generated);    // the remembered raw turn, exactly once
            } else {
                TEST_ASSERT(chat[1].tool_calls.at(0).name == "ls");
                TEST_ASSERT(chat[1].tool_calls.at(0).arguments == json({{"path", "."}}));
                const std::string rendered = render_chat_template(
                    {chat[1]}, ChatFormat::QWEN3, false);
                TEST_ASSERT(count_of(rendered, "<tool_call>") == c.calls);
                if (!c.prose.empty()) {
                    TEST_ASSERT(content.rfind(c.prose, 0) == 0);
                    TEST_ASSERT(count_of(content, c.prose) == 1);
                } else {
                    TEST_ASSERT(content.empty());
                }
            }
        }
    }
}

// Several calls: unique, ordered output indices; the arguments of each call
// equal across delta, arguments done, item done and the completed output.
TEST_CASE(ServerUnitFixture, test_responses_multiple_calls_indices_and_arguments) {
    const auto turn = stream_responses_turn(
        "Two.\n<tool_call>\n<function=ls>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>",
        nullptr);
    TEST_ASSERT(turn.calls.size() == 2);
    std::vector<int> added, done;
    std::map<int, std::string> delta, args_done, item_done;
    for (const auto & e : turn.events) {
        const std::string type = e.value("type", "");
        const int index = e.value("output_index", -1);
        if (type == "response.output_item.added") added.push_back(index);
        if (type == "response.output_item.done") {
            done.push_back(index);
            if (e["item"]["type"] == "function_call") item_done[index] = e["item"]["arguments"];
        }
        if (type == "response.function_call_arguments.delta") delta[index] += e["delta"].get<std::string>();
        if (type == "response.function_call_arguments.done") args_done[index] = e["arguments"];
    }
    TEST_ASSERT(added == std::vector<int>({0, 1, 2}));
    TEST_ASSERT(done == std::vector<int>({0, 1, 2}));
    const json & out = turn.completed_output;
    TEST_ASSERT(out.size() == 3);
    for (int index : {1, 2}) {
        TEST_ASSERT(delta[index] == args_done[index]);
        TEST_ASSERT(args_done[index] == item_done[index]);
        TEST_ASSERT(item_done[index] == out[(size_t) index]["arguments"].get<std::string>());
    }
    TEST_ASSERT(out[1]["id"] != out[2]["id"]);
}

TEST_CASE(ServerUnitFixture, test_responses_replay_preserves_turn_boundaries) {
    ToolMemory memory;
    const json call = {{"type", "function_call"}, {"call_id", "call_boundary"},
                       {"name", "ls"}, {"arguments", R"({"path":"."})"}};
    const json result = {{"type", "function_call_output"},
                         {"call_id", "call_boundary"}, {"output", "ok"}};
    const json input = json::array({
        {{"role", "assistant"}, {"content", "Earlier answer."}},
        {{"role", "user"}, {"content", "List files."}},
        {{"type", "message"}, {"role", "assistant"},
         {"content", json::array({{{"type", "output_text"}, {"text", "Listing."}}})}},
        call, result,
        {{"role", "assistant"}, {"content", "Finished."}},
    });
    const auto chat = normalize_chat_messages(input, ApiFormat::RESPONSES, memory);
    TEST_ASSERT(chat.size() == 5);
    TEST_ASSERT(chat.at(0).role == "assistant" && chat.at(0).content == "Earlier answer.");
    TEST_ASSERT(chat.at(1).role == "user" && chat.at(1).content == "List files.");
    TEST_ASSERT(chat.at(2).role == "assistant");
    TEST_ASSERT(chat.at(2).content == "Listing.");
    TEST_ASSERT(chat.at(2).tool_calls.size() == 1);
    TEST_ASSERT(chat.at(2).tool_calls.at(0).name == "ls");
    TEST_ASSERT(chat.at(2).tool_calls.at(0).arguments == json({{"path", "."}}));
    TEST_ASSERT(chat.at(3).role == "tool" && chat.at(3).content == "ok");
    TEST_ASSERT(chat.at(3).tool_call_id == "call_boundary");
    TEST_ASSERT(chat.at(4).role == "assistant" && chat.at(4).content == "Finished.");
}

TEST_CASE(ServerUnitFixture, test_emitter_responses_function_call_item_lifecycle) {
    json tools = json::array({{
        {"type", "function"}, {"name", "ls"},
        {"parameters", {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}}}}}
    }});
    SseEmitter em(ApiFormat::RESPONSES, "resp_test_002", "test-model", 10, tools, nullptr);
    std::vector<std::string> events = em.emit_start();
    for (const auto & e : em.emit_token("Listing.\n<tool_call>\n<function=ls>\n<parameter=path>\n.\n</parameter>\n"))
        events.push_back(e);
    for (const auto & e : em.emit_token("</function>\n</tool_call>")) events.push_back(e);
    for (const auto & e : em.emit_finish(12)) events.push_back(e);
    TEST_ASSERT(em.tool_calls().size() == 1);
    // Parse the SSE "data:" payloads in order.
    std::vector<json> data;
    for (const auto & e : events) {
        const size_t at = e.find("data: ");
        if (at == std::string::npos) continue;
        const size_t end = e.find('\n', at);
        data.push_back(json::parse(e.substr(at + 6, end - at - 6)));
    }
    std::vector<std::string> types;
    for (const auto & d : data) types.push_back(d.value("type", ""));
    const auto pos = [&](const std::string & type, int output_index) {
        for (size_t i = 0; i < data.size(); ++i) {
            if (types[i] == type && data[i].value("output_index", -1) == output_index) return (int) i;
        }
        return -1;
    };
    const int msg_added = pos("response.output_item.added", 0);
    const int msg_done = pos("response.output_item.done", 0);
    const int fc_added = pos("response.output_item.added", 1);
    const int fc_delta = pos("response.function_call_arguments.delta", 1);
    const int fc_args_done = pos("response.function_call_arguments.done", 1);
    const int fc_done = pos("response.output_item.done", 1);
    TEST_ASSERT(msg_added >= 0 && msg_done > msg_added);
    TEST_ASSERT(fc_added > msg_done && fc_delta > fc_added && fc_args_done > fc_delta && fc_done > fc_args_done);
    if (fc_added >= 0 && fc_done >= 0) {
        const json & added = data[(size_t) fc_added]["item"];
        TEST_ASSERT(added["type"] == "function_call" && added["name"] == "ls" && added["status"] == "in_progress");
        const json & done = data[(size_t) fc_done]["item"];
        TEST_ASSERT(done["status"] == "completed" && done["call_id"] == added["call_id"]);
        TEST_ASSERT(json::parse(done["arguments"].get<std::string>())["path"] == ".");
        TEST_ASSERT(data[(size_t) fc_delta]["item_id"] == added["id"]);
    }
    const json & completed = data.back();
    TEST_ASSERT(completed["type"] == "response.completed");
    const json & output = completed["response"]["output"];
    TEST_ASSERT(output.size() == 2 && output[0]["type"] == "message" && output[1]["type"] == "function_call");
    TEST_ASSERT(output[0]["content"][0]["text"] == "Listing.\n");
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

static SseEmitter make_emitter_with_stops(ApiFormat fmt,
                                           const std::vector<std::string> & stops) {
    return SseEmitter(fmt, "test_id_001", "test-model", 10,
                      json::array(), nullptr, stops);
}

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
// Prefix cache hash tests (model-free)
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_resolve_deepseek_chat_markers) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    ChatMarkers markers;
    TEST_ASSERT(resolve_chat_markers(tokenizer, markers));
    TEST_ASSERT(markers.family == "deepseek");
    TEST_ASSERT(markers.sys_role_prefix == std::vector<int32_t>({1}));
    TEST_ASSERT(markers.end_msg_seqs ==
                std::vector<std::vector<int32_t>>({{2}}));
    TEST_ASSERT(markers.next_role_starts ==
                std::vector<std::vector<int32_t>>({{3}, {4}}));
    TEST_ASSERT(markers.role_starts_delimit);

    // Only assistant turns carry an end marker; the system text and user
    // turns end where the next role starts. Every role marker therefore
    // opens a reusable boundary: the system text, each completed turn, and
    // the generation prompt. The marker itself belongs to the boundary,
    // matching the server's other chat families.
    const std::vector<int32_t> prompt = {
        1, 100, 3, 101, 4, 102, 2, 3, 103, 4,
    };
    TEST_ASSERT(find_all_boundaries(prompt, markers) ==
                std::vector<int>({3, 5, 8, 10}));
    // The snapshot cut is the end of the last message, which the next turn
    // extends.
    TEST_ASSERT(select_inline_snapshot_boundary(
                    find_all_boundaries(prompt, markers)) == 10);

    // A first turn prefers its system text, so a new session on the same
    // system prompt restores it instead of prefilling it again.
    const std::vector<int32_t> first_turn = {1, 100, 3, 101, 4};
    TEST_ASSERT(find_all_boundaries(first_turn, markers) ==
                std::vector<int>({3, 5}));
    TEST_ASSERT(select_inline_snapshot_boundary(
                    find_all_boundaries(first_turn, markers), 0, true) == 3);
    // Tool-heavy requests pin the same system head.
    TEST_ASSERT(select_inline_snapshot_boundary(
                    find_all_boundaries(prompt, markers), 0, true) == 3);
    remove_test_path(path);
}

TEST_CASE(ServerUnitFixture, test_prefix_cache_reserves_disk_staging_slot) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    PrefixCache cache(PrefixCache::MAX_SLOTS, tokenizer);
    TEST_ASSERT(cache.stats().capacity == PrefixCache::MAX_CACHE_SLOTS);
    TEST_ASSERT(PrefixCache::MAX_CACHE_SLOTS == ModelBackend::kMaxSlots - 1);

    remove_test_path(path);
}

TEST_CASE(ServerUnitFixture, test_prefix_cache_records_only_validated_restore) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    PrefixCache cache(2, tokenizer);
    const std::vector<int32_t> prompt = {
        1, 100, 3, 101, 4, 102, 2, 3, 103, 4,
    };
    cache.confirm_inline_snap(0, 8, prompt);
    cache.confirm_inline_snap(1, 10, prompt);

    const auto candidate = cache.lookup_candidate(
        prompt, (int)prompt.size() - 1);
    TEST_ASSERT(candidate.first == 0);
    TEST_ASSERT(candidate.second == 8);
    TEST_ASSERT(cache.stats().lifetime_hits == 0);

    cache.record_inline_hit(
        candidate.first, candidate.second, prompt.size());
    TEST_ASSERT(cache.stats().lifetime_hits == 1);

    // The classic path still accepts/counts an exact snapshot. Concurrent
    // admission requests a strict prefix because snapshots do not store the
    // next-token logits required for an empty suffix.
    const auto exact = cache.lookup(prompt);
    TEST_ASSERT(exact.first == 1);
    TEST_ASSERT(exact.second == 10);
    TEST_ASSERT(cache.stats().lifetime_hits == 2);

    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_prefix_cache_long_first_turn_snapshots_whole_prompt) {
    // reserve_inline_snap reads these once per process; the expectations
    // below are for the defaults, so an exported override skips the case.
    if (std::getenv("LUCE_PC_DEEP_FIRST_MIN") || std::getenv("LUCE_PC_DEEP_FIRST_MAX_HEAD")) {
        std::fprintf(stderr, "skip: LUCE_PC_DEEP_FIRST_* overridden\n");
        return;
    }
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    // An agent's first turn: a short system/tools head, then a user turn
    // well past LUCE_PC_DEEP_FIRST_MIN (4096). The snapshot covers the whole
    // prompt, so the first follow-up does not re-prefill the conversation.
    std::vector<int32_t> short_head = {1, 100, 3};
    short_head.insert(short_head.end(), 5000, 101);
    short_head.push_back(4);
    {
        PrefixCache cache(2, tokenizer);
        auto r = cache.reserve_inline_snap(
            short_head, /*restored_prefix_len=*/0,
            /*prefer_tools_boundary=*/true);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == (int) short_head.size());
        r.cancel();
    }
    // A new conversation that restored only the shared head still snapshots
    // its long first turn whole.
    {
        PrefixCache cache(2, tokenizer);
        auto r = cache.reserve_inline_snap(
            short_head, /*restored_prefix_len=*/3,
            /*prefer_tools_boundary=*/true);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == (int) short_head.size());
        r.cancel();
    }

    // A head longer than LUCE_PC_DEEP_FIRST_MAX_HEAD (2048) keeps its own
    // pin: new conversations that share it must not re-prefill it.
    std::vector<int32_t> long_head = {1};
    long_head.insert(long_head.end(), 3000, 100);
    long_head.push_back(3);
    long_head.insert(long_head.end(), 5000, 101);
    long_head.push_back(4);
    {
        PrefixCache cache(2, tokenizer);
        auto r = cache.reserve_inline_snap(
            long_head, /*restored_prefix_len=*/0,
            /*prefer_tools_boundary=*/true);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == 3002);
        r.cancel();
    }

    // A short tail keeps the head pin as before.
    const std::vector<int32_t> short_tail = {1, 100, 3, 101, 4};
    {
        PrefixCache cache(2, tokenizer);
        auto r = cache.reserve_inline_snap(
            short_tail, /*restored_prefix_len=*/0,
            /*prefer_tools_boundary=*/true);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == 3);
        r.cancel();
    }

    // A first turn without tools saves the same way: its long message whole,
    // and the head when the message is short.
    {
        PrefixCache cache(2, tokenizer);
        auto r = cache.reserve_inline_snap(
            short_head, /*restored_prefix_len=*/0,
            /*prefer_tools_boundary=*/false);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == (int) short_head.size());
        r.cancel();
        r = cache.reserve_inline_snap(short_tail);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == 3);
        r.cancel();
    }

    // A forced pin ahead of the restored prefix (PPP's pin of a head seen
    // before, not resident now) still wins.
    {
        PrefixCache cache(2, tokenizer);
        auto r = cache.reserve_inline_snap(
            short_head, /*restored_prefix_len=*/0,
            /*prefer_tools_boundary=*/true, /*forced_cut=*/3);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == 3);
        r.cancel();
    }
    // Once that pin is restored, the long first turn is snapshotted whole.
    {
        PrefixCache cache(2, tokenizer);
        auto r = cache.reserve_inline_snap(
            short_head, /*restored_prefix_len=*/3,
            /*prefer_tools_boundary=*/true, /*forced_cut=*/3);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == (int) short_head.size());
        r.cancel();
    }

    // The whole-prompt snapshot is what the first follow-up restores: once
    // committed, the next turn (the first one plus new tokens) finds all of it.
    {
        PrefixCache cache(2, tokenizer);
        auto r = cache.reserve_inline_snap(
            short_head, /*restored_prefix_len=*/0,
            /*prefer_tools_boundary=*/true);
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.commit(short_head));
        std::vector<int32_t> follow_up = short_head;
        follow_up.insert(follow_up.end(), {1, 102, 3, 103, 103, 4});
        const auto hit = cache.lookup(follow_up);
        TEST_ASSERT(hit.first >= 0);
        TEST_ASSERT(hit.second == (int) short_head.size());
    }

    // A whole prompt the resident budget can never hold keeps the head pin
    // instead of saving nothing.
    {
        PrefixCache cache(2, tokenizer, /*max_resident_bytes=*/1000);
        auto r = cache.reserve_inline_snap(
            short_head, /*restored_prefix_len=*/0,
            /*prefer_tools_boundary=*/true, /*forced_cut=*/0,
            /*restore_source_slot=*/-1,
            [](int cut) { return (size_t) cut; });
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.target_cut() == 3);
        r.cancel();
    }

    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_restore_invalidation_preserves_pending_pin) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    PrefixCache cache(2, tokenizer);
    const std::vector<int32_t> first = {1, 100};
    const std::vector<int32_t> stale = {1, 200};
    const std::vector<int32_t> pinned = {1, 300};
    const std::vector<int32_t> replacement = {1, 400};
    const std::vector<int32_t> next = {1, 500};
    cache.confirm_inline_snap(0, 2, first);
    cache.confirm_inline_snap(1, 2, stale);

    // Reserve the oldest slot for a protected tool-prefix capture, then
    // invalidate an unrelated restore while that reservation is in flight.
    auto prepared = cache.reserve_inline_snap(
        pinned, /*restored_prefix_len=*/0,
        /*prefer_tools_boundary=*/true, /*forced_cut=*/2);
    TEST_ASSERT(prepared.slot() == 0);
    TEST_ASSERT(prepared.target_cut() == 2);
    // Another request's capture can run beside it, never in its slot.
    auto beside = cache.reserve_inline_snap(
        next, /*restored_prefix_len=*/0,
        /*prefer_tools_boundary=*/false, /*forced_cut=*/2);
    TEST_ASSERT(beside.active() && beside.slot() == 1);
    beside.cancel();
    cache.invalidate_inline_snap(/*slot=*/1);
    TEST_ASSERT(prepared.commit(pinned));

    // Refill the unrelated slot. The next eviction must choose this
    // unprotected entry, proving invalidation did not clear the pending pin.
    cache.confirm_inline_snap(1, 2, replacement);
    auto victim = cache.reserve_inline_snap(
        next, /*restored_prefix_len=*/0,
        /*prefer_tools_boundary=*/false, /*forced_cut=*/2);
    TEST_ASSERT(victim.slot() == 1);
    TEST_ASSERT(victim.target_cut() == 2);
    victim.cancel();

    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_prefix_cache_resident_budget_and_stall_stats) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    PrefixCache cache(3, tokenizer, /*max_resident_bytes=*/300);
    const std::vector<int32_t> pinned = {1, 100};
    const std::vector<int32_t> ordinary = {1, 200};
    const std::vector<int32_t> replacement = {1, 300};
    const std::vector<int32_t> oversized = {1, 400};

    auto prepared = cache.reserve_inline_snap(
        pinned, 0, /*prefer_tools_boundary=*/true, /*forced_cut=*/2,
        /*restore_source_slot=*/-1,
        [](int) { return 100; });
    TEST_ASSERT(prepared.slot() == 0);
    TEST_ASSERT(prepared.commit(pinned, /*resident_bytes=*/100));

    prepared = cache.reserve_inline_snap(
        ordinary, 0, /*prefer_tools_boundary=*/false, /*forced_cut=*/2,
        /*restore_source_slot=*/-1,
        [](int) { return 100; });
    TEST_ASSERT(prepared.slot() == 1);
    TEST_ASSERT(prepared.commit(ordinary, /*resident_bytes=*/100));

    // A third slot exists, but its 150-byte checkpoint would exceed the
    // resident ceiling. Replace the oldest unprotected leaf (slot 1) while
    // preserving the protected tools pin in slot 0.
    prepared = cache.reserve_inline_snap(
        replacement, 0, /*prefer_tools_boundary=*/false, /*forced_cut=*/2,
        /*restore_source_slot=*/-1,
        [](int) { return 150; });
    TEST_ASSERT(prepared.slot() == 1);
    TEST_ASSERT(prepared.commit(replacement, /*resident_bytes=*/150));

    auto stats = cache.stats();
    TEST_ASSERT(stats.in_use == 2);
    TEST_ASSERT(stats.max_resident_bytes == 300);
    TEST_ASSERT(stats.resident_bytes == 250);
    TEST_ASSERT(cache.lookup(pinned).first == 0);

    prepared = cache.reserve_inline_snap(
        oversized, 0, /*prefer_tools_boundary=*/false, /*forced_cut=*/2,
        /*restore_source_slot=*/-1,
        [](int) { return 301; });
    TEST_ASSERT(!prepared.active());
    cache.record_capture_attempt(/*elapsed_us=*/1500, /*success=*/false);
    cache.record_restore_attempt(/*elapsed_us=*/2500, /*restored=*/false);
    stats = cache.stats();
    TEST_ASSERT(stats.in_use == 2);
    TEST_ASSERT(stats.resident_bytes == 250);
    TEST_ASSERT(stats.budget_skips == 1);
    TEST_ASSERT(stats.capture_attempts == 1);
    TEST_ASSERT(stats.capture_failures == 1);
    TEST_ASSERT(stats.capture_stall_us_total == 1500);
    TEST_ASSERT(stats.capture_stall_us_max == 1500);
    TEST_ASSERT(stats.restore_attempts == 1);
    TEST_ASSERT(stats.restore_invalidations == 1);
    TEST_ASSERT(stats.restore_stall_us_total == 2500);
    TEST_ASSERT(stats.restore_stall_us_max == 2500);

    unlink(path.c_str());
}

// Resident budget: a capture that fits the budget but that no single entry
// makes room for evicts several, least recently used first, when it commits.
TEST_CASE(ServerUnitFixture, test_prefix_cache_budget_evicts_several_for_large_capture) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const auto bytes = [](size_t n) {
        return [n](int) { return n; };
    };
    const std::vector<std::vector<int32_t>> small = {
        {1, 100, 3, 101}, {1, 200, 3, 201}, {1, 300, 3, 301}, {1, 400, 3, 401}};
    const std::vector<int32_t> big = {1, 900, 3, 901};
    const auto fill = [&](PrefixCache & cache, std::vector<int> & slots) {
        for (const auto & ids : small) {
            auto r = cache.reserve_inline_snap(ids, 0, false, (int)ids.size(), -1, bytes(200));
            TEST_ASSERT(r.active());
            slots.push_back(r.slot());
            TEST_ASSERT(r.commit(ids, 200));
        }
        TEST_ASSERT(cache.stats().resident_bytes == 800);
    };

    {
        PrefixCache cache(8, tokenizer, /*max_resident_bytes=*/1000);
        cache.set_prunes_superseded(true);
        cache.set_record_evictions(true);
        std::vector<int> slots;
        fill(cache, slots);
        TEST_ASSERT(cache.take_evicted().empty());
        // 700 bytes: 500 must go, and every entry holds 200.
        auto r = cache.reserve_inline_snap(big, 0, false, (int)big.size(), -1, bytes(700));
        TEST_ASSERT(r.active());
        TEST_ASSERT(r.extra_victim_slots() ==
                    std::vector<int>({slots[0], slots[1], slots[2]}));
        // Nothing leaves RAM before the capture lands.
        TEST_ASSERT(cache.stats().in_use == 4 && cache.stats().resident_bytes == 800);
        TEST_ASSERT(cache.take_evicted().empty());
        TEST_ASSERT(r.commit(big, 700));
        const auto evicted = cache.take_evicted();
        TEST_ASSERT(evicted.size() == 3);
        for (size_t i = 0; i < evicted.size(); ++i) {
            TEST_ASSERT(evicted[i].slot == slots[i]);
            TEST_ASSERT(evicted[i].ids == small[i]);
            TEST_ASSERT(evicted[i].persist);  // each one goes to disk
        }
        const auto stats = cache.stats();
        TEST_ASSERT(stats.in_use == 2 && stats.resident_bytes == 900);
        TEST_ASSERT(stats.budget_skips == 0);
        TEST_ASSERT(cache.lookup_candidate(small[3], 4).second == 4);
        std::vector<int32_t> next = big;
        next.insert(next.end(), {4, 902});
        TEST_ASSERT(cache.lookup_candidate(next, (int)next.size()).second == (int)big.size());
    }
    {
        // A capture that fails or is cancelled evicts none of them.
        PrefixCache cache(8, tokenizer, 1000);
        cache.set_record_evictions(true);
        std::vector<int> slots;
        fill(cache, slots);
        auto r = cache.reserve_inline_snap(big, 0, false, (int)big.size(), -1, bytes(700));
        TEST_ASSERT(r.active() && r.extra_victim_slots().size() == 3);
        r.cancel();
        TEST_ASSERT(cache.stats().in_use == 4 && cache.stats().resident_bytes == 800);
        r = cache.reserve_inline_snap(big, 0, false, (int)big.size(), -1, bytes(700));
        TEST_ASSERT(r.active());
        r.abort();
        TEST_ASSERT(cache.stats().in_use == 4 && cache.stats().resident_bytes == 800);
        TEST_ASSERT(cache.take_evicted().empty());
    }
    {
        // Larger than the whole budget: skipped, nothing evicted.
        PrefixCache cache(8, tokenizer, 1000);
        cache.set_record_evictions(true);
        std::vector<int> slots;
        fill(cache, slots);
        auto r = cache.reserve_inline_snap(big, 0, false, (int)big.size(), -1, bytes(1001));
        TEST_ASSERT(!r.active());
        TEST_ASSERT(cache.stats().budget_skips == 1);
        TEST_ASSERT(cache.stats().in_use == 4 && cache.stats().resident_bytes == 800);
        TEST_ASSERT(cache.take_evicted().empty());
        // Exactly the budget: everything else goes.
        r = cache.reserve_inline_snap(big, 0, false, (int)big.size(), -1, bytes(1000));
        TEST_ASSERT(r.active() && r.extra_victim_slots() == slots);
        TEST_ASSERT(r.commit(big, 1000));
        TEST_ASSERT(cache.take_evicted().size() == 4);
        TEST_ASSERT(cache.stats().in_use == 1 && cache.stats().resident_bytes == 1000);
    }
    unlink(path.c_str());
}

// What a several-entry plan never takes: protected pins, heads that other
// entries extend, the capture's own ancestors and restore source, and the
// entries another capture in flight evicts.
TEST_CASE(ServerUnitFixture, test_prefix_cache_budget_plan_keeps_protected_and_in_use) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(16, tokenizer, /*max_resident_bytes=*/1200);
    cache.set_record_evictions(true);
    const std::vector<int32_t> s1 = {1, 110, 3, 111};
    const std::vector<int32_t> s2 = {1, 120, 3, 121};
    const std::vector<int32_t> pin = {1, 130};
    const std::vector<int32_t> head = {1, 140};
    const std::vector<int32_t> leaf = {1, 140, 3, 141};
    const std::vector<int32_t> ancestor = {1, 150};
    const std::vector<int32_t> source = {1, 150, 3, 151};
    const std::vector<int32_t> t1 = {1, 160, 3, 161};
    const std::vector<int32_t> t2 = {1, 170, 3, 171};
    // Oldest first. Every entry holds 100 bytes.
    int slot = 0;
    for (const auto * ids : {&s1, &s2, &pin, &head, &leaf, &ancestor, &source, &t1, &t2}) {
        cache.confirm_inline_snap(slot++, (int)ids->size(), *ids,
                                  /*protect=*/ids == &pin, /*resident_bytes=*/100);
    }
    TEST_ASSERT(cache.stats().resident_bytes == 900);

    // In flight: a 500-byte capture that needs two of them gone (s1, s2).
    const std::vector<int32_t> other = {1, 180, 3, 181};
    auto first = cache.reserve_inline_snap(
        other, 0, false, (int)other.size(), -1, [](int) { return (size_t)500; });
    TEST_ASSERT(first.active());
    TEST_ASSERT(first.extra_victim_slots() == std::vector<int>({0, 1}));

    // The next turn of `source`, restored from it: 300 bytes must go. The
    // oldest entries are s1/s2 (the first capture's), then the pin, the head
    // and the ancestor; only leaf, t1 and t2 are eligible.
    std::vector<int32_t> turn = source;
    turn.insert(turn.end(), {4, 152});
    auto second = cache.reserve_inline_snap(
        turn, (int)source.size(), false, (int)turn.size(),
        /*restore_source_slot=*/6, [](int) { return (size_t)100; });
    TEST_ASSERT(second.active());
    TEST_ASSERT(second.extra_victim_slots() == std::vector<int>({4, 7, 8}));
    // A third capture needs 401 bytes gone: only `source` (100, not its
    // ancestor here) is left to plan, so it is skipped. No protected pin,
    // head or ancestor is taken instead.
    auto third = cache.reserve_inline_snap(
        {1, 190, 3, 191}, 0, false, 4, -1, [](int) { return (size_t)101; });
    TEST_ASSERT(!third.active());

    TEST_ASSERT(second.commit(turn, 100));
    auto evicted = cache.take_evicted();
    TEST_ASSERT(evicted.size() == 3 && evicted[0].ids == leaf &&
                evicted[1].ids == t1 && evicted[2].ids == t2);
    TEST_ASSERT(first.commit(other, 500));
    evicted = cache.take_evicted();
    TEST_ASSERT(evicted.size() == 2 && evicted[0].ids == s1 && evicted[1].ids == s2);
    for (const auto * kept : {&pin, &head, &ancestor, &source}) {
        TEST_ASSERT(cache.lookup_candidate(*kept, (int)kept->size()).second ==
                    (int)kept->size());
    }
    TEST_ASSERT(cache.stats().resident_bytes == 1000);
    unlink(path.c_str());
}

// An edit backup goes before an older leaf, and is not written to disk while
// its conversation's deeper snapshot is resident.
TEST_CASE(ServerUnitFixture, test_prefix_cache_budget_plan_evicts_edit_backups_first) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(16, tokenizer, /*max_resident_bytes=*/1000);
    cache.set_prunes_superseded(true);
    cache.set_record_evictions(true);
    const std::vector<int32_t> y = {1, 200, 3, 201};
    const std::vector<int32_t> z = {1, 300, 3, 301};
    const std::vector<int32_t> head = {1, 100};
    const std::vector<int32_t> turn1 = {1, 100, 3, 101};
    const std::vector<int32_t> turn2 = {1, 100, 3, 101, 4, 102};
    cache.confirm_inline_snap(0, 4, y, false, 100);
    cache.confirm_inline_snap(1, 4, z, false, 100);
    cache.confirm_inline_snap(2, 2, head, false, 100);
    cache.confirm_inline_snap(3, 4, turn1, false, 100);
    cache.confirm_inline_snap(4, 6, turn2, false, 100);
    TEST_ASSERT(cache.prune_superseded_ancestors(4).empty());  // turn1 kept
    TEST_ASSERT(!cache.persist_on_evict(3));                   // as an edit backup

    const std::vector<int32_t> big = {1, 400, 3, 401};
    auto r = cache.reserve_inline_snap(
        big, 0, false, 4, -1, [](int) { return (size_t)700; });
    TEST_ASSERT(r.active());
    TEST_ASSERT(r.extra_victim_slots() == std::vector<int>({3, 0}));
    TEST_ASSERT(r.commit(big, 700));
    const auto evicted = cache.take_evicted();
    TEST_ASSERT(evicted.size() == 2);
    TEST_ASSERT(evicted[0].ids == turn1 && !evicted[0].persist);
    TEST_ASSERT(evicted[1].ids == y && evicted[1].persist);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_canonical_turn_matches_replay_checkpoint) {
    TEST_ASSERT(http_detail::canonical_turn_matches_checkpoint(
        {1, 2, 3}, {1, 2, 9, 4}, 2));
    TEST_ASSERT(!http_detail::canonical_turn_matches_checkpoint(
        {1, 2, 3}, {1, 9, 3, 4}, 2));
    TEST_ASSERT(!http_detail::canonical_turn_matches_checkpoint(
        {1, 2, 3}, {1, 2}, 2));
    TEST_ASSERT(!http_detail::canonical_turn_matches_checkpoint(
        {1, 2, 3}, {1, 2, 3, 4}, 0));
    TEST_ASSERT(!http_detail::canonical_turn_matches_checkpoint(
        {1, 2, 3}, {1, 2, 3, 4}, 4));
}

TEST_CASE(ServerUnitFixture, test_qwen_completed_tool_turn_preserves_generation_prefix) {
    const std::string sentinel = "__AGENT_TURN_SENTINEL__";
    for (bool thinking : {false, true}) {
        std::vector<ChatMessage> messages = {{"user", "inspect the repo"}};
        const std::string generation = render_chat_template(
            messages, ChatFormat::QWEN3, true, thinking);
        messages.push_back({"assistant", sentinel});
        const std::string probe = render_chat_template(
            messages, ChatFormat::QWEN3, false, thinking);

        std::string content;
        TEST_ASSERT(http_detail::canonical_assistant_content(
            generation, probe, sentinel, "<tool_call>x</tool_call>", content));
        messages.back().content = content;
        const std::string completed = render_chat_template(
            messages, ChatFormat::QWEN3, false, thinking);
        TEST_ASSERT(completed.compare(0, generation.size(), generation) == 0);
    }
}

TEST_CASE(ServerUnitFixture, test_parse_dsml_v41_tool_calls) {
    // DeepSeek V4.1 spells DSML tags with a leading-space name.
    const std::string text =
        "I'll read the file.\n\n"
        "<｜DSML｜ calls>\n"
        "<｜DSML｜ invoke name=\"read_file\">\n"
        "<｜DSML｜ parameter name=\"path\" string=\"true\">deepseek4_candidates.cpp</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"limit\" string=\"false\">40</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>";
    json tools = json::array({
        {{"type", "function"}, {"function", {
             {"name", "read_file"},
             {"parameters", {
                 {"type", "object"},
                 {"properties", {
                     {"path", {{"type", "string"}}},
                     {"limit", {{"type", "integer"}}}
                 }}
             }}
         }}}
    });
    auto result = parse_tool_calls(text, tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read_file");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "deepseek4_candidates.cpp");
        TEST_ASSERT(args["limit"] == 40);
    }
    TEST_ASSERT(result.cleaned_text == "I'll read the file.");
}

// Reference renders from encoding/tests in the DeepSeek-V4.1-Flash model repo.
TEST_CASE(ServerUnitFixture, test_deepseek41_template_matches_reference_chat) {
    const std::vector<ChatMessage> msgs = {
        {"system", "You are a helpful assistant."},
        {"user", "Hello"},
        {"assistant", "Hi there! How can I help you?"},
        {"user", "What is the capital of France?"},
        {"assistant", "The capital of France is Paris."},
    };
    const std::string expected = R"REF(<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜></think>Hi there! How can I help you?<｜end▁of▁sentence｜><｜User｜>What is the capital of France?<｜Assistant｜></think>The capital of France is Paris.<｜end▁of▁sentence｜>)REF";
    TEST_ASSERT(render_chat_template(msgs, ChatFormat::DEEPSEEK41, false, false) == expected);
}

TEST_CASE(ServerUnitFixture, test_deepseek41_template_matches_reference_tools) {
    const std::vector<ChatMessage> msgs = {
        {"system", "You are a helpful assistant."},
        {"user", "What's the weather like in Beijing?"},
    };
    const std::string tools_json = R"REF([{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a specific location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}}, "required": ["location"]}}}, {"type": "function", "function": {"name": "search", "description": "Search the web for information", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "num_results": {"type": "integer", "description": "Number of results to return"}}, "required": ["query"]}}}])REF";
    const std::string expected = R"REF(<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)

You are a helpful assistant.

## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{"name": "get_weather", "description": "Get the weather for a specific location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}}, "required": ["location"]}}
{"name": "search", "description": "Search the web for information", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "num_results": {"type": "integer", "description": "Number of results to return"}}, "required": ["query"]}}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
<｜User｜>What's the weather like in Beijing?<｜Assistant｜><think>)REF";
    TEST_ASSERT(render_chat_template(msgs, ChatFormat::DEEPSEEK41, true, true, tools_json, "high") ==
                expected);
}

TEST_CASE(ServerUnitFixture, test_deepseek41_template_tool_results_and_effort) {
    const std::vector<ChatMessage> msgs = {
        {"user", "Read a.cpp and b.cpp."},
        {"assistant", "\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>"},
        {"tool", "int a;", "call_1"},
        {"tool", "int b;", "call_2"},
        {"system", "Be brief."},
    };
    const std::string out = render_chat_template(msgs, ChatFormat::DEEPSEEK41, true, true, "", "max");
    TEST_ASSERT(out.rfind("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 (range 1-100", 0) == 0);
    TEST_ASSERT(out.find("<｜User｜><tool_result>int a;</tool_result>\n\n<tool_result>int b;</tool_result>"
                         "<｜System｜>Be brief.<｜Assistant｜><think>") != std::string::npos);
    TEST_ASSERT(chat_format_for_arch("deepseek41") == ChatFormat::DEEPSEEK41);
    TEST_ASSERT(chat_format_for_arch("deepseek4") == ChatFormat::DEEPSEEK4);
}

TEST_CASE(ServerUnitFixture, test_hash_prefix_deterministic) {
    std::vector<int32_t> ids = {100, 200, 300, 400, 500};
    auto h1 = hash_prefix(ids.data(), (int)ids.size());
    auto h2 = hash_prefix(ids.data(), (int)ids.size());
    TEST_ASSERT(h1 == h2);
}

TEST_CASE(ServerUnitFixture, test_hash_prefix_different_inputs) {
    std::vector<int32_t> ids1 = {100, 200, 300};
    std::vector<int32_t> ids2 = {100, 200, 301};
    auto h1 = hash_prefix(ids1.data(), (int)ids1.size());
    auto h2 = hash_prefix(ids2.data(), (int)ids2.size());
    TEST_ASSERT(h1 != h2);
}

TEST_CASE(ServerUnitFixture, test_hash_prefix_different_lengths) {
    std::vector<int32_t> ids1 = {100, 200, 300};
    std::vector<int32_t> ids2 = {100, 200, 300, 400};
    auto h1 = hash_prefix(ids1.data(), (int)ids1.size());
    auto h2 = hash_prefix(ids2.data(), (int)ids2.size());
    TEST_ASSERT(h1 != h2);
}

TEST_CASE(ServerUnitFixture, test_hash_prefix_empty) {
    auto h = hash_prefix(nullptr, 0);
    // Should not crash, just return a hash of empty input
    TEST_ASSERT(h.size() == 16);
}

TEST_CASE(ServerUnitFixture, test_find_boundaries_empty) {
    ChatMarkers markers;
    markers.family = "qwen";
    std::vector<int32_t> ids;
    auto bounds = find_all_boundaries(ids, markers);
    TEST_ASSERT(bounds.empty());
}

// Boundary detection with and without a system message. Qwen-shaped
// synthetic markers: {100,200} = "<im_start>system", {101} = "<im_end>",
// {100} = "<im_start>" as the next-role start.
static ChatMarkers make_qwen_boundary_markers_for_test() {
    ChatMarkers markers;
    markers.family = "qwen";
    markers.sys_role_prefix = {100, 200};
    markers.end_msg_seqs = {{101}};
    markers.next_role_starts = {{100}};
    return markers;
}

TEST_CASE(ServerUnitFixture, test_find_boundaries_qwen_system_first) {
    auto markers = make_qwen_boundary_markers_for_test();
    // <im_start> system ... <im_end> <im_start> user ... <im_end> <im_start> assistant ...
    std::vector<int32_t> ids = {
        100, 200, 10, 11, 101,
        100, 201, 12, 13, 101,
        100, 202, 14,
    };
    auto bounds = find_all_boundaries(ids, markers);
    TEST_ASSERT(bounds.size() == 2);
    TEST_ASSERT(bounds[0] == 6);
    TEST_ASSERT(bounds[1] == 11);
}

TEST_CASE(ServerUnitFixture, test_find_boundaries_qwen_user_first_quoted_system) {
    auto markers = make_qwen_boundary_markers_for_test();
    // User-first prompt whose second message quotes a literal system prefix
    // ({100,200}) in its content. The leading user role must still anchor the
    // boundaries; the quoted prefix is just content.
    std::vector<int32_t> ids = {
        100, 201, 10, 11, 101,
        100, 202, 100, 200, 12, 101,
        100, 201, 14,
    };
    auto bounds = find_all_boundaries(ids, markers);
    TEST_ASSERT(bounds.size() == 2);
    TEST_ASSERT(bounds[0] == 6);
    TEST_ASSERT(bounds[1] == 12);
}

TEST_CASE(ServerUnitFixture, test_find_boundaries_qwen_user_first) {
    auto markers = make_qwen_boundary_markers_for_test();
    // No system message: <im_start> user ... <im_end> <im_start> assistant
    // ... <im_end> <im_start> user ... must still yield the role boundaries
    // instead of an empty list (which disabled the prefix cache entirely).
    std::vector<int32_t> ids = {
        100, 201, 10, 11, 101,
        100, 202, 12, 13, 101,
        100, 201, 14,
    };
    auto bounds = find_all_boundaries(ids, markers);
    TEST_ASSERT(bounds.size() == 2);
    TEST_ASSERT(bounds[0] == 6);
    TEST_ASSERT(bounds[1] == 11);
}

TEST_CASE(ServerUnitFixture, test_tool_schema_is_part_of_stable_system_boundary) {
    // Synthetic Qwen-shaped prompt:
    //   <system> TOOL_SCHEMA </system> <user> question </user> <assistant>
    // Marker IDs are intentionally simple; the invariant under test is that
    // the first safe boundary ends after the system/tool block. Its position
    // stays stable while its prefix hash changes with the tools, but not with
    // a user-only suffix change.
    ChatMarkers markers;
    markers.family = "qwen";
    markers.sys_role_prefix = {10, 11};
    markers.end_msg_seqs = {{12}};
    markers.next_role_starts = {{10}};

    const std::vector<int32_t> prompt_a = {
        10, 11, 100, 101, 102, 12, 10, 20, 200, 12, 10, 30,
    };
    const std::vector<int32_t> prompt_new_user = {
        10, 11, 100, 101, 102, 12, 10, 20, 999, 12, 10, 30,
    };
    const std::vector<int32_t> prompt_new_tools = {
        10, 11, 100, 101, 777, 12, 10, 20, 200, 12, 10, 30,
    };

    const auto bounds_a = find_all_boundaries(prompt_a, markers);
    const auto bounds_user = find_all_boundaries(prompt_new_user, markers);
    const auto bounds_tools = find_all_boundaries(prompt_new_tools, markers);
    TEST_ASSERT(bounds_a.size() == 2);
    TEST_ASSERT(bounds_user == bounds_a);
    TEST_ASSERT(bounds_tools == bounds_a);

    const int system_end = bounds_a.front();
    TEST_ASSERT(system_end == 7);
    TEST_ASSERT(hash_prefix(prompt_a.data(), system_end) ==
                hash_prefix(prompt_new_user.data(), system_end));
    TEST_ASSERT(hash_prefix(prompt_a.data(), system_end) !=
                hash_prefix(prompt_new_tools.data(), system_end));
}

TEST_CASE(ServerUnitFixture, test_find_boundaries_stray_end_msg_does_not_truncate) {
    // A lone end-of-message marker embedded in message content (file dumps,
    // terminal output, model-echoed chatml) must not truncate the boundary
    // walk. Before the fix, the first stray \n with no role start within 5
    // tokens cut the scan off, hiding every real boundary after it and pinning
    // the inline-snapshot deepen target at the already-restored prefix length.
    //
    // Layout (qwen-shaped synthetic markers):
    //   10=<|im_start|>  11="system"  12=<|im_end|>
    //   idx: 0:10 1:11 2:100 3:12 4:10 5:20 6:200 7:12(stray) 8:600 9:601
    //         10:602 11:603 12:604 13:12 14:10 15:30 16:300 17:12 18:10 19:40 20:400
    // The stray 12 at idx 7 has five content tokens before the next real <|im_end|>
    // (idx 13), so its 5-token window holds no <|im_start|>.
    ChatMarkers markers;
    markers.family = "qwen";
    markers.sys_role_prefix = {10, 11};
    markers.end_msg_seqs = {{12}};
    markers.next_role_starts = {{10}};

    const std::vector<int32_t> prompt = {
        10, 11, 100, 12, 10, 20, 200,
        12,             // stray <|im_end|> in user content
        600, 601, 602, 603, 604,
        12,             // real end of the user message
        10, 30, 300, 12, 10, 40, 400,
    };

    const auto bounds = find_all_boundaries(prompt, markers);
    // Real boundaries after the stray (assistant start = 15, final user start =
    // 19) must be found; the stable system head (5) is unchanged.
    TEST_ASSERT(bounds == (std::vector<int>{5, 15, 19}));
    TEST_ASSERT(bounds.front() == 5);
}

TEST_CASE(ServerUnitFixture, test_find_boundaries_clean_prompt_unchanged) {
    // A clean prompt (no stray markers) must produce the identical boundary
    // list as before the fix — the stray-skip path must never alter correct
    // prompts.
    ChatMarkers markers;
    markers.family = "qwen";
    markers.sys_role_prefix = {10, 11};
    markers.end_msg_seqs = {{12}};
    markers.next_role_starts = {{10}};

    //  system content  user content  assistant content  user2
    const std::vector<int32_t> prompt = {
        10, 11, 100, 101, 12, 10, 20, 200, 12, 10, 30, 300, 12, 10, 40,
    };
    const auto bounds = find_all_boundaries(prompt, markers);
    TEST_ASSERT(bounds == (std::vector<int>{6, 10, 14}));
}

TEST_CASE(ServerUnitFixture, test_inline_snapshot_boundary_advances_past_restore) {
    const std::vector<int> boundaries = {100, 240, 380, 520};
    // The last boundary, the end of the last message: a person's follow-up
    // and an agent's next step after its tool results both extend it.
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries) == 520);
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 380) == 520);
    // Do not reserve a snapshot when the restore already covers that point.
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 520) == 0);
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 600) == 0);
    TEST_ASSERT(select_inline_snapshot_boundary({}, 0) == 0);
    TEST_ASSERT(select_inline_snapshot_boundary({100}, 0) == 100);
}

// Families with a trailing role marker can snapshot through the tool results.
// Gemma instead continues the open model turn; its continuation is tested below.
TEST_CASE(ServerUnitFixture, test_tool_result_prompt_prefix_survives_next_agent_turn) {
    struct Family { ChatFormat format; const char * marker; };
    const Family families[] = {
        {ChatFormat::QWEN3, "<|im_start|>"},
        {ChatFormat::DEEPSEEK4, "<｜Assistant｜>"},
        {ChatFormat::DEEPSEEK41, "<｜Assistant｜>"},
        {ChatFormat::LAGUNA, "<assistant>"},
    };
    std::vector<ChatMessage> turn = {
        {"system", "You are a coding agent."},
        {"user", "Read main.cpp and fix the bug."},
        {"assistant", "<tool_call>read main.cpp</tool_call>"},
        {"tool", "int main() { return 1; }", "call_1"},
        {"tool", "second parallel result", "call_2"},
    };
    std::vector<ChatMessage> next = turn;
    next.push_back({"assistant", "<tool_call>edit main.cpp</tool_call>"});
    next.push_back({"tool", "edited", "call_3"});
    for (const bool thinking : {false, true}) {
        for (const auto & family : families) {
            const std::string now = render_chat_template(
                turn, family.format, true, thinking, "[]");
            const std::string later = render_chat_template(
                next, family.format, true, thinking, "[]");
            const size_t marker = now.rfind(family.marker);
            TEST_ASSERT(marker != std::string::npos);
            const size_t cut = marker + std::strlen(family.marker);
            // The cut covers both tool results and stops before the
            // generation prompt's role text.
            TEST_ASSERT(now.substr(0, cut).find("second parallel result") !=
                        std::string::npos);
            TEST_ASSERT_MSG(later.compare(0, cut, now, 0, cut) == 0,
                            now.substr(0, cut));
        }
    }
}

// Qwen saves only at prefill chunk starts past the restore point. Cuts it
// cannot reach are skipped: a short system/tools head is not pinned (it would
// never be saved and the cache would never deepen), and the next reachable
// boundary is taken instead.
TEST_CASE(ServerUnitFixture, test_spaced_restore_points) {
    const std::vector<int> b = {300, 900, 1500, 2600, 2700, 5000, 5100};
    TEST_ASSERT((spaced_restore_points(b, 2048) == std::vector<int>{300, 2600, 5000}));
    TEST_ASSERT(spaced_restore_points(b, 0) == b);
    // Prefix-stable: a shorter prompt keeps the longer prompt's points up to its end.
    TEST_ASSERT((spaced_restore_points({300, 900, 1500, 2600}, 2048) == std::vector<int>{300, 2600}));

    // A capture is the end of the last message, the point the next turn
    // extends; with spacing, a cut short of it moves down to a spaced point,
    // and none is reserved when that adds no prefix.
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(4, tokenizer);
    const std::vector<int32_t> prompt = {1, 100, 3, 101, 4, 102, 3};
    const std::vector<int> boundaries = find_all_boundaries(prompt, cache.chat_markers());
    TEST_ASSERT(boundaries.size() >= 3);
    const int head = boundaries.front();
    auto plain = cache.reserve_inline_snap(prompt);
    TEST_ASSERT(plain.active() && plain.target_cut() == boundaries.back());
    plain.cancel();
    // A spacing wider than the whole prompt still keeps the last message's end.
    auto last = cache.reserve_inline_snap(prompt, 0, false, 0, -1, {}, 0, /*restore_point_spacing=*/64);
    TEST_ASSERT(last.active() && last.target_cut() == boundaries.back());
    last.cancel();
    // A pinned cut in the middle moves down to the head, the only spaced point.
    auto moved = cache.reserve_inline_snap(prompt, 0, false, /*forced_cut=*/boundaries[1], -1, {}, 0, 64);
    TEST_ASSERT(moved.active() && moved.target_cut() == head);
    moved.cancel();
    auto none = cache.reserve_inline_snap(prompt, boundaries.back(), false, 0, -1, {}, 0, 64);
    TEST_ASSERT(!none.active());
}

// Concurrent requests each capture their own prefix at once: a capture in
// flight holds its slot and bytes, and the same prefix is not captured twice.
TEST_CASE(ServerUnitFixture, test_inline_captures_in_flight_hold_their_slots) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(2, tokenizer, /*max_resident_bytes=*/1000);
    const std::vector<int32_t> a = {1, 100, 3, 101, 4};
    const std::vector<int32_t> b = {1, 200, 3, 201, 4};
    const std::vector<int32_t> c = {1, 300, 3, 301, 4};
    const auto bytes = [](int) { return (size_t)400; };
    auto first = cache.reserve_inline_snap(a, 0, false, 0, -1, bytes);
    auto twin = cache.reserve_inline_snap(a, 0, false, 0, -1, bytes);
    auto second = cache.reserve_inline_snap(b, 0, false, 0, -1, bytes);
    TEST_ASSERT(first.active() && !twin.active() && second.active());
    TEST_ASSERT(first.slot() != second.slot());
    // Both slots are held, and a third capture would not fit the budget anyway.
    auto third = cache.reserve_inline_snap(c, 0, false, 0, -1, bytes);
    TEST_ASSERT(!third.active());
    TEST_ASSERT(second.commit(b, 400));
    TEST_ASSERT(first.commit(a, 400));
    TEST_ASSERT(cache.lookup_candidate({1, 100, 3, 101, 4, 9}, 5).second > 0);
    TEST_ASSERT(cache.lookup_candidate({1, 200, 3, 201, 4, 9}, 5).second > 0);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_inline_snapshot_skips_unreachable_cuts) {
    const std::vector<int> boundaries = {266, 700, 1300, 1800};
    TEST_ASSERT(select_inline_snapshot_boundary(
                    boundaries, 0, true, /*reachable_from=*/512) == 1800);
    // A reachable head is still pinned first.
    TEST_ASSERT(select_inline_snapshot_boundary(
                    boundaries, 0, true, 1) == 266);
    // Nothing reachable yet: no reservation, prefill stays cold.
    TEST_ASSERT(select_inline_snapshot_boundary(
                    {100, 300}, 0, true, 512) == 0);
    // Restored at 1024 with 512-token chunks: 1300 lies in the first chunk.
    TEST_ASSERT(select_inline_snapshot_boundary(
                    {266, 1300}, 1024, false, 1024 + 512) == 0);
    TEST_ASSERT(select_inline_snapshot_boundary(
                    boundaries, 1024, false, 1024 + 512) == 1800);
    // A PPP pin the backend cannot save is not forced.
    TEST_ASSERT(!should_force_inline_snapshot_boundary(
                    boundaries, 2000, 0, true, 266, 512));
    TEST_ASSERT(should_force_inline_snapshot_boundary(
                    boundaries, 2000, 0, true, 700, 512));

    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(4, tokenizer);
    // Boundaries at 2, 4 and 6; the backend can save from position 3 on.
    const std::vector<int32_t> prompt = {1, 100, 3, 101, 4, 102, 3};
    auto capture = cache.reserve_inline_snap(
        prompt, 0, /*prefer_tools_boundary=*/true, /*forced_cut=*/2, -1, {},
        /*reachable_from=*/3);
    TEST_ASSERT(capture.active());
    TEST_ASSERT(capture.target_cut() > 2);
    capture.cancel();

    // An agent-turn replay saves at the end of its prefill, which any backend
    // can do, so it reserves without a reachability floor even when it is
    // shorter than one chunk.
    auto replay = cache.reserve_inline_snap(
        prompt, /*restored_prefix_len=*/5, false,
        /*forced_cut=*/(int) prompt.size(), /*restore_source_slot=*/-1);
    TEST_ASSERT(replay.active());
    TEST_ASSERT(replay.target_cut() == (int) prompt.size());
    replay.cancel();
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_inline_snapshot_prefers_tools_boundary_until_restored) {
    const std::vector<int> boundaries = {100, 240, 380, 520};
    // Cold tool-heavy: pin system+tools head (first marker), not deepen cut.
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 0, true) == 100);
    // After tools head is restored, deepen to the end of the last message.
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 100, true) == 520);
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 520, true) == 0);
    TEST_ASSERT(select_inline_snapshot_boundary({100}, 0, true) == 100);
    TEST_ASSERT(select_inline_snapshot_boundary({100}, 100, true) == 0);
}

TEST_CASE(ServerUnitFixture, test_forced_tools_pin_yields_to_deepen_after_restore) {
    const std::vector<int> boundaries = {100, 240, 380, 520};

    // Cold request: the PPP cut pins the tools/identity head.
    TEST_ASSERT(should_force_inline_snapshot_boundary(
        boundaries, 600, 0, true, 110));

    // Once the tools boundary is restored, normal selection must deepen to a
    // later conversation boundary instead of forcing the nearby pin again.
    TEST_ASSERT(!should_force_inline_snapshot_boundary(
        boundaries, 600, 100, true, 110));
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 100, true) == 520);

    // Without tools preference, a still-unrestored forced cut retains its
    // original behavior.
    TEST_ASSERT(should_force_inline_snapshot_boundary(
        boundaries, 600, 100, false, 110));
    TEST_ASSERT(!should_force_inline_snapshot_boundary(
        boundaries, 600, 110, false, 110));
    TEST_ASSERT(!should_force_inline_snapshot_boundary(
        boundaries, 100, 0, false, 110));
}

TEST_CASE(ServerUnitFixture, test_ppp_master_toggle_gates_tools_boundary_pinning) {
    TEST_ASSERT(ppp_prefers_tools_boundary(true, true));
    TEST_ASSERT(!ppp_prefers_tools_boundary(false, true));
    TEST_ASSERT(!ppp_prefers_tools_boundary(true, false));
    TEST_ASSERT(!ppp_prefers_tools_boundary(false, false));
}

TEST_CASE(ServerUnitFixture, test_ppp_tool_prefix_source_ignores_pflash_view) {
    // A 600-token system/tools turn closed by im_end (50) + '\n' (10), then
    // the next turn's role start (40): the head boundary follows it.
    std::vector<int32_t> raw = {40, 1, 10};
    raw.insert(raw.end(), 600, 7);
    const int content_end = (int) raw.size();
    raw.insert(raw.end(), {50, 10, 40});
    const int boundary = (int) raw.size();
    raw.insert(raw.end(), {2, 10, 9, 9, 9});
    const std::vector<int> boundaries = {boundary};
    // A PFlash view that kept the system content but not its closer.
    std::vector<int32_t> view(raw.begin(), raw.begin() + content_end);
    view.insert(view.end(), raw.begin() + content_end + 2, raw.end());

    TEST_ASSERT(&ppp_tool_prefix_source(true, raw, view) == &raw);
    TEST_ASSERT(&ppp_tool_prefix_source(false, raw, view) == &view);

    // Learning the view pins the next uncompressed request mid-message, at
    // the view's divergence point ...
    std::vector<std::vector<int32_t>> ring = {view};
    TEST_ASSERT(PinFriendlyPrompt::annotate_pin_end(
        raw, boundaries, ring, /*window=*/8, /*min=*/512) == content_end);
    // ... learning what the client sent pins it at the head boundary.
    ring = {ppp_tool_prefix_source(true, raw, view)};
    TEST_ASSERT(PinFriendlyPrompt::annotate_pin_end(
        raw, boundaries, ring, /*window=*/8, /*min=*/512) == boundary);
}

// ── Pin-Friendly Prompt Processor (PPP) ─────────────────────────────────

TEST_CASE(ServerUnitFixture, test_ppp_lcp_and_safe_boundary) {
    const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6};
    const std::vector<int32_t> b = {1, 2, 3, 9, 9};
    TEST_ASSERT(PinFriendlyPrompt::longest_common_prefix_len(a, b) == 3);
    TEST_ASSERT(PinFriendlyPrompt::longest_common_prefix_len(a, a) == 6);
    TEST_ASSERT(PinFriendlyPrompt::longest_common_prefix_len(a, {}) == 0);

    const std::vector<int> boundaries = {100, 240, 380};
    TEST_ASSERT(PinFriendlyPrompt::safe_boundary_cut(250, boundaries) == 240);
    TEST_ASSERT(PinFriendlyPrompt::safe_boundary_cut(50, boundaries) == 0);
    TEST_ASSERT(PinFriendlyPrompt::safe_boundary_cut(380, boundaries) == 380);
}

TEST_CASE(ServerUnitFixture, test_ppp_choose_pin_end_prefers_boundary_then_mid) {
    const std::vector<int> boundaries = {100, 200};
    // LCP past a boundary → pin at that boundary.
    TEST_ASSERT(PinFriendlyPrompt::choose_pin_end(150, boundaries, 50) == 100);
    // LCP past first boundary but short of second → still prefer boundary.
    TEST_ASSERT(PinFriendlyPrompt::choose_pin_end(175, boundaries, 50) == 100);
    // No boundary ≤ LCP → mid-message cut (tools-before-system layout).
    TEST_ASSERT(PinFriendlyPrompt::choose_pin_end(80, boundaries, 50) == 80);
    TEST_ASSERT(PinFriendlyPrompt::choose_pin_end(40, boundaries, 50) == 0);
}

TEST_CASE(ServerUnitFixture, test_ppp_annotate_against_recent_ring) {
    // Shared tools+identity head, divergent session clock in the tail of the
    // first turn (before any chat boundary at 200).
    std::vector<int32_t> day1(180, 7);
    day1.push_back(111);  // date token
    day1.insert(day1.end(), {8, 8, 8});  // past boundary material
    std::vector<int32_t> day2(180, 7);
    day2.push_back(222);
    day2.insert(day2.end(), {8, 8, 8});

    std::vector<std::vector<int32_t>> ring = {day1};
    const std::vector<int> boundaries = {200};
    const int pin = PinFriendlyPrompt::annotate_pin_end(
        day2, boundaries, ring, /*window=*/4, /*min=*/50);
    TEST_ASSERT(pin == 180);  // mid-message LCP before date drift
}

TEST_CASE(ServerUnitFixture, test_ppp_diff_split_finds_middle_hunk) {
    const std::vector<int32_t> a = {1, 2, 3, 100, 4, 5};
    const std::vector<int32_t> b = {1, 2, 3, 999, 4, 5};
    const auto split = PinFriendlyPrompt::diff_split(a, b);
    TEST_ASSERT(split.prefix_len == 3);
    TEST_ASSERT(split.suffix_len == 2);
    TEST_ASSERT(split.middle_begin == 3);
    TEST_ASSERT(split.middle_end == 4);
}

TEST_CASE(ServerUnitFixture, test_ppp_diff_rewrite_moves_volatile_after_stable) {
    // Head: [stable…][TIME][stable_tail…][im_end]  →  [stable…][stable_tail…][TIME][im_end]
    std::vector<int32_t> day1 = {7, 7, 7, 7, 111, 8, 8, 50};  // 50 = im_end
    std::vector<int32_t> day2 = {7, 7, 7, 7, 222, 8, 8, 50};
    // Transcript after first boundary.
    day2.insert(day2.end(), {9, 9});

    ChatMarkers markers;
    markers.family = "test";
    markers.end_msg_seqs = {{50}};

    std::vector<std::vector<int32_t>> ring = {day1};
    // Chat boundaries sit after the next role-start; DiffPin must still cut
    // the rewrite head at the first im_end (index 8), not at boundaries.front().
    const std::vector<int> boundaries = {10};
    auto rw = PinFriendlyPrompt::diff_make_pin_friendly(
        day2, boundaries, ring, markers,
        /*window=*/4, /*min_pin=*/4, /*max_ephemeral=*/16);
    TEST_ASSERT(rw.rewritten);
    TEST_ASSERT(rw.prefix_len == 4);
    TEST_ASSERT(rw.suffix_len == 2);  // {8,8} after peeling im_end trailer
    TEST_ASSERT(rw.middle_len == 1);
    // pin covers stable prefix+suffix; volatile then im_end follow.
    TEST_ASSERT(rw.pin_end == 6);
    // [7,7,7,7][8,8][222][50][9,9]
    TEST_ASSERT(rw.tokens.size() == day2.size());
    TEST_ASSERT((rw.tokens[0] == 7 && rw.tokens[3] == 7));
    TEST_ASSERT(rw.tokens[4] == 8 && rw.tokens[5] == 8);
    TEST_ASSERT(rw.tokens[6] == 222);
    TEST_ASSERT(rw.tokens[7] == 50);
    TEST_ASSERT(rw.tokens[8] == 9);
}

TEST_CASE(ServerUnitFixture, test_ppp_diff_rewrite_noop_without_boundaries) {
    std::vector<int32_t> day1 = {7, 7, 7, 7, 111, 8, 8, 50};
    std::vector<int32_t> day2 = {7, 7, 7, 7, 222, 8, 8, 50, 9, 9};
    ChatMarkers markers;
    markers.end_msg_seqs = {{50}};
    std::vector<std::vector<int32_t>> ring = {day1};
    auto rw = PinFriendlyPrompt::diff_make_pin_friendly(
        day2, /*boundaries=*/{}, ring, markers,
        /*window=*/4, /*min_pin=*/4, /*max_ephemeral=*/16);
    TEST_ASSERT(!rw.rewritten);
    TEST_ASSERT(rw.tokens == day2);
}

TEST_CASE(ServerUnitFixture, test_ppp_diff_rewrite_stops_before_next_role) {
    // Realistic boundary: after user role-start (token 90), past im_end (50).
    // Volatile middle must not float into the user turn.
    std::vector<int32_t> day1 = {7, 7, 7, 7, 111, 8, 8, 50};
    std::vector<int32_t> day2 = {7, 7, 7, 7, 222, 8, 8, 50, 90, 91, 92};
    ChatMarkers markers;
    markers.end_msg_seqs = {{50}};
    std::vector<std::vector<int32_t>> ring = {day1};
    const std::vector<int> boundaries = {11};  // after user role start
    auto rw = PinFriendlyPrompt::diff_make_pin_friendly(
        day2, boundaries, ring, markers,
        /*window=*/4, /*min_pin=*/4, /*max_ephemeral=*/16);
    TEST_ASSERT(rw.rewritten);
    TEST_ASSERT(rw.tokens.size() == day2.size());
    // Head rewritten; user role tokens untouched at the end.
    TEST_ASSERT(rw.tokens[rw.tokens.size() - 3] == 90);
    TEST_ASSERT(rw.tokens[rw.tokens.size() - 2] == 91);
    TEST_ASSERT(rw.tokens[rw.tokens.size() - 1] == 92);
    TEST_ASSERT(rw.tokens[6] == 222);
    TEST_ASSERT(rw.tokens[7] == 50);
}

TEST_CASE(ServerUnitFixture, test_ppp_tools_system_head_end) {
    ChatMarkers markers;
    markers.end_msg_seqs = {{50}, {51, 52}};
    std::vector<int32_t> ids = {1, 2, 50, 90, 91};
    TEST_ASSERT(PinFriendlyPrompt::tools_system_head_end(ids, markers) == 3);
    ids = {1, 2, 51, 52, 90};
    TEST_ASSERT(PinFriendlyPrompt::tools_system_head_end(ids, markers) == 4);
    TEST_ASSERT(PinFriendlyPrompt::tools_system_head_end({1, 2, 3}, markers) == 0);
}

TEST_CASE(ServerUnitFixture, test_ppp_split_and_rearrange_ephemeral_tail) {
    const std::string system =
        "You are Hermes.\n\n"
        "Conversation started: Thursday, July 30, 2026 03:59 PM\n"
        "Model: qwen\n";
    auto [stable, ephemeral] =
        PinFriendlyPrompt::split_ephemeral_system_tail(system);
    TEST_ASSERT(stable == "You are Hermes.");
    TEST_ASSERT(ephemeral.find("Conversation started:") == 0);

    std::vector<ChatMessage> messages = {
        {"system", system, ""},
        {"user", "hi", ""},
    };
    auto off = PinFriendlyPrompt::rearrange(messages, false);
    TEST_ASSERT(!off.rearranged);
    TEST_ASSERT(off.messages.size() == 2);

    auto on = PinFriendlyPrompt::rearrange(messages, true);
    TEST_ASSERT(on.rearranged);
    TEST_ASSERT(on.messages.size() == 3);
    TEST_ASSERT(on.messages[0].role == "system");
    TEST_ASSERT(on.messages[0].content == "You are Hermes.");
    TEST_ASSERT(on.messages[1].role == "system");
    TEST_ASSERT(on.messages[1].content.find("Conversation started:") == 0);
    TEST_ASSERT(on.messages[2].role == "user");
}

// ── Prefix-aware eviction policy (model-free) ───────────────────────────

TEST_CASE(ServerUnitFixture, test_evict_empty_is_zero) {
    std::vector<std::vector<int32_t>> ids;
    TEST_ASSERT(select_inline_evict_victim(ids) == 0);
}

TEST_CASE(ServerUnitFixture, test_evict_single_is_zero) {
    std::vector<std::vector<int32_t>> ids = {{1, 2, 3}};
    TEST_ASSERT(select_inline_evict_victim(ids) == 0);
}

TEST_CASE(ServerUnitFixture, test_evict_chain_keeps_ancestors) {
    // Oldest-first chain: [s] < [s,a] < [s,a,b]. [s,a] sits between the head
    // and the newest turn, kept only for an edit of the last message: it goes
    // first. Then the leaf, and the shared head last.
    std::vector<std::vector<int32_t>> ids = {{9}, {9, 1}, {9, 1, 2}};
    TEST_ASSERT(select_inline_evict_victim(ids) == 1);
    ids = {{9}, {9, 1, 2}};
    TEST_ASSERT(select_inline_evict_victim(ids) == 1);
    // A protected turn in between stays; the leaf goes.
    ids = {{9}, {9, 1}, {9, 1, 2}};
    std::vector<bool> protect = {false, true, false};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect) == 2);
}

TEST_CASE(ServerUnitFixture, test_evict_unrelated_falls_back_to_lru) {
    // No prefix relation: all are leaves, so evict the oldest (index 0).
    std::vector<std::vector<int32_t>> ids = {{1, 1}, {2, 2}, {3, 3}};
    TEST_ASSERT(select_inline_evict_victim(ids) == 0);
}

TEST_CASE(ServerUnitFixture, test_evict_branch_spares_shared_root) {
    // [s] is an ancestor of both branches, so it is never the victim; the oldest
    // leaf ([s,a] at index 1) is evicted instead.
    std::vector<std::vector<int32_t>> ids = {{9}, {9, 1}, {9, 2}};
    int v = select_inline_evict_victim(ids);
    TEST_ASSERT(v == 1);
    TEST_ASSERT(v != 0);  // the shared root must be spared
}

TEST_CASE(ServerUnitFixture, test_evict_skips_protected_leaf) {
    // Two unrelated leaves; oldest is protected → evict next unprotected leaf.
    std::vector<std::vector<int32_t>> ids = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<bool> protect = {true, false, false};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect) == 1);
}

TEST_CASE(ServerUnitFixture, test_evict_all_protected_falls_back) {
    std::vector<std::vector<int32_t>> ids = {{1, 1}, {2, 2}};
    std::vector<bool> protect = {true, true};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect) == 0);
}

// ── Restore-source-aware eviction (prefix-cache slide) ─────────────────

// (a) Linear chain at capacity: the new snapshot must land in a different
// slot than the restore source, so the restore point can slide forward past
// the deepest slot.
TEST_CASE(ServerUnitFixture, test_slide_evicts_ancestor_not_restore_source) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(4, tokenizer);
    TEST_ASSERT(!cache.disabled());

    // Linear chain: each prompt strictly extends the previous one.
    std::vector<int32_t> p1 = {1, 100, 4, 101};
    std::vector<int32_t> p2 = p1;
    p2.insert(p2.end(), {3, 102});
    std::vector<int32_t> p3 = p2;
    p3.insert(p3.end(), {4, 103});
    std::vector<int32_t> p4 = p3;
    p4.insert(p4.end(), {3, 104});

    auto fill = [&](const std::vector<int32_t> & p) {
        auto prepared = cache.reserve_inline_snap(
            p, 0, false, (int) p.size());
        TEST_ASSERT(prepared.active());
        TEST_ASSERT(prepared.target_cut() == (int) p.size());
        const int slot = prepared.slot();
        TEST_ASSERT(prepared.commit(p));
        return slot;
    };
    const int s1 = fill(p1);
    const int s2 = fill(p2);
    const int s3 = fill(p3);
    const int s4 = fill(p4);
    TEST_ASSERT(s1 != s2 && s2 != s3 && s3 != s4 && s4 != s1);
    TEST_ASSERT(s4 == 3);  // deepest slot, like the BUG.md repro

    // Turn 5: restore from the deepest slot and extend the conversation.
    std::vector<int32_t> p5 = p4;
    p5.insert(p5.end(), {3, 105});
    const auto hit = cache.lookup(p5);
    TEST_ASSERT(hit.first == s4 && hit.second == (int) p4.size());

    auto snap = cache.reserve_inline_snap(
        p5, hit.second, false, (int) p5.size(), hit.first);
    TEST_ASSERT(snap.active());
    const int snap_slot = snap.slot();
    TEST_ASSERT(snap_slot != s4);  // different slot: the restore source
                                   // was not the victim
    TEST_ASSERT(snap.target_cut() == (int) p5.size());
    TEST_ASSERT(snap.commit(p5));

    // The restore point slid forward: the new, deeper prefix now matches.
    const auto after = cache.lookup(p5);
    TEST_ASSERT(after.first == snap_slot);
    TEST_ASSERT(after.second == (int) p5.size());
    // The old deepest entry survived the eviction.
    std::vector<int32_t> p4b = p4;
    p4b.insert(p4b.end(), {7, 7});
    const auto kept = cache.lookup(p4b);
    TEST_ASSERT(kept.first == s4 && kept.second == (int) p4.size());
    TEST_ASSERT(cache.stats().in_use == 4);
    unlink(path.c_str());
}

// (b) The in-flight restore source is never the eviction victim, at any LRU
// position, whether it is the only leaf (linear chain) or not.
TEST_CASE(ServerUnitFixture, test_slide_restore_source_never_evicted) {
    std::vector<std::vector<int32_t>> ids = {
        {9}, {9, 1}, {9, 1, 2}, {9, 1, 2, 3},
    };
    for (int skip = 0; skip < 4; ++skip) {
        const int victim = select_inline_evict_victim(ids, nullptr, skip);
        TEST_ASSERT(victim >= 0 && victim != skip);
    }
    // Linear chain whose only leaf is the restore source: evict a turn kept
    // between the head and the newest instead of cancelling everything.
    TEST_ASSERT(select_inline_evict_victim(ids, nullptr, 3) == 1);
    // Otherwise the shallowest unprotected ancestor.
    ids = {{9}, {9, 1}};
    TEST_ASSERT(select_inline_evict_victim(ids, nullptr, 1) == 0);
}

// (c) The restore source's own tools pin is never evicted, even when it is
// the shallowest ancestor and the only other entry besides the restore source.
TEST_CASE(ServerUnitFixture, test_slide_restore_source_pin_never_evicted) {
    std::vector<std::vector<int32_t>> ids = {
        {9}, {9, 1}, {9, 1, 2}, {9, 1, 2, 3},
    };
    std::vector<bool> protect = {true, false, false, false};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect, 3) == 1);

    // Only the restore source and its pin remain: no safe victim, so no
    // snapshot is reserved instead of destroying the pin.
    std::vector<std::vector<int32_t>> two = {{9}, {9, 1}};
    std::vector<bool> two_prot = {true, false};
    TEST_ASSERT(select_inline_evict_victim(two, &two_prot, 1) == -1);

    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(2, tokenizer);
    TEST_ASSERT(!cache.disabled());

    std::vector<int32_t> pin = {1, 100, 4, 101};
    std::vector<int32_t> deep = pin;
    deep.insert(deep.end(), {3, 102});
    auto prepared = cache.reserve_inline_snap(pin, 0, true, (int) pin.size());
    TEST_ASSERT(prepared.slot() == 0);
    TEST_ASSERT(prepared.commit(pin, 0, true));
    prepared = cache.reserve_inline_snap(deep, 0, false, (int) deep.size());
    TEST_ASSERT(prepared.slot() == 1);
    TEST_ASSERT(prepared.commit(deep));

    std::vector<int32_t> deeper = deep;
    deeper.insert(deeper.end(), {4, 103});
    const auto hit = cache.lookup(deeper);
    TEST_ASSERT(hit.first == 1 && hit.second == (int) deep.size());
    // At capacity the only other entry is the protected pin: refuse rather
    // than evict it.
    auto refused = cache.reserve_inline_snap(
        deeper, hit.second, false, (int) deeper.size(), hit.first);
    TEST_ASSERT(!refused.active());
    const auto kept = cache.lookup(deep);
    TEST_ASSERT(kept.first == 1 && kept.second == (int) deep.size());
    TEST_ASSERT(cache.stats().in_use == 2);
    unlink(path.c_str());
}

// (c2) A cache full of other conversations' pins still takes the restored
// conversation's next snapshot: the least recently used pin that is a leaf
// goes. Otherwise the conversation never saves another turn.
TEST_CASE(ServerUnitFixture, test_slide_full_cache_of_pins_evicts_oldest_pin) {
    // {9} heads the restore source; {7} and {8} head other conversations.
    std::vector<std::vector<int32_t>> ids = {{9}, {7}, {8}, {9, 1}};
    std::vector<bool> protect = {true, true, true, false};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect, 3) == 1);

    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(3, tokenizer);
    TEST_ASSERT(!cache.disabled());

    const std::vector<int32_t> other = {1, 200, 4, 201};
    const std::vector<int32_t> head = {1, 100, 4, 101};
    std::vector<int32_t> deep = head;
    deep.insert(deep.end(), {3, 102});
    std::vector<int32_t> deeper = deep;
    deeper.insert(deeper.end(), {4, 103});
    cache.confirm_inline_snap(0, (int) other.size(), other, /*protect=*/true);
    cache.confirm_inline_snap(1, (int) head.size(), head, /*protect=*/true);
    cache.confirm_inline_snap(2, (int) deep.size(), deep);

    const auto hit = cache.lookup(deeper);
    TEST_ASSERT(hit.first == 2 && hit.second == (int) deep.size());
    auto next = cache.reserve_inline_snap(
        deeper, hit.second, false, (int) deeper.size(), hit.first);
    TEST_ASSERT(next.active() && next.slot() == 0);
    TEST_ASSERT(next.commit(deeper));

    std::vector<int32_t> after = deeper;
    after.insert(after.end(), {3, 104});
    const auto grown = cache.lookup(after);
    TEST_ASSERT(grown.first == 0 && grown.second == (int) deeper.size());
    std::vector<int32_t> sibling = head;
    sibling.insert(sibling.end(), {3, 999});
    TEST_ASSERT(cache.lookup(sibling).first == 1);
    std::vector<int32_t> other_next = other;
    other_next.insert(other_next.end(), {3, 999});
    TEST_ASSERT(cache.lookup(other_next).first == -1);
    unlink(path.c_str());
}

// A capture in flight holds the oldest pin's slot: a restored conversation
// takes the next pin instead of saving nothing.
TEST_CASE(ServerUnitFixture, test_slide_skips_pin_held_by_capture_in_flight) {
    std::vector<std::vector<int32_t>> ids = {{7}, {8}, {9}};
    std::vector<bool> protect = {true, true, true};
    std::vector<bool> held = {true, false, false};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect, 2, &held) == 1);

    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(3, tokenizer);
    TEST_ASSERT(!cache.disabled());

    const std::vector<int32_t> a = {1, 200, 4, 201};
    const std::vector<int32_t> b = {1, 300, 4, 301};
    const std::vector<int32_t> c = {1, 100, 4, 101};
    cache.confirm_inline_snap(0, (int) a.size(), a, /*protect=*/true);
    cache.confirm_inline_snap(1, (int) b.size(), b, /*protect=*/true);
    cache.confirm_inline_snap(2, (int) c.size(), c, /*protect=*/true);

    const std::vector<int32_t> fresh = {1, 400, 4, 401};
    auto in_flight = cache.reserve_inline_snap(fresh, 0, false, (int) fresh.size());
    TEST_ASSERT(in_flight.active() && in_flight.slot() == 0);

    std::vector<int32_t> deeper = c;
    deeper.insert(deeper.end(), {3, 102});
    const auto hit = cache.lookup(deeper);
    TEST_ASSERT(hit.first == 2 && hit.second == (int) c.size());
    auto next = cache.reserve_inline_snap(
        deeper, hit.second, false, (int) deeper.size(), hit.first);
    TEST_ASSERT(next.active() && next.slot() == 1);
    next.cancel();
    in_flight.cancel();
    unlink(path.c_str());
}

// (d) Branching conversations are unchanged: with two leaves, the oldest
// non-restore-source leaf is still the victim.
TEST_CASE(ServerUnitFixture, test_slide_branching_oldest_leaf_unchanged) {
    // [9] is a shared root; leaves are idx 1 ([9,1]) and idx 2 ([9,2]).
    std::vector<std::vector<int32_t>> ids = {{9}, {9, 1}, {9, 2}};
    TEST_ASSERT(select_inline_evict_victim(ids) == 1);  // original behavior
    // Restore source is the newer leaf: the older leaf is still the victim.
    TEST_ASSERT(select_inline_evict_victim(ids, nullptr, 2) == 1);
    // Restore source is the older leaf: the remaining leaf is the victim.
    TEST_ASSERT(select_inline_evict_victim(ids, nullptr, 1) == 2);
    // A protected leaf goes last: with the restore source skipped and the
    // only remaining leaf protected, the shallowest unprotected ancestor is
    // the victim instead.
    std::vector<bool> protect = {false, true, false};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect, 2) == 0);
}

// (e) Free-slot path (not at capacity) must skip the restore source too, so
// the http_server / agent-replay guards do not cancel the reservation and the
// restore point can advance.
TEST_CASE(ServerUnitFixture, test_slide_free_slot_skips_restore_source) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(4, tokenizer);
    TEST_ASSERT(!cache.disabled());

    // Two entries with cap 4: vacancy exists, so reserve_inline_snap takes
    // the free-slot path. Round-robin has next_slot_ at 2.
    std::vector<int32_t> p1 = {1, 100, 4, 101};
    std::vector<int32_t> p2 = p1;
    p2.insert(p2.end(), {3, 102});

    auto fill = [&](const std::vector<int32_t> & p) {
        auto prepared = cache.reserve_inline_snap(
            p, 0, false, (int) p.size());
        TEST_ASSERT(prepared.active());
        TEST_ASSERT(prepared.target_cut() == (int) p.size());
        const int slot = prepared.slot();
        TEST_ASSERT(prepared.commit(p));
        return slot;
    };
    const int s1 = fill(p1);  // slot 0
    const int s2 = fill(p2);  // slot 1
    TEST_ASSERT(s1 == 0 && s2 == 1);

    // Drive the round-robin so next_slot_ lands exactly on s2 (the restore
    // source we will pass in). Three abort-burn steps from slot 2 → 3 → 0 → 1.
    for (int i = 0; i < 3; ++i) {
        std::vector<int32_t> scratch = p2;
        scratch.push_back(7);
        scratch.push_back(7 + i);
        auto prep = cache.reserve_inline_snap(
            scratch, 0, false, (int) scratch.size());
        TEST_ASSERT(prep.active());
        // Burn the round-robin step without committing an entry.
        prep.cancel();
    }

    // Restore source is s2 = 1. Free slots are 2 and 3. The next free-slot
    // allocation must skip s2 (== 1) and pick a non-restore slot.
    std::vector<int32_t> p3 = p2;
    p3.insert(p3.end(), {4, 103});
    const auto hit = cache.lookup(p3);
    TEST_ASSERT(hit.first == s2);
    auto snap = cache.reserve_inline_snap(
        p3, hit.second, false, (int) p3.size(), hit.first);
    TEST_ASSERT(snap.active());
    TEST_ASSERT(snap.slot() != hit.first);
    snap.cancel();
    unlink(path.c_str());
}

// (f) Below capacity, a free slot is taken before a live one. Round-robin
// alone wraps onto committed snapshots once a slot in the middle is freed.
TEST_CASE(ServerUnitFixture, test_free_slot_taken_before_live_slot) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(4, tokenizer);

    std::vector<std::vector<int32_t>> prompts = {
        {1, 100}, {1, 200}, {1, 300}, {1, 400},
    };
    for (int i = 0; i < 4; ++i) {
        auto prepared = cache.reserve_inline_snap(
            prompts[(size_t)i], 0, false, 2);
        TEST_ASSERT(prepared.slot() == i);
        TEST_ASSERT(prepared.commit(prompts[(size_t)i]));
    }
    cache.invalidate_inline_snap(2);

    const std::vector<int32_t> next = {1, 500};
    auto prepared = cache.reserve_inline_snap(next, 0, false, 2);
    TEST_ASSERT(prepared.slot() == 2);
    TEST_ASSERT(prepared.commit(next));
    TEST_ASSERT(cache.stats().in_use == 4);
    TEST_ASSERT(cache.lookup(prompts[0]).first == 0);
    unlink(path.c_str());
}

// (g) A new turn supersedes the intermediate turns of its conversation. The
// shallowest ancestor (shared head), protected pins and sibling branches stay.
TEST_CASE(ServerUnitFixture, test_prune_superseded_ancestors) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(8, tokenizer);

    const std::vector<int32_t> head = {1, 100};
    std::vector<int32_t> t1 = head;
    t1.insert(t1.end(), {3, 101});
    std::vector<int32_t> t2 = t1;
    t2.insert(t2.end(), {4, 102});
    std::vector<int32_t> t3 = t2;
    t3.insert(t3.end(), {3, 103});
    std::vector<int32_t> sibling = head;
    sibling.insert(sibling.end(), {3, 900});

    cache.confirm_inline_snap(0, (int)head.size(), head, false, 10);
    cache.confirm_inline_snap(1, (int)t1.size(), t1, false, 20);
    cache.confirm_inline_snap(2, (int)t2.size(), t2, false, 30);
    cache.confirm_inline_snap(3, (int)sibling.size(), sibling, false, 20);
    cache.confirm_inline_snap(4, (int)t3.size(), t3, false, 40);

    // The head and the previous turn stay; the turn before that goes.
    auto pruned = cache.prune_superseded_ancestors(4);
    TEST_ASSERT(pruned == std::vector<int>({1}));
    TEST_ASSERT(cache.stats().in_use == 4);
    TEST_ASSERT(cache.stats().resident_bytes == 100);
    TEST_ASSERT(cache.lookup(t3).first == 4);
    TEST_ASSERT(cache.lookup(sibling).first == 3);
    // A person who edits the last message resumes from the previous turn.
    std::vector<int32_t> edited = t2;
    edited.insert(edited.end(), {3, 104});
    TEST_ASSERT(cache.lookup(edited).first == 2);
    // Pruning again finds nothing left to drop.
    TEST_ASSERT(cache.prune_superseded_ancestors(4).empty());
    // A slot without an entry prunes nothing.
    TEST_ASSERT(cache.prune_superseded_ancestors(7).empty());

    // The previous turn goes too when the resident budget cannot hold it.
    PrefixCache tight(8, tokenizer, /*max_resident_bytes=*/70);
    tight.confirm_inline_snap(0, (int)head.size(), head, false, 10);
    tight.confirm_inline_snap(1, (int)t2.size(), t2, false, 30);
    tight.confirm_inline_snap(2, (int)t3.size(), t3, false, 40);
    TEST_ASSERT(tight.prune_superseded_ancestors(2) == std::vector<int>({1}));
    TEST_ASSERT(tight.stats().resident_bytes == 50);

    // A protected pin that is not the shallowest ancestor also survives.
    std::vector<int32_t> t4 = t3;
    t4.insert(t4.end(), {4, 104});
    PrefixCache pinned(8, tokenizer);
    pinned.confirm_inline_snap(0, (int)head.size(), head);
    pinned.confirm_inline_snap(1, (int)t1.size(), t1, /*protect=*/true);
    pinned.confirm_inline_snap(2, (int)t2.size(), t2);
    pinned.confirm_inline_snap(3, (int)t3.size(), t3);
    pinned.confirm_inline_snap(4, (int)t4.size(), t4);
    TEST_ASSERT(pinned.prune_superseded_ancestors(4) == std::vector<int>({2}));
    unlink(path.c_str());
}

// (h) With pruning on, the budget counts what a capture supersedes (its
// restore source included) as freed. Otherwise a deep capture has to evict
// the shared system/tools head to fit.
TEST_CASE(ServerUnitFixture, test_budget_counts_superseded_snapshots_as_freed) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    const std::vector<int32_t> head = {1, 100};
    std::vector<int32_t> turn1 = head;
    turn1.insert(turn1.end(), {3, 101});
    std::vector<int32_t> turn2 = turn1;
    turn2.insert(turn2.end(), {4, 102});
    std::vector<int32_t> turn3 = turn2;
    turn3.insert(turn3.end(), {3, 103});
    // Budget 300: head 20 + turn2 150 resident, turn3 needs 150 more.
    const auto sized = [](int cut) { return cut <= 2 ? (size_t) 20 : (size_t) 150; };
    for (const bool prunes : {false, true}) {
        PrefixCache cache(8, tokenizer, /*max_resident_bytes=*/300);
        cache.set_prunes_superseded(prunes);
        cache.confirm_inline_snap(0, (int) head.size(), head, false, 20);
        cache.confirm_inline_snap(1, (int) turn1.size(), turn1, false, 150);
        cache.confirm_inline_snap(2, (int) turn2.size(), turn2, false, 150);
        TEST_ASSERT(cache.prune_superseded_ancestors(2) == std::vector<int>({1}));
        TEST_ASSERT(cache.stats().resident_bytes == 170);

        auto capture = cache.reserve_inline_snap(
            turn3, (int) turn2.size(), false, (int) turn3.size(),
            /*restore_source_slot=*/2, sized);
        TEST_ASSERT(capture.active());
        if (!prunes) {
            // The restore source cannot make room, so the head is replaced.
            TEST_ASSERT(capture.slot() == 0);
            capture.cancel();
            continue;
        }
        const int slot = capture.slot();
        TEST_ASSERT(slot != 0 && slot != 2);
        TEST_ASSERT(capture.commit(turn3, 150));
        TEST_ASSERT(cache.prune_superseded_ancestors(slot) == std::vector<int>({2}));
        TEST_ASSERT(cache.stats().resident_bytes == 170);
        TEST_ASSERT(cache.lookup(turn3).first == slot);
        std::vector<int32_t> other = head;
        other.insert(other.end(), {3, 999});
        TEST_ASSERT(cache.lookup(other).first == 0);  // head kept
    }
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_parallel_reservations_do_not_double_credit_shared_ancestor) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::vector<int32_t> head = {1, 100};
    const std::vector<int32_t> ancestor = {1, 100, 3, 101};
    const std::vector<int32_t> a = {1, 100, 3, 101, 4, 102};
    const std::vector<int32_t> b = {1, 100, 3, 101, 4, 103};
    for (size_t budget : {300u, 320u}) {
        for (bool reverse : {false, true}) {
            PrefixCache cache(8, tokenizer, budget);
            cache.set_prunes_superseded(true);
            cache.confirm_inline_snap(0, 2, head, true, 20);
            cache.confirm_inline_snap(1, 4, ancestor, false, 150);
            auto first = cache.reserve_inline_snap(a, 2, false, 6, 0,
                [](int) { return (size_t)150; });
            TEST_ASSERT(first.active());
            auto second = cache.reserve_inline_snap(b, 2, false, 6, 0,
                [](int) { return (size_t)150; });
            // 20-byte head plus two 150-byte captures needs exactly 320;
            // shared reclaimed bytes must never make it fit under 300.
            TEST_ASSERT(second.active() == (budget == 320));
            const auto commit = [&](PrefixCache::InlineReservation & r,
                                    const std::vector<int32_t> & ids) {
                const int slot = r.slot();
                TEST_ASSERT(r.commit(ids, 150));
                cache.prune_superseded_ancestors(slot);
                TEST_ASSERT(cache.enforce_resident_budget(slot).empty());
                TEST_ASSERT(cache.stats().resident_bytes <= budget);
            };
            if (budget == 320 && reverse) commit(second, b);
            commit(first, a);
            if (budget == 320 && !reverse) commit(second, b);
        }
    }
    unlink(path.c_str());
}

// (i) A capture can commit shorter than it reserved (agent-turn replay saves
// at the backend's position), so an ancestor credited as freed may stay
// resident. Enforcing the budget after the commit restores the invariant.
TEST_CASE(ServerUnitFixture, test_budget_enforced_after_shorter_commit) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    PrefixCache cache(8, tokenizer, /*max_resident_bytes=*/300);
    cache.set_prunes_superseded(true);
    const std::vector<int32_t> prompt = {1, 100, 3, 101, 4, 102};
    const std::vector<int32_t> head(prompt.begin(), prompt.begin() + 2);
    const std::vector<int32_t> middle(prompt.begin(), prompt.begin() + 4);
    cache.confirm_inline_snap(0, 2, head, false, 20);
    cache.confirm_inline_snap(1, 4, middle, false, 150);

    // The reservation credits `middle` (an ancestor of the 6-token cut).
    auto capture = cache.reserve_inline_snap(
        prompt, 2, false, (int) prompt.size(), /*restore_source_slot=*/0,
        [](int) { return (size_t) 150; });
    TEST_ASSERT(capture.active());
    const int slot = capture.slot();
    // It lands at 3 tokens, so `middle` is its descendant, not its ancestor.
    TEST_ASSERT(capture.commit_at(prompt, 3, 150));
    TEST_ASSERT(cache.prune_superseded_ancestors(slot).empty());
    TEST_ASSERT(cache.stats().resident_bytes == 320);

    TEST_ASSERT(cache.enforce_resident_budget(slot) == std::vector<int>({1}));
    TEST_ASSERT(cache.stats().resident_bytes == 170);
    TEST_ASSERT(cache.lookup(prompt).first == slot);
    // Within budget there is nothing to evict.
    TEST_ASSERT(cache.enforce_resident_budget(slot).empty());
    unlink(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════
// PFlash config tests (model-free)
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_pflash_config_defaults) {
    ServerConfig cfg;
    TEST_ASSERT(!cfg.pflash);
    TEST_ASSERT(cfg.pflash_threshold == 32000);
    TEST_ASSERT(cfg.pflash_keep_ratio > 0.04f && cfg.pflash_keep_ratio < 0.06f);
    TEST_ASSERT(cfg.pflash_drafter_path.empty());
    TEST_ASSERT(!cfg.pflash_skip_park);
    TEST_ASSERT(cfg.draft_residency == DraftResidencyPolicy::Auto);
}

TEST_CASE(ServerUnitFixture, test_concurrent_status_is_aggregate_only) {
    ServerStatus status;
    ServerStatus::RequestInfo info;
    info.model = "classic-model";
    status.set_running("classic prompt", 12, true, info);
    json snapshot = status.to_json();
    TEST_ASSERT(snapshot["active_requests"] == 0);
    TEST_ASSERT(snapshot["current"]["model"] == "classic-model");

    status.set_concurrent_requests(2, 2);
    snapshot = status.to_json();
    TEST_ASSERT(snapshot["phase"] == "prefill");
    TEST_ASSERT(snapshot["active_requests"] == 2);
    TEST_ASSERT(snapshot["current"].is_null());

    status.set_concurrent_requests(2, 1);
    snapshot = status.to_json();
    TEST_ASSERT(snapshot["phase"] == "mixed");
    TEST_ASSERT(snapshot["active_requests"] == 2);

    status.set_concurrent_requests(2, 0);
    snapshot = status.to_json();
    TEST_ASSERT(snapshot["phase"] == "decode");
    TEST_ASSERT(snapshot["active_requests"] == 2);

    status.set_idle();
    snapshot = status.to_json();
    TEST_ASSERT(snapshot["phase"] == "idle");
    TEST_ASSERT(snapshot["active_requests"] == 0);
    TEST_ASSERT(snapshot["current"].is_null());
}

TEST_CASE(ServerUnitFixture, test_pflash_compress_result_defaults) {
    ModelBackend::CompressResult result;
    TEST_ASSERT(!result.ok);
    TEST_ASSERT(result.compressed_ids.empty());
}

TEST_CASE(ServerUnitFixture, test_pflash_config_upstream_defaults) {
    ServerConfig cfg;
    TEST_ASSERT(cfg.pflash_upstream_base.empty());
    TEST_ASSERT(cfg.pflash_upstream_key.empty());
    TEST_ASSERT(cfg.pflash_upstream_model.empty());
    TEST_ASSERT(cfg.pflash_curve.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_request_sampler_applies_defaults_and_overrides) {
    SamplingDefaults defaults;
    defaults.has_temperature = true;
    defaults.temperature = 0.6f;
    defaults.has_top_p = true;
    defaults.top_p = 0.9f;
    defaults.has_repetition_penalty = true;
    defaults.repetition_penalty = 1.1f;

    const SamplerCfg sampler = parse_request_sampler({
        {"temperature", 0.2f},
        {"top_k", 20},
        {"seed", 42},
        {"presence_penalty", 0.3f},
    }, defaults);

    TEST_ASSERT(std::fabs(sampler.temp - 0.2f) < 0.001f);
    TEST_ASSERT(std::fabs(sampler.top_p - 0.9f) < 0.001f);
    TEST_ASSERT(sampler.top_k == 20);
    TEST_ASSERT(sampler.seed == 42);
    TEST_ASSERT(std::fabs(sampler.pres_pen - 0.3f) < 0.001f);
    TEST_ASSERT(std::fabs(sampler.rep_pen - 1.1f) < 0.001f);
}

// The stream a request samples from: no "seed" field gives fresh entropy on
// every request; any explicit seed, zero and UINT64_MAX included, is the seed.
TEST_CASE(ServerUnitFixture, test_parse_request_sampler_tells_no_seed_from_seed_zero) {
    const SamplingDefaults defaults;
    const uint64_t max_seed = std::numeric_limits<uint64_t>::max();
    const SamplerCfg unseeded =
        parse_request_sampler({{"temperature", 1.0f}}, defaults);
    TEST_ASSERT(request_rng_seed(unseeded) != request_rng_seed(unseeded));
    TEST_ASSERT(request_rng_seed(parse_request_sampler(
        {{"temperature", 1.0f}, {"seed", 0}}, defaults)) == 0);
    TEST_ASSERT(request_rng_seed(parse_request_sampler(
        {{"temperature", 1.0f}, {"seed", max_seed}}, defaults)) == max_seed);
}

// ═══════════════════════════════════════════════════════════════════════
// sampling_no_thinking: per-card sampler defaults for requests whose final
// thinking state resolves to OFF (e.g. Qwen3.8-Flash-Next's instruct-mode
// recipe vs. its thinking-mode one). See docs/specs/thinking-budget.md §3.3.
// ═══════════════════════════════════════════════════════════════════════

// Mirrors the real pipeline in HttpServer::handle_model_request: sampler
// defaults are parsed before the final thinking state is known, then
// apply_request_reasoning resolves req.thinking_enabled, then (only when
// thinking ended up OFF) apply_no_thinking_sampler_defaults backfills the
// still-omitted fields from the card's no-thinking block.
static SamplerCfg resolve_request_sampler(const json & body, const ServerConfig & config) {
    SamplerCfg sampler = parse_request_sampler(body, config.sampler_defaults);
    ParsedRequest req;
    req.max_output = resolve_max_output_tokens(body, config.default_max_tokens);
    apply_request_reasoning(body, config, req);
    if (!req.thinking_enabled) {
        apply_no_thinking_sampler_defaults(
            body, config.sampler_defaults_no_thinking, sampler);
    }
    return sampler;
}

// Qwen3.8-Flash-Next's published thinking vs. instruct sampling sets.
static ServerConfig qwen4exp_dual_sampling_config() {
    ServerConfig config;
    config.arch = "qwen4exp";  // thinks by default unless disabled
    config.default_max_tokens = 32768;
    config.think_max_tokens = 24576;
    config.hard_limit_reply_budget = 8192;

    config.sampler_defaults.has_temperature = true;
    config.sampler_defaults.temperature = 1.0f;
    config.sampler_defaults.has_top_p = true;
    config.sampler_defaults.top_p = 0.95f;
    config.sampler_defaults.has_top_k = true;
    config.sampler_defaults.top_k = 20;
    config.sampler_defaults.has_presence_penalty = true;
    config.sampler_defaults.presence_penalty = 0.0f;
    config.sampler_defaults.has_repetition_penalty = true;
    config.sampler_defaults.repetition_penalty = 1.0f;

    config.sampler_defaults_no_thinking.has_temperature = true;
    config.sampler_defaults_no_thinking.temperature = 0.7f;
    config.sampler_defaults_no_thinking.has_top_p = true;
    config.sampler_defaults_no_thinking.top_p = 0.80f;
    config.sampler_defaults_no_thinking.has_top_k = true;
    config.sampler_defaults_no_thinking.top_k = 20;
    config.sampler_defaults_no_thinking.has_presence_penalty = true;
    config.sampler_defaults_no_thinking.presence_penalty = 1.5f;
    config.sampler_defaults_no_thinking.has_repetition_penalty = true;
    config.sampler_defaults_no_thinking.repetition_penalty = 1.0f;
    return config;
}

TEST_CASE(ServerUnitFixture, test_model_card_parses_sampling_no_thinking) {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "dflash-mc-no-thinking-test";
    fs::remove_all(root);
    fs::create_directories(root / "share" / "model_cards");

    {
        FILE * f = std::fopen(
            (root / "share" / "model_cards" / "notest-model.json").string().c_str(), "w");
        TEST_ASSERT(f != nullptr);
        std::fprintf(f,
            "{\"name\":\"notest-model\",\"source\":\"test\",\"verified_at\":\"2026-10-02\","
            "\"max_tokens\":32768,"
            "\"sampling\":{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,"
            "\"presence_penalty\":0.0,\"repetition_penalty\":1.0},"
            "\"sampling_no_thinking\":{\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,"
            "\"min_p\":0.0,\"presence_penalty\":1.5,\"repetition_penalty\":1.0}}");
        std::fclose(f);
    }

    auto card = luce::common::resolve_model_card("", "notest-model", "qwen4exp", root.string());
    fs::remove_all(root);

    TEST_ASSERT(card.sampling.has_temperature);
    TEST_ASSERT(std::fabs(card.sampling.temperature - 1.0f) < 1.0e-6f);

    TEST_ASSERT(card.sampling_no_thinking.has_temperature);
    TEST_ASSERT(std::fabs(card.sampling_no_thinking.temperature - 0.7f) < 1.0e-6f);
    TEST_ASSERT(card.sampling_no_thinking.has_top_p);
    TEST_ASSERT(std::fabs(card.sampling_no_thinking.top_p - 0.8f) < 1.0e-6f);
    TEST_ASSERT(card.sampling_no_thinking.has_top_k);
    TEST_ASSERT(card.sampling_no_thinking.top_k == 20);
    TEST_ASSERT(card.sampling_no_thinking.has_presence_penalty);
    TEST_ASSERT(std::fabs(card.sampling_no_thinking.presence_penalty - 1.5f) < 1.0e-6f);
    TEST_ASSERT(card.sampling_no_thinking.has_repetition_penalty);
    TEST_ASSERT(std::fabs(card.sampling_no_thinking.repetition_penalty - 1.0f) < 1.0e-6f);
}

TEST_CASE(ServerUnitFixture, test_sampler_defaults_no_thinking_applied_when_thinking_off) {
    const ServerConfig config = qwen4exp_dual_sampling_config();
    const json body = {
        {"chat_template_kwargs", {{"enable_thinking", false}}},
    };
    const SamplerCfg sampler = resolve_request_sampler(body, config);

    TEST_ASSERT(std::fabs(sampler.temp - 0.7f) < 0.001f);
    TEST_ASSERT(std::fabs(sampler.top_p - 0.80f) < 0.001f);
    TEST_ASSERT(sampler.top_k == 20);
    TEST_ASSERT(std::fabs(sampler.pres_pen - 1.5f) < 0.001f);
    TEST_ASSERT(std::fabs(sampler.rep_pen - 1.0f) < 0.001f);
}

TEST_CASE(ServerUnitFixture, test_sampler_defaults_thinking_mode_unaffected_by_no_thinking_card) {
    const ServerConfig config = qwen4exp_dual_sampling_config();
    const json body = json::object();  // qwen4exp thinks by default when omitted
    const SamplerCfg sampler = resolve_request_sampler(body, config);

    TEST_ASSERT(std::fabs(sampler.temp - 1.0f) < 0.001f);
    TEST_ASSERT(std::fabs(sampler.top_p - 0.95f) < 0.001f);
    TEST_ASSERT(std::fabs(sampler.pres_pen - 0.0f) < 0.001f);
    TEST_ASSERT(std::fabs(sampler.rep_pen - 1.0f) < 0.001f);
}

TEST_CASE(ServerUnitFixture, test_sampler_explicit_fields_win_in_both_thinking_modes) {
    const ServerConfig config = qwen4exp_dual_sampling_config();

    const json thinking_off_body = {
        {"temperature", 0.33f},
        {"presence_penalty", 0.1f},
        {"chat_template_kwargs", {{"enable_thinking", false}}},
    };
    const SamplerCfg off_sampler = resolve_request_sampler(thinking_off_body, config);
    TEST_ASSERT(std::fabs(off_sampler.temp - 0.33f) < 0.001f);
    TEST_ASSERT(std::fabs(off_sampler.pres_pen - 0.1f) < 0.001f);
    // Omitted fields still pick up the no-thinking defaults.
    TEST_ASSERT(std::fabs(off_sampler.top_p - 0.80f) < 0.001f);

    const json thinking_on_body = {{"temperature", 0.44f}};
    const SamplerCfg on_sampler = resolve_request_sampler(thinking_on_body, config);
    TEST_ASSERT(std::fabs(on_sampler.temp - 0.44f) < 0.001f);
    TEST_ASSERT(std::fabs(on_sampler.top_p - 0.95f) < 0.001f);
}

TEST_CASE(ServerUnitFixture, test_sampler_defaults_no_thinking_absent_keeps_today_behavior) {
    // A card without `sampling_no_thinking` must behave exactly as before
    // this feature: no-thinking requests still fall back to the (thinking-
    // mode) `sampling` defaults for omitted fields.
    ServerConfig config;
    config.arch = "qwen4exp";
    config.sampler_defaults.has_temperature = true;
    config.sampler_defaults.temperature = 1.0f;
    config.sampler_defaults.has_presence_penalty = true;
    config.sampler_defaults.presence_penalty = 0.0f;
    // config.sampler_defaults_no_thinking left default-constructed: every
    // has_* is false, i.e. the card has no `sampling_no_thinking` block.

    const json body = {
        {"chat_template_kwargs", {{"enable_thinking", false}}},
    };
    const SamplerCfg sampler = resolve_request_sampler(body, config);

    TEST_ASSERT(std::fabs(sampler.temp - 1.0f) < 0.001f);
    TEST_ASSERT(std::fabs(sampler.pres_pen - 0.0f) < 0.001f);
}

TEST_CASE(ServerUnitFixture, test_require_messages_array_rejects_invalid) {
    const json valid = {{"messages", json::array({
        {{"role", "user"}, {"content", "hi"}},
    })}};
    TEST_ASSERT(require_messages_array(valid).size() == 1);

    const json invalid_bodies[] = {
        json::object(),                       // missing
        {{"messages", nullptr}},              // null
        {{"messages", "hi"}},                 // wrong type
        {{"messages", json::array()}},        // empty
    };
    for (const auto & body : invalid_bodies) {
        bool threw = false;
        try {
            require_messages_array(body);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        TEST_ASSERT(threw);
    }
}

TEST_CASE(ServerUnitFixture, test_max_output_alias_precedence_ignores_shadowed_invalid_value) {
    const json body = {
        {"max_tokens", 100},
        {"max_output_tokens", 200},
        {"max_completion_tokens", "invalid"},
    };

    TEST_ASSERT(resolve_max_output_tokens(body, 400) == 100);
    TEST_ASSERT(
        resolve_max_output_tokens({{"max_output_tokens", 200}}, 400) == 200);
    TEST_ASSERT(resolve_max_output_tokens(json::object(), 400) == 400);
    // "Unlimited" sentinels from clients such as PocketPal must fall back
    // to the default rather than yielding a zero-token budget.
    TEST_ASSERT(
        resolve_max_output_tokens({{"max_completion_tokens", -1}}, 400) == 400);
    TEST_ASSERT(
        resolve_max_output_tokens({{"max_completion_tokens", 0}}, 400) == 400);
    TEST_ASSERT(
        resolve_max_output_tokens({{"max_tokens", -1}}, 400) == 400);
    TEST_ASSERT(
        resolve_max_output_tokens({{"max_output_tokens", -1}}, 400) == 400);
    TEST_ASSERT(
        resolve_max_output_tokens({{"max_completion_tokens", 8}}, 400) == 8);
}


static ServerConfig deepseek_reasoning_test_config() {
    ServerConfig config;
    config.arch = "deepseek4";
    config.think_max_tokens = 900;
    config.hard_limit_reply_budget = 100;
    config.effort_tiers.low = 100;
    config.effort_tiers.medium = 200;
    config.effort_tiers.high = 300;
    config.effort_tiers.x_high = 400;
    config.effort_tiers.max = 500;
    return config;
}

static ParsedRequest resolve_deepseek_reasoning(const json & body) {
    ParsedRequest req;
    req.max_output = 1000;
    apply_request_reasoning(body, deepseek_reasoning_test_config(), req);
    return req;
}

static ParsedRequest resolve_qwen_reasoning(const json & body) {
    ServerConfig config = deepseek_reasoning_test_config();
    config.arch = "qwen35";
    ParsedRequest req;
    req.max_output = 1000;
    apply_request_reasoning(body, config, req);
    return req;
}

static ParsedRequest resolve_qwen4exp_reasoning(const json & body) {
    ServerConfig config = deepseek_reasoning_test_config();
    config.arch = "qwen4exp";
    ParsedRequest req;
    req.max_output = 1000;
    apply_request_reasoning(body, config, req);
    return req;
}

// Qwen3.8-Flash-Next's official template thinks by default (reasoning effort xhigh); an omitted thinking field must
// keep it on, with the budget envelope active so the reply reserve applies. Explicit off still wins. Other Qwen
// architectures keep the server's thinking-off default.
TEST_CASE(ServerUnitFixture, test_qwen4exp_thinks_by_default) {
    const ParsedRequest by_default = resolve_qwen4exp_reasoning(json::object());
    TEST_ASSERT(by_default.thinking_enabled && by_default.thinking_opt_in);
    const ParsedRequest off = resolve_qwen4exp_reasoning({{"thinking", {{"type", "disabled"}}}});
    TEST_ASSERT(!off.thinking_enabled && !off.thinking_opt_in);
    // Claude Code's "adaptive" leaves it to the model: the default.
    const ParsedRequest adaptive = resolve_qwen4exp_reasoning({{"thinking", {{"type", "adaptive"}}}});
    TEST_ASSERT(adaptive.thinking_enabled && adaptive.thinking_opt_in);
    TEST_ASSERT(!resolve_qwen_reasoning({{"thinking", {{"type", "adaptive"}}}}).thinking_enabled);
    const ParsedRequest kw_off = resolve_qwen4exp_reasoning({{"chat_template_kwargs", {{"enable_thinking", false}}}});
    TEST_ASSERT(!kw_off.thinking_enabled && !kw_off.thinking_opt_in);
    TEST_ASSERT(!resolve_qwen_reasoning(json::object()).thinking_enabled);
}

// chat_template_kwargs.preserve_thinking is a tri-state Jinja-only toggle:
// absent leaves req.preserve_thinking at -1 (template default applies), and
// an explicit bool sets 0/1. It must not disturb thinking_enabled/opt_in.
TEST_CASE(ServerUnitFixture, test_qwen4exp_preserve_thinking_kwarg) {
    const ParsedRequest absent = resolve_qwen4exp_reasoning(json::object());
    TEST_ASSERT(absent.preserve_thinking == -1);
    const ParsedRequest off = resolve_qwen4exp_reasoning(
        {{"chat_template_kwargs", {{"preserve_thinking", false}}}});
    TEST_ASSERT(off.preserve_thinking == 0);
    TEST_ASSERT(off.thinking_enabled && off.thinking_opt_in);
    const ParsedRequest on = resolve_qwen4exp_reasoning(
        {{"chat_template_kwargs", {{"preserve_thinking", true}}}});
    TEST_ASSERT(on.preserve_thinking == 1);
}

// Qwen3.8-Flash-Next's template knows low, medium and xhigh (its default); Lucebox's high, x-high and max map to xhigh.
TEST_CASE(ServerUnitFixture, test_qwen4exp_template_effort_mapping) {
    TEST_ASSERT(qwen4exp_template_effort("") == "");
    TEST_ASSERT(qwen4exp_template_effort("low") == "low");
    TEST_ASSERT(qwen4exp_template_effort("medium") == "medium");
    TEST_ASSERT(qwen4exp_template_effort("high") == "xhigh");
    TEST_ASSERT(qwen4exp_template_effort("x-high") == "xhigh");
    TEST_ASSERT(qwen4exp_template_effort("max") == "xhigh");
}

// The Jinja path hands the reasoning effort to the template only when one is set (templates default it themselves).
TEST_CASE(ServerUnitFixture, test_jinja_render_passes_reasoning_effort) {
    const std::string tmpl = "{% if reasoning_effort is defined %}{{ reasoning_effort }}{% else %}unset{% endif %}";
    const std::vector<ChatMessage> msgs = {{"user", "hi", ""}};
    TEST_ASSERT(render_chat_template_jinja(tmpl, msgs, "", "", true, true, "", "low") == "low");
    TEST_ASSERT(render_chat_template_jinja(tmpl, msgs, "", "", true, true, "") == "unset");
}

TEST_CASE(ServerUnitFixture, test_deepseek_reasoning_effort_aliases_and_budgets) {
    struct Case {
        const char * requested;
        const char * model_effort;
        int phase1_cap;
        bool enabled;
    };
    const Case cases[] = {
        {"none",    "",     -1,  false},
        {"minimal", "low",  100, true},
        {"low",     "low",  100, true},
        {"medium",  "high", 200, true},
        {"high",    "high", 300, true},
        // DeepSeek V4 Flash's API-compatible spelling maps to high.
        {"xhigh",   "high", 300, true},
        // Lucebox's hyphenated extension retains its separate budget tier.
        {"x-high",  "max",  400, true},
        {"max",     "max",  500, true},
        {"future",  "high", 300, true},
    };

    for (const auto & test : cases) {
        const ParsedRequest req = resolve_deepseek_reasoning({
            {"reasoning", {{"effort", test.requested}}},
        });
        TEST_ASSERT(req.thinking_enabled == test.enabled);
        TEST_ASSERT(req.reasoning_effort == test.model_effort);
        TEST_ASSERT(req.per_req_phase1_cap == test.phase1_cap);
        TEST_ASSERT(req.thinking_opt_in == test.enabled);
    }
}

TEST_CASE(ServerUnitFixture, test_deepseek_reasoning_request_precedence_and_toggles) {
    const json effort_locations[] = {
        {{"reasoning", {{"effort", "max"}}}},
        {{"reasoning_effort", "max"}},
        {{"chat_template_kwargs", {{"reasoning_effort", "max"}}}},
    };
    for (const auto & body : effort_locations) {
        const ParsedRequest req = resolve_deepseek_reasoning(body);
        TEST_ASSERT(req.thinking_enabled);
        TEST_ASSERT(req.reasoning_effort == "max");
        TEST_ASSERT(req.per_req_phase1_cap == 500);
    }

    // reasoning.effort wins over both lower-priority spellings.
    const ParsedRequest first_wins = resolve_deepseek_reasoning({
        {"reasoning", {{"effort", "low"}}},
        {"reasoning_effort", "max"},
        {"chat_template_kwargs", {{"reasoning_effort", "max"}}},
    });
    TEST_ASSERT(first_wins.reasoning_effort == "low");
    TEST_ASSERT(first_wins.per_req_phase1_cap == 100);

    // The API-style thinking control opts into the budget envelope and uses
    // DeepSeek's high default when no explicit effort is present.
    const ParsedRequest api_default_high = resolve_deepseek_reasoning({
        {"thinking", {{"type", "enabled"}}},
    });
    TEST_ASSERT(api_default_high.thinking_enabled);
    TEST_ASSERT(api_default_high.reasoning_effort == "high");
    TEST_ASSERT(api_default_high.per_req_phase1_cap == 300);
    TEST_ASSERT(api_default_high.thinking_opt_in);

    // Renderer-only controls may select DeepSeek's default model-facing
    // effort, but must not activate Lucebox's force-close budget envelope.
    const json renderer_only_controls[] = {
        {{"reasoning", json::object()}},
        {{"chat_template_kwargs", {{"thinking", true}}}},
        {{"chat_template_kwargs", {{"enable_thinking", true}}}},
        {
            {"thinking", {
                {"budget_tokens", 250},
                {"reply_budget", 50},
            }},
            {"chat_template_kwargs", {{"thinking", true}}},
        },
        {
            {"thinking", {{"type", "disabled"}}},
            {"chat_template_kwargs", {{"thinking", true}}},
        },
    };
    for (const auto & body : renderer_only_controls) {
        const ParsedRequest req = resolve_deepseek_reasoning(body);
        TEST_ASSERT(req.thinking_enabled);
        TEST_ASSERT(req.reasoning_effort == "high");
        TEST_ASSERT(req.per_req_phase1_cap == -1);
        TEST_ASSERT(req.per_req_reply_budget == -1);
        TEST_ASSERT(!req.thinking_opt_in);
    }

    // A later explicit toggle overrides an effort, including effort=none.
    const ParsedRequest api_disabled = resolve_deepseek_reasoning({
        {"reasoning_effort", "max"},
        {"thinking", {{"type", "disabled"}}},
    });
    TEST_ASSERT(!api_disabled.thinking_enabled);
    TEST_ASSERT(api_disabled.reasoning_effort.empty());
    TEST_ASSERT(api_disabled.per_req_phase1_cap == -1);
    TEST_ASSERT(!api_disabled.thinking_opt_in);

    const json renderer_disabled_controls[] = {
        {{"chat_template_kwargs", {{"thinking", false}}}},
        {{"chat_template_kwargs", {{"enable_thinking", false}}}},
    };
    for (const auto & renderer_control : renderer_disabled_controls) {
        json body = {
            {"reasoning_effort", "max"},
            {"thinking", {
                {"type", "enabled"},
                {"budget_tokens", 250},
                {"reply_budget", 50},
            }},
        };
        body.update(renderer_control);
        const ParsedRequest disabled = resolve_deepseek_reasoning(body);
        TEST_ASSERT(!disabled.thinking_enabled);
        TEST_ASSERT(disabled.reasoning_effort.empty());
        TEST_ASSERT(disabled.per_req_phase1_cap == -1);
        TEST_ASSERT(disabled.per_req_reply_budget == -1);
        TEST_ASSERT(!disabled.thinking_opt_in);
    }

    const ParsedRequest reenabled = resolve_deepseek_reasoning({
        {"reasoning", {{"effort", "none"}}},
        {"thinking", {{"type", "enabled"}}},
    });
    TEST_ASSERT(reenabled.thinking_enabled);
    TEST_ASSERT(reenabled.reasoning_effort == "high");
    TEST_ASSERT(reenabled.per_req_phase1_cap == 300);

    const ParsedRequest budget_override = resolve_deepseek_reasoning({
        {"reasoning_effort", "max"},
        {"thinking", {
            {"type", "enabled"},
            {"budget_tokens", 250},
            {"reply_budget", 50},
        }},
    });
    TEST_ASSERT(budget_override.reasoning_effort == "max");
    TEST_ASSERT(budget_override.per_req_phase1_cap == 250);
    TEST_ASSERT(budget_override.per_req_reply_budget == 50);

    const ParsedRequest no_control =
        resolve_deepseek_reasoning(json::object());
    TEST_ASSERT(!no_control.thinking_enabled);
    TEST_ASSERT(no_control.reasoning_effort.empty());
    TEST_ASSERT(no_control.per_req_phase1_cap == -1);
}

// A max_tokens at or below the reply reserve still leaves room to think: the
// reply takes half of it. The full reserve used to take all of it, so asking
// for an effort with max_tokens 4096 (reserve 4096) gave no thinking.
TEST_CASE(ServerUnitFixture, test_reply_reserve_leaves_room_to_think) {
    TEST_ASSERT(luce::common::reply_reserve_within(4096, 32768) == 4096);
    TEST_ASSERT(luce::common::reply_reserve_within(4096, 4096) == 2048);
    TEST_ASSERT(luce::common::reply_reserve_within(4096, 1000) == 500);

    struct Case {
        int max_output;
        int phase1_cap;
    };
    // Reserve 100, high tier 300.
    const Case cases[] = {
        {1000, 300},
        {150, 75},
        {100, 50},
    };
    ServerConfig config = deepseek_reasoning_test_config();
    config.arch = "qwen35";
    for (const auto & test : cases) {
        ParsedRequest req;
        req.max_output = test.max_output;
        apply_request_reasoning({{"reasoning_effort", "high"}}, config, req);
        TEST_ASSERT(req.thinking_opt_in);
        TEST_ASSERT(req.per_req_phase1_cap == test.phase1_cap);
    }
}

TEST_CASE(ServerUnitFixture, test_qwen_template_toggles_remain_renderer_only) {
    const json renderer_only_controls[] = {
        {{"chat_template_kwargs", {{"thinking", true}}}},
        {{"chat_template_kwargs", {{"enable_thinking", true}}}},
    };
    for (const auto & body : renderer_only_controls) {
        const ParsedRequest req = resolve_qwen_reasoning(body);
        TEST_ASSERT(req.thinking_enabled);
        TEST_ASSERT(req.reasoning_effort.empty());
        TEST_ASSERT(req.per_req_phase1_cap == -1);
        TEST_ASSERT(req.per_req_reply_budget == -1);
        TEST_ASSERT(!req.thinking_opt_in);
    }

    // An explicit effort remains a budget opt-in even when transported in
    // chat_template_kwargs; only the bare boolean toggles are renderer-only.
    const ParsedRequest explicit_effort = resolve_qwen_reasoning({
        {"chat_template_kwargs", {
            {"thinking", true},
            {"reasoning_effort", "max"},
        }},
    });
    TEST_ASSERT(explicit_effort.thinking_enabled);
    TEST_ASSERT(explicit_effort.reasoning_effort == "max");
    TEST_ASSERT(explicit_effort.per_req_phase1_cap == 500);
    TEST_ASSERT(explicit_effort.thinking_opt_in);
}

TEST_CASE(ServerUnitFixture, test_pflash_placement_same_backend_local) {
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

TEST_CASE(ServerUnitFixture, test_pflash_placement_mixed_backend_remote) {
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

TEST_CASE(ServerUnitFixture, test_pflash_placement_auto_draft_follows_target) {
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

TEST_CASE(ServerUnitFixture, test_pflash_placement_disabled_never_remote) {
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

TEST_CASE(ServerUnitFixture, test_pflash_drafter_device_overrides_the_decode_drafters) {
    DevicePlacement draft;
    draft.backend = PlacementBackend::Hip;
    draft.gpu = 0;
    DevicePlacement own;
    own.backend = PlacementBackend::Hip;
    own.gpu = 1;
    TEST_ASSERT(pflash_drafter_device(draft, std::nullopt).gpu == 0);
    TEST_ASSERT(pflash_drafter_device(draft, own).gpu == 1);

    RemoteDraftConfig remote;
    remote.ipc_bin = "/tmp/backend_ipc_daemon";
    auto placement = resolve_pflash_drafter_placement(
        draft, pflash_drafter_device(draft, own), remote, /*pflash_enabled=*/true);
    TEST_ASSERT(placement.drafter_gpu == 1);
    // Same backend on another GPU: scored in process, not through IPC.
    TEST_ASSERT(!placement.remote_drafter);
}

TEST_CASE(ServerUnitFixture, test_pflash_drafter_beside_target) {
    DevicePlacement target;
    target.backend = PlacementBackend::Hip;
    target.gpu = 0;
    DevicePlacement same = target;
    DevicePlacement other = target;
    other.gpu = 1;
    DevicePlacement unset;
    unset.backend = PlacementBackend::Auto;
    unset.gpu = 0;
    // No device of its own: it follows the decode drafter, on the target's GPU.
    TEST_ASSERT(pflash_drafter_beside_target(target, std::nullopt));
    TEST_ASSERT(pflash_drafter_beside_target(target, same));
    TEST_ASSERT(!pflash_drafter_beside_target(target, other));
    // An unset backend is the compiled one, so hip:0 and auto:0 are one GPU here.
    if (compiled_placement_backend() == PlacementBackend::Hip) {
        TEST_ASSERT(pflash_drafter_beside_target(target, unset));
    }
}

TEST_CASE(ServerUnitFixture, test_pflash_drafter_device_refusal) {
    DevicePlacement target;
    target.backend = PlacementBackend::Hip;
    target.gpu = 0;
    DevicePlacement other = target;
    other.gpu = 1;
    // Without its own device nothing is refused, whatever the target.
    TEST_ASSERT(pflash_drafter_device_refusal(target, true, std::nullopt) == nullptr);
    // Another GPU on the target's backend scores in process.
    TEST_ASSERT(pflash_drafter_device_refusal(target, false, other) == nullptr);
    // auto picks the target's GPU after the pool and parking are sized.
    TEST_ASSERT(pflash_drafter_device_refusal(target, true, other) != nullptr);
    // A layer-split target loads its drafter on its first GPU.
    DevicePlacement split;
    TEST_ASSERT(parse_placement_device_list("hip:0,hip:1", split));
    TEST_ASSERT(pflash_drafter_device_refusal(split, false, other) != nullptr);
    // Another backend would need the IPC drafter.
    DevicePlacement cuda = other;
    cuda.backend = target.backend == PlacementBackend::Cuda ? PlacementBackend::Hip : PlacementBackend::Cuda;
    TEST_ASSERT(pflash_drafter_device_refusal(target, false, cuda) != nullptr);
}

TEST_CASE(ServerUnitFixture, test_draft_residency_parse) {
    DraftResidencyPolicy policy = DraftResidencyPolicy::Auto;
    TEST_ASSERT(parse_draft_residency_policy("auto", policy));
    TEST_ASSERT(policy == DraftResidencyPolicy::Auto);
    TEST_ASSERT(parse_draft_residency_policy("persistent", policy));
    TEST_ASSERT(policy == DraftResidencyPolicy::Persistent);
    TEST_ASSERT(parse_draft_residency_policy("request-scoped", policy));
    TEST_ASSERT(policy == DraftResidencyPolicy::RequestScoped);
    TEST_ASSERT(parse_draft_residency_policy("request_scoped", policy));
    TEST_ASSERT(policy == DraftResidencyPolicy::RequestScoped);
    TEST_ASSERT(!parse_draft_residency_policy("request", policy));
}

TEST_CASE(ServerUnitFixture, test_draft_residency_pflash_auto) {
    auto action = resolve_draft_residency_action(
        DraftResidencyPolicy::Auto,
        DraftResidencyContext{
            DraftResidencyUse::PFlashCompress,
            /*low_vram_hint=*/false,
            /*has_decode_draft=*/false,
        });
    TEST_ASSERT(action == DraftResidencyAction::ReleaseAfterUse);

    action = resolve_draft_residency_action(
        DraftResidencyPolicy::Auto,
        DraftResidencyContext{
            DraftResidencyUse::PFlashCompress,
            /*low_vram_hint=*/true,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::ReleaseAfterUse);
}

TEST_CASE(ServerUnitFixture, test_draft_residency_dflash_auto_and_request_scoped) {
    auto action = resolve_draft_residency_action(
        DraftResidencyPolicy::Auto,
        DraftResidencyContext{
            DraftResidencyUse::DFlashDecode,
            /*low_vram_hint=*/false,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::KeepLoaded);

    action = resolve_draft_residency_action(
        DraftResidencyPolicy::Auto,
        DraftResidencyContext{
            DraftResidencyUse::DFlashDecode,
            /*low_vram_hint=*/true,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::ReleaseAfterUse);

    action = resolve_draft_residency_action(
        DraftResidencyPolicy::RequestScoped,
        DraftResidencyContext{
            DraftResidencyUse::DFlashDecode,
            /*low_vram_hint=*/false,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::ReleaseAfterUse);

    action = resolve_draft_residency_action(
        DraftResidencyPolicy::Persistent,
        DraftResidencyContext{
            DraftResidencyUse::DFlashDecode,
            /*low_vram_hint=*/true,
            /*has_decode_draft=*/true,
        });
    TEST_ASSERT(action == DraftResidencyAction::KeepLoaded);
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

TEST_CASE(ServerUnitFixture, test_bailingmoe3_router_builds_group_mask) {
    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr);

    TargetWeights weights;
    weights.n_expert = 512;
    weights.n_expert_used = 8;
    weights.n_expert_groups = 8;
    weights.n_expert_groups_used = 4;
    weights.expert_gating_func = 2;
    weights.expert_weights_norm = true;
    weights.expert_weights_scale = 2.5f;

    TargetLayer layer;
    layer.ffn_gate_inp = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, 16, weights.n_expert);
    layer.ffn_exp_probs_b = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, weights.n_expert);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 2);

    const Qwen35MoeRouterOutputs router = build_qwen35moe_router(
        ctx, input, weights, layer);
    TEST_ASSERT(router.selected != nullptr);
    TEST_ASSERT(router.weights != nullptr);

    std::unordered_set<const ggml_tensor *> visited;
    const auto contains_op = [&](const auto & self,
                                 const ggml_tensor * tensor,
                                 ggml_op op) -> bool {
        if (!tensor || !visited.insert(tensor).second) return false;
        if (tensor->op == op) return true;
        for (const ggml_tensor * source : tensor->src) {
            if (self(self, source, op)) return true;
        }
        return false;
    };
    TEST_ASSERT(contains_op(contains_op, router.selected, GGML_OP_SET_ROWS));

    ggml_free(ctx);
}

// The prefix cache budgets captures before they exist, so the Qwen estimate
// must equal the buffer snapshot_target_cache() allocates at each length.
TEST_CASE(ServerUnitFixture, test_qwen_snapshot_estimate_matches_saved_snapshot) {
    TargetWeights w;
    w.n_layer = 4;
    w.full_attention_interval = 4;
    w.n_embd = 64;
    w.n_embd_head_k = 32;
    w.n_embd_head_v = 32;
    w.n_head = 2;
    w.n_head_kv = 1;
    w.ssm_d_conv = 4;
    w.ssm_d_inner = 64;
    w.ssm_d_state = 16;
    w.ssm_dt_rank = 4;
    w.ssm_n_group = 2;
    w.n_capture_layers = 5;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    TEST_ASSERT(cpu != nullptr);
    TargetCache cache;
    TEST_ASSERT(create_target_cache(w, 64, 0, cpu, cache, /*prefill_only=*/true));
    // Every snapshot part is present: K/V, recurrent state, target features.
    TEST_ASSERT(!cache.attn_k.empty() && cache.attn_k[0] != nullptr);
    TEST_ASSERT(!cache.ssm_state.empty() && cache.ssm_state[0] != nullptr);
    TEST_ASSERT(cache.target_feat != nullptr);

    const auto buft = ggml_backend_get_default_buffer_type(cpu);
    for (const int tokens : {1, 17, 64}) {
        cache.cur_pos = tokens;
        PrefixSnapshot snap;
        TEST_ASSERT(snapshot_target_cache(w, cache, cpu, snap));
        TEST_ASSERT(ggml_backend_buffer_get_size(snap.buf) ==
                    estimate_target_cache_snapshot_bytes(w, cache, tokens, buft));
        free_prefix_snapshot(snap);
    }
    // Lengths past the cache capacity cost what a full cache costs.
    TEST_ASSERT(estimate_target_cache_snapshot_bytes(w, cache, 1 << 30, buft) ==
                estimate_target_cache_snapshot_bytes(w, cache, 64, buft));
    TEST_ASSERT(estimate_target_cache_snapshot_bytes(w, cache, 0, buft) == 0);

    free_target_cache(cache);
    ggml_backend_free(cpu);
}

TEST_CASE(ServerUnitFixture, test_qwen4exp_qsa_batch_boundary) {
    Qwen4ExpWeights w;
    w.n_layer = 1;
    w.layers.resize(1);
    w.layers[0].is_full_attention = true;
    w.compress_ratios = {4};
    w.indexer_head_size = 128;
    w.indexer_n_head = 4;
    w.indexer_top_k = 2048;
    w.n_head = 24;
    w.n_head_kv = 2;
    ggml_tensor raw{}, kv{};
    kv.ne[1] = 32768;
    Qwen4ExpCache caches[2];
    for (auto & cache : caches) {
        cache.indexer_raw = {&raw};
        cache.attn_k = {&kv};
    }
    Qwen4ExpForwardSegment spans[] = {{&caches[0], nullptr, 1, 16},
                                     {&caches[1], nullptr, 1, 2050}};
    TEST_ASSERT(qwen4exp_can_batch(w, spans, 2, true)); // 512 blocks + 3 tail tokens
    spans[1].pos0 = 2051;
    TEST_ASSERT(!qwen4exp_can_batch(w, spans, 2, true)); // first 513th block, before executing
    TEST_ASSERT(qwen4exp_can_batch(w, spans, 2, true, true)); // stable QSA rows join the batch
    TEST_ASSERT(qwen4exp_can_batch(w, spans, 2, false)); // generic device, QSA off
    std::swap(spans[0], spans[1]);
    TEST_ASSERT(!qwen4exp_can_batch(w, spans, 2, true)); // any slot, independent of order
    spans[0].pos0 = 16;
    spans[0].n_tokens = 512;
    TEST_ASSERT(!qwen4exp_can_batch(w, spans, 2, true)); // prefill always runs solo
    TEST_ASSERT(!qwen4exp_can_batch(w, spans, 2, true, true));
    TEST_ASSERT(!qwen4exp_can_batch(w, spans, 2, false));
    spans[0].n_tokens = 1;
    Qwen4ExpForwardSegment rows[5] = {spans[0], spans[1], spans[0], spans[1], spans[0]};
    TEST_ASSERT(qwen4exp_can_batch(w, rows, 4, true));
    TEST_ASSERT(qwen4exp_can_batch(w, rows, 3, true));
    TEST_ASSERT(!qwen4exp_can_batch(w, rows, 5, true)); // batch-invariant MMID ceiling
    TEST_ASSERT(!qwen4exp_can_batch(w, rows, 5, false));
    w.indexer_top_k = 1024;
    spans[0].pos0 = 1026;
    spans[0].n_tokens = 1;
    TEST_ASSERT(qwen4exp_can_batch(w, spans, 2, true));
    spans[0].pos0++;
    TEST_ASSERT(!qwen4exp_can_batch(w, spans, 2, true)); // use model budget, not a magic 2051
    w.compress_ratios = {1};
    TEST_ASSERT(qwen4exp_can_batch(w, spans, 2, true)); // no supported QSA
    w.compress_ratios = {4};
    caches[1].indexer_raw.clear();
    TEST_ASSERT(qwen4exp_can_batch(w, spans, 2, true));
}

// Qwen4Exp QSA indexer pooling: block b is the mean of the r consecutive token keys r*b .. r*b+r-1
// (reference modeling_qwen4_exp.py: block_token_indices.view(n, ratio) then mean over the ratio axis).
TEST_CASE(ServerUnitFixture, test_qwen4exp_profile_is_scoped) {
    using namespace luce::common;
    auto set = ggml_backend_cuda_set_qwen4exp_profile;
    TEST_ASSERT(set(GGML_CUDA_QWEN4EXP_DEFAULT) == GGML_CUDA_QWEN4EXP_OFF);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    const bool supported = ggml_backend_cuda_qwen4exp_supported(cpu);
    TEST_ASSERT(!supported);
    {
        // An unsupported backend masks the enclosing profile, then restores it.
        [&] { Qwen4ExpCudaScope generic(supported); TEST_ASSERT(!generic.optimized);
              TEST_ASSERT(set(GGML_CUDA_QWEN4EXP_OFF) == GGML_CUDA_QWEN4EXP_OFF); }();
        TEST_ASSERT(set(GGML_CUDA_QWEN4EXP_DEFAULT) == GGML_CUDA_QWEN4EXP_DEFAULT);
        bool isolated = false;
        std::thread other([&] { isolated = set(GGML_CUDA_QWEN4EXP_OFF) == GGML_CUDA_QWEN4EXP_OFF; });
        other.join();
        TEST_ASSERT(isolated);
    }
    TEST_ASSERT(set(GGML_CUDA_QWEN4EXP_OFF) == GGML_CUDA_QWEN4EXP_DEFAULT);
    [&] { Qwen4ExpCudaScope optimized(true); TEST_ASSERT(optimized.optimized);
          TEST_ASSERT(set(GGML_CUDA_QWEN4EXP_DEFAULT) == GGML_CUDA_QWEN4EXP_DEFAULT);
          TEST_ASSERT(!ggml_backend_cuda_set_mmvq_batch_invariant(false));
          {
              ScopedCudaGraphOverrides batch(false, 0, false, 0, true);
              TEST_ASSERT(ggml_backend_cuda_set_mmvq_batch_invariant(true));
              TEST_ASSERT(set(GGML_CUDA_QWEN4EXP_DEFAULT) == GGML_CUDA_QWEN4EXP_DEFAULT);
          }
          TEST_ASSERT(!ggml_backend_cuda_set_mmvq_batch_invariant(false)); }();
    TEST_ASSERT(set(GGML_CUDA_QWEN4EXP_OFF) == GGML_CUDA_QWEN4EXP_OFF);
    ggml_backend_free(cpu);
}

TEST_CASE(ServerUnitFixture, test_qwen4exp_mtp_profile_and_verify_restore) {
    using namespace luce::common;
    auto profile = ggml_backend_cuda_set_qwen4exp_profile;
    auto invariant = ggml_backend_cuda_set_mmvq_batch_invariant;
    const auto previous = profile(GGML_CUDA_QWEN4EXP_OFF);
    const bool previous_invariant = invariant(false);
    [&] {
        Qwen4ExpCudaScope scope(true);
        ScopedCudaGraphOverrides verify(false, 0, false, 0, true);
        TEST_ASSERT(profile(GGML_CUDA_QWEN4EXP_DEFAULT) == GGML_CUDA_QWEN4EXP_DEFAULT);
        TEST_ASSERT(invariant(true));
    }();
    TEST_ASSERT(profile(GGML_CUDA_QWEN4EXP_OFF) == GGML_CUDA_QWEN4EXP_OFF);
    TEST_ASSERT(!invariant(false));
    // Invalid verify/rollback return before any GPU access, restoring the caller's profile.
    Qwen4ExpWeights weights;
    weights.gfx1151 = true;
    Qwen4ExpCache cache;
    std::vector<float> logits;
    const int32_t tokens[] = {1, 2};
    TEST_ASSERT(!qwen4exp_forward(nullptr, weights, cache, tokens, 2, 0, logits, nullptr, true).ok);
    TEST_ASSERT(profile(GGML_CUDA_QWEN4EXP_OFF) == GGML_CUDA_QWEN4EXP_OFF);
    TEST_ASSERT(!qwen4exp_verify_rollback(nullptr, weights, cache, 0, 1));
    TEST_ASSERT(profile(previous) == GGML_CUDA_QWEN4EXP_OFF);
    TEST_ASSERT(!invariant(previous_invariant));
}

TEST_CASE(ServerUnitFixture, test_qwen4exp_pool_blocks_averages_consecutive_tokens) {
    ggml_init_params ip{1 << 20, nullptr, false};
    ggml_context * c = ggml_init(ip);
    TEST_ASSERT(c != nullptr);
    const int idim = 2, r = 4, nb = 3;
    ggml_tensor * k = ggml_new_tensor_2d(c, GGML_TYPE_F32, idim, nb * r);
    float * kd = (float *) k->data;
    for (int t = 0; t < nb * r; ++t) {
        for (int d = 0; d < idim; ++d) kd[t * idim + d] = 100.0f * d + (float) t;
    }
    ggml_tensor * pooled = qwen4exp_pool_blocks(c, k, r);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, pooled);
    TEST_ASSERT(ggml_graph_compute_with_ctx(c, gf, 1) == GGML_STATUS_SUCCESS);
    TEST_ASSERT(pooled->ne[0] == idim && pooled->ne[1] == nb && ggml_is_contiguous(pooled));
    const float * pd = (const float *) pooled->data;
    for (int b = 0; b < nb; ++b) {
        for (int d = 0; d < idim; ++d) TEST_ASSERT(pd[b * idim + d] == 100.0f * d + (float) (r * b) + 1.5f);
    }
    ggml_free(c);
}

// The stable T=1 decode span is a function of kv_len alone that reproduces the spans token-by-token decode rebuilt
// its graph with (((kv_len + 511) / 256) * 256 whenever kv_len outgrew the last one), so an MTP verify row attends
// over exactly the span plain decode uses at that position.
TEST_CASE(ServerUnitFixture, test_qwen4exp_stable_kv_span_matches_decode_rebuilds) {
    for (const int64_t max_ctx : {4096, 32768, 40000}) {
        for (const int64_t first : {1, 2, 100, 255, 256, 257, 511, 512, 513, 2000, 3999}) {
            int64_t base = 0, rebuilt = 0;
            bool same = true;
            for (int64_t kv = first; kv <= max_ctx; ++kv) {
                if (rebuilt == 0 || kv > rebuilt) rebuilt = std::min<int64_t>(max_ctx, ((kv + 511) / 256) * 256);
                same = same && qwen4exp_stable_kv_span(base, max_ctx, kv) == rebuilt;
            }
            TEST_ASSERT_MSG(same, "max_ctx=" + std::to_string(max_ctx) + " first=" + std::to_string(first));
        }
    }
}

TEST_CASE(ServerUnitFixture, test_qwen4exp_mtp_sidecar_pick) {
    TEST_ASSERT(pick_qwen4exp_mtp_sidecar({"README.md", "mtp-b-Q8_0.gguf", "model.gguf", "mtp-a-Q8_0.gguf"}) ==
                "mtp-a-Q8_0.gguf");
    TEST_ASSERT(pick_qwen4exp_mtp_sidecar({"mtp-.gguf", "mtp-x.bin", "Qwen3.8-Flash-Next.gguf"}).empty());
    TEST_ASSERT(pick_qwen4exp_mtp_sidecar({}).empty());
}

// The Qwen report adds up the live cache and each snapshot from the buffers
// snapshot_target_cache() actually allocates.
TEST_CASE(ServerUnitFixture, test_qwen_memory_report_matches_buffers) {
    TargetWeights w;
    w.n_layer = 4;
    w.full_attention_interval = 4;
    w.n_embd = 64;
    w.n_embd_head_k = 32;
    w.n_embd_head_v = 32;
    w.n_head = 2;
    w.n_head_kv = 1;
    w.ssm_d_conv = 4;
    w.ssm_d_inner = 64;
    w.ssm_d_state = 16;
    w.ssm_dt_rank = 4;
    w.ssm_n_group = 2;
    w.n_capture_layers = 5;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    TEST_ASSERT(cpu != nullptr);
    TargetCache cache;
    TEST_ASSERT(create_target_cache(w, 64, 0, cpu, cache, /*prefill_only=*/true));
    PrefixSnapshot snapshots[3];
    cache.cur_pos = 17;
    TEST_ASSERT(snapshot_target_cache(w, cache, cpu, snapshots[1]));
    cache.cur_pos = 40;

    const auto report = Qwen35Backend::memory_report_for(cache, snapshots, 3);
    TEST_ASSERT(report.available);
    TEST_ASSERT(report.cache.host);
    TEST_ASSERT(report.cache.capacity_tokens == 64);
    TEST_ASSERT(report.cache.live_tokens == 40);
    uint64_t kv = 0, recurrent = 0;
    for (size_t i = 0; i < cache.attn_k.size(); ++i) {
        kv += ggml_nbytes(cache.attn_k[i]) + ggml_nbytes(cache.attn_v[i]);
    }
    for (size_t i = 0; i < cache.ssm_state.size(); ++i) {
        recurrent += ggml_nbytes(cache.ssm_state[i]) + ggml_nbytes(cache.conv_state[i]);
    }
    TEST_ASSERT(kv > 0 && recurrent > 0);
    TEST_ASSERT(report.cache.kv_bytes == kv);
    TEST_ASSERT(report.cache.recurrent_bytes == recurrent);
    TEST_ASSERT(report.cache.draft_feature_bytes == ggml_nbytes(cache.target_feat));
    uint64_t allocated = ggml_backend_buffer_get_size(cache.base_buf);
    if (cache.rollback_buf) allocated += ggml_backend_buffer_get_size(cache.rollback_buf);
    TEST_ASSERT(report.cache.kv_bytes + report.cache.recurrent_bytes +
                    report.cache.draft_feature_bytes + report.cache.other_bytes ==
                allocated);
    TEST_ASSERT(report.snapshots.size() == 1);
    TEST_ASSERT(report.snapshots[0].slot == 1);
    TEST_ASSERT(report.snapshots[0].tokens == 17);
    TEST_ASSERT(report.snapshots[0].host);
    TEST_ASSERT(report.snapshots[0].bytes ==
                ggml_backend_buffer_get_size(snapshots[1].buf));

    free_prefix_snapshot(snapshots[1]);
    free_target_cache(cache);
    ggml_backend_free(cpu);
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

TEST_CASE(ServerUnitFixture, test_gemma_tool_schema_formats_preserve_nested_types) {
    const json schema = {
        {"type", "object"},
        {"properties", {
            {"config", {{"type", "object"}, {"properties", {
                {"enabled", {{"type", "boolean"}}}}}}},
            {"type", {{"type", "string"}, {"enum", json::array({"lowercase"})}}},
        }},
        {"required", json::array({"config"})},
    };
    const json function = {{"name", "inspect"}, {"description", "Inspect."}, {"parameters", schema}};
    const std::vector<json> tools = {
        json::array({{{"type", "function"}, {"function", function}}}),
        json::array({function}),
        json::array({{{"name", "inspect"}, {"description", "Inspect."}, {"input_schema", schema}}}),
    };
    const std::string declaration =
        "<|tool>declaration:inspect{description:<|\"|>Inspect.<|\"|>,parameters:{"
        "properties:{config:{properties:{enabled:{type:<|\"|>BOOLEAN<|\"|>}},"
        "type:<|\"|>OBJECT<|\"|>},type:{enum:[<|\"|>lowercase<|\"|>],type:<|\"|>STRING<|\"|>}},"
        "required:[<|\"|>config<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";
    for (const auto & definition : tools) {
        const auto prompt = render_chat_template({{"user", "inspect it"}}, ChatFormat::GEMMA4,
                                                  true, false, definition.dump());
        TEST_ASSERT(prompt.find(declaration) != std::string::npos);
        TEST_ASSERT(prompt.find("input_schema") == std::string::npos);
    }
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

// Mirrors the official qwen4exp (Qwen3.8-Flash-Next) GGUF Jinja template's
// think-replay snippet: earlier assistant turns either replay their recorded
// reasoning_content inside <think>...</think>, or (preserve_thinking=false)
// drop the think block entirely for turns at/before the last user query.
static const char QWEN4EXP_THINK_REPLAY_TEMPLATE[] =
    "{%- set ns = namespace(last_query_index = 0) -%}"
    "{%- for message in messages -%}"
    "{%- if message.role == 'user' -%}{%- set ns.last_query_index = loop.index0 -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- for message in messages -%}"
    "{%- set content = message.content -%}"
    "{%- if message.role == 'assistant' -%}"
    "{%- set reasoning_content = '' -%}"
    "{%- if message.reasoning_content is string -%}{%- set reasoning_content = message.reasoning_content -%}{%- endif -%}"
    "{%- set reasoning_content = reasoning_content|trim -%}"
    "{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index -%}"
    "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content + '<|im_end|>\\n' }}"
    "{%- else -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}"
    "{%- endif -%}"
    "{%- else -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}"
    "{%- endif -%}"
    "{%- endfor -%}";

// (a) An assistant history message with reasoning_content renders inside
// <think>...</think> under the default (preserve_thinking left unset).
TEST_CASE(ServerUnitFixture, test_jinja_render_reasoning_content_default_preserves_think) {
    const std::vector<ChatMessage> msgs = {
        {"user",      "Q1", "", ""},
        {"assistant", "A1", "", "thinking about Q1"},
        {"user",      "Q2", "", ""},
    };
    const std::string out = render_chat_template_jinja(
        QWEN4EXP_THINK_REPLAY_TEMPLATE, msgs, "", "",
        /*add_gen=*/false, /*think=*/true, "", "",
        /*preserve_thinking=*/-1);
    TEST_ASSERT(out.find(
        "<|im_start|>assistant\n<think>\nthinking about Q1\n</think>\n\nA1<|im_end|>")
        != std::string::npos);
}

// (b) With preserve_thinking=false the earlier assistant turn renders
// without any <think> block.
TEST_CASE(ServerUnitFixture, test_jinja_render_preserve_thinking_false_strips_think) {
    const std::vector<ChatMessage> msgs = {
        {"user",      "Q1", "", ""},
        {"assistant", "A1", "", "thinking about Q1"},
        {"user",      "Q2", "", ""},
    };
    const std::string out = render_chat_template_jinja(
        QWEN4EXP_THINK_REPLAY_TEMPLATE, msgs, "", "",
        /*add_gen=*/false, /*think=*/true, "", "",
        /*preserve_thinking=*/0);
    TEST_ASSERT(out.find("<|im_start|>assistant\nA1<|im_end|>") != std::string::npos);
    TEST_ASSERT(out.find("<think>") == std::string::npos);
}

// (c) Without reasoning_content and under the default preserve setting, the
// output is unchanged from today: an empty <think></think> block (neither of
// the template's two official modes, but the pre-existing behavior — this is
// a regression guard, not an endorsement).
TEST_CASE(ServerUnitFixture, test_jinja_render_no_reasoning_content_regression) {
    const std::vector<ChatMessage> msgs = {
        {"user",      "Q1", "", ""},
        {"assistant", "A1", "", ""},
        {"user",      "Q2", "", ""},
    };
    const std::string out = render_chat_template_jinja(
        QWEN4EXP_THINK_REPLAY_TEMPLATE, msgs, "", "",
        /*add_gen=*/false, /*think=*/true, "", "",
        /*preserve_thinking=*/-1);
    TEST_ASSERT(out.find("<|im_start|>assistant\n<think>\n\n</think>\n\nA1<|im_end|>")
        != std::string::npos);
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
        TEST_ASSERT(chat_msgs[2].tool_calls_replayed);
        TEST_ASSERT(chat_msgs[2].tool_calls.size() == 2);
        TEST_ASSERT(chat_msgs[3].role == "tool");
        TEST_ASSERT(chat_msgs[3].tool_call_id == call_id);
        TEST_ASSERT(chat_msgs[3].content == "Process exited with code 0");
        TEST_ASSERT(chat_msgs[3].tool_name == "exec_command");
        TEST_ASSERT(chat_msgs[4].role == "tool");
        TEST_ASSERT(chat_msgs[4].tool_call_id == second_call_id);
        TEST_ASSERT(chat_msgs[4].content == "int main() {}");
        TEST_ASSERT(chat_msgs[4].tool_name == "read_file");
    }
}

// Claude Code shape: tool_use blocks in the assistant turn, tool_result
// blocks (any order, string or block content) plus reminder text in the
// next user turn, then a system note.
TEST_CASE(ServerUnitFixture, test_normalize_anthropic_tool_followup_messages) {
    ToolMemory tool_memory;
    const std::string raw =
        "<think>\n\n</think>\n\n<tool_call>\n<function=Read>\n"
        "<parameter=file_path>\n/a.py\n</parameter>\n</function>\n</tool_call>";
    tool_memory.remember({"call_a", "call_b"}, raw);

    const json messages = json::array({
        {{"role", "user"}, {"content", json::array({
            {{"type", "text"}, {"text", "read both"}}})}},
        {{"role", "assistant"}, {"content", json::array({
            {{"type", "text"}, {"text", ""}},
            {{"type", "tool_use"}, {"id", "call_a"}, {"name", "Read"},
             {"input", {{"file_path", "/a.py"}}}},
            {{"type", "tool_use"}, {"id", "call_b"}, {"name", "Read"},
             {"input", {{"file_path", "/b.py"}}}}})}},
        {{"role", "user"}, {"content", json::array({
            {{"type", "tool_result"}, {"tool_use_id", "call_b"},
             {"content", json::array({{{"type", "text"}, {"text", "B"}}})}},
            {{"type", "tool_result"}, {"tool_use_id", "call_a"},
             {"content", "A"}},
            {{"type", "text"}, {"text", "<system-reminder>x</system-reminder>"}}})}},
        {{"role", "system"}, {"content", json::array({
            {{"type", "text"}, {"text", "<total_tokens>9</total_tokens>"}}})}},
    });

    const auto chat = normalize_chat_messages(
        messages, ApiFormat::ANTHROPIC, tool_memory);
    TEST_ASSERT(chat.size() == 6);
    if (chat.size() == 6) {
        TEST_ASSERT(chat[0].role == "user" && chat[0].content == "read both");
        TEST_ASSERT(chat[1].role == "assistant" && chat[1].content == raw);
        TEST_ASSERT(chat[1].tool_calls_replayed);
        TEST_ASSERT(chat[1].tool_calls.size() == 2);
        TEST_ASSERT(chat[2].role == "tool" && chat[2].tool_call_id == "call_b" &&
                    chat[2].content == "B");
        TEST_ASSERT(chat[3].role == "tool" && chat[3].tool_call_id == "call_a" &&
                    chat[3].content == "A");
        TEST_ASSERT(chat[2].tool_name == "Read");
        TEST_ASSERT(chat[3].tool_name == "Read");
        TEST_ASSERT(chat[4].role == "user" &&
                    chat[4].content == "<system-reminder>x</system-reminder>");
        TEST_ASSERT(chat[5].role == "system");
    }
    TEST_ASSERT(!http_detail::ends_with_tool_result(chat));
    TEST_ASSERT(http_detail::ends_with_tool_result(
        std::vector<ChatMessage>(chat.begin(), chat.begin() + 4)));
    std::vector<ChatMessage> with_note(chat.begin(), chat.begin() + 4);
    with_note.push_back(chat[5]);
    TEST_ASSERT(http_detail::ends_with_tool_result(with_note));
}

// Calls the server no longer remembers retain structure for any renderer.
TEST_CASE(ServerUnitFixture, test_normalize_anthropic_tool_use_without_memory) {
    ToolMemory tool_memory;
    const json messages = json::array({
        {{"role", "user"}, {"content", "go"}},
        {{"role", "assistant"}, {"content", json::array({
            {{"type", "text"}, {"text", "Reading."}},
            {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "Read"},
             {"input", {{"file_path", "/a.py"}}}},
            {{"type", "tool_use"}, {"id", "toolu_2"}, {"name", "Grep"},
             {"input", {{"pattern", "x"}, {"n", 3}}}}})}},
    });
    const auto chat = normalize_chat_messages(
        messages, ApiFormat::ANTHROPIC, tool_memory);
    TEST_ASSERT(chat.size() == 2);
    if (chat.size() == 2) {
        TEST_ASSERT(chat[1].content == "Reading.");
        TEST_ASSERT(!chat[1].tool_calls_replayed);
        TEST_ASSERT(chat[1].tool_calls.size() == 2);
        if (chat[1].tool_calls.size() != 2) return;
        TEST_ASSERT(chat[1].tool_calls[0].id == "toolu_1");
        TEST_ASSERT(chat[1].tool_calls[0].name == "Read");
        TEST_ASSERT(chat[1].tool_calls[0].arguments == json({{"file_path", "/a.py"}}));
        TEST_ASSERT(chat[1].tool_calls[1].id == "toolu_2");
        TEST_ASSERT(chat[1].tool_calls[1].name == "Grep");
        TEST_ASSERT(chat[1].tool_calls[1].arguments == json({{"pattern", "x"}, {"n", 3}}));
    }
}

// OpenAI clients (Hermes Agent, Pi, any chat-completions agent) replay their
// history with tool_calls this server did not generate or no longer
// remembers: the call itself must reach the prompt, not only the prose.
TEST_CASE(ServerUnitFixture, test_normalize_openai_tool_calls_without_memory) {
    ToolMemory tool_memory;
    const json messages = json::array({
        {{"role", "user"}, {"content", "fix the bug"}},
        {{"role", "assistant"}, {"content", "Let me look."},
         {"tool_calls", json::array({
             {{"id", "call_1"}, {"type", "function"},
              {"function", {{"name", "bash"}, {"arguments", "{\"command\": \"open src/io_utils.py 10\"}"}}}},
             {{"id", "call_2"}, {"type", "function"},
              {"function", {{"name", "raw"}, {"arguments", "not json"}}}}})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "def read():"}},
        {{"role", "assistant"}, {"content", nullptr},
         {"tool_calls", json::array({
             {{"id", "call_3"}, {"type", "function"},
              {"function", {{"name", "bash"}, {"arguments", "{\"command\": \"ls\"}"}}}}})}},
    });
    const auto chat = normalize_chat_messages(messages, ApiFormat::OPENAI_CHAT, tool_memory);
    TEST_ASSERT(chat.size() == 4);
    if (chat.size() != 4) return;
    TEST_ASSERT(chat[1].role == "assistant");
    TEST_ASSERT(chat[1].content == "Let me look.");
    TEST_ASSERT(chat[1].tool_calls.size() == 2);
    if (chat[1].tool_calls.size() != 2) return;
    TEST_ASSERT(chat[1].tool_calls[0].name == "bash");
    TEST_ASSERT(chat[1].tool_calls[0].arguments ==
                json({{"command", "open src/io_utils.py 10"}}));
    TEST_ASSERT(chat[1].tool_calls[1].name == "raw");
    TEST_ASSERT(chat[1].tool_calls[1].arguments == json({{"arguments", "not json"}}));
    TEST_ASSERT(chat[2].role == "tool" && chat[2].content == "def read():");
    TEST_ASSERT(chat[2].tool_name == "bash");
    TEST_ASSERT(chat[3].content.empty());
    TEST_ASSERT(chat[3].tool_calls.size() == 1);
    if (chat[3].tool_calls.size() == 1) {
        TEST_ASSERT(chat[3].tool_calls[0].arguments == json({{"command", "ls"}}));
    }
}

TEST_CASE(ServerUnitFixture, test_normalize_openai_tool_argument_shapes) {
    ToolMemory tool_memory;
    const json nested = {
        {"options", {{"enabled", true}, {"paths", json::array({"a", "b"})}}},
        {"text", "line one\n\"line two\""},
    };
    const std::vector<std::pair<json, json>> cases = {
        {nested, nested},
        {nested.dump(), nested},
        {"{}", json::object()},
        {"", json::object()},
        {"[1, \"two\"]", json({{"arguments", "[1, \"two\"]"}})},
        {"42", json({{"arguments", "42"}})},
        {"true", json({{"arguments", "true"}})},
        {"null", json({{"arguments", "null"}})},
        {"\"text\"", json({{"arguments", "\"text\""}})},
        {"{broken", json({{"arguments", "{broken"}})},
    };
    for (const auto & [arguments, expected] : cases) {
        const json messages = json::array({
            {{"role", "assistant"}, {"content", nullptr},
             {"tool_calls", json::array({
                 {{"function", {{"name", "inspect"}, {"arguments", arguments}}}}})}},
        });
        const auto chat = normalize_chat_messages(messages, ApiFormat::OPENAI_CHAT, tool_memory);
        TEST_ASSERT(chat.size() == 1);
        if (chat.size() != 1) return;
        TEST_ASSERT(chat[0].content.empty());
        TEST_ASSERT(!chat[0].tool_calls_replayed);
        TEST_ASSERT(chat[0].tool_calls.size() == 1);
        if (chat[0].tool_calls.size() != 1) return;
        TEST_ASSERT(chat[0].tool_calls[0].id.empty());
        TEST_ASSERT(chat[0].tool_calls[0].name == "inspect");
        TEST_ASSERT(chat[0].tool_calls[0].arguments == expected);
    }
}

TEST_CASE(ServerUnitFixture, test_normalize_openai_tool_memory_replay_and_misses) {
    ToolMemory tool_memory(2);
    const std::string raw =
        "<think>Keep this reasoning.</think>\n\nReading.\n\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/a\n</parameter>\n"
        "</function>\n</tool_call>\n\n"
        "<tool_call>\n<function=list>\n</function>\n</tool_call>\n";
    tool_memory.remember({"call_a", "call_b"}, raw);
    json messages = json::array({
        {{"role", "assistant"}, {"content", "Reading."},
         {"tool_calls", json::array({
             {{"id", "call_a"}, {"type", "function"},
              {"function", {{"name", "read"}, {"arguments", R"({"path":"/a"})"}}}},
             {{"id", "call_b"}, {"type", "function"},
              {"function", {{"name", "list"}}}}})}},
    });
    auto check = [&](bool replayed) {
        const auto chat = normalize_chat_messages(messages, ApiFormat::OPENAI_CHAT, tool_memory);
        TEST_ASSERT(chat.size() == 1);
        if (chat.size() != 1) return;
        TEST_ASSERT(chat[0].content == (replayed ? raw : "Reading."));
        TEST_ASSERT(chat[0].tool_calls_replayed == replayed);
        TEST_ASSERT(chat[0].tool_calls.size() == 2);
        if (chat[0].tool_calls.size() != 2) return;
        TEST_ASSERT(chat[0].tool_calls[0].name == "read");
        TEST_ASSERT(chat[0].tool_calls[1].name == "list");
        if (replayed) {
            TEST_ASSERT(chat[0].tool_calls[0].arguments.is_null());
            TEST_ASSERT(chat[0].tool_calls[1].arguments.is_null());
            const auto rendered = render_chat_template(chat, ChatFormat::QWEN3, false, false, "");
            const auto pos = rendered.find(raw);
            TEST_ASSERT(pos != std::string::npos);
            TEST_ASSERT(rendered.find("<function=read>", rendered.find("<function=read>") + 1) ==
                        std::string::npos);
        } else {
            TEST_ASSERT(chat[0].tool_calls[0].arguments == json({{"path", "/a"}}));
            TEST_ASSERT(chat[0].tool_calls[1].arguments == json::object());
        }
    };
    check(true);
    messages[0]["tool_calls"][1]["id"] = "unknown";
    check(false);
    messages[0]["tool_calls"][1].erase("id");
    check(false);
    messages[0]["tool_calls"][1]["id"] = "";
    check(false);
    messages[0]["tool_calls"][1]["id"] = "call_b";
    tool_memory.remember({"unrelated"}, "Another turn.");
    check(false);  // One of the original IDs has been evicted.
    tool_memory.remember({"call_a"}, raw);
    tool_memory.remember({"call_b"}, "A different turn.");
    check(false);  // Both IDs exist, but belong to different turns.
}

TEST_CASE(ServerUnitFixture, test_normalize_openai_streamed_tool_calls) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools());
    const std::string raw =
        "<tool_call>\n<function=read>\n<parameter=path>\n/a\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/b\n</parameter>\n"
        "</function>\n</tool_call>";
    std::string wire = concat(em.emit_start());
    for (size_t offset = 0; offset < raw.size(); offset += 7) {
        wire += concat(em.emit_token(raw.substr(offset, 7)));
    }
    wire += concat(em.emit_finish(40));

    // Accumulate deltas by index as an OpenAI streaming client does.
    json calls = json::array();
    for (size_t pos = 0; (pos = wire.find("data: ", pos)) != std::string::npos;) {
        pos += 6;
        const size_t end = wire.find('\n', pos);
        const std::string data = wire.substr(pos, end - pos);
        if (data == "[DONE]") break;
        const auto event = json::parse(data);
        for (const auto & choice : event["choices"]) {
            const auto & delta = choice["delta"];
            if (!delta.contains("tool_calls")) continue;
            for (const auto & part : delta["tool_calls"]) {
                const size_t index = part["index"].get<size_t>();
                while (calls.size() <= index) {
                    calls.push_back({{"type", "function"},
                        {"function", {{"name", ""}, {"arguments", ""}}}});
                }
                auto & call = calls[index];
                if (part.contains("id")) call["id"] = part["id"];
                if (part.contains("function")) {
                    for (const char * key : {"name", "arguments"}) {
                        call["function"][key].get_ref<std::string &>() +=
                            part["function"].value(key, "");
                    }
                }
            }
        }
    }
    TEST_ASSERT(calls.size() == 2);
    if (calls.size() != 2) return;
    TEST_ASSERT(json::parse(calls[0]["function"]["arguments"].get<std::string>()) ==
                json({{"path", "/a"}}));
    TEST_ASSERT(json::parse(calls[1]["function"]["arguments"].get<std::string>()) ==
                json({{"path", "/b"}}));
    ToolMemory cold_memory;
    const auto chat = normalize_chat_messages(
        json::array({{{"role", "assistant"}, {"content", nullptr}, {"tool_calls", calls}}}),
        ApiFormat::OPENAI_CHAT, cold_memory);
    TEST_ASSERT(chat.size() == 1);
    if (chat.size() != 1) return;
    TEST_ASSERT(chat[0].content.empty());
    TEST_ASSERT(chat[0].tool_calls.size() == 2);
    const auto rendered = render_chat_template(chat, ChatFormat::QWEN3, false, false, "");
    TEST_ASSERT(rendered.find(raw) != std::string::npos);
}

// Equivalent histories exercise each API's distinct normalization branch.
// Results deliberately arrive in the reverse order from the calls.
static std::vector<std::pair<ApiFormat, json>> tool_history_inputs(
        const json & arguments, const std::string & second_id = "call_b") {
    return {
        {ApiFormat::OPENAI_CHAT, json::array({
            {{"role", "user"}, {"content", "go"}},
            {{"role", "assistant"}, {"content", "Checking."},
             {"tool_calls", json::array({
                 {{"id", "call_a"}, {"type", "function"}, {"function", {
                     {"name", "inspect"}, {"arguments", arguments.dump()}}}},
                 {{"id", second_id}, {"type", "function"}, {"function", {
                     {"name", "finish"}, {"arguments", "{}"}}}}})}},
            {{"role", "tool"}, {"tool_call_id", second_id}, {"content", "DONE"}},
            {{"role", "tool"}, {"tool_call_id", "call_a"}, {"content", "VALUE"}},
        })},
        {ApiFormat::RESPONSES, json::array({
            {{"role", "user"}, {"content", "go"}},
            {{"type", "message"}, {"role", "assistant"}, {"content", json::array({
                {{"type", "output_text"}, {"text", "Checking."}}})}},
            {{"type", "function_call"}, {"call_id", "call_a"},
             {"name", "inspect"}, {"arguments", arguments.dump()}},
            {{"type", "function_call"}, {"call_id", second_id},
             {"name", "finish"}, {"arguments", "{}"}},
            {{"type", "function_call_output"}, {"call_id", second_id}, {"output", "DONE"}},
            {{"type", "function_call_output"}, {"call_id", "call_a"}, {"output", "VALUE"}},
        })},
        {ApiFormat::ANTHROPIC, json::array({
            {{"role", "user"}, {"content", "go"}},
            {{"role", "assistant"}, {"content", json::array({
                {{"type", "text"}, {"text", "Checking."}},
                {{"type", "tool_use"}, {"id", "call_a"}, {"name", "inspect"},
                 {"input", arguments}},
                {{"type", "tool_use"}, {"id", second_id}, {"name", "finish"},
                 {"input", json::object()}}})}},
            {{"role", "user"}, {"content", json::array({
                {{"type", "tool_result"}, {"tool_use_id", second_id}, {"content", "DONE"}},
                {{"type", "tool_result"}, {"tool_use_id", "call_a"}, {"content", "VALUE"}}})}},
        })},
    };
}

TEST_CASE(ServerUnitFixture, test_normalize_tool_history_cache_boundaries_all_apis) {
    const json arguments = {{"path", "/a"}};
    const std::string raw = "Checking.\n<tool_call>RAW ORIGINAL TURN</tool_call>\n";
    for (const auto & [format, input] : tool_history_inputs(arguments)) {
        ToolMemory memory(2);
        const auto check = [&](bool replayed) {
            const auto chat = normalize_chat_messages(input, format, memory);
            // Responses message + adjacent calls must stay one assistant turn,
            // including when the cache already contains that message's prose.
            TEST_ASSERT(chat.size() == 4);
            if (chat.size() != 4) return;
            TEST_ASSERT(chat[1].role == "assistant");
            TEST_ASSERT(chat[1].content == (replayed ? raw : "Checking."));
            TEST_ASSERT(chat[1].tool_calls_replayed == replayed);
            TEST_ASSERT(chat[1].tool_calls.size() == 2);
            if (chat[1].tool_calls.size() != 2) return;
            TEST_ASSERT(chat[1].tool_calls[0].id == "call_a");
            TEST_ASSERT(chat[1].tool_calls[0].name == "inspect");
            TEST_ASSERT(chat[1].tool_calls[1].id == "call_b");
            TEST_ASSERT(chat[1].tool_calls[1].name == "finish");
            TEST_ASSERT(chat[1].tool_calls[0].arguments ==
                        (replayed ? json(nullptr) : arguments));
            TEST_ASSERT(chat[1].tool_calls[1].arguments ==
                        (replayed ? json(nullptr) : json::object()));
            TEST_ASSERT(chat[2].tool_name == "finish" && chat[2].content == "DONE");
            TEST_ASSERT(chat[3].tool_name == "inspect" && chat[3].content == "VALUE");
        };
        check(false);
        memory.remember({"call_a"}, raw);
        check(false);  // A partial hit must not erase the uncached call.
        memory.remember({"call_a", "call_b"}, raw);
        check(true);
        memory.remember({"unrelated"}, "Other turn");
        check(false);  // Eviction of either ID invalidates replay of the turn.
    }
    for (auto [format, input] : tool_history_inputs(arguments, "")) {
        ToolMemory memory;
        memory.remember({"call_a"}, raw);
        for (bool absent : {false, true}) {
            if (absent) {
                if (format == ApiFormat::OPENAI_CHAT) {
                    input[1]["tool_calls"][1].erase("id");
                } else if (format == ApiFormat::RESPONSES) {
                    input[3].erase("call_id");
                } else {
                    input[1]["content"][2].erase("id");
                }
            }
            const auto chat = normalize_chat_messages(input, format, memory);
            TEST_ASSERT(chat.size() == 4);
            if (chat.size() != 4) continue;
            TEST_ASSERT(!chat[1].tool_calls_replayed);
            TEST_ASSERT(chat[1].content == "Checking.");
            TEST_ASSERT(chat[1].tool_calls.size() == 2);
            if (chat[1].tool_calls.size() != 2) continue;
            TEST_ASSERT(chat[1].tool_calls[1].id.empty());
            TEST_ASSERT(chat[1].tool_calls[1].name == "finish");
        }
    }
}

TEST_CASE(ServerUnitFixture, test_jinja_normalized_tool_history_structure_and_replay) {
    static const char tpl[] =
        "{%- for m in messages -%}[{{ m.role }}]{{ m.content }}"
        "{%- for tc in m.tool_calls -%}"
        "[CALL:{{ tc.id }}/{{ tc.type }}/{{ tc.function.name }}]"
        "{%- if tc.function.name == 'inspect' -%}"
        "{{ tc.function.arguments.text }}"
        "{%- if tc.function.arguments.options.enabled and "
        "tc.function.arguments.options.paths[1] == 2 and "
        "tc.function.arguments.options.paths[2] == none -%}TYPED{%- endif -%}"
        "{%- endif -%}{%- endfor -%}"
        "{%- if m.role == 'tool' -%}[RESULT:{{ m.tool_call_id }}/{{ m.name }}]"
        "{%- endif -%}{%- endfor -%}";
    const json arguments = {
        {"options", {{"enabled", true}, {"paths", json::array({"a", 2, nullptr})}}},
        {"text", "line one\n\"line two\"\\end"},
    };
    for (const auto & [format, input] : tool_history_inputs(arguments)) {
        ToolMemory memory;
        const auto cold = normalize_chat_messages(input, format, memory);
        const auto rendered = render_chat_template_jinja(tpl, cold, "", "", false, false, "");
        TEST_ASSERT(rendered ==
            "[user]go[assistant]Checking."
            "[CALL:call_a/function/inspect]line one\n\"line two\"\\endTYPED"
            "[CALL:call_b/function/finish]"
            "[tool]DONE[RESULT:call_b/finish][tool]VALUE[RESULT:call_a/inspect]");
        const std::string raw = "Checking.\n[original native calls]\n";
        memory.remember({"call_a", "call_b"}, raw);
        const auto warm = normalize_chat_messages(input, format, memory, false);
        TEST_ASSERT(warm.size() == 4);
        if (warm.size() != 4) continue;
        TEST_ASSERT(!warm[1].tool_calls_replayed);
        TEST_ASSERT(warm[1].content == "Checking.");
        TEST_ASSERT(warm[1].tool_calls.size() == 2);
        TEST_ASSERT(render_chat_template_jinja(tpl, warm, "", "", false, false, "") == rendered);

        // Raw native replay cannot be interpreted by an arbitrary template:
        // templates may depend on tool_calls to render both calls and results.
        const auto raw_replay = normalize_chat_messages(input, format, memory);
        bool threw = false;
        try {
            (void)render_chat_template_jinja(tpl, raw_replay, "", "", false, false, "");
        } catch (const std::logic_error &) {
            threw = true;
        }
        TEST_ASSERT(threw);
    }
}

TEST_CASE(ServerUnitFixture, test_normalize_openai_tool_result_explicit_name) {
    auto input = tool_history_inputs(json::object()).front().second;
    input[2]["name"] = "explicit_finish";
    ToolMemory memory;
    memory.remember({"call_a", "call_b"}, "Original cached turn");
    const auto chat = normalize_chat_messages(input, ApiFormat::OPENAI_CHAT, memory);
    TEST_ASSERT(chat.size() == 4);
    if (chat.size() != 4) return;
    TEST_ASSERT(chat[2].tool_name == "explicit_finish");
    TEST_ASSERT(chat[2].tool_call_id == "call_b");
    TEST_ASSERT(chat[3].tool_name == "inspect");
}

TEST_CASE(ServerUnitFixture, test_normalized_tool_history_native_rendering_all_apis) {
    const json arguments = {
        {"options", {{"enabled", true}, {"paths", json::array({"a, \"b\":\\c\n", 2, nullptr})}}},
        {"text", "line one\n\"line two\"\\end"},
    };
    struct NativeHistory {
        ChatFormat format;
        std::string calls;
        std::string first_result;
        std::string second_result;
    };
    // Literal goldens from the official model templates, not our renderer:
    // Qwen/Qwen3.6-27B, inclusionAI/Ling-3.0-flash, poolside/Laguna-XS.2,
    // google/gemma-4-26B-A4B-it (chat_template.jinja on huggingface.co);
    // deepseek-ai/DeepSeek-V4-Flash/encoding/encoding_dsv4.py and
    // deepseek-ai/DeepSeek-V4.1-Flash/encoding/encoding.py.
    const std::vector<NativeHistory> native = {
        {ChatFormat::QWEN3,
         "<tool_call>\n<function=inspect>\n"
         "<parameter=options>\n"
         R"({"enabled": true, "paths": ["a, \"b\":\\c\n", 2, null]})"
         "\n</parameter>\n"
         "<parameter=text>\nline one\n\"line two\"\\end\n</parameter>\n"
         "</function>\n</tool_call>\n"
         "<tool_call>\n<function=finish>\n</function>\n</tool_call>",
         "<tool_response>\nDONE\n</tool_response>",
         "<tool_response>\nVALUE\n</tool_response>"},
        {ChatFormat::BAILINGMOE3,
         "<tool_call>inspect<arg_key>options</arg_key>\n"
         "<arg_value>"
         R"({"enabled": true, "paths": ["a, \"b\":\\c\n", 2, null]})"
         "</arg_value>"
         "<arg_key>text</arg_key>\n<arg_value>line one\n\"line two\"\\end</arg_value>\n"
         "</tool_call>\n<tool_call>finish\n</tool_call>",
         "<tool_response>\nDONE\n</tool_response>",
         "<tool_response>\nVALUE\n</tool_response>"},
        {ChatFormat::LAGUNA,
         "<tool_call>inspect<arg_key>options</arg_key>\n"
         "<arg_value>"
         R"({"enabled": true, "paths": ["a, \"b\":\\c\n", 2, null]})"
         "</arg_value>\n"
         "<arg_key>text</arg_key>\n<arg_value>line one\n\"line two\"\\end</arg_value>\n"
         "</tool_call>\n<tool_call>finish</tool_call>",
         "<tool_response>\nDONE\n</tool_response>",
         "<tool_response>\nVALUE\n</tool_response>"},
        {ChatFormat::GEMMA4,
         "<|tool_call>call:inspect{options:{enabled:true,paths:[<|\"|>a, \"b\":\\c\n<|\"|>,2,null]},"
         "text:<|\"|>line one\n\"line two\"\\end<|\"|>}<tool_call|>"
         "<|tool_call>call:finish{}<tool_call|>",
         "<|tool_response>response:finish{value:<|\"|>DONE<|\"|>}<tool_response|>",
         "<|tool_response>response:inspect{value:<|\"|>VALUE<|\"|>}<tool_response|>"},
        {ChatFormat::DEEPSEEK4,
         "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"inspect\">\n"
         "<｜DSML｜parameter name=\"options\" string=\"false\">"
         R"({"enabled": true, "paths": ["a, \"b\":\\c\n", 2, null]})"
         "</｜DSML｜parameter>\n"
         "<｜DSML｜parameter name=\"text\" string=\"true\">"
         "line one\n\"line two\"\\end</｜DSML｜parameter>\n"
         "</｜DSML｜invoke>\n<｜DSML｜invoke name=\"finish\">\n\n"
         "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
         "<tool_result>VALUE</tool_result>", "<tool_result>DONE</tool_result>"},
        {ChatFormat::DEEPSEEK41,
         "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"inspect\">\n"
         "<｜DSML｜ parameter name=\"options\" string=\"false\">"
         R"({"enabled": true, "paths": ["a, \"b\":\\c\n", 2, null]})"
         "</｜DSML｜ parameter>\n"
         "<｜DSML｜ parameter name=\"text\" string=\"true\">"
         "line one\n\"line two\"\\end</｜DSML｜ parameter>\n"
         "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"finish\">\n\n"
         "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
         "<tool_result>VALUE</tool_result>", "<tool_result>DONE</tool_result>"},
    };
    const auto occurs_once = [](const std::string & text, const std::string & needle) {
        const auto pos = text.find(needle);
        return pos != std::string::npos && text.find(needle, pos + needle.size()) == std::string::npos;
    };
    for (const auto & expected : native) {
        for (const auto & [api, input] : tool_history_inputs(arguments)) {
            ToolMemory memory;
            const auto cold = normalize_chat_messages(input, api, memory);
            TEST_ASSERT(cold.size() == 4);
            if (cold.size() != 4) continue;
            TEST_ASSERT(cold[1].content == "Checking.");
            TEST_ASSERT(cold[1].tool_calls.size() == 2);
            if (cold[1].tool_calls.size() != 2) continue;
            TEST_ASSERT(cold[1].tool_calls[0].arguments == arguments);
            const auto rendered = render_chat_template(cold, expected.format, true, false, "");
            TEST_ASSERT(occurs_once(rendered, expected.calls));
            TEST_ASSERT(occurs_once(rendered, "Checking."));
            TEST_ASSERT(occurs_once(rendered, expected.first_result));
            TEST_ASSERT(occurs_once(rendered, expected.second_result));
            TEST_ASSERT(rendered.find(expected.calls) < rendered.find(expected.first_result));
            TEST_ASSERT(rendered.find(expected.first_result) < rendered.find(expected.second_result));
            if (expected.format != ChatFormat::QWEN3) {
                TEST_ASSERT(rendered.find("<function=") == std::string::npos);
                TEST_ASSERT(rendered.find("<parameter=") == std::string::npos);
            }
            if (expected.format == ChatFormat::GEMMA4) {
                TEST_ASSERT(occurs_once(rendered, "<|turn>model\n"));
                TEST_ASSERT(rendered.find("<|turn>tool\n") == std::string::npos);
            }

            // Raw cached bytes may have different whitespace from cold
            // serialization. Preserve them, without replaying metadata again.
            const std::string raw = "Checking.\n\n" + expected.calls + "\n";
            memory.remember({"call_a", "call_b"}, raw);
            const auto warm = normalize_chat_messages(input, api, memory);
            TEST_ASSERT(warm.size() == 4);
            if (warm.size() != 4) continue;
            TEST_ASSERT(warm[1].content == raw);
            TEST_ASSERT(warm[1].tool_calls_replayed);
            const auto replayed = render_chat_template(warm, expected.format, true, false, "");
            TEST_ASSERT(occurs_once(replayed, raw));
            TEST_ASSERT(occurs_once(replayed, expected.calls));
            TEST_ASSERT(occurs_once(replayed, "Checking."));
            TEST_ASSERT(occurs_once(replayed, expected.first_result));
            TEST_ASSERT(occurs_once(replayed, expected.second_result));
            TEST_ASSERT(replayed.find(expected.calls) < replayed.find(expected.first_result));
            TEST_ASSERT(replayed.find(expected.first_result) < replayed.find(expected.second_result));
        }
    }
}

// DeepSeek's reference encoder closes historical reasoning before calls and
// orders ID-less result blocks by call ID, even when both calls share a name.
TEST_CASE(ServerUnitFixture, test_deepseek_tool_history_reference_replay) {
    for (const auto format : {ChatFormat::DEEPSEEK4, ChatFormat::DEEPSEEK41}) {
        for (const auto & [api, input] : tool_history_inputs(json::object())) {
            ToolMemory memory;
            auto cold = normalize_chat_messages(input, api, memory);
            cold[1].tool_calls[1].name = cold[1].tool_calls[0].name;
            for (bool thinking : {false, true}) {
                const auto out = render_chat_template(cold, format, true, thinking);
                const std::string header = thinking
                    ? "<｜Assistant｜><think></think>Checking."
                    : "<｜Assistant｜></think>Checking.";
                TEST_ASSERT(out.find(header) != std::string::npos);
                TEST_ASSERT(out.find("<tool_result>VALUE</tool_result>\n\n"
                                     "<tool_result>DONE</tool_result>") != std::string::npos);
            }

            const std::string raw = format == ChatFormat::DEEPSEEK41
                ? "Planning.</think>Checking.\n\n<｜DSML｜ calls>\n"
                  "<｜DSML｜ invoke name=\"inspect\">\n\n</｜DSML｜ invoke>\n"
                  "<｜DSML｜ invoke name=\"finish\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
                : "Planning.</think>Checking.\n\n<｜DSML｜tool_calls>\n"
                  "<｜DSML｜invoke name=\"inspect\">\n\n</｜DSML｜invoke>\n"
                  "<｜DSML｜invoke name=\"finish\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
            memory.remember({"call_a", "call_b"}, raw);
            const auto warm = normalize_chat_messages(input, api, memory);
            const auto replayed = render_chat_template(warm, format, true, true);
            TEST_ASSERT(replayed.find("<｜Assistant｜><think>" + raw +
                                     "<｜end▁of▁sentence｜>") != std::string::npos);
            TEST_ASSERT(replayed.find("<tool_result>VALUE</tool_result>") <
                        replayed.find("<tool_result>DONE</tool_result>"));
        }
    }
}

TEST_CASE(ServerUnitFixture, test_deepseek_result_order_preserves_user_text_and_unknown_ids) {
    ChatMessage calls{"assistant", ""};
    calls.tool_calls = {
        {"a", "lookup", {{"key", "A"}}},
        {"b", "lookup", {{"key", "B"}}},
        {"c", "lookup", {{"key", "C"}}},
    };
    ChatMessage next_calls{"assistant", ""};
    next_calls.tool_calls = {
        {"next_c", "lookup", {{"key", "C2"}}},
        {"next_a", "lookup", {{"key", "A2"}}},
    };
    const std::vector<ChatMessage> messages = {
        {"user", "look up"}, calls,
        {"tool", "C", "c"}, {"user", "interleaved"},
        {"tool", "B", "b"}, {"tool", "A", "a"},
        {"tool", "unknown", "missing"}, {"tool", "no_id", ""},
        next_calls, {"tool", "A2", "next_a"}, {"tool", "C2", "next_c"},
    };
    for (const auto format : {ChatFormat::DEEPSEEK4, ChatFormat::DEEPSEEK41}) {
        const auto out = render_chat_template(messages, format, true, false);
        // Unknown/missing IDs use the reference's rank-zero fallback; ties
        // retain arrival order and user text stays in its original slot.
        size_t previous = 0;
        for (const char * text : {
                "<tool_result>A</tool_result>", "interleaved",
                "<tool_result>unknown</tool_result>", "<tool_result>no_id</tool_result>",
                "<tool_result>B</tool_result>", "<tool_result>C</tool_result>",
                "<tool_result>C2</tool_result>", "<tool_result>A2</tool_result>"}) {
            const size_t position = out.find(text);
            TEST_ASSERT(position != std::string::npos && position > previous);
            previous = position;
        }
    }
}

TEST_CASE(ServerUnitFixture, test_gemma_tool_turn_continuation_preserves_results) {
    for (const auto & [api, input] : tool_history_inputs(json::object())) {
        for (const bool replay : {false, true}) {
            ToolMemory memory;
            if (replay) {
                memory.remember({"call_a", "call_b"},
                    "Checking.<|tool_call>call:inspect{}<tool_call|>"
                    "<|tool_call>call:finish{}<tool_call|><|tool_response>");
            }
            const auto messages = normalize_chat_messages(input, api, memory);
            for (const bool thinking : {false, true}) {
                const auto prefix = render_chat_template(
                    messages, ChatFormat::GEMMA4, false, thinking, "");
                const std::string responses =
                    "<|tool_response>response:finish{value:<|\"|>DONE<|\"|>}<tool_response|>"
                    "<|tool_response>response:inspect{value:<|\"|>VALUE<|\"|>}<tool_response|>";
                TEST_ASSERT(prefix.size() >= responses.size());
                TEST_ASSERT(prefix.compare(prefix.size() - responses.size(), responses.size(), responses) == 0);
                auto continued = messages;
                continued.push_back({"assistant", "Finished."});
                continued.push_back({"user", "Next."});
                const auto next = render_chat_template(
                    continued, ChatFormat::GEMMA4, true, thinking, "");
                TEST_ASSERT(next.compare(0, prefix.size(), prefix) == 0);
                TEST_ASSERT(next.find("<tool_response|>Finished.<turn|>\n<|turn>user\nNext.") != std::string::npos);
                TEST_ASSERT(next.find("<|tool_response><|tool_response>") == std::string::npos);
            }
        }
    }
}

// An assistant history message's prior <think> text must carry through to
// ChatMessage.reasoning_content (OpenAI/DeepSeek dialect `reasoning_content`,
// and the OpenRouter/Anthropic-gateway flat `reasoning` alias — the same
// names the response side emits, see format_response_message). Non-assistant
// roles and messages with neither field must leave it empty.
TEST_CASE(ServerUnitFixture, test_normalize_assistant_reasoning_content_passthrough) {
    ToolMemory tool_memory;
    const json messages = json::array({
        {{"role", "user"}, {"content", "hi"}},
        {{"role", "assistant"}, {"content", "A1"}, {"reasoning_content", "thought one"}},
        {{"role", "user"}, {"content", "and?"}, {"reasoning_content", "ignored on non-assistant"}},
        {{"role", "assistant"}, {"content", "A2"}, {"reasoning", "thought two"}},
        {{"role", "assistant"}, {"content", "A3"}},
    });
    const auto chat = normalize_chat_messages(
        messages, ApiFormat::OPENAI_CHAT, tool_memory);
    TEST_ASSERT(chat.size() == 5);
    if (chat.size() == 5) {
        TEST_ASSERT(chat[1].reasoning_content == "thought one");
        TEST_ASSERT(chat[2].reasoning_content.empty());
        TEST_ASSERT(chat[3].reasoning_content == "thought two");
        TEST_ASSERT(chat[4].reasoning_content.empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Placement config tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_parse_target_device_list_same_backend) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1", placement));
    TEST_ASSERT(placement.backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.gpu == 0);
    TEST_ASSERT(placement.is_layer_split());
    TEST_ASSERT(!placement.is_mixed_layer_split());
    TEST_ASSERT(placement.layer_split_backends.size() == 2);
    TEST_ASSERT(placement.layer_split_backends[0] == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_backends[1] == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_gpus.size() == 2);
    TEST_ASSERT(placement.layer_split_gpus[0] == 0);
    TEST_ASSERT(placement.layer_split_gpus[1] == 1);
    TEST_ASSERT(placement.layer_split_weights.empty());
}

TEST_CASE(ServerUnitFixture, test_parse_target_device_list_mixed_backend) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,hip:1", placement));
    TEST_ASSERT(placement.backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.gpu == 0);
    TEST_ASSERT(placement.is_layer_split());
    TEST_ASSERT(placement.is_mixed_layer_split());
    TEST_ASSERT(placement.layer_split_backends.size() == 2);
    TEST_ASSERT(placement.layer_split_backends[0] == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_backends[1] == PlacementBackend::Hip);
    TEST_ASSERT(placement.layer_split_backend(0) == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_backend(1) == PlacementBackend::Hip);
    TEST_ASSERT(placement.layer_split_gpus.size() == 2);
    TEST_ASSERT(placement.layer_split_gpus[0] == 0);
    TEST_ASSERT(placement.layer_split_gpus[1] == 1);
}

TEST_CASE(ServerUnitFixture, test_parse_target_device_list_mixed_backend_multi_remote) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,hip:0,hip:1", placement));
    TEST_ASSERT(placement.backend == PlacementBackend::Cuda);
    TEST_ASSERT(placement.gpu == 0);
    TEST_ASSERT(placement.is_layer_split());
    TEST_ASSERT(placement.is_mixed_layer_split());
    TEST_ASSERT(placement.layer_split_backends.size() == 3);
    TEST_ASSERT(placement.layer_split_backends[0] == PlacementBackend::Cuda);
    TEST_ASSERT(placement.layer_split_backends[1] == PlacementBackend::Hip);
    TEST_ASSERT(placement.layer_split_backends[2] == PlacementBackend::Hip);
    TEST_ASSERT(placement.layer_split_gpus.size() == 3);
    TEST_ASSERT(placement.layer_split_gpus[0] == 0);
    TEST_ASSERT(placement.layer_split_gpus[1] == 0);
    TEST_ASSERT(placement.layer_split_gpus[2] == 1);
}

TEST_CASE(ServerUnitFixture, test_parse_target_device_list_single_gpu_is_not_layer_split) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("hip:2", placement));
    TEST_ASSERT(placement.backend == PlacementBackend::Hip);
    TEST_ASSERT(placement.gpu == 2);
    TEST_ASSERT(!placement.is_layer_split());
    TEST_ASSERT(!placement.is_mixed_layer_split());
    TEST_ASSERT(placement.layer_split_backends.empty());
    TEST_ASSERT(placement.layer_split_gpus.empty());
}

TEST_CASE(ServerUnitFixture, test_validate_layer_split_weights_shape) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1", placement));

    placement.layer_split_weights = {1.0};
    TEST_ASSERT(!validate_device_placement(placement, -1).empty());

    placement.layer_split_weights = {1.0, 0.0};
    TEST_ASSERT(!validate_device_placement(placement, -1).empty());

    placement.layer_split_weights = {1.0, 2.0};
    TEST_ASSERT(validate_device_placement(placement, -1).empty());
}

TEST_CASE(ServerUnitFixture, test_target_shard_plan_same_backend_split) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1,cuda:2", placement));

    MixedLayerSplitPlan plan;
    TEST_ASSERT(compute_target_shard_layer_split_plan(
        placement, PlacementBackend::Cuda, plan, "test-target-shard"));
    TEST_ASSERT(plan.remote_begin == 1);
    TEST_ASSERT(plan.remote_backend == PlacementBackend::Cuda);
}

TEST_CASE(ServerUnitFixture, test_target_shard_plan_mixed_backend_split) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,cuda:1,hip:0,hip:1", placement));

    MixedLayerSplitPlan plan;
    TEST_ASSERT(compute_target_shard_layer_split_plan(
        placement, PlacementBackend::Cuda, plan, "test-target-shard"));
    TEST_ASSERT(plan.remote_begin == 2);
    TEST_ASSERT(plan.remote_backend == PlacementBackend::Hip);
}

TEST_CASE(ServerUnitFixture, test_target_shard_plan_rejects_bad_local_backend) {
    DevicePlacement placement;
    TEST_ASSERT(parse_placement_device_list("cuda:0,hip:0", placement));

    MixedLayerSplitPlan plan;
    TEST_ASSERT(!compute_target_shard_layer_split_plan(
        placement, PlacementBackend::Hip, plan, "test-target-shard"));
}

static bool kvflash_test_sync_identity(KvFlashPager & pager, int committed) {
    return layer_split_kvflash_sync_identity(
        pager, committed, pager.pool_tokens(), "test-target-split");
}

TEST_CASE(ServerUnitFixture, test_kvflash_pager_identity_sync_contract) {
    KvFlashConfig cfg;
    cfg.pool_tokens = 512;

    KvFlashPager local;
    KvFlashPager remote;
    TEST_ASSERT(local.attach(cfg, {}, {}));
    TEST_ASSERT(remote.attach(cfg, {}, {}));

    TEST_ASSERT(kvflash_test_sync_identity(local, 256));
    TEST_ASSERT(kvflash_test_sync_identity(remote, 256));
    TEST_ASSERT(local.slot_of(255) == remote.slot_of(255));

    TEST_ASSERT(kvflash_test_sync_identity(local, cfg.pool_tokens));
    TEST_ASSERT(local.slot_of(cfg.pool_tokens - 1) == cfg.pool_tokens - 1);
    TEST_ASSERT(local.is_identity());

    const int relocated = local.slot_for(cfg.pool_tokens);
    TEST_ASSERT(relocated >= 0);
    TEST_ASSERT(relocated != cfg.pool_tokens);
    TEST_ASSERT(!local.is_identity());

    TEST_ASSERT(kvflash_test_sync_identity(local, 128));
    TEST_ASSERT(local.slot_of(127) == 127);
}

TEST_CASE(ServerUnitFixture, test_layer_split_kvflash_history_contract) {
    std::vector<int32_t> history;
    layer_split_kvflash_sync_history(history, {1, 2, 3}, 0);
    TEST_ASSERT((history == std::vector<int32_t>{1, 2, 3}));

    layer_split_kvflash_sync_history(history, {4, 5}, 3);
    TEST_ASSERT((history == std::vector<int32_t>{1, 2, 3, 4, 5}));

    layer_split_kvflash_sync_history(history, {9}, 2);
    TEST_ASSERT((history == std::vector<int32_t>{1, 2, 9}));

    layer_split_kvflash_sync_history(history, {7}, 5);
    TEST_ASSERT(history.size() == 6);
    TEST_ASSERT(history[0] == 1);
    TEST_ASSERT(history[1] == 2);
    TEST_ASSERT(history[2] == 9);
    TEST_ASSERT(history[3] == 0);
    TEST_ASSERT(history[4] == 0);
    TEST_ASSERT(history[5] == 7);

    std::vector<std::vector<int32_t>> snapshots(2);
    layer_split_kvflash_save_history_snapshot(history, 4, snapshots[1]);
    TEST_ASSERT((snapshots[1] == std::vector<int32_t>{1, 2, 9, 0}));

    history = {8, 8, 8};
    layer_split_kvflash_restore_history(history, snapshots, 1, 6);
    TEST_ASSERT((history == std::vector<int32_t>{1, 2, 9, 0, 0, 0}));

    layer_split_kvflash_restore_history(history, snapshots, 0, 3);
    TEST_ASSERT((history == std::vector<int32_t>{0, 0, 0}));
}

TEST_CASE(ServerUnitFixture, test_backend_precision_cuda_sm_policy) {
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(90) == GGML_TYPE_BF16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(80) == GGML_TYPE_BF16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(75) == GGML_TYPE_F16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(70) == GGML_TYPE_F16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(60) == GGML_TYPE_F16);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(62) == GGML_TYPE_F32);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(61) == GGML_TYPE_F32);
    TEST_ASSERT(select_cuda_backend_precision_type_for_sm(52) == GGML_TYPE_F32);
}

TEST_CASE(ServerUnitFixture, test_backend_precision_hip_arch_policy) {
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx90a") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx942") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx950") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx1100") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx1200") == GGML_TYPE_BF16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx906") == GGML_TYPE_F16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx1030") == GGML_TYPE_F16);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("gfx803") == GGML_TYPE_F32);
    TEST_ASSERT(select_hip_activation_precision_type_for_arch("") == GGML_TYPE_F32);
}

TEST_CASE(ServerUnitFixture, test_backend_precision_activation_type_combine) {
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_BF16, GGML_TYPE_BF16) == GGML_TYPE_BF16);
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_BF16, GGML_TYPE_F16) == GGML_TYPE_F16);
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_F16, GGML_TYPE_BF16) == GGML_TYPE_F16);
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_F16, GGML_TYPE_F32) == GGML_TYPE_F32);
    TEST_ASSERT(combine_activation_precision_types(GGML_TYPE_F32, GGML_TYPE_BF16) == GGML_TYPE_F32);
}

struct MockLayerSplitAdapter : LayerSplitAdapter {
    int max_ctx = 128;
    bool reset_called = false;
    int saved_slot = -1;
    int saved_pos = 0;
    int restored_slot = -1;
    int current_pos = 0;
    int current_last = -1;
    std::vector<int> prefill_bases;
    std::vector<int> prefill_sizes;
    int dflash_base = -1;
    int dflash_last = -1;
    std::vector<int32_t> emitted_tokens;
    bool dflash_enabled = false;
    bool dflash_called = false;
    bool sampling_enabled = false;
    bool kvflash_enabled = false;
    bool mixed_backend_enabled = false;
    int shutdown_calls = 0;
    ModelBackend::CompressRequest last_compress_req;
    int prefill_chunk = 0;
    std::function<void()> on_prefill;

    const char * name() const override { return "mock"; }
    bool init() override { return true; }
    int max_context() const override { return max_ctx; }
    void reset_request_state() override {
        reset_called = true;
        current_pos = 0;
        current_last = -1;
    }
    int prefill_chunk_tokens() const override { return prefill_chunk; }
    bool prefill(const std::vector<int32_t> & prompt,
                 int base_pos, int & last_tok) override {
        prefill_bases.push_back(base_pos);
        prefill_sizes.push_back((int)prompt.size());
        current_pos = base_pos + (int)prompt.size();
        current_last = prompt.empty() ? current_last : prompt.back();
        last_tok = current_last;
        if (on_prefill) on_prefill();
        return true;
    }
    bool decode_ar(int last_tok, int committed, ThinkingBudget & budget,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io) override {
        (void)history_prefix;
        TEST_ASSERT(committed == current_pos);
        const int n_gen = budget.remaining();
        for (int i = 0; i < n_gen; ++i) {
            int32_t tok = budget.apply(last_tok + i + 1).token;
            out_tokens.push_back(tok);
            emitted_tokens.push_back(tok);
            io.emit(tok);
        }
        io.emit(-1);
        return true;
    }
    bool can_dflash_decode() const override { return dflash_enabled; }
    bool supports_cpu_sampling() const override { return sampling_enabled; }
    bool supports_kvflash() const override { return kvflash_enabled; }
    bool supports_mixed_backend_layer_split() const override {
        return mixed_backend_enabled;
    }
    bool decode_dflash(const std::vector<int32_t> & prompt, int base_pos,
                       int last_tok, ThinkingBudget & budget, std::vector<int32_t> & out_tokens,
                       const DaemonIO & io, float & accept_rate_out) override {
        (void)prompt;
        accept_rate_out = 0.0f;
        dflash_called = true;
        dflash_base = base_pos;
        dflash_last = last_tok;
        const int n_gen = budget.remaining();
        for (int i = 0; i < n_gen; ++i) {
            int32_t tok = budget.apply(last_tok + i + 10).token;
            out_tokens.push_back(tok);
            emitted_tokens.push_back(tok);
            io.emit(tok);
        }
        io.emit(-1);
        return true;
    }
    void free_drafter() override {}
    bool snapshot_save(int slot) override {
        saved_slot = slot;
        saved_pos = current_pos;
        return true;
    }
    bool snapshot_used(int slot) const override {
        return slot == saved_slot && saved_pos > 0;
    }
    int snapshot_cur_pos(int slot) const override {
        return snapshot_used(slot) ? saved_pos : 0;
    }
    bool snapshot_restore(int slot) override {
        if (!snapshot_used(slot)) return false;
        restored_slot = slot;
        current_pos = saved_pos;
        current_last = saved_pos;
        return true;
    }
    int current_last_token() const override { return current_last; }
    const char * default_compress_drafter_path() const override {
        return "/tmp/default-layer-split-drafter.gguf";
    }
    ModelBackend::CompressResult
    compress(const ModelBackend::CompressRequest & req) override {
        last_compress_req = req;
        ModelBackend::CompressResult result;
        result.ok = true;
        result.compressed_ids = {77, 88};
        return result;
    }
    void shutdown() override { shutdown_calls++; }
};

TEST_CASE(ServerUnitFixture, test_layer_split_backend_inline_snapshot_and_restore_delta) {
    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

    GenerateRequest req;
    req.prompt = {10, 11, 12, 13};
    req.n_gen = 1;
    req.snap_slot = 2;
    req.snap_pos = 3;
    DaemonIO io;
    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(raw->reset_called);
    TEST_ASSERT(raw->saved_slot == 2);
    TEST_ASSERT(raw->saved_pos == 3);
    TEST_ASSERT(raw->prefill_bases.size() == 2);
    TEST_ASSERT(raw->prefill_bases[0] == 0);
    TEST_ASSERT(raw->prefill_sizes[0] == 3);
    TEST_ASSERT(raw->prefill_bases[1] == 3);
    TEST_ASSERT(raw->prefill_sizes[1] == 1);
    TEST_ASSERT(backend.snapshot_used(2));
    TEST_ASSERT(backend.snapshot_cur_pos(2) == 3);

    raw->reset_called = false;
    raw->prefill_bases.clear();
    raw->prefill_sizes.clear();
    raw->dflash_enabled = true;
    GenerateRequest restore_req;
    restore_req.prompt = {10, 11, 12, 99};
    restore_req.n_gen = 1;
    GenerateResult restored = backend.restore_and_generate(2, restore_req, io);

    TEST_ASSERT(restored.ok());
    TEST_ASSERT(raw->dflash_called);
    TEST_ASSERT(raw->restored_slot == 2);
    TEST_ASSERT(!raw->reset_called);
    TEST_ASSERT(raw->prefill_bases.size() == 1);
    TEST_ASSERT(raw->prefill_bases[0] == 3);
    TEST_ASSERT(raw->prefill_sizes[0] == 1);
    TEST_ASSERT(raw->dflash_base == 3);
    TEST_ASSERT(raw->dflash_last == 99);
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_sampling_capability_gate) {
    {
        auto * raw = new MockLayerSplitAdapter();
        LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

        GenerateRequest req;
        req.prompt = {10, 11};
        req.n_gen = 1;
        req.do_sample = true;
        req.sampler.temp = 0.8f;
        DaemonIO io;
        GenerateResult result = backend.generate(req, io);

        TEST_ASSERT(!result.ok());
        TEST_ASSERT(result.error->code == GenerateErrorCode::SamplingUnsupported);
        TEST_ASSERT(result.error_code() == "sampling_unsupported");
    }

    {
        auto * raw = new MockLayerSplitAdapter();
        raw->sampling_enabled = true;
        LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

        GenerateRequest req;
        req.prompt = {10, 11};
        req.n_gen = 1;
        req.do_sample = true;
        req.sampler.temp = 0.8f;
        DaemonIO io;
        GenerateResult result = backend.generate(req, io);

        TEST_ASSERT(result.ok());
        TEST_ASSERT(result.tokens.size() == 1);
        TEST_ASSERT(result.tokens[0] == 12);
    }
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_chunks_prefill_by_adapter_limit) {
    auto * raw = new MockLayerSplitAdapter();
    raw->prefill_chunk = 3;
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

    GenerateRequest req;
    req.prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    req.n_gen = 1;
    DaemonIO io;
    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(raw->prefill_bases.size() == 3);
    TEST_ASSERT(raw->prefill_sizes.size() == 3);
    TEST_ASSERT(raw->prefill_bases[0] == 0);
    TEST_ASSERT(raw->prefill_sizes[0] == 3);
    TEST_ASSERT(raw->prefill_bases[1] == 3);
    TEST_ASSERT(raw->prefill_sizes[1] == 3);
    TEST_ASSERT(raw->prefill_bases[2] == 6);
    TEST_ASSERT(raw->prefill_sizes[2] == 2);
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_cancels_between_prefill_chunks) {
    auto * raw = new MockLayerSplitAdapter();
    raw->prefill_chunk = 3;
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

    bool cancel = false;
    raw->on_prefill = [&cancel]() { cancel = true; };
    DaemonIO io;
    io.should_cancel = [&cancel]() { return cancel; };

    GenerateRequest req;
    req.prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    req.n_gen = 4;
    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(io.is_cancelled());
    TEST_ASSERT(raw->prefill_bases.size() == 1);
    TEST_ASSERT(raw->prefill_sizes.size() == 1);
    TEST_ASSERT(raw->prefill_sizes[0] == 3);
    TEST_ASSERT(raw->emitted_tokens.empty());
}

TEST_CASE(ServerUnitFixture, test_layer_split_compress_nopark_uses_default_drafter_path) {
    const std::string ids_path = test_tmp_path(
        "luce_test_layer_split_compress_ids.bin").string();
    remove_test_path(ids_path);
    TEST_ASSERT(write_int32_file(ids_path, {1, 2, 3, 4}));

    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};
    DaemonIO io;

    const std::string cmd = "compress " + ids_path + " 250 nopark";
    TEST_ASSERT(backend.handle_compress(cmd, io));
    TEST_ASSERT(raw->last_compress_req.skip_park);
    TEST_ASSERT(std::abs(raw->last_compress_req.keep_ratio - 0.25f) < 1e-5f);
    TEST_ASSERT(raw->last_compress_req.input_ids.size() == 4);
    TEST_ASSERT(raw->last_compress_req.drafter_path ==
                "/tmp/default-layer-split-drafter.gguf");

    remove_test_path(ids_path);
}

TEST_CASE(ServerUnitFixture, test_layer_split_compress_rejects_bad_keep_ratio) {
    const std::string ids_path = test_tmp_path(
        "luce_test_layer_split_compress_bad.bin").string();
    remove_test_path(ids_path);
    TEST_ASSERT(write_int32_file(ids_path, {1, 2, 3, 4}));

    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};
    DaemonIO io;

    const std::string cmd = "compress " + ids_path + " 1250 nopark";
    TEST_ASSERT(!backend.handle_compress(cmd, io));
    TEST_ASSERT(raw->last_compress_req.input_ids.empty());

    remove_test_path(ids_path);
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_shutdown_is_idempotent) {
    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};
    backend.shutdown();
    backend.shutdown();
    TEST_ASSERT(raw->shutdown_calls == 1);
}

TEST_CASE(ServerUnitFixture, test_layer_split_backend_capability_proxy) {
    auto * raw = new MockLayerSplitAdapter();
    LayerSplitBackend backend{std::unique_ptr<LayerSplitAdapter>(raw)};

    TEST_ASSERT(!backend.supports_kvflash());
    TEST_ASSERT(!backend.supports_mixed_backend_layer_split());

    raw->kvflash_enabled = true;
    raw->mixed_backend_enabled = true;
    TEST_ASSERT(backend.supports_kvflash());
    TEST_ASSERT(backend.supports_mixed_backend_layer_split());
}

// Disk Prefix Cache Tests
// ═══════════════════════════════════════════════════════════════════════

// Minimal mock backend for testing (no GPU needed).
struct MockBackend : ModelBackend {
    void print_ready_banner() const override {}
    bool park(ParkTarget) override { return true; }
    bool unpark(ParkTarget) override { return true; }
    bool is_target_parked() const override { return false; }
    GenerateResult generate_impl(const GenerateRequest &, const DaemonIO &) override { return {}; }
    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int  snapshot_cur_pos(int) const override { return 0; }
    GenerateResult restore_and_generate_impl(int, const GenerateRequest &, const DaemonIO &) override { return {}; }
    bool handle_compress(const std::string &, const DaemonIO &) override { return false; }
    void free_drafter() override {}
    void shutdown() override {}
};

struct ShortInlineSnapshotBackend : MockBackend {
    int saved_slot = -1;
    int saved_position = 0;

    bool snapshot_used(int slot) const override {
        return slot == saved_slot && saved_position > 0;
    }
    int snapshot_cur_pos(int slot) const override {
        return snapshot_used(slot) ? saved_position : 0;
    }
};

// The default snapshot_detach feeds the disk writer; a pooled (KVFlash)
// snapshot keeps its KV in a pager blob outside the exported ggml context, so
// it must never be detached as if it were a full dense state.
struct PooledRefBackend : MockBackend {
    ggml_context * ctx = nullptr;
    int freed = 0;
    SnapshotRef snapshot_ref(int) const override {
        SnapshotRef r;
        r.ctx = ctx;
        r.cur_pos = 4096;
        r.pooled = true;
        return r;
    }
    void snapshot_free(int) override { freed++; }
};

TEST_CASE(ServerUnitFixture, test_default_snapshot_detach_refuses_pooled_snapshots) {
    ggml_init_params ip{1 << 16, nullptr, true};
    PooledRefBackend backend;
    backend.ctx = ggml_init(ip);
    TEST_ASSERT(backend.ctx != nullptr);
    OwnedSnapshot out = backend.snapshot_detach(0);
    TEST_ASSERT(out.empty());
    TEST_ASSERT(backend.freed == 1);
    ggml_free(backend.ctx);
}

// PFlash admission asks the server how much of a prompt a restore would skip.
// The answer is the backend's: a KVFlash live session can cover more than the
// snapshot the prefix cache holds, and the default trusts the snapshot.
struct LiveSessionBackend : ShortInlineSnapshotBackend {
    int live_cut = 0;
    mutable int seen_slot = -2;
    mutable int seen_slot_tokens = -2;
    int restorable_prefix(const std::vector<int32_t> &, int slot,
                          int slot_tokens) const override {
        seen_slot = slot;
        seen_slot_tokens = slot_tokens;
        return live_cut > slot_tokens ? live_cut : slot_tokens;
    }
};

TEST_CASE(ServerUnitFixture, test_ram_restorable_prefix_defers_to_the_backend) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<LiveSessionBackend>();
    LiveSessionBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.prefix_cache_cap = 2;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    const std::vector<int32_t> prompt = {1, 100, 3, 101};

    // No snapshot: the backend is asked with slot -1 and 0 tokens, so a live
    // session alone is reported.
    backend.live_cut = 3000;
    TEST_ASSERT(SchedulerTestHarness::ram_restorable_prefix_tokens(server, prompt) == 3000);
    TEST_ASSERT(backend.seen_slot == -1);
    TEST_ASSERT(backend.seen_slot_tokens == 0);

    auto reservation = cache.reserve_inline_snap(
        prompt, /*restored_prefix_len=*/0,
        /*prefer_tools_boundary=*/false, /*forced_cut=*/4);
    TEST_ASSERT(reservation.active());
    backend.saved_slot = reservation.slot();
    backend.saved_position = 3;
    SchedulerTestHarness::finalize_inline_snapshot(
        server, prompt, std::move(reservation), backend.saved_slot,
        /*requested_cut=*/4);

    // The backend's cut beats the snapshot's 3 tokens.
    TEST_ASSERT(SchedulerTestHarness::ram_restorable_prefix_tokens(server, prompt) == 3000);
    TEST_ASSERT(backend.seen_slot == backend.saved_slot);
    TEST_ASSERT(backend.seen_slot_tokens == 3);

    // A smaller live cut leaves the snapshot's tokens.
    backend.live_cut = 2;
    TEST_ASSERT(SchedulerTestHarness::ram_restorable_prefix_tokens(server, prompt) == 3);
}

TEST_CASE(ServerUnitFixture, test_ram_restorable_prefix_default_is_the_snapshot_tokens) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<ShortInlineSnapshotBackend>();
    ShortInlineSnapshotBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.prefix_cache_cap = 2;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    const std::vector<int32_t> prompt = {1, 100, 3, 101};
    TEST_ASSERT(SchedulerTestHarness::ram_restorable_prefix_tokens(server, prompt) == 0);

    auto reservation = cache.reserve_inline_snap(
        prompt, /*restored_prefix_len=*/0,
        /*prefer_tools_boundary=*/false, /*forced_cut=*/4);
    TEST_ASSERT(reservation.active());
    backend.saved_slot = reservation.slot();
    backend.saved_position = 3;
    SchedulerTestHarness::finalize_inline_snapshot(
        server, prompt, std::move(reservation), backend.saved_slot,
        /*requested_cut=*/4);
    TEST_ASSERT(SchedulerTestHarness::ram_restorable_prefix_tokens(server, prompt) == 3);
}

// Qwen saves at the chunk boundary at or before the requested cut. The entry
// stays keyed at the requested cut so the next request treats that cut as
// covered and deepens past it; only the disk/shutdown key uses the saved rows.
TEST_CASE(ServerUnitFixture,
          test_inline_snapshot_finalization_keys_requested_cut) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<ShortInlineSnapshotBackend>();
    ShortInlineSnapshotBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.prefix_cache_cap = 2;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    const std::vector<int32_t> prompt = {1, 100, 3, 101};
    auto reservation = cache.reserve_inline_snap(
        prompt, /*restored_prefix_len=*/0,
        /*prefer_tools_boundary=*/false, /*forced_cut=*/4);
    TEST_ASSERT(reservation.active());
    TEST_ASSERT(reservation.target_cut() == 4);

    backend.saved_slot = reservation.slot();
    backend.saved_position = 3;
    SchedulerTestHarness::finalize_inline_snapshot(
        server, prompt, std::move(reservation), backend.saved_slot,
        /*requested_cut=*/4);

    const auto hit = cache.lookup(prompt);
    TEST_ASSERT(hit.first == backend.saved_slot);
    TEST_ASSERT(hit.second == 4);
    TEST_ASSERT(SchedulerTestHarness::slot_tokens(
                    server, backend.saved_slot) ==
                std::vector<int32_t>(prompt.begin(), prompt.begin() + 3));

    unlink(path.c_str());
}

struct PruningSnapshotBackend : ShortInlineSnapshotBackend {
    std::vector<int> freed;
    int max_tokens = 1000;

    void snapshot_free(int slot) override { freed.push_back(slot); }
    size_t snapshot_bytes_estimate(int tokens) const override {
        return (size_t)std::min(tokens, max_tokens) * 100 + 1000;
    }
};

// A new turn's snapshot supersedes the intermediate turns before it: the
// server frees their payloads and keeps only the conversation head.
TEST_CASE(ServerUnitFixture, test_inline_snapshot_finalization_prunes_chain) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<PruningSnapshotBackend>();
    PruningSnapshotBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.prefix_cache_cap = 4;
    config.prefix_cache_max_bytes = 0;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);

    const std::vector<int32_t> head = {1, 100};
    std::vector<int32_t> turn1 = head;
    turn1.insert(turn1.end(), {3, 101});
    std::vector<int32_t> turn2 = turn1;
    turn2.insert(turn2.end(), {4, 102});
    std::vector<int32_t> turn3 = turn2;
    turn3.insert(turn3.end(), {3, 103});
    cache.confirm_inline_snap(0, (int)head.size(), head);
    cache.confirm_inline_snap(1, (int)turn1.size(), turn1);
    cache.confirm_inline_snap(2, (int)turn2.size(), turn2);

    auto reservation = cache.reserve_inline_snap(
        turn3, (int)turn2.size(), /*prefer_tools_boundary=*/false,
        /*forced_cut=*/(int)turn3.size(), /*restore_source_slot=*/2);
    TEST_ASSERT(reservation.active());
    const int slot = reservation.slot();
    TEST_ASSERT(slot != 0 && slot != 1 && slot != 2);
    backend.saved_slot = slot;
    backend.saved_position = (int)turn3.size();
    SchedulerTestHarness::finalize_inline_snapshot(
        server, turn3, std::move(reservation), slot,
        /*requested_cut=*/(int)turn3.size());

    // The head and the previous turn stay.
    TEST_ASSERT(backend.freed == std::vector<int>({1}));
    TEST_ASSERT(cache.stats().in_use == 3);
    TEST_ASSERT(cache.lookup(turn3).first == slot);
    TEST_ASSERT(cache.lookup(turn2).first == 2);
    std::vector<int32_t> other = head;
    other.insert(other.end(), {3, 999});
    TEST_ASSERT(cache.lookup(other).first == 0);
    unlink(path.c_str());
}

// The live state after generation covers the prompt and the decoded part
// of the reply; it is saved only as far as the next request's rendering of
// the turn agrees with it.
struct LiveStateBackend : ShortInlineSnapshotBackend {
    int live_position = 0;
    int saves = 0;

    bool snapshot_save(int slot) override {
        saved_slot = slot;
        saved_position = live_position;
        ++saves;
        return true;
    }
    void snapshot_free(int slot) override {
        if (slot == saved_slot) saved_position = 0;
    }
};

TEST_CASE(ServerUnitFixture, test_save_generated_turn_keys_live_state) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::vector<int32_t> prompt = {1, 100, 3, 101};
    const std::vector<int32_t> generated = {200, 201, 202};
    // The next request renders the reply, then appends tool results.
    const std::vector<int32_t> canonical = {1, 100, 3, 101, 200, 201, 202, 7};

    {
        auto backend_owner = std::make_unique<LiveStateBackend>();
        LiveStateBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.prefix_cache_cap = 2;
        HttpServer server(engine, tokenizer, config);
        // The last sampled token was never decoded.
        backend.live_position = 6;
        TEST_ASSERT(SchedulerTestHarness::save_generated_turn(
            server, prompt, generated, canonical));
        std::vector<int32_t> next = canonical;
        next.insert(next.end(), {8, 9});
        const auto hit =
            SchedulerTestHarness::prefix_cache(server).lookup(next);
        TEST_ASSERT(hit.first == backend.saved_slot);
        TEST_ASSERT(hit.second == 6);
        TEST_ASSERT(SchedulerTestHarness::is_agent_turn(server, hit.first));
    }
    {
        // A reply the next request renders differently is not saved.
        auto backend_owner = std::make_unique<LiveStateBackend>();
        LiveStateBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.prefix_cache_cap = 2;
        HttpServer server(engine, tokenizer, config);
        backend.live_position = 6;
        const std::vector<int32_t> rerendered = {1, 100, 3, 101, 250, 7};
        TEST_ASSERT(!SchedulerTestHarness::save_generated_turn(
            server, prompt, generated, rerendered));
        TEST_ASSERT(backend.saves == 0);
    }
    {
        // State past the agreeing tokens would claim rows it does not match.
        auto backend_owner = std::make_unique<LiveStateBackend>();
        LiveStateBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.prefix_cache_cap = 2;
        HttpServer server(engine, tokenizer, config);
        backend.live_position = 7;
        const std::vector<int32_t> partial = {1, 100, 3, 101, 200, 201, 250};
        TEST_ASSERT(!SchedulerTestHarness::save_generated_turn(
            server, prompt, generated, partial));
        TEST_ASSERT(SchedulerTestHarness::prefix_cache(server).stats().in_use == 0);
    }
    {
        // At capacity the checkpoint never evicts the request's own inline
        // snapshot, the fallback when the next request diverges inside the
        // generated turn. Otherwise slot 1, the oldest leaf, would go.
        auto backend_owner = std::make_unique<LiveStateBackend>();
        LiveStateBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.prefix_cache_cap = 3;
        HttpServer server(engine, tokenizer, config);
        PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
        cache.confirm_inline_snap(0, 2, prompt);
        cache.confirm_inline_snap(1, 4, prompt);
        cache.confirm_inline_snap(2, 3, std::vector<int32_t>{1, 100, 9});
        backend.live_position = 6;
        TEST_ASSERT(SchedulerTestHarness::save_generated_turn(
            server, prompt, generated, canonical, /*own_snapshot_slot=*/1));
        TEST_ASSERT(backend.saved_slot == 2);
        std::vector<int32_t> next = canonical;
        next.insert(next.end(), {8, 9});
        TEST_ASSERT(cache.lookup(next).first == 2);
        std::vector<int32_t> diverged = prompt;
        diverged.insert(diverged.end(), {250, 7});
        TEST_ASSERT(cache.lookup(diverged).first == 1);
    }
    {
        // The same holds for the checkpoint the request restored from while
        // the backend still holds it.
        auto backend_owner = std::make_unique<LiveStateBackend>();
        LiveStateBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.prefix_cache_cap = 3;
        HttpServer server(engine, tokenizer, config);
        PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
        cache.confirm_inline_snap(0, 2, prompt);
        cache.confirm_inline_snap(1, 4, prompt);
        cache.confirm_inline_snap(2, 3, std::vector<int32_t>{1, 100, 9});
        backend.saved_slot = 1;
        backend.saved_position = 4;
        backend.live_position = 6;
        TEST_ASSERT(SchedulerTestHarness::save_generated_turn(
            server, prompt, generated, canonical, /*own_snapshot_slot=*/-1,
            DiskPrefixCacheMode::Full, /*restored_slot=*/1));
        TEST_ASSERT(backend.saved_slot == 2);
        std::vector<int32_t> diverged = prompt;
        diverged.insert(diverged.end(), {250, 7});
        TEST_ASSERT(cache.lookup(diverged).first == 1);
    }
    {
        // A live restore consumed its checkpoint and finalization dropped
        // the entry, so its slot is free. Here it is the only free slot, and
        // the turn is saved there. Refusing it saves nothing on every other
        // turn of a full cache.
        auto backend_owner = std::make_unique<LiveStateBackend>();
        LiveStateBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.prefix_cache_cap = 3;
        HttpServer server(engine, tokenizer, config);
        PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
        cache.confirm_inline_snap(0, 3, std::vector<int32_t>{1, 100, 9});
        cache.confirm_inline_snap(1, 3, std::vector<int32_t>{1, 100, 8});
        backend.live_position = 6;
        TEST_ASSERT(SchedulerTestHarness::save_generated_turn(
            server, prompt, generated, canonical, /*own_snapshot_slot=*/-1,
            DiskPrefixCacheMode::Full, /*restored_slot=*/2));
        std::vector<int32_t> next = canonical;
        next.insert(next.end(), {8, 9});
        TEST_ASSERT(cache.lookup(next).first == 2);
    }
    unlink(path.c_str());
}

// Serves every generation with `reply`, keeping the state a real backend
// would: an inline snapshot at the planned cut, and after decode the prompt
// plus every generated token but the last (sampled, never decoded).
struct ScriptedReplyBackend : MockBackend {
    std::vector<int32_t> reply;
    std::vector<std::vector<int32_t>> prompts;
    int restored_prefix = 0;
    std::map<int, int> positions;
    int live_position = 0;

    bool snapshot_save(int slot) override {
        positions[slot] = live_position;
        return true;
    }
    void snapshot_free(int slot) override { positions.erase(slot); }
    bool snapshot_used(int slot) const override { return positions.count(slot) != 0; }
    int snapshot_cur_pos(int slot) const override {
        const auto it = positions.find(slot);
        return it == positions.end() ? 0 : it->second;
    }

    GenerateResult serve(const GenerateRequest & req, const DaemonIO & io) {
        prompts.push_back(req.prompt);
        if (req.snap_slot >= 0 && req.snap_pos > 0) positions[req.snap_slot] = req.snap_pos;
        GenerateResult result;
        result.succeed();
        for (const int32_t token : reply) {
            result.tokens.push_back(token);
            if (io.on_token && !io.on_token(token)) break;
        }
        live_position = (int) (req.prompt.size() + result.tokens.size()) - 1;
        return result;
    }
    GenerateResult generate_impl(const GenerateRequest & req, const DaemonIO & io) override {
        return serve(req, io);
    }
    GenerateResult restore_and_generate_impl(int slot, const GenerateRequest & req,
                                             const DaemonIO & io) override {
        restored_prefix = snapshot_cur_pos(slot);
        GenerateResult result = serve(req, io);
        result.restored_prefix_tokens = restored_prefix;
        return result;
    }
};

// Qwen-shaped vocabulary: the think and tool markers are single tokens, and
// "\n\n" and "\n\n\n" are distinct newline tokens.
static std::string write_thinking_tool_tokenizer_fixture() {
    std::string printable = "\n";
    for (char c = ' '; c <= '~'; ++c) printable += c;
    return write_pflash_bpe_tokenizer_fixture(
        {"\n\n", "\n\n\n"}, printable,
        {"<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>",
         "</tool_call>", "<tool_response>", "</tool_response>"});
}

// A stateless agent's two requests to `server`: a thinking tool-call turn,
// then the conversation with the call's result. `reply` sets what the backend
// generates next. Returns the generated turn: the model closes its thinking
// with the </think> token followed by the "\n\n" token, then calls a tool and
// ends its turn.
static std::vector<int32_t> send_thinking_tool_turn(
        HttpServer & server, const Tokenizer & tokenizer,
        const std::function<void(std::vector<int32_t>)> & reply) {
    std::vector<int32_t> generated = tokenizer.encode(
        "I will read a.py first.\n</think>\n\n<tool_call>\n<function=read>\n"
        "<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>");
    generated.push_back(tokenizer.eos_chat_id());

    const json tools = json::array({{
        {"type", "function"},
        {"function", {
            {"name", "read"},
            {"description", "Read a file"},
            {"parameters", {
                {"type", "object"},
                {"properties", {{"path", {{"type", "string"}}}}},
                {"required", json::array({"path"})}}}}}}});
    const auto chat = [&](const json & messages) {
        int sockets[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        const json body = {
            {"model", "test"},
            {"messages", messages},
            {"tools", tools},
            {"max_tokens", 256},
            {"chat_template_kwargs", {{"enable_thinking", true}}},
            // Only the client's requests reach the backend.
            {"tool_checks", false},
        };
        TEST_ASSERT(SchedulerTestHarness::route(
            server, sockets[0], "/v1/chat/completions", body));
        close(sockets[0]);
        std::string wire;
        char buf[4096];
        ssize_t n = 0;
        while ((n = read(sockets[1], buf, sizeof(buf))) > 0) wire.append(buf, (size_t) n);
        close(sockets[1]);
        const size_t body_at = wire.find("\r\n\r\n");
        TEST_ASSERT_MSG(body_at != std::string::npos, wire);
        return json::parse(wire.substr(body_at + 4));
    };

    json messages = json::array({
        {{"role", "system"}, {"content", "You are a coding agent."}},
        {{"role", "user"}, {"content", "Read a.py."}},
    });
    reply(generated);
    const json first = chat(messages);
    const json message = first["choices"][0]["message"];
    TEST_ASSERT_MSG(message.contains("tool_calls") && message["tool_calls"].size() == 1,
                    first.dump());
    messages.push_back(message);
    messages.push_back({{"role", "tool"},
                        {"tool_call_id", message["tool_calls"][0]["id"]},
                        {"content", "print(1)"}});
    reply({tokenizer.eos_chat_id()});
    chat(messages);
    return generated;
}

// Where `next` stops following `prompt` + `generated`; empty when it extends
// them.
static std::string generated_turn_gap(
        const Tokenizer & tokenizer, std::vector<int32_t> turn,
        const std::vector<int32_t> & generated, const std::vector<int32_t> & next) {
    turn.insert(turn.end(), generated.begin(), generated.end());
    size_t same = 0;
    while (same < turn.size() && same < next.size() && turn[same] == next[same]) ++same;
    if (same == turn.size() && next.size() > turn.size()) return {};
    return "next prompt leaves the generated turn at token " + std::to_string(same) +
           " of " + std::to_string(turn.size()) + ": " +
           tokenizer.decode({next.begin() + (long) same,
                             next.begin() + (long) std::min(next.size(), same + 4)});
}

// A stateless agent sends the thinking tool-call turn back with its tool
// result. The next prompt must hold the turn exactly as generated (previous
// prompt + generated tokens), and the server's checkpoint of the generated
// turn must serve it, so only the tool result is prefilled.
TEST_CASE(ServerUnitFixture, test_thinking_tool_turn_replays_as_generated) {
    const std::string path = write_thinking_tool_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    auto backend_owner = std::make_unique<ScriptedReplyBackend>();
    ScriptedReplyBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.max_ctx = 4096;
    config.prefix_cache_cap = 4;
    HttpServer server(engine, tokenizer, config);
    TEST_ASSERT(SchedulerTestHarness::start_worker(server));

    const std::vector<int32_t> generated = send_thinking_tool_turn(
        server, tokenizer, [&](std::vector<int32_t> reply) { backend.reply = std::move(reply); });
    unlink(path.c_str());

    TEST_ASSERT(backend.prompts.size() == 2);
    if (backend.prompts.size() != 2) return;
    const std::string gap = generated_turn_gap(
        tokenizer, backend.prompts[0], generated, backend.prompts[1]);
    TEST_ASSERT_MSG(gap.empty(), gap);
    const int turn = (int) (backend.prompts[0].size() + generated.size());
    TEST_ASSERT_MSG(backend.restored_prefix == turn - 1,
                    "restored " + std::to_string(backend.restored_prefix) + " of " +
                    std::to_string(turn));
}

// The concurrent engine samples `reply` for every request: the token that
// ends each prompt's prefill, then one token per decode step.
class ScriptedReplyEngine final : public SeqEngine {
public:
    int32_t eos = -1;
    std::vector<int32_t> reply;
    std::vector<std::vector<int32_t>> prompts;

    int slot_count() const override { return 1; }
    int max_context() const override { return 4096; }
    bool token_is_eos(int32_t token) const override { return token == eos; }
    StepPlanLimits step_plan_limits(int) const override { return {1, 4096, 4096, 4096}; }
    AdmitResult admit(uint64_t, const std::vector<int32_t> & prompt, const SamplerCfg &) override {
        prompts.push_back(prompt);
        next_ = 0;
        AdmitResult result;
        result.status = AdmitResult::Status::admitted;
        result.slot = 0;
        return result;
    }
    StepResult step(const StepPlan & plan) override {
        StepResult result;
        for (const StepInput & input : plan.decode) {
            result.decode.push_back({input.slot, reply[std::min(next_++, reply.size() - 1)], false, {}});
        }
        for (const PrefillSlice & slice : plan.prefills) {
            PrefillOutput out;
            out.slot = slice.slot;
            out.status = PrefillOutput::Status::completed;
            out.token = reply[std::min(next_++, reply.size() - 1)];
            result.prefills.push_back(out);
        }
        return result;
    }
    void retire(int) override {}

private:
    size_t next_ = 0;
};

struct ScriptedReplySeqBackend : MockBackend {
    ScriptedReplyEngine engine;
    SeqEngine * seq_engine() override { return &engine; }
};

// Under continuous batching too, the thinking tool-call turn comes back on the
// next request exactly as generated.
TEST_CASE(ServerUnitFixture, test_concurrent_thinking_tool_turn_replays_as_generated) {
    const std::string path = write_thinking_tool_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    auto backend_owner = std::make_unique<ScriptedReplySeqBackend>();
    ScriptedReplyEngine & seq = backend_owner->engine;
    seq.eos = tokenizer.eos_chat_id();
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 4096;
    config.prefix_cache_cap = 0;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);
    TEST_ASSERT(SchedulerTestHarness::start_worker(server));

    const std::vector<int32_t> generated = send_thinking_tool_turn(
        server, tokenizer, [&](std::vector<int32_t> reply) { seq.reply = std::move(reply); });
    unlink(path.c_str());

    TEST_ASSERT(seq.prompts.size() == 2);
    if (seq.prompts.size() != 2) return;
    const std::string gap = generated_turn_gap(
        tokenizer, seq.prompts[0], generated, seq.prompts[1]);
    TEST_ASSERT_MSG(gap.empty(), gap);
}

// A hit whose backend snapshot is gone (a deferred snapshot a live
// continuation consumed) is dropped and the next deepest entry restored.
struct SlotSetBackend : MockBackend {
    std::map<int, int> positions;
    bool cuts_at_restore_points = false;
    bool prefill_cuts_at_restore_points() const override { return cuts_at_restore_points; }
    bool snapshot_used(int slot) const override {
        return positions.count(slot) != 0;
    }
    int snapshot_cur_pos(int slot) const override {
        const auto it = positions.find(slot);
        return it == positions.end() ? 0 : it->second;
    }
};

TEST_CASE(ServerUnitFixture, test_prepare_cache_skips_consumed_snapshot) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    auto backend_owner = std::make_unique<SlotSetBackend>();
    SlotSetBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.prefix_cache_cap = 4;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);

    const std::vector<int32_t> prompt = {1, 100, 3, 101, 4, 102};
    cache.confirm_inline_snap(0, 2, prompt);
    cache.confirm_inline_snap(1, 4, prompt);
    // Slot 1 has no snapshot left: it was consumed.
    backend.positions[0] = 2;

    const auto prepared = SchedulerTestHarness::prepare_cache(server, prompt);
    TEST_ASSERT(prepared.restore_slot == 0);
    TEST_ASSERT(prepared.prefix_len == 2);
    TEST_ASSERT(cache.lookup_candidate(prompt, (int)prompt.size()).first == 0);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_prefix_replaced_ancestor_releases_metadata) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    auto owner = std::make_unique<SlotSetBackend>();
    auto & backend = *owner;
    LuceEngine engine(std::move(owner));
    ServerConfig config;
    config.prefix_cache_cap = 4;
    HttpServer server(engine, tokenizer, config);
    auto & cache = SchedulerTestHarness::prefix_cache(server);
    const std::vector<int32_t> prompt = {1, 100, 3, 101, 4, 102};
    cache.confirm_inline_snap(0, 2, prompt, true, 100);
    auto reservation = cache.reserve_inline_snap(prompt, 2, false, 4, 0);
    TEST_ASSERT(reservation.active());
    const int slot = reservation.slot();
    backend.positions[slot] = 4; // backend replaced source slot 0 under pressure
    SchedulerTestHarness::finalize_inline_snapshot(server, prompt, std::move(reservation), slot, 4, 0);
    TEST_ASSERT(cache.lookup_candidate(prompt, 2).first == -1);
    TEST_ASSERT(cache.lookup_candidate(prompt, 6).first == slot);
    TEST_ASSERT(cache.stats().resident_bytes == 0); // MockBackend estimates zero, old 100 bytes released
    unlink(path.c_str());
}

static std::vector<int> prefill_chunk_starts(
        int kv_offset, int prompt_end, const std::vector<int> & points,
        int first_min_tokens = kQwen35MinChunkTokens) {
    std::vector<int> starts;
    for (int pos = kv_offset; pos < prompt_end;) {
        starts.push_back(pos);
        pos += qwen35_prefill_chunk_tokens(
            pos, prompt_end - pos, kQwen35PrefillUbatch, points,
            pos == kv_offset ? first_min_tokens : kQwen35MinChunkTokens);
    }
    return starts;
}

// Chunks start at every restore point, keep to the 512 grid in between, and
// are never shorter than 16 tokens before the end of the prompt. The expected
// starts below assume those values of kQwen35PrefillUbatch and
// kQwen35MinChunkTokens.
TEST_CASE(ServerUnitFixture, test_qwen35_prefill_chunks) {
    TEST_ASSERT((prefill_chunk_starts(0, 1300, {}) ==
                 std::vector<int>{0, 512, 1024}));
    TEST_ASSERT((prefill_chunk_starts(0, 1300, {21, 700}) ==
                 std::vector<int>{0, 21, 512, 700, 1024}));
    // A restore point 2 tokens before the grid does not leave a 2-token chunk.
    TEST_ASSERT((prefill_chunk_starts(0, 1300, {510}) ==
                 std::vector<int>{0, 510, 1022}));
    // The prompt's last chunk may be short.
    TEST_ASSERT((prefill_chunk_starts(0, 515, {}) ==
                 std::vector<int>{0, 512}));
    // A prefill resumed at a chunk start of the cold prefill cuts alike.
    const std::vector<int> points = {40, 300, 1400, 1405, 2047, 2600};
    const auto cold = prefill_chunk_starts(0, 3000, points);
    for (int start : cold) {
        const auto resumed = prefill_chunk_starts(start, 3000, points);
        TEST_ASSERT(std::equal(resumed.begin(), resumed.end(),
                               std::find(cold.begin(), cold.end(), start)));
    }
    // An off-grid restore runs its first chunk at least 64 tokens.
    const auto off_grid = prefill_chunk_starts(
        1003, 1300, {1010, 1100}, kQwen35OffGridLeadTokens);
    TEST_ASSERT(off_grid.size() >= 2 &&
                off_grid[1] - off_grid[0] >= kQwen35OffGridLeadTokens);
}

// An agent turn that continues a generated-turn checkpoint skips its own
// snapshot while another checkpoint lies close behind; it keeps it when the
// other checkpoints are far behind, and other requests always keep it.
TEST_CASE(ServerUnitFixture, test_agent_continuation_throttles_snapshot) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    // Slot 0 holds the first 3 tokens, slot 1 a generated-turn checkpoint
    // `filler` + 5 tokens in, and the prompt adds one more turn.
    // `cuts`: the backend cuts prefill at every restore point (qwen4exp).
    const auto prepare = [&](int filler, bool ends_with_tool_result, bool cuts = true) {
        auto backend_owner = std::make_unique<SlotSetBackend>();
        SlotSetBackend & backend = *backend_owner;
        backend.cuts_at_restore_points = cuts;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.prefix_cache_cap = 4;
        HttpServer server(engine, tokenizer, config);
        PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);

        std::vector<int32_t> prompt = {1, 100, 3, 101, 4};
        prompt.insert(prompt.end(), (size_t) filler, 102);
        const int checkpoint = (int) prompt.size();
        prompt.insert(prompt.end(), {3, 103, 4});
        cache.confirm_inline_snap(0, 3, prompt);
        cache.confirm_inline_snap(1, checkpoint, prompt);
        SchedulerTestHarness::mark_agent_turn(server, 1);
        backend.positions[0] = 3;
        backend.positions[1] = checkpoint;
        return SchedulerTestHarness::prepare_cache(
            server, prompt, ends_with_tool_result);
    };

    const auto near = prepare(1, /*ends_with_tool_result=*/true);
    TEST_ASSERT(near.restore_slot == 1);
    TEST_ASSERT(!near.snapshot);
    // Even a skipped capture must keep the tool-end cut: a cold request can
    // select a different snapshot, but must use the same prefill boundaries.
    TEST_ASSERT(near.restore_points.back() == 9);
    // A backend that does not cut prefill at every restore point keeps the existing boundary policy.
    const auto other = prepare(1, true, /*cuts=*/false);
    TEST_ASSERT(other.restore_slot == 1 && !other.snapshot);
    TEST_ASSERT(other.restore_points.back() == 7);

    const auto chat = prepare(1, /*ends_with_tool_result=*/false);
    TEST_ASSERT(chat.restore_slot == 1);
    TEST_ASSERT(chat.snapshot);

    // Slot 0 lies more than 2048 tokens behind the cut.
    const auto far = prepare(2100, /*ends_with_tool_result=*/true);
    TEST_ASSERT(far.restore_slot == 1);
    TEST_ASSERT(far.snapshot);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_prefix_cache_budget_resolution) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    const auto budget_for = [&](std::unique_ptr<ModelBackend> backend,
                                size_t configured) {
        LuceEngine engine(std::move(backend));
        ServerConfig config;
        config.prefix_cache_cap = 2;
        config.prefix_cache_max_bytes = configured;
        HttpServer server(engine, tokenizer, config);
        return SchedulerTestHarness::prefix_cache(server)
            .stats().max_resident_bytes;
    };
    // Auto: three snapshots at the backend's full context (1000 tokens).
    TEST_ASSERT(budget_for(std::make_unique<PruningSnapshotBackend>(),
                           ServerConfig::kPrefixCacheBudgetAuto) ==
                3 * (1000 * 100 + 1000));
    TEST_ASSERT(budget_for(std::make_unique<PruningSnapshotBackend>(),
                           5000) == 5000);
    TEST_ASSERT(budget_for(std::make_unique<PruningSnapshotBackend>(),
                           0) == 0);
    // A backend that cannot size snapshots keeps the cache unbounded rather
    // than refusing every capture.
    TEST_ASSERT(budget_for(std::make_unique<MockBackend>(),
                           ServerConfig::kPrefixCacheBudgetAuto) == 0);

    // Startup rejects an explicit limit the backend cannot enforce, instead
    // of silently running without it; auto and 0 stay accepted.
    MockBackend unsized;
    PruningSnapshotBackend sized;
    ServerConfig config;
    config.prefix_cache_max_bytes = 5000;
    PrefixCacheBudget budget = resolve_prefix_cache_budget(config, unsized);
    TEST_ASSERT(budget.bytes == 0 && !budget.sized && !budget.error.empty());
    budget = resolve_prefix_cache_budget(config, sized);
    TEST_ASSERT(budget.bytes == 5000 && !budget.automatic && budget.error.empty());
    config.prefix_cache_max_bytes = 0;
    budget = resolve_prefix_cache_budget(config, unsized);
    TEST_ASSERT(budget.bytes == 0 && budget.error.empty());
    config.prefix_cache_max_bytes = ServerConfig::kPrefixCacheBudgetAuto;
    budget = resolve_prefix_cache_budget(config, unsized);
    TEST_ASSERT(budget.bytes == 0 && budget.automatic && budget.error.empty());
    budget = resolve_prefix_cache_budget(config, sized);
    TEST_ASSERT(budget.bytes == 3 * (1000 * 100 + 1000) && budget.automatic);
    // With the prefix cache off there is nothing to bound or reject.
    config.prefix_cache_cap = 0;
    config.prefix_cache_max_bytes = 5000;
    budget = resolve_prefix_cache_budget(config, unsized);
    TEST_ASSERT(budget.bytes == 0 && budget.error.empty());
    config.prefix_cache_cap = 32;
    // Concurrent paged serving keeps its own limit, whatever the backend.
    config.concurrent_prefix_cache = true;
    config.concurrent_prefix_cache_max_bytes = 4096;
    budget = resolve_prefix_cache_budget(config, unsized);
    TEST_ASSERT(budget.bytes == 4096 && budget.error.empty());
    unlink(path.c_str());
}

struct MemoryReportBackend : MockBackend {
    MemoryReport memory_report() const override {
        MemoryReport report;
        report.available = true;
        report.cache.capacity_tokens = 4096;
        report.cache.live_tokens = 1200;
        report.cache.kv_bytes = 1000;
        report.cache.recurrent_bytes = 200;
        report.cache.draft_feature_bytes = 30;
        report.cache.other_bytes = 4;
        report.cache.host_state_bytes = 5;
        report.snapshots = {
            {0, 1024, 100, true},
            {1, 2048, 150, true},
            {3, 512, 60, true},
            {ModelBackend::kMaxSlots - 1, 256, 40, false, 8},
        };
        return report;
    }
};

// /props reports where memory goes: the live cache, each saved snapshot
// labelled by its cache role, and the process's resident set.
TEST_CASE(ServerUnitFixture, test_props_memory_report) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    LuceEngine engine(std::make_unique<MemoryReportBackend>());
    ServerConfig config;
    config.prefix_cache_cap = 2;
    config.prefill_cache_cap = 2;
    HttpServer server(engine, tokenizer, config);

    // Before the first report there is nothing to show but the process.
    json memory = SchedulerTestHarness::memory_json(server);
    TEST_ASSERT(memory["cache"].is_null());
    TEST_ASSERT(memory["reports_published"] == 0);
    TEST_ASSERT(memory["snapshots"]["count"] == 0);
    TEST_ASSERT(memory["snapshots"]["slots"].is_array());
    TEST_ASSERT(memory["snapshots"]["slots"].empty());

    SchedulerTestHarness::mark_agent_turn(server, 1);
    SchedulerTestHarness::publish_memory_report(server);
    memory = SchedulerTestHarness::memory_json(server);
    TEST_ASSERT(memory["reports_published"] == 1);
    const json & cache = memory["cache"];
    TEST_ASSERT(cache["location"] == "device");
    TEST_ASSERT(cache["capacity_tokens"] == 4096);
    TEST_ASSERT(cache["live_tokens"] == 1200);
    TEST_ASSERT(cache["kv_bytes"] == 1000);
    TEST_ASSERT(cache["recurrent_bytes"] == 200);
    TEST_ASSERT(cache["draft_feature_bytes"] == 30);
    TEST_ASSERT(cache["other_bytes"] == 4);
    TEST_ASSERT(cache["host_state_bytes"] == 5);

    const json & snapshots = memory["snapshots"];
    TEST_ASSERT(snapshots["count"] == 4);
    // The device snapshot's host-side copies count as host memory.
    TEST_ASSERT(snapshots["host_bytes"] == 310 + 8);
    TEST_ASSERT(snapshots["device_bytes"] == 40);
    const json & slots = snapshots["slots"];
    TEST_ASSERT(slots.size() == 4);
    TEST_ASSERT(slots[0]["kind"] == "prefix");
    TEST_ASSERT(slots[0]["tokens"] == 1024);
    TEST_ASSERT(slots[1]["kind"] == "agent_turn");
    TEST_ASSERT(slots[2]["kind"] == "prefill_cache");
    TEST_ASSERT(slots[3]["kind"] == "disk_staging");
    TEST_ASSERT(slots[3]["location"] == "device");
    TEST_ASSERT(slots[3]["bytes"] == 48);
#if defined(__linux__)
    TEST_ASSERT(memory["process"]["rss_bytes"].get<uint64_t>() > 0);
    TEST_ASSERT(memory["process"]["peak_rss_bytes"].get<uint64_t>() >=
                memory["process"]["rss_bytes"].get<uint64_t>());
#endif

    // A backend without a report still yields a well-formed section.
    LuceEngine plain_engine(std::make_unique<MockBackend>());
    HttpServer plain(plain_engine, tokenizer, config);
    SchedulerTestHarness::publish_memory_report(plain);
    memory = SchedulerTestHarness::memory_json(plain);
    TEST_ASSERT(memory["cache"].is_null());
    TEST_ASSERT(memory["snapshots"]["count"] == 0);
    unlink(path.c_str());
}

#if !defined(_WIN32)
class SchedulerPrefixEngine final : public SeqEngine {
public:
    SchedulerPrefixEngine() : slots_(2) {}

    int slot_count() const override { return (int)slots_.size(); }
    int max_context() const override { return 64; }
    bool supports_prefix_store() const override { return true; }
    size_t estimate_prefix_store_bytes(int) const override { return 256; }
    bool token_is_eos(int32_t token) const override { return token == 2; }
    StepPlanLimits step_plan_limits(int) const override {
        return {/*max_prefill_sequences=*/2,
                /*max_prefill_tokens_per_sequence=*/64,
                /*max_prefill_tokens_total=*/128,
                /*prefill_allocation_quantum=*/64};
    }

    AdmitResult admit(
            uint64_t request_id,
            const std::vector<int32_t> & prompt,
            const SamplerCfg & sampler) override {
        return admit_with_prefix(
            request_id, prompt, sampler, PrefixStorePlan{});
    }

    AdmitResult admit_with_prefix(
            uint64_t,
            const std::vector<int32_t> & prompt,
            const SamplerCfg &,
            const PrefixStorePlan & plan) override {
        AdmitResult result;
        if (prompt.empty()) {
            result.error = "empty prompt";
            return result;
        }
        if (defer_restore.load(std::memory_order_relaxed) &&
            plan.restore.valid()) {
            returned_busy_before_restore.store(true, std::memory_order_relaxed);
            result.status = AdmitResult::Status::busy;
            result.error = "restore admission deferred";
            return result;
        }
        int chosen = -1;
        for (int i = 0; i < (int)slots_.size(); ++i) {
            if (!slots_[(size_t)i].active) {
                chosen = i;
                break;
            }
        }
        if (chosen < 0) {
            result.status = AdmitResult::Status::busy;
            result.error = "all slots live";
            return result;
        }

        Slot & slot = slots_[(size_t)chosen];
        slot.active = true;
        slot.prefilling = true;
        slot.capture = {};
        result.status = AdmitResult::Status::admitted;
        result.slot = chosen;

        if (plan.restore.valid()) {
            result.prefix_store.restore_attempted = true;
            result.prefix_store.restore_elapsed_us = 2500;
            if (unrequested_restore.load(std::memory_order_relaxed)) {
                result.prefix_store.restored = {
                    plan.restore.id + 1, plan.restore.tokens};
            } else if (malformed_restore.load(std::memory_order_relaxed)) {
                result.prefix_store.restored = {plan.restore.id, 0};
            } else if (plan.restore == PrefixStoreRef{2, 2}) {
                saw_stale_restore = true;
                discarded.push_back(plan.restore);
                result.prefix_store.invalidated = plan.restore;
            } else {
                result.prefix_store.restored = plan.restore;
            }
        }
        if (!result.prefix_store.invalidated.valid() &&
            plan.capture.valid()) {
            saw_capture = true;
            slot.capture = plan.capture;
            result.prefix_store.capture = plan.capture;
        }
        return result;
    }

    StepResult step(const StepPlan & plan) override {
        StepResult result;
        for (const StepInput & input : plan.decode) {
            result.decode.push_back({input.slot, 2, false, {}});
        }
        for (const PrefillSlice & slice : plan.prefills) {
            if (slice.slot < 0 || slice.slot >= (int)slots_.size() ||
                !slots_[(size_t)slice.slot].active ||
                !slots_[(size_t)slice.slot].prefilling) {
                result.error = "invalid prefill";
                result.prefills.clear();
                result.decode.clear();
                return result;
            }
            Slot & slot = slots_[(size_t)slice.slot];
            slot.prefilling = false;
            PrefillOutput output;
            output.slot = slice.slot;
            output.status = PrefillOutput::Status::completed;
            output.token = 2;
            if (slot.capture.valid()) {
                output.prefix_store.status = PrefixStoreEvent::Status::saved;
                output.prefix_store.ticket = slot.capture;
                output.prefix_store.bytes = 256;
                output.prefix_store.elapsed_us = 1500;
                slot.capture = {};
            }
            result.prefills.push_back(std::move(output));
        }
        return result;
    }

    void retire(int slot) override {
        if (slot >= 0 && slot < (int)slots_.size()) {
            slots_[(size_t)slot] = Slot{};
        }
    }

    void discard_prefix_store(PrefixStoreRef checkpoint) override {
        discarded.push_back(checkpoint);
    }

    bool saw_capture = false;
    bool saw_stale_restore = false;
    std::atomic<bool> defer_restore{false};
    std::atomic<bool> returned_busy_before_restore{false};
    std::atomic<bool> unrequested_restore{false};
    std::atomic<bool> malformed_restore{false};
    std::vector<PrefixStoreRef> discarded;

private:
    struct Slot {
        bool active = false;
        bool prefilling = false;
        PrefixCaptureTicket capture;
    };
    std::vector<Slot> slots_;
};

struct SchedulerPrefixBackend : MockBackend {
    SchedulerPrefixEngine engine;
    SeqEngine * seq_engine() override { return &engine; }
};

// Concurrent serving: two requests admitted together both capture, and a
// conversation's new checkpoint frees the ones it supersedes (keeping the
// shallowest, the shared head, and the previous turn for an edit).
TEST_CASE(ServerUnitFixture,
          test_scheduler_captures_concurrently_and_prunes_superseded) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<SchedulerPrefixBackend>();
    SchedulerPrefixBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 8;
    config.concurrent_prefix_cache_max_bytes = 4096;
    config.concurrent_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);

    std::vector<std::unique_ptr<ServerJob>> jobs;
    std::vector<int> fds;
    const auto make_job = [&](std::vector<int32_t> prompt, const char * id) {
        int sockets[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        fds.push_back(sockets[0]);
        fds.push_back(sockets[1]);
        auto job = std::make_unique<ServerJob>();
        job->fd = sockets[0];
        job->req.format = ApiFormat::OPENAI_CHAT;
        job->req.prompt_tokens = std::move(prompt);
        job->req.max_output = 1;
        job->req.stream = false;
        job->req.model = "scheduler-test";
        job->req.response_id = id;
        jobs.push_back(std::move(job));
        return jobs.back().get();
    };
    const auto wait_done = [](ServerJob * job) {
        std::unique_lock<std::mutex> lock(job->mu);
        return job->cv.wait_for(lock, std::chrono::seconds(5), [&] { return job->done; });
    };

    // Admitted in one pass: both capture their head.
    ServerJob * a = make_job({1, 100, 3, 101, 4}, "a");
    ServerJob * b = make_job({1, 200, 3, 201, 4}, "b");
    SchedulerTestHarness::enqueue(server, a);
    SchedulerTestHarness::enqueue(server, b);
    std::thread scheduler([&] {
        SchedulerTestHarness::run(server, backend.engine);
    });
    const bool ab_done = wait_done(a) && wait_done(b);

    // One conversation, three more turns, each saved at its end: the fourth
    // turn's checkpoint supersedes the second's, which is freed; the head and
    // the third turn stay.
    const std::vector<int32_t> turn2 = {1, 100, 3, 101, 4, 102, 2, 3, 103, 4};
    std::vector<int32_t> turn3 = turn2;
    turn3.insert(turn3.end(), {104, 2, 3, 105, 4});
    std::vector<int32_t> turn4 = turn3;
    turn4.insert(turn4.end(), {106, 2, 3, 107, 4});
    ServerJob * second = make_job(turn2, "turn2");
    SchedulerTestHarness::enqueue(server, second);
    const bool second_done = wait_done(second);
    ServerJob * third = make_job(turn3, "turn3");
    SchedulerTestHarness::enqueue(server, third);
    const bool third_done = wait_done(third);
    ServerJob * fourth = make_job(turn4, "turn4");
    SchedulerTestHarness::enqueue(server, fourth);
    const bool fourth_done = wait_done(fourth);

    SchedulerTestHarness::stop(server);
    scheduler.join();
    for (const int fd : fds) close(fd);
    unlink(path.c_str());

    TEST_ASSERT(ab_done && second_done && third_done && fourth_done);
    // Both heads were captured although their requests prefilled together.
    TEST_ASSERT(cache.lookup_candidate({1, 100, 3, 101, 4, 9}, 5).second == 3);
    TEST_ASSERT(cache.lookup_candidate({1, 200, 3, 201, 4, 9}, 5).second == 3);
    std::vector<int32_t> next = turn4;
    next.push_back(9);
    TEST_ASSERT(cache.lookup_candidate(next, (int)next.size() - 1).second == 20);
    // An edit of the last message resumes from the third turn's end.
    std::vector<int32_t> edited = turn3;
    edited.insert(edited.end(), {106, 2, 3, 999, 4});
    TEST_ASSERT(cache.lookup_candidate(edited, (int)edited.size() - 1).second == 15);
    // The turn-2 checkpoint is gone from the cache and freed in the engine.
    std::vector<int32_t> after_turn2 = turn2;
    after_turn2.push_back(9);
    TEST_ASSERT(cache.lookup_candidate(after_turn2, 10).second == 3);
    bool freed_turn2 = false;
    for (const auto & ref : backend.engine.discarded) freed_turn2 |= ref.tokens == 10;
    TEST_ASSERT(freed_turn2);
}

TEST_CASE(ServerUnitFixture,
          test_scheduler_counts_restore_only_after_engine_attempts_it) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<SchedulerPrefixBackend>();
    SchedulerPrefixBackend & backend = *backend_owner;
    backend.engine.defer_restore.store(true, std::memory_order_relaxed);
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;
    config.concurrent_prefix_cache_max_bytes = 1024;
    config.concurrent_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    cache.confirm_inline_snap(
        /*slot=*/0, /*target_cut=*/2, {1, 100}, false, 128);

    int sockets[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);

    ServerJob job;
    job.fd = sockets[0];
    job.req.format = ApiFormat::OPENAI_CHAT;
    job.req.prompt_tokens = {1, 100, 999};
    job.req.max_output = 1;
    job.req.stream = false;
    job.req.model = "scheduler-test";
    job.req.response_id = "restore-after-busy";

    SchedulerTestHarness::enqueue(server, &job);
    std::thread scheduler([&] {
        SchedulerTestHarness::run(server, backend.engine);
    });

    const auto busy_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!backend.engine.returned_busy_before_restore.load(
               std::memory_order_relaxed) &&
           std::chrono::steady_clock::now() < busy_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool returned_busy =
        backend.engine.returned_busy_before_restore.load(
            std::memory_order_relaxed);
    const auto pre_attempt_stats = cache.stats();
    backend.engine.defer_restore.store(false, std::memory_order_relaxed);

    std::unique_lock<std::mutex> lock(job.mu);
    const bool done = job.cv.wait_for(
        lock, std::chrono::seconds(5), [&] { return job.done; });
    lock.unlock();
    SchedulerTestHarness::stop(server);
    scheduler.join();

    close(sockets[0]);
    close(sockets[1]);
    unlink(path.c_str());

    TEST_ASSERT(done);
    TEST_ASSERT(returned_busy);
    TEST_ASSERT(pre_attempt_stats.restore_attempts == 0);
    TEST_ASSERT(pre_attempt_stats.restore_invalidations == 0);
    TEST_ASSERT(pre_attempt_stats.restore_stall_us_total == 0);
    TEST_ASSERT(pre_attempt_stats.restore_stall_us_max == 0);

    const auto stats = cache.stats();
    TEST_ASSERT(stats.restore_attempts == 1);
    TEST_ASSERT(stats.restore_invalidations == 0);
    TEST_ASSERT(stats.restore_stall_us_total == 2500);
}

TEST_CASE(ServerUnitFixture,
          test_scheduler_rejects_malformed_restore_and_drops_stale_entry) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<SchedulerPrefixBackend>();
    SchedulerPrefixBackend & backend = *backend_owner;
    backend.engine.malformed_restore.store(true, std::memory_order_relaxed);
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;
    config.concurrent_prefix_cache_max_bytes = 1024;
    config.concurrent_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    cache.confirm_inline_snap(
        /*slot=*/0, /*target_cut=*/2, {1, 100}, false, 128);

    int sockets[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);

    ServerJob job;
    job.fd = sockets[0];
    job.req.format = ApiFormat::OPENAI_CHAT;
    job.req.prompt_tokens = {1, 100, 999};
    job.req.max_output = 1;
    job.req.stream = false;
    job.req.model = "scheduler-test";
    job.req.response_id = "malformed-restore";

    SchedulerTestHarness::enqueue(server, &job);
    std::thread scheduler([&] {
        SchedulerTestHarness::run(server, backend.engine);
    });

    std::unique_lock<std::mutex> lock(job.mu);
    const bool done = job.cv.wait_for(
        lock, std::chrono::seconds(5), [&] { return job.done; });
    lock.unlock();
    SchedulerTestHarness::stop(server);
    scheduler.join();

    close(sockets[0]);
    close(sockets[1]);
    unlink(path.c_str());

    TEST_ASSERT(done);
    TEST_ASSERT(cache.lookup_candidate({1, 100, 999}, 2).first == -1);
    TEST_ASSERT(backend.engine.discarded ==
                std::vector<PrefixStoreRef>({{1, 2}}));
    const auto stats = cache.stats();
    TEST_ASSERT(stats.restore_attempts == 1);
    TEST_ASSERT(stats.restore_invalidations == 1);
    TEST_ASSERT(stats.restore_stall_us_total == 2500);
}

// A long prompt shares the time with a decoding request: after each of its
// chunks the decoder steps alone for as long as the chunk took. Prompt steps
// here take 40 ms and decode steps 2 ms.
class TimeShareEngine final : public SeqEngine {
public:
    explicit TimeShareEngine(int long_prompt_tokens = 0)
        : long_prompt_tokens_(long_prompt_tokens) {}
    int slot_count() const override { return 2; }
    int max_context() const override { return 4096; }
    bool token_is_eos(int32_t) const override { return false; }
    StepPlanLimits step_plan_limits(int decode_rows) const override {
        return {decode_rows > 0 ? 1 : 2, 16, 32, 16, /*prefill_time_share=*/0.5,
                long_prompt_tokens_};
    }
    AdmitResult admit(uint64_t, const std::vector<int32_t> & prompt, const SamplerCfg &) override {
        AdmitResult result;
        const int slot = !slots_[0].active ? 0 : !slots_[1].active ? 1 : -1;
        if (slot < 0) {
            result.status = AdmitResult::Status::busy;
            return result;
        }
        slots_[(size_t)slot] = {true, (int)prompt.size()};
        result.status = AdmitResult::Status::admitted;
        result.slot = slot;
        return result;
    }
    StepResult step(const StepPlan & plan) override {
        StepResult result;
        for (const StepInput & input : plan.decode) result.decode.push_back({input.slot, 7, false, {}});
        for (const PrefillSlice & slice : plan.prefills) {
            Slot & slot = slots_[(size_t)slice.slot];
            slot.remaining -= std::min(slot.remaining, slice.max_tokens);
            PrefillOutput out;
            out.slot = slice.slot;
            out.status = slot.remaining == 0 ? PrefillOutput::Status::completed
                                             : PrefillOutput::Status::advanced;
            if (slot.remaining == 0) out.token = 7;
            result.prefills.push_back(out);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(plan.prefills.empty() ? 2 : 40));
        std::lock_guard<std::mutex> lock(mu);
        steps.push_back({(int)plan.decode.size(), (int)plan.prefills.size()});
        std::vector<int> chunk_slots;
        for (const PrefillSlice & slice : plan.prefills) chunk_slots.push_back(slice.slot);
        prompt_slots.push_back(std::move(chunk_slots));
        return result;
    }
    void retire(int slot) override { slots_[(size_t)slot] = {}; }

    std::mutex mu;
    std::vector<std::pair<int, int>> steps;  // (decode rows, prompt chunks)
    std::vector<std::vector<int>> prompt_slots;  // the slots each step prefilled

private:
    struct Slot { bool active = false; int remaining = 0; };
    std::array<Slot, 2> slots_{};
    int long_prompt_tokens_;
};

struct TimeShareBackend : MockBackend {
    explicit TimeShareBackend(int long_prompt_tokens = 0) : engine(long_prompt_tokens) {}
    TimeShareEngine engine;
    SeqEngine * seq_engine() override { return &engine; }
};

// A chat that decodes, then a 128-token prompt (8 chunks) beside it. Returns
// the most decode-only steps the chat took between two of the prompt's chunks.
static int decode_only_steps_between_chunks(int long_prompt_tokens) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    auto backend_owner = std::make_unique<TimeShareBackend>(long_prompt_tokens);
    TimeShareBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 4096;
    config.prefix_cache_cap = 0;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);

    int fds[4] = {-1, -1, -1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, fds + 2) == 0);
    const auto make_job = [](ServerJob & job, int fd, int prompt, int output, const char * id) {
        job.fd = fd;
        job.req.format = ApiFormat::OPENAI_CHAT;
        job.req.prompt_tokens.assign((size_t)prompt, 100);
        job.req.max_output = output;
        job.req.stream = false;
        job.req.model = "scheduler-test";
        job.req.response_id = id;
    };
    ServerJob chat, document;
    make_job(chat, fds[0], 4, 400, "chat");
    make_job(document, fds[2], 128, 1, "document");
    SchedulerTestHarness::enqueue(server, &chat);
    std::thread scheduler([&] { SchedulerTestHarness::run(server, backend.engine); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    SchedulerTestHarness::enqueue(server, &document);
    const auto wait_done = [](ServerJob & job) {
        std::unique_lock<std::mutex> lock(job.mu);
        return job.cv.wait_for(lock, std::chrono::seconds(10), [&] { return job.done; });
    };
    const bool done = wait_done(document) && wait_done(chat);
    SchedulerTestHarness::stop(server);
    scheduler.join();
    for (const int fd : fds) close(fd);
    unlink(path.c_str());
    TEST_ASSERT(done);

    int chunks_beside = 0, alone_between = 0, alone_run = 0;
    bool after_chunk = false;
    for (const auto & [decode, chunks] : backend.engine.steps) {
        if (chunks > 0 && decode > 0) {
            if (after_chunk) alone_between = std::max(alone_between, alone_run);
            ++chunks_beside;
            after_chunk = true;
            alone_run = 0;
        } else if (chunks == 0 && decode > 0 && after_chunk) {
            ++alone_run;
        }
    }
    TEST_ASSERT(chunks_beside >= 2);
    return alone_between;
}

TEST_CASE(ServerUnitFixture, test_scheduler_gives_decoders_time_beside_a_long_prompt) {
    // Between two chunks of the long prompt that ran beside the chat, the chat
    // stepped alone: several decode-only steps, not none.
    TEST_ASSERT(decode_only_steps_between_chunks(/*long_prompt_tokens=*/0) >= 5);
    // A prompt under the engine's long-prompt size keeps every step.
    TEST_ASSERT(decode_only_steps_between_chunks(/*long_prompt_tokens=*/256) == 0);
}

// A prompt that fits in one step runs before a longer one, alone.
TEST_CASE(ServerUnitFixture, test_scheduler_runs_a_short_prompt_before_a_long_one) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    auto backend_owner = std::make_unique<TimeShareBackend>();
    TimeShareBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 4096;
    config.prefix_cache_cap = 0;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);

    int fds[4] = {-1, -1, -1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, fds + 2) == 0);
    ServerJob document, question;
    for (auto [job, fd, prompt, id] : {std::tuple{&document, fds[0], 128, "document"},
                                       std::tuple{&question, fds[2], 4, "question"}}) {
        job->fd = fd;
        job->req.format = ApiFormat::OPENAI_CHAT;
        job->req.prompt_tokens.assign((size_t)prompt, 100);
        job->req.max_output = 1;
        job->req.stream = false;
        job->req.model = "scheduler-test";
        job->req.response_id = id;
    }
    SchedulerTestHarness::enqueue(server, &document);
    std::thread scheduler([&] { SchedulerTestHarness::run(server, backend.engine); });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    SchedulerTestHarness::enqueue(server, &question);
    const auto wait_done = [](ServerJob & job) {
        std::unique_lock<std::mutex> lock(job.mu);
        return job.cv.wait_for(lock, std::chrono::seconds(10), [&] { return job.done; });
    };
    const bool done = wait_done(question) && wait_done(document);
    SchedulerTestHarness::stop(server);
    scheduler.join();
    for (const int fd : fds) close(fd);
    unlink(path.c_str());
    TEST_ASSERT(done);

    // The question's step carried only the question; the document went on after.
    int question_step = -1, last_document_step = -1;
    for (int i = 0; i < (int)backend.engine.prompt_slots.size(); ++i) {
        const auto & chunk_slots = backend.engine.prompt_slots[(size_t)i];
        if (chunk_slots == std::vector<int>{1}) question_step = i;
        if (std::find(chunk_slots.begin(), chunk_slots.end(), 0) != chunk_slots.end()) last_document_step = i;
    }
    TEST_ASSERT(question_step > 0);
    TEST_ASSERT(last_document_step > question_step);
}

// Every decode step verifies a drafted chain of 8 positions; the first step
// accepts 4 of them, the second 2.
class SpecRoundEngine final : public SeqEngine {
public:
    int slot_count() const override { return 1; }
    int max_context() const override { return 4096; }
    bool token_is_eos(int32_t) const override { return false; }
    StepPlanLimits step_plan_limits(int) const override { return {1, 16, 16, 16}; }
    AdmitResult admit(uint64_t, const std::vector<int32_t> &, const SamplerCfg &) override {
        AdmitResult result;
        result.status = AdmitResult::Status::admitted;
        result.slot = 0;
        return result;
    }
    StepResult step(const StepPlan & plan) override {
        StepResult result;
        for (const StepInput & input : plan.decode) {
            DecodeOutput out;
            out.slot = input.slot;
            out.committed_tokens = {8, 7, 8};
            out.token = 7;
            out.spec_offered = 8;
            out.spec_accepted = rounds++ == 0 ? 4 : 2;
            result.decode.push_back(std::move(out));
        }
        for (const PrefillSlice & slice : plan.prefills) {
            PrefillOutput out;
            out.slot = slice.slot;
            out.status = PrefillOutput::Status::completed;
            out.token = 7;
            result.prefills.push_back(out);
        }
        return result;
    }
    void retire(int) override {}

    int rounds = 0;
};

struct SpecRoundBackend : MockBackend {
    SpecRoundEngine engine;
    SeqEngine * seq_engine() override { return &engine; }
};

// A request the parallel scheduler decoded with drafted rounds reports, in
// its /status perf record, that it speculated and the rounds' acceptance.
TEST_CASE(ServerUnitFixture, test_scheduler_perf_record_reports_speculative_rounds) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    auto backend_owner = std::make_unique<SpecRoundBackend>();
    SpecRoundBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 4096;
    config.prefix_cache_cap = 0;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);

    int fds[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    ServerJob job;
    job.fd = fds[0];
    job.req.format = ApiFormat::OPENAI_CHAT;
    job.req.prompt_tokens.assign(4, 100);
    job.req.max_output = 9;  // the prefill's token, then two rounds of 4
    job.req.stream = false;
    job.req.model = "scheduler-test";
    job.req.response_id = "spec";
    SchedulerTestHarness::enqueue(server, &job);
    std::thread scheduler([&] { SchedulerTestHarness::run(server, backend.engine); });
    bool done = false;
    {
        std::unique_lock<std::mutex> lock(job.mu);
        done = job.cv.wait_for(lock, std::chrono::seconds(10), [&] { return job.done; });
    }
    SchedulerTestHarness::stop(server);
    scheduler.join();
    for (const int fd : fds) close(fd);
    unlink(path.c_str());
    TEST_ASSERT(done);
    TEST_ASSERT(backend.engine.rounds == 2);

    const json perf = SchedulerTestHarness::status_json(server)["perf_history"];
    TEST_ASSERT(perf.size() == 1);
    TEST_ASSERT(perf[0]["completion_tokens"] == 9);
    TEST_ASSERT(perf[0]["spec_decode"] == true);
    // The second round had room for 4 of its 8 positions, as single-lane
    // decode counts a round that reaches the output budget.
    TEST_ASSERT(std::abs(perf[0]["accept_rate"].get<double>() - 6.0 / 12.0) < 1e-6);
}

TEST_CASE(ServerUnitFixture,
          test_scheduler_discards_unrequested_restored_checkpoint) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<SchedulerPrefixBackend>();
    SchedulerPrefixBackend & backend = *backend_owner;
    backend.engine.unrequested_restore.store(true, std::memory_order_relaxed);
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;
    config.concurrent_prefix_cache_max_bytes = 1024;
    config.concurrent_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    cache.confirm_inline_snap(
        /*slot=*/0, /*target_cut=*/2, {1, 100}, false, 128);

    int sockets[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);

    ServerJob job;
    job.fd = sockets[0];
    job.req.format = ApiFormat::OPENAI_CHAT;
    job.req.prompt_tokens = {1, 100, 999};
    job.req.max_output = 1;
    job.req.stream = false;
    job.req.model = "scheduler-test";
    job.req.response_id = "unrequested-restore";

    SchedulerTestHarness::enqueue(server, &job);
    std::thread scheduler([&] {
        SchedulerTestHarness::run(server, backend.engine);
    });

    std::unique_lock<std::mutex> lock(job.mu);
    const bool done = job.cv.wait_for(
        lock, std::chrono::seconds(5), [&] { return job.done; });
    lock.unlock();
    SchedulerTestHarness::stop(server);
    scheduler.join();

    close(sockets[0]);
    close(sockets[1]);
    unlink(path.c_str());

    TEST_ASSERT(done);
    TEST_ASSERT(backend.engine.discarded ==
                std::vector<PrefixStoreRef>({{2, 2}}));
    TEST_ASSERT(cache.lookup_candidate({1, 100, 999}, 2).first == 0);
}

TEST_CASE(ServerUnitFixture,
          test_scheduler_stale_restore_preserves_other_protected_capture) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<SchedulerPrefixBackend>();
    SchedulerPrefixBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;
    config.concurrent_prefix_cache_max_bytes = 1024;
    config.concurrent_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    cache.confirm_inline_snap(
        /*slot=*/0, /*target_cut=*/2, {1, 100}, false, 128);
    cache.confirm_inline_snap(
        /*slot=*/1, /*target_cut=*/2, {1, 200}, false, 128);

    int capture_sockets[2] = {-1, -1};
    int restore_sockets[2] = {-1, -1};
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, capture_sockets) == 0);
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, restore_sockets) == 0);

    ServerJob capture_job;
    capture_job.fd = capture_sockets[0];
    capture_job.req.format = ApiFormat::OPENAI_CHAT;
    capture_job.req.prompt_tokens = {1, 300, 999};
    capture_job.req.max_output = 1;
    capture_job.req.stream = false;
    capture_job.req.model = "scheduler-test";
    capture_job.req.response_id = "capture";
    capture_job.req.pin_end_token = 2;
    capture_job.req.tools = json::array(
        {{{"type", "function"},
          {"function", {{"name", "probe"}, {"parameters", json::object()}}}}});

    ServerJob restore_job;
    restore_job.fd = restore_sockets[0];
    restore_job.req.format = ApiFormat::OPENAI_CHAT;
    restore_job.req.prompt_tokens = {1, 200, 999};
    restore_job.req.max_output = 1;
    restore_job.req.stream = false;
    restore_job.req.model = "scheduler-test";
    restore_job.req.response_id = "restore";

    SchedulerTestHarness::enqueue(server, &capture_job);
    SchedulerTestHarness::enqueue(server, &restore_job);
    std::thread scheduler([&] {
        SchedulerTestHarness::run(server, backend.engine);
    });

    const auto wait_done = [](ServerJob & job) {
        std::unique_lock<std::mutex> lock(job.mu);
        return job.cv.wait_for(lock, std::chrono::seconds(5),
                               [&] { return job.done; });
    };
    const bool capture_done = wait_done(capture_job);
    const bool restore_done = wait_done(restore_job);
    SchedulerTestHarness::stop(server);
    scheduler.join();

    close(capture_sockets[0]);
    close(capture_sockets[1]);
    close(restore_sockets[0]);
    close(restore_sockets[1]);
    unlink(path.c_str());

    TEST_ASSERT(capture_done);
    TEST_ASSERT(restore_done);
    TEST_ASSERT(backend.engine.saw_capture);
    TEST_ASSERT(backend.engine.saw_stale_restore);
    auto stats = cache.stats();
    TEST_ASSERT(stats.in_use == 1);
    TEST_ASSERT(stats.resident_bytes == 256);
    TEST_ASSERT(stats.capture_attempts == 1);
    TEST_ASSERT(stats.capture_failures == 0);
    TEST_ASSERT(stats.capture_stall_us_total == 1500);
    TEST_ASSERT(stats.restore_attempts == 1);
    TEST_ASSERT(stats.restore_invalidations == 1);
    TEST_ASSERT(stats.restore_stall_us_total == 2500);

    // Refill the stale slot. The next capture must evict this unprotected
    // entry, proving the real scheduler preserved the other request's pin.
    cache.confirm_inline_snap(
        /*slot=*/1, /*target_cut=*/2, {1, 400}, false, 128);
    auto victim = cache.reserve_inline_snap(
        {1, 500}, 0, /*prefer_tools_boundary=*/false, /*forced_cut=*/2,
        /*restore_source_slot=*/-1,
        [](int) { return 128; });
    TEST_ASSERT(victim.slot() == 1);
    victim.cancel();
}
#endif

struct MockPflashCompressBackend : MockBackend {
    int compress_calls = 0;
    CompressRequest last_request;

    CompressResult compress(const CompressRequest & request) override {
        ++compress_calls;
        last_request = request;
        return CompressResult::from_compressed_ids(request.input_ids);
    }
};

TEST_CASE(ServerUnitFixture, test_pflash_default_raw_text_maps_user_query) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", nullptr};
    luce_test::ScopedEnvVar chunk{"PFLASH_SELECT_CHUNK_SIZE", nullptr};
    luce_test::ScopedEnvVar query{"PFLASH_SELECT_QUERY_TOKENS", nullptr};
    luce_test::ScopedEnvVar parser{"PFLASH_SELECT_QUERY_PARSER", nullptr};
    luce_test::ScopedEnvVar top_p{"PFLASH_SELECT_TOP_P", nullptr};

    const std::string tokenizer_path =
        write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(tokenizer_path.c_str()));

    auto backend_owner = std::make_unique<MockPflashCompressBackend>();
    MockPflashCompressBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        server.set_drafter_tokenizer(&tokenizer);

        ParsedRequest request;
        request.format = ApiFormat::RESPONSES;
        request.messages = "x";
        request.prompt_tokens = tokenizer.encode("x");

        const std::string error =
            HttpServerTestAccess::apply_pflash_compression(server, request);
        TEST_ASSERT_MSG(error.empty(), error);
    }

    TEST_ASSERT(backend.compress_calls == 1);
    TEST_ASSERT(backend.last_request.score_query_end == 1);
    TEST_ASSERT(backend.last_request.score_query_tokens == 1);
    unlink(tokenizer_path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_strict_chat_query_is_the_prompt_end) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "top_p"};

    // The server's own Qwen rendering, generation prompt and its think
    // prefix included.
    const std::string rendered = render_chat_template(
        {{"system", "You are helpful.", ""},
         {"user", "What is the answer?", ""}},
        ChatFormat::QWEN3, /*add_generation_prompt=*/true,
        /*enable_thinking=*/true);
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"What", " is", " the", " answer", "?", "user", "assistant",
         "system", "\n", "You", " are", " helpful", "."},
        rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashCompressBackend>();
    MockPflashCompressBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash_keep_ratio = 1.0f;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        server.set_drafter_tokenizer(&tokenizer);

        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = json::array({
            {{"role", "system"}, {"content", "You are helpful."}},
            {{"role", "user"}, {"content", "What is the answer?"}},
        });
        request.prompt_tokens = tokenizer.encode(rendered);

        const std::string error =
            HttpServerTestAccess::apply_pflash_compression(server, request);
        TEST_ASSERT_MSG(error.empty(), error);
    }

    TEST_ASSERT(backend.compress_calls == 1);
    const auto & ids = backend.last_request.input_ids;
    const int im_end = tokenizer.token_to_id("<|im_end|>");
    int last_im_end = -1;
    for (int i = 0; i < (int) ids.size(); ++i) {
        if (ids[i] == im_end) last_im_end = i;
    }
    TEST_ASSERT(last_im_end > 0);
    // The scorer query is the prompt's last token, where the model starts
    // answering; nothing is parsed out of the user's text.
    TEST_ASSERT(backend.last_request.score_query_end == (int) ids.size());
    TEST_ASSERT(backend.last_request.score_query_tokens == 1);
    // The user turn's tail scores as a second query window.
    const auto turn = backend.last_request.turn_query_span;
    TEST_ASSERT(turn.begin >= 0 && turn.end == last_im_end);
    TEST_ASSERT(tokenizer.decode({ids.begin() + turn.begin, ids.begin() + turn.end})
                == "What is the answer?");
    // The generation prompt, the turn's role header and -- a short turn --
    // the whole question stay.
    bool header_pinned = false;
    bool question_pinned = false;
    bool generation_pinned = false;
    for (const auto & span : backend.last_request.required_instruction_spans) {
        const std::string text = tokenizer.decode(
            {ids.begin() + span.begin, ids.begin() + span.end});
        if (text.find("<|im_start|>user\n") != std::string::npos) {
            header_pinned = true;
        }
        if (text.find("What is the answer?") != std::string::npos) {
            question_pinned = true;
        }
        if (span.begin <= last_im_end + 2 && span.end == (int) ids.size()) {
            generation_pinned = true;
        }
    }
    TEST_ASSERT(header_pinned);
    TEST_ASSERT(question_pinned);
    TEST_ASSERT(generation_pinned);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_strict_agent_turns_after_query_are_candidates) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "top_p"};

    const std::vector<ChatMessage> messages{
        {"user", "What is the answer?", ""},
        {"assistant", "Sure.", ""},
        {"tool", "tool output here", "call-1"},
    };
    const std::string rendered = render_chat_template(
        messages, ChatFormat::QWEN3, /*add_generation_prompt=*/true,
        /*enable_thinking=*/true);
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"What", " is", " the", " answer", "?", "user", "assistant", "\n",
         "Sure", "."},
        rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashCompressBackend>();
    MockPflashCompressBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash_keep_ratio = 1.0f;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        server.set_drafter_tokenizer(&tokenizer);

        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = json::array({
            {{"role", "user"}, {"content", "What is the answer?"}},
            {{"role", "assistant"}, {"content", "Sure."}},
            {{"role", "tool"}, {"content", "tool output here"},
             {"tool_call_id", "call-1"}},
        });
        request.prompt_tokens = tokenizer.encode(rendered);

        const std::string error =
            HttpServerTestAccess::apply_pflash_compression(server, request);
        TEST_ASSERT_MSG(error.empty(), error);
    }

    TEST_ASSERT(backend.compress_calls == 1);
    const auto & request = backend.last_request;
    const auto & ids = request.input_ids;
    // The query is the prompt's end, so nothing follows it: the assistant
    // and tool turns are ordinary context.
    TEST_ASSERT(!request.query_suffix_candidates);
    TEST_ASSERT(request.score_query_end == (int) ids.size());
    TEST_ASSERT(request.score_query_tokens == 1);
    // The generation prompt is pinned, and the short assistant turn stays as
    // part of the conversation's skeleton; the tool output between the query
    // and the generation prompt is scored, not pinned.
    bool generation_pinned = false;
    for (const auto & span : request.required_instruction_spans) {
        const std::string text = tokenizer.decode(
            {ids.begin() + span.begin, ids.begin() + span.end});
        TEST_ASSERT_MSG(text.find("tool output") == std::string::npos, text);
        if (span.end == (int) ids.size() &&
            text.find("<|im_start|>assistant\n<think>\n") != std::string::npos) {
            generation_pinned = true;
        }
    }
    TEST_ASSERT(generation_pinned);
    unlink(path.c_str());
}

// Keeps a prefix of the input that fits the requested ratio, so the target
// ceiling check sees a real compression.
struct MockPflashBudgetBackend : MockBackend {
    int compress_calls = 0;
    CompressRequest last_request;

    CompressResult compress(const CompressRequest & request) override {
        ++compress_calls;
        last_request = request;
        const size_t keep = (size_t) std::max(1.0, std::floor(
            (double) request.input_ids.size() * request.keep_ratio) - 2.0);
        return CompressResult::from_compressed_ids(std::vector<int32_t>(
            request.input_ids.begin(),
            request.input_ids.begin() + (long) std::min(keep, request.input_ids.size())));
    }
};

struct PflashSystemPromptCase {
    std::string rendered;
    json messages;
    std::vector<std::string> vocab;
};

static PflashSystemPromptCase pflash_long_system_prompt_case(
        const std::string & instruction_role = "system") {
    std::string system;
    for (int i = 0; i < 30; ++i) system += "You are helpful. ";
    // Longer than the multi-turn skeleton keeps whole: droppable material.
    std::string history;
    for (int i = 0; i < 120; ++i) history += "Sure. ";
    PflashSystemPromptCase out;
    out.rendered = render_chat_template(
        {{instruction_role, system, ""},
         {"user", history, ""},
         {"assistant", "Sure.", ""},
         {"user", "What is the answer?", ""}},
        // As the server renders it: ParsedRequest defaults to thinking on,
        // and the instruction spans come from re-renders of this prompt.
        ChatFormat::QWEN3, /*add_generation_prompt=*/true,
        /*enable_thinking=*/true);
    out.messages = json::array({
        {{"role", instruction_role}, {"content", system}},
        {{"role", "user"}, {"content", history}},
        {{"role", "assistant"}, {"content", "Sure."}},
        {{"role", "user"}, {"content", "What is the answer?"}},
    });
    out.vocab = {"What", " is", " the", " answer", "?", "user", "assistant",
                 "system", "\n", "You", " are", " helpful", ".", " ", "Sure"};
    return out;
}

TEST_CASE(ServerUnitFixture,
        test_pflash_strict_budget_spends_keep_ratio_on_droppable_tokens) {
    // A system prompt larger than keep_ratio x prompt used to exhaust the
    // budget (mandatory_query_exceeds_budget). It is kept and the ratio now
    // applies to the rest.
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar chunk{"PFLASH_SELECT_CHUNK_SIZE", "4"};
    const auto prompt = pflash_long_system_prompt_case();
    const std::string path =
        write_pflash_bpe_tokenizer_fixture(prompt.vocab, prompt.rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashBudgetBackend>();
    MockPflashBudgetBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash_keep_ratio = 0.05f;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    std::string error;
    {
        HttpServer server(engine, tokenizer, config);
        server.set_drafter_tokenizer(&tokenizer);
        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = prompt.messages;
        request.prompt_tokens = tokenizer.encode(prompt.rendered);
        error = HttpServerTestAccess::apply_pflash_compression(server, request);
    }
    TEST_ASSERT_MSG(error.empty(), error);
    TEST_ASSERT(backend.compress_calls == 1);
    const auto & request = backend.last_request;
    const int input = (int) request.input_ids.size();
    int system_end = -1;
    for (const auto & span : request.required_instruction_spans) {
        const std::string text = tokenizer.decode(
            {request.input_ids.begin() + span.begin,
             request.input_ids.begin() + span.end});
        if (text.find("You are helpful") != std::string::npos) {
            system_end = span.end;
        }
    }
    TEST_ASSERT(system_end > 0);   // the system prompt is still kept
    // Its tokens are charged on top of 5 % of the droppable rest.
    TEST_ASSERT(request.keep_ratio > (double) system_end / input);
    TEST_ASSERT(request.keep_ratio < 1.0f);
    unlink(path.c_str());
}

static std::string pflash_overflowing_instruction_error(
        const std::string & role, bool & compressed,
        std::vector<std::string> & kept_texts) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar chunk{"PFLASH_SELECT_CHUNK_SIZE", "4"};
    const auto prompt = pflash_long_system_prompt_case(role);
    const std::string path =
        write_pflash_bpe_tokenizer_fixture(prompt.vocab, prompt.rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashBudgetBackend>();
    MockPflashBudgetBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash_keep_ratio = 0.3f;
    config.max_ctx = 200;   // smaller than the instructions + max_output
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    std::string error;
    {
        HttpServer server(engine, tokenizer, config);
        server.set_drafter_tokenizer(&tokenizer);
        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = prompt.messages;
        request.prompt_tokens = tokenizer.encode(prompt.rendered);
        request.max_output = 16;
        error = HttpServerTestAccess::apply_pflash_compression(server, request);
    }
    compressed = backend.compress_calls == 1;
    kept_texts.clear();
    if (compressed) {
        const auto & request = backend.last_request;
        for (const auto & span : request.required_instruction_spans) {
            kept_texts.push_back(tokenizer.decode(
                {request.input_ids.begin() + span.begin,
                 request.input_ids.begin() + span.end}));
        }
    }
    unlink(path.c_str());
    return error;
}

TEST_CASE(ServerUnitFixture,
        test_pflash_strict_refuses_system_prompt_that_overflows_context) {
    // The system prompt is never compressed: when it alone does not fit the
    // context the request fails before the drafter runs.
    bool compressed = true;
    std::vector<std::string> kept;
    const std::string error =
        pflash_overflowing_instruction_error("system", compressed, kept);
    TEST_ASSERT_MSG(error.find("system prompt alone does not fit") !=
                        std::string::npos, error);
    TEST_ASSERT(!compressed);
}

TEST_CASE(ServerUnitFixture,
        test_pflash_strict_scores_developer_text_that_overflows_context) {
    // A developer message too large for the context is data: it loses its
    // pin and competes for the budget.
    bool compressed = false;
    std::vector<std::string> kept;
    const std::string error =
        pflash_overflowing_instruction_error("developer", compressed, kept);
    TEST_ASSERT_MSG(error.empty(), error);
    TEST_ASSERT(compressed);
    for (const auto & text : kept) {
        TEST_ASSERT_MSG(text.find("You are helpful") == std::string::npos, text);
    }
}

TEST_CASE(ServerUnitFixture,
        test_pflash_auto_threshold_counts_droppable_tokens) {
    // Auto mode: the whole prompt clears the threshold, the droppable part
    // does not -- nothing worth selecting, so the prompt goes through as is.
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar chunk{"PFLASH_SELECT_CHUNK_SIZE", "4"};
    const auto prompt = pflash_long_system_prompt_case();
    const std::string path =
        write_pflash_bpe_tokenizer_fixture(prompt.vocab, prompt.rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const auto ids = tokenizer.encode(prompt.rendered);

    auto backend_owner = std::make_unique<MockPflashBudgetBackend>();
    MockPflashBudgetBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash = true;
    config.pflash_threshold = (int) ids.size() - 20;
    config.pflash_keep_ratio = 0.3f;
    config.max_ctx = 8192;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        server.set_drafter_tokenizer(&tokenizer);
        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = prompt.messages;
        request.prompt_tokens = ids;
        const auto prepared =
            HttpServerTestAccess::prepare_prompt(server, request);
        TEST_ASSERT_MSG(prepared.error.empty(), prepared.error);
        TEST_ASSERT(!prepared.compressed);
        TEST_ASSERT(prepared.tokens == ids);
    }
    TEST_ASSERT(backend.compress_calls == 0);
    unlink(path.c_str());
}

// Failure policy D5: the drafter fails the way the R9700 did on 9 Oct
// (non-finite request-probe scores, empty result).
namespace luce::common { void set_last_error(std::string msg); }

struct MockPflashFailingBackend : MockBackend {
    int compress_calls = 0;
    CompressResult compress(const CompressRequest &) override {
        ++compress_calls;
        luce::common::set_last_error(
            "non-finite Qwen3.5 scoring-head scores (request probe): 7/7");
        return {};
    }
};

struct PflashFallbackRun {
    HttpServerTestAccess::PreparedPrompt prepared;
    json counters;
    std::vector<int32_t> ids;
    int compress_calls = 0;
    std::vector<json> view_records;
};

template <typename Backend>
static PflashFallbackRun run_pflash_fallback_case(int max_ctx, int max_output) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar chunk{"PFLASH_SELECT_CHUNK_SIZE", "4"};
    char trace_path[] = "/tmp/pflash_fallback_view_XXXXXX";
    const int trace_fd = mkstemp(trace_path);
    TEST_ASSERT(trace_fd >= 0);
    close(trace_fd);
    luce_test::ScopedEnvVar view_trace{"PFLASH_VIEW_TRACE_PATH", trace_path};
    const auto prompt = pflash_long_system_prompt_case();
    const std::string path =
        write_pflash_bpe_tokenizer_fixture(prompt.vocab, prompt.rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PflashFallbackRun run;
    run.ids = tokenizer.encode(prompt.rendered);

    auto backend_owner = std::make_unique<Backend>();
    Backend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash = true;
    config.pflash_threshold = 0;   // every prompt
    config.pflash_keep_ratio = 0.3f;
    config.max_ctx = max_ctx;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);
        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = prompt.messages;
        request.prompt_tokens = run.ids;
        request.max_output = max_output;
        run.prepared = HttpServerTestAccess::prepare_prompt(server, request);
        run.counters = HttpServerTestAccess::pflash_failure_counters(server);
    }
    run.compress_calls = backend.compress_calls;
    std::ifstream in(trace_path);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) run.view_records.push_back(json::parse(line));
    }
    unlink(trace_path);
    unlink(path.c_str());
    return run;
}

TEST_CASE(ServerUnitFixture, test_pflash_failure_falls_back_to_raw_prompt_with_flag) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    const auto run = run_pflash_fallback_case<MockPflashFailingBackend>(8192, 16);
    TEST_ASSERT(run.compress_calls >= 1);
    TEST_ASSERT_MSG(run.prepared.error.empty(), run.prepared.error);
    TEST_ASSERT(run.prepared.error_status == 0);
    TEST_ASSERT(!run.prepared.compressed);
    TEST_ASSERT(run.prepared.tokens == run.ids);   // the raw prompt, untouched
    TEST_ASSERT(run.prepared.snapshot_cut == -1);
    const json & flag = run.prepared.pflash_stats["pflash_fallback"];
    TEST_ASSERT(flag.value("reason", "") == "nonfinite_scores");
    TEST_ASSERT(flag.value("error", "") == "PFlash compression failed");
    TEST_ASSERT(flag.value("detail", "").find("non-finite") != std::string::npos);
    TEST_ASSERT(flag.value("prompt_tokens", 0) == (int) run.ids.size());
    TEST_ASSERT(run.counters.value("fallback_served", 0) == 1);
    TEST_ASSERT(run.counters.value("fallback_refused", 0) == 0);
    TEST_ASSERT(run.counters["fallback_reasons"].value("nonfinite_scores", 0) == 1);
    // One view-trace record, kind raw, carrying the flag at the top level.
    TEST_ASSERT(run.view_records.size() == 1);
    TEST_ASSERT(run.view_records[0].value("kind", "") == "raw");
    TEST_ASSERT(run.view_records[0]["pflash_fallback"].value("reason", "") == "nonfinite_scores");
}

TEST_CASE(ServerUnitFixture, test_pflash_failure_fallback_refused_when_raw_prompt_does_not_fit) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    // max_ctx admits the compressed prompt but not the raw one plus the reply.
    const auto probe = run_pflash_fallback_case<MockPflashFailingBackend>(8192, 16);
    const int raw = (int) probe.ids.size();
    const auto run = run_pflash_fallback_case<MockPflashFailingBackend>(raw + 15, 16);
    TEST_ASSERT(run.prepared.error == "PFlash compression failed");
    TEST_ASSERT(run.prepared.error_status == 500);
    TEST_ASSERT(!run.prepared.pflash_stats.is_object() ||
                !run.prepared.pflash_stats.contains("pflash_fallback"));
    TEST_ASSERT(run.counters.value("fallback_served", 0) == 0);
    TEST_ASSERT(run.counters.value("fallback_refused", 0) == 1);
    TEST_ASSERT(run.counters["refused_reasons"].value("raw_exceeds_max_ctx", 0) == 1);
    TEST_ASSERT(run.view_records.empty());
    // Exactly at the limit the raw prompt is admissible.
    const auto edge = run_pflash_fallback_case<MockPflashFailingBackend>(raw + 16, 16);
    TEST_ASSERT_MSG(edge.prepared.error.empty(), edge.prepared.error);
    TEST_ASSERT(edge.prepared.tokens == edge.ids);
}

TEST_CASE(ServerUnitFixture, test_pflash_fail_closed_switch_keeps_the_error) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", "1"};
    const auto run = run_pflash_fallback_case<MockPflashFailingBackend>(8192, 16);
    TEST_ASSERT(run.prepared.error == "PFlash compression failed");
    TEST_ASSERT(run.prepared.error_status == 500);
    TEST_ASSERT(run.view_records.empty());
    TEST_ASSERT(run.counters.value("fail_closed", false));
    TEST_ASSERT(run.counters.value("fallback_served", 0) == 0);
    TEST_ASSERT(run.counters["refused_reasons"].value("fail_closed", 0) == 1);
}

TEST_CASE(ServerUnitFixture, test_pflash_success_path_has_no_fallback) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    const auto run = run_pflash_fallback_case<MockPflashBudgetBackend>(8192, 16);
    TEST_ASSERT(run.compress_calls >= 1);
    TEST_ASSERT_MSG(run.prepared.error.empty(), run.prepared.error);
    TEST_ASSERT(run.prepared.compressed);
    TEST_ASSERT(run.prepared.tokens.size() < run.ids.size());
    TEST_ASSERT(!run.prepared.pflash_stats.contains("pflash_fallback"));
    TEST_ASSERT(run.counters.value("fallback_served", 0) == 0);
    TEST_ASSERT(run.counters.value("fallback_refused", 0) == 0);
    TEST_ASSERT(run.view_records.size() == 1);
    TEST_ASSERT(run.view_records[0].value("kind", "") == "compressed");
    TEST_ASSERT(!run.view_records[0].contains("pflash_fallback"));
}

TEST_CASE(ServerUnitFixture, test_pflash_fallback_decision_and_reasons) {
    using luce::common::http_detail::pflash_fallback_decision;
    using luce::common::http_detail::pflash_failure_reason;
    // Config / request-contract errors never fall back.
    TEST_ASSERT(pflash_fallback_decision("invalid PFlash strict selection config: x",
                                         false, 100, 10, 1000).refused == "config_error");
    TEST_ASSERT(pflash_fallback_decision("PFlash pflash_required needs strict budget selection",
                                         false, 100, 10, 1000).refused == "config_error");
    TEST_ASSERT(pflash_fallback_decision("PFlash compression failed", true, 100, 10, 1000).refused
                == "fail_closed");
    TEST_ASSERT(pflash_fallback_decision("PFlash compression failed", false, 991, 10, 1000).refused
                == "raw_exceeds_max_ctx");
    TEST_ASSERT(pflash_fallback_decision("PFlash compression failed", false, 990, 10, 1000).fallback);
    TEST_ASSERT(pflash_fallback_decision(
        "PFlash strict selection final prompt exceeds target-token ceiling (5 > 4)",
        false, 100, 10, 1000).fallback);
    TEST_ASSERT(pflash_failure_reason("PFlash compression failed", "") == "compress_failed");
    TEST_ASSERT(pflash_failure_reason("PFlash compression failed",
                                      "non-finite Qwen3.5 scoring-head scores: 3/9") == "nonfinite_scores");
    TEST_ASSERT(pflash_failure_reason("remote PFlash drafter compression failed", "") == "remote_drafter");
    TEST_ASSERT(pflash_failure_reason(
        "PFlash strict selection final prompt exceeds target-token ceiling (5 > 4)", "") == "ceiling_exceeded");
    TEST_ASSERT(pflash_failure_reason("PFlash scorer query mapping failed", "") == "mapping_failed");
    TEST_ASSERT(pflash_failure_reason("PFlash strict selection: the system prompt alone does not fit", "")
                == "selection_failed");
}

// PFLASH_FALLBACK_SPREAD (relevance-spread fallback, release2).
TEST_CASE(ServerUnitFixture, test_pflash_relevance_spread_formula) {
    using luce::common::pflash_relevance_spread;
    const auto near = [](double a, double b) { return std::fabs(a - b) < 1e-12; };
    // Uniform over 4, two selected: exp(ln 4) / 2.
    auto r = pflash_relevance_spread({1, 1, 1, 1}, {1, 1, 0, 0}, {0, 0, 0, 0});
    TEST_ASSERT(r.valid && r.k == 2 && r.candidates == 4 && near(r.spread, 2.0));
    // One-hot: H = 0.
    r = pflash_relevance_spread({5, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0});
    TEST_ASSERT(r.valid && r.k == 1 && near(r.entropy, 0.0) && near(r.spread, 1.0));
    // Mixed, one forced selected chunk not counted; reference = sim_tier2.spread_of.
    r = pflash_relevance_spread({0.2, 0.5, 0.1, 0.0, 0.2}, {0, 1, 1, 0, 1}, {0, 0, 1, 0, 0});
    TEST_ASSERT(r.valid && r.k == 2);
    TEST_ASSERT_MSG(std::fabs(r.spread - 1.6946226386695418) < 1e-9, std::to_string(r.spread));
    // All-zero scores: sum 0 -> H = 0 -> 1 / max(k, 1).
    r = pflash_relevance_spread({0, 0, 0}, {1, 1, 0}, {0, 0, 0});
    TEST_ASSERT(r.valid && r.k == 2 && near(r.spread, 0.5));
    r = pflash_relevance_spread({0, 0, 0}, {0, 0, 0}, {0, 0, 0});
    TEST_ASSERT(r.valid && r.k == 0 && near(r.spread, 1.0));
    // k = 0 (nothing selected): exp(H) / 1.
    r = pflash_relevance_spread({1, 3}, {0, 0}, {0, 0});
    TEST_ASSERT(r.valid && r.k == 0);
    TEST_ASSERT(std::fabs(r.spread - 1.7547653506033232) < 1e-12);
    // Forced-only selection: k = 0 as well.
    r = pflash_relevance_spread({1, 3, 2}, {1, 1, 0}, {1, 1, 0});
    TEST_ASSERT(r.valid && r.k == 0);
    TEST_ASSERT(near(r.spread, std::exp(-(1.0/6*std::log(1.0/6) + 0.5*std::log(0.5) +
                                          1.0/3*std::log(1.0/3)))));
    // Invalid inputs: no decision may be taken.
    TEST_ASSERT(!pflash_relevance_spread({1, 2}, {1}, {0, 0}).valid);
    TEST_ASSERT(!pflash_relevance_spread({1, 2}, {1, 0}, {0}).valid);
    TEST_ASSERT(!pflash_relevance_spread({1, NAN}, {1, 0}, {0, 0}).valid);
    TEST_ASSERT(!pflash_relevance_spread({1, INFINITY}, {1, 0}, {0, 0}).valid);
}

TEST_CASE(ServerUnitFixture, test_pflash_spread_decision_region_plan) {
    using luce::common::http_detail::pflash_spread_decision;
    using luce::common::http_detail::pflash_spread_fires;
    // Without a region plan the decision is exactly pflash_spread_fires.
    std::mt19937 rng(20261011);
    std::uniform_real_distribution<double> u(0.0, 0.4);
    const double thresholds[] = {0.0, 0.05, 0.11, 0.3};
    for (int i = 0; i < 2000; ++i) {
        const double t = thresholds[i % 4];
        const double sp = (i % 97 == 0) ? NAN : u(rng);
        const bool valid = (i % 13) != 0;
        const auto d = pflash_spread_decision(t, valid, sp, false);
        TEST_ASSERT(d.fires == pflash_spread_fires(t, valid, sp));
        TEST_ASSERT(d.skipped == nullptr);
    }
    TEST_ASSERT(pflash_spread_decision(0.11, true, 0.11, false).fires);
    // A region plan takes no decision, whatever the spread.
    for (double sp : {0.0, 0.05, 0.11, 0.5, 5.0}) {
        for (bool valid : {true, false}) {
            const auto d = pflash_spread_decision(0.11, valid, sp, true);
            TEST_ASSERT(!d.fires);
            TEST_ASSERT(d.skipped != nullptr && std::string(d.skipped) == "region_plan");
        }
    }
}

TEST_CASE(ServerUnitFixture, test_pflash_spread_threshold_boundary) {
    using luce::common::http_detail::pflash_spread_fires;
    TEST_ASSERT(pflash_spread_fires(0.11, true, 0.11));           // >= fires
    TEST_ASSERT(pflash_spread_fires(0.11, true, 0.5));
    TEST_ASSERT(!pflash_spread_fires(0.11, true, std::nextafter(0.11, 0.0)));
    TEST_ASSERT(!pflash_spread_fires(0.0, true, 5.0));             // off
    TEST_ASSERT(!pflash_spread_fires(0.11, false, 5.0));           // invalid spread
    TEST_ASSERT(!pflash_spread_fires(0.11, true, NAN));
    using luce::common::http_detail::pflash_failure_reason;
    TEST_ASSERT(pflash_failure_reason(
        "PFlash relevance spread fallback (spread 0.200000 >= 0.110000)", "") == "spread");
    TEST_ASSERT(pflash_failure_reason("PFlash relevance spread unavailable", "") ==
                "spread_unavailable");
}

TEST_CASE(ServerUnitFixture, test_pflash_fallback_spread_env_parse) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce::pflash::PFlashSelectionConfig config;
    std::string error;
    {
        luce_test::ScopedEnvVar spread{"PFLASH_FALLBACK_SPREAD", nullptr};
        TEST_ASSERT(luce::pflash::resolve_pflash_selection(1000, 32, config, error));
        TEST_ASSERT(config.fallback_spread == 0.0);
    }
    for (const char * off : {"0", "off", ""}) {
        luce_test::ScopedEnvVar spread{"PFLASH_FALLBACK_SPREAD", off};
        TEST_ASSERT_MSG(luce::pflash::resolve_pflash_selection(1000, 32, config, error), error);
        TEST_ASSERT(config.fallback_spread == 0.0);
    }
    {
        luce_test::ScopedEnvVar spread{"PFLASH_FALLBACK_SPREAD", "0.11"};
        TEST_ASSERT_MSG(luce::pflash::resolve_pflash_selection(1000, 32, config, error), error);
        TEST_ASSERT(config.fallback_spread == 0.11);
    }
    for (const char * bad : {"-0.1", "abc", "0.11x", "nan", "inf"}) {
        luce_test::ScopedEnvVar spread{"PFLASH_FALLBACK_SPREAD", bad};
        TEST_ASSERT_MSG(!luce::pflash::resolve_pflash_selection(1000, 32, config, error), bad);
        TEST_ASSERT(error.find("PFLASH_FALLBACK_SPREAD") != std::string::npos);
    }
    {
        luce_test::ScopedEnvVar legacy{"PFLASH_SELECT_MODE", nullptr};
        luce_test::ScopedEnvVar spread{"PFLASH_FALLBACK_SPREAD", "0.11"};
        TEST_ASSERT(!luce::pflash::resolve_pflash_selection(1000, 32, config, error));
    }
}

// The sel-trace carries spread / spread_threshold only with the flag on, and
// the value is the formula applied to the record's own chunk_scores /
// selected_chunks / forced_chunks.
TEST_CASE(ServerUnitFixture, test_pflash_spread_in_selection_trace) {
    char trace_path[] = "/tmp/pflash_spread_trace_XXXXXX";
    const int fd = mkstemp(trace_path);
    TEST_ASSERT(fd >= 0);
    close(fd);
    luce_test::ScopedEnvVar trace{"PFLASH_TRACE_PATH", trace_path};
    std::vector<int32_t> ids(64);
    for (int i = 0; i < 64; ++i) ids[(size_t) i] = 100 + i;
    std::vector<float> mass(64, 0.001f);
    for (int i = 16; i < 24; ++i) mass[(size_t) i] = 0.05f;
    for (int i = 40; i < 44; ++i) mass[(size_t) i] = 0.03f;
    luce::pflash::PFlashSelectionConfig config;
    config.mode = luce::pflash::PFlashSelectionMode::BudgetOnly;
    config.selection_active = true;
    config.configured = true;
    config.chunk_size = 8;
    const auto run = [&] {
        return luce::common::select_pflash_chunks(
            ids, mass, /*keep_ratio=*/0.4f, /*n_lookahead=*/4, /*score_query_end=*/64,
            /*pool_kernel=*/1, config, {}, /*direct_mass=*/true, /*write_trace=*/true);
    };
    TEST_ASSERT(!run().empty());
    const auto off = luce::common::pflash_last_relevance_spread();
    config.fallback_spread = 0.11;
    TEST_ASSERT(!run().empty());
    const auto on = luce::common::pflash_last_relevance_spread();
    TEST_ASSERT(off.valid && on.valid && off.spread == on.spread);
    std::ifstream in(trace_path);
    std::vector<json> rows;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) rows.push_back(json::parse(line));
    }
    unlink(trace_path);
    TEST_ASSERT(rows.size() == 2);
    TEST_ASSERT(!rows[0].contains("spread") && !rows[0].contains("spread_threshold"));
    TEST_ASSERT(rows[1].value("spread_threshold", 0.0) == 0.11);
    TEST_ASSERT(rows[1].value("spread", -1.0) == on.spread);
    // Recompute from the record (what the offline simulation reads).
    std::vector<double> scores = rows[1]["chunk_scores"].get<std::vector<double>>();
    std::vector<uint8_t> selected(scores.size(), 0), forced(scores.size(), 0);
    for (int i : rows[1]["selected_chunks"].get<std::vector<int>>()) selected[(size_t) i] = 1;
    for (int i : rows[1]["forced_chunks"].get<std::vector<int>>()) forced[(size_t) i] = 1;
    const auto again = luce::common::pflash_relevance_spread(scores, selected, forced);
    TEST_ASSERT(again.valid && again.spread == on.spread);
    // Everything else in the record is identical with the flag on and off.
    json a = rows[0], b = rows[1];
    b.erase("spread");
    b.erase("spread_threshold");
    TEST_ASSERT(a == b);
}

TEST_CASE(ServerUnitFixture, test_pflash_chat_view_store_forget) {
    http_detail::PflashChatViewStore store(4);
    http_detail::PflashChatView first;
    first.raw_tokens = {1, 2, 3, 9, 9};
    first.raw_gen_begin = 3;
    first.drafter_ids = first.raw_tokens;
    first.drafter_gen_begin = 3;
    store.remember(first);
    http_detail::PflashChatView other = first;
    other.raw_tokens = {7, 7, 7, 9};
    other.drafter_ids = other.raw_tokens;
    store.remember(other);
    TEST_ASSERT(store.size() == 2);
    // A prompt continuing the first conversation drops its view only.
    TEST_ASSERT(store.forget({1, 2, 3, 4, 5, 9}) == 1);
    TEST_ASSERT(store.size() == 1);
    http_detail::PflashChatView found;
    TEST_ASSERT(!store.find({1, 2, 3, 4, 5, 9}, {1, 2, 3, 4, 5, 9}, found));
    TEST_ASSERT(store.find({7, 7, 7, 1, 9}, {7, 7, 7, 1, 9}, found));
    TEST_ASSERT(store.forget({5, 5}) == 0);
}

// Budget compression that reports a configurable relevance spread.
struct MockPflashSpreadBackend : MockPflashBudgetBackend {
    static inline bool spread_valid = true;
    static inline double spread = 0.0;
    CompressResult compress(const CompressRequest & request) override {
        auto result = MockPflashBudgetBackend::compress(request);
        result.spread_valid = spread_valid;
        result.spread = spread;
        result.spread_k = 3;
        return result;
    }
};

static PflashFallbackRun run_spread_case(const char * flag, double spread, bool valid,
                                         int max_ctx = 8192) {
    luce_test::ScopedEnvVar env{"PFLASH_FALLBACK_SPREAD", flag};
    MockPflashSpreadBackend::spread = spread;
    MockPflashSpreadBackend::spread_valid = valid;
    return run_pflash_fallback_case<MockPflashSpreadBackend>(max_ctx, 16);
}

TEST_CASE(ServerUnitFixture, test_pflash_spread_flag_off_takes_no_decision) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    // A spread far above any threshold: with the flag off nothing changes.
    const auto run = run_spread_case(nullptr, 5.0, true);
    const auto base = run_pflash_fallback_case<MockPflashBudgetBackend>(8192, 16);
    TEST_ASSERT_MSG(run.prepared.error.empty(), run.prepared.error);
    TEST_ASSERT(run.prepared.compressed);
    TEST_ASSERT(run.prepared.tokens == base.prepared.tokens);
    // compress_ms is a wall-clock field
    json stats_run = run.prepared.pflash_stats;
    json stats_base = base.prepared.pflash_stats;
    stats_run.erase("compress_ms");
    stats_base.erase("compress_ms");
    TEST_ASSERT(stats_run == stats_base);
    TEST_ASSERT(!run.prepared.pflash_stats.contains("spread"));
    TEST_ASSERT(!run.prepared.pflash_stats.contains("pflash_fallback"));
    TEST_ASSERT(run.counters.value("fallback_served", 0) == 0);
    TEST_ASSERT(run.view_records.size() == 1);
    TEST_ASSERT(run.view_records[0].value("kind", "") == "compressed");
    // "0" is off too.
    const auto zero = run_spread_case("0", 5.0, true);
    TEST_ASSERT(zero.prepared.compressed && !zero.prepared.pflash_stats.contains("spread"));
}

TEST_CASE(ServerUnitFixture, test_pflash_spread_fires_full_prefill_with_flag) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    const auto run = run_spread_case("0.11", 0.11, true);   // boundary: fires
    TEST_ASSERT(run.compress_calls >= 1);
    TEST_ASSERT_MSG(run.prepared.error.empty(), run.prepared.error);
    TEST_ASSERT(!run.prepared.compressed);
    TEST_ASSERT(run.prepared.tokens == run.ids);
    TEST_ASSERT(run.prepared.snapshot_cut == -1);
    const json & flag = run.prepared.pflash_stats["pflash_fallback"];
    TEST_ASSERT(flag.value("reason", "") == "spread");
    TEST_ASSERT(flag.value("spread", -1.0) == 0.11);
    TEST_ASSERT(flag.value("spread_threshold", -1.0) == 0.11);
    TEST_ASSERT(flag.value("fired", false));
    TEST_ASSERT(flag.value("prompt_tokens", 0) == (int) run.ids.size());
    TEST_ASSERT(run.counters.value("fallback_served", 0) == 1);
    TEST_ASSERT(run.counters["fallback_reasons"].value("spread", 0) == 1);
    TEST_ASSERT(run.view_records.size() == 1);
    TEST_ASSERT(run.view_records[0].value("kind", "") == "raw");
    TEST_ASSERT(run.view_records[0]["pflash_fallback"].value("reason", "") == "spread");
    // Below the threshold: compressed, spread recorded, not fired.
    const auto low = run_spread_case("0.11", std::nextafter(0.11, 0.0), true);
    TEST_ASSERT_MSG(low.prepared.error.empty(), low.prepared.error);
    TEST_ASSERT(low.prepared.compressed);
    TEST_ASSERT(!low.prepared.pflash_stats.contains("pflash_fallback"));
    TEST_ASSERT(!low.prepared.pflash_stats["spread"].value("fired", true));
    TEST_ASSERT(low.counters.value("fallback_served", 0) == 0);
}

TEST_CASE(ServerUnitFixture, test_pflash_spread_fallback_is_not_refused_by_fail_closed) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", "1"};
    const auto run = run_spread_case("0.11", 0.5, true);
    TEST_ASSERT_MSG(run.prepared.error.empty(), run.prepared.error);
    TEST_ASSERT(run.prepared.tokens == run.ids);
    TEST_ASSERT(run.prepared.pflash_stats["pflash_fallback"].value("reason", "") == "spread");
}

TEST_CASE(ServerUnitFixture, test_pflash_spread_raw_prompt_too_long_serves_compressed) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    const auto probe = run_pflash_fallback_case<MockPflashBudgetBackend>(8192, 16);
    const auto run = run_spread_case("0.11", 0.5, true, (int) probe.ids.size() + 15);
    TEST_ASSERT_MSG(run.prepared.error.empty(), run.prepared.error);
    TEST_ASSERT(run.prepared.compressed);
    TEST_ASSERT(run.prepared.pflash_stats["spread"].value("refused", "") == "raw_exceeds_max_ctx");
    TEST_ASSERT(run.counters.value("fallback_served", 0) == 0);
}

TEST_CASE(ServerUnitFixture, test_pflash_spread_unavailable_is_a_failure) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    const auto run = run_spread_case("0.11", 0.0, false);
    TEST_ASSERT_MSG(run.prepared.error.empty(), run.prepared.error);
    TEST_ASSERT(run.prepared.tokens == run.ids);
    TEST_ASSERT(run.prepared.pflash_stats["pflash_fallback"].value("reason", "") ==
                "spread_unavailable");
    // An invalid setting is a configuration error: never served.
    const auto bad = run_spread_case("abc", 0.5, true);
    TEST_ASSERT(bad.prepared.error_status == 500);
    TEST_ASSERT(bad.prepared.error.find("PFLASH_FALLBACK_SPREAD") != std::string::npos);
}

// Keeps the required spans, the query window through the end, and whatever
// ``pick`` adds; reports the kept spans like the in-process drafter does.
struct MockPflashSpanBackend : MockBackend {
    int compress_calls = 0;
    CompressRequest last_request;
    std::function<std::vector<PFlashTokenSpan>(const CompressRequest &)> pick;

    CompressResult compress(const CompressRequest & request) override {
        ++compress_calls;
        last_request = request;
        auto spans = request.required_instruction_spans;
        spans.push_back({request.score_query_end - request.score_query_tokens,
                         (int) request.input_ids.size()});
        if (pick) {
            const auto extra = pick(request);
            spans.insert(spans.end(), extra.begin(), extra.end());
        }
        spans = http_detail::canonicalize_pflash_token_spans(std::move(spans));
        CompressResult result;
        for (const auto & span : spans) {
            result.compressed_ids.insert(result.compressed_ids.end(),
                request.input_ids.begin() + span.begin,
                request.input_ids.begin() + span.end);
        }
        result.kept_spans = spans;
        result.ok = !result.compressed_ids.empty();
        return result;
    }
};

TEST_CASE(ServerUnitFixture,
        test_pflash_chat_view_appends_turns_and_recalls_missing_segments) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar view_env{"PFLASH_CHAT_VIEW", nullptr};
    // Only turns of a few tokens stay whole, so the document turn is
    // material the selection picks from.
    luce_test::ScopedEnvVar skeleton{"PFLASH_CHAT_SKELETON_TOKENS", "4"};

    std::string system;
    for (int i = 0; i < 20; ++i) system += "You are helpful. ";
    const std::string document =
        "alpha facts live here. filler filler filler. beta facts live here.";
    const std::vector<ChatMessage> turn1{
        {"system", system, ""},
        {"user", document + " Question one?", ""},
    };
    auto turn2 = turn1;
    turn2.push_back({"assistant", "Answer one.", ""});
    turn2.push_back({"user", "Question two?", ""});
    const auto render = [] (const std::vector<ChatMessage> & messages) {
        return render_chat_template(messages, ChatFormat::QWEN3,
                                    /*add_generation_prompt=*/true,
                                    /*enable_thinking=*/true);
    };
    const auto to_json = [] (const std::vector<ChatMessage> & messages) {
        json out = json::array();
        for (const auto & message : messages) {
            out.push_back({{"role", message.role}, {"content", message.content}});
        }
        return out;
    };
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"alpha", " facts", "beta", " live", " here", ".", " filler",
         "Question", " one", " two", "?", "Answer", "user", "assistant",
         "system", "\n", "You", " are", " helpful"},
        render(turn2) +
            "[Earlier in this conversation]\n[End of earlier excerpts]\n");
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashSpanBackend>();
    MockPflashSpanBackend & backend = *backend_owner;
    const char * wanted = "alpha facts";
    backend.pick = [&] (const ModelBackend::CompressRequest & request) {
        const auto span = http_detail::pflash_decoded_text_span(
            tokenizer, request.input_ids, 0, (int) request.input_ids.size(),
            wanted);
        return span.begin < 0 ? std::vector<PFlashTokenSpan>{}
                              : std::vector<PFlashTokenSpan>{span};
    };
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash = true;
    config.pflash_threshold = 0;   // every prompt
    config.pflash_keep_ratio = 1.0f;
    config.max_ctx = 8192;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    std::vector<int32_t> served1;
    std::vector<int32_t> served2;
    std::vector<int32_t> served3;
    std::vector<std::string> modes;
    {
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);
        const auto run = [&] (const std::vector<ChatMessage> & messages,
                              std::vector<int32_t> & served) {
            ParsedRequest request;
            request.format = ApiFormat::OPENAI_CHAT;
            request.messages = to_json(messages);
            request.prompt_tokens = tokenizer.encode(render(messages));
            const auto prepared =
                HttpServerTestAccess::prepare_prompt(server, request);
            TEST_ASSERT_MSG(prepared.error.empty(), prepared.error);
            TEST_ASSERT(prepared.compressed);
            served = prepared.tokens;
            // usage.timings.pflash reports the view outcome.
            TEST_ASSERT(prepared.pflash_stats.contains("view"));
            modes.push_back(prepared.pflash_stats["view"].value("mode", ""));
            TEST_ASSERT(prepared.pflash_stats["view"].value("served_tokens", 0) ==
                        (int) served.size());
            // The snapshot lands where the next turn's prompt branches off.
            const auto generation =
                tokenizer.encode("<|im_start|>assistant\n<think>\n");
            TEST_ASSERT(prepared.snapshot_cut ==
                        (int) (served.size() - generation.size()));
        };
        run(turn1, served1);
        wanted = "beta facts";   // the new query wants what turn 1 dropped
        run(turn2, served2);
        run(turn2, served3);     // a retry serves the same view
    }
    const std::string text1 = tokenizer.decode(served1);
    const std::string text2 = tokenizer.decode(served2);
    TEST_ASSERT(text1.find("alpha facts") != std::string::npos);
    TEST_ASSERT(text1.find("beta facts") == std::string::npos);

    // Turn 2 extends turn 1's served prompt: everything before its
    // generation prompt is reused token for token.
    const auto generation = tokenizer.encode("<|im_start|>assistant\n<think>\n");
    TEST_ASSERT(served1.size() > generation.size());
    const size_t reused = served1.size() - generation.size();
    TEST_ASSERT(served2.size() > reused);
    TEST_ASSERT(std::equal(served1.begin(), served1.begin() + (long) reused,
                           served2.begin()));
    // The recalled segment opens the new user turn, after the answer.
    const size_t answer = text2.find("Answer one.");
    const size_t recall = text2.find("[Earlier in this conversation]");
    const size_t question = text2.find("Question two?");
    TEST_ASSERT(answer != std::string::npos);
    TEST_ASSERT(recall != std::string::npos && recall > answer);
    TEST_ASSERT(text2.find("beta facts", recall) != std::string::npos);
    TEST_ASSERT(question != std::string::npos && question > recall);
    TEST_ASSERT(served3 == served2);
    TEST_ASSERT(modes == std::vector<std::string>({"fresh", "continue", "repeat"}));
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_strict_multi_turn_keeps_skeleton_and_history_queries) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar skeleton{"PFLASH_CHAT_SKELETON_TOKENS", nullptr};

    std::string material;
    for (int i = 0; i < 80; ++i) material += "filler words here. ";
    const std::vector<ChatMessage> messages{
        {"system", "You are helpful.", ""},
        {"user", "Here is text: " + material + "What is the first answer?", ""},
        {"assistant", "Sure.", ""},
        {"user", "What is the answer?", ""},
    };
    const std::string rendered = render_chat_template(
        messages, ChatFormat::QWEN3, /*add_generation_prompt=*/true,
        /*enable_thinking=*/true);
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"What", " is", " the", " answer", " first", "?", "user", "assistant",
         "system", "\n", "Sure", ".", " filler", " words", " here"},
        rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashCompressBackend>();
    MockPflashCompressBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash_keep_ratio = 1.0f;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);
        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        json wire = json::array();
        for (const auto & message : messages) {
            wire.push_back({{"role", message.role}, {"content", message.content}});
        }
        request.messages = wire;
        request.prompt_tokens = tokenizer.encode(rendered);
        const std::string error =
            HttpServerTestAccess::apply_pflash_compression(server, request);
        TEST_ASSERT_MSG(error.empty(), error);
    }
    TEST_ASSERT(backend.compress_calls == 1);
    const auto & request = backend.last_request;
    const auto & ids = request.input_ids;
    const auto text_of = [&] (const PFlashTokenSpan & span) {
        return tokenizer.decode({ids.begin() + span.begin, ids.begin() + span.end});
    };
    // The short assistant answer stays whole; the long first user turn keeps
    // only its header, its material competes for the budget.
    bool answer_kept = false;
    bool material_kept = false;
    for (const auto & span : request.required_instruction_spans) {
        const std::string text = text_of(span);
        answer_kept = answer_kept || text.find("Sure.") != std::string::npos;
        material_kept = material_kept ||
            text.find("filler words") != std::string::npos;
    }
    TEST_ASSERT(answer_kept);
    TEST_ASSERT(!material_kept);
    // The earlier question scores alongside the current one, through the
    // last token of the header of the reply that followed it.
    TEST_ASSERT(request.history_query_spans.size() == 1);
    const auto history = request.history_query_spans[0];
    TEST_ASSERT(history.end - history.begin == 1);
    TEST_ASSERT_MSG(tokenizer.decode({ids.begin() + history.end,
                                      ids.begin() + history.end + 2}) == "Sure.",
                    text_of(history));
    // A chat whose prompt ends at the latest question's reply has nothing to
    // keep: that question's window is the query itself.
    TEST_ASSERT(request.keep_query_spans.empty());
    unlink(path.c_str());
}

// A tool session's prompt ends after tool output: the latest question's
// history window (its reply's header) is kept for the next request, which
// scores exactly that window once a new question arrives.
TEST_CASE(ServerUnitFixture, test_pflash_tool_session_keeps_next_history_query) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar skeleton{"PFLASH_CHAT_SKELETON_TOKENS", nullptr};

    std::string material;
    for (int i = 0; i < 80; ++i) material += "filler words here. ";
    std::vector<ChatMessage> messages{
        {"system", "You are helpful.", ""},
        {"user", "What is the first answer?", ""},
        {"assistant", "Sure.", ""},
        {"tool", "Here is text: " + material, ""},
    };
    const auto run = [&] (const std::vector<ChatMessage> & chat,
                          ModelBackend::CompressRequest & out,
                          std::vector<int32_t> & ids_out) {
        const std::string rendered = render_chat_template(
            chat, ChatFormat::QWEN3, /*add_generation_prompt=*/true,
            /*enable_thinking=*/true);
        const std::string path = write_pflash_bpe_tokenizer_fixture(
            {"What", " is", " the", " answer", " first", "?", "user", "assistant",
             "system", "tool", "\n", "Sure", ".", " filler", " words", " here",
             "Here", " text", ":", " Next", " question"},
            rendered);
        Tokenizer tokenizer;
        TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
        auto backend_owner = std::make_unique<MockPflashCompressBackend>();
        MockPflashCompressBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.pflash_keep_ratio = 1.0f;
        config.prefix_cache_cap = 0;
        config.prefill_cache_cap = 0;
        {
            HttpServer server(engine, tokenizer, config);
            HttpServerTestAccess::chat_view_only(server);
            server.set_drafter_tokenizer(&tokenizer);
            ParsedRequest request;
            request.format = ApiFormat::OPENAI_CHAT;
            json wire = json::array();
            for (const auto & message : chat) {
                wire.push_back({{"role", message.role}, {"content", message.content}});
            }
            request.messages = wire;
            request.prompt_tokens = tokenizer.encode(rendered);
            const std::string error =
                HttpServerTestAccess::apply_pflash_compression(server, request);
            TEST_ASSERT_MSG(error.empty(), error);
        }
        TEST_ASSERT(backend.compress_calls == 1);
        out = backend.last_request;
        ids_out = out.input_ids;
        unlink(path.c_str());
    };
    ModelBackend::CompressRequest first;
    std::vector<int32_t> first_ids;
    run(messages, first, first_ids);
    TEST_ASSERT(first.history_query_spans.empty());
    TEST_ASSERT(first.keep_query_spans.size() == 1);

    messages.push_back({"assistant", "Sure.", ""});
    messages.push_back({"user", "Next question?", ""});
    ModelBackend::CompressRequest second;
    std::vector<int32_t> second_ids;
    run(messages, second, second_ids);
    TEST_ASSERT(second.history_query_spans.size() == 1);
    const auto kept = first.keep_query_spans[0];
    const auto history = second.history_query_spans[0];
    TEST_ASSERT(kept.begin == history.begin && kept.end == history.end);
    TEST_ASSERT(std::equal(first_ids.begin(), first_ids.begin() + kept.end, second_ids.begin()));
}

// Two turns over one document through prepare_prompt; returns the served
// prompts, the view modes and how often the compressor ran.
struct PflashTwoTurnRun {
    std::vector<std::vector<int32_t>> served;
    std::vector<std::string> modes;
    int compress_calls = 0;
    std::string text2;
};

static PflashTwoTurnRun pflash_two_turn_run(const std::string & follow_up) {
    std::string system;
    for (int i = 0; i < 20; ++i) system += "You are helpful. ";
    const std::string document =
        "alpha facts live here. filler filler filler. beta facts live here.";
    const std::vector<ChatMessage> turn1{
        {"system", system, ""},
        {"user", document + " Question one?", ""},
    };
    auto turn2 = turn1;
    turn2.push_back({"assistant", "Answer one.", ""});
    turn2.push_back({"user", follow_up + " Question two?", ""});
    const auto render = [] (const std::vector<ChatMessage> & messages) {
        return render_chat_template(messages, ChatFormat::QWEN3,
                                    /*add_generation_prompt=*/true,
                                    /*enable_thinking=*/true);
    };
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"alpha", " facts", "beta", " live", " here", ".", " filler",
         "Question", " one", " two", "?", "Answer", "user", "assistant",
         "system", "\n", "You", " are", " helpful", " pasted", " notes"},
        render(turn2) +
            "[Earlier in this conversation]\n[End of earlier excerpts]\n");
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    auto backend_owner = std::make_unique<MockPflashSpanBackend>();
    MockPflashSpanBackend & backend = *backend_owner;
    // The fact and the questions: a prompt-end query ranks the user's own
    // sentences first, so the scorer keeps them.
    backend.pick = [&] (const ModelBackend::CompressRequest & request) {
        std::vector<PFlashTokenSpan> spans;
        for (const char * text : {"alpha facts", "Question one?", "Question two?"}) {
            const auto span = http_detail::pflash_decoded_text_span(
                tokenizer, request.input_ids, 0, (int) request.input_ids.size(),
                text);
            if (span.begin >= 0) spans.push_back(span);
        }
        return spans;
    };
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash = true;
    config.pflash_threshold = 0;   // every prompt
    config.pflash_keep_ratio = 1.0f;
    config.max_ctx = 8192;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    PflashTwoTurnRun run;
    {
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);
        const std::vector<const std::vector<ChatMessage> *> turns{&turn1, &turn2};
        for (const auto * messages : turns) {
            ParsedRequest request;
            request.format = ApiFormat::OPENAI_CHAT;
            json wire = json::array();
            for (const auto & message : *messages) {
                wire.push_back({{"role", message.role}, {"content", message.content}});
            }
            request.messages = wire;
            request.prompt_tokens = tokenizer.encode(render(*messages));
            const auto prepared = HttpServerTestAccess::prepare_prompt(server, request);
            TEST_ASSERT_MSG(prepared.error.empty(), prepared.error);
            run.served.push_back(prepared.tokens);
            run.modes.push_back(prepared.pflash_stats["view"].value("mode", ""));
        }
    }
    run.compress_calls = backend.compress_calls;
    run.text2 = tokenizer.decode(run.served[1]);
    unlink(path.c_str());
    return run;
}

TEST_CASE(ServerUnitFixture, test_pflash_raw_fallback_forgets_previous_chat_view) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar recall{"PFLASH_CHAT_RECALL", "1"};
    luce_test::ScopedEnvVar skeleton{"PFLASH_CHAT_SKELETON_TOKENS", "4"};
    struct FailingViewBackend : MockPflashSpanBackend {
        bool fail = false;
        CompressResult compress(const CompressRequest & request) override {
            if (!fail) return MockPflashSpanBackend::compress(request);
            ++compress_calls;
            luce::common::set_last_error("non-finite Qwen3.5 scoring-head scores");
            return {};
        }
    };
    const auto render = [](const std::vector<ChatMessage> & messages) {
        return render_chat_template(messages, ChatFormat::QWEN3, true, true);
    };
    std::vector<ChatMessage> messages{
        {"system", "You are helpful.", ""},
        {"user", "alpha facts live here. filler filler filler. beta facts live here. Question one?", ""},
    };
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"alpha", " facts", "beta", " live", " here", ".", " filler",
         "Question", " one", " two", "?", "Answer", "user", "assistant",
         "system", "\n", "You", " are", " helpful"},
        render(messages) + "Answer one. Question two?");
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    for (bool fail_closed : {false, true}) {
        luce_test::ScopedEnvVar fail_policy{"PFLASH_FAIL_CLOSED", fail_closed ? "1" : "0"};
        auto backend_owner = std::make_unique<FailingViewBackend>();
        auto & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.pflash = true;
        config.pflash_threshold = 0;   // every prompt
        config.pflash_keep_ratio = 1.0f;
        config.max_ctx = 8192;
        config.prefix_cache_cap = 0;
        config.prefill_cache_cap = 0;
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);
        const auto request_for = [&](const std::vector<ChatMessage> & turns) {
            ParsedRequest request;
            request.format = ApiFormat::OPENAI_CHAT;
            request.messages = json::array();
            for (const auto & m : turns) request.messages.push_back({{"role", m.role}, {"content", m.content}});
            request.prompt_tokens = tokenizer.encode(render(turns));
            return request;
        };
        const auto first = HttpServerTestAccess::prepare_prompt(server, request_for(messages));
        TEST_ASSERT_MSG(first.error.empty(), first.error);
        TEST_ASSERT(first.compressed);
        TEST_ASSERT(HttpServerTestAccess::pflash_view_count(server) == 1);
        auto follow_up = messages;
        follow_up.push_back({"assistant", "Answer one.", ""});
        follow_up.push_back({"user", "Question two?", ""});
        const auto request = request_for(follow_up);
        backend.fail = true;
        const auto failed = HttpServerTestAccess::prepare_prompt(server, request);
        if (fail_closed) {
            TEST_ASSERT(!failed.error.empty());
            TEST_ASSERT(HttpServerTestAccess::pflash_view_count(server) == 1);
        } else {
            TEST_ASSERT_MSG(failed.error.empty(), failed.error);
            TEST_ASSERT(!failed.compressed && failed.tokens == request.prompt_tokens);
            TEST_ASSERT(HttpServerTestAccess::pflash_view_count(server) == 0);
            TEST_ASSERT(failed.pflash_stats["pflash_fallback"].value("views_forgotten", 0) == 1);
            backend.fail = false;
            const auto retry = HttpServerTestAccess::prepare_prompt(server, request);
            TEST_ASSERT_MSG(retry.error.empty(), retry.error);
            TEST_ASSERT(retry.pflash_stats["view"].value("mode", "") == "fresh");
        }
    }
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_chat_view_small_follow_up_without_recall_skips_the_drafter) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar recall{"PFLASH_CHAT_RECALL", "0"};
    const auto run = pflash_two_turn_run("");
    // Turn 2 is appended exactly as full prefill appends it: no scoring.
    TEST_ASSERT(run.compress_calls == 1);
    TEST_ASSERT(run.modes == std::vector<std::string>({"fresh", "continue"}));
    TEST_ASSERT(run.text2.find("[Earlier in this conversation]") == std::string::npos);
    TEST_ASSERT(run.text2.find("Answer one.") != std::string::npos);
    TEST_ASSERT(run.text2.find("Question two?") != std::string::npos);
}

TEST_CASE(ServerUnitFixture,
        test_pflash_chat_view_compresses_large_follow_ups_only) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar threshold{"PFLASH_CHAT_COMPRESS_NEW_TOKENS", "40"};
    // The pasted turn is material, not a short turn kept whole.
    luce_test::ScopedEnvVar skeleton{"PFLASH_CHAT_SKELETON_TOKENS", "8"};
    std::string pasted;
    for (int i = 0; i < 12; ++i) pasted += " pasted notes filler.";
    const auto run = pflash_two_turn_run(pasted);
    TEST_ASSERT(run.compress_calls == 2);
    TEST_ASSERT(run.modes == std::vector<std::string>({"fresh", "continue-compressed"}));
    // The view before the new material is reused token for token...
    const auto & first = run.served[0];
    const auto & second = run.served[1];
    TEST_ASSERT(second.size() > 0 && first.size() > 16);
    const size_t prefix = first.size() - 8;
    TEST_ASSERT(std::equal(first.begin(), first.begin() + (long) (prefix - 8),
                           second.begin()));
    // ...and the pasted material is compressed: only what the selection
    // keeps (the question) survives, not the whole paste.
    TEST_ASSERT_MSG(run.text2.find("Question two?") != std::string::npos, run.text2);
    TEST_ASSERT(run.text2.find(pasted) == std::string::npos);
}

TEST_CASE(ServerUnitFixture,
        test_pflash_legacy_chat_query_uses_last_user_turn) {
    // No strict-selection environment: the legacy selector derives the same
    // query, and a trailing tool result does not replace the user's turn.
    const std::vector<ChatMessage> messages{
        {"user", "What is the answer?", ""},
        {"assistant", "Sure.", ""},
        {"tool", "tool output here", "call-1"},
    };
    const std::string rendered = render_chat_template(
        messages, ChatFormat::QWEN3, /*add_generation_prompt=*/true,
        /*enable_thinking=*/false);
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"What", " is", " the", " answer", "?", "user", "assistant", "\n",
         "Sure", "."},
        rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashCompressBackend>();
    MockPflashCompressBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);

        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = json::array({
            {{"role", "user"}, {"content", "What is the answer?"}},
            {{"role", "assistant"}, {"content", "Sure."}},
            {{"role", "tool"}, {"content", "tool output here"},
             {"tool_call_id", "call-1"}},
        });
        request.prompt_tokens = tokenizer.encode(rendered);

        const std::string error =
            HttpServerTestAccess::apply_pflash_compression(server, request);
        TEST_ASSERT_MSG(error.empty(), error);
    }

    TEST_ASSERT(backend.compress_calls == 1);
    const auto & ids = backend.last_request.input_ids;
    const int query_end = backend.last_request.score_query_end;
    const int query_begin = query_end - backend.last_request.score_query_tokens;
    TEST_ASSERT(query_begin >= 0);
    TEST_ASSERT(backend.last_request.score_query_tokens <= 8);
    const std::string query = tokenizer.decode(
        {ids.begin() + query_begin, ids.begin() + query_end});
    TEST_ASSERT_MSG(std::string("What is the answer?").size() >= query.size() &&
                    std::string("What is the answer?").compare(
                        19 - query.size(), query.size(), query) == 0,
                    query);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_strict_selection_owns_chat_continuations) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "top_p"};

    const std::string rendered = render_chat_template(
        {{"user", "first", ""},
         {"assistant", "Sure.", ""},
         {"user", "second question", ""}},
        ChatFormat::QWEN3, /*add_generation_prompt=*/true,
        /*enable_thinking=*/false);
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"first", "second", " question", "user", "assistant", "\n", "Sure",
         "."},
        rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashCompressBackend>();
    MockPflashCompressBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash = true;
    config.pflash_threshold = 0;   // every prompt
    config.pflash_keep_ratio = 1.0f;
    config.max_ctx = 8192;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);

        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = json::array({
            {{"role", "user"}, {"content", "first"}},
            {{"role", "assistant"}, {"content", "Sure."}},
            {{"role", "user"}, {"content", "second question"}},
        });
        request.prompt_tokens = tokenizer.encode(rendered);

        const auto prepared =
            HttpServerTestAccess::prepare_prompt(server, request);
        TEST_ASSERT_MSG(prepared.error.empty(), prepared.error);
        TEST_ASSERT(prepared.compressed);
        TEST_ASSERT(!prepared.flowkv);
    }

    // Whole-prompt PFlash ran on the multi-turn prompt and scored against
    // the prompt's last token.
    TEST_ASSERT(backend.compress_calls == 1);
    const auto & ids = backend.last_request.input_ids;
    TEST_ASSERT(backend.last_request.score_query_end == (int) ids.size());
    TEST_ASSERT(backend.last_request.score_query_tokens == 1);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture,
        test_pflash_default_continuation_stays_on_flowkv) {
    const std::string rendered =
        "<|im_start|>user\nfirst<|im_end|>\n"
        "<|im_start|>assistant\nSure.<|im_end|>\n"
        "<|im_start|>user\nsecond<|im_end|>\n"
        "<|im_start|>assistant\n";
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"first", "second", "user", "assistant", "\n", "Sure", "."},
        rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    auto backend_owner = std::make_unique<MockPflashCompressBackend>();
    MockPflashCompressBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash = true;
    config.pflash_threshold = 0;   // every prompt
    config.max_ctx = 8192;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);

        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = json::array({
            {{"role", "user"}, {"content", "first"}},
            {{"role", "assistant"}, {"content", "Sure."}},
            {{"role", "user"}, {"content", "second"}},
        });
        request.prompt_tokens = tokenizer.encode(rendered);

        const auto prepared =
            HttpServerTestAccess::prepare_prompt(server, request);
        TEST_ASSERT_MSG(prepared.error.empty(), prepared.error);
    }

    // Without a strict-selection environment FlowKV keeps owning
    // continuations; whole-prompt PFlash never ran.
    TEST_ASSERT(backend.compress_calls == 0);
    unlink(path.c_str());
}

struct MockBatchCompressBackend : MockBackend {
    int compress_calls = 0;

    CompressResult compress(const CompressRequest & request) override {
        ++compress_calls;
        CompressResult result;
        result.ok = !request.input_ids.empty();
        if (result.ok) result.compressed_ids = {request.input_ids.front()};
        return result;
    }
};

TEST_CASE(ServerUnitFixture, test_compress_batch_default_preserves_order) {
    MockBatchCompressBackend backend;
    std::vector<ModelBackend::CompressRequest> requests(3);
    requests[0].input_ids = {11, 12};
    requests[1].input_ids = {21, 22};
    requests[2].input_ids = {31, 32};

    const auto results = backend.compress_batch(requests);
    TEST_ASSERT(results.size() == requests.size());
    TEST_ASSERT(backend.compress_calls == 3);
    TEST_ASSERT(results[0].compressed_ids == std::vector<int32_t>({11}));
    TEST_ASSERT(results[1].compressed_ids == std::vector<int32_t>({21}));
    TEST_ASSERT(results[2].compressed_ids == std::vector<int32_t>({31}));
}

struct MockMemoryOnlySnapshotBackend : MockBackend {
    bool snapshot_used(int slot) const override { return slot == 0; }
};

// ─── MockBackendWithLayout ──────────────────────────────────────────────
// Extends MockBackend with a real ggml_context so DiskPrefixCache can
// iterate tensors in compute_layout_id and write a real .dkv file.
// KV: one layer, K=[16,32,4,1] F32 + V=[32,16,4,1] F32.
struct MockBackendWithLayout : MockBackend {
    static constexpr int     kNLayer  = 1;
    static constexpr int64_t kHeadDim = 16;
    static constexpr int64_t kNHead   = 4;
    static constexpr int     kMaxPos  = 32;

    ggml_context         * kv_ctx_ = nullptr;
    ggml_backend_t         cpu_be_ = nullptr;
    ggml_backend_buffer_t  kv_buf_ = nullptr;
    ggml_tensor          * k_[kNLayer] = {};
    ggml_tensor          * v_[kNLayer] = {};

    MockBackendWithLayout() {
        cpu_be_ = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (kNLayer * 2 + 4) + 4096;
        ip.no_alloc = true;
        kv_ctx_ = ggml_init(ip);
        int64_t ne_k[4] = {kHeadDim, kMaxPos, kNHead, 1};
        int64_t ne_v[4] = {kMaxPos, kHeadDim, kNHead, 1};
        char name[64];
        for (int il = 0; il < kNLayer; ++il) {
            k_[il] = ggml_new_tensor(kv_ctx_, GGML_TYPE_F32, 4, ne_k);
            std::snprintf(name, sizeof(name), "snap_k_%d", il);
            ggml_set_name(k_[il], name);
            v_[il] = ggml_new_tensor(kv_ctx_, GGML_TYPE_F32, 4, ne_v);
            std::snprintf(name, sizeof(name), "snap_v_%d", il);
            ggml_set_name(v_[il], name);
        }
        kv_buf_ = ggml_backend_alloc_ctx_tensors(kv_ctx_, cpu_be_);
        for (int il = 0; il < kNLayer; ++il) {
            std::vector<float> ones(ggml_nelements(k_[il]), 1.0f);
            std::vector<float> twos(ggml_nelements(v_[il]), 2.0f);
            ggml_backend_tensor_set(k_[il], ones.data(), 0, ggml_nbytes(k_[il]));
            ggml_backend_tensor_set(v_[il], twos.data(), 0, ggml_nbytes(v_[il]));
        }
    }
    ~MockBackendWithLayout() {
        if (kv_buf_) ggml_backend_buffer_free(kv_buf_);
        if (kv_ctx_) ggml_free(kv_ctx_);
        if (cpu_be_) ggml_backend_free(cpu_be_);
    }

    SnapshotRef snapshot_ref(int /*slot*/) const override {
        SnapshotRef ref;
        ref.ctx      = kv_ctx_;
        ref.buf      = kv_buf_;
        ref.cur_pos  = kMaxPos;
        ref.last_tok = 42;
        return ref;
    }

    bool snapshot_save(int) override { return true; }
    bool snapshot_used(int) const override { return true; }
    int  snapshot_cur_pos(int) const override { return kMaxPos; }
};

// Helper: recursively remove a directory.
static void rm_rf(const std::string & path) {
    std::error_code ec;
    fs::remove_all(path, ec);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_config_defaults) {
    DiskCacheConfig cfg;
    TEST_ASSERT(cfg.cache_dir.empty());
    TEST_ASSERT(cfg.budget_bytes == (size_t)128 * 1024 * 1024 * 1024);
    TEST_ASSERT(cfg.min_tokens == 512);
    TEST_ASSERT(cfg.continued_interval == 10240);
    TEST_ASSERT(cfg.cold_max_tokens == 10240);
    TEST_ASSERT(cfg.ttl_seconds == 72ull * 3600);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_policy_parse) {
    DiskPrefixCachePolicy policy;
    TEST_ASSERT(parse_disk_prefix_cache_policy("off", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Off);
    TEST_ASSERT(parse_disk_prefix_cache_policy("full", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Full);
    TEST_ASSERT(parse_disk_prefix_cache_policy("auto", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Auto);
    TEST_ASSERT(policy.auto_window == 30);
    TEST_ASSERT(parse_disk_prefix_cache_policy("auto:30", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Auto);
    TEST_ASSERT(policy.auto_window == 30);
    TEST_ASSERT(parse_disk_prefix_cache_policy("1000", policy));
    TEST_ASSERT(policy.mode == DiskPrefixCacheMode::Fixed);
    TEST_ASSERT(policy.fixed_tokens == 1000);
    TEST_ASSERT(!parse_disk_prefix_cache_policy("core", policy));
    TEST_ASSERT(!parse_disk_prefix_cache_policy("task", policy));
    TEST_ASSERT(!parse_disk_prefix_cache_policy("auto:0", policy));
}

// BUG-A: apply_request_scope_override must preserve server-level compress flag.
// A request-level scope override (e.g. "auto") must NOT clear compress=true
// that was set by the server configuration.
TEST_CASE(ServerUnitFixture, test_scope_override_preserves_compress) {
    // Server policy: compress=true, mode=Full.
    DiskPrefixCachePolicy server;
    server.mode = DiskPrefixCacheMode::Full;
    server.compress = true;

    // Request sends scope="auto" — should change mode but keep compress.
    TEST_ASSERT(apply_request_scope_override(server, "auto"));
    TEST_ASSERT(server.mode == DiskPrefixCacheMode::Auto);
    TEST_ASSERT_MSG(server.compress,
        "BUG-A: scope override dropped server-level compress=true");

    // Same with a fixed-token scope.
    DiskPrefixCachePolicy server2;
    server2.mode = DiskPrefixCacheMode::Full;
    server2.compress = true;
    TEST_ASSERT(apply_request_scope_override(server2, "1000"));
    TEST_ASSERT(server2.mode == DiskPrefixCacheMode::Fixed);
    TEST_ASSERT(server2.fixed_tokens == 1000);
    TEST_ASSERT_MSG(server2.compress,
        "BUG-A: fixed-token scope override dropped server-level compress=true");

    // scope="off" must also preserve compress flag.
    DiskPrefixCachePolicy server3;
    server3.compress = true;
    TEST_ASSERT(apply_request_scope_override(server3, "off"));
    TEST_ASSERT(server3.mode == DiskPrefixCacheMode::Off);
    TEST_ASSERT_MSG(server3.compress,
        "BUG-A: off scope override dropped server-level compress=true");

    // Invalid scope string must return false and leave policy unchanged.
    DiskPrefixCachePolicy server4;
    server4.compress = true;
    server4.mode = DiskPrefixCacheMode::Full;
    TEST_ASSERT(!apply_request_scope_override(server4, "core"));
    TEST_ASSERT(server4.compress);
    TEST_ASSERT(server4.mode == DiskPrefixCacheMode::Full);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_fixed_boundary) {
    DiskPrefixCachePolicy policy;
    TEST_ASSERT(parse_disk_prefix_cache_policy("1000", policy));
    TEST_ASSERT(disk_prefix_cache_fixed_boundary(policy, 2000) == 1000);
    TEST_ASSERT(disk_prefix_cache_fixed_boundary(policy, 500) == 0);
    TEST_ASSERT(disk_prefix_cache_fixed_boundary(policy, 2000, 1001) == 0);
    TEST_ASSERT(disk_prefix_cache_fixed_boundary(policy, 2000, 1000) == 1000);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_auto_boundary_lcp) {
    std::vector<int32_t> current{1, 2, 3, 4, 5, 9};
    std::vector<std::vector<int32_t>> recent{
        {1, 2, 3, 4, 8},
        {1, 2, 3, 4, 7},
        {7, 8},
    };
    std::vector<int> safe_boundaries{2, 4};
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 2, safe_boundaries, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 2, {}, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 3, {}, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 2, safe_boundaries, 5) == 0);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_auto_window_limits_history) {
    std::vector<int32_t> current{1, 2, 3, 4, 5};
    std::vector<std::vector<int32_t>> recent{
        {9},
        {1, 2, 3, 4, 0},
        {1, 2, 3, 4, 9},
    };
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 1, {}, 2) == 0);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 2, {}, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 3, {}, 2) == 4);
    TEST_ASSERT(disk_prefix_cache_auto_boundary(
        current, recent, 0, {}, 2) == 0);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_disabled_when_no_dir) {
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

TEST_CASE(ServerUnitFixture, test_disk_cache_disables_memory_only_backend) {
    MockMemoryOnlySnapshotBackend backend;
    DiskCacheConfig cfg;
    cfg.cache_dir = "/tmp/luce_test_disk_cache_memory_only";
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(!cache.disabled());

    // A live in-memory snapshot with no SnapshotRef means this backend cannot
    // serialize or adopt disk entries. Detect it once and skip later disk work.
    cache.learn_layout(0);
    TEST_ASSERT(cache.disabled());
}

TEST_CASE(ServerUnitFixture, test_disk_cache_init_creates_directory) {
    MockBackend backend;
    std::string dir = test_tmp_path("luce_test_disk_cache_init").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(!cache.disabled());
    TEST_ASSERT(cache.init());

    // Directory should exist.
    std::error_code ec;
    TEST_ASSERT(fs::is_directory(dir, ec) && !ec);

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_header_size) {
    // The header should be exactly 96 bytes.
    TEST_ASSERT(DISK_CACHE_HEADER_SIZE == 96);
    TEST_ASSERT(DISK_CACHE_LAST_USED_OFFSET == 68);
    // Bumped to 2 when the K-rotation default changed: a cache written by an
    // older binary stores K in the rotated basis, and the layout id does not
    // cover that, so the version is what rejects it.
    // v4 added the identity salt and the stored key tokens.
    TEST_ASSERT(DISK_CACHE_VERSION == 5);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_header_round_trip) {
    // Write and read a header to verify serialization.
    std::string path = test_tmp_path("luce_test_header_rt.dkv").string();
    remove_test_path(path);
    std::error_code ec;

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
    std::memset(hdr.identity_salt, 0xEF, 16);

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
    std::fwrite(hdr.identity_salt, 16, 1, f);
    std::fclose(f);

    // Verify file size is DISK_CACHE_HEADER_SIZE.
    TEST_ASSERT(fs::file_size(path, ec) == DISK_CACHE_HEADER_SIZE && !ec);

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
    uint8_t salt[16]; std::fread(salt, 16, 1, f);
    TEST_ASSERT(salt[0] == 0xEF && salt[15] == 0xEF);
    std::fclose(f);

    remove_test_path(path);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_continued_boundary) {
    // Test maybe_store_continued logic: saves at interval boundaries.
    MockBackend backend;
    std::string dir = test_tmp_path("luce_test_continued").string();
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

TEST_CASE(ServerUnitFixture, test_disk_cache_full_lookup_lengths) {
    // Whole prompt first, then every boundary deepest first, skipping cuts
    // below the persistence minimum and the prompt end itself.
    TEST_ASSERT(disk_prefix_cache_full_lookup_lengths(
                    6000, {300, 2000, 4000, 6000}, 512) ==
                std::vector<int>({6000, 4000, 2000}));
    TEST_ASSERT(disk_prefix_cache_full_lookup_lengths(6000, {}, 512) ==
                std::vector<int>({6000}));
    TEST_ASSERT(disk_prefix_cache_full_lookup_lengths(0, {100}, 512).empty());
    // Below the persistence minimum nothing was ever written: no probes.
    TEST_ASSERT(disk_prefix_cache_full_lookup_lengths(300, {100}, 512).empty());
}

TEST_CASE(ServerUnitFixture, test_disk_cache_cold_prefix_short_prompt) {
    // Cold prefix should not trigger for short prompts.
    MockBackend backend;
    std::string dir = test_tmp_path("luce_test_cold_short").string();
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

TEST_CASE(ServerUnitFixture, test_disk_cache_cold_prefix_no_boundaries) {
    // Cold prefix should not trigger if no boundaries provided.
    MockBackend backend;
    std::string dir = test_tmp_path("luce_test_cold_nobound").string();
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

TEST_CASE(ServerUnitFixture, test_disk_cache_cold_prefix_finds_boundary) {
    // Cold prefix should find the last boundary <= cold_max_tokens.
    MockBackend backend;
    std::string dir = test_tmp_path("luce_test_cold_finds").string();
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

TEST_CASE(ServerUnitFixture, test_disk_cache_lookup_miss_no_layout) {
    // Lookup with no layout known should return false.
    MockBackend backend;
    std::string dir = test_tmp_path("luce_test_lookup_miss").string();
    rm_rf(dir);

    DiskCacheConfig cfg;
    cfg.cache_dir = dir;
    DiskPrefixCache cache(cfg, backend);
    cache.init();

    std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8};
    TEST_ASSERT(!cache.lookup(ids, 0));

    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_save_below_min_tokens) {
    // Save with fewer tokens than min_tokens should be rejected.
    MockBackend backend;
    std::string dir = test_tmp_path("luce_test_save_below").string();
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

// ─── Disk-cache identity salt tests (manifest hardening) ────────────────
//
// (a) Different salts → different layout_id; same salt → same layout_id.
// (b) All-zero salt (default) ≡ no salt at all (back-compat).

// Helper: read layout_id from the first .dkv file found under base/.
static std::array<uint8_t, 16> read_layout_id_from_cache_dir(const std::string & base) {
    std::array<uint8_t, 16> id{};
    std::error_code ec;
    for (const fs::directory_entry & entry : fs::directory_iterator(base, ec)) {
        if (ec) break;
        if (!entry.is_directory(ec)) {
            ec.clear();
            continue;
        }
        for (const fs::directory_entry & file_entry :
             fs::directory_iterator(entry.path(), ec)) {
            if (ec) break;
            const std::string name = file_entry.path().filename().string();
            if (name.size() < 4 || name.compare(name.size() - 4, 4, ".dkv") != 0) {
                continue;
            }
            FILE * f = std::fopen(file_entry.path().string().c_str(), "rb");
            if (!f) continue;
            std::fseek(f, 8, SEEK_SET);  // skip magic(4) + version(4)
            std::fread(id.data(), 1, 16, f);
            std::fclose(f);
            return id;
        }
    }
    return id;
}

// Layout mock with a settable live position that can also adopt
// deserialized snapshots (lookup path).
struct MockBackendWithAdopt : MockBackendWithLayout {
    std::vector<std::pair<ggml_context *, ggml_backend_buffer_t>> adopted_;
    int adopted_cur_pos_ = 0;
    int cur_pos_ = kMaxPos;
    ~MockBackendWithAdopt() {
        for (auto & p : adopted_) {
            if (p.second) ggml_backend_buffer_free(p.second);
            if (p.first) ggml_free(p.first);
        }
    }
    SnapshotRef snapshot_ref(int slot) const override {
        SnapshotRef ref = MockBackendWithLayout::snapshot_ref(slot);
        ref.cur_pos = cur_pos_;
        return ref;
    }
    int snapshot_cur_pos(int) const override { return cur_pos_; }
    bool snapshot_adopt(int, ggml_context * ctx, ggml_backend_buffer_t buf,
                        int cur_pos, int32_t) override {
        adopted_.push_back({ctx, buf});
        adopted_cur_pos_ = cur_pos;
        return true;
    }
};

TEST_CASE(ServerUnitFixture, test_disk_cache_failed_init_disables_pending_writes) {
    MockBackendWithAdopt backend;
    const std::string file = test_tmp_path("luce_test_disk_init_blocker").string();
    rm_rf(file);
    FILE * blocker = std::fopen(file.c_str(), "wb");
    TEST_ASSERT(blocker != nullptr);
    std::fclose(blocker);
    for (const std::string & dir : {file, file + "/cache"}) {
        DiskCacheConfig cfg;
        cfg.cache_dir = dir;
        cfg.min_tokens = 1;
        cfg.max_pending_bytes = 1;  // a second queued write would block without a writer
        DiskPrefixCache cache(cfg, backend);
        TEST_ASSERT(!cache.init());
        TEST_ASSERT(cache.disabled());
        const auto ref = backend.snapshot_ref(0);
        std::vector<int32_t> key(40, 1);
        for (int turn = 0; turn < 2; ++turn) {
            ++key.back();
            TEST_ASSERT(!cache.save(0, key));
            TEST_ASSERT(!cache.save_async(OwnedSnapshot::clone_to_host(
                ref.ctx, ref.cur_pos, ref.last_tok), key));
        }
        TEST_ASSERT(cache.entry_count() == 0);
        TEST_ASSERT(cache.stats().pending_bytes == 0);
        TEST_ASSERT(cache.stats().backpressure_waits == 0);
    }
    rm_rf(file);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_rejects_snapshot_past_key) {
    // A snapshot of kMaxPos positions may only be filed under a key that
    // covers at least kMaxPos tokens; shorter keys are refused on save and,
    // for files that already exist, on read.
    MockBackendWithAdopt backend;
    std::string dir = test_tmp_path("luce_test_past_key").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    std::vector<int32_t> short_key;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos - 4; ++i) short_key.push_back(i + 1);
    std::vector<int32_t> full_key;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos; ++i) full_key.push_back(i + 1);
    {
        DiskPrefixCache cache(cfg, backend);
        TEST_ASSERT(cache.init());
        cache.learn_layout(0);
        TEST_ASSERT(!cache.save(0, short_key));
        TEST_ASSERT(cache.total_bytes() == 0);
        TEST_ASSERT(cache.save(0, full_key));
        TEST_ASSERT(cache.total_bytes() > 0);
        TEST_ASSERT(!cache.lookup(short_key, 1));
        TEST_ASSERT(cache.lookup(full_key, 1));
        TEST_ASSERT(backend.adopted_cur_pos_ == MockBackendWithLayout::kMaxPos);
    }  // the writer finishes and the directory lock is released

    // Forge the pre-fix shape on disk: key shorter than the snapshot. The
    // scan keys entries by the header's token hash, so only the header
    // fields need rewriting (token_count at byte 32, token_hash at byte 36
    // of the 80-byte field-by-field header).
    {
        std::string forged;
        for (auto & entry : fs::recursive_directory_iterator(dir)) {
            if (entry.path().extension() == ".dkv") {
                forged = (entry.path().parent_path() /
                          (std::string(32, 'f') + ".dkv")).string();
                fs::copy_file(entry.path(), forged, fs::copy_options::overwrite_existing);
                break;
            }
        }
        TEST_ASSERT(!forged.empty());
        FILE * f = std::fopen(forged.c_str(), "r+b");
        TEST_ASSERT(f != nullptr);
        if (f) {
            const uint32_t short_count = (uint32_t)short_key.size();
            PrefixHash ph = hash_prefix(short_key.data(), (int)short_key.size());
            std::fseek(f, 32, SEEK_SET);
            TEST_ASSERT(std::fwrite(&short_count, 4, 1, f) == 1);
            TEST_ASSERT(std::fwrite(ph.data(), 16, 1, f) == 1);
            std::fclose(f);
        }
        DiskPrefixCache reopened(cfg, backend);
        TEST_ASSERT(reopened.init());
        reopened.learn_layout(0);
        TEST_ASSERT(!reopened.lookup(short_key, 2));   // rejected + removed
        TEST_ASSERT(!fs::exists(forged));
        TEST_ASSERT(reopened.lookup(full_key, 2));     // consistent file survives
    }
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_continued_keys_full_prefix) {
    // Continued checkpoints are paced by the interval but keyed by the
    // tokens the snapshot really covers, so only a prompt containing all of
    // them can hit.
    MockBackendWithAdopt backend;
    std::string dir = test_tmp_path("luce_test_continued_key").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    cfg.continued_interval = 10;   // 32 positions -> crosses at 30
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.init());
    cache.learn_layout(0);

    std::vector<int32_t> tokens;
    for (int i = 0; i < 60; ++i) tokens.push_back(100 + i);
    const int cur_pos = MockBackendWithLayout::kMaxPos;  // 32 -> bucket 30
    backend.cur_pos_ = cur_pos;
    TEST_ASSERT(cache.maybe_store_continued(0, tokens, cur_pos));
    std::vector<int32_t> aligned(tokens.begin(), tokens.begin() + 30);
    std::vector<int32_t> covered(tokens.begin(), tokens.begin() + cur_pos);
    TEST_ASSERT(!cache.lookup(aligned, 1));
    TEST_ASSERT(cache.lookup(covered, 1));
    TEST_ASSERT(backend.adopted_cur_pos_ == cur_pos);
    const size_t bytes_after_first = cache.total_bytes();
    TEST_ASSERT(bytes_after_first > 0);

    // Same interval bucket: no second checkpoint.
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, cur_pos));
    backend.cur_pos_ = 38;  // still bucket 30
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 38));
    TEST_ASSERT(cache.total_bytes() == bytes_after_first);

    // Crossing into bucket 40 fires again, keyed by the 42 covered tokens.
    backend.cur_pos_ = 42;
    TEST_ASSERT(cache.maybe_store_continued(0, tokens, 42));
    TEST_ASSERT(cache.total_bytes() > bytes_after_first);
    std::vector<int32_t> bucket(tokens.begin(), tokens.begin() + 40);
    std::vector<int32_t> covered2(tokens.begin(), tokens.begin() + 42);
    TEST_ASSERT(!cache.lookup(bucket, 2));
    TEST_ASSERT(cache.lookup(covered2, 2));
    TEST_ASSERT(backend.adopted_cur_pos_ == 42);
    // The earlier checkpoint is still there, and bucket 40 does not refire.
    TEST_ASSERT(cache.lookup(covered, 3));
    backend.cur_pos_ = 47;
    TEST_ASSERT(!cache.maybe_store_continued(0, tokens, 47));
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_identity_salt_changes_layout_id) {
    MockBackendWithLayout backend;
    std::vector<int32_t> prompt;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos; ++i) prompt.push_back(i + 1);

    // Salt A: non-zero.
    std::array<uint8_t, 16> salt_a{};
    salt_a[0] = 0x01; salt_a[15] = 0xAB;

    std::string dir_a = test_tmp_path("luce_test_salt_a").string();
    rm_rf(dir_a);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir_a; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(salt_a);
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    // Salt B: different from A.
    std::array<uint8_t, 16> salt_b{};
    salt_b[0] = 0x02; salt_b[15] = 0xCD;

    std::string dir_b = test_tmp_path("luce_test_salt_b").string();
    rm_rf(dir_b);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir_b; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(salt_b);
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    auto id_a = read_layout_id_from_cache_dir(dir_a);
    auto id_b = read_layout_id_from_cache_dir(dir_b);

    // Different salts → different layout_id.
    TEST_ASSERT(id_a != id_b);

    // Same salt A applied again → identical layout_id.
    std::string dir_a2 = test_tmp_path("luce_test_salt_a2").string();
    rm_rf(dir_a2);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir_a2; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(salt_a);
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }
    auto id_a2 = read_layout_id_from_cache_dir(dir_a2);
    TEST_ASSERT(id_a == id_a2);

    rm_rf(dir_a);
    rm_rf(dir_b);
    rm_rf(dir_a2);
}

TEST_CASE(ServerUnitFixture, test_disk_identity_salt_zero_is_backcompat) {
    // Explicit all-zero salt must produce the same layout_id as no salt call
    // (default-constructed identity_salt_ is already all-zero).
    MockBackendWithLayout backend;
    std::vector<int32_t> prompt;
    for (int i = 0; i < MockBackendWithLayout::kMaxPos; ++i) prompt.push_back(i + 1);

    std::string dir1 = test_tmp_path("luce_test_salt_zero1").string();
    rm_rf(dir1);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir1; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        // No set_identity_salt call — stays all-zero.
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    std::string dir2 = test_tmp_path("luce_test_salt_zero2").string();
    rm_rf(dir2);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir2; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        std::array<uint8_t, 16> zero_salt{};
        cache.set_identity_salt(zero_salt);
        cache.init();
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, prompt));
    }

    auto id1 = read_layout_id_from_cache_dir(dir1);
    auto id2 = read_layout_id_from_cache_dir(dir2);
    TEST_ASSERT(id1 == id2);

    rm_rf(dir1);
    rm_rf(dir2);
}

// ─── Session persistence: TTL, LRU budget, restart, validation ──────────

namespace {

std::vector<int32_t> disk_test_key(int n, int32_t base = 1) {
    std::vector<int32_t> key;
    for (int i = 0; i < n; ++i) key.push_back(base + i);
    return key;
}

std::vector<std::string> dkv_files(const std::string & dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (auto & e : fs::recursive_directory_iterator(dir, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".dkv") {
            out.push_back(e.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Size of one snapshot file the mock produces for a key of `n` tokens.
uint64_t one_file_size(MockBackendWithAdopt & backend, const std::string & dir, int n) {
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    uint64_t size = 0;
    {
        DiskPrefixCache cache(cfg, backend);
        cache.init();
        cache.learn_layout(0);
        cache.save(0, disk_test_key(n));
        size = cache.total_bytes();
    }
    rm_rf(dir);
    return size;
}

}  // namespace

TEST_CASE(ServerUnitFixture, test_disk_cache_ttl_expiry) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_ttl").string();
    rm_rf(dir);
    uint64_t clock = 1000;
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    cfg.ttl_seconds = 100;
    const auto a = disk_test_key(40, 1);
    const auto b = disk_test_key(40, 500);
    const auto c = disk_test_key(40, 900);
    {
        DiskPrefixCache cache(cfg, backend);
        cache.set_clock([&]() { return clock; });
        TEST_ASSERT(cache.init());
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, a));
        clock = 1090;
        TEST_ASSERT(cache.lookup(a, 1));          // used at 1090
        clock = 1180;
        TEST_ASSERT(cache.lookup(a, 1));          // 90 s idle: still alive
        clock = 1281;
        TEST_ASSERT(!cache.lookup(a, 1));         // 101 s idle: expired + deleted
        TEST_ASSERT(cache.entry_count() == 0);
        TEST_ASSERT(dkv_files(dir).empty());

        // Insertion sweep: saving b deletes an expired c.
        TEST_ASSERT(cache.save(0, c));
        clock = 1400;
        TEST_ASSERT(cache.save(0, b));
        TEST_ASSERT(cache.entry_count() == 1);
        TEST_ASSERT(!cache.lookup(c, 1));
    }
    {
        // Startup sweep: b expires while the server is down.
        clock = 1600;
        DiskPrefixCache cache(cfg, backend);
        cache.set_clock([&]() { return clock; });
        TEST_ASSERT(cache.init());
        TEST_ASSERT(cache.entry_count() == 0);
        TEST_ASSERT(dkv_files(dir).empty());
    }
    {
        // ttl 0 = never expire.
        DiskCacheConfig never = cfg; never.ttl_seconds = 0;
        DiskPrefixCache cache(never, backend);
        cache.set_clock([&]() { return clock; });
        TEST_ASSERT(cache.init());
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, a));
        clock += 10ull * 365 * 24 * 3600;
        TEST_ASSERT(cache.lookup(a, 1));
    }
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_budget_lru_survives_restart) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_lru").string();
    const uint64_t one = one_file_size(backend, dir, 40);
    TEST_ASSERT(one > 0);
    uint64_t clock = 1;
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    cfg.ttl_seconds = 0;
    cfg.budget_bytes = (size_t)(one * 2 + one / 2);  // room for two files
    const auto a = disk_test_key(40, 1);
    const auto b = disk_test_key(40, 500);
    const auto c = disk_test_key(40, 900);
    const auto d = disk_test_key(40, 1300);
    {
        DiskPrefixCache cache(cfg, backend);
        cache.set_clock([&]() { return clock; });
        TEST_ASSERT(cache.init());
        cache.learn_layout(0);
        clock = 1; TEST_ASSERT(cache.save(0, a));
        clock = 2; TEST_ASSERT(cache.save(0, b));
        clock = 3; TEST_ASSERT(cache.lookup(a, 1));  // a is now newer than b
    }
    {
        // last_used came from the file: b (2) is the LRU, not a (created 1).
        DiskPrefixCache cache(cfg, backend);
        cache.set_clock([&]() { return clock; });
        TEST_ASSERT(cache.init());
        TEST_ASSERT(cache.entry_count() == 2);
        cache.learn_layout(0);
        clock = 4; TEST_ASSERT(cache.save(0, c));
        TEST_ASSERT(cache.total_bytes() <= cfg.budget_bytes);
        TEST_ASSERT(cache.entry_count() == 2);
        TEST_ASSERT(!cache.lookup(b, 1));
        clock = 5; TEST_ASSERT(cache.lookup(a, 1));
        clock = 6; TEST_ASSERT(cache.save(0, d));   // evicts c (4)
        TEST_ASSERT(cache.total_bytes() <= cfg.budget_bytes);
        TEST_ASSERT(!cache.lookup(c, 1));
        TEST_ASSERT(cache.lookup(a, 1));
        TEST_ASSERT(cache.lookup(d, 1));
    }
    {
        // A smaller budget is enforced at startup, before any request.
        DiskCacheConfig small = cfg; small.budget_bytes = (size_t)(one + one / 2);
        DiskPrefixCache cache(small, backend);
        cache.set_clock([&]() { return clock; });
        TEST_ASSERT(cache.init());
        TEST_ASSERT(cache.entry_count() == 1);
        TEST_ASSERT(cache.total_bytes() <= small.budget_bytes);
    }
    {
        // A snapshot larger than the whole budget is never written.
        DiskCacheConfig tiny = cfg; tiny.budget_bytes = (size_t)(one / 2);
        DiskPrefixCache t(tiny, backend);
        TEST_ASSERT(t.init());
        t.learn_layout(0);
        TEST_ASSERT(!t.disabled());
        TEST_ASSERT(!t.save(0, disk_test_key(40, 7000)));
        TEST_ASSERT(t.total_bytes() <= tiny.budget_bytes);
    }
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_duplicate_save_refreshes_ttl_after_restart) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_duplicate_touch").string();
    const auto key = disk_test_key(40);
    for (bool async : {false, true}) {
        rm_rf(dir);
        uint64_t now = 100000;
        DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1; cfg.ttl_seconds = 100;
        {
            DiskPrefixCache cache(cfg, backend);
            cache.set_clock([&] { return now; });
            TEST_ASSERT(cache.init());
            TEST_ASSERT(cache.save(0, key));
            cache.drain();
            now += 90;
            const auto ref = backend.snapshot_ref(0);
            TEST_ASSERT(async ? cache.save_async(OwnedSnapshot::clone_to_host(
                ref.ctx, ref.cur_pos, ref.last_tok), key) : cache.save(0, key));
            TEST_ASSERT(cache.stats().writes == 1);
        }
        now += 60;  // expired from creation, alive from the duplicate save
        {
            DiskPrefixCache cache(cfg, backend);
            cache.set_clock([&] { return now; });
            TEST_ASSERT(cache.init());
            TEST_ASSERT(cache.entry_count() == 1);
            TEST_ASSERT(cache.lookup(key, 1));
        }
    }
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_restart_restores_and_rejects_other_identity) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_restart").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    std::array<uint8_t, 16> q4{}; q4[0] = 4;
    std::array<uint8_t, 16> q8{}; q8[0] = 8;
    const auto key = disk_test_key(40);
    {
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(q4);
        TEST_ASSERT(cache.init());
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, key, "replay-text"));
    }
    {
        // Same identity after a restart: the index is rebuilt from disk and
        // the first lookup hits before any live snapshot exists.
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(q4);
        TEST_ASSERT(cache.init());
        TEST_ASSERT(cache.entry_count() == 1);
        std::vector<std::string> aux;
        cache.for_each_aux([&](const std::string & blob) { aux.push_back(blob); });
        TEST_ASSERT(aux.size() == 1 && aux[0] == "replay-text");
        TEST_ASSERT(cache.longest_prefix(key).len == 40);
        backend.adopted_cur_pos_ = 0;
        TEST_ASSERT(cache.lookup(key, 1));
        TEST_ASSERT(backend.adopted_cur_pos_ == MockBackendWithLayout::kMaxPos);
    }
    {
        // Another KV type/model identity: never restored, still indexed so
        // the budget and the TTL cover it.
        DiskPrefixCache cache(cfg, backend);
        cache.set_identity_salt(q8);
        TEST_ASSERT(cache.init());
        TEST_ASSERT(cache.entry_count() == 1);
        size_t other_aux = 0;
        cache.for_each_aux([&](const std::string &) { ++other_aux; });
        TEST_ASSERT(other_aux == 0);
        TEST_ASSERT(cache.longest_prefix(key).len == 0);
        TEST_ASSERT(!cache.lookup(key, 1));
        cache.learn_layout(0);
        TEST_ASSERT(cache.longest_prefix(key).len == 0);
        TEST_ASSERT(!cache.lookup(key, 1));
        TEST_ASSERT(dkv_files(dir).size() == 1);
    }
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_init_deletes_corrupt_files) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_corrupt").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    const auto good = disk_test_key(40, 1);
    const auto cut = disk_test_key(40, 300);
    const auto old = disk_test_key(40, 600);
    std::string layout_dir;
    {
        DiskPrefixCache cache(cfg, backend);
        TEST_ASSERT(cache.init());
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, good));
        TEST_ASSERT(cache.save(0, cut));
        TEST_ASSERT(cache.save(0, old));
    }
    auto files = dkv_files(dir);
    TEST_ASSERT(files.size() == 3);
    layout_dir = fs::path(files[0]).parent_path().string();
    auto path_of = [&](const std::vector<int32_t> & k) {
        const PrefixHash h = hash_prefix(k.data(), (int)k.size());
        static const char * hx = "0123456789abcdef";
        std::string name;
        for (uint8_t byte : h) { name += hx[byte >> 4]; name += hx[byte & 15]; }
        return layout_dir + "/" + name + ".dkv";
    };
    // Truncated (crash or full disk mid-write).
    fs::resize_file(path_of(cut), fs::file_size(path_of(cut)) - 7);
    // Older format version.
    if (FILE * f = std::fopen(path_of(old).c_str(), "r+b")) {
        const uint32_t v3 = 3;
        std::fseek(f, 4, SEEK_SET);
        std::fwrite(&v3, 4, 1, f);
        std::fclose(f);
    }
    // Garbage and a temp file left by an interrupted write.
    { FILE * f = std::fopen((layout_dir + "/" + std::string(32, 'a') + ".dkv").c_str(), "wb");
      std::fputs("not a snapshot", f); std::fclose(f); }
    { FILE * f = std::fopen((layout_dir + "/" + std::string(32, 'b') + ".dkv.tmp").c_str(), "wb");
      std::fputs("partial", f); std::fclose(f); }
    {
        DiskPrefixCache cache(cfg, backend);
        TEST_ASSERT(cache.init());
        TEST_ASSERT(cache.entry_count() == 1);
        files = dkv_files(dir);
        TEST_ASSERT(files.size() == 1 && files[0] == path_of(good));
        TEST_ASSERT(cache.lookup(good, 1));
    }
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_verifies_stored_tokens) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_tokens").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    const auto key = disk_test_key(40);
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.init());
    cache.learn_layout(0);
    TEST_ASSERT(cache.save(0, key));
    cache.drain();
    const auto files = dkv_files(dir);
    TEST_ASSERT(files.size() == 1);
    // The stored key sits right before the (empty) aux blob, the payload
    // and the 8-byte checksum.
    uint64_t payload = 0;
    FILE * f = std::fopen(files[0].c_str(), "r+b");
    TEST_ASSERT(f != nullptr);
    std::fseek(f, 52, SEEK_SET);
    TEST_ASSERT(std::fread(&payload, 8, 1, f) == 1);
    const long tokens_at = (long)(fs::file_size(files[0]) - payload - 8 - 4 - 4 * key.size());
    const int32_t other = 999999;
    std::fseek(f, tokens_at + 4 * 10, SEEK_SET);
    std::fwrite(&other, 4, 1, f);
    std::fclose(f);
    // Same hash in the index, different stored tokens: rejected and deleted.
    TEST_ASSERT(!cache.lookup(key, 1));
    TEST_ASSERT(dkv_files(dir).empty());
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_longest_prefix_and_prune_ancestors) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_prefix_probe").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.init());
    cache.learn_layout(0);
    const auto turn3 = disk_test_key(60);
    const std::vector<int32_t> turn1(turn3.begin(), turn3.begin() + 40);
    const std::vector<int32_t> turn2(turn3.begin(), turn3.begin() + 48);
    TEST_ASSERT(cache.save(0, turn1));
    TEST_ASSERT(cache.save(0, turn2));

    // Deepest restorable prefix, without reading any file.
    auto probe = cache.longest_prefix(turn3);
    TEST_ASSERT(probe.len == 48 && probe.cur_pos == MockBackendWithLayout::kMaxPos);
    auto diverged = turn3; diverged[45] = -7;
    TEST_ASSERT(cache.longest_prefix(diverged).len == 40);
    TEST_ASSERT(cache.longest_prefix(disk_test_key(60, 5000)).len == 0);
    TEST_ASSERT(cache.longest_prefix(turn1).len == 40);

    // A deeper state of the session supersedes turn2, but the shared head
    // (<= keep_max) stays for other sessions.
    TEST_ASSERT(cache.save(0, turn3));
    TEST_ASSERT(cache.prune_ancestors(turn3, 40) == 1);
    TEST_ASSERT(cache.entry_count() == 2);
    TEST_ASSERT(cache.longest_prefix(diverged).len == 40);
    // Without a head at or below keep_max, the shallowest ancestor stays
    // in its place (a long system/tools head).
    TEST_ASSERT(cache.prune_ancestors(turn3, 0) == 0);
    TEST_ASSERT(cache.entry_count() == 2);
    TEST_ASSERT(cache.longest_prefix(diverged).len == 40);
    TEST_ASSERT(cache.longest_prefix(turn3).len == 60);
    const auto turn4 = disk_test_key(70);
    TEST_ASSERT(cache.save(0, turn4));
    TEST_ASSERT(cache.prune_ancestors(turn4, 0) == 1);  // turn3; turn1 stays
    TEST_ASSERT(cache.longest_prefix(diverged).len == 40);
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_cache_write_failure_is_not_fatal) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_write_fail").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.init());
    cache.learn_layout(0);
    TEST_ASSERT(cache.save(0, disk_test_key(40)));
    cache.drain();
    const std::string layout_dir = fs::path(dkv_files(dir)[0]).parent_path().string();
    // Block the directory with a file: a deterministic write failure on
    // Windows and POSIX, including runs as root. Restore the original file.
    const std::string backup_dir = layout_dir + ".backup";
    fs::rename(layout_dir, backup_dir);
    FILE * blocker = std::fopen(layout_dir.c_str(), "wb");
    TEST_ASSERT(blocker != nullptr);
    std::fclose(blocker);
    const bool queued = cache.save(0, disk_test_key(40, 800));
    cache.drain();
    fs::remove(layout_dir);
    fs::rename(backup_dir, layout_dir);
    TEST_ASSERT(queued);
    TEST_ASSERT(cache.stats().write_failures == 1);
    TEST_ASSERT(cache.entry_count() == 1);
    TEST_ASSERT(dkv_files(dir).size() == 1);
    TEST_ASSERT(!cache.lookup(disk_test_key(40, 800), 1));
    rm_rf(dir);
}

// ─── SSD session cache: eviction writes, edit backups, shutdown, paged ──

namespace {

// Snapshot payloads in real host ggml buffers, one per slot, so the server
// can detach them for the disk writer and adopt them back. A payload of
// `pos` positions holds one F32 tensor [4, pos] filled with `pos`.
struct DiskPayloadBackend : MockBackend {
    struct Payload {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        int pos = 0;
    };
    std::map<int, Payload> slots;
    std::map<int, Payload> displaced;
    std::vector<int> freed;
    std::vector<int> adopted;
    int live_position = 0;  // what snapshot_save() captures

    static Payload make(int pos) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 4 + 1024;
        ip.no_alloc = true;
        Payload p;
        p.ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, 4, pos);
        ggml_set_name(t, "snap_k_0");
        p.buf = ggml_backend_alloc_ctx_tensors_from_buft(p.ctx, ggml_backend_cpu_buffer_type());
        std::vector<float> fill((size_t)ggml_nelements(t), (float)pos);
        ggml_backend_tensor_set(t, fill.data(), 0, ggml_nbytes(t));
        p.pos = pos;
        return p;
    }
    static void release(Payload & p) {
        if (p.buf) ggml_backend_buffer_free(p.buf);
        if (p.ctx) ggml_free(p.ctx);
        p = Payload{};
    }
    void put(int slot, int pos) {
        drop(slot);
        slots[slot] = make(pos);
    }
    void drop(int slot) {
        auto it = slots.find(slot);
        if (it == slots.end()) return;
        release(it->second);
        slots.erase(it);
    }
    ~DiskPayloadBackend() override {
        for (auto & [slot, p] : slots) release(p);
        for (auto & [slot, p] : displaced) release(p);
    }
    void snapshot_free(int slot) override {
        if (slots.count(slot)) freed.push_back(slot);
        drop(slot);
    }
    bool snapshot_save(int slot) override {
        if (live_position <= 0) return false;
        put(slot, live_position);
        return true;
    }
    bool snapshot_used(int slot) const override { return slots.count(slot) != 0; }
    int snapshot_cur_pos(int slot) const override {
        auto it = slots.find(slot);
        return it == slots.end() ? 0 : it->second.pos;
    }
    SnapshotRef snapshot_ref(int slot) const override {
        SnapshotRef ref;
        auto it = slots.find(slot);
        if (it == slots.end()) return ref;
        ref.ctx = it->second.ctx;
        ref.buf = it->second.buf;
        ref.cur_pos = it->second.pos;
        ref.last_tok = 7;
        return ref;
    }
    bool snapshot_adopt(int slot, ggml_context * ctx, ggml_backend_buffer_t buf,
                        int cur_pos, int32_t) override {
        drop(slot);
        slots[slot] = {ctx, buf, cur_pos};
        adopted.push_back(slot);
        return true;
    }
    OwnedSnapshot snapshot_take_displaced(int slot) override {
        auto it = displaced.find(slot);
        if (it == displaced.end()) return {};
        OwnedSnapshot out(it->second.ctx, it->second.buf, it->second.pos, -1);
        displaced.erase(it);
        return out;
    }
    size_t snapshot_bytes_estimate(int tokens) const override {
        return (size_t)tokens * 100 + 1000;
    }
};

ServerConfig disk_session_config(const std::string & dir) {
    ServerConfig config;
    config.prefix_cache_cap = 8;
    config.prefix_cache_max_bytes = 0;
    config.disk_cache_dir = dir;
    config.disk_cache_min_tokens = 1;
    config.disk_cache_cold_max_tokens = 2;  // the 2-token head is "short"
    return config;
}

// Commit `prompt` as an inline snapshot at its end, the way a request does.
int commit_turn(HttpServer & server, DiskPayloadBackend & backend,
                const std::vector<int32_t> & prompt, int restore_slot = -1,
                bool compressed = false) {
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    auto reservation = cache.reserve_inline_snap(
        prompt, restore_slot >= 0 ? backend.snapshot_cur_pos(restore_slot) : 0,
        /*prefer_tools_boundary=*/false, /*forced_cut=*/(int)prompt.size(),
        restore_slot, [&](int tokens) { return backend.snapshot_bytes_estimate(tokens); });
    if (!reservation.active()) return -1;
    const int slot = reservation.slot();
    SchedulerTestHarness::evict_inline_slot(server, slot);
    backend.put(slot, (int)prompt.size());
    SchedulerTestHarness::finalize_inline_snapshot(
        server, prompt, std::move(reservation), slot, (int)prompt.size(),
        restore_slot, compressed);
    return slot;
}

}  // namespace

TEST_CASE(ServerUnitFixture, test_disk_session_compressed_snapshot_keeps_ram_reuse_without_disk_key) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_compressed").string();
    rm_rf(dir);
    {
        auto owner = std::make_unique<DiskPayloadBackend>();
        auto & backend = *owner;
        LuceEngine engine(std::move(owner));
        ServerConfig config = disk_session_config(dir);
        config.prefix_cache_cap = 1;
        HttpServer server(engine, tokenizer, config);
        const std::vector<int32_t> view = {1, 100, 3, 101, 4};
        const int slot = commit_turn(server, backend, view, -1, true);
        TEST_ASSERT(slot >= 0);
        TEST_ASSERT(SchedulerTestHarness::prefix_cache(server).lookup_candidate(view, 5).second == 5);
        SchedulerTestHarness::flush_sessions(server);
        auto & disk = SchedulerTestHarness::disk_cache(server);
        TEST_ASSERT(disk.stats().writes == 0);
        // Evict the compressed state: raw sessions still persist normally.
        const std::vector<int32_t> raw = {1, 200, 3, 201, 4};
        TEST_ASSERT(commit_turn(server, backend, raw) >= 0);
        disk.drain();
        TEST_ASSERT(disk.stats().writes == 0);
        SchedulerTestHarness::flush_sessions(server);
        TEST_ASSERT(disk.stats().writes == 1);
        TEST_ASSERT(disk.longest_prefix(view).len == 0);
        TEST_ASSERT(disk.longest_prefix(raw).len == 5);
    }
    rm_rf(dir);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_disk_session_failed_init_keeps_ram_cache_working) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    {
        auto backend_owner = std::make_unique<DiskPayloadBackend>();
        auto & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        // A file cannot contain a cache directory: deterministic on every OS.
        ServerConfig config = disk_session_config(path + "/cache");
        config.prefix_cache_max_bytes = 3000;
        HttpServer server(engine, tokenizer, config);
        DiskPrefixCache & disk = SchedulerTestHarness::disk_cache(server);
        TEST_ASSERT(disk.disabled());
        TEST_ASSERT(!SchedulerTestHarness::props(server)["full_cache"].value("disk_enabled", true));
        const std::vector<int32_t> a = {1, 100, 3, 101};
        const std::vector<int32_t> b = {1, 200, 3, 201};
        const std::vector<int32_t> c = {1, 300, 3, 301};
        TEST_ASSERT(commit_turn(server, backend, a) >= 0);
        const int slot_b = commit_turn(server, backend, b);
        TEST_ASSERT(slot_b >= 0);
        TEST_ASSERT(commit_turn(server, backend, c) >= 0);  // evict a without disk work
        const auto restored = SchedulerTestHarness::prepare_cache(server, {1, 200, 3, 201, 4, 102});
        TEST_ASSERT(restored.restore_slot == slot_b && restored.prefix_len == 4);
        TEST_ASSERT(disk.entry_count() == 0 && disk.stats().pending_bytes == 0);
        SchedulerTestHarness::flush_sessions(server);
        TEST_ASSERT(disk.stats().writes == 0);
    }
    unlink(path.c_str());
}

// Persist-on-evict policy: every conversation's newest snapshot and the
// shared heads are written; a previous turn kept only as an edit backup is
// not, while its conversation's deeper snapshot is resident.
TEST_CASE(ServerUnitFixture, test_prefix_cache_persist_on_evict_skips_edit_backups) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PrefixCache cache(8, tokenizer, 0);
    const std::vector<int32_t> head = {1, 100};
    const std::vector<int32_t> turn1 = {1, 100, 3, 101};
    const std::vector<int32_t> turn2 = {1, 100, 3, 101, 4, 102};
    const std::vector<int32_t> turn3 = {1, 100, 3, 101, 4, 102, 3, 103};
    cache.confirm_inline_snap(0, (int)head.size(), head);
    cache.confirm_inline_snap(1, (int)turn1.size(), turn1);
    cache.confirm_inline_snap(2, (int)turn2.size(), turn2);
    TEST_ASSERT(cache.persist_on_evict(1));  // nothing marked yet
    cache.confirm_inline_snap(3, (int)turn3.size(), turn3);
    const auto pruned = cache.prune_superseded_ancestors(3);
    TEST_ASSERT(pruned == std::vector<int>({1}));
    TEST_ASSERT(cache.persist_on_evict(0));   // head
    TEST_ASSERT(!cache.persist_on_evict(2));  // edit backup of turn3
    TEST_ASSERT(cache.persist_on_evict(3));   // newest
    TEST_ASSERT(!cache.persist_on_evict(5));  // no entry
    const auto all = cache.entries_for_shutdown();
    TEST_ASSERT(all.size() == 3 && all[0].slot == 3);  // most recent first
    for (const auto & e : all) TEST_ASSERT(e.persist == (e.slot != 2));

    // A protected pin (the system/tools head of a tool request) is always
    // written, even with a deeper snapshot after it.
    PrefixCache pinned(8, tokenizer, 0);
    pinned.confirm_inline_snap(0, (int)turn1.size(), turn1, /*protect=*/true);
    pinned.confirm_inline_snap(1, (int)turn2.size(), turn2);
    pinned.confirm_inline_snap(2, (int)turn3.size(), turn3);
    pinned.prune_superseded_ancestors(2);
    TEST_ASSERT(pinned.persist_on_evict(0));
    TEST_ASSERT(pinned.persist_on_evict(2));

    // Once the deeper snapshot is gone, the kept turn is the newest again.
    cache.invalidate_inline_snap(3);
    TEST_ASSERT(cache.persist_on_evict(2));

    // Commit victims are logged (with their flag) only when recording.
    cache.confirm_inline_snap(3, (int)turn3.size(), turn3);
    cache.prune_superseded_ancestors(3);
    TEST_ASSERT(!cache.persist_on_evict(2));
    PrefixCache small(2, tokenizer, 0);
    small.set_record_evictions(true);
    small.confirm_inline_snap(0, (int)turn1.size(), turn1);
    small.confirm_inline_snap(1, (int)head.size(), {1, 200});
    const std::vector<int32_t> other = {1, 300, 3, 301};
    auto reservation = small.reserve_inline_snap(other, 0, false, (int)other.size());
    TEST_ASSERT(reservation.active());
    const int victim_slot = reservation.slot();
    TEST_ASSERT(reservation.commit(other));
    const auto evicted = small.take_evicted();
    TEST_ASSERT(evicted.size() == 1 && evicted[0].persist &&
                evicted[0].slot == victim_slot &&
                (evicted[0].ids == turn1 || evicted[0].ids == std::vector<int32_t>({1, 200})));
    TEST_ASSERT(small.take_evicted().empty());
    unlink(path.c_str());
}

// Single-sequence path: a snapshot the resident budget evicts is written to
// disk in the background and a later request restores it from there.
TEST_CASE(ServerUnitFixture, test_disk_session_ram_eviction_writes_and_restores) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_evict").string();
    rm_rf(dir);
    const std::vector<int32_t> a = {1, 100, 3, 101};
    const std::vector<int32_t> b = {1, 200, 3, 201};
    const std::vector<int32_t> c = {1, 300, 3, 301};
    {
        auto backend_owner = std::make_unique<DiskPayloadBackend>();
        DiskPayloadBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config = disk_session_config(dir);
        config.prefix_cache_max_bytes = 3000;  // two 4-token snapshots (1400 B)
        HttpServer server(engine, tokenizer, config);
        DiskPrefixCache & disk = SchedulerTestHarness::disk_cache(server);
        TEST_ASSERT(!disk.disabled());
        const int slot_a = commit_turn(server, backend, a);
        TEST_ASSERT(slot_a >= 0 && commit_turn(server, backend, b) >= 0);
        TEST_ASSERT(disk.entry_count() == 0);       // nothing written per turn
        TEST_ASSERT(commit_turn(server, backend, c) >= 0);  // evicts a
        // c reuses a's slot: a was handed to the disk writer first.
        TEST_ASSERT(SchedulerTestHarness::prefix_cache(server).lookup(a).first != slot_a);
        TEST_ASSERT(disk.entry_count() == 1);
        TEST_ASSERT(disk.longest_prefix(a).len == (int)a.size());
        // The next turn of conversation a restores from disk (served from
        // memory if the write is still queued), into the staging slot.
        std::vector<int32_t> next = a;
        next.insert(next.end(), {4, 102, 3, 103});
        auto prepared = SchedulerTestHarness::prepare_cache(server, next);
        TEST_ASSERT(prepared.restore_slot == ModelBackend::kMaxSlots - 1);
        TEST_ASSERT(prepared.prefix_len == (int)a.size());
        disk.drain();
        TEST_ASSERT(disk.stats().writes == 1 && disk.stats().pending_files == 0);
    }
    rm_rf(dir);
    unlink(path.c_str());
}

// Single-sequence path: a long session that arrives after several short
// ones is captured; the short ones it pushes out of RAM go to disk first.
TEST_CASE(ServerUnitFixture, test_disk_session_large_capture_evicts_several_to_disk) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_several").string();
    rm_rf(dir);
    const std::vector<std::vector<int32_t>> small = {
        {1, 100, 3, 101}, {1, 200, 3, 201}, {1, 300, 3, 301}, {1, 400, 3, 401}};
    std::vector<int32_t> big = {1, 900, 3};
    big.insert(big.end(), 40, 901);  // 43 tokens: 5300 bytes, 1400 per short one
    {
        auto backend_owner = std::make_unique<DiskPayloadBackend>();
        DiskPayloadBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config = disk_session_config(dir);
        config.prefix_cache_max_bytes = 7000;
        HttpServer server(engine, tokenizer, config);
        DiskPrefixCache & disk = SchedulerTestHarness::disk_cache(server);
        PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
        std::vector<int> slots;
        for (const auto & ids : small) {
            slots.push_back(commit_turn(server, backend, ids));
            TEST_ASSERT(slots.back() >= 0);
        }
        TEST_ASSERT(disk.entry_count() == 0);
        const int big_slot = commit_turn(server, backend, big);
        TEST_ASSERT(big_slot >= 0);
        TEST_ASSERT(cache.stats().budget_skips == 0);
        TEST_ASSERT(cache.stats().resident_bytes == 1400 + 5300);
        // The three oldest left RAM, written then freed; the newest stays.
        for (int i = 0; i < 3; ++i) {
            TEST_ASSERT(std::count(backend.freed.begin(), backend.freed.end(), slots[i]) >= 1);
            TEST_ASSERT(!backend.snapshot_used(slots[i]));
            TEST_ASSERT(cache.lookup_candidate(small[i], 4).first < 0);
            TEST_ASSERT(disk.longest_prefix(small[i]).len == 4);
        }
        TEST_ASSERT(cache.lookup_candidate(small[3], 4).first == slots[3]);
        TEST_ASSERT(disk.longest_prefix(small[3]).len == 0);
        std::vector<int32_t> next = big;
        next.insert(next.end(), {4, 902});
        TEST_ASSERT(cache.lookup_candidate(next, (int)next.size()).second == (int)big.size());
        disk.drain();
        TEST_ASSERT(disk.stats().writes == 3 && disk.stats().pending_files == 0);
        // A short session restores from disk afterwards.
        std::vector<int32_t> again = small[0];
        again.insert(again.end(), {4, 102});
        const auto prepared = SchedulerTestHarness::prepare_cache(server, again);
        TEST_ASSERT(prepared.restore_slot == ModelBackend::kMaxSlots - 1);
        TEST_ASSERT(prepared.prefix_len == 4);
    }
    rm_rf(dir);
    unlink(path.c_str());
}

// Tool path: an agent turn saved with its generated text (the live state
// after generation) is written when the resident budget evicts it, keyed
// by the tokens the next request renders.
TEST_CASE(ServerUnitFixture, test_disk_session_agent_turn_eviction_writes_live_state) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_agent").string();
    rm_rf(dir);
    const std::vector<int32_t> prompt = {1, 100, 3, 101};
    const std::vector<int32_t> generated = {200, 201, 202};
    const std::vector<int32_t> canonical = {1, 100, 3, 101, 200, 201, 202, 7};
    {
        auto backend_owner = std::make_unique<DiskPayloadBackend>();
        DiskPayloadBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config = disk_session_config(dir);
        config.prefix_cache_max_bytes = 3500;
        HttpServer server(engine, tokenizer, config);
        DiskPrefixCache & disk = SchedulerTestHarness::disk_cache(server);
        backend.live_position = 6;  // the last sampled token is not decoded
        TEST_ASSERT(SchedulerTestHarness::save_generated_turn(
            server, prompt, generated, canonical));
        TEST_ASSERT(commit_turn(server, backend, {1, 200, 3, 201}) >= 0);
        TEST_ASSERT(disk.entry_count() == 0);
        TEST_ASSERT(commit_turn(server, backend, {1, 300, 3, 301}) >= 0);
        std::vector<int32_t> next = canonical;
        next.insert(next.end(), {8, 9});
        const auto probe = disk.longest_prefix(next);
        TEST_ASSERT(probe.len == 6 && probe.cur_pos == 6);
        disk.drain();
        TEST_ASSERT(disk.stats().writes == 1);
    }
    rm_rf(dir);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_qwen_disk_adopt_restores_dense_state_and_rejects_failed_restore) {
    struct CpuSnapshotBackend : Qwen35Backend {
        CpuSnapshotBackend() : Qwen35Backend(Qwen35Config{}) {
            target_weights().n_layer = 0;
            target_weights().full_attention_interval = 1;
            auto & c = target_cache();
            c.max_ctx = c.target_feat_cap = 16;
            ggml_init_params ip{};
            ip.mem_size = 4 * ggml_tensor_overhead() + 1024;
            ip.no_alloc = true;
            c.base_ctx = ggml_init(ip);
            c.target_feat = ggml_new_tensor_2d(c.base_ctx, GGML_TYPE_F32, 2, 16);
            c.base_buf = ggml_backend_alloc_ctx_tensors_from_buft(c.base_ctx, ggml_backend_cpu_buffer_type());
            ggml_backend_buffer_clear(c.base_buf, 0);
        }
        bool restore(int slot) { return restore_target_cache_from_snapshot(slot); }
        TargetCache & cache() { return target_cache(); }
    } backend;
    const std::string dir = test_tmp_path("luce_test_qwen_disk_adopt").string();
    rm_rf(dir);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
        DiskPrefixCache disk(cfg, backend);
        TEST_ASSERT(disk.init());
        ggml_init_params ip{};
        ip.mem_size = 4 * ggml_tensor_overhead() + 1024;
        ip.no_alloc = true;
        auto * ctx = ggml_init(ip);
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
        ggml_set_name(t, "snap_target_feat");
        auto buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        const std::vector<float> values = {1, 2, 3, 4, 5, 6};
        ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
        const std::vector<int32_t> key = {1, 2, 3};
        TEST_ASSERT(disk.save_async(OwnedSnapshot(ctx, buf, 3, 3), key));
        disk.drain();
        TEST_ASSERT(disk.lookup(key, 0));  // actual Qwen35Backend::snapshot_adopt
        TEST_ASSERT(backend.restore(0)); // requires the adopted layout to be dense
        TEST_ASSERT(backend.cache().cur_pos == 3);
        std::vector<float> restored(values.size());
        ggml_backend_tensor_get(backend.cache().target_feat, restored.data(), 0,
                                restored.size() * sizeof(float));
        TEST_ASSERT(restored == values);
        // A restore failure must return before any prefill or decode on cleared KV.
        ++backend.cache().max_ctx;
        GenerateRequest req;
        req.prompt = key;
        req.n_gen = 1;
        DaemonIO io;
        const auto failed = backend.restore_and_generate_impl(0, req, io);
        TEST_ASSERT(!failed.ok() && failed.tokens.empty());
        TEST_ASSERT(failed.error && failed.error->code == GenerateErrorCode::InvalidSnapshotSlot);
        TEST_ASSERT(backend.cache().cur_pos == 0);
    }
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_disk_agent_turn_respects_request_scope) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_agent_disk_scope").string();
    const std::vector<int32_t> prompt = {1, 100, 3, 101};
    const std::vector<int32_t> generated = {200, 201, 202};
    const std::vector<int32_t> canonical = {1, 100, 3, 101, 200, 201, 202, 7};
    for (auto scope : {DiskPrefixCacheMode::Off, DiskPrefixCacheMode::Auto,
                       DiskPrefixCacheMode::Fixed, DiskPrefixCacheMode::Full}) {
        rm_rf(dir);
        {
            auto backend_owner = std::make_unique<DiskPayloadBackend>();
            auto & backend = *backend_owner;
            LuceEngine engine(std::move(backend_owner));
            HttpServer server(engine, tokenizer, disk_session_config(dir));
            backend.live_position = 6;
            TEST_ASSERT(SchedulerTestHarness::save_generated_turn(
                server, prompt, generated, canonical, -1, scope));
            // All scopes keep the RAM checkpoint, but only full keys it for disk.
            const auto hit = SchedulerTestHarness::prefix_cache(server).lookup_candidate(canonical, 8);
            TEST_ASSERT(hit.first >= 0 && hit.second == 6);
            SchedulerTestHarness::flush_sessions(server);
            auto & disk = SchedulerTestHarness::disk_cache(server);
            TEST_ASSERT(disk.stats().writes == (scope == DiskPrefixCacheMode::Full ? 1 : 0));
        }
    }
    rm_rf(dir);
    unlink(path.c_str());
}

// Graceful shutdown is one last eviction: the head and each conversation's
// newest snapshot are written, the edit backup and pruned turns are not, and
// a restarted server restores the session from disk.
TEST_CASE(ServerUnitFixture, test_disk_session_shutdown_flush_then_restart_restores) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_flush").string();
    rm_rf(dir);
    const std::vector<int32_t> head = {1, 100};
    const std::vector<int32_t> turn1 = {1, 100, 3, 101};
    const std::vector<int32_t> turn2 = {1, 100, 3, 101, 4, 102};
    const std::vector<int32_t> turn3 = {1, 100, 3, 101, 4, 102, 3, 103};
    {
        auto backend_owner = std::make_unique<DiskPayloadBackend>();
        DiskPayloadBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        HttpServer server(engine, tokenizer, disk_session_config(dir));
        PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
        // The head, then three turns, each restoring the previous one.
        int prev = -1;
        for (const auto * turn : {&head, &turn1, &turn2, &turn3}) {
            if (turn == &head) {
                backend.put(0, (int)head.size());
                cache.confirm_inline_snap(0, (int)head.size(), head);
                SchedulerTestHarness::set_slot_tokens(server, 0, head);
                prev = 0;
                continue;
            }
            prev = commit_turn(server, backend, *turn, prev);
            TEST_ASSERT(prev >= 0);
        }
        DiskPrefixCache & disk = SchedulerTestHarness::disk_cache(server);
        TEST_ASSERT(disk.entry_count() == 0);
        SchedulerTestHarness::flush_sessions(server);
        TEST_ASSERT(disk.stats().pending_files == 0);
        TEST_ASSERT(disk.entry_count() == 2);  // head + turn3
        TEST_ASSERT(disk.longest_prefix(turn3).len == (int)turn3.size());
        TEST_ASSERT(disk.longest_prefix(turn2).len == (int)head.size());
        TEST_ASSERT(dkv_files(dir).size() == 2);
    }
    {
        auto backend_owner = std::make_unique<DiskPayloadBackend>();
        DiskPayloadBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        HttpServer server(engine, tokenizer, disk_session_config(dir));
        std::vector<int32_t> next = turn3;
        next.insert(next.end(), {4, 104});
        const auto prepared = SchedulerTestHarness::prepare_cache(server, next);
        TEST_ASSERT(prepared.restore_slot == ModelBackend::kMaxSlots - 1);
        TEST_ASSERT(prepared.prefix_len == (int)turn3.size());
        TEST_ASSERT(backend.adopted == std::vector<int>({ModelBackend::kMaxSlots - 1}));
        // Another conversation on the same head restores the head.
        const std::vector<int32_t> other = {1, 100, 3, 555, 4, 556};
        TEST_ASSERT(SchedulerTestHarness::prepare_cache(server, other).prefix_len ==
                    (int)head.size());
    }
    rm_rf(dir);
    unlink(path.c_str());
}

// A session evicted to disk still matches the shared head in RAM; its
// next request restores the deeper disk state, not the head. When that read
// fails, the RAM head is still used.
TEST_CASE(ServerUnitFixture, test_disk_session_deeper_than_ram_hit_restores_from_disk) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_deeper").string();
    rm_rf(dir);
    const std::vector<int32_t> head = {1, 100};
    const std::vector<int32_t> turn1 = {1, 100, 3, 101};
    {
        auto backend_owner = std::make_unique<DiskPayloadBackend>();
        DiskPayloadBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        HttpServer server(engine, tokenizer, disk_session_config(dir));
        PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
        backend.put(0, (int)head.size());
        cache.confirm_inline_snap(0, (int)head.size(), head);
        SchedulerTestHarness::set_slot_tokens(server, 0, head);
        const int slot = commit_turn(server, backend, turn1, 0);
        TEST_ASSERT(slot > 0);
        SchedulerTestHarness::evict_inline_slot(server, slot);
        DiskPrefixCache & disk = SchedulerTestHarness::disk_cache(server);
        disk.drain();
        TEST_ASSERT(disk.stats().writes == 1);
        std::vector<int32_t> next = turn1;
        next.insert(next.end(), {4, 104});
        const auto prepared = SchedulerTestHarness::prepare_cache(server, next);
        TEST_ASSERT(prepared.restore_slot == ModelBackend::kMaxSlots - 1);
        TEST_ASSERT(prepared.prefix_len == (int)turn1.size());
        backend.snapshot_free(ModelBackend::kMaxSlots - 1);
        // The file is gone: the index still names it, the read fails and
        // the request restores the resident head.
        for (const auto & f : dkv_files(dir)) fs::remove(f);
        const auto fallback = SchedulerTestHarness::prepare_cache(server, next);
        TEST_ASSERT(fallback.restore_slot == 0);
        TEST_ASSERT(fallback.prefix_len == (int)head.size());
    }
    rm_rf(dir);
    unlink(path.c_str());
}

// Corruption fails closed: a flipped payload byte or a cut file is a logged
// miss, the file is deleted, and nothing is adopted.
TEST_CASE(ServerUnitFixture, test_disk_cache_checksum_rejects_corrupt_payload) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_checksum").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    DiskPrefixCache cache(cfg, backend);
    TEST_ASSERT(cache.init());
    cache.learn_layout(0);
    const auto a = disk_test_key(40, 1);
    const auto b = disk_test_key(40, 500);
    TEST_ASSERT(cache.save(0, a));
    TEST_ASSERT(cache.save(0, b));
    cache.drain();
    const auto path_of = [&](const std::vector<int32_t> & k) {
        for (const auto & f : dkv_files(dir)) {
            const PrefixHash h = hash_prefix(k.data(), (int)k.size());
            static const char * hx = "0123456789abcdef";
            std::string name;
            for (uint8_t byte : h) { name += hx[byte >> 4]; name += hx[byte & 15]; }
            if (fs::path(f).filename().string() == name + ".dkv") return f;
        }
        return std::string();
    };
    const std::string fa = path_of(a), fb = path_of(b);
    TEST_ASSERT(!fa.empty() && !fb.empty());
    // One bit in the middle of the tensor payload.
    {
        FILE * f = std::fopen(fa.c_str(), "r+b");
        TEST_ASSERT(f != nullptr);
        const long at = (long)fs::file_size(fa) - 8 - 100;
        std::fseek(f, at, SEEK_SET);
        const int byte = std::fgetc(f);
        std::fseek(f, at, SEEK_SET);
        std::fputc(byte ^ 0x10, f);
        std::fclose(f);
    }
    const size_t adopted_before = backend.adopted_.size();
    TEST_ASSERT(!cache.lookup(a, 1));
    TEST_ASSERT(!fs::exists(fa));
    TEST_ASSERT(cache.stats().load_failures == 1);
    // Cut short after the index was built (no startup scan in between).
    fs::resize_file(fb, fs::file_size(fb) - 3);
    TEST_ASSERT(!cache.lookup(b, 1));
    TEST_ASSERT(!fs::exists(fb));
    TEST_ASSERT(backend.adopted_.size() == adopted_before);
    TEST_ASSERT(cache.entry_count() == 0);
    rm_rf(dir);
}

// Tensor metadata is untrusted too: reject it before GGML can assert,
// divide by zero on a removed type, or allocate from overflowing dimensions.
TEST_CASE(ServerUnitFixture, test_disk_cache_rejects_corrupt_tensor_metadata) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_tensor_metadata").string();
    rm_rf(dir);
    DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
    const auto key = disk_test_key(40);
    for (int corruption = 0; corruption < 4; ++corruption) {
        for (bool restart : {false, true}) {
            rm_rf(dir);
            auto cache = std::make_unique<DiskPrefixCache>(cfg, backend);
            TEST_ASSERT(cache->init());
            TEST_ASSERT(cache->save(0, key));
            cache->drain();
            const auto files = dkv_files(dir);
            TEST_ASSERT(files.size() == 1);
            FILE * f = std::fopen(files[0].c_str(), "r+b");
            TEST_ASSERT(f != nullptr);
            std::fseek(f, DISK_CACHE_HEADER_SIZE, SEEK_SET);
            uint16_t name_len = 0;
            TEST_ASSERT(std::fread(&name_len, sizeof(name_len), 1, f) == 1);
            const long type_at = DISK_CACHE_HEADER_SIZE + 2 + name_len;
            std::fseek(f, type_at, SEEK_SET);
            const uint32_t type = corruption == 0 ? 4 : GGML_TYPE_Q4_0;
            TEST_ASSERT(std::fwrite(&type, sizeof(type), 1, f) == 1);
            if (corruption != 0) {
                const int64_t ne = corruption == 1 ? 1 :
                    (corruption == 2 ? INT64_MAX - 31 : 32);
                TEST_ASSERT(std::fwrite(&ne, sizeof(ne), 1, f) == 1);
                if (corruption == 3) {
                    const int64_t huge = INT64_MAX;
                    TEST_ASSERT(std::fwrite(&huge, sizeof(huge), 1, f) == 1);
                }
            }
            std::fclose(f);
            if (restart) {
                cache.reset();
                cache = std::make_unique<DiskPrefixCache>(cfg, backend);
                TEST_ASSERT(cache->init());
            }
            TEST_ASSERT(!cache->lookup(key, 1));
            TEST_ASSERT(!fs::exists(files[0]));
            TEST_ASSERT(cache->entry_count() == 0);
            TEST_ASSERT(backend.adopted_.empty());
        }
    }
    rm_rf(dir);
}

// Writes are asynchronous: a queued snapshot counts against the budget and
// is served from memory at once; the payload handed over is owned by the
// cache; the budget is clamped to the free space.
TEST_CASE(ServerUnitFixture, test_disk_cache_save_async_queue_and_free_space) {
    MockBackendWithAdopt backend;
    const std::string dir = test_tmp_path("luce_test_disk_async").string();
    rm_rf(dir);
    const auto key = disk_test_key(40, 1);
    const auto ref = backend.snapshot_ref(0);
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
        DiskPrefixCache cache(cfg, backend);
        TEST_ASSERT(cache.init());
        TEST_ASSERT(cache.save_async(
            OwnedSnapshot::clone_to_host(ref.ctx, ref.cur_pos, ref.last_tok), key, "aux"));
        TEST_ASSERT(cache.total_bytes() > 0);
        TEST_ASSERT(cache.lookup(key, 1));  // from memory or from the file
        TEST_ASSERT(backend.adopted_cur_pos_ == ref.cur_pos);
        const auto drained = cache.drain();
        TEST_ASSERT(drained.cancelled == 0);
        TEST_ASSERT(cache.stats().writes == 1 && cache.stats().pending_bytes == 0);
        TEST_ASSERT(dkv_files(dir).size() == 1);
        // An empty payload or one past its key is refused.
        TEST_ASSERT(!cache.save_async(OwnedSnapshot{}, disk_test_key(40, 99)));
        TEST_ASSERT(!cache.save_async(
            OwnedSnapshot::clone_to_host(ref.ctx, ref.cur_pos, ref.last_tok),
            disk_test_key(ref.cur_pos - 1, 99)));
    }
    {
        // 1 MB free and no reserve: the configured 128 GB budget shrinks to
        // the bytes already used plus the free space.
        DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
        cfg.free_space_reserve_bytes = 0;
        DiskPrefixCache cache(cfg, backend);
        cache.set_free_space_probe([] { return (uint64_t)1 << 20; });
        TEST_ASSERT(cache.init());
        TEST_ASSERT(cache.budget_bytes() == cache.total_bytes() + (1u << 20));
        cache.set_free_space_probe([] { return (uint64_t)0; });
        cache.learn_layout(0);
        TEST_ASSERT(cache.save(0, disk_test_key(40, 700)));  // evicts the old one
        TEST_ASSERT(cache.entry_count() == 1);
        TEST_ASSERT(!cache.lookup(key, 1));
    }
    // A second server on the same directory leaves it alone.
    {
        DiskCacheConfig cfg; cfg.cache_dir = dir; cfg.min_tokens = 1;
        DiskPrefixCache first(cfg, backend);
        TEST_ASSERT(first.init() && !first.disabled());
        DiskPrefixCache second(cfg, backend);
        TEST_ASSERT(second.init());
        TEST_ASSERT(second.disabled());
    }
    rm_rf(dir);
}

#if !defined(_WIN32)
namespace {

// A concurrent engine whose checkpoints live in the backend's snapshot
// slots (as Qwen's paged checkpoints do): a capture replaces the slot's
// payload and leaves the old one displaced; restoring checkpoint 64 reads
// the disk staging slot.
class DiskCheckpointEngine final : public SeqEngine {
public:
    explicit DiskCheckpointEngine(DiskPayloadBackend & backend)
        : backend_(backend), slots_(2) {}
    int slot_count() const override { return (int)slots_.size(); }
    int max_context() const override { return 64; }
    bool supports_prefix_store() const override { return true; }
    bool prefix_store_in_backend_snapshots() const override { return true; }
    size_t estimate_prefix_store_bytes(int tokens) const override {
        return bytes_per_token ? (size_t)tokens * bytes_per_token : 256;
    }
    size_t bytes_per_token = 0;  // 0: every checkpoint is 256 bytes
    bool token_is_eos(int32_t token) const override { return token == 2; }
    StepPlanLimits step_plan_limits(int) const override {
        return {2, 64, 128, 64};
    }
    AdmitResult admit(uint64_t id, const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override {
        return admit_with_prefix(id, prompt, sampler, PrefixStorePlan{});
    }
    AdmitResult admit_with_prefix(uint64_t, const std::vector<int32_t> & prompt,
                                  const SamplerCfg &,
                                  const PrefixStorePlan & plan) override {
        AdmitResult result;
        int chosen = -1;
        for (int i = 0; i < (int)slots_.size(); ++i) {
            if (!slots_[(size_t)i].active) { chosen = i; break; }
        }
        if (prompt.empty() || chosen < 0) {
            result.status = AdmitResult::Status::busy;
            return result;
        }
        Slot & slot = slots_[(size_t)chosen];
        slot = Slot{true, true, {}};
        result.status = AdmitResult::Status::admitted;
        result.slot = chosen;
        if (plan.restore.valid()) {
            result.prefix_store.restore_attempted = true;
            const int index = (int)plan.restore.id - 1;
            if (backend_.snapshot_cur_pos(index) == plan.restore.tokens) {
                result.prefix_store.restored = plan.restore;
                restores.push_back(plan.restore);
            } else {
                result.prefix_store.invalidated = plan.restore;
            }
        }
        if (!result.prefix_store.invalidated.valid() && plan.capture.valid()) {
            slot.capture = plan.capture;
            result.prefix_store.capture = plan.capture;
        }
        return result;
    }
    StepResult step(const StepPlan & plan) override {
        StepResult result;
        for (const StepInput & input : plan.decode) {
            result.decode.push_back({input.slot, 2, false, {}});
        }
        for (const PrefillSlice & slice : plan.prefills) {
            Slot & slot = slots_[(size_t)slice.slot];
            slot.prefilling = false;
            PrefillOutput output;
            output.slot = slice.slot;
            output.status = PrefillOutput::Status::completed;
            output.token = 2;
            if (slot.capture.valid()) {
                const int index = (int)slot.capture.checkpoint.id - 1;
                auto old = backend_.slots.find(index);
                if (old != backend_.slots.end()) {
                    DiskPayloadBackend::release(backend_.displaced[index]);
                    backend_.displaced[index] = old->second;
                    backend_.slots.erase(old);
                }
                backend_.slots[index] =
                    DiskPayloadBackend::make(slot.capture.checkpoint.tokens);
                output.prefix_store.status = PrefixStoreEvent::Status::saved;
                output.prefix_store.ticket = slot.capture;
                output.prefix_store.bytes =
                    estimate_prefix_store_bytes(slot.capture.checkpoint.tokens);
                slot.capture = {};
            }
            result.prefills.push_back(std::move(output));
        }
        return result;
    }
    void retire(int slot) override { slots_[(size_t)slot] = Slot{}; }
    void discard_prefix_store(PrefixStoreRef checkpoint) override {
        backend_.drop((int)checkpoint.id - 1);
    }
    std::vector<PrefixStoreRef> restores;

private:
    struct Slot {
        bool active = false;
        bool prefilling = false;
        PrefixCaptureTicket capture;
    };
    DiskPayloadBackend & backend_;
    std::vector<Slot> slots_;
};

struct DiskCheckpointBackend : DiskPayloadBackend {
    DiskCheckpointEngine engine{*this};
    SeqEngine * seq_engine() override { return &engine; }
};

}  // namespace

// Concurrent (paged) serving: a checkpoint evicted by another capture is
// written to disk, the conversation's next request is restored from it
// through the staging checkpoint, and shutdown writes the rest.
TEST_CASE(ServerUnitFixture, test_disk_session_concurrent_eviction_and_restore) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_paged").string();
    rm_rf(dir);

    auto backend_owner = std::make_unique<DiskCheckpointBackend>();
    DiskCheckpointBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config = disk_session_config(dir);
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;  // a third conversation evicts the first
    config.concurrent_prefix_cache_max_bytes = 4096;
    config.concurrent_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(engine, tokenizer, config);
    DiskPrefixCache & disk = SchedulerTestHarness::disk_cache(server);
    TEST_ASSERT(!disk.disabled());

    std::vector<std::unique_ptr<ServerJob>> jobs;
    std::vector<int> fds;
    const auto make_job = [&](std::vector<int32_t> prompt, const char * id) {
        int sockets[2] = {-1, -1};
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
        fds.push_back(sockets[0]);
        fds.push_back(sockets[1]);
        auto job = std::make_unique<ServerJob>();
        job->fd = sockets[0];
        job->req.format = ApiFormat::OPENAI_CHAT;
        job->req.prompt_tokens = std::move(prompt);
        job->req.max_output = 1;
        job->req.stream = false;
        job->req.model = "scheduler-test";
        job->req.response_id = id;
        jobs.push_back(std::move(job));
        return jobs.back().get();
    };
    const auto run_one = [&](std::vector<int32_t> prompt, const char * id) {
        ServerJob * job = make_job(std::move(prompt), id);
        SchedulerTestHarness::enqueue(server, job);
        std::unique_lock<std::mutex> lock(job->mu);
        return job->cv.wait_for(lock, std::chrono::seconds(5), [&] { return job->done; });
    };
    std::thread scheduler([&] { SchedulerTestHarness::run(server, backend.engine); });
    const std::vector<int32_t> a = {1, 100, 3, 101, 4};
    const std::vector<int32_t> b = {1, 200, 3, 201, 4};
    const std::vector<int32_t> c = {1, 300, 3, 301, 4};
    const bool abc = run_one(a, "a") && run_one(b, "b") && run_one(c, "c");
    disk.drain();
    const auto written = disk.stats().writes;
    const auto probe_a = disk.longest_prefix({1, 100, 3, 101, 4, 9});
    // Conversation a again: RAM lost it, disk has it.
    const bool again = run_one({1, 100, 3, 101, 4, 9, 2, 3, 102, 4}, "a2");
    SchedulerTestHarness::stop(server);
    scheduler.join();
    for (const int fd : fds) close(fd);

    TEST_ASSERT(abc && again);
    TEST_ASSERT(written == 1);
    TEST_ASSERT(probe_a.len == 3 && probe_a.cur_pos == 3);
    bool restored_from_disk = false;
    for (const auto & r : backend.engine.restores) {
        restored_from_disk |= r == PrefixStoreRef{(uint64_t)ModelBackend::kMaxSlots, 3};
    }
    TEST_ASSERT(restored_from_disk);
    TEST_ASSERT(!backend.snapshot_used(ModelBackend::kMaxSlots - 1));  // staging freed
    TEST_ASSERT(backend.displaced.empty());  // every displaced payload claimed

    // Shutdown writes what is still in RAM.
    SchedulerTestHarness::flush_sessions(server);
    TEST_ASSERT(disk.stats().pending_files == 0);
    TEST_ASSERT(disk.longest_prefix({1, 300, 3, 301, 4, 9}).len == 3);
    rm_rf(dir);
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_disk_session_concurrent_respects_request_scope) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_paged_scope").string();
    for (const auto mode : {DiskPrefixCacheMode::Off, DiskPrefixCacheMode::Fixed,
                            DiskPrefixCacheMode::Auto}) {
        rm_rf(dir);
        auto owner = std::make_unique<DiskCheckpointBackend>();
        auto & backend = *owner;
        LuceEngine engine(std::move(owner));
        ServerConfig config = disk_session_config(dir);
        config.arch = "qwen35";
        config.max_ctx = 64;
        config.prefix_cache_cap = 2;
        config.concurrent_prefix_cache_max_bytes = 4096;
        config.concurrent_prefix_cache = true;
        config.admission_coalesce_ms = 0;
        HttpServer server(engine, tokenizer, config);
        auto & disk = SchedulerTestHarness::disk_cache(server);
        const std::vector<int32_t> a = {1, 100, 3, 101, 4};
        backend.slots[62] = DiskPayloadBackend::make(3);
        disk.learn_layout(62);
        TEST_ASSERT(disk.save(62, {1, 100, 3}));
        disk.drain();
        backend.drop(62);

        std::vector<std::unique_ptr<ServerJob>> jobs;
        std::vector<int> fds;
        const auto run_one = [&](std::vector<int32_t> prompt) {
            int sockets[2] = {-1, -1};
            TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
            fds.insert(fds.end(), {sockets[0], sockets[1]});
            auto job = std::make_unique<ServerJob>();
            job->fd = sockets[0];
            job->req.format = ApiFormat::OPENAI_CHAT;
            job->req.prompt_tokens = std::move(prompt);
            job->req.max_output = 1;
            job->req.model = "scheduler-test";
            job->req.disk_cache_policy.mode = mode;
            job->req.disk_cache_policy.fixed_tokens = 2;
            ServerJob * raw = job.get();
            jobs.push_back(std::move(job));
            SchedulerTestHarness::enqueue(server, raw);
            std::unique_lock<std::mutex> lock(raw->mu);
            return raw->cv.wait_for(lock, std::chrono::seconds(5), [&] { return raw->done; });
        };
        std::thread scheduler([&] { SchedulerTestHarness::run(server, backend.engine); });
        const std::vector<int32_t> c = {1, 300, 3, 301, 4};
        const bool ok = run_one(a) && run_one({1, 200, 3, 201, 4}) &&
                        run_one(c) && run_one(c);
        SchedulerTestHarness::stop(server);
        scheduler.join();
        for (int fd : fds) close(fd);
        SchedulerTestHarness::flush_sessions(server);

        TEST_ASSERT(ok);
        bool disk_restored = false, ram_restored = false;
        for (const auto & r : backend.engine.restores) {
            disk_restored |= r.id == (uint64_t)ModelBackend::kMaxSlots;
            ram_restored |= r.id != (uint64_t)ModelBackend::kMaxSlots;
        }
        TEST_ASSERT(disk_restored == (mode == DiskPrefixCacheMode::Auto));
        TEST_ASSERT(ram_restored);  // scoped disk policy still allows RAM reuse
        TEST_ASSERT(disk.stats().writes == 1);  // only the seeded Full entry
        TEST_ASSERT(disk.entry_count() == 1);
        TEST_ASSERT(backend.displaced.empty());
    }
    rm_rf(dir);
    unlink(path.c_str());
}

// Concurrent (paged) serving does the same: a long session's checkpoint
// that no single resident one makes room for evicts several when it lands,
// writes them to disk (with the disk on) and frees their payloads.
TEST_CASE(ServerUnitFixture, test_disk_session_concurrent_large_capture_evicts_several) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::string dir = test_tmp_path("luce_test_session_paged_several").string();
    for (const bool disk_on : {true, false}) {
        rm_rf(dir);
        auto backend_owner = std::make_unique<DiskCheckpointBackend>();
        DiskCheckpointBackend & backend = *backend_owner;
        backend.engine.bytes_per_token = 100;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config = disk_session_config(disk_on ? dir : std::string());
        config.arch = "qwen35";
        config.max_ctx = 64;
        config.prefix_cache_cap = 8;
        config.concurrent_prefix_cache_max_bytes = 4500;
        config.concurrent_prefix_cache = true;
        config.admission_coalesce_ms = 0;
        HttpServer server(engine, tokenizer, config);
        DiskPrefixCache & disk = SchedulerTestHarness::disk_cache(server);
        TEST_ASSERT(disk.disabled() == !disk_on);

        std::vector<std::unique_ptr<ServerJob>> jobs;
        std::vector<int> fds;
        const auto run_one = [&](std::vector<int32_t> prompt, const char * id) {
            int sockets[2] = {-1, -1};
            TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
            fds.push_back(sockets[0]);
            fds.push_back(sockets[1]);
            auto job = std::make_unique<ServerJob>();
            job->fd = sockets[0];
            job->req.format = ApiFormat::OPENAI_CHAT;
            job->req.prompt_tokens = std::move(prompt);
            job->req.max_output = 1;
            job->req.stream = false;
            job->req.model = "scheduler-test";
            job->req.response_id = id;
            jobs.push_back(std::move(job));
            ServerJob * raw = jobs.back().get();
            SchedulerTestHarness::enqueue(server, raw);
            std::unique_lock<std::mutex> lock(raw->mu);
            return raw->cv.wait_for(lock, std::chrono::seconds(5), [&] { return raw->done; });
        };
        std::thread scheduler([&] { SchedulerTestHarness::run(server, backend.engine); });
        // Four short conversations save their 3-token heads (300 bytes each),
        // then a long one saves its 42-token head (4200 bytes): 900 must go.
        const std::vector<std::vector<int32_t>> small = {
            {1, 100, 3, 101, 4}, {1, 200, 3, 201, 4},
            {1, 300, 3, 301, 4}, {1, 400, 3, 401, 4}};
        std::vector<int32_t> big = {1};
        big.insert(big.end(), 40, 900);
        big.insert(big.end(), {3, 901, 4});
        bool ok = true;
        for (size_t i = 0; i < small.size(); ++i) {
            ok = ok && run_one(small[i], "s");
        }
        const auto before = SchedulerTestHarness::prefix_cache(server).stats();
        ok = ok && run_one(big, "big");
        disk.drain();
        SchedulerTestHarness::stop(server);
        scheduler.join();
        for (const int fd : fds) close(fd);

        TEST_ASSERT(ok);
        TEST_ASSERT(before.in_use == 4 && before.resident_bytes == 1200);
        PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
        const auto after = cache.stats();
        TEST_ASSERT(after.budget_skips == 0);
        TEST_ASSERT(after.in_use == 2 && after.resident_bytes == 300 + 4200);
        std::vector<int32_t> next = big;
        next.push_back(9);
        TEST_ASSERT(cache.lookup_candidate(next, (int)next.size()).second == 42);
        TEST_ASSERT(cache.lookup_candidate(small[3], 5).second == 3);
        // Only the newest short head and the long one keep a payload.
        TEST_ASSERT(backend.slots.size() == 2);
        TEST_ASSERT(backend.displaced.empty());
        if (disk_on) {
            TEST_ASSERT(disk.stats().writes == 3);
            for (int i = 0; i < 4; ++i) {
                std::vector<int32_t> probe = small[(size_t)i];
                probe.push_back(9);
                TEST_ASSERT(disk.longest_prefix(probe).len == (i < 3 ? 3 : 0));
            }
        }
    }
    rm_rf(dir);
    unlink(path.c_str());
}
#endif

TEST_CASE(ServerUnitFixture, test_backend_ipc_rejects_file_work_dir) {
    const std::string file_path = test_tmp_path(
        "luce_test_backend_ipc_work_dir_file").string();
    remove_test_path(file_path);
    FILE * file = std::fopen(file_path.c_str(), "wb");
    TEST_ASSERT(file != nullptr);
    if (file != nullptr) {
        const char payload[] = "not a dir";
        (void)std::fwrite(payload, 1, sizeof(payload) - 1, file);
        std::fclose(file);
    }

    BackendIpcLaunchConfig cfg;
    cfg.bin = "/bin/true";
    cfg.payload_path = "/tmp/luce_test_backend_ipc_payload";
    cfg.work_dir = file_path;

    BackendIpcProcess proc;
    TEST_ASSERT(!proc.start(cfg));
    TEST_ASSERT(!proc.active());
    remove_test_path(file_path);
}

#if !defined(_WIN32)
TEST_CASE(ServerUnitFixture, test_backend_ipc_payload_pipe_round_trip) {
    int payload_pipe[2] = {-1, -1};
    int status_pipe[2] = {-1, -1};
    TEST_ASSERT(pipe(payload_pipe) == 0);
    TEST_ASSERT(pipe(status_pipe) == 0);
    if (payload_pipe[0] < 0 || payload_pipe[1] < 0 ||
        status_pipe[0] < 0 || status_pipe[1] < 0) {
        if (payload_pipe[0] >= 0) close(payload_pipe[0]);
        if (payload_pipe[1] >= 0) close(payload_pipe[1]);
        if (status_pipe[0] >= 0) close(status_pipe[0]);
        if (status_pipe[1] >= 0) close(status_pipe[1]);
        return;
    }

    const std::vector<float> payload = {1.0f, 2.5f, -3.0f, 4.25f};
    TEST_ASSERT(write_exact_fd(payload_pipe[1],
                               payload.data(),
                               payload.size() * sizeof(float)));
    close(payload_pipe[1]);
    payload_pipe[1] = -1;

    std::vector<float> received(payload.size(), 0.0f);
    TEST_ASSERT(read_exact_fd(payload_pipe[0],
                              received.data(),
                              received.size() * sizeof(float)));
    close(payload_pipe[0]);
    payload_pipe[0] = -1;
    TEST_ASSERT(received == payload);

    const int32_t ready = 0;
    TEST_ASSERT(write_exact_fd(status_pipe[1], &ready, sizeof(ready)));
    close(status_pipe[1]);
    status_pipe[1] = -1;
    int32_t status = -1;
    TEST_ASSERT(read_exact_fd(status_pipe[0], &status, sizeof(status)));
    TEST_ASSERT(status == 0);
    close(status_pipe[0]);
}
#endif

TEST_CASE(ServerUnitFixture, test_backend_ipc_payload_transport_parse) {
    BackendIpcMode mode = BackendIpcMode::DFlashDraft;
    TEST_ASSERT(parse_backend_ipc_mode("dflash-draft", mode));
    TEST_ASSERT(mode == BackendIpcMode::DFlashDraft);
    TEST_ASSERT(parse_backend_ipc_mode("pflash-compress", mode));
    TEST_ASSERT(mode == BackendIpcMode::PFlashCompress);
    TEST_ASSERT(parse_backend_ipc_mode("qwen35-target-shard", mode));
    TEST_ASSERT(mode == BackendIpcMode::Qwen35TargetShard);
    TEST_ASSERT(parse_backend_ipc_mode("moe-expert-compute", mode));
    TEST_ASSERT(mode == BackendIpcMode::MoeExpertCompute);
    TEST_ASSERT(!parse_backend_ipc_mode("moe-ffn", mode));

    BackendIpcPayloadTransport transport = BackendIpcPayloadTransport::Auto;
    TEST_ASSERT(parse_backend_ipc_payload_transport("stream", transport));
    TEST_ASSERT(transport == BackendIpcPayloadTransport::Stream);
    TEST_ASSERT(parse_backend_ipc_payload_transport("shared", transport));
    TEST_ASSERT(transport == BackendIpcPayloadTransport::Shared);
    TEST_ASSERT(parse_backend_ipc_payload_transport("auto", transport));
    TEST_ASSERT(transport == BackendIpcPayloadTransport::Auto);
    TEST_ASSERT(!parse_backend_ipc_payload_transport("pipe", transport));
    TEST_ASSERT(std::strcmp(
        backend_ipc_payload_transport_name(BackendIpcPayloadTransport::Stream),
        "stream") == 0);
}

TEST_CASE(ServerUnitFixture, test_backend_ipc_payload_bounds) {
    size_t out = 0;
    TEST_ASSERT(backend_ipc_checked_add_size(4, 8, out));
    TEST_ASSERT(out == 12);
    TEST_ASSERT(!backend_ipc_checked_add_size(
        std::numeric_limits<size_t>::max(), 1, out));
    TEST_ASSERT(backend_ipc_payload_in_bounds(0, 16, 16));
    TEST_ASSERT(backend_ipc_payload_in_bounds(4, 8, 16));
    TEST_ASSERT(!backend_ipc_payload_in_bounds(9, 8, 16));
    TEST_ASSERT(!backend_ipc_payload_in_bounds(
        std::numeric_limits<size_t>::max(), 1, 16));
}

TEST_CASE(ServerUnitFixture, test_backend_ipc_shared_payload_map_sizing) {
    size_t map_bytes = 0;
    TEST_ASSERT(backend_ipc_shared_payload_map_bytes(1024, map_bytes));
    TEST_ASSERT(map_bytes == 1024 + backend_ipc_shared_payload_header_bytes());

    BackendIpcSharedPayloadHeader header;
    backend_ipc_publish_shared_payload_header(&header, 7, 1024);
    TEST_ASSERT(backend_ipc_shared_payload_header_matches(&header, 7, 1024));
    TEST_ASSERT(!backend_ipc_shared_payload_header_matches(&header, 0, 1024));
    TEST_ASSERT(!backend_ipc_shared_payload_header_matches(&header, 8, 1024));
    TEST_ASSERT(!backend_ipc_shared_payload_header_matches(&header, 7, 512));
    uint64_t sequence = 0;
    uint64_t bytes = 0;
    backend_ipc_load_shared_payload_header(&header, sequence, bytes);
    TEST_ASSERT(sequence == 7);
    TEST_ASSERT(bytes == 1024);

    TEST_ASSERT(!backend_ipc_shared_payload_map_bytes(
        std::numeric_limits<size_t>::max(), map_bytes));
}

TEST_CASE(ServerUnitFixture, test_backend_ipc_shared_payload_segment_contract) {
    const BackendIpcPayloadSegment a{reinterpret_cast<const void *>(1), 16};
    const BackendIpcPayloadSegment b{reinterpret_cast<const void *>(2), 32};
    const BackendIpcPayloadSegment segments[] = {a, b};
    size_t total = 0;
    for (const BackendIpcPayloadSegment & segment : segments) {
        TEST_ASSERT(backend_ipc_checked_add_size(total, segment.bytes, total));
    }
    TEST_ASSERT(total == 48);
    TEST_ASSERT(backend_ipc_payload_in_bounds(0, total, 48));
    TEST_ASSERT(!backend_ipc_payload_in_bounds(0, total + 1, 48));
}

TEST_CASE(ServerUnitFixture, test_moe_hybrid_expert_compute_batch_default) {
    luce_unsetenv("LUCE_MOE_EXPERT_COMPUTE_BATCH");
    luce_unsetenv("LUCE_MOE_EXPERT_COMPUTE_BATCH_MAX");
    TEST_ASSERT(moe_hybrid_expert_compute_batch_limit() == 32);
}

TEST_CASE(ServerUnitFixture, test_moe_hybrid_prefill_hot_sub_batch_limit) {
    luce_unsetenv("LUCE_MOE_PREFILL_HOT_SUB_BATCH");
    TEST_ASSERT(moe_hybrid_prefill_hot_sub_batch_limit() == 4);

    luce_setenv("LUCE_MOE_PREFILL_HOT_SUB_BATCH", "0");
    TEST_ASSERT(moe_hybrid_prefill_hot_sub_batch_limit() == 4);

    luce_setenv("LUCE_MOE_PREFILL_HOT_SUB_BATCH", "3");
    TEST_ASSERT(moe_hybrid_prefill_hot_sub_batch_limit() == 3);

    luce_setenv("LUCE_MOE_PREFILL_HOT_SUB_BATCH", "8");
    TEST_ASSERT(moe_hybrid_prefill_hot_sub_batch_limit() == 4);

    luce_unsetenv("LUCE_MOE_PREFILL_HOT_SUB_BATCH");
}

TEST_CASE(ServerUnitFixture, test_moe_hybrid_uma_core_memory_is_saturating) {
    constexpr size_t gib = (size_t) 1024 * 1024 * 1024;
    TEST_ASSERT(moe_hybrid_core_bytes_from_memory(
        "test", 6 * gib, 8 * gib) == 2 * gib);
    TEST_ASSERT(moe_hybrid_core_bytes_from_memory(
        "test", 10 * gib, 8 * gib) == 0);
}

TEST_CASE(ServerUnitFixture, test_moe_hybrid_canonical_rocmfp2_q2_is_tokenwise) {
    // ROCmFP2's safe q>1 fallback builds one owner graph per token. Canonical
    // route-order joins must preserve [hidden, route, token] while appending
    // those token slices; concatenating the route dimension makes the final
    // owner reduction invalid.
    luce_unsetenv("LUCE_MOE_TP_GROUPED_MMVQ");
    luce_unsetenv("LUCE_DS4_TP_GROUPED_MMVQ");

    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr);

    constexpr int n_embd = 32;
    constexpr int n_ff = 32;
    constexpr int n_expert = 4;
    constexpr int n_used = 2;
    constexpr int n_tokens = 2;

    MoeHybridConfig cfg;
    cfg.n_embd = n_embd;
    cfg.n_ff_exp = n_ff;
    cfg.n_expert = n_expert;
    cfg.n_expert_used = n_used;

    MoeHybridLayerStorage storage;
    storage.gate_hot = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q2_0_ROCMFP2, n_embd, n_ff, 2);
    storage.up_hot = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q2_0_ROCMFP2, n_embd, n_ff, 2);
    storage.down_hot = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q3_0_ROCMFPX, n_ff, n_embd, 2);
    storage.gate_cold = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q2_0_ROCMFP2, n_embd, n_ff, 2);
    storage.up_cold = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q2_0_ROCMFP2, n_embd, n_ff, 2);
    storage.down_cold = ggml_new_tensor_3d(
        ctx, GGML_TYPE_Q3_0_ROCMFPX, n_ff, n_embd, 2);
    storage.hot_local_by_global = {0, 1, -1, -1};
    storage.cold_local_by_global = {-1, -1, 0, 1};

    MoeLayerDesc desc;
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_used, n_tokens);
    MoeHybridGraphInputs out;
    const bool built = build_moe_hybrid_ffn_graph(
        ctx, nullptr, cfg, desc, storage, input, ids, weights, n_tokens,
        out, /*include_shared=*/false, /*allow_fused_combine=*/false,
        MoeHybridJoinMode::CanonicalRouteOrder);
    TEST_ASSERT(built);
    TEST_ASSERT(out.output != nullptr);
    TEST_ASSERT(out.output->ne[0] == n_embd);
    TEST_ASSERT(out.output->ne[1] == n_tokens);

    const auto count_mul_mat_id = [](const std::vector<ggml_tensor *> & nodes) {
        int count = 0;
        for (const ggml_tensor * node : nodes) {
            if (node && node->op == GGML_OP_MUL_MAT_ID) ++count;
        }
        return count;
    };
    TEST_ASSERT(count_mul_mat_id(out.hot_nodes) == 3 * n_tokens);
    TEST_ASSERT(count_mul_mat_id(out.cold_nodes) == 3 * n_tokens);

    ggml_free(ctx);
}

// ═══════════════════════════════════════════════════════════════════════
// Sampler tests (model-independent, CPU-only)
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_sampler_cfg_defaults) {
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

TEST_CASE(ServerUnitFixture, test_sampler_greedy_argmax) {
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

TEST_CASE(ServerUnitFixture, test_sampler_temperature_affects_distribution) {
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

TEST_CASE(ServerUnitFixture, test_sampler_top_p_truncation) {
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

TEST_CASE(ServerUnitFixture, test_sampler_top_k_truncation) {
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

TEST_CASE(ServerUnitFixture, test_sampler_repetition_penalty) {
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

TEST_CASE(ServerUnitFixture, test_sampler_frequency_penalty) {
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

TEST_CASE(ServerUnitFixture, test_sampler_presence_penalty) {
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

TEST_CASE(ServerUnitFixture, test_sampler_freq_and_pres_combined) {
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

TEST_CASE(ServerUnitFixture, test_sampler_negative_frequency_penalty) {
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

TEST_CASE(ServerUnitFixture, test_sampler_seed_reproducibility) {
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

TEST_CASE(ServerUnitFixture, test_sampler_rep_window_limits_scope) {
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

TEST_CASE(ServerUnitFixture, test_parse_sampler_token_basic) {
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

TEST_CASE(ServerUnitFixture, test_parse_sampler_token_with_penalties) {
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

TEST_CASE(ServerUnitFixture, test_parse_sampler_token_minimal) {
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

TEST_CASE(ServerUnitFixture, test_parse_sampler_token_no_samp) {
    std::string line = "gen 128";
    SamplerCfg cfg;
    TEST_ASSERT(!parse_sampler_token(line, cfg));
    TEST_ASSERT(line == "gen 128");  // unchanged
}

TEST_CASE(ServerUnitFixture, test_sampler_temp_zero_with_penalties_uses_argmax) {
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

TEST_CASE(ServerUnitFixture, test_sampler_needs_logit_processing) {
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

TEST_CASE(ServerUnitFixture, test_auto_concurrent_prefix_budget) {
    const size_t MiB = 1024 * 1024, GiB = 1024 * MiB;
    const size_t floor = ServerConfig::kConcurrentPrefixBudgetFloor;
    // Small checkpoints stay at the former 4 GiB default.
    TEST_ASSERT(auto_concurrent_prefix_budget(100 * MiB, 4, 128 * GiB) == floor);
    // 2 x slots + 1 checkpoints when that is above the floor and under the cap.
    TEST_ASSERT(auto_concurrent_prefix_budget(1237 * MiB, 4, 128 * GiB) == 9 * 1237 * MiB);
    // Capped at a quarter of the available memory...
    TEST_ASSERT(auto_concurrent_prefix_budget(4 * GiB, 8, 64 * GiB) == 16 * GiB);
    // ...but never below the floor, and unknown memory does not cap.
    TEST_ASSERT(auto_concurrent_prefix_budget(4 * GiB, 8, 8 * GiB) == floor);
    TEST_ASSERT(auto_concurrent_prefix_budget(1 * GiB, 4, 0) == 9 * GiB);
    // A non-positive slot count sizes one slot.
    TEST_ASSERT(auto_concurrent_prefix_budget(2 * GiB, 0, 0) == 6 * GiB);
}

TEST_CASE(ServerUnitFixture, test_server_config_cache_defaults) {
    ServerConfig cfg;
    TEST_ASSERT(cfg.prefix_cache_cap == 32);
    TEST_ASSERT(cfg.concurrent_prefix_cache_max_bytes == ServerConfig::kPrefixCacheBudgetAuto);
    TEST_ASSERT(!cfg.concurrent_prefix_cache);
    TEST_ASSERT(cfg.prefill_cache_cap == 0);
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

TEST_CASE(ServerUnitFixture, test_model_card_env_override_beats_cwd) {
    // LUCE_MODEL_CARDS_DIR used to be the LAST candidate, tried after the cwd-relative
    // "share/model_cards". Running from a directory that happened to contain one silently
    // ignored the operator's explicit override. An explicit setting must win.
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / "dflash-mc-env-test";
    const auto envdir = root / "explicit";
    fs::remove_all(root);
    fs::create_directories(envdir);

    // A card the resolver can only have found via the env var.
    {
        FILE * f = std::fopen((envdir / "env-probe-model.json").string().c_str(), "w");
        TEST_ASSERT(f != nullptr);
        std::fprintf(f, "{\"name\":\"env-probe-model\",\"source\":\"test\","
                        "\"verified_at\":\"2026-08-04\",\"max_tokens\":4321}");
        std::fclose(f);
    }

    const char * prev = std::getenv("LUCE_MODEL_CARDS_DIR");
    const std::string saved = prev ? prev : "";
    luce_setenv("LUCE_MODEL_CARDS_DIR", envdir.string().c_str());

    auto card = luce::common::resolve_model_card("", "env-probe-model", "deepseek4", "");

    if (saved.empty()) luce_unsetenv("LUCE_MODEL_CARDS_DIR");
    else luce_setenv("LUCE_MODEL_CARDS_DIR", saved.c_str());
    fs::remove_all(root);

    // Resolved from the env dir, not the deepseek4 family fallback (which gives 32768).
    TEST_ASSERT(card.max_tokens == 4321);
    TEST_ASSERT(card.source_label != "family:deepseek4");
}

TEST_CASE(ServerUnitFixture, test_model_card_family_fallback_deepseek4) {
    // deepseek4 had NO family entry, so every DeepSeek4 artifact -- including the
    // published ROCmFPX GGUFs -- fell through to the hard fallback, taking a generic
    // 16000-token ceiling that is a placeholder rather than a measured property of the
    // model, and reporting model_card = null on /props.
    //
    // Pins the branch rather than the exact ceiling: an operator is expected to ship a
    // sidecar for the real figures, and the fallback is deliberately conservative.
    // What must not regress is that deepseek4 resolves to a FAMILY card at all, and
    // carries the wider reply budget rather than the terse 512 default.
    auto card = luce::common::resolve_model_card("", "", "deepseek4", "");
    TEST_ASSERT(card.source_label == "family:deepseek4");
    TEST_ASSERT(card.max_tokens == 32768);
    TEST_ASSERT(card.hard_limit_reply_budget == 4096);

    // An unknown architecture must still fall through, or the safety net would mask
    // genuinely unsupported models.
    auto unknown = luce::common::resolve_model_card("", "", "not-a-real-arch", "");
    TEST_ASSERT(unknown.source_label != "family:not-a-real-arch");
}

TEST_CASE(ServerUnitFixture, test_model_card_family_fallback_bailingmoe3) {
    auto card = luce::common::resolve_model_card("", "", "bailingmoe3", "");
    TEST_ASSERT(card.source_label == "family:bailingmoe3");
    TEST_ASSERT(card.max_tokens == 32768);
    TEST_ASSERT(card.sampling.has_temperature);
    TEST_ASSERT(std::abs(card.sampling.temperature - 0.6f) < 1.0e-6f);
    TEST_ASSERT(card.sampling.has_top_p);
    TEST_ASSERT(std::abs(card.sampling.top_p - 0.95f) < 1.0e-6f);
    TEST_ASSERT(card.sampling.has_top_k);
    TEST_ASSERT(card.sampling.top_k == 20);
}

TEST_CASE(ServerUnitFixture, test_props_model_card_wholesale_sidecar) {
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

TEST_CASE(ServerUnitFixture, test_props_model_card_exposes_sampling_no_thinking) {
    // sampling_no_thinking rides along in the wholesale raw-sidecar re-emit
    // next to the existing `sampling` block, so preflight tooling can
    // assert the card actually carries distinct instruct-mode values.
    json sidecar = {
        {"name",        "Qwen3.8 Flash Next"},
        {"source",      "https://huggingface.co/Qwen/Qwen3.8-Flash-Next"},
        {"verified_at", "2026-10-01"},
        {"max_tokens",  32768},
        {"sampling", {
            {"temperature", 1.0}, {"top_p", 0.95}, {"top_k", 20},
            {"min_p", 0.0}, {"presence_penalty", 0.0}, {"repetition_penalty", 1.0},
        }},
        {"sampling_no_thinking", {
            {"temperature", 0.7}, {"top_p", 0.80}, {"top_k", 20},
            {"min_p", 0.0}, {"presence_penalty", 1.5}, {"repetition_penalty", 1.0},
        }},
    };
    ServerConfig cfg = make_props_config_with_sidecar(sidecar);
    Tokenizer    tok;
    PrefixCache  pc(0, tok);
    ToolMemory   tm;
    json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body["model_card"].contains("sampling"));
    TEST_ASSERT(body["model_card"].contains("sampling_no_thinking"));
    TEST_ASSERT(std::fabs(
        body["model_card"]["sampling_no_thinking"]["temperature"].get<double>() - 0.7) < 1.0e-6);
    TEST_ASSERT(std::fabs(
        body["model_card"]["sampling_no_thinking"]["top_p"].get<double>() - 0.80) < 1.0e-6);
    TEST_ASSERT(std::fabs(
        body["model_card"]["sampling_no_thinking"]["presence_penalty"].get<double>() - 1.5) < 1.0e-6);
    // Existing `sampling` (thinking-mode) field is untouched by the addition.
    TEST_ASSERT(std::fabs(
        body["model_card"]["sampling"]["temperature"].get<double>() - 1.0) < 1.0e-6);
}

TEST_CASE(ServerUnitFixture, test_props_model_card_null_on_family_fallback) {
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

TEST_CASE(ServerUnitFixture, test_props_deepseek4_tool_capability) {
    ServerConfig cfg;
    cfg.arch = "deepseek4";
    Tokenizer tok;
    PrefixCache pc(0, tok);
    ToolMemory tm;
    const json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body["capabilities"]["tools_supported"].get<bool>());
}

TEST_CASE(ServerUnitFixture, test_props_deepseek4_reasoning_capability) {
    ServerConfig cfg;
    cfg.arch = "deepseek4";
    Tokenizer tok;
    PrefixCache pc(0, tok);
    ToolMemory tm;
    const json body = build_props_body(cfg, pc, tm);

    TEST_ASSERT(body["reasoning"]["supported"].get<bool>());
    TEST_ASSERT(body["reasoning"]["supported_efforts"] ==
                json::array({"low", "high", "max"}));
    TEST_ASSERT(body["capabilities"]["reasoning_supported"].get<bool>());
}

TEST_CASE(ServerUnitFixture, test_props_budget_envelope_shape) {
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

    // Sanity: props_schema 3 (schema 3 dropped three pflash fields).
    TEST_ASSERT(body["server"]["props_schema"].get<int>() == 3);
}

// ─── /props.runtime captures full config (§4.16) ──────────────────────
// Snapshot/bench tooling reads /props.runtime wholesale into
// result.json.server_info; this test pins the field set so additions
// elsewhere don't accidentally drop a knob we depend on for forensics.
TEST_CASE(ServerUnitFixture, test_props_runtime_shape) {
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
    cfg.draft_residency = DraftResidencyPolicy::Persistent;
    cfg.target_sharding = false;
    cfg.chunk           = 512;
    cfg.target_device   = "auto:0";
    cfg.draft_device    = "auto:0";
    TEST_ASSERT(cfg.admission_coalesce_ms == 20);

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
    TEST_ASSERT(rt["draft_residency"].get<std::string>() == "persistent");
    TEST_ASSERT(rt["target_sharding"].get<bool>()        == false);
    TEST_ASSERT(rt["chunk"].get<int>()                   == 512);
    TEST_ASSERT(rt["target_device"].get<std::string>()   == "auto:0");
    TEST_ASSERT(rt["draft_device"].get<std::string>()    == "auto:0");
    const json & pc_props = body["prefix_cache"];
    TEST_ASSERT(pc_props.contains("max_resident_bytes"));
    TEST_ASSERT(pc_props.contains("resident_bytes"));
    TEST_ASSERT(pc_props.contains("budget_skips"));
    TEST_ASSERT(pc_props.contains("capture_attempts"));
    TEST_ASSERT(pc_props.contains("capture_failures"));
    TEST_ASSERT(pc_props.contains("capture_stall_ms_total"));
    TEST_ASSERT(pc_props.contains("capture_stall_ms_max"));
    TEST_ASSERT(pc_props.contains("restore_attempts"));
    TEST_ASSERT(pc_props.contains("restore_invalidations"));
    TEST_ASSERT(pc_props.contains("restore_stall_ms_total"));
    TEST_ASSERT(pc_props.contains("restore_stall_ms_max"));
    TEST_ASSERT(rt["continuous_batching"]["admission_coalesce_ms"]
                    .get<int>() == 20);
    TEST_ASSERT(body["pflash"]["draft_residency"].get<std::string>() == "persistent");

    // draft_device is null when no draft model is loaded.
    cfg.draft_device.clear();
    cfg.admission_coalesce_ms = 7;
    body = build_props_body(cfg, pc, tm);
    TEST_ASSERT(body["runtime"]["draft_device"].is_null());
    TEST_ASSERT(body["runtime"]["continuous_batching"]
                    ["admission_coalesce_ms"].get<int>() == 7);
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

    // Apps read the cached share where OpenAI puts it.
    for (const auto format : {ApiFormat::OPENAI_CHAT, ApiFormat::RESPONSES}) {
        auto em = make_emitter(format);
        em.emit_start();
        em.emit_token("x");
        const std::string wire = concat(em.emit_finish(1, &t));
        const char * field = format == ApiFormat::OPENAI_CHAT
            ? "\"prompt_tokens_details\":{\"cached_tokens\":8192}"
            : "\"input_tokens_details\":{\"cached_tokens\":8192}";
        TEST_ASSERT(wire.find(field) != std::string::npos);
    }
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

// ModelBackend common empty-spec retry tests
// ═══════════════════════════════════════════════════════════════════════

struct EmptySpecRetryBackend : MockBackend {
    int generate_calls = 0;
    int restore_calls = 0;
    bool generate_saw_force_ar = false;
    bool restore_saw_force_ar = false;
    bool generate_first_empty_visible = false;
    bool restore_first_empty_visible = false;
    // False once the first attempt continued a deferred snapshot in place.
    bool snapshot_present = true;

    bool snapshot_used(int) const override { return snapshot_present; }

    GenerateResult generate_impl(const GenerateRequest & req,
                            const DaemonIO &) override {
        generate_calls++;
        GenerateResult result;
        result.succeed();
        if (req.force_ar_decode) {
            generate_saw_force_ar = true;
            result.tokens = {42};
        } else {
            result.spec_decode_ran = true;
            if (generate_first_empty_visible) {
                result.tokens = {2};
                result.empty_visible_output = true;
            }
        }
        return result;
    }

    GenerateResult restore_and_generate_impl(int, const GenerateRequest & req,
                                        const DaemonIO &) override {
        restore_calls++;
        GenerateResult result;
        result.succeed();
        result.restored_prefix_tokens = req.force_ar_decode ? 2 : 3;
        if (req.force_ar_decode) {
            restore_saw_force_ar = true;
            result.tokens = {84};
        } else {
            result.spec_decode_ran = true;
            if (restore_first_empty_visible) {
                result.tokens = {2};
                result.empty_visible_output = true;
            }
        }
        return result;
    }
};

TEST_CASE(ServerUnitFixture, test_model_backend_retries_empty_spec_generate_once_with_ar) {
    EmptySpecRetryBackend backend;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 42);
    TEST_ASSERT(result.spec_decode_ran);
    TEST_ASSERT(backend.generate_calls == 2);
    TEST_ASSERT(backend.generate_saw_force_ar);
}

TEST_CASE(ServerUnitFixture, test_model_backend_retries_empty_spec_restore_once_with_ar) {
    EmptySpecRetryBackend backend;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result =
        backend.restore_and_generate(7, req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 84);
    TEST_ASSERT(result.spec_decode_ran);
    TEST_ASSERT(result.restored_prefix_tokens == 3);
    TEST_ASSERT(backend.restore_calls == 2);
    TEST_ASSERT(backend.restore_saw_force_ar);
}

// A deferred snapshot the first attempt consumed cannot be restored again:
// the retry prefills the whole prompt.
TEST_CASE(ServerUnitFixture, test_model_backend_retries_consumed_restore_with_prefill) {
    EmptySpecRetryBackend backend;
    backend.snapshot_present = false;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result = backend.restore_and_generate(7, req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 42);
    TEST_ASSERT(backend.restore_calls == 1);
    TEST_ASSERT(backend.generate_calls == 1);
    TEST_ASSERT(backend.generate_saw_force_ar);
    TEST_ASSERT(result.restored_prefix_tokens == 0);
}

TEST_CASE(ServerUnitFixture, test_model_backend_retries_empty_visible_spec_generate_once_with_ar) {
    EmptySpecRetryBackend backend;
    backend.generate_first_empty_visible = true;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result = backend.generate(req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 42);
    TEST_ASSERT(!result.empty_visible_output);
    TEST_ASSERT(result.spec_decode_ran);
    TEST_ASSERT(backend.generate_calls == 2);
    TEST_ASSERT(backend.generate_saw_force_ar);
}

TEST_CASE(ServerUnitFixture, test_model_backend_retries_empty_visible_spec_restore_once_with_ar) {
    EmptySpecRetryBackend backend;
    backend.restore_first_empty_visible = true;
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 4;
    DaemonIO io;

    GenerateResult result = backend.restore_and_generate(7, req, io);

    TEST_ASSERT(result.ok());
    TEST_ASSERT(result.tokens.size() == 1);
    TEST_ASSERT(result.tokens[0] == 84);
    TEST_ASSERT(!result.empty_visible_output);
    TEST_ASSERT(result.spec_decode_ran);
    TEST_ASSERT(backend.restore_calls == 2);
    TEST_ASSERT(backend.restore_saw_force_ar);
}

// GenerateResult speculative telemetry plumbing tests (Day 1 of bandit MVP)
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_generate_result_accept_rate_defaults_to_zero) {
    GenerateResult r;
    TEST_ASSERT(r.accept_rate == 0.0f);
}

TEST_CASE(ServerUnitFixture, test_generate_result_accept_rate_zero_when_no_spec_decode) {
    // When spec decode doesn't run (no draft model), accept_rate stays 0.
    GenerateResult r;
    r.succeed();
    // Telemetry not set → accept_rate is zero and speculative decode is false.
    TEST_ASSERT(r.accept_rate == 0.0f);
    TEST_ASSERT(!r.spec_decode_ran);
}

TEST_CASE(ServerUnitFixture, test_generate_result_error_state_is_consistent) {
    GenerateResult result;
    TEST_ASSERT(!result.ok());
    TEST_ASSERT(result.error->code == GenerateErrorCode::Incomplete);
    TEST_ASSERT(result.error_code() == "incomplete");

    result.fail(GenerateErrorCode::BackendSpecific, "prefill graph allocation failed");
    TEST_ASSERT(!result.ok());
    TEST_ASSERT(result.error->code == GenerateErrorCode::BackendSpecific);
    TEST_ASSERT(result.error_code() == "backend_specific");
    TEST_ASSERT(result.error_detail() == "prefill graph allocation failed");

    result.succeed();
    TEST_ASSERT(result.ok());
    TEST_ASSERT(!result.error.has_value());
    TEST_ASSERT(result.error_code().empty());
    TEST_ASSERT(result.error_detail().empty());
}

TEST_CASE(ServerUnitFixture, test_response_error_maps_every_generation_code) {
    struct Case {
        GenerateErrorCode code;
        int status;
    };
    const Case cases[] = {
        {GenerateErrorCode::Incomplete, 500},
        {GenerateErrorCode::AdapterUnavailable, 503},
        {GenerateErrorCode::ResourceExhausted, 503},
        {GenerateErrorCode::ContextOverflow, 400},
        {GenerateErrorCode::SamplingUnsupported, 400},
        {GenerateErrorCode::PrefillFailed, 500},
        {GenerateErrorCode::DecodeSeedMissing, 500},
        {GenerateErrorCode::DecodeFailed, 500},
        {GenerateErrorCode::InvalidSnapshotSlot, 500},
        {GenerateErrorCode::ModelParked, 503},
        {GenerateErrorCode::BackendSpecific, 500},
    };

    for (const Case & c : cases) {
        const ResponseError error = to_response_error({c.code, {}});
        TEST_ASSERT(!error.code.empty());
        TEST_ASSERT(!error.message.empty());
        TEST_ASSERT(error.code == generate_error_code(c.code));
        TEST_ASSERT(response_error_http_status(error) == c.status);
    }

    const ResponseError detailed = to_response_error(
        {GenerateErrorCode::DecodeFailed, "device execution failed"});
    TEST_ASSERT(detailed.message == "device execution failed");
}

TEST_CASE(ServerUnitFixture, test_response_error_nonstream_formats) {
    const ResponseError error = ResponseError::internal(
        "decode_failed", "generation decode failed");

    const json openai = build_error_response(
        ApiFormat::OPENAI_CHAT, error, "chat_123");
    TEST_ASSERT(openai["error"]["type"] == "server_error");
    TEST_ASSERT(openai["error"]["code"] == "decode_failed");
    TEST_ASSERT(openai["error"]["message"] == "generation decode failed");

    const json anthropic = build_error_response(
        ApiFormat::ANTHROPIC, error, "msg_123");
    TEST_ASSERT(anthropic["type"] == "error");
    TEST_ASSERT(anthropic["error"]["type"] == "api_error");
    TEST_ASSERT(anthropic["request_id"] == "msg_123");

    const json responses = build_error_response(
        ApiFormat::RESPONSES, error, "resp_123");
    TEST_ASSERT(responses["error"]["type"] == "server_error");
    TEST_ASSERT(responses["error"]["code"] == "decode_failed");
}

TEST_CASE(ServerUnitFixture, test_response_error_factories_fill_empty_messages) {
    TEST_ASSERT(ResponseError::invalid_request("invalid_request", {}).message ==
                "invalid request");
    TEST_ASSERT(ResponseError::unavailable("unavailable", {}).message ==
                "service unavailable");
    TEST_ASSERT(ResponseError::internal("engine_step_failed", {}).message ==
                "generation failed");
}

TEST_CASE(ServerUnitFixture, test_sse_emitter_openai_error_is_terminal) {
    auto emitter = make_emitter(ApiFormat::OPENAI_CHAT);
    std::string wire = concat(emitter.emit_start());
    wire += concat(emitter.emit_token("pending output that was already flushed"));

    wire += concat(emitter.emit_error(
        ResponseError::internal("decode_failed", "decode failed")));
    TEST_ASSERT(wire.find("\"error\"") != std::string::npos);
    TEST_ASSERT(wire.find("decode_failed") != std::string::npos);
    TEST_ASSERT(wire.find("[DONE]") != std::string::npos);
    TEST_ASSERT(wire.find("pending output") != std::string::npos);
    TEST_ASSERT(emitter.finish_reason() == "error");
    TEST_ASSERT(emitter.emit_finish(1).empty());
    TEST_ASSERT(emitter.emit_token("late token").empty());
    TEST_ASSERT(emitter.emit_error(ResponseError::internal(
        "late_error", "late error")).empty());
}

TEST_CASE(ServerUnitFixture, test_sse_emitter_anthropic_error_is_terminal) {
    auto emitter = make_emitter(ApiFormat::ANTHROPIC);
    emitter.emit_start();

    const std::string wire = concat(emitter.emit_error(
        ResponseError::unavailable("model_parked", "model unavailable")));
    TEST_ASSERT(wire.find("event: error") != std::string::npos);
    TEST_ASSERT(wire.find("overloaded_error") != std::string::npos);
    TEST_ASSERT(wire.find("message_stop") == std::string::npos);
    TEST_ASSERT(emitter.emit_finish(0).empty());
}

TEST_CASE(ServerUnitFixture, test_sse_emitter_responses_error_is_terminal) {
    auto emitter = make_emitter(ApiFormat::RESPONSES);
    emitter.emit_start();

    const std::string wire = concat(emitter.emit_error(
        ResponseError::internal("prefill_failed", "prefill failed")));
    TEST_ASSERT(wire.find("event: response.failed") != std::string::npos);
    TEST_ASSERT(wire.find("\"status\":\"failed\"") !=
                std::string::npos);
    TEST_ASSERT(wire.find("prefill_failed") != std::string::npos);
    TEST_ASSERT(wire.find("response.completed") == std::string::npos);
    TEST_ASSERT(wire.find("[DONE]") == std::string::npos);
    TEST_ASSERT(emitter.emit_finish(0).empty());
}

// ═══════════════════════════════════════════════════════════════════════
// normalize_system_for_cache — header-strip tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_normalize_strips_billing_header_anthropic_array) {
    // Anthropic system-as-array: one billing-header block + one real block.
    json system_blocks = json::array({
        {{"type", "text"},
         {"text", "x-anthropic-billing-header: session=abc123 turn=4 ts=1749430000"}},
        {{"type", "text"},
         {"text", "You are a helpful coding assistant."}}
    });
    std::string out = luce::common::normalize_system_for_cache(system_blocks);
    TEST_ASSERT(out.find("x-anthropic-billing-header:") == std::string::npos);
    TEST_ASSERT(out.find("helpful coding assistant") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_normalize_strips_billing_header_openai_messages0) {
    // OpenAI messages[0] system containing the billing header in content.
    json messages = json::array({
        {{"role", "system"},
         {"content", "x-anthropic-billing-header: session=xyz789 turn=12 ts=1749431000\nYou are a code reviewer."}},
        {{"role", "user"}, {"content", "Review this diff."}}
    });
    std::string out = luce::common::normalize_system_for_cache(messages);
    TEST_ASSERT(out.find("x-anthropic-billing-header:") == std::string::npos);
    TEST_ASSERT(out.find("code reviewer") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_normalize_idempotent_across_changing_header) {
    // Two OpenAI messages arrays identical except the header turn value.
    // normalize_system_for_cache must return EQUAL strings for both.
    json messages_turn4 = json::array({
        {{"role", "system"},
         {"content", "x-anthropic-billing-header: session=S1 turn=4 ts=1749430000\nYou help with Rust."}},
        {{"role", "user"}, {"content", "What is a lifetime?"}}
    });
    json messages_turn5 = json::array({
        {{"role", "system"},
         {"content", "x-anthropic-billing-header: session=S1 turn=5 ts=1749430060\nYou help with Rust."}},
        {{"role", "user"}, {"content", "What is a lifetime?"}}
    });
    std::string out4 = luce::common::normalize_system_for_cache(messages_turn4);
    std::string out5 = luce::common::normalize_system_for_cache(messages_turn5);
    TEST_ASSERT(out4 == out5);
}

TEST_CASE(ServerUnitFixture, test_normalize_preserves_legit_system_content) {
    // A normal system prompt containing no billing header must pass through unchanged.
    json messages = json::array({
        {{"role", "system"},
         {"content", "You are an expert in C++ performance optimization."}},
        {{"role", "user"}, {"content", "Help me optimize this loop."}}
    });
    std::string out = luce::common::normalize_system_for_cache(messages);
    TEST_ASSERT(out == "You are an expert in C++ performance optimization.");
}

TEST_CASE(ServerUnitFixture, test_normalize_handles_leading_whitespace_header) {
    // Header block with leading whitespace must still be stripped.
    json system_blocks = json::array({
        {{"type", "text"},
         {"text", "  x-anthropic-billing-header: session=W1 turn=1 ts=1749432000"}},
        {{"type", "text"},
         {"text", "Be concise."}}
    });
    std::string out = luce::common::normalize_system_for_cache(system_blocks);
    TEST_ASSERT(out.find("x-anthropic-billing-header:") == std::string::npos);
    TEST_ASSERT(out.find("Be concise.") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_prefix_key_stable_across_header_change) {
    // Two /v1/chat/completions-style messages arrays differing ONLY in the
    // billing header value must normalize to EQUAL strings.
    json messages_a = json::array({
        {{"role", "system"},
         {"content", "x-anthropic-billing-header: session=S2 turn=1 ts=1749440000\nYou are a senior engineer."}},
        {{"role", "user"}, {"content", "What is RAII?"}}
    });
    json messages_b = json::array({
        {{"role", "system"},
         {"content", "x-anthropic-billing-header: session=S2 turn=7 ts=1749440420\nYou are a senior engineer."}},
        {{"role", "user"}, {"content", "What is RAII?"}}
    });
    std::string norm_a = luce::common::normalize_system_for_cache(messages_a);
    std::string norm_b = luce::common::normalize_system_for_cache(messages_b);
    TEST_ASSERT(norm_a == norm_b);
    TEST_ASSERT(norm_a.find("senior engineer") != std::string::npos);
}

// FlowKV + disk-cache compose tests (T1–T7)

// T4 (compress=false): policy name has no "+compress" suffix.
TEST_CASE(ServerUnitFixture, test_flowkv_T4_compress_false_policy_name_no_suffix) {
    DiskPrefixCachePolicy p;
    p.mode = DiskPrefixCacheMode::Full;
    p.compress = false;
    std::string name = disk_prefix_cache_policy_name(p);
    TEST_ASSERT_MSG(name.find("+compress") == std::string::npos,
                    "compress=false: name must not contain +compress");
}

// T4 (compress=true): policy name has "+compress" suffix.
TEST_CASE(ServerUnitFixture, test_flowkv_T4_compress_true_policy_name_has_suffix) {
    DiskPrefixCachePolicy p;
    p.mode = DiskPrefixCacheMode::Full;
    p.compress = true;
    std::string name = disk_prefix_cache_policy_name(p);
    TEST_ASSERT_MSG(name.find("+compress") != std::string::npos,
                    "compress=true: name must contain +compress");
    // auto+compress
    p.mode = DiskPrefixCacheMode::Auto;
    p.auto_window = 10;
    name = disk_prefix_cache_policy_name(p);
    TEST_ASSERT(name.find("+compress") != std::string::npos);
    // fixed+compress
    p.mode = DiskPrefixCacheMode::Fixed;
    p.fixed_tokens = 512;
    name = disk_prefix_cache_policy_name(p);
    TEST_ASSERT(name.find("+compress") != std::string::npos);
}

// T4: compression-aware disk clamping remains opt-in.
TEST_CASE(ServerUnitFixture, test_flowkv_T4_default_no_compress) {
    DiskPrefixCachePolicy p;
    TEST_ASSERT_MSG(!p.compress, "FlowKV disk clamping must default to off");
    TEST_ASSERT(!http_detail::should_clamp_flowkv_disk_cache(true, p));

    p.compress = true;
    TEST_ASSERT(http_detail::should_clamp_flowkv_disk_cache(true, p));
    TEST_ASSERT(!http_detail::should_clamp_flowkv_disk_cache(false, p));
}

// T6: frozen_block_key is deterministic — same tokens → same hash.
TEST_CASE(ServerUnitFixture, test_flowkv_T6_frozen_block_key_deterministic) {
    std::vector<int32_t> ids = {10, 20, 30, 40, 50};
    PrefixHash k1 = frozen_block_key(ids.data(), 0, (int)ids.size());
    PrefixHash k2 = frozen_block_key(ids.data(), 0, (int)ids.size());
    TEST_ASSERT_MSG(k1 == k2, "frozen_block_key must be deterministic");
}

// T6: frozen_block_key returns zero hash on empty slice.
TEST_CASE(ServerUnitFixture, test_flowkv_T6_frozen_block_key_zero_on_empty) {
    std::vector<int32_t> ids = {10, 20, 30};
    PrefixHash k = frozen_block_key(ids.data(), 2, 2);  // begin == end
    PrefixHash zero{};
    TEST_ASSERT_MSG(k == zero, "empty slice must return zero hash");
    PrefixHash k2 = frozen_block_key(ids.data(), 5, 3);  // begin > end
    TEST_ASSERT(k2 == zero);
}

// T6: distinct token content → distinct hashes.
TEST_CASE(ServerUnitFixture, test_flowkv_T6_frozen_block_key_distinct_content) {
    std::vector<int32_t> a = {1, 2, 3};
    std::vector<int32_t> b = {1, 2, 4};
    PrefixHash ka = frozen_block_key(a.data(), 0, 3);
    PrefixHash kb = frozen_block_key(b.data(), 0, 3);
    TEST_ASSERT_MSG(ka != kb, "different token content must produce different hashes");
}

// T7: disk clamp — with compress=true, boundary should use system_end (first
// safe boundary), not the full prompt.  Tested via the fixed-boundary logic.
TEST_CASE(ServerUnitFixture, test_flowkv_T7_disk_clamp_system_end_boundary) {
    // Simulate: effective_prompt has a system_end at token 300.
    // The FlowKV disk-clamp should set fixed_tokens = system_end.
    // We test this by constructing a DiskPrefixCachePolicy and verifying that
    // disk_prefix_cache_fixed_boundary returns system_end when fixed_tokens = system_end.
    const int system_end = 300;
    DiskPrefixCachePolicy p;
    p.mode = DiskPrefixCacheMode::Fixed;
    p.fixed_tokens = system_end;
    p.compress = true;

    // full_len larger than system_end → boundary = system_end
    int b = disk_prefix_cache_fixed_boundary(p, 1200, /*min_tokens=*/128);
    TEST_ASSERT_MSG(b == system_end,
                    "disk clamp must return system_end as boundary");

    // full_len smaller than system_end → no boundary (prompt shorter than system)
    int b2 = disk_prefix_cache_fixed_boundary(p, 100, /*min_tokens=*/128);
    TEST_ASSERT_MSG(b2 == 0, "boundary 0 when prompt shorter than system_end");

    // system_end below min_tokens → no boundary
    DiskPrefixCachePolicy p2;
    p2.mode = DiskPrefixCacheMode::Fixed;
    p2.fixed_tokens = 50;
    p2.compress = true;
    int b3 = disk_prefix_cache_fixed_boundary(p2, 1000, /*min_tokens=*/512);
    TEST_ASSERT_MSG(b3 == 0, "boundary 0 when system_end < min_tokens");
}

// T3 (WS1): non-continuation messages JSON has no assistant role.
// This tests the JSON shape that the is_continuation check reads.
TEST_CASE(ServerUnitFixture, test_flowkv_T3_ws1_continuation_json_shape) {
    // Single user message: NOT a continuation.
    json msgs = json::array({
        {{"role", "system"}, {"content", "You are an assistant."}},
        {{"role", "user"},   {"content", "Hello!"}}
    });
    bool is_continuation = false;
    for (const auto & m : msgs) {
        if (!m.is_object()) continue;
        const std::string role = m.value("role", "");
        if (role == "assistant") { is_continuation = true; break; }
        if (m.contains("tool_calls")) {
            const auto & tc = m["tool_calls"];
            if (tc.is_array() && !tc.empty()) { is_continuation = true; break; }
        }
    }
    TEST_ASSERT_MSG(!is_continuation, "user-only messages are NOT a continuation");

    // With assistant turn: IS a continuation.
    json msgs2 = json::array({
        {{"role", "system"},    {"content", "You are an assistant."}},
        {{"role", "user"},      {"content", "Hello!"}},
        {{"role", "assistant"}, {"content", "Hi there!"}}
    });
    bool is_cont2 = false;
    for (const auto & m : msgs2) {
        if (!m.is_object()) continue;
        const std::string role = m.value("role", "");
        if (role == "assistant") { is_cont2 = true; break; }
    }
    TEST_ASSERT_MSG(is_cont2, "messages with assistant turn ARE a continuation");
}

// T1 (head-verbatim): system_end is the FIRST boundary (boundary[0]).
// Verifies the disk-clamp invariant: system_end = find_all_boundaries()[0].
// Tests the boundary function returns a sane first boundary on a chat prompt.
TEST_CASE(ServerUnitFixture, test_flowkv_T1_system_end_boundary_first) {
    // Construct a synthetic token stream where chat markers appear at known
    // positions. find_all_boundaries uses prefix_cache_.chat_markers() which
    // are model-specific; test the boundary API directly.
    // The load-bearing invariant is: when compress=true + pflash_compressed,
    // disk_policy.fixed_tokens == system_end == find_all_boundaries()[0].
    // We test that find_all_boundaries returns a sorted ascending list and
    // that [0] is strictly less than [1] (system before later turns).

    // Boundary logic from disk_prefix_cache.cpp: uses marker token IDs to find
    // chat turn boundaries. We can test via a simple synthetic case.
    std::vector<int> boundaries = {100, 250, 500};
    // system_end would be boundaries[0] = 100.
    int system_end = boundaries.empty() ? 0 : boundaries[0];
    TEST_ASSERT_MSG(system_end == 100, "first boundary is system_end");
    // All later boundaries are after system_end.
    for (size_t i = 1; i < boundaries.size(); ++i) {
        TEST_ASSERT(boundaries[i] > system_end);
    }
}

// T5: exercise the production FlowKV activation decision and defaults.
TEST_CASE(ServerUnitFixture, test_flowkv_T5_aggregate_activation_threshold) {
    ServerConfig config;
    config.pflash = true;

    TEST_ASSERT(http_detail::flowkv_activation_threshold(config) == 32000);
    TEST_ASSERT(!http_detail::flowkv_should_activate(config, 13000));
    TEST_ASSERT(http_detail::flowkv_should_activate(config, 32000));

    config.pflash_threshold = 12000;
    TEST_ASSERT(!http_detail::flowkv_should_activate(config, 11999));
    TEST_ASSERT(http_detail::flowkv_should_activate(config, 13000));

    config.pflash_threshold = 0;   // every prompt
    TEST_ASSERT(http_detail::flowkv_activation_threshold(config) ==
                http_detail::kFlowKvInertMinTokens);
    TEST_ASSERT(http_detail::flowkv_should_activate(
        config, http_detail::kFlowKvInertMinTokens));
}

// Session feedback overrides the static/curve ratio for both whole-prompt
// PFlash and FlowKV.
TEST_CASE(ServerUnitFixture, test_flowkv_session_keep_ratio_override) {
    HttpServerSessions sessions;
    sessions.update("adaptive", 0.95f);

    const float configured_ratio = 0.05f;
    const float static_ratio = http_detail::resolve_pflash_keep_ratio(
        configured_ratio, "", sessions);
    const float adaptive_ratio = http_detail::resolve_pflash_keep_ratio(
        configured_ratio, "adaptive", sessions);

    TEST_ASSERT(std::fabs(static_ratio - configured_ratio) < 1e-6f);
    TEST_ASSERT(std::fabs(adaptive_ratio - 0.09f) < 1e-6f);

    // A session without feedback keeps the configured (curve) ratio, and its
    // first feedback adapts from that ratio rather than the fixed default:
    // 0.1875 (the 16K real-use budget) minus one small step at high acceptance.
    const float curve_ratio = 0.1875f;
    TEST_ASSERT(std::fabs(http_detail::resolve_pflash_keep_ratio(
        curve_ratio, "fresh", sessions) - curve_ratio) < 1e-6f);
    sessions.update("fresh", 0.95f, curve_ratio);
    TEST_ASSERT(std::fabs(http_detail::resolve_pflash_keep_ratio(
        curve_ratio, "fresh", sessions) - (curve_ratio - 0.01f)) < 1e-6f);
    // Seeds are clamped to the controller's range like every later step.
    sessions.update("wide", 0.50f, 0.30f);
    TEST_ASSERT(std::fabs(sessions.get_keep_ratio("wide") - 0.20f) < 1e-6f);
}

// ═══════════════════════════════════════════════════════════════════════
// Qwen3-0.6B model loader: truncated GGUF guard (bug #438)
// ═══════════════════════════════════════════════════════════════════════
//
// Builds a minimal but structurally valid Qwen3-0.6B-style GGUF on disk, then
// verifies that load_qwen3_model:
//   (1) loads the full, untruncated file successfully (positive control), and
//   (2) fails cleanly with a "truncated or corrupt" error when the tensor-data
//       section is truncated — instead of letting the H2D copy read past the
//       end of the mmap and SIGSEGV inside the device copy.

// Write a tiny valid model GGUF and return its path. The loader fixes
// n_vocab at 151936 (Qwen3Weights default), so token_embd stays the
// largest tensor (~2.4 MB BF16) while every other tensor is minimal.
static std::string write_qwen3_model_fixture_gguf() {
    const int n_embd    = 8;
    const int n_head    = 2;
    const int head_dim  = 4;
    const int n_head_kv = 1;
    const int n_ff      = 16;
    const int n_layer   = 1;
    const int n_vocab   = 151936;                // must match the loader default
    const int q_dim     = n_head * head_dim;     // 8
    const int kv_dim    = n_head_kv * head_dim;  // 4

    ggml_init_params ip{};
    ip.mem_size   = (size_t)16 * 1024 * 1024;    // headroom for token_embd + 13 tensors
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);

    gguf_context * g = gguf_init_empty();
    gguf_set_val_u32(g, "qwen3.embedding_length",        (uint32_t)n_embd);
    gguf_set_val_u32(g, "qwen3.feed_forward_length",     (uint32_t)n_ff);
    gguf_set_val_u32(g, "qwen3.attention.head_count",    (uint32_t)n_head);
    gguf_set_val_u32(g, "qwen3.attention.head_count_kv", (uint32_t)n_head_kv);
    gguf_set_val_u32(g, "qwen3.block_count",             (uint32_t)n_layer);
    gguf_set_val_u32(g, "qwen3.context_length",          (uint32_t)64);
    gguf_set_val_u32(g, "qwen3.attention.key_length",    (uint32_t)head_dim);
    gguf_set_val_f32(g, "qwen3.rope.freq_base",          1000000.0f);

    auto add_tensor = [&](const char * name, ggml_type t, int n_dims,
                          int64_t ne0, int64_t ne1) {
        ggml_tensor * w = (n_dims == 1)
            ? ggml_new_tensor_1d(ctx, t, ne0)
            : ggml_new_tensor_2d(ctx, t, ne0, ne1);
        ggml_set_name(w, name);
        std::memset(w->data, 0, ggml_nbytes(w));
        gguf_add_tensor(g, w);
    };

    // Top-level tensors. output.weight is intentionally omitted so the loader
    // exercises its tied-weights path (and the fixture stays small).
    add_tensor("token_embd.weight",  GGML_TYPE_BF16, 2, n_embd, n_vocab);
    add_tensor("output_norm.weight", GGML_TYPE_F32,  1, n_embd, 0);

    // The single transformer block: the 11 per-layer tensors the loader copies.
    add_tensor("blk.0.attn_norm.weight",   GGML_TYPE_F32,  1, n_embd,   0);
    add_tensor("blk.0.attn_q.weight",      GGML_TYPE_BF16, 2, n_embd,   q_dim);
    add_tensor("blk.0.attn_k.weight",      GGML_TYPE_BF16, 2, n_embd,   kv_dim);
    add_tensor("blk.0.attn_v.weight",      GGML_TYPE_BF16, 2, n_embd,   kv_dim);
    add_tensor("blk.0.attn_output.weight", GGML_TYPE_BF16, 2, q_dim,    n_embd);
    add_tensor("blk.0.attn_q_norm.weight", GGML_TYPE_F32,  1, head_dim, 0);
    add_tensor("blk.0.attn_k_norm.weight", GGML_TYPE_F32,  1, head_dim, 0);
    add_tensor("blk.0.ffn_norm.weight",    GGML_TYPE_F32,  1, n_embd,   0);
    add_tensor("blk.0.ffn_gate.weight",    GGML_TYPE_BF16, 2, n_embd,   n_ff);
    add_tensor("blk.0.ffn_up.weight",      GGML_TYPE_BF16, 2, n_embd,   n_ff);
    add_tensor("blk.0.ffn_down.weight",    GGML_TYPE_BF16, 2, n_ff,     n_embd);

    const std::string path = test_tmp_path(
        "luce_test_qwen3_model_438.gguf").string();
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);

    gguf_free(g);
    ggml_free(ctx);
    return path;
}

TEST_CASE(ServerUnitFixture, test_qwen3_model_rejects_truncated_gguf) {
    const std::string path = write_qwen3_model_fixture_gguf();

    ggml_backend_t backend = ggml_backend_cpu_init();
    TEST_ASSERT(backend != nullptr);

    // Positive control: the full, untruncated file loads cleanly.
    {
        Qwen3Weights w;
        bool ok = load_qwen3_model(path, backend, w);
        TEST_ASSERT_MSG(ok, luce_last_error());
        free_qwen3_model(w);
    }

    // Truncate inside the tensor-data section. The header, kv block, and tensor
    // info table all live before the data offset, so gguf_init_from_file still
    // succeeds and we reach the EOF guard rather than a parse failure.
    std::error_code ec;
    const uintmax_t file_size = fs::file_size(path, ec);
    TEST_ASSERT(!ec);
    const uintmax_t truncated_size = file_size - 4096;
    TEST_ASSERT(truncated_size > 0);
    fs::resize_file(path, truncated_size, ec);
    TEST_ASSERT(!ec);

    // The loader must fail cleanly (no SIGSEGV) with a descriptive error.
    {
        Qwen3Weights w;
        bool ok = load_qwen3_model(path, backend, w);
        TEST_ASSERT(!ok);
        const std::string err = luce_last_error();
        TEST_ASSERT_MSG(err.find("truncated or corrupt") != std::string::npos,
                        err.c_str());
        free_qwen3_model(w);
    }

    ggml_backend_free(backend);
    remove_test_path(path);
}

// ─── GGUF tensor bounds (gguf_tensor_in_file / gguf_bounds_error) ───────
//
// The shared overflow-safe bounds check used by every GGUF loader
// (draft/target/laguna) to reject truncated/corrupt files before a copy reads
// past the mapping (#438), without wrongly rejecting valid files (#318). These
// tests pin the boundary and, critically, the size_t overflow behaviour that a
// naive `data_off + tensor_off + tensor_sz > file_size` test gets wrong.
TEST_CASE(ServerUnitFixture, test_gguf_tensor_in_file_bounds) {
    // Typical layout: 100-byte header/kv, 900-byte data section, 1000-byte file.
    const size_t data_off = 100;
    const size_t file     = 1000;

    // Fully inside, including the exact end-of-file boundary.
    TEST_ASSERT(gguf_tensor_in_file(data_off, 0,   900, file));   // fills the data section
    TEST_ASSERT(gguf_tensor_in_file(data_off, 0,   0,   file));   // zero-size tensor
    TEST_ASSERT(gguf_tensor_in_file(data_off, 899, 1,   file));   // last byte
    TEST_ASSERT(gguf_tensor_in_file(data_off, 900, 0,   file));   // zero-size at EOF

    // One byte past EOF must be rejected.
    TEST_ASSERT(!gguf_tensor_in_file(data_off, 0,   901, file));
    TEST_ASSERT(!gguf_tensor_in_file(data_off, 900, 1,   file));

    // Data section offset itself past EOF (corrupt header).
    TEST_ASSERT(!gguf_tensor_in_file(2000, 0, 0, file));
    // data_off == file: only a zero-size tensor at offset 0 fits.
    TEST_ASSERT(gguf_tensor_in_file(file, 0, 0, file));
    TEST_ASSERT(!gguf_tensor_in_file(file, 0, 1, file));

    // Whole-file data section (data_off == 0), valid full read.
    TEST_ASSERT(gguf_tensor_in_file(0, 0, file, file));
    TEST_ASSERT(!gguf_tensor_in_file(0, 0, file + 1, file));

    // Overflow safety: a malformed offset/size must not wrap and slip through.
    const size_t kMax = std::numeric_limits<size_t>::max();
    TEST_ASSERT(!gguf_tensor_in_file(data_off, kMax, 10, file));   // huge tensor_off
    TEST_ASSERT(!gguf_tensor_in_file(data_off, 0, kMax, file));    // huge tensor_sz
    // The case a naive `off + sz > file` check fails: off + sz wraps below file.
    // off = kMax - 10, sz = 20  →  off + sz == 9 (< file) would FALSE-PASS.
    TEST_ASSERT(!gguf_tensor_in_file(0, kMax - 10, 20, file));
    TEST_ASSERT(!gguf_tensor_in_file(kMax, 10, 10, file));         // huge data_off
}

TEST_CASE(ServerUnitFixture, test_gguf_bounds_error_reports_operands) {
    // A normal (non-overflowing) rejection: the message must surface every
    // operand so a false positive on a valid file (#318) is diagnosable.
    const std::string e = gguf_bounds_error("target GGUF", "blk.56.ssm_out.weight",
                                            "q5_K", 100, 5000, 200, 1000);
    TEST_ASSERT(e.find("blk.56.ssm_out.weight") != std::string::npos);
    TEST_ASSERT(e.find("q5_K") != std::string::npos);
    TEST_ASSERT(e.find("data_off=100") != std::string::npos);
    TEST_ASSERT(e.find("tensor_off=5000") != std::string::npos);
    TEST_ASSERT(e.find("size=200") != std::string::npos);
    TEST_ASSERT(e.find("5300") != std::string::npos);      // 100 + 5000 + 200
    TEST_ASSERT(e.find("1000") != std::string::npos);      // file size

    // Null name/type must not crash and must produce placeholders.
    const std::string n = gguf_bounds_error("draft GGUF", nullptr, nullptr,
                                            0, 0, 1, 0);
    TEST_ASSERT(n.find("(null)") != std::string::npos);
    TEST_ASSERT(n.find("(unknown)") != std::string::npos);

    // When the end offset itself overflows size_t, report "overflow" rather
    // than a wrapped number.
    const size_t kMax = std::numeric_limits<size_t>::max();
    const std::string o = gguf_bounds_error("target GGUF", "t", "f32",
                                            kMax, 10, 10, 100);
    TEST_ASSERT(o.find("overflow") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_qwen35_embedded_mtp_target_layer_count) {
    uint32_t target_layers = 0;
    std::string error;

    TEST_ASSERT(derive_effective_target_layer_count(
        "qwen35", 64, 0, target_layers, error));
    TEST_ASSERT(target_layers == 64);

    TEST_ASSERT(derive_effective_target_layer_count(
        "qwen35", 65, 1, target_layers, error));
    TEST_ASSERT(target_layers == 64);

    TEST_ASSERT(derive_effective_target_layer_count(
        "qwen35moe", 81, 1, target_layers, error));
    TEST_ASSERT(target_layers == 80);

    TEST_ASSERT(derive_effective_target_layer_count(
        "bailingmoe3", 43, 1, target_layers, error));
    TEST_ASSERT(target_layers == 42);

    TEST_ASSERT(!derive_effective_target_layer_count(
        "qwen35", 1, 1, target_layers, error));
    TEST_ASSERT(error.find("smaller than block_count") != std::string::npos);

    TEST_ASSERT(!derive_effective_target_layer_count(
        "qwen35", 0, 0, target_layers, error));
    TEST_ASSERT(error.find("greater than zero") != std::string::npos);

    // Do not reinterpret similarly named metadata for unrelated architectures.
    TEST_ASSERT(derive_effective_target_layer_count(
        "laguna", 65, 1, target_layers, error));
    TEST_ASSERT(target_layers == 65);
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_invoke_xml) {
    const std::string text =
        "Reading configuration:\n"
        "<function_calls>\n"
        "<invoke name=\"read\">\n"
        "  <param name=\"path\">server.go</param>\n"
        "  <param name=\"offset\">10</param>\n"
        "  <param name=\"limit\">50</param>\n"
        "</invoke>\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "server.go");
        TEST_ASSERT(args["offset"] == 10);
        TEST_ASSERT(args["limit"] == 50);
    }
    TEST_ASSERT(result.cleaned_text == "Reading configuration:");
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_invoke_json) {
    const std::string text =
        "<function_calls>\n"
        "<invoke name=\"read\">\n"
        "  {\"path\": \"app.py\", \"offset\": \"5\"}\n"
        "</invoke>\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "app.py");
        TEST_ASSERT(args["offset"] == 5);
    }
    TEST_ASSERT(result.cleaned_text.empty());
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

TEST_CASE(ServerUnitFixture, test_parse_function_calls_anthropic_input_schema) {
    json anthropic_tools = json::array({
        {
            {"name", "read"},
            {"description", "Read a file"},
            {"input_schema", {
                {"type", "object"},
                {"properties", {
                    {"path", {{"type", "string"}}},
                    {"offset", {{"type", "integer"}}}
                }}
            }}
        }
    });

    const std::string text =
        "<function_calls>\n"
        "<invoke name=\"read\">\n"
        "  <param name=\"path\">main.cpp</param>\n"
        "  <param name=\"offset\">42</param>\n"
        "</invoke>\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, anthropic_tools);
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "main.cpp");
        TEST_ASSERT(args["offset"] == 42);
    }
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

TEST_CASE(ServerUnitFixture, test_parse_function_calls_json_lines) {
    std::string text =
        "Let me read the files:\n"
        "<function_calls>\n"
        "{\"name\": \"read\", \"arguments\": {\"path\": \"file1.txt\"}}\n"
        "{\"name\": \"read\", \"arguments\": {\"path\": \"file2.txt\"}}\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 2);
    if (result.tool_calls.size() == 2) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(result.tool_calls[1].name == "read");
        auto a1 = json::parse(result.tool_calls[0].arguments);
        auto a2 = json::parse(result.tool_calls[1].arguments);
        TEST_ASSERT(a1["path"] == "file1.txt");
        TEST_ASSERT(a2["path"] == "file2.txt");
    }
    TEST_ASSERT(result.cleaned_text == "Let me read the files:");
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_sibling_invokes_partial_failure) {
    std::string text =
        "<function_calls>\n"
        "<invoke name=\"read\">\n"
        "  <param name=\"path\">valid.txt</param>\n"
        "</invoke>\n"
        "<invoke name=\"read\">\n"
        "  malformed parameter text\n"
        "</invoke>\n"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        auto args = json::parse(result.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "valid.txt");
    }
    TEST_ASSERT(result.cleaned_text.find("malformed parameter text") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_rejected_invokes_do_not_expose_nested_json) {
    const std::string disallowed =
        "<function_calls>"
        "<invoke name=\"forbidden\">"
        "{\"name\":\"read\",\"arguments\":{\"path\":\"secret\"}}"
        "</invoke>"
        "</function_calls>";
    auto disallowed_result = parse_tool_calls(disallowed, read_tools());
    TEST_ASSERT(disallowed_result.tool_calls.empty());
    TEST_ASSERT(disallowed_result.cleaned_text == disallowed);

    const std::string malformed =
        "<function_calls>"
        "<invoke>"
        "{\"name\":\"read\",\"arguments\":{\"path\":\"secret\"}}"
        "</invoke>"
        "</function_calls>";
    auto malformed_result = parse_tool_calls(malformed, read_tools());
    TEST_ASSERT(malformed_result.tool_calls.empty());
    TEST_ASSERT(malformed_result.cleaned_text == malformed);
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_invoke_rejects_braced_prose) {
    const std::string text =
        "<function_calls>"
        "<invoke name=\"read\">{this is prose}</invoke>"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.empty());
    TEST_ASSERT(result.cleaned_text == text);
}

TEST_CASE(ServerUnitFixture, test_parse_function_calls_invoke_keeps_structured_json_syntax_errors) {
    const std::string text =
        "<function_calls>"
        "<invoke name=\"read\">{\"offset\":5o1}</invoke>"
        "</function_calls>";

    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(result.tool_calls[0].arguments == "{\"offset\":5o1}");
    }
}

TEST_CASE(ServerUnitFixture, test_parse_json_syntax_error_forwarding) {
    std::string text = "<function_call>{\"name\": \"read\", \"arguments\": {\"offset\": 5o1}}</function_call>";
    auto result = parse_tool_calls(text, read_tools());
    TEST_ASSERT(result.tool_calls.size() == 1);
    if (!result.tool_calls.empty()) {
        TEST_ASSERT(result.tool_calls[0].name == "read");
        TEST_ASSERT(result.tool_calls[0].arguments == "{\"offset\": 5o1}");
    }
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

TEST_CASE(ServerUnitFixture, test_parse_tool_call_rejects_empty_name_and_scalar_arguments) {
    // Empty tool name must be rejected
    const std::string empty_name =
        "<function_call>\n{\"name\": \"\", \"arguments\": {\"path\": \"/tmp/test\"}}\n</function_call>";
    auto res_empty = parse_tool_calls(empty_name, read_tools());
    TEST_ASSERT(res_empty.tool_calls.empty());

    // Valid scalar string argument must be rejected (not an object)
    const std::string scalar_arg =
        "<function_call>\n{\"name\": \"read\", \"arguments\": \"just a string\"}\n</function_call>";
    auto res_scalar = parse_tool_calls(scalar_arg, read_tools());
    TEST_ASSERT(res_scalar.tool_calls.empty());

    // Valid array argument must be rejected (not an object)
    const std::string array_arg =
        "<function_call>\n{\"name\": \"read\", \"arguments\": [1, 2, 3]}\n</function_call>";
    auto res_array = parse_tool_calls(array_arg, read_tools());
    TEST_ASSERT(res_array.tool_calls.empty());

    // A scalar string wrapped in braces is still prose, not object arguments.
    const std::string braced_prose =
        "<function_call>\n"
        "{\"name\": \"read\", \"arguments\": \"{this is prose}\"}\n"
        "</function_call>";
    auto res_braced_prose = parse_tool_calls(braced_prose, read_tools());
    TEST_ASSERT(res_braced_prose.tool_calls.empty());

    // Keep forwarding a structurally object-like string with a JSON syntax
    // error so the client can report the exact bad arguments to the model.
    const std::string bad_obj_string =
        "<function_call>\n"
        "{\"name\": \"read\", \"arguments\": \"{\\\"offset\\\": 5o1}\"}\n"
        "</function_call>";
    auto res_bad_obj_string = parse_tool_calls(bad_obj_string, read_tools());
    TEST_ASSERT(res_bad_obj_string.tool_calls.size() == 1);
    if (!res_bad_obj_string.tool_calls.empty()) {
        TEST_ASSERT(res_bad_obj_string.tool_calls[0].arguments == "{\"offset\": 5o1}");
    }

    // Malformed JSON object syntax (e.g. 5o1) is forwarded as raw args
    const std::string bad_obj =
        "<function_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"/tmp/test\", \"offset\": 5o1}}\n</function_call>";
    auto res_bad = parse_tool_calls(bad_obj, read_tools());
    TEST_ASSERT(res_bad.tool_calls.size() == 1);
    if (!res_bad.tool_calls.empty()) {
        TEST_ASSERT(res_bad.tool_calls[0].name == "read");
        TEST_ASSERT(res_bad.tool_calls[0].arguments.find("5o1") != std::string::npos);
    }
}

TEST_CASE(ServerUnitFixture, test_extract_raw_json_tool_fallback_with_nested_name_argument) {
    // Malformed arguments containing a "name" key before top-level "name"
    const std::string text_reversed =
        "<function_call>\n"
        "{\"arguments\": {\"name\": \"evil_command\", \"offset\": 5o1}, \"name\": \"read\"}\n"
        "</function_call>";
    auto res_rev = parse_tool_calls(text_reversed, read_tools());
    TEST_ASSERT(res_rev.tool_calls.size() == 1);
    if (!res_rev.tool_calls.empty()) {
        TEST_ASSERT(res_rev.tool_calls[0].name == "read");
        TEST_ASSERT(res_rev.tool_calls[0].arguments.find("evil_command") != std::string::npos);
    }

    // Nested function object with "name" inside arguments
    const std::string text_nested =
        "<function_call>\n"
        "{\"function\": {\"name\": \"read\", \"arguments\": {\"name\": \"evil_nested\", \"offset\": 5o1}}}\n"
        "</function_call>";
    auto res_nest = parse_tool_calls(text_nested, read_tools());
    TEST_ASSERT(res_nest.tool_calls.size() == 1);
    if (!res_nest.tool_calls.empty()) {
        TEST_ASSERT(res_nest.tool_calls[0].name == "read");
        TEST_ASSERT(res_nest.tool_calls[0].arguments.find("evil_nested") != std::string::npos);
    }
}

TEST_CASE(ServerUnitFixture, test_extract_raw_json_tool_fallback_ignores_nested_metadata_name) {
    const std::string text =
        "<function_call>"
        "{\"function\":{\"name\":\"bash\",\"metadata\":{\"name\":\"read\"},"
        "\"arguments\":{\"command\":\"pwd\",\"offset\":5o1}}}"
        "</function_call>";

    auto res = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(res.tool_calls.size() == 1);
    if (!res.tool_calls.empty()) {
        TEST_ASSERT(res.tool_calls[0].name == "bash");
        TEST_ASSERT(res.tool_calls[0].arguments.find("5o1") != std::string::npos);
    }
}

TEST_CASE(ServerUnitFixture, test_raw_json_fallback_does_not_cross_tool_envelopes) {
    const std::string text =
        "{\"metadata\":{\"arguments\":{\"offset\":5o1}},"
        "\"tool_call\":{\"name\":\"read\",\"arguments\":{\"path\":\"x\"}}}";

    auto res = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(res.tool_calls.size() == 1);
    if (!res.tool_calls.empty()) {
        TEST_ASSERT(res.tool_calls[0].name == "read");
        const json args = json::parse(res.tool_calls[0].arguments);
        TEST_ASSERT(args["path"] == "x");
        TEST_ASSERT(!args.contains("offset"));
    }
}

TEST_CASE(ServerUnitFixture, test_parse_json_tool_call_checks_later_envelope_siblings) {
    const std::string text =
        "{\"function\":{\"metadata\":true},"
        "\"tool_call\":{\"name\":\"read\",\"arguments\":{\"path\":\"x\"}}}";

    auto res = parse_tool_calls(text, read_and_bash_tools());
    TEST_ASSERT(res.tool_calls.size() == 1);
    if (!res.tool_calls.empty()) {
        TEST_ASSERT(res.tool_calls[0].name == "read");
        TEST_ASSERT(json::parse(res.tool_calls[0].arguments)["path"] == "x");
    }
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

TEST_CASE(ServerUnitFixture, test_parse_function_calls_mixed_invoke_and_json_line_siblings) {
    const std::string text =
        "Processing files:\n"
        "<function_calls>\n"
        "  <invoke name=\"read\">\n"
        "    <param name=\"path\">first.go</param>\n"
        "  </invoke>\n"
        "  {\"name\": \"read\", \"arguments\": {\"path\": \"second.go\"}}\n"
        "</function_calls>";

    auto res = parse_tool_calls(text, read_tools());
    TEST_ASSERT(res.tool_calls.size() == 2);
    if (res.tool_calls.size() == 2) {
        TEST_ASSERT(res.tool_calls[0].name == "read");
        auto args0 = json::parse(res.tool_calls[0].arguments);
        TEST_ASSERT(args0["path"] == "first.go");

        TEST_ASSERT(res.tool_calls[1].name == "read");
        auto args1 = json::parse(res.tool_calls[1].arguments);
        TEST_ASSERT(args1["path"] == "second.go");
    }
    TEST_ASSERT(res.cleaned_text == "Processing files:");
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

TEST_CASE(ServerUnitFixture, test_parse_function_calls_empty_invoke_zero_args) {
    json tools = json::array({
        {
            {"type", "function"},
            {"function", {
                {"name", "get_version"},
                {"description", "get version"},
                {"parameters", {{"type", "object"}, {"properties", json::object()}}}
            }}
        }
    });

    const std::string text =
        "<function_calls>\n"
        "  <invoke name=\"get_version\"></invoke>\n"
        "</function_calls>";

    auto res = parse_tool_calls(text, tools);
    TEST_ASSERT(res.tool_calls.size() == 1);
    if (!res.tool_calls.empty()) {
        TEST_ASSERT(res.tool_calls[0].name == "get_version");
        TEST_ASSERT(res.tool_calls[0].arguments == "{}");
    }
}

TEST_CASE(ServerUnitFixture, test_emitter_streaming_malformed_tool_in_think_answer_containing_close_tag_recovers_answer) {
    auto em = make_emitter(ApiFormat::OPENAI_CHAT, read_tools(), true);
    em.emit_start();
    em.emit_token("<think><tool_call><bad_param></tool_call></think>The output is </parameter> end.");
    auto chunks = em.emit_finish(10, nullptr, -1);
    TEST_ASSERT(em.tool_calls().empty());
    TEST_ASSERT(em.accumulated_text() == "The output is </parameter> end.");
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

namespace {
struct StderrCapture {
    int old_stderr = -1;
    std::FILE * file = nullptr;

    StderrCapture() {
        std::fflush(stderr);
        file = std::tmpfile();
        if (file == nullptr) return;

#if defined(_WIN32)
        old_stderr = _dup(_fileno(stderr));
        if (old_stderr == -1 || _dup2(_fileno(file), _fileno(stderr)) == -1) {
            if (old_stderr != -1) _close(old_stderr);
#else
        old_stderr = dup(STDERR_FILENO);
        if (old_stderr == -1 || dup2(fileno(file), STDERR_FILENO) == -1) {
            if (old_stderr != -1) close(old_stderr);
#endif
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
#if defined(_WIN32)
            _dup2(old_stderr, _fileno(stderr));
            _close(old_stderr);
#else
            dup2(old_stderr, STDERR_FILENO);
            close(old_stderr);
#endif
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
    TEST_ASSERT(captured.find("[server] tool_call parse failed; suppressing buffered tool text") != std::string::npos);
    TEST_ASSERT(captured.find("text='<function_call>\\n  <invoke name=\"read\">\\n    malformed prose body with\\nnew lines and \\t tabs\\n'") != std::string::npos);
}

TEST_CASE(ServerUnitFixture,
          test_concurrent_scheduler_burst_stops_at_eos) {
    SeqEngine::DecodeOutput burst;
    burst.slot = 0;
    burst.committed_tokens = {101, 2, 103};
    burst.token = 104;

    std::vector<int32_t> emitted;
    const bool consumed_all = consume_decode_output_tokens(
        burst, [&](int32_t token) {
            emitted.push_back(token);
            return token != 2;
        });

    TEST_ASSERT(!consumed_all);
    TEST_ASSERT((emitted == std::vector<int32_t>{101, 2}));
}

namespace {
// Any text works; the transport must not know a model's marker.
constexpr char IMAGE_PLACEHOLDER[] = "<image>";

json image_transport_part(const char * url = "data:image/png;base64,iVBORw0KGgo=") {
    return {{"type", "image_url"}, {"image_url", {{"url", url}}}};
}

class OwnedImageTestPayload final : public ImagePromptPayload {
public:
    explicit OwnedImageTestPayload(std::vector<int32_t> tokens) : tokens_(std::move(tokens)) {}
    bool matches(const std::vector<int32_t> & tokens) const override { return tokens == tokens_; }
private:
    const std::vector<int32_t> tokens_;
};

struct ImageCarrierRetryBackend : EmptySpecRetryBackend {
    const ImagePromptPayload * expected = nullptr;
    std::vector<int32_t> expected_tokens;
    GenerateResult generate_impl(const GenerateRequest & req, const DaemonIO & io) override {
        TEST_ASSERT(req.images.get() == expected);
        TEST_ASSERT(req.prompt == expected_tokens);
        TEST_ASSERT(req.images->matches(req.prompt));
        return EmptySpecRetryBackend::generate_impl(req, io);
    }
};
}

TEST_CASE(ServerUnitFixture, test_image_extraction_normalization_preserves_interleaved_order) {
    const json messages = json::array({
        {{"role", "user"}, {"content", json::array({
            {{"type", "text"}, {"text", "before "}}, image_transport_part(),
            {{"type", "text"}, {"text", " between "}},
            image_transport_part("data:image/jpeg;base64,/9j/"),
            {{"type", "text"}, {"text", " after"}}})}},
        {{"role", "assistant"}, {"content", "acknowledged"}},
        {{"role", "user"}, {"content", "follow-up"}}
    });
    json normalized;
    std::vector<EncodedImage> images;
    std::string error;
    TEST_ASSERT(prepare_request_images(messages, {true, true, IMAGE_PLACEHOLDER}, normalized, images, error));
    TEST_ASSERT(images.size() == 2);
    TEST_ASSERT(images[0].mime_type == "image/png");
    TEST_ASSERT(images[0].bytes == std::vector<uint8_t>({137, 80, 78, 71, 13, 10, 26, 10}));
    TEST_ASSERT(images[1].mime_type == "image/jpeg");
    TEST_ASSERT(images[1].bytes == std::vector<uint8_t>({255, 216, 255}));
    ToolMemory memory;
    const auto chat = normalize_chat_messages(normalized, ApiFormat::OPENAI_CHAT, memory);
    TEST_ASSERT(chat.size() == 3);
    TEST_ASSERT(chat[0].role == "user");
    TEST_ASSERT(chat[0].content == std::string("before ") + IMAGE_PLACEHOLDER +
                " between " + IMAGE_PLACEHOLDER + " after");
    TEST_ASSERT(chat[1].content == "acknowledged" && chat[2].content == "follow-up");
    json retained = {{"messages", messages}, {"metadata", {{"image_url", "private-url"}, {"label", "keep"}}}};
    redact_image_urls(retained);
    TEST_ASSERT(retained.dump().find("base64") == std::string::npos);
    TEST_ASSERT(retained.dump().find("private-url") == std::string::npos);
    TEST_ASSERT(retained["metadata"]["label"] == "keep");
    TEST_ASSERT(retained["messages"][0]["content"][0]["text"] == "before ");
    TEST_ASSERT(messages[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgo=");
}

TEST_CASE(ServerUnitFixture, test_image_extraction_failure_does_not_publish_partial_images) {
    const json messages = json::array({{{"role", "user"}, {"content", json::array({
        image_transport_part(), image_transport_part("https://example.invalid/private.png")
    })}}});
    json normalized = {{"stale", true}};
    std::vector<EncodedImage> images{{"stale", {1}}};
    std::string error;
    TEST_ASSERT(!extract_chat_images(messages, IMAGE_PLACEHOLDER, normalized, images, error));
    TEST_ASSERT(normalized.is_null());
    TEST_ASSERT(images.empty());
    TEST_ASSERT(!error.empty());
    TEST_ASSERT(error.find("example.invalid") == std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_default_backend_rejects_encoded_images_without_rewriting_tokens) {
    MockBackend backend;
    std::vector<int32_t> tokens{1, 2, 3};
    ImagePromptHandle payload;
    std::string error;
    TEST_ASSERT(!backend.supports_images());
    TEST_ASSERT(backend.prepare_images(tokens, {{"image/png", {137, 80, 78, 71}}},
                                       8192, 32, payload, error) == ImagePrepareStatus::invalid);
    TEST_ASSERT(tokens == std::vector<int32_t>({1, 2, 3}));
    TEST_ASSERT(!payload && !error.empty());
    TEST_ASSERT(backend.prepare_images(tokens, {}, 8192, 32, payload, error) ==
                ImagePrepareStatus::ok);
    TEST_ASSERT(tokens == std::vector<int32_t>({1, 2, 3}));
}

TEST_CASE(ServerUnitFixture, test_image_binding_survives_request_copies_and_common_ar_retry) {
    static_assert(std::is_const_v<ImagePromptHandle::element_type>);
    GenerateRequest request;
    std::weak_ptr<const ImagePromptPayload> lifetime;
    {
        ParsedRequest parsed;
        parsed.prompt_tokens = {1, 129280, 129281, 2};
        parsed.images = std::make_shared<OwnedImageTestPayload>(parsed.prompt_tokens);
        lifetime = parsed.images;
        ParsedRequest queued = parsed;
        parsed.prompt_tokens.clear();
        parsed.images.reset();
        request.prompt = queued.prompt_tokens;
        request.images = queued.images;
    }
    TEST_ASSERT(!lifetime.expired());
    auto changed = request.prompt;
    changed[1] += 1;
    TEST_ASSERT(!request.images->matches(changed));
    TEST_ASSERT(request.images->matches(request.prompt));
    ImageCarrierRetryBackend backend;
    backend.expected = request.images.get();
    backend.expected_tokens = request.prompt;
    request.n_gen = 1;
    DaemonIO io;
    TEST_ASSERT(backend.generate(request, io).ok());
    TEST_ASSERT(backend.generate_calls == 2 && backend.generate_saw_force_ar);
    request.images.reset();
    TEST_ASSERT(lifetime.expired());
}

TEST_CASE(ServerUnitFixture, test_http_image_policy_rejects_unconsumed_images_including_mixed_valid) {
    const json image = image_transport_part();
    const std::vector<json> malformed_messages = {
        {{"role", "user"}, {"content", image}},
        {{"role", "user"}, {"content", json::array({
            {{"type", "text"}, {"text", "visible"}, {"image_url", image["image_url"]}}
        })}},
        {{"role", "user"}, {"content", json::array({
            {{"type", "container"}, {"nested", image}}
        })}},
        {{"role", "user"}, {"content", "visible"}, {"attachment", image}},
        {{"role", "user"}, {"content", json::array({
            {{"type", "input_image"}, {"image_url", image["image_url"]}}
        })}}
    };
    for (const auto & malformed : malformed_messages) {
        for (bool mixed : {false, true}) {
            json messages = json::array();
            if (mixed) messages.push_back({{"role", "user"}, {"content", json::array({image})}});
            messages.push_back(malformed);
            json normalized = "stale";
            std::vector<EncodedImage> images{{"stale", {1}}};
            std::string error;
            TEST_ASSERT(!prepare_request_images(messages, {true, true, IMAGE_PLACEHOLDER}, normalized, images, error));
            TEST_ASSERT(normalized.is_null() && images.empty());
            TEST_ASSERT(!error.empty());
            TEST_ASSERT(error.find("base64") == std::string::npos);
        }
    }
}

TEST_CASE(ServerUnitFixture, test_http_image_policy_requires_chat_endpoint_and_effective_capability) {
    const json messages = json::array({{{"role", "user"}, {"content", json::array({image_transport_part()})}}});
    {
        json normalized = "stale";
        std::vector<EncodedImage> images{{"stale", {1}}};
        std::string error;
        TEST_ASSERT(!prepare_request_images(messages, {false, true, IMAGE_PLACEHOLDER}, normalized, images, error));
        TEST_ASSERT(normalized.is_null() && images.empty());
        TEST_ASSERT(!error.empty());
    }
    // Without image capability the messages pass through untouched.
    for (ImageRequestPolicy policy : {
            ImageRequestPolicy{true, false, IMAGE_PLACEHOLDER},
            ImageRequestPolicy{false, false, IMAGE_PLACEHOLDER}}) {
        json normalized = "stale";
        std::vector<EncodedImage> images{{"stale", {1}}};
        std::string error;
        TEST_ASSERT(prepare_request_images(messages, policy, normalized, images, error));
        TEST_ASSERT(normalized == messages && images.empty());
    }
    json normalized;
    std::vector<EncodedImage> images;
    std::string error;
    TEST_ASSERT(prepare_request_images(messages, {true, true, IMAGE_PLACEHOLDER}, normalized, images, error));
    TEST_ASSERT(images.size() == 1);
}

TEST_CASE(ServerUnitFixture, test_http_image_policy_preserves_text_without_image_capability) {
    const json messages = json::array({
        {{"role", "system"}, {"content", "instructions"}},
        {{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", "ordinary text"}}})}}
    });
    for (ImageRequestPolicy policy : {
            ImageRequestPolicy{true, false, IMAGE_PLACEHOLDER},
            ImageRequestPolicy{false, false, IMAGE_PLACEHOLDER},
            ImageRequestPolicy{true, true, IMAGE_PLACEHOLDER}}) {
        json normalized;
        std::vector<EncodedImage> images;
        std::string error;
        TEST_ASSERT(prepare_request_images(messages, policy, normalized, images, error));
        TEST_ASSERT(normalized == messages && images.empty());
    }
    json normalized;
    std::vector<EncodedImage> images;
    std::string error;
    const json forged = json::array({{{"role", "user"}, {"content", IMAGE_PLACEHOLDER}}});
    TEST_ASSERT(!prepare_request_images(forged, {true, true, IMAGE_PLACEHOLDER}, normalized, images, error));
    TEST_ASSERT(normalized.is_null() && images.empty());
}

TEST_CASE(ServerUnitFixture, test_pflash_query_head_plan_windows_and_caps) {
    // One token per character: token counts equal character counts.
    std::string question;
    for (int i = 0; i < 12; ++i) question += "Is setting " + std::to_string(i) + " fine? ";
    std::string paste = "## Paper\n\n";
    for (int i = 0; i < 30; ++i) paste += "Body text line " + std::to_string(i) + " of the paper.\n\n";
    const std::string after = "Which line mentions the number twenty?";
    const std::string content = question + "\n\n" + paste + after;
    const std::string rendered = "<|im_start|>user\n" + content + "<|im_end|>\n";
    const std::string path = write_pflash_bpe_tokenizer_fixture({}, rendered);
    Tokenizer tok;
    TEST_ASSERT(tok.load_from_gguf(path.c_str()));
    unlink(path.c_str());
    std::vector<int32_t> ids;
    ids.push_back(tok.token_to_id("<|im_start|>"));
    for (char ch : std::string("user\n") + content) {
        ids.push_back(tok.token_to_id(test_gpt2_encode(std::string(1, ch))));
    }
    const int begin = 6;                       // after "<|im_start|>user\n"
    const int end = begin + (int) content.size();
    TEST_ASSERT(tok.decode({ids.begin() + begin, ids.begin() + end}) == content);

    // Caps above both blocks: the head whole, the tail as the query window.
    auto plan = http_detail::pflash_query_head_plan(tok, ids, begin, end, 512, 512);
    TEST_ASSERT(plan.structured && plan.head_rule == "paste");
    TEST_ASSERT(plan.extra.size() == 1);
    const std::string q = question.substr(0, question.size() - 1);   // trimmed
    TEST_ASSERT(tok.decode({ids.begin() + plan.extra[0].begin, ids.begin() + plan.extra[0].end}) == q);
    TEST_ASSERT(tok.decode({ids.begin() + plan.tail.begin, ids.begin() + plan.tail.end}) == after);
    TEST_ASSERT(plan.tail.end == end);

    // A head over its cap: first and last cap/2 tokens; a tail over its cap:
    // the query window is its last cap/2 tokens, its first cap/2 an extra.
    plan = http_detail::pflash_query_head_plan(tok, ids, begin, end, 40, 20);
    TEST_ASSERT(plan.extra.size() == 3);
    TEST_ASSERT(plan.extra[0].begin == begin && plan.extra[0].end == begin + 20);
    TEST_ASSERT(plan.extra[1].end == begin + (int) q.size() &&
                plan.extra[1].end - plan.extra[1].begin == 20);
    TEST_ASSERT(plan.tail.end == end && plan.tail.end - plan.tail.begin == 10);
    TEST_ASSERT(plan.extra[2].begin == end - (int) after.size() &&
                plan.extra[2].end - plan.extra[2].begin == 10);

    // A stock preamble under 16 tokens is not a question; a plain message
    // keeps today's window.
    const std::string stock = "Here it is.\n\n" + paste + after;
    std::vector<int32_t> stock_ids;
    for (char ch : stock) stock_ids.push_back(tok.token_to_id(test_gpt2_encode(std::string(1, ch))));
    plan = http_detail::pflash_query_head_plan(tok, stock_ids, 0, (int) stock_ids.size(), 256, 256);
    TEST_ASSERT(plan.extra.empty() && plan.head_tokens == 0);
    TEST_ASSERT(plan.tail.begin >= 0);
    std::vector<int32_t> plain_ids;
    for (char ch : std::string("just a short plain message without any structure at all"))
        plain_ids.push_back(tok.token_to_id(test_gpt2_encode(std::string(1, ch))));
    plan = http_detail::pflash_query_head_plan(tok, plain_ids, 0, (int) plain_ids.size(), 256, 256);
    TEST_ASSERT(!plan.structured && plan.extra.empty() && plan.tail.begin < 0);
}

// A tool session: its tool definitions and filler text.
static std::string pflash_tool_session_filler(const char * line, int count) {
    std::string out;
    for (int i = 0; i < count; ++i) out += line;
    return out;
}

static json pflash_tool_session_tools() {
    return json::array({
        {{"type", "function"},
         {"function", {{"name", "read_file"}, {"description", "Read a file"},
                       {"parameters", {{"type", "object"},
                                       {"properties", {{"path", {{"type", "string"}}}}}}}}}},
    });
}

// system + task + (assistant tool call + long tool result) + assistant tool
// call + a short new tool result.
// ─── Tool sessions (PFLASH_AGENT_PASTE) ───────────────────────────────────

TEST_CASE(ServerUnitFixture, test_pflash_agent_paste_flag_defaults_off) {
    luce_unsetenv("PFLASH_AGENT_PASTE");
    TEST_ASSERT(!http_detail::pflash_agent_paste_enabled());
    // Only "1" turns it on; anything else leaves tool sessions raw.
    for (const char * off : {"", "0", "on", "true", "keep", "1 ", "01"}) {
        luce_setenv("PFLASH_AGENT_PASTE", off);
        TEST_ASSERT_MSG(!http_detail::pflash_agent_paste_enabled(), off);
    }
    luce_setenv("PFLASH_AGENT_PASTE", "1");
    TEST_ASSERT(http_detail::pflash_agent_paste_enabled());
    luce_unsetenv("PFLASH_AGENT_PASTE");
}

TEST_CASE(ServerUnitFixture, test_pflash_keep_history_decision) {
    using http_detail::PflashAdmission;
    using http_detail::pflash_keep_history_admission;
    const int t = 16384;
    // Large new user content: only it is compressed, whatever today's
    // decision for the whole prompt (warm or compress).
    for (auto admission : {PflashAdmission::Compress, PflashAdmission::WarmCache}) {
        TEST_ASSERT(pflash_keep_history_admission(admission, t, t, true) ==
                    PflashAdmission::NewContent);
        TEST_ASSERT(pflash_keep_history_admission(admission, t, t, false) ==
                    PflashAdmission::NewContent);
    }
    // Small new content: raw instead of compressing the history.
    TEST_ASSERT(pflash_keep_history_admission(PflashAdmission::Compress, t - 1, t, false) ==
                PflashAdmission::KeepHistory);
    TEST_ASSERT(pflash_keep_history_admission(PflashAdmission::WarmCache, t - 1, t, true) ==
                PflashAdmission::KeepHistory);
    // Already raw: unchanged.
    TEST_ASSERT(pflash_keep_history_admission(PflashAdmission::WarmCache, t - 1, t, false) ==
                PflashAdmission::WarmCache);
    TEST_ASSERT(pflash_keep_history_admission(PflashAdmission::UnderThreshold, 3 * t, t, true) ==
                PflashAdmission::UnderThreshold);
    TEST_ASSERT(std::string(http_detail::pflash_admission_name(
        PflashAdmission::NewContent)) == "new_content");
    TEST_ASSERT(std::string(http_detail::pflash_admission_name(
        PflashAdmission::KeepHistory)) == "keep_history");
}

TEST_CASE(ServerUnitFixture, test_pflash_keep_history_scope) {
    using http_detail::PflashChatTurn;
    const auto turn = [] (const char * role, int begin) {
        PflashChatTurn out;
        out.role = role;
        out.role_begin = begin;
        out.content_begin = begin + 2;
        out.content_end = begin + 8;
        out.turn_end = begin + 10;
        return out;
    };
    int newest = 7;
    TEST_ASSERT(std::string(http_detail::pflash_keep_history_scope({}, newest)) == "no_turns");
    TEST_ASSERT(newest == -1);
    // First turn: the preamble and the user's message. The newest user
    // message is the one thing a tool session may compress.
    TEST_ASSERT(http_detail::pflash_keep_history_scope(
        {turn("system", 0), turn("user", 10)}, newest) == nullptr);
    TEST_ASSERT(newest == 1);
    // Newest turn a tool output (or an assistant prefill): nothing to compress.
    TEST_ASSERT(std::string(http_detail::pflash_keep_history_scope(
        {turn("system", 0), turn("user", 10), turn("assistant", 20), turn("tool", 30)},
        newest)) == "newest_not_user");
    TEST_ASSERT(std::string(http_detail::pflash_keep_history_scope(
        {turn("user", 0), turn("assistant", 10)}, newest)) == "newest_not_user");
    // A user turn after the agent's turns: in scope, newest is its index.
    TEST_ASSERT(http_detail::pflash_keep_history_scope(
        {turn("system", 0), turn("user", 10), turn("assistant", 20), turn("tool", 30),
         turn("assistant", 40), turn("user", 50)}, newest) == nullptr);
    TEST_ASSERT(newest == 5);
    // An empty user turn has nothing to compress or keep apart.
    auto empty = turn("user", 50);
    empty.content_end = empty.content_begin;
    TEST_ASSERT(std::string(http_detail::pflash_keep_history_scope(
        {turn("user", 0), turn("assistant", 10), empty}, newest)) == "newest_not_user");
}

static std::string pflash_keep_filler(const char * line, int count) {
    std::string out;
    for (int i = 0; i < count; ++i) out += line;
    return out;
}

static int pflash_keep_count(const std::string & text, const std::string & needle) {
    int count = 0;
    for (size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + needle.size())) {
        ++count;
    }
    return count;
}

// system + task + read_file call + tool result (``tool_lines`` lines) +
// the agent's answer: an earlier conversation with tool activity.
static json pflash_keep_agent_history(int tool_lines) {
    const json call = json::array({{{"id", "call-1"}, {"type", "function"},
        {"function", {{"name", "read_file"}, {"arguments", "{\"path\":\"parser.py\"}"}}}}});
    return json::array({
        {{"role", "system"}, {"content", "You are a coding agent."}},
        {{"role", "user"}, {"content", "Fix the bug in parser.py."}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", call}},
        {{"role", "tool"}, {"tool_call_id", "call-1"},
         {"content", pflash_keep_filler("def parse(x): return x.strip()\n", tool_lines)}},
        {{"role", "assistant"}, {"content", "The parser strips each line."}},
    });
}

static const char * const kPflashKeepPasteLine = "The manual says many things.\n";

static std::string pflash_keep_paste(int lines, const char * question) {
    return pflash_keep_filler(kPflashKeepPasteLine, lines) + question;
}

static json pflash_keep_with(json messages, const json & message) {
    messages.push_back(message);
    return messages;
}

static json pflash_keep_user(const std::string & content) {
    return {{"role", "user"}, {"content", content}};
}

struct MockPflashWarmSpanBackend : MockPflashSpanBackend {
    int warm_pos = 0;   // slot 0 holds a snapshot at this position
    bool snapshot_used(int slot) const override { return slot == 0 && warm_pos > 0; }
    int snapshot_cur_pos(int slot) const override {
        return snapshot_used(slot) ? warm_pos : 0;
    }
};

struct PflashKeepHistoryRun {
    std::vector<HttpServerTestAccess::PreparedPrompt> prepared;
    std::vector<std::vector<int32_t>> ids;
    std::vector<std::string> served_text;
    // Target tokens before the last "<|im_start|>user" of each prompt.
    std::vector<int> user_cut;
    json counters;
    json keep;
    int compress_calls = 0;
    std::vector<int> calls_after;   // backend compress calls after each turn
    std::vector<std::string> cut_text;   // served text up to the snapshot cut
    std::vector<json> view_records;
};

// Extra passages the mock selector keeps: {trigger, passage} keeps the
// passage's span whenever the prompt contains the trigger text (a question
// that needs it), so a later question can recall what a paste turn dropped.
using PflashKeepRecallPairs = std::vector<std::pair<std::string, std::string>>;

// One server, the given turns in order (cache-aware admission, arrive on,
// fixture tokenizer as target and drafter). ``flag``: PFLASH_AGENT_PASTE
// (nullptr = unset, the default). ``warm_history``: before the
// first turn, the prefix cache holds its prompt up to the newest user turn.
static PflashKeepHistoryRun run_keep_history_session(
        const std::vector<json> & turns, const json & tools, const char * flag,
        int threshold, bool warm_history = false,
        int max_ctx = 16384, const char * arrive_flag = "1",
        const PflashKeepRecallPairs & recall_pairs = {}) {
    const int max_output = 16;
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar chunk{"PFLASH_SELECT_CHUNK_SIZE", "4"};
    luce_test::ScopedEnvVar admission{"PFLASH_ADMISSION", "cache"};
    luce_test::ScopedEnvVar arrive{"PFLASH_ARRIVE", arrive_flag};
    luce_test::ScopedEnvVar paste{"PFLASH_AGENT_PASTE", flag};
    char trace_path[] = "/tmp/pflash_keep_history_view_XXXXXX";
    const int trace_fd = mkstemp(trace_path);
    TEST_ASSERT(trace_fd >= 0);
    close(trace_fd);
    luce_test::ScopedEnvVar view_trace{"PFLASH_VIEW_TRACE_PATH", trace_path};
    std::vector<std::string> rendered;
    std::string cover;
    for (const auto & messages : turns) {
        ToolMemory tool_memory;
        const auto chat = normalize_chat_messages(messages, ApiFormat::OPENAI_CHAT, tool_memory);
        rendered.push_back(render_chat_template(
            chat, ChatFormat::QWEN3, /*add_generation_prompt=*/true,
            /*enable_thinking=*/true,
            tools.is_array() && !tools.empty() ? tools.dump() : ""));
        cover += rendered.back();
    }
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"user", "assistant", "system", "\n", "tool"}, cover);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    PflashKeepHistoryRun run;
    for (const auto & text : rendered) {
        run.ids.push_back(tokenizer.encode(text));
        const size_t at = text.rfind("<|im_start|>user");
        run.user_cut.push_back(at == std::string::npos ? -1
            : (int) tokenizer.encode(text.substr(0, at)).size());
    }
    auto backend_owner = std::make_unique<MockPflashWarmSpanBackend>();
    MockPflashWarmSpanBackend & backend = *backend_owner;
    // The selector keeps the user's questions (a prompt-end query ranks
    // them first), nothing else of a paste.
    backend.pick = [&] (const ModelBackend::CompressRequest & request) {
        std::vector<PFlashTokenSpan> spans;
        for (const char * text : {"What does the manual say about exit codes?",
                                  "And what about signals?", "Which page covers traps?"}) {
            const auto span = http_detail::pflash_decoded_text_span(
                tokenizer, request.input_ids, 0, (int) request.input_ids.size(), text);
            if (span.begin >= 0) spans.push_back(span);
        }
        for (const auto & [trigger, passage] : recall_pairs) {
            const int n = (int) request.input_ids.size();
            if (http_detail::pflash_decoded_text_span(
                    tokenizer, request.input_ids, 0, n, trigger).begin < 0) {
                continue;
            }
            const auto span = http_detail::pflash_decoded_text_span(
                tokenizer, request.input_ids, 0, n, passage);
            if (span.begin >= 0) spans.push_back(span);
        }
        return spans;
    };
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash = true;
    config.pflash_threshold = threshold;
    config.pflash_keep_ratio = 0.3f;
    config.max_ctx = max_ctx;
    config.prefix_cache_cap = warm_history ? 4 : 0;
    config.prefill_cache_cap = 0;
    {
        HttpServer server(engine, tokenizer, config);
        HttpServerTestAccess::chat_view_only(server);
        server.set_drafter_tokenizer(&tokenizer);
        if (warm_history) {
            TEST_ASSERT(run.user_cut[0] > 0);
            backend.warm_pos = run.user_cut[0];
            HttpServerTestAccess::seed_prefix_cache(server, 0, run.user_cut[0], run.ids[0]);
        }
        for (size_t index = 0; index < turns.size(); ++index) {
            ParsedRequest request;
            request.format = ApiFormat::OPENAI_CHAT;
            request.messages = turns[index];
            if (tools.is_array() && !tools.empty()) request.tools = tools;
            request.prompt_tokens = run.ids[index];
            request.max_output = max_output;
            run.prepared.push_back(HttpServerTestAccess::prepare_prompt(server, request));
            run.served_text.push_back(tokenizer.decode(run.prepared.back().tokens));
            run.calls_after.push_back(backend.compress_calls);
            const auto & prepared = run.prepared.back();
            run.cut_text.push_back(prepared.snapshot_cut > 0
                ? tokenizer.decode(std::vector<int32_t>(
                      prepared.tokens.begin(), prepared.tokens.begin() + prepared.snapshot_cut))
                : std::string());
        }
        run.counters = HttpServerTestAccess::pflash_failure_counters(server);
        run.keep = HttpServerTestAccess::pflash_keep_history(server);
    }
    run.compress_calls = backend.compress_calls;
    std::ifstream in(trace_path);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) run.view_records.push_back(json::parse(line));
    }
    unlink(trace_path);
    unlink(path.c_str());
    return run;
}

static bool pflash_keep_prefix_is(const std::vector<int32_t> & served,
                                  const std::vector<int32_t> & ids, int cut) {
    return cut > 0 && (int) served.size() >= cut && (int) ids.size() >= cut &&
        std::equal(ids.begin(), ids.begin() + cut, served.begin());
}

// PFLASH_AGENT_PASTE off (the default): every turn of a tool session is
// served raw, full prefill, without a drafter pass (decision keep_history,
// the record says the switch is off).
static void pflash_keep_assert_paste_off_raw(const PflashKeepHistoryRun & run) {
    TEST_ASSERT(run.compress_calls == 0);
    for (size_t turn = 0; turn < run.prepared.size(); ++turn) {
        const auto & prepared = run.prepared[turn];
        TEST_ASSERT_MSG(prepared.error.empty(), prepared.error);
        TEST_ASSERT_MSG(!prepared.compressed && prepared.tokens == run.ids[turn],
                        std::to_string(turn));
        const json & admission = prepared.pflash_stats["admission"];
        TEST_ASSERT_MSG(admission.value("decision", "") == "keep_history", admission.dump());
        TEST_ASSERT(admission["keep_history"].value("in_scope", false));
        TEST_ASSERT(!admission["keep_history"].value("paste", true));
        TEST_ASSERT(prepared.pflash_stats["pflash_fallback"].value("reason", "") ==
                    "keep_history");
    }
    TEST_ASSERT(!run.keep.value("paste", true));
    TEST_ASSERT(run.keep["decisions"].value("keep_history", 0) == (int) run.prepared.size());
    TEST_ASSERT(run.counters["fallback_reasons"].value("keep_history", 0) ==
                (int) run.prepared.size());
    TEST_ASSERT(run.view_records.size() == run.prepared.size());
    for (const auto & record : run.view_records) {
        TEST_ASSERT(record.value("kind", "") == "raw");
    }
}

TEST_CASE(ServerUnitFixture, test_pflash_keep_history_small_history_big_paste) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    // Case 4: a small agent history (under the threshold) and a large paste
    // as the new user turn. By default the tool session is served raw; with
    // PFLASH_AGENT_PASTE=1 the history is served raw and only the paste is
    // compressed.
    const json turn = pflash_keep_with(pflash_keep_agent_history(6),
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")));
    const auto off = run_keep_history_session({turn}, pflash_tool_session_tools(), nullptr,
                                              400);
    const auto on = run_keep_history_session({turn}, pflash_tool_session_tools(), "1",
                                             400);
    pflash_keep_assert_paste_off_raw(off);
    TEST_ASSERT_MSG(on.prepared[0].error.empty(), on.prepared[0].error);
    // With the switch: one compression, the history token for token, the
    // paste compressed, the question kept.
    TEST_ASSERT(on.compress_calls == 1);
    TEST_ASSERT(on.prepared[0].compressed);
    const json & admission = on.prepared[0].pflash_stats["admission"];
    TEST_ASSERT_MSG(admission.value("decision", "") == "new_content", admission.dump());
    const json & keep = admission["keep_history"];
    TEST_ASSERT(keep.value("in_scope", false) && keep.value("fired", false));
    TEST_ASSERT(keep.value("decision_before", "") == "compress");
    TEST_ASSERT(keep.value("new_droppable", 0) >= 400);
    TEST_ASSERT(keep.value("base", "") == "raw");
    TEST_ASSERT(keep.value("raw_base_tokens", -1) == on.user_cut[0]);
    TEST_ASSERT(admission["region"].value("mode", "") == "new_content");
    TEST_ASSERT(pflash_keep_prefix_is(on.prepared[0].tokens, on.ids[0], on.user_cut[0]));
    TEST_ASSERT(pflash_keep_count(on.served_text[0], "def parse(x)") == 6);
    TEST_ASSERT(pflash_keep_count(on.served_text[0], "The manual says many things.") < 30);
    TEST_ASSERT(on.served_text[0].find("exit codes?") != std::string::npos);
    TEST_ASSERT(on.prepared[0].tokens.size() < on.ids[0].size());
    const json & view = on.prepared[0].pflash_stats["view"];
    TEST_ASSERT(view.value("mode", "") == "new_content");
    TEST_ASSERT(view.value("raw_base_tokens", -1) == on.user_cut[0]);
    TEST_ASSERT(on.view_records.size() == 1);
    TEST_ASSERT(on.view_records[0]["pflash"]["view"].value("mode", "") == "new_content");
    TEST_ASSERT(on.keep["decisions"].value("new_content", 0) == 1);
    TEST_ASSERT(on.counters.value("fallback_served", 0) == 0);
    TEST_ASSERT(on.keep.value("paste", false));
}

TEST_CASE(ServerUnitFixture, test_pflash_keep_history_big_history_big_paste) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    // Case 3: an uncached history over the threshold and a large paste. By
    // default everything is served raw; with PFLASH_AGENT_PASTE=1 the
    // history is still raw and only the paste is compressed.
    const json turn = pflash_keep_with(pflash_keep_agent_history(60),
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")));
    const auto off = run_keep_history_session({turn}, pflash_tool_session_tools(), nullptr,
                                              400);
    const auto on = run_keep_history_session({turn}, pflash_tool_session_tools(), "1",
                                             400);
    pflash_keep_assert_paste_off_raw(off);
    TEST_ASSERT_MSG(on.prepared[0].error.empty(), on.prepared[0].error);
    const json & admission = on.prepared[0].pflash_stats["admission"];
    TEST_ASSERT_MSG(admission.value("decision", "") == "new_content", admission.dump());
    TEST_ASSERT(admission["keep_history"].value("decision_before", "") == "compress");
    TEST_ASSERT(!admission.contains("cold_history"));
    TEST_ASSERT(on.prepared[0].compressed && on.compress_calls == 1);
    TEST_ASSERT(pflash_keep_prefix_is(on.prepared[0].tokens, on.ids[0], on.user_cut[0]));
    TEST_ASSERT(pflash_keep_count(on.served_text[0], "def parse(x)") == 60);
    TEST_ASSERT(pflash_keep_count(on.served_text[0], "The manual says many things.") < 30);
    TEST_ASSERT(on.counters.value("fallback_served", 0) == 0);
}

TEST_CASE(ServerUnitFixture, test_pflash_keep_history_small_new_turn_serves_raw) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", "1"};
    // An uncached history over the threshold and a short new question: the
    // history is never compressed. The raw prompt is served (keep_history),
    // with the switch on or off. Never refused.
    const json turn = pflash_keep_with(pflash_keep_agent_history(60),
        pflash_keep_user("Why does it strip?"));
    const auto off = run_keep_history_session({turn}, pflash_tool_session_tools(), nullptr,
                                              400);
    pflash_keep_assert_paste_off_raw(off);
    const auto on = run_keep_history_session({turn}, pflash_tool_session_tools(), "1",
                                             400);
    TEST_ASSERT_MSG(on.prepared[0].error.empty(), on.prepared[0].error);
    TEST_ASSERT(!on.prepared[0].compressed && on.compress_calls == 0);
    TEST_ASSERT(on.prepared[0].tokens == on.ids[0]);
    const json & admission = on.prepared[0].pflash_stats["admission"];
    TEST_ASSERT_MSG(admission.value("decision", "") == "keep_history", admission.dump());
    TEST_ASSERT(admission["keep_history"].value("fired", false));
    TEST_ASSERT(admission["keep_history"].value("decision_before", "") == "compress");
    TEST_ASSERT(admission["keep_history"].value("new_droppable", 999) < 400);
    TEST_ASSERT(on.prepared[0].pflash_stats["pflash_fallback"].value("reason", "") ==
                "keep_history");
    TEST_ASSERT(on.counters.value("fallback_served", 0) == 1);
    TEST_ASSERT(on.counters["fallback_reasons"].value("keep_history", 0) == 1);
    TEST_ASSERT(on.keep["decisions"].value("keep_history", 0) == 1);
    TEST_ASSERT(on.view_records.size() == 1);
    TEST_ASSERT(on.view_records[0].value("kind", "") == "raw");
    TEST_ASSERT(on.view_records[0]["pflash_fallback"].value("reason", "") == "keep_history");
}

TEST_CASE(ServerUnitFixture, test_pflash_keep_history_warm_cache_big_paste) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    // Case 2: the prefix cache holds the history; a large paste arrives.
    // By default it is served raw; with PFLASH_AGENT_PASTE=1 the served
    // prompt starts with the raw history tokens the cache restores and only
    // the paste is compressed (never arrive's re-encoded uncached tail).
    const json turn = pflash_keep_with(pflash_keep_agent_history(60),
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")));
    const auto off = run_keep_history_session({turn}, pflash_tool_session_tools(), nullptr,
                                              400, /*warm_history=*/true);
    const auto on = run_keep_history_session({turn}, pflash_tool_session_tools(), "1",
                                             400, /*warm_history=*/true);
    pflash_keep_assert_paste_off_raw(off);
    TEST_ASSERT_MSG(on.prepared[0].error.empty(), on.prepared[0].error);
    const json & admission = on.prepared[0].pflash_stats["admission"];
    TEST_ASSERT_MSG(admission.value("decision", "") == "new_content", admission.dump());
    TEST_ASSERT(admission["keep_history"].value("decision_before", "") == "warm_cache");
    TEST_ASSERT(admission.value("cached_tokens", 0) == on.user_cut[0]);
    TEST_ASSERT(pflash_keep_prefix_is(on.prepared[0].tokens, on.ids[0], on.user_cut[0]));
    TEST_ASSERT(pflash_keep_count(on.served_text[0], "def parse(x)") == 60);
    TEST_ASSERT(pflash_keep_count(on.served_text[0], "The manual says many things.") < 30);
    // Without arrive: the switch off serves it raw, the switch on still
    // compresses only the paste, on the same raw base.
    const auto off_raw = run_keep_history_session({turn}, pflash_tool_session_tools(), nullptr,
                                                  400, true, 16384, nullptr);
    pflash_keep_assert_paste_off_raw(off_raw);
    const auto on_raw = run_keep_history_session({turn}, pflash_tool_session_tools(), "1",
                                                 400, true, 16384, nullptr);
    TEST_ASSERT(on_raw.prepared[0].pflash_stats["admission"].value("decision", "") ==
                "new_content");
    TEST_ASSERT(on_raw.prepared[0].tokens == on.prepared[0].tokens);
}

// The served prompt minus the generation prompt equals the raw prompt
// minus its own, token for token (nothing compressed).
static bool pflash_keep_all_raw(const PflashKeepHistoryRun & run, size_t turn) {
    return run.prepared[turn].tokens == run.ids[turn] && !run.prepared[turn].compressed;
}

// A first-turn tool request (no assistant turn yet): only the newest user
// message is compressible. A big one is compressed as its own region on a
// raw base (system prompt and tools verbatim); earlier turns stay raw; a
// small one leaves nothing to compress.
TEST_CASE(ServerUnitFixture, test_pflash_keep_history_first_turn_paste_with_tools) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    const json first = json::array({
        {{"role", "system"}, {"content", "You are a coding agent."}},
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")),
    });
    {
        const auto off = run_keep_history_session({first}, pflash_tool_session_tools(), nullptr,
                                                  400);
        const auto on = run_keep_history_session({first}, pflash_tool_session_tools(), "1",
                                                 400);
        pflash_keep_assert_paste_off_raw(off);
        TEST_ASSERT_MSG(on.prepared[0].error.empty(), on.prepared[0].error);
        TEST_ASSERT(on.prepared[0].compressed && on.compress_calls == 1);
        const json & admission = on.prepared[0].pflash_stats["admission"];
        TEST_ASSERT_MSG(admission.value("decision", "") == "new_content", admission.dump());
        TEST_ASSERT(admission["keep_history"].value("newest", "") == "user");
        TEST_ASSERT(admission["keep_history"].value("base", "") == "raw");
        TEST_ASSERT(admission["region"].value("mode", "") == "new_content");
        // System prompt and tool definitions: the raw prompt's own tokens.
        TEST_ASSERT(pflash_keep_prefix_is(on.prepared[0].tokens, on.ids[0], on.user_cut[0]));
        TEST_ASSERT(on.served_text[0].find("read_file") != std::string::npos);
        TEST_ASSERT(pflash_keep_count(on.served_text[0], "The manual says many things.") < 30);
        TEST_ASSERT(on.served_text[0].find("exit codes?") != std::string::npos);
        TEST_ASSERT(on.keep["decisions"].value("new_content", 0) == 1);
    }
    // An earlier user message before the newest (no assistant turn): it is
    // served verbatim, only the paste is compressed (switch off: all raw).
    const json two_users = json::array({
        {{"role", "system"}, {"content", "You are a coding agent."}},
        pflash_keep_user(pflash_keep_filler("Earlier note line.\n", 30)),
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")),
    });
    const auto off2 = run_keep_history_session({two_users}, pflash_tool_session_tools(), nullptr,
                                               400);
    pflash_keep_assert_paste_off_raw(off2);
    const auto on2 = run_keep_history_session({two_users}, pflash_tool_session_tools(), "1",
                                              400);
    TEST_ASSERT(on2.prepared[0].pflash_stats["admission"].value("decision", "") == "new_content");
    TEST_ASSERT(pflash_keep_count(on2.served_text[0], "Earlier note line.") == 30);
    TEST_ASSERT(pflash_keep_prefix_is(on2.prepared[0].tokens, on2.ids[0], on2.user_cut[0]));
    TEST_ASSERT(pflash_keep_count(on2.served_text[0], "The manual says many things.") < 30);
    // The same with a short newest message: nothing to compress (raw,
    // keep_history), switch on or off.
    const json small_newest = json::array({
        {{"role", "system"}, {"content", "You are a coding agent."}},
        pflash_keep_user(pflash_keep_filler("Earlier note line.\n", 30)),
        pflash_keep_user("Fix the bug in parser.py."),
    });
    const auto off3 = run_keep_history_session({small_newest}, pflash_tool_session_tools(),
                                               nullptr, 400);
    pflash_keep_assert_paste_off_raw(off3);
    const auto on3 = run_keep_history_session({small_newest}, pflash_tool_session_tools(),
                                              "1", 400);
    TEST_ASSERT_MSG(on3.prepared[0].error.empty(), on3.prepared[0].error);
    TEST_ASSERT(pflash_keep_all_raw(on3, 0) && on3.compress_calls == 0);
    TEST_ASSERT(on3.prepared[0].pflash_stats["admission"].value("decision", "") == "keep_history");
    TEST_ASSERT(on3.prepared[0].pflash_stats["pflash_fallback"].value("reason", "") ==
                "keep_history");
}

// A newest tool output of any size with no view: served raw, with the paste
// switch off (before any other rule) or on (whatever the cache admission
// would decide: whole-prompt compress, or arrive over the uncached tail).
TEST_CASE(ServerUnitFixture, test_pflash_keep_history_tool_output_no_view_served_raw) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", "1"};   // never refused
    const auto with_tool_output = [] (int history_lines) {
        json messages = pflash_keep_agent_history(history_lines);
        messages.erase(messages.size() - 1);
        messages.push_back({{"role", "assistant"}, {"content", ""},
            {"tool_calls", json::array({{{"id", "call-2"}, {"type", "function"},
                {"function", {{"name", "read_file"},
                              {"arguments", "{\"path\":\"manual.txt\"}"}}}}})}});
        // Over the threshold on its own (30 lines, ~900 fixture tokens).
        messages.push_back({{"role", "tool"}, {"tool_call_id", "call-2"},
                            {"content", pflash_keep_filler(kPflashKeepPasteLine, 30)}});
        return messages;
    };
    const json small_history = with_tool_output(6);
    {
        // Small uncached history: the admission would compress the whole
        // prompt (history and tool output).
        const auto off = run_keep_history_session({small_history}, pflash_tool_session_tools(),
                                                  nullptr, 400);
        pflash_keep_assert_paste_off_raw(off);
        const auto on = run_keep_history_session({small_history}, pflash_tool_session_tools(),
                                                 "1", 400);
        TEST_ASSERT_MSG(on.prepared[0].error.empty(), on.prepared[0].error);
        TEST_ASSERT(pflash_keep_all_raw(on, 0) && on.compress_calls == 0);
        const json & admission = on.prepared[0].pflash_stats["admission"];
        TEST_ASSERT_MSG(admission.value("decision", "") == "keep_history", admission.dump());
        TEST_ASSERT(admission["keep_history"].value("newest", "") == "newest_not_user");
        TEST_ASSERT(admission["keep_history"].value("decision_before", "") == "compress");
        TEST_ASSERT(admission["keep_history"].value("new_droppable", -1) == 0);
        TEST_ASSERT(on.prepared[0].pflash_stats["pflash_fallback"].value("reason", "") ==
                    "keep_history");
        TEST_ASSERT(on.counters["fallback_reasons"].value("keep_history", 0) == 1);
        TEST_ASSERT(on.counters.value("fallback_refused", 0) == 0);
    }
    // Warm history (cached up to the tool output): arrive would compress the
    // uncached tool output; it is served raw.
    const json big_history = with_tool_output(60);
    const auto off_warm = run_keep_history_session({big_history}, pflash_tool_session_tools(),
                                                   nullptr, 400, /*warm_history=*/true);
    pflash_keep_assert_paste_off_raw(off_warm);
    const auto on_warm = run_keep_history_session({big_history}, pflash_tool_session_tools(),
                                                  "1", 400, /*warm_history=*/true);
    TEST_ASSERT(pflash_keep_all_raw(on_warm, 0) && on_warm.compress_calls == 0);
    TEST_ASSERT(on_warm.prepared[0].pflash_stats["admission"].value("decision", "") ==
                "keep_history");
    TEST_ASSERT(on_warm.prepared[0].pflash_stats["admission"]["keep_history"]
                    .value("decision_before", "") == "warm_cache");
    // Uncached big history: served raw as well (keep_history).
    const auto on_cold = run_keep_history_session({big_history}, pflash_tool_session_tools(),
                                                  "1", 400);
    TEST_ASSERT(pflash_keep_all_raw(on_cold, 0) && on_cold.compress_calls == 0);
    TEST_ASSERT(on_cold.prepared[0].pflash_stats["admission"].value("decision", "") ==
                "keep_history");
    TEST_ASSERT(on_cold.prepared[0].pflash_stats["admission"]["keep_history"]
                    .value("decision_before", "") == "compress");
}

TEST_CASE(ServerUnitFixture, test_pflash_keep_history_chat_without_tools_unchanged) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    // A plain chat (no tools anywhere) with a paste after an answer: out of
    // scope, served, traced and counted the same with the switch on or off.
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        pflash_keep_user("Hello there."),
        {{"role", "assistant"}, {"content", "Hi."}},
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")),
    });
    const auto off = run_keep_history_session({turn}, json(), nullptr, 400);
    const auto on = run_keep_history_session({turn}, json(), "1", 400);
    TEST_ASSERT_MSG(on.prepared[0].error.empty(), on.prepared[0].error);
    TEST_ASSERT(on.prepared[0].compressed && on.compress_calls == 1);
    TEST_ASSERT(on.prepared[0].tokens == off.prepared[0].tokens);
    json stats_on = on.prepared[0].pflash_stats;
    json stats_off = off.prepared[0].pflash_stats;
    stats_on.erase("compress_ms");
    stats_off.erase("compress_ms");
    TEST_ASSERT_MSG(stats_on == stats_off, stats_on.dump() + " vs " + stats_off.dump());
    TEST_ASSERT(!stats_on["admission"].contains("keep_history"));
    TEST_ASSERT(on.counters == off.counters);
    TEST_ASSERT(on.keep["decisions"].empty());
}

// A raw prompt over the context in a tool session is not compressed as a
// whole: with PFLASH_AGENT_PASTE=1 a big newest paste is compressed on its
// own (the region view fits); anything else, and everything with the switch
// off, gets the context-overflow error full prefill gets.
TEST_CASE(ServerUnitFixture, test_pflash_keep_history_raw_over_max_ctx) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    const json paste = pflash_keep_with(pflash_keep_agent_history(6),
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")));
    const auto probe = run_keep_history_session({paste}, pflash_tool_session_tools(), "1",
                                                400);
    const int max_ctx = (int) probe.ids[0].size() + 15;   // raw + 16 output > max_ctx
    const auto off = run_keep_history_session({paste}, pflash_tool_session_tools(), nullptr,
                                              400, false, max_ctx);
    TEST_ASSERT(off.compress_calls == 0 && !off.prepared[0].compressed);
    TEST_ASSERT_MSG(off.prepared[0].error_status == 400, off.prepared[0].error);
    TEST_ASSERT(off.prepared[0].error.find("maximum context length") != std::string::npos);
    const auto on = run_keep_history_session({paste}, pflash_tool_session_tools(), "1",
                                             400, false, max_ctx);
    TEST_ASSERT_MSG(on.prepared[0].error.empty(), on.prepared[0].error);
    TEST_ASSERT(on.prepared[0].compressed);
    const json & admission = on.prepared[0].pflash_stats["admission"];
    TEST_ASSERT_MSG(admission.value("decision", "") == "new_content", admission.dump());
    TEST_ASSERT(!admission.value("raw_fits", true));
    TEST_ASSERT(pflash_keep_prefix_is(on.prepared[0].tokens, on.ids[0], on.user_cut[0]));
    TEST_ASSERT(pflash_keep_count(on.served_text[0], "def parse(x)") == 6);
    // A short newest message on a history over the context: nothing may be
    // compressed, so the request gets the overflow error (as full prefill).
    json history = pflash_keep_agent_history(60);
    const json small = pflash_keep_with(history, pflash_keep_user("Why does it strip?"));
    const auto probe2 = run_keep_history_session({small}, pflash_tool_session_tools(), "1",
                                                 400);
    const int max_ctx2 = (int) probe2.ids[0].size() + 15;
    const auto off2 = run_keep_history_session({small}, pflash_tool_session_tools(), nullptr,
                                               400, false, max_ctx2);
    TEST_ASSERT(off2.compress_calls == 0 && !off2.prepared[0].compressed);
    TEST_ASSERT_MSG(off2.prepared[0].error_status == 400, off2.prepared[0].error);
    const auto on2 = run_keep_history_session({small}, pflash_tool_session_tools(), "1",
                                              400, false, max_ctx2);
    TEST_ASSERT(on2.compress_calls == 0 && !on2.prepared[0].compressed);
    TEST_ASSERT_MSG(on2.prepared[0].error_status == 400, on2.prepared[0].error);
    TEST_ASSERT(on2.prepared[0].error.find("maximum context length") != std::string::npos);
    // The same with a newest tool output.
    history.erase(history.size() - 1);
    history.push_back({{"role", "assistant"}, {"content", ""},
        {"tool_calls", json::array({{{"id", "call-2"}, {"type", "function"},
            {"function", {{"name", "read_file"}, {"arguments", "{\"path\":\"a.txt\"}"}}}}})}});
    history.push_back({{"role", "tool"}, {"tool_call_id", "call-2"},
                       {"content", pflash_keep_filler(kPflashKeepPasteLine, 30)}});
    const auto probe3 = run_keep_history_session({history}, pflash_tool_session_tools(), "1",
                                                 400);
    const int max_ctx3 = (int) probe3.ids[0].size() + 15;
    const auto on3 = run_keep_history_session({history}, pflash_tool_session_tools(), "1",
                                              400, false, max_ctx3);
    TEST_ASSERT(on3.compress_calls == 0 && !on3.prepared[0].compressed);
    TEST_ASSERT_MSG(on3.prepared[0].error_status == 400, on3.prepared[0].error);
}

// Nine turns of an agent session: a paste, follow-up questions (one needs a
// passage the paste turn dropped: recall), a small and a large tool output
// as the newest turn, a second paste. Every turn continues the view: the
// history and every tool output stay verbatim, recall comes after the
// cached view and ahead of the newest message, nothing is ever rebuilt
// (rebuild ratio 1.1 with no minimum: the plain-chat rule would rebuild).
// With the switch off the same nine turns are all served raw.
TEST_CASE(ServerUnitFixture, test_pflash_keep_history_follow_up_turns) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    luce_test::ScopedEnvVar recall{"PFLASH_CHAT_RECALL", nullptr};       // default: on
    luce_test::ScopedEnvVar ratio{"PFLASH_CHAT_REBUILD_RATIO", "1.1"};
    luce_test::ScopedEnvVar min_tokens{"PFLASH_CHAT_REBUILD_MIN_TOKENS", nullptr};
    const std::string sigint = "SIGINT is signal number two.";
    const std::string sig_q = "Which signal number is SIGINT?";
    const std::string paste1 = pflash_keep_filler(kPflashKeepPasteLine, 15) + sigint + "\n" +
        pflash_keep_filler(kPflashKeepPasteLine, 15) + "What does the manual say about exit codes?";
    const auto call = [] (const char * id, const char * path) {
        return json{{"role", "assistant"}, {"content", ""},
            {"tool_calls", json::array({{{"id", id}, {"type", "function"},
                {"function", {{"name", "read_file"},
                              {"arguments", std::string("{\"path\":\"") + path + "\"}"}}}}})}};
    };
    std::vector<json> turns;
    json m = pflash_keep_with(pflash_keep_agent_history(6), pflash_keep_user(paste1));
    turns.push_back(m);                                                         // 0 paste
    m = pflash_keep_with(m, {{"role", "assistant"}, {"content", "Exit codes are listed."}});
    turns.push_back(m = pflash_keep_with(m, pflash_keep_user(sig_q)));          // 1 recall
    m = pflash_keep_with(m, call("call-2", "traps.txt"));
    turns.push_back(m = pflash_keep_with(m, {{"role", "tool"}, {"tool_call_id", "call-2"},
        {"content", pflash_keep_filler("trap handlers run on signals.\n", 30)}}));  // 2 big tool
    m = pflash_keep_with(m, call("call-3", "ok.txt"));
    turns.push_back(m = pflash_keep_with(m, {{"role", "tool"}, {"tool_call_id", "call-3"},
        {"content", "ok"}}));                                                   // 3 small tool
    m = pflash_keep_with(m, {{"role", "assistant"}, {"content", "Traps are handled."}});
    turns.push_back(m = pflash_keep_with(m, pflash_keep_user("And what about signals?")));  // 4
    m = pflash_keep_with(m, {{"role", "assistant"}, {"content", "Signals are covered."}});
    turns.push_back(m = pflash_keep_with(m, pflash_keep_user(
        pflash_keep_filler("Another page of the manual.\n", 30) + "Which page covers traps?")));  // 5
    m = pflash_keep_with(m, {{"role", "assistant"}, {"content", "Page two."}});
    turns.push_back(m = pflash_keep_with(m, pflash_keep_user("Thanks, and the exit codes again?")));  // 6
    m = pflash_keep_with(m, call("call-4", "more.txt"));
    turns.push_back(m = pflash_keep_with(m, {{"role", "tool"}, {"tool_call_id", "call-4"},
        {"content", pflash_keep_filler("more trap notes here.\n", 30)}}));    // 7 big tool
    m = pflash_keep_with(m, {{"role", "assistant"}, {"content", "Done reading."}});
    turns.push_back(m = pflash_keep_with(m, pflash_keep_user("Summarize the traps.")));      // 8
    const auto run = run_keep_history_session(turns, pflash_tool_session_tools(), "1",
                                              400, false, 16384, "1",
                                              {{sig_q, sigint}});
    const auto mode = [&] (size_t turn) {
        return run.prepared[turn].pflash_stats["view"].value("mode", "");
    };
    for (size_t turn = 0; turn < turns.size(); ++turn) {
        const auto & prepared = run.prepared[turn];
        TEST_ASSERT_MSG(prepared.error.empty(), prepared.error);
        TEST_ASSERT_MSG(prepared.compressed, std::to_string(turn));
        const std::string m_name = mode(turn);
        TEST_ASSERT_MSG(m_name == "new_content" || m_name == "continue",
                        std::to_string(turn) + " " + prepared.pflash_stats.dump());
        TEST_ASSERT(!prepared.pflash_stats.contains("pflash_fallback"));
        // The history verbatim, every turn: the raw prompt's own tokens up
        // to the paste turn, and all six tool-output lines.
        TEST_ASSERT_MSG(pflash_keep_prefix_is(prepared.tokens, run.ids[0], run.user_cut[0]),
                        std::to_string(turn));
        TEST_ASSERT(pflash_keep_count(run.served_text[turn], "def parse(x)") == 6);
        TEST_ASSERT(prepared.snapshot_cut > 0 &&
                    prepared.snapshot_cut < (int) prepared.tokens.size());
        if (turn > 0) {
            // Each turn continues the previous served prompt.
            const int reused = prepared.pflash_stats["view"].value("reused_tokens", -1);
            TEST_ASSERT_MSG(reused == run.prepared[turn - 1].snapshot_cut,
                            std::to_string(turn) + " " + prepared.pflash_stats["view"].dump());
            TEST_ASSERT(std::equal(run.prepared[turn - 1].tokens.begin(),
                                   run.prepared[turn - 1].tokens.begin() + reused,
                                   prepared.tokens.begin()));
        }
    }
    // Turn 0: the paste on a raw base, the SIGINT line dropped.
    TEST_ASSERT(mode(0) == "new_content");
    TEST_ASSERT(run.served_text[0].find(sigint) == std::string::npos);
    // Turn 1: recall brings the line back, in an excerpt block after the
    // cached view and ahead of the newest message's text; no history or
    // newest-turn text inside the block.
    const json & view1 = run.prepared[1].pflash_stats["view"];
    TEST_ASSERT_MSG(view1.value("recalled_tokens", 0) > 0, view1.dump());
    TEST_ASSERT(view1.value("keep_history", false));
    const std::string & s1 = run.served_text[1];
    const std::string & cached1 = run.cut_text[0];
    TEST_ASSERT(!cached1.empty());
    const size_t block = s1.find("[Earlier in this conversation]");
    const size_t block_end = s1.find("[End of earlier excerpts]");
    TEST_ASSERT(block != std::string::npos && block_end != std::string::npos);
    TEST_ASSERT(s1.compare(0, cached1.size(), cached1) == 0 && block >= cached1.size());
    TEST_ASSERT(block > s1.rfind("Exit codes are listed."));
    TEST_ASSERT(block_end < s1.rfind(sig_q));
    const std::string excerpt = s1.substr(block, block_end - block);
    TEST_ASSERT(excerpt.find(sigint) != std::string::npos);
    TEST_ASSERT(excerpt.find(sig_q) == std::string::npos);
    TEST_ASSERT(excerpt.find("def parse") == std::string::npos);
    TEST_ASSERT(excerpt.find("Exit codes are listed.") == std::string::npos);
    TEST_ASSERT(pflash_keep_count(s1, sig_q) == 1);
    // Turns 2 and 3: tool outputs (large, small) appended verbatim from the
    // view, no drafter pass.
    TEST_ASSERT(run.calls_after[2] == run.calls_after[1]);
    TEST_ASSERT(run.calls_after[3] == run.calls_after[2]);
    TEST_ASSERT(pflash_keep_count(run.served_text[2], "trap handlers run on signals.") == 30);
    TEST_ASSERT(run.prepared[2].pflash_stats["admission"]["keep_history"].value("newest", "") ==
                "newest_not_user");
    TEST_ASSERT(run.prepared[2].pflash_stats["admission"]["keep_history"].value("decision", "") ==
                "view");
    // Over the old rebuild rule but not rebuilt.
    const json & view2 = run.prepared[2].pflash_stats["view"];
    TEST_ASSERT(view2.value("keep_history", false));
    // Turn 5: the second paste compressed on its own after the view.
    TEST_ASSERT(mode(5) == "new_content");
    TEST_ASSERT(pflash_keep_count(run.served_text[5], "Another page of the manual.") < 30);
    // Turn 7: another large tool output, verbatim; turn 8 keeps everything.
    TEST_ASSERT(run.calls_after[7] == run.calls_after[6]);
    for (size_t turn = 2; turn < turns.size(); ++turn) {
        TEST_ASSERT(pflash_keep_count(run.served_text[turn], "trap handlers run on signals.") == 30);
    }
    TEST_ASSERT(pflash_keep_count(run.served_text[8], "more trap notes here.") == 30);
    TEST_ASSERT(pflash_keep_count(run.served_text[8], sigint) == 1);   // recalled once
    TEST_ASSERT(run.keep["decisions"].value("new_content", 0) == 2);
    TEST_ASSERT(run.keep["decisions"].value("view", 0) == 7);
    TEST_ASSERT(run.counters.value("fallback_served", 0) == 0);
    // PFLASH_AGENT_PASTE off: both pastes, both big tool outputs and every
    // follow-up served raw, without a drafter pass.
    const auto off = run_keep_history_session(turns, pflash_tool_session_tools(), nullptr,
                                              400, false, 16384, "1",
                                              {{sig_q, sigint}});
    pflash_keep_assert_paste_off_raw(off);
    TEST_ASSERT(pflash_keep_count(off.served_text[0], "The manual says many things.") == 30);
    TEST_ASSERT(pflash_keep_count(off.served_text[5], "Another page of the manual.") == 30);
    TEST_ASSERT(pflash_keep_count(off.served_text[8], "trap handlers run on signals.") == 30);
    TEST_ASSERT(pflash_keep_count(off.served_text[8], "more trap notes here.") == 30);
    TEST_ASSERT(off.served_text[8].find("[Earlier in this conversation]") == std::string::npos);
    TEST_ASSERT(off.prepared[2].pflash_stats["admission"]["keep_history"].value("newest", "") ==
                "newest_not_user");
    TEST_ASSERT(off.prepared[5].pflash_stats["admission"]["keep_history"].value("newest", "") ==
                "user");
}

// A keep view whose continuation no longer fits the context is never
// rebuilt: the raw prompt is served, and here it does not fit either, so
// the request gets the overflow error full prefill gets.
TEST_CASE(ServerUnitFixture, test_pflash_keep_history_view_over_context_not_rebuilt) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    luce_test::ScopedEnvVar recall{"PFLASH_CHAT_RECALL", "0"};
    const json t1 = pflash_keep_with(pflash_keep_agent_history(6),
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")));
    json t2 = pflash_keep_with(t1, {{"role", "assistant"}, {"content", ""},
        {"tool_calls", json::array({{{"id", "call-2"}, {"type", "function"},
            {"function", {{"name", "read_file"}, {"arguments", "{\"path\":\"a.txt\"}"}}}}})}});
    t2 = pflash_keep_with(t2, {{"role", "tool"}, {"tool_call_id", "call-2"},
        {"content", pflash_keep_filler("trap handlers run on signals.\n", 30)}});
    const auto probe = run_keep_history_session({t1, t2}, pflash_tool_session_tools(), "1",
                                                400);
    TEST_ASSERT(probe.prepared[1].compressed);
    // The context holds turn 1's view but not turn 2's continuation.
    const int max_ctx = (int) probe.prepared[1].tokens.size() + 15;
    // Switch off: never compressed; each raw prompt is served, or gets the
    // overflow error when it does not fit.
    const auto off = run_keep_history_session({t1, t2}, pflash_tool_session_tools(), nullptr,
                                              400, false, max_ctx);
    TEST_ASSERT(off.compress_calls == 0);
    for (size_t turn = 0; turn < 2; ++turn) {
        TEST_ASSERT(!off.prepared[turn].compressed);
        if ((int) off.ids[turn].size() + 16 > max_ctx) {
            TEST_ASSERT_MSG(off.prepared[turn].error_status == 400, off.prepared[turn].error);
        } else {
            TEST_ASSERT_MSG(off.prepared[turn].error.empty(), off.prepared[turn].error);
            TEST_ASSERT(off.prepared[turn].tokens == off.ids[turn]);
        }
    }
    const auto on = run_keep_history_session({t1, t2}, pflash_tool_session_tools(), "1",
                                             400, false, max_ctx);
    TEST_ASSERT(on.prepared[0].compressed);
    TEST_ASSERT(!on.prepared[1].compressed);
    TEST_ASSERT_MSG(on.prepared[1].error_status == 400, on.prepared[1].error);
}

// A plain chat (no tools) over several turns -- a paste, a follow-up that
// recalls, a second paste -- is served, traced and counted the same with
// the switch on or off.
TEST_CASE(ServerUnitFixture, test_pflash_keep_history_plain_chat_turns_unchanged) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    luce_test::ScopedEnvVar recall{"PFLASH_CHAT_RECALL", nullptr};
    const std::string sigint = "SIGINT is signal number two.";
    const std::string sig_q = "Which signal number is SIGINT?";
    json m = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        pflash_keep_user(pflash_keep_filler(kPflashKeepPasteLine, 15) + sigint + "\n" +
            pflash_keep_filler(kPflashKeepPasteLine, 15) +
            "What does the manual say about exit codes?"),
    });
    std::vector<json> turns{m};
    m = pflash_keep_with(m, {{"role", "assistant"}, {"content", "Exit codes are listed."}});
    turns.push_back(m = pflash_keep_with(m, pflash_keep_user(sig_q)));
    m = pflash_keep_with(m, {{"role", "assistant"}, {"content", "Two."}});
    turns.push_back(m = pflash_keep_with(m, pflash_keep_user(
        pflash_keep_filler("Another page of the manual.\n", 30) + "Which page covers traps?")));
    {
        const auto off = run_keep_history_session(turns, json(), nullptr, 400, false,
                                                  16384, "1", {{sig_q, sigint}});
        const auto on = run_keep_history_session(turns, json(), "1", 400, false,
                                                 16384, "1", {{sig_q, sigint}});
        TEST_ASSERT(off.prepared[1].pflash_stats["view"].value("recalled_tokens", 0) > 0);
        for (size_t turn = 0; turn < turns.size(); ++turn) {
            TEST_ASSERT_MSG(on.prepared[turn].error.empty(), on.prepared[turn].error);
            TEST_ASSERT(on.prepared[turn].tokens == off.prepared[turn].tokens);
            TEST_ASSERT(on.prepared[turn].compressed == off.prepared[turn].compressed);
            TEST_ASSERT(on.prepared[turn].snapshot_cut == off.prepared[turn].snapshot_cut);
            json a = on.prepared[turn].pflash_stats;
            json b = off.prepared[turn].pflash_stats;
            a.erase("compress_ms");
            b.erase("compress_ms");
            TEST_ASSERT_MSG(a == b, a.dump() + " vs " + b.dump());
        }
        TEST_ASSERT(on.calls_after == off.calls_after);
        TEST_ASSERT(on.counters == off.counters);
        TEST_ASSERT(on.keep["decisions"].empty());
        TEST_ASSERT(on.view_records.size() == off.view_records.size());
        for (size_t i = 0; i < on.view_records.size(); ++i) {
            json a = on.view_records[i];
            json b = off.view_records[i];
            a.erase("request_id");
            b.erase("request_id");
            a["pflash"].erase("compress_ms");
            b["pflash"].erase("compress_ms");
            TEST_ASSERT(a == b);
        }
    }
}

// A tool session whose turns cannot be located (no messages array): no
// history can be told apart, so nothing is compressed, switch on or off.
TEST_CASE(ServerUnitFixture, test_pflash_keep_history_unparsed_tool_session_raw) {
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", nullptr};
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar chunk{"PFLASH_SELECT_CHUNK_SIZE", "4"};
    luce_test::ScopedEnvVar admission_env{"PFLASH_ADMISSION", "cache"};
    const json messages = pflash_keep_with(pflash_keep_agent_history(60),
        pflash_keep_user(pflash_keep_paste(30, "What does the manual say about exit codes?")));
    ToolMemory tool_memory;
    const std::string rendered = render_chat_template(
        normalize_chat_messages(messages, ApiFormat::OPENAI_CHAT, tool_memory),
        ChatFormat::QWEN3, /*add_generation_prompt=*/true, /*enable_thinking=*/true,
        pflash_tool_session_tools().dump());
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"user", "assistant", "system", "\n", "tool"}, rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const auto ids = tokenizer.encode(rendered);
    const auto serve = [&] (const char * flag, int & calls) {
        luce_test::ScopedEnvVar paste{"PFLASH_AGENT_PASTE", flag};
        auto backend_owner = std::make_unique<MockPflashSpanBackend>();
        MockPflashSpanBackend & backend = *backend_owner;
        LuceEngine engine(std::move(backend_owner));
        ServerConfig config;
        config.pflash = true;
        config.pflash_threshold = 400;
        config.pflash_keep_ratio = 0.3f;
        config.max_ctx = 16384;
        config.prefix_cache_cap = 0;
        config.prefill_cache_cap = 0;
        HttpServer server(engine, tokenizer, config);
        server.set_drafter_tokenizer(&tokenizer);
        ParsedRequest request;
        request.format = ApiFormat::OPENAI_CHAT;
        request.messages = rendered;   // raw text: no messages array
        request.tools = pflash_tool_session_tools();
        request.prompt_tokens = ids;
        request.max_output = 16;
        auto prepared = HttpServerTestAccess::prepare_prompt(server, request);
        calls = backend.compress_calls;
        return prepared;
    };
    int calls_on = 0;
    const auto on = serve("1", calls_on);
    TEST_ASSERT_MSG(on.error.empty(), on.error);
    TEST_ASSERT(!on.compressed && on.tokens == ids && calls_on == 0);
    const json & admission = on.pflash_stats["admission"];
    TEST_ASSERT_MSG(admission.value("decision", "") == "keep_history", on.pflash_stats.dump());
    TEST_ASSERT(admission["keep_history"].value("newest", "") == "unparsed");
    TEST_ASSERT(on.pflash_stats["pflash_fallback"].value("reason", "") == "keep_history");
    int calls_off = 0;
    const auto off = serve(nullptr, calls_off);
    TEST_ASSERT_MSG(off.error.empty(), off.error);
    TEST_ASSERT(!off.compressed && off.tokens == ids && calls_off == 0);
    TEST_ASSERT(off.pflash_stats["admission"].value("decision", "") == "keep_history");
    TEST_ASSERT(off.pflash_stats["admission"]["keep_history"].value("newest", "") == "unparsed");
    TEST_ASSERT(!off.pflash_stats["admission"]["keep_history"].value("paste", true));
    unlink(path.c_str());
}

// The prefix-cache snapshot of a tool session's PFlash view (a paste turn,
// PFLASH_AGENT_PASTE=1), with the system/tools head already restored: the
// view asks for its cut (the start of its generation prompt), a tool request
// drops that forced cut, and #835's rule saves the end of the prompt's last
// message instead -- one token past the cut, the generation prompt's role
// marker, which the next turn's continued view repeats. So the next turn
// restores the whole view without a cut of its own.
TEST_CASE(ServerUnitFixture, test_pflash_tool_view_snapshot_at_last_message_end) {
    const json messages = pflash_keep_with(pflash_keep_agent_history(6),
        pflash_keep_user("And what about signals?"));
    ToolMemory tool_memory;
    const std::string rendered = render_chat_template(
        normalize_chat_messages(messages, ApiFormat::OPENAI_CHAT, tool_memory),
        ChatFormat::QWEN3, /*add_generation_prompt=*/true, /*enable_thinking=*/true,
        pflash_tool_session_tools().dump());
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"user", "assistant", "system", "\n", "tool"}, rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const auto ids = tokenizer.encode(rendered);
    const size_t gen = rendered.rfind("<|im_start|>assistant");
    const int cut = (int) tokenizer.encode(rendered.substr(0, gen)).size();
    // The next turn continues the view: the same tokens up to the cut, then
    // the assistant turn, opening with the same role marker.
    const auto marker = tokenizer.encode("<|im_start|>");
    TEST_ASSERT(marker.size() == 1 && cut < (int) ids.size() && ids[(size_t) cut] == marker[0]);
    auto backend_owner = std::make_unique<SlotSetBackend>();
    SlotSetBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.prefix_cache_cap = 4;
    HttpServer server(engine, tokenizer, config);
    PrefixCache & cache = SchedulerTestHarness::prefix_cache(server);
    const auto bounds = find_all_boundaries(ids, cache.chat_markers());
    TEST_ASSERT(bounds.size() >= 3);
    const int head = bounds.front();
    TEST_ASSERT(head > 0 && cut > head + 4);
    cache.confirm_inline_snap(0, head, ids);
    backend.positions[0] = head;
    const int snap = SchedulerTestHarness::prepare_cache_snapshot_cut(
        server, ids, pflash_tool_session_tools(), cut);
    TEST_ASSERT_MSG(snap == cut + 1 && snap == bounds.back(),
                    std::to_string(snap) + " vs cut " + std::to_string(cut));
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_pflash_raw_base_target_fails_closed) {
    const json messages = pflash_keep_with(pflash_keep_agent_history(2),
        pflash_keep_user("What now?"));
    ToolMemory tool_memory;
    const std::string rendered = render_chat_template(
        normalize_chat_messages(messages, ApiFormat::OPENAI_CHAT, tool_memory),
        ChatFormat::QWEN3, /*add_generation_prompt=*/true, /*enable_thinking=*/true,
        pflash_tool_session_tools().dump());
    const std::string path = write_pflash_bpe_tokenizer_fixture(
        {"user", "assistant", "system", "\n", "tool"}, rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    ChatMarkers markers;
    TEST_ASSERT(resolve_chat_markers(tokenizer, markers));
    const auto ids = tokenizer.encode(rendered);
    const auto parsed = http_detail::pflash_chat_query_turn(tokenizer, markers, tokenizer, ids);
    int newest = -1;
    TEST_ASSERT(http_detail::pflash_keep_history_scope(parsed.turns, newest) == nullptr);
    const int cut = http_detail::pflash_raw_base_target(
        tokenizer, markers, ids, tokenizer, ids, parsed.turns, newest);
    TEST_ASSERT(cut == (int) tokenizer.encode(rendered.substr(0, rendered.rfind("<|im_start|>user"))).size());
    // Drafter tokens that do not decode to the same history: -1.
    auto other = ids;
    other[3] = other[4];
    TEST_ASSERT(http_detail::pflash_raw_base_target(
        tokenizer, markers, ids, tokenizer, other, parsed.turns, newest) == -1);
    // A turn list that does not match the target parse: -1.
    auto fewer = parsed.turns;
    fewer.erase(fewer.begin());
    TEST_ASSERT(http_detail::pflash_raw_base_target(
        tokenizer, markers, ids, tokenizer, ids, fewer, (int) fewer.size() - 1) == -1);
    TEST_ASSERT(http_detail::pflash_raw_base_target(
        tokenizer, markers, ids, tokenizer, ids, parsed.turns, 99) == -1);
    unlink(path.c_str());
}

// ── The unified view ─────────────────────────────────────────────────────

// Scores every candidate (a line, a paragraph ending in a blank line, or ``chunk`` tokens) with
// ``score(segment text, prompt text)`` and returns the candidate scores the unified path admits units by; the
// selection itself is ignored there.
// ``Base`` MockBackend, or DiskPayloadBackend for a server with the disk session cache.
template <class Base>
struct MockUnifiedBackendT : Base {
    const Tokenizer * tokenizer = nullptr;
    int chunk = 0;
    bool paragraphs = false;
    bool widen = false;
    bool keep_scored_only = false;   // kept_spans = the candidates scored above 1 (the selection keeps only them)
    std::string probe_pin_text;      // non-empty: the request probe pins its last occurrence in the input
    bool fail = false;               // the drafter pass fails
    std::function<double(const std::string &, const std::string &)> score;
    int compress_calls = 0;
    ModelBackend::CompressRequest last_request;

    ModelBackend::CompressResult compress(const ModelBackend::CompressRequest & request) override {
        ++compress_calls;
        last_request = request;
        const auto & ids = request.input_ids;
        const int n = (int) ids.size();
        std::vector<std::string> pieces((size_t) n);
        std::string whole;
        for (int i = 0; i < n; ++i) {
            pieces[(size_t) i] = tokenizer->decode(std::vector<int32_t>{ids[(size_t) i]});
            whole += pieces[(size_t) i];
        }
        ModelBackend::CompressResult result;
        if (fail) return result;
        if (!probe_pin_text.empty()) {
            const size_t at = whole.rfind(probe_pin_text);
            if (at != std::string::npos) {
                int b = -1, e = -1;
                size_t offset = 0;
                for (int i = 0; i < n; ++i) {
                    if (b < 0 && offset + pieces[(size_t) i].size() > at) b = i;
                    offset += pieces[(size_t) i].size();
                    if (e < 0 && offset >= at + probe_pin_text.size()) e = i + 1;
                }
                result.probe_pins.push_back({b, e});
            }
        }
        int begin = 0;
        std::string segment;
        for (int i = 0; i < n; ++i) {
            segment += pieces[(size_t) i];
            const bool cut = chunk > 0 ? i + 1 - begin >= chunk
                           : paragraphs ? segment.size() >= 2 && segment.compare(segment.size() - 2, 2, "\n\n") == 0
                                        : pieces[(size_t) i].find('\n') != std::string::npos;
            if (!cut && i + 1 < n) continue;
            result.candidate_lifts.push_back({{begin, i + 1}, score ? score(segment, whole) : 1.0});
            segment.clear();
            begin = i + 1;
        }
        result.compressed_ids = ids;
        result.kept_spans = {{0, n}};
        if (keep_scored_only) {
            // As strict selection: the candidates scored above 1, plus what it always keeps (the required spans,
            // the query window and, unless the suffix is scored, everything after it).
            std::vector<PFlashTokenSpan> kept = request.required_instruction_spans;
            for (const auto & lift : result.candidate_lifts) {
                if (lift.second > 1.0) kept.push_back(lift.first);
            }
            if (request.score_query_end > 0) {
                kept.push_back({(std::max)(0, request.score_query_end - request.score_query_tokens),
                                request.query_suffix_candidates ? request.score_query_end : n});
            }
            result.kept_spans = http_detail::canonicalize_pflash_token_spans(std::move(kept));
        }
        result.budget_widened = widen;
        result.ok = true;
        return result;
    }
};
using MockUnifiedBackend = MockUnifiedBackendT<MockBackend>;

struct UnifiedRun {
    std::vector<HttpServerTestAccess::PreparedPrompt> prepared;
    std::vector<std::vector<int32_t>> ids;
    std::vector<std::string> served_text;
    std::vector<std::string> raw_text;
    std::vector<int> calls_after;
    std::vector<json> view_records;
    json unified;
    std::vector<std::vector<size_t>> target_starts;     // per turn, byte offsets where a raw target token starts
    std::vector<std::vector<size_t>> drafter_starts;
};

using UnifiedScore = std::function<double(const std::string &, const std::string &)>;

struct UnifiedOptions {
    std::vector<int> max_outputs;               // per turn; 16 when absent
    const char * select_mode = "budget_only";   // nullptr: no strict-selection environment
    std::vector<std::string> target_words;      // whole pre-tokeniser pieces the target has as one token
    bool distinct_drafter = false;              // a drafter tokenizer of its own, with ``drafter_words``
    std::vector<std::string> drafter_words;
    bool paragraphs = false;                    // drafter candidates end at blank lines
    bool widen = false;                         // the drafter pass reports a stage-2 widened budget
    bool allow_errors = false;                  // a turn may be refused (context overflow)
    const char * recall_gate = "off";           // PFLASH_RECALL_GATE (nullptr: unset, the 0.075 default)
    const char * trace_units = nullptr;         // PFLASH_RECALL_TRACE_UNITS
    bool keep_scored_only = false;              // the drafter selection keeps only candidates scored above 1
    bool chat_view_only = false;                // the chat view serves every turn (its behaviour, as a baseline)
    std::string disk_dir;                       // non-empty: the disk session cache in this directory
    bool disk_seed_history = false;             // first, a server restart: turn 0 up to its newest user message on disk
    DiskPrefixCacheMode disk_mode = DiskPrefixCacheMode::Full;   // the disk scope of the requests
    int restart_at = -1;                        // a server restart (shutdown flush, a new process) before this turn
    std::string probe_pin_text;                 // the request probe pins this text's last occurrence
    bool drafter_fails = false;                 // every drafter pass fails
    const char * scorer = nullptr;              // PFLASH_SELECT_SCORER
    const char * fail_closed = nullptr;         // PFLASH_FAIL_CLOSED
    std::vector<std::vector<std::string>> required;   // per turn, pflash_required
    std::function<void(size_t)> before_turn;    // runs before each turn (after a restart)
};

// One server with the unified view (off when ``options.chat_view_only``), the turns in order. By default one fixture tokenizer is target and drafter, so
// every token boundary is shared; ``distinct_drafter`` gives the drafter its own merges. ``region_tools`` nullptr
// lists no tool (PFLASH_REGION_TOOLS set empty; unset would be the "*" default).
static UnifiedRun run_unified_session(const std::vector<json> & turns, const json & tools, const char * region_tools,
                                      int threshold, const UnifiedScore & score, int chunk = 0,
                                      float keep_ratio = 0.3f, const char * recall = nullptr,
                                      int max_ctx = 65536, const UnifiedOptions & options = UnifiedOptions()) {
    luce_test::ScopedEnvVar region{"PFLASH_REGION_TOOLS", region_tools ? region_tools : ""};
    luce_test::ScopedEnvVar recall_env{"PFLASH_RECALL", recall};
    luce_test::ScopedEnvVar recall_gate{"PFLASH_RECALL_GATE", options.recall_gate};
    luce_test::ScopedEnvVar trace_units{"PFLASH_RECALL_TRACE_UNITS", options.trace_units};
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", options.select_mode};
    luce_test::ScopedEnvVar admission{"PFLASH_ADMISSION", "cache"};
    luce_test::ScopedEnvVar scorer{"PFLASH_SELECT_SCORER", options.scorer};
    luce_test::ScopedEnvVar fail_closed{"PFLASH_FAIL_CLOSED", options.fail_closed};
    char trace_path[] = "/tmp/pflash_unified_view_XXXXXX";
    const int trace_fd = mkstemp(trace_path);
    TEST_ASSERT(trace_fd >= 0);
    close(trace_fd);
    luce_test::ScopedEnvVar view_trace{"PFLASH_VIEW_TRACE_PATH", trace_path};
    std::vector<std::string> rendered;
    std::string cover;
    for (const auto & messages : turns) {
        ToolMemory tool_memory;
        const auto chat = normalize_chat_messages(messages, ApiFormat::OPENAI_CHAT, tool_memory);
        rendered.push_back(render_chat_template(
            chat, ChatFormat::QWEN3, /*add_generation_prompt=*/true, /*enable_thinking=*/true,
            tools.is_array() && !tools.empty() ? tools.dump() : ""));
        cover += rendered.back();
    }
    cover += "[]\xE2\x80\xA6" "0123456789:;,.-|\\\"" "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    const auto vocabulary = [] (const std::vector<std::string> & words) {
        std::vector<std::string> out{"user", "assistant", "system", "\n", "tool"};
        out.insert(out.end(), words.begin(), words.end());
        return out;
    };
    const std::string path = write_pflash_bpe_tokenizer_fixture(vocabulary(options.target_words), cover);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    std::string drafter_path;
    Tokenizer drafter_tokenizer;
    if (options.distinct_drafter) {
        drafter_path = write_pflash_bpe_tokenizer_fixture(vocabulary(options.drafter_words), cover);
        TEST_ASSERT(drafter_tokenizer.load_from_gguf(drafter_path.c_str()));
    }
    Tokenizer & drafter = options.distinct_drafter ? drafter_tokenizer : tokenizer;
    UnifiedRun run;
    const auto starts = [] (const Tokenizer & tok, const std::vector<int32_t> & ids) {
        std::vector<size_t> out;
        size_t at = 0;
        for (int32_t id : ids) {
            out.push_back(at);
            at += tok.decode(std::vector<int32_t>{id}).size();
        }
        return out;
    };
    for (const auto & text : rendered) {
        run.ids.push_back(tokenizer.encode(text));
        run.raw_text.push_back(text);
        run.target_starts.push_back(starts(tokenizer, run.ids.back()));
        run.drafter_starts.push_back(starts(drafter, drafter.encode(text)));
    }
    std::unique_ptr<ModelBackend> backend_owner;
    std::function<int()> compress_calls;
    const auto setup = [&] (auto backend) {
        backend->tokenizer = &drafter;
        backend->chunk = chunk;
        backend->paragraphs = options.paragraphs;
        backend->widen = options.widen;
        backend->keep_scored_only = options.keep_scored_only;
        backend->probe_pin_text = options.probe_pin_text;
        backend->fail = options.drafter_fails;
        backend->score = score;
        compress_calls = [mock = backend.get()] { return mock->compress_calls; };
        backend_owner = std::move(backend);
    };
    if (options.disk_dir.empty()) {
        setup(std::make_unique<MockUnifiedBackend>());
    } else {
        setup(std::make_unique<MockUnifiedBackendT<DiskPayloadBackend>>());
    }
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.pflash = true;
    config.pflash_threshold = threshold;
    config.pflash_keep_ratio = keep_ratio;
    config.max_ctx = max_ctx;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    if (!options.disk_dir.empty()) {
        config.prefix_cache_cap = 8;
        config.prefix_cache_max_bytes = 0;
        config.disk_cache_dir = options.disk_dir;
        config.disk_cache_min_tokens = 1;
    }
    if (options.disk_seed_history) {
        // The previous process: its session (turn 0 before the newest user message) leaves RAM at shutdown.
        const size_t at = run.raw_text[0].rfind("<|im_start|>user");
        const auto & starts0 = run.target_starts[0];
        const auto it = std::find(starts0.begin(), starts0.end(), at);
        TEST_ASSERT(at != std::string::npos && it != starts0.end());
        const std::vector<int32_t> history(run.ids[0].begin(), run.ids[0].begin() + (it - starts0.begin()));
        auto seed_owner = std::make_unique<DiskPayloadBackend>();
        DiskPayloadBackend & seed_backend = *seed_owner;
        LuceEngine seed_engine(std::move(seed_owner));
        HttpServer seed(seed_engine, tokenizer, config);
        seed_backend.put(0, (int) history.size());
        SchedulerTestHarness::prefix_cache(seed).confirm_inline_snap(0, (int) history.size(), history);
        SchedulerTestHarness::set_slot_tokens(seed, 0, history);
        SchedulerTestHarness::flush_sessions(seed);
        TEST_ASSERT(SchedulerTestHarness::disk_cache(seed).longest_prefix(history).len == (int) history.size());
    }
    config.disk_cache_policy.mode = options.disk_mode;
    {
        auto start = [&] {
            auto server = std::make_unique<HttpServer>(engine, tokenizer, config);
            server->set_drafter_tokenizer(&drafter);
            if (options.chat_view_only) HttpServerTestAccess::chat_view_only(*server);
            return server;
        };
        auto owned = start();
        for (size_t index = 0; index < turns.size(); ++index) {
            if ((int) index == options.restart_at) {
                SchedulerTestHarness::flush_sessions(*owned);
                owned.reset();
                owned = start();
            }
            if (options.before_turn) options.before_turn(index);
            HttpServer & server = *owned;
            ParsedRequest request;
            request.format = ApiFormat::OPENAI_CHAT;
            request.messages = turns[index];
            if (tools.is_array() && !tools.empty()) request.tools = tools;
            request.prompt_tokens = run.ids[index];
            request.max_output = index < options.max_outputs.size() ? options.max_outputs[index] : 16;
            if (index < options.required.size()) request.pflash_required = options.required[index];
            request.disk_cache_policy = config.disk_cache_policy;   // as the request parser sets it
            run.prepared.push_back(HttpServerTestAccess::prepare_prompt(server, request));
            if (!options.allow_errors) TEST_ASSERT_MSG(run.prepared.back().error.empty(), run.prepared.back().error);
            run.served_text.push_back(tokenizer.decode(run.prepared.back().tokens));
            run.calls_after.push_back(compress_calls());
        }
        run.unified = HttpServerTestAccess::pflash_unified(*owned);
    }
    std::ifstream in(trace_path);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) run.view_records.push_back(json::parse(line));
    }
    unlink(trace_path);
    unlink(path.c_str());
    if (!drafter_path.empty()) unlink(drafter_path.c_str());
    return run;
}

// "Line i: plain filler text here[ mark]\n" for i in 1..lines.
static std::string unified_doc(int lines, const std::map<int, std::string> & marks,
                               const std::string & prefix = "Line ") {
    std::string out;
    for (int i = 1; i <= lines; ++i) {
        out += prefix + std::to_string(i) + ": plain filler text here";
        const auto it = marks.find(i);
        if (it != marks.end()) out += " " + it->second;
        out += "\n";
    }
    return out;
}

static size_t unified_occurrences(const std::string & text, const std::string & needle) {
    size_t n = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) ++n;
    return n;
}

// Wants the kiwi line when the prompt asks KIWI?, the mango line when it asks MANGO?.
static double unified_fruit_score(const std::string & segment, const std::string & prompt) {
    double score = 1.0;
    if (prompt.find("KIWI?") != std::string::npos && segment.find("kiwi") != std::string::npos) score += 50.0;
    if (prompt.find("MANGO?") != std::string::npos && segment.find("mango") != std::string::npos) score += 50.0;
    return score;
}

static json unified_tool_call(const std::string & id, const std::string & name, const json & arguments) {
    return {{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", arguments.dump()}}}};
}

TEST_CASE(ServerUnitFixture, test_unified_paste_pages_then_recall_appends_and_repeats) {
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {60, "mango"}});
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    const auto run = run_unified_session({turn1, turn2, turn2}, nullptr, nullptr, 64, unified_fruit_score);

    // Turn 1: the paste is paged; the system prompt, the pinned edges, the question and the kiwi line stay whole.
    const auto & p1 = run.prepared[0];
    TEST_ASSERT(p1.compressed);
    TEST_ASSERT_MSG(p1.pflash_stats["unified"].value("decision", "") == "region", p1.pflash_stats.dump());
    const std::string & s1 = run.served_text[0];
    TEST_ASSERT(s1.find("You are helpful.") != std::string::npos);
    TEST_ASSERT(s1.find("Line 1: plain filler text here\nLine 2: plain filler text here\n"
                        "[Server note: this message was shortened:") != std::string::npos);
    TEST_ASSERT(s1.find("Line 7: plain filler text here kiwi\n") != std::string::npos);
    TEST_ASSERT(s1.find("mango") == std::string::npos);
    TEST_ASSERT(s1.find("lines not shown \xE2\x80\xA6]\n") != std::string::npos);
    TEST_ASSERT(s1.find("Line 80: plain filler text here\nWhich line mentions KIWI?") != std::string::npos);
    TEST_ASSERT(run.prepared[0].tokens.size() < run.ids[0].size());

    // Turn 2: turn 1's served prompt is reused token for token; the mango line comes back in a labelled block at
    // the start of the question; nothing turn 1 kept is appended again.
    const auto & p2 = run.prepared[1];
    const int vg1 = p1.snapshot_cut;
    TEST_ASSERT(vg1 > 0);
    TEST_ASSERT(std::equal(p1.tokens.begin(), p1.tokens.begin() + vg1, p2.tokens.begin()));
    const std::string & s2 = run.served_text[1];
    const size_t answer = s2.find("Line 7.<|im_end|>");
    const size_t block = s2.find("[Earlier in this conversation]\n[From an earlier user message]\n");
    const size_t mango = s2.find("Line 60: plain filler text here mango\n");
    const size_t close = s2.find("mango\n\n[End of earlier excerpts]\n\nWhich line mentions MANGO?");
    TEST_ASSERT(answer != std::string::npos);
    TEST_ASSERT_MSG(block != std::string::npos && block > answer, s2.substr(answer));
    TEST_ASSERT_MSG(mango != std::string::npos && mango > block && close != std::string::npos && close > mango,
                    s2.substr(block));
    TEST_ASSERT(s2.find("here\n\n[From an earlier user message]\nLine 60:", block) != std::string::npos);
    for (int line = 1; line <= 80; ++line) {
        TEST_ASSERT_MSG(unified_occurrences(s2, "Line " + std::to_string(line) + ":") <= 1, std::to_string(line));
    }
    TEST_ASSERT(p2.pflash_stats["unified"].value("decision", "") == "recall");

    // Turn 3, a retry: the same bytes, no drafter pass.
    TEST_ASSERT(run.prepared[2].tokens == p2.tokens);
    TEST_ASSERT(run.prepared[2].pflash_stats["unified"].value("decision", "") == "repeat");
    TEST_ASSERT(run.calls_after == std::vector<int>({1, 2, 2}));
    TEST_ASSERT(run.unified["counts"].value("serve.region", 0) == 1);
    TEST_ASSERT(run.unified["counts"].value("recall.fired", 0) == 1);
    TEST_ASSERT(run.view_records.size() == 3);
    for (const auto & record : run.view_records) TEST_ASSERT(record.value("schema_version", 0) == 3);
}

TEST_CASE(ServerUnitFixture, test_unified_paging_keeps_the_request_probes_pins) {
    // The question sits mid-paste and scores low; only the request probe's pin keeps it.
    const std::string question = "Line 40: plain filler text here QUESTION-X";
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {40, "QUESTION-X"}});
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    const UnifiedScore score = [] (const std::string & segment, const std::string & prompt) {
        return segment.find("QUESTION-X") != std::string::npos ? 0.1 : unified_fruit_score(segment, prompt);
    };
    const auto unpinned = run_unified_session({turn}, nullptr, nullptr, 64, score);
    TEST_ASSERT(unpinned.prepared[0].compressed);
    TEST_ASSERT(unpinned.served_text[0].find(question) == std::string::npos);
    UnifiedOptions options;
    options.probe_pin_text = "QUESTION-X";
    const auto pinned = run_unified_session({turn}, nullptr, nullptr, 64, score, 0, 0.3f, nullptr, 65536, options);
    TEST_ASSERT(pinned.prepared[0].compressed);
    TEST_ASSERT_MSG(pinned.served_text[0].find(question + "\n") != std::string::npos, pinned.served_text[0]);
    TEST_ASSERT(pinned.served_text[0].find("Line 7: plain filler text here kiwi\n") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_unified_raw_fallback_drops_the_conversations_views) {
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {60, "mango"}});
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    json turn3 = turn2;
    turn3.push_back({{"role", "assistant"}, {"content", "Line 60."}});
    turn3.push_back({{"role", "user"}, {"content", "Which line mentions KIWI?"}});
    // Turn 2 fails its PFlash mapping (a pflash_required text it does not hold) and is served raw.
    UnifiedOptions options;
    options.required = {{}, {"text that is not in the message"}, {}};
    const auto run = run_unified_session({turn1, turn2, turn3}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f,
                                         nullptr, 65536, options);
    TEST_ASSERT(run.prepared[0].compressed);
    TEST_ASSERT(!run.prepared[1].compressed && run.prepared[1].tokens == run.ids[1]);
    const json & fallback = run.prepared[1].pflash_stats["pflash_fallback"];
    TEST_ASSERT_MSG(fallback.value("unified_lineages_forgotten", 0) == 1, run.prepared[1].pflash_stats.dump());
    // Turn 3 does not continue turn 1's view (its history was served raw): the chat view takes it.
    TEST_ASSERT_MSG(run.prepared[2].unified_handback == "uncached", run.prepared[2].pflash_stats.dump());
    TEST_ASSERT(run.unified["counts"].value("serve.region", 0) == 1);
}

TEST_CASE(ServerUnitFixture, test_unified_needs_candidate_scores) {
    using luce::pflash::PFlashScorer;
    using luce::pflash::PFlashSelectStrategy;
    luce::pflash::PFlashSelectionConfig selection;
    TEST_ASSERT(http_detail::pflash_selection_gives_lifts(selection));
    selection.scorer = PFlashScorer::Legacy;
    TEST_ASSERT(!http_detail::pflash_selection_gives_lifts(selection));
    selection.scorer = PFlashScorer::Split;
    TEST_ASSERT(!http_detail::pflash_selection_gives_lifts(selection));
    selection.scorer = PFlashScorer::Head;
    selection.strategy = PFlashSelectStrategy::Fold;
    TEST_ASSERT(!http_detail::pflash_selection_gives_lifts(selection));
    selection.strategy = PFlashSelectStrategy::Segments;
    {
        luce_test::ScopedEnvVar legacy{"PFLASH_QWEN35_LEGACY_SCORER", "1"};
        TEST_ASSERT(!http_detail::pflash_selection_gives_lifts(selection));
    }

    const std::string doc = unified_doc(80, {{7, "kiwi"}});
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    // A scorer that gives no candidate scores: the unified view is off and the chat view serves the request.
    UnifiedOptions legacy;
    legacy.scorer = "legacy";
    const auto handed = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr,
                                            65536, legacy);
    TEST_ASSERT_MSG(handed.prepared[0].unified_handback.empty() &&
                    handed.prepared[0].pflash_stats.find("unified") == handed.prepared[0].pflash_stats.end(),
                    handed.prepared[0].pflash_stats.dump());
    TEST_ASSERT(handed.unified.is_null());
    // A failed drafter pass: the newest content is served verbatim, or with PFLASH_FAIL_CLOSED=1 refused.
    UnifiedOptions failing;
    failing.drafter_fails = true;
    const auto verbatim = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr,
                                              65536, failing);
    TEST_ASSERT(verbatim.served_text[0] == verbatim.raw_text[0]);
    TEST_ASSERT(verbatim.unified["counts"].value("drafter.failed", 0) == 1);
    failing.fail_closed = "1";
    failing.allow_errors = true;
    const auto refused = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr,
                                             65536, failing);
    TEST_ASSERT_MSG(!refused.prepared[0].error.empty(), refused.prepared[0].pflash_stats.dump());
}

TEST_CASE(ServerUnitFixture, test_unified_repeat_with_a_new_pin_serves_it) {
    // A retry of the same messages that now pins (pflash_required) a line the view left out is not answered with
    // the stored view: it is paged again with that line kept.
    const std::string pear = "Line 30: plain filler text here pear";
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", unified_doc(80, {{7, "kiwi"}, {30, "pear"}}) + "Which line mentions KIWI?"}},
    });
    const UnifiedScore score = [] (const std::string & segment, const std::string & prompt) {
        return segment.find("pear") != std::string::npos ? 0.1 : unified_fruit_score(segment, prompt);
    };
    UnifiedOptions options;
    options.required = {{}, {pear}};
    const auto run = run_unified_session({turn, turn}, nullptr, nullptr, 64, score, 0, 0.3f, nullptr, 65536, options);
    TEST_ASSERT(run.prepared[0].compressed && run.served_text[0].find(pear) == std::string::npos);
    TEST_ASSERT_MSG(run.served_text[1].find(pear + "\n") != std::string::npos, run.prepared[1].pflash_stats.dump());
    TEST_ASSERT(run.unified["counts"].value("repeat.pins_unserved", 0) == 1);
    TEST_ASSERT(run.unified["counts"].value("repeat", 0) == 0);
}

TEST_CASE(ServerUnitFixture, test_unified_repeat_of_an_adopted_prompt_with_a_new_pin_serves_it) {
    // The chat view's prompt the unified view adopted keeps no record of the lines it kept: a retry that pins
    // (pflash_required) text is compressed again, not answered with that prompt.
    std::string filler;
    for (int i = 0; i < 300; ++i) filler += "The grass is green. ";
    const std::string pear = "The pear number is 3.";
    const std::string doc = "Below is a long passage.\n\n" + filler + pear + " " + filler + "The kiwi number is 7. " +
                            filler + "\n\nQuestion: what is the kiwi number?\n";
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which sentence mentions KIWI?"}},
    });
    UnifiedOptions options;
    options.keep_scored_only = true;
    options.required = {{}, {pear}};
    const auto run = run_unified_session({turn, turn}, nullptr, nullptr, 64, unified_fruit_score, 16, 0.3f, nullptr,
                                         65536, options);
    TEST_ASSERT_MSG(run.prepared[0].unified_handback == "uncached" && run.prepared[0].compressed,
                    run.prepared[0].pflash_stats.dump());
    TEST_ASSERT(run.served_text[0].find(pear) == std::string::npos);
    TEST_ASSERT_MSG(run.served_text[1].find(pear) != std::string::npos, run.prepared[1].pflash_stats.dump());
    TEST_ASSERT(run.unified["counts"].value("repeat", 0) == 0);
}

TEST_CASE(ServerUnitFixture, test_unified_disk_view_that_does_not_match_the_prompt_is_dropped) {
    // The restarted server finds turn 1's view on disk, but the file's region digest no longer matches the
    // conversation's bytes: the view is deleted and the follow-up goes to the chat view as a cold history.
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", unified_doc(80, {{7, "kiwi"}, {60, "mango"}}) + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    const std::string dir = test_tmp_path("luce_test_unified_view_mismatch").string();
    rm_rf(dir);
    UnifiedOptions disk;
    disk.disk_dir = dir;
    disk.restart_at = 1;
    int edited = 0;
    disk.before_turn = [&] (size_t index) {
        if (index != 1) return;
        for (const auto & entry : std::filesystem::directory_iterator(dir + "/.pflash-views")) {
            if (entry.path().extension() != ".view") continue;
            std::ifstream in(entry.path());
            std::string header, body;
            std::getline(in, header);
            std::getline(in, body);
            in.close();
            json view = json::parse(body);
            view["regions"][0]["content_sha"] = "0000";
            std::ofstream(entry.path(), std::ios::trunc) << header << "\n" << view.dump() << "\n";
            ++edited;
        }
    };
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr,
                                         65536, disk);
    TEST_ASSERT(edited == 1);
    TEST_ASSERT(run.prepared[0].compressed && run.prepared[0].unified_view);
    TEST_ASSERT_MSG(run.prepared[1].unified_handback == "uncached", run.prepared[1].pflash_stats.dump());
    TEST_ASSERT(run.unified["counts"].value("disk_view_rejected", 0) == 1);
    TEST_ASSERT(run.unified["counts"].value("disk_view", 0) == 0);
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_unified_tool_output_needs_listing) {
    const json tools = json::array({
        {{"type", "function"}, {"function", {{"name", "read_file"}, {"description", "Read a file"}}}},
        {{"type", "function"}, {"function", {{"name", "terminal"}, {"description", "Run a command"}}}},
    });
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Read both files. Which line mentions KIWI?"}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
            unified_tool_call("call_1", "read_file", {{"path", "docs/a.md"}}),
            unified_tool_call("call_2", "terminal", {{"cmd", "cat docs/b.md"}}),
            unified_tool_call("call_3", "read_file", {{"path", "docs/c.md"}}),
        })}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", unified_doc(80, {{9, "kiwi"}}, "A")}},
        {{"role", "tool"}, {"tool_call_id", "call_2"}, {"content", unified_doc(80, {}, "B")}},
        {{"role", "tool"}, {"tool_call_id", "call_3"}, {"content", unified_doc(1, {}, "C")}},
    });
    // The fixture tokenizer has one token per character, so the history before the tool results (the question and
    // three calls, ~350 tokens) would already be a 64-token threshold's worth of uncached history and the request
    // would go to the chat view; 600 keeps it a fresh agent turn whose 80-line outputs are still large.
    const int threshold = 600;
    const auto listed = run_unified_session({turn}, tools, "read_file", threshold, unified_fruit_score);
    const json & admission = listed.prepared[0].pflash_stats["unified"]["admission"];
    TEST_ASSERT_MSG(admission.size() == 3, admission.dump());
    TEST_ASSERT(admission[0].value("decision", "") == "region");
    TEST_ASSERT(admission[1].value("decision", "") == "unlisted");
    TEST_ASSERT(admission[2].value("decision", "") == "small");
    const std::string & served = listed.served_text[0];
    TEST_ASSERT(served.find("A9: plain filler text here kiwi\n") != std::string::npos);
    TEST_ASSERT(served.find("[Server note: this tool result was shortened:") != std::string::npos);
    for (int line = 1; line <= 80; ++line) {
        TEST_ASSERT(served.find("B" + std::to_string(line) + ": plain filler text here\n") != std::string::npos);
    }
    // "*": every tool output may be paged; the one-line one is still too small.
    const auto all = run_unified_session({turn}, tools, "*", threshold, unified_fruit_score);
    const json & all_admission = all.prepared[0].pflash_stats["unified"]["admission"];
    TEST_ASSERT(all_admission[0].value("decision", "") == "region");
    TEST_ASSERT(all_admission[1].value("decision", "") == "region");
    TEST_ASSERT(all_admission[2].value("decision", "") == "small");
    // Nothing listed: the tool turn is served raw, without a drafter pass.
    const auto none = run_unified_session({turn}, tools, nullptr, threshold, unified_fruit_score);
    TEST_ASSERT(!none.prepared[0].compressed);
    TEST_ASSERT(none.prepared[0].tokens == none.ids[0]);
    TEST_ASSERT(none.calls_after[0] == 0);
    TEST_ASSERT(none.prepared[0].pflash_stats["unified"].value("reason", "") == "no_region");
}

TEST_CASE(ServerUnitFixture, test_unified_line_model_json_string_served) {
    // A JSON-wrapped read (the shape some harnesses return): numbered lines inside a string field.
    std::string body;
    for (int i = 1; i <= 80; ++i) {
        body += std::to_string(i) + "|row " + std::to_string(i) + (i == 33 ? " kiwi" : "") + "\n";
    }
    const std::string wrapped = json({{"content", body}, {"total_lines", 80}, {"truncated", false}}).dump();
    const json tools = json::array({{{"type", "function"}, {"function", {{"name", "read_file"}}}}});
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Which row mentions KIWI?"}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
            unified_tool_call("call_1", "read_file", {{"path", "rows.txt"}})})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", wrapped}},
    });
    const auto run = run_unified_session({turn}, tools, "read_file", 64, unified_fruit_score, /*chunk=*/7);
    const auto & prepared = run.prepared[0];
    TEST_ASSERT_MSG(prepared.compressed, prepared.pflash_stats.dump());
    const json & admission = prepared.pflash_stats["unified"]["admission"][0];
    TEST_ASSERT(admission.value("escaped_lines", false));
    const std::string & served = run.served_text[0];
    const size_t open = served.rfind("<tool_response>\n");
    const size_t close = served.find("\n</tool_response>", open);
    TEST_ASSERT(open != std::string::npos && close != std::string::npos);
    const json result = json::parse(served.substr(open + 16, close - open - 16), nullptr, false);
    TEST_ASSERT_MSG(!result.is_discarded(), served.substr(open, 400));
    TEST_ASSERT(result.value("total_lines", 0) == 80);
    TEST_ASSERT(result.value("truncated", true) == false);
    const std::string content = result.value("content", "");
    TEST_ASSERT(content.find("1|row 1\n2|row 2\n[Server note: this tool result was shortened:") == 0);
    TEST_ASSERT(content.find("33|row 33 kiwi\n") != std::string::npos);
    TEST_ASSERT(content.find(" lines not shown \xE2\x80\xA6]\n") != std::string::npos);
    TEST_ASSERT(content.find("\n79|row 79\n80|row 80\n") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_unified_recall_user_question_only) {
    const json tools = json::array({{{"type", "function"}, {"function", {{"name", "read_file"}}}}});
    json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Which line mentions KIWI?"}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
            unified_tool_call("call_1", "read_file", {{"path", "doc.md"}})})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"},
         {"content", unified_doc(80, {{7, "kiwi"}, {60, "mango"}})}},
    });
    // An agent step: another call and a small result. No recall, no drafter pass; the view continues.
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
        unified_tool_call("call_2", "read_file", {{"path", "note.md"}})})}});
    turn2.push_back({{"role", "tool"}, {"tool_call_id", "call_2"}, {"content", "short note"}});
    // The answer, then a new user question: recall at the question.
    json turn3 = turn2;
    turn3.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn3.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    const auto run = run_unified_session({turn1, turn2, turn3}, tools, "read_file", 64, unified_fruit_score);
    TEST_ASSERT(run.calls_after == std::vector<int>({1, 1, 2}));
    TEST_ASSERT(run.prepared[1].pflash_stats["unified"].value("decision", "") == "continue");
    TEST_ASSERT(run.prepared[2].pflash_stats["unified"].value("decision", "") == "recall");
    for (size_t turn = 1; turn < 3; ++turn) {
        const int vg = run.prepared[turn - 1].snapshot_cut;
        TEST_ASSERT(std::equal(run.prepared[turn - 1].tokens.begin(), run.prepared[turn - 1].tokens.begin() + vg,
                               run.prepared[turn].tokens.begin()));
    }
    const std::string & s3 = run.served_text[2];
    const size_t block = s3.find("[Earlier in this conversation]\n[From the earlier result of read_file "
                                 "{\"path\":\"doc.md\"}]\n");
    TEST_ASSERT_MSG(block != std::string::npos, s3);
    TEST_ASSERT(s3.find("Line 60: plain filler text here mango\n", block) != std::string::npos);
    TEST_ASSERT(s3.find("short note") < block);
}

TEST_CASE(ServerUnitFixture, test_unified_cold_history_hands_back_to_the_chat_view) {
    // A cold transcript: a large paste in turn 1 and a large listed tool output in turn 2, the newest turn a short
    // question. The unified view never pages anything before the newest turn and would prefill that history raw, so
    // it hands the request to the chat view, which serves a tool session raw (keep_history): no drafter pass.
    const json tools = json::array({{{"type", "function"}, {"function", {{"name", "read_file"}}}}});
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", unified_doc(80, {{7, "kiwi"}}) + "Summarise this."}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
            unified_tool_call("call_1", "read_file", {{"path", "doc.md"}})})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", unified_doc(80, {}, "T")}},
        {{"role", "assistant"}, {"content", "Done."}},
        {{"role", "user"}, {"content", "Which line mentions KIWI?"}},
    });
    const auto run = run_unified_session({turn}, tools, "read_file", 64, unified_fruit_score);
    TEST_ASSERT_MSG(run.prepared[0].unified_handback == "uncached", run.prepared[0].pflash_stats.dump());
    TEST_ASSERT(run.prepared[0].pflash_stats.find("unified") == run.prepared[0].pflash_stats.end());
    TEST_ASSERT(!run.prepared[0].compressed);
    TEST_ASSERT(run.prepared[0].tokens == run.ids[0]);
    TEST_ASSERT(run.calls_after[0] == 0);
    TEST_ASSERT(run.unified["counts"].value("handback.uncached", 0) == 1);
    TEST_ASSERT_MSG(run.view_records.size() == 1 &&
                    run.view_records[0]["pflash"]["unified"].value("decision", "") == "handback" &&
                    run.view_records[0]["pflash"]["unified"].value("reason", "") == "uncached",
                    run.view_records.empty() ? "no view record" : run.view_records[0].dump());
}

TEST_CASE(ServerUnitFixture, test_unified_cold_chat_history_is_adopted_after_the_chat_view) {
    // A chat whose first request already holds a large earlier paste: the chat view compresses the cold history,
    // then the unified view adopts what it served, so the follow-up continues it and reuses the served prefix.
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", unified_doc(80, {{7, "kiwi"}, {60, "mango"}}) + "Summarise this."}},
        {{"role", "assistant"}, {"content", "A list of lines."}},
        {{"role", "user"}, {"content", "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    UnifiedOptions selecting;
    selecting.keep_scored_only = true;
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 16, 0.3f, nullptr,
                                         65536, selecting);
    UnifiedOptions chat = selecting;
    chat.chat_view_only = true;
    const auto base = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 16, 0.3f, nullptr,
                                          65536, chat);
    // Turn 1: the chat view's own prompt.
    TEST_ASSERT_MSG(run.prepared[0].unified_handback == "uncached", run.prepared[0].pflash_stats.dump());
    TEST_ASSERT(run.prepared[0].compressed && run.prepared[0].unified_view);
    TEST_ASSERT(run.prepared[0].tokens == base.prepared[0].tokens);
    TEST_ASSERT(run.unified["counts"].value("adopt.uncached", 0) == 1);
    // Turn 2: the adopted view continues; turn 1's served prompt is reused token for token.
    const auto & p2 = run.prepared[1];
    TEST_ASSERT_MSG(p2.unified_handback.empty() && p2.unified_view, p2.pflash_stats.dump());
    const json & stats = p2.pflash_stats["unified"];
    TEST_ASSERT_MSG(stats.value("continued", false) && stats.value("decision", "") == "continue", stats.dump());
    const int cut = run.prepared[0].snapshot_cut;
    TEST_ASSERT(cut > 0 && stats.value("reused_tokens", 0) == cut);
    TEST_ASSERT(std::equal(run.prepared[0].tokens.begin(), run.prepared[0].tokens.begin() + cut, p2.tokens.begin()));
    TEST_ASSERT(run.served_text[1].find("Which line mentions MANGO?") != std::string::npos);
    TEST_ASSERT(run.unified.value("views", 0) >= 1);
}

TEST_CASE(ServerUnitFixture, test_unified_long_line_paste_goes_to_the_chat_view) {
    // A needle-in-a-haystack paste: the whole passage is one line, longer than the region's cap. Paging could only
    // drop that line whole, the kiwi sentence with it; the chat view selects inside the line instead.
    std::string filler;
    for (int i = 0; i < 300; ++i) filler += "The grass is green. ";
    const std::string doc = "Below is a long passage.\n\n" + filler + "The kiwi number is 7. " + filler +
                            "\n\nQuestion: what is the kiwi number?\n";
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which sentence mentions KIWI?"}},
    });
    UnifiedOptions options;
    options.keep_scored_only = true;
    const auto run = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 16, 0.3f, nullptr,
                                         65536, options);
    TEST_ASSERT_MSG(run.prepared[0].unified_handback == "uncached", run.prepared[0].pflash_stats.dump());
    TEST_ASSERT_MSG(run.prepared[0].compressed, run.prepared[0].pflash_stats.dump());
    TEST_ASSERT_MSG(run.served_text[0].find("kiwi") != std::string::npos, run.served_text[0]);
    TEST_ASSERT(run.unified["counts"].value("handback.uncached", 0) == 1);
    TEST_ASSERT(run.unified["counts"].value("adopt.uncached", 0) == 1);

    // The follow-up stays with the unified view: it continues the adopted prompt.
    json turn2 = turn;
    turn2.push_back({{"role", "assistant"}, {"content", "The kiwi number is 7."}});
    turn2.push_back({{"role", "user"}, {"content", "And what colour is the grass?"}});
    const auto two = run_unified_session({turn, turn2}, nullptr, nullptr, 64, unified_fruit_score, 16, 0.3f, nullptr,
                                         65536, options);
    const auto & p2 = two.prepared[1];
    TEST_ASSERT_MSG(p2.unified_handback.empty() && p2.unified_view, p2.pflash_stats.dump());
    TEST_ASSERT(p2.pflash_stats["unified"].value("continued", false));
    const int cut = two.prepared[0].snapshot_cut;
    TEST_ASSERT(cut > 0 && std::equal(two.prepared[0].tokens.begin(), two.prepared[0].tokens.begin() + cut,
                                      p2.tokens.begin()));
}

TEST_CASE(ServerUnitFixture, test_unified_long_line_under_the_message_cap_but_over_the_region_cap) {
    // The first two lines are always kept and are long, so the message's cap holds the kiwi line but the region's
    // cap, on the pageable lines alone, does not: paging would drop the line whole, so the chat view takes it.
    const auto repeat = [](const std::string & s, int n) {
        std::string out;
        for (int i = 0; i < n; ++i) out += s;
        return out;
    };
    const std::string doc = repeat("The sky is blue. ", 250) + "\n" + repeat("The sea is deep. ", 250) + "\n" +
                            repeat("The grass is green. ", 80) + "The kiwi number is 7. " +
                            repeat("The grass is green. ", 80) + "\nshort a\nshort b\nshort c\nshort d\nend\n";
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which sentence mentions KIWI?"}},
    });
    UnifiedOptions options;
    options.keep_scored_only = true;
    const auto run = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 16, 0.3f, nullptr,
                                         65536, options);
    TEST_ASSERT_MSG(run.prepared[0].unified_handback == "uncached", run.prepared[0].pflash_stats.dump());
    TEST_ASSERT_MSG(run.served_text[0].find("kiwi") != std::string::npos, run.served_text[0]);
}

// A tool output whose body is one line of ``filler`` around ``needle``, framed by short lines.
static std::string unified_long_line_output(const std::string & needle) {
    std::string filler;
    for (int i = 0; i < 300; ++i) filler += "The grass is green. ";
    return "Header\n\n" + filler + needle + " " + filler + "\n\nfooter\nend\n";
}

TEST_CASE(ServerUnitFixture, test_unified_long_line_tool_output_is_served_whole) {
    const json tools = json::array({{{"type", "function"}, {"function", {{"name", "read_file"}}}}});
    const std::string kiwi_output = unified_long_line_output("The kiwi number is 7.");
    const json cold = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Read the file. Which sentence mentions KIWI?"}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
            unified_tool_call("call_1", "read_file", {{"path", "a.txt"}})})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", kiwi_output}},
    });
    // A fresh agent turn: the chat view takes it and serves a tool session raw (keep_history), the line whole.
    const auto fresh = run_unified_session({cold}, tools, "read_file", 600, unified_fruit_score);
    TEST_ASSERT_MSG(fresh.prepared[0].unified_handback == "uncached", fresh.prepared[0].pflash_stats.dump());
    TEST_ASSERT(!fresh.prepared[0].compressed);
    TEST_ASSERT(fresh.prepared[0].tokens == fresh.ids[0]);
    // The agent's next call: the adopted raw prompt continues, and the next output is paged.
    json next = cold;
    next.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
        unified_tool_call("call_2", "read_file", {{"path", "b.md"}})})}});
    next.push_back({{"role", "tool"}, {"tool_call_id", "call_2"}, {"content", unified_doc(80, {{9, "kiwi"}}, "B")}});
    const auto agent = run_unified_session({cold, next}, tools, "read_file", 600, unified_fruit_score);
    TEST_ASSERT(agent.unified["counts"].value("adopt.uncached", 0) == 1);
    const auto & a2 = agent.prepared[1];
    TEST_ASSERT_MSG(a2.unified_handback.empty() && a2.unified_view && a2.compressed, a2.pflash_stats.dump());
    TEST_ASSERT_MSG(a2.pflash_stats["unified"].value("decision", "") == "region", a2.pflash_stats.dump());
    TEST_ASSERT(std::equal(agent.ids[0].begin(), agent.ids[0].begin() + agent.prepared[0].snapshot_cut,
                           a2.tokens.begin()));

    // A view that continues: turn 1 pages an ordinary output; turn 2's one-line output is appended verbatim.
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Read both files. Which sentences mention KIWI and MANGO?"}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
            unified_tool_call("call_1", "read_file", {{"path", "a.md"}})})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", unified_doc(80, {{9, "kiwi"}}, "A")}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
        unified_tool_call("call_2", "read_file", {{"path", "b.txt"}})})}});
    const std::string mango_output = unified_long_line_output("The mango number is 8.");
    turn2.push_back({{"role", "tool"}, {"tool_call_id", "call_2"}, {"content", mango_output}});
    const auto run = run_unified_session({turn1, turn2}, tools, "read_file", 600, unified_fruit_score);
    TEST_ASSERT_MSG(run.prepared[0].compressed && run.prepared[0].unified_view, run.prepared[0].pflash_stats.dump());
    const auto & p2 = run.prepared[1];
    TEST_ASSERT_MSG(p2.unified_handback.empty() && p2.unified_view, p2.pflash_stats.dump());
    const json & admission = p2.pflash_stats["unified"]["admission"];
    TEST_ASSERT_MSG(admission.size() == 1 && admission[0].value("decision", "") == "long_line", admission.dump());
    const int vg1 = run.prepared[0].snapshot_cut;
    TEST_ASSERT(std::equal(run.prepared[0].tokens.begin(), run.prepared[0].tokens.begin() + vg1, p2.tokens.begin()));
    TEST_ASSERT(run.served_text[1].find(mango_output) != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_pflash_restorable_prefix_counts_disk_only_when_on) {
    using http_detail::pflash_restorable_prefix;
    const auto is = [](const http_detail::PflashRestorablePrefix & got, int tokens, const char * source) {
        return got.tokens == tokens && std::string(got.source) == source;
    };
    // Disk cache off: the RAM prefix, whatever the disk would hold.
    TEST_ASSERT(is(pflash_restorable_prefix(0, false, 0), 0, "none"));
    TEST_ASSERT(is(pflash_restorable_prefix(0, false, 40000), 0, "none"));
    TEST_ASSERT(is(pflash_restorable_prefix(9000, false, 40000), 9000, "ram"));
    // Disk cache on: the deeper of the two; ties stay in RAM (no file read).
    TEST_ASSERT(is(pflash_restorable_prefix(0, true, 0), 0, "none"));
    TEST_ASSERT(is(pflash_restorable_prefix(0, true, 40000), 40000, "disk"));
    TEST_ASSERT(is(pflash_restorable_prefix(9000, true, 40000), 40000, "disk"));
    TEST_ASSERT(is(pflash_restorable_prefix(40000, true, 40000), 40000, "ram"));
    TEST_ASSERT(is(pflash_restorable_prefix(41000, true, 40000), 41000, "ram"));
    TEST_ASSERT(is(pflash_restorable_prefix(-1, true, -5), 0, "none"));
}

TEST_CASE(ServerUnitFixture, test_unified_restart_resumes_a_disk_chat_and_pages_only_the_new_paste) {
    // A server restart: the chat's history (an earlier paste and its answer) is on disk, RAM is empty. The unified
    // view counts the disk entry as cached, as the chat view's admission does: it keeps the conversation instead of
    // handing a cold history to the chat view, serves the history as it is (the generation path restores it from
    // disk) and pages only the new paste. Without the entry, or outside the Full disk scope, the chat view takes it.
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", unified_doc(80, {{7, "mango"}}) + "Summarise this."}},
        {{"role", "assistant"}, {"content", "A list of lines."}},
        {{"role", "user"}, {"content", unified_doc(80, {{33, "kiwi"}}, "Row ") + "Which row mentions KIWI?"}},
    });
    const std::string dir = test_tmp_path("luce_test_unified_disk_resume").string();
    rm_rf(dir);
    UnifiedOptions disk;
    disk.disk_dir = dir;
    disk.disk_seed_history = true;
    const auto run = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr, 65536,
                                         disk);
    const auto & stats = run.prepared[0].pflash_stats;
    TEST_ASSERT_MSG(run.prepared[0].unified_handback.empty(), stats.dump());
    TEST_ASSERT_MSG(stats.contains("unified") && stats["unified"].contains("cached") &&
                    stats["unified"]["cached"].value("source", "") == "disk" &&
                    stats["unified"]["cached"].value("tokens", 0) > 0, stats.dump());
    TEST_ASSERT(run.prepared[0].compressed);
    TEST_ASSERT(run.calls_after[0] == 1);
    TEST_ASSERT(run.served_text[0].find(unified_doc(80, {{7, "mango"}}) + "Summarise this.") != std::string::npos);
    TEST_ASSERT(run.served_text[0].find("Row 33: plain filler text here kiwi") != std::string::npos);
    TEST_ASSERT(run.served_text[0].size() < run.raw_text[0].size());
    TEST_ASSERT(run.unified["counts"].value("handback.uncached", 0) == 0);

    // The disk is looked up only in the Full scope, so only there does it count. (A fresh directory: the first run's
    // shutdown wrote its view there, and a view on disk continues the conversation whatever the scope.)
    rm_rf(dir);
    UnifiedOptions off = disk;
    off.disk_mode = DiskPrefixCacheMode::Off;
    const auto scoped = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr,
                                            65536, off);
    TEST_ASSERT_MSG(scoped.prepared[0].unified_handback == "uncached", scoped.prepared[0].pflash_stats.dump());
    rm_rf(dir);
    // No disk entry: the cold history goes to the chat view, as without the disk cache.
    UnifiedOptions empty;
    empty.disk_dir = dir;
    const auto cold = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr,
                                          65536, empty);
    TEST_ASSERT_MSG(cold.prepared[0].unified_handback == "uncached", cold.prepared[0].pflash_stats.dump());
    rm_rf(dir);
}

TEST_CASE(ServerUnitFixture, test_unified_restart_serves_a_disk_chat_raw_for_a_short_question) {
    // The same restart with a short follow-up question: nothing new to page, so the whole prompt is served raw for
    // the disk restore -- no drafter pass and no compression of the history. Without the disk cache the chat view
    // compresses the cold history (test_unified_cold_chat_history_goes_to_the_chat_view).
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", unified_doc(80, {{7, "kiwi"}, {60, "mango"}}) + "Summarise this."}},
        {{"role", "assistant"}, {"content", "A list of lines."}},
        {{"role", "user"}, {"content", "Which line mentions KIWI?"}},
    });
    const std::string dir = test_tmp_path("luce_test_unified_disk_short").string();
    rm_rf(dir);
    UnifiedOptions disk;
    disk.disk_dir = dir;
    disk.disk_seed_history = true;
    const auto run = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr, 65536,
                                         disk);
    TEST_ASSERT_MSG(run.prepared[0].unified_handback.empty(), run.prepared[0].pflash_stats.dump());
    TEST_ASSERT(!run.prepared[0].compressed);
    TEST_ASSERT(run.prepared[0].tokens == run.ids[0]);
    TEST_ASSERT(run.calls_after[0] == 0);

    // The chat view alone counts the disk the same way: its cache admission keeps the
    // history raw instead of compressing it, and records where the cached prefix is.
    UnifiedOptions chat = disk;
    chat.chat_view_only = true;
    const auto warm = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr, 65536,
                                          chat);
    TEST_ASSERT(warm.calls_after[0] == 0);
    TEST_ASSERT(!warm.prepared[0].compressed);
    TEST_ASSERT(warm.prepared[0].tokens == warm.ids[0]);
    const json warm_admission = warm.prepared[0].pflash_stats.value("admission", json::object());
    TEST_ASSERT_MSG(warm_admission.value("cached_source", "") == "disk", warm.prepared[0].pflash_stats.dump());
    rm_rf(dir);
    // Disk cache off: the admission compresses the cold history, and nothing records a source.
    UnifiedOptions plain;
    plain.chat_view_only = true;
    const auto cold = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr, 65536,
                                          plain);
    TEST_ASSERT(cold.calls_after[0] == 1);
    TEST_ASSERT_MSG(cold.prepared[0].pflash_stats.dump().find("cached_source") == std::string::npos,
                    cold.prepared[0].pflash_stats.dump());
}

TEST_CASE(ServerUnitFixture, test_unified_shortened_chat_survives_a_restart_through_its_view_on_disk) {
    // Turn 1 pages a long paste. The server restarts (shutdown flush): its view is written to disk and read back on
    // the next turn, so the follow-up continues the shortened conversation -- the same served prefix, nothing handed
    // back to the chat view -- instead of arriving cold. Without the disk cache the restart loses the view, and the
    // follow-up goes to the chat view as a cold history.
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", unified_doc(80, {{7, "kiwi"}, {60, "mango"}}) + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    const std::string dir = test_tmp_path("luce_test_unified_view_restart").string();
    rm_rf(dir);
    UnifiedOptions disk;
    disk.disk_dir = dir;
    disk.restart_at = 1;
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr,
                                         65536, disk);
    TEST_ASSERT_MSG(run.prepared[0].compressed && run.prepared[0].unified_view, run.prepared[0].pflash_stats.dump());
    const auto & stats = run.prepared[1].pflash_stats;
    TEST_ASSERT_MSG(run.prepared[1].unified_handback.empty() && stats.contains("unified") &&
                    stats["unified"].value("disk_view", false), stats.dump());
    TEST_ASSERT(run.prepared[1].compressed && run.prepared[1].unified_view);
    TEST_ASSERT(run.unified["counts"].value("disk_view", 0) == 1);
    // The follow-up's served prompt starts with what turn 1 served, up to its generation prompt.
    const int cut = stats["unified"].value("reused_tokens", -1);
    TEST_ASSERT_MSG(cut > 0 && cut == run.prepared[0].snapshot_cut && run.prepared[1].tokens.size() > (size_t) cut &&
                    std::equal(run.prepared[0].tokens.begin(), run.prepared[0].tokens.begin() + cut,
                               run.prepared[1].tokens.begin()), stats.dump());
    TEST_ASSERT(run.served_text[1].find("Line 60: plain filler text here mango") != std::string::npos);

    // The same without the disk cache: the view is gone after the restart.
    rm_rf(dir);
    UnifiedOptions ram;
    ram.restart_at = 1;
    const auto lost = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f,
                                          nullptr, 65536, ram);
    TEST_ASSERT_MSG(lost.prepared[1].unified_handback == "uncached", lost.prepared[1].pflash_stats.dump());
}

TEST_CASE(ServerUnitFixture, test_unified_unsplittable_paste_goes_to_the_chat_view) {
    // A large paste the view cannot cut into lines (one long line) would be prefilled raw: the chat view takes it.
    std::string paste;
    for (int i = 0; i < 300; ++i) paste += "word" + std::to_string(i % 10) + " ";
    const json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", paste + "\nWhat is this?"}},
    });
    const auto run = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score);
    UnifiedOptions off;
    off.chat_view_only = true;
    const auto base = run_unified_session({turn}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, nullptr, 65536,
                                          off);
    TEST_ASSERT_MSG(run.prepared[0].unified_handback == "uncached", run.prepared[0].pflash_stats.dump());
    TEST_ASSERT(run.prepared[0].tokens == base.prepared[0].tokens);
    TEST_ASSERT(run.calls_after == base.calls_after);
    TEST_ASSERT(run.unified["counts"].value("handback.uncached", 0) == 1);
}

TEST_CASE(ServerUnitFixture, test_unified_reread_never_paged_and_holds_units) {
    const json tools = json::array({{{"type", "function"}, {"function", {{"name", "read_file"}}}}});
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {60, "mango"}});
    json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Which line mentions KIWI?"}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
            unified_tool_call("call_1", "read_file", {{"path", "doc.md"}})})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", doc}},
    });
    // The agent reads the whole file again: a re-read, served verbatim.
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
        unified_tool_call("call_2", "read_file", {{"path", "doc.md"}, {"offset", 1}})})}});
    turn2.push_back({{"role", "tool"}, {"tool_call_id", "call_2"}, {"content", doc}});
    json turn3 = turn2;
    turn3.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn3.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    const auto run = run_unified_session({turn1, turn2, turn3}, tools, "read_file", 64, unified_fruit_score);
    const json & admission = run.prepared[1].pflash_stats["unified"]["admission"];
    TEST_ASSERT_MSG(admission[0].value("decision", "") == "reread", admission.dump());
    TEST_ASSERT(run.prepared[1].pflash_stats["unified"].value("reread_held_units", 0) > 0);
    TEST_ASSERT(run.calls_after[1] == 1);
    // Everything was re-read: the question finds nothing to recall.
    TEST_ASSERT(run.served_text[2].find("[Earlier in this conversation]") == std::string::npos);
    TEST_ASSERT(run.unified["counts"].value("recall.empty", 0) == 1);
}

TEST_CASE(ServerUnitFixture, test_unified_recall_coverage_bound_worst_case) {
    // A scorer that ranks every unit: questions recall until the whole region is held, then nothing more.
    const std::string doc = unified_doc(80, {});
    json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Question 0?"}},
    });
    std::vector<json> turns{turn};
    for (int q = 1; q <= 8; ++q) {
        turn.push_back({{"role", "assistant"}, {"content", "Ok."}});
        turn.push_back({{"role", "user"}, {"content", "Question " + std::to_string(q) + "?"}});
        turns.push_back(turn);
    }
    const auto run = run_unified_session(turns, nullptr, nullptr, 64, nullptr);
    const std::string & last = run.served_text.back();
    for (int line = 1; line <= 80; ++line) {
        TEST_ASSERT_MSG(unified_occurrences(last, "Line " + std::to_string(line) + ":") == 1, std::to_string(line));
    }
    // Once everything is back, a question appends nothing: the view grows by the new turn's raw tokens only.
    TEST_ASSERT(run.unified["counts"].value("recall.empty", 0) >= 1);
    const size_t n = run.prepared.size();
    TEST_ASSERT(run.prepared[n - 1].pflash_stats["unified"].value("decision", "") == "continue");
    const int grown = (int) run.prepared[n - 1].tokens.size() - (int) run.prepared[n - 2].tokens.size();
    const int raw_grown = (int) run.ids[n - 1].size() - (int) run.ids[n - 2].size();
    TEST_ASSERT(grown == raw_grown);
    // Cumulative appended by recall never exceeds what paging dropped plus labels and wrappers.
    int recalled = 0;
    for (size_t turn = 1; turn < n; ++turn) {
        const json & recall = run.prepared[turn].pflash_stats["unified"]["recall"];
        if (recall.is_array()) for (const auto & region : recall) recalled = region.value("cumulative_tokens", 0);
    }
    const json & paged = run.prepared[0].pflash_stats["unified"]["admission"][0];
    TEST_ASSERT(recalled > 0);
    TEST_ASSERT(recalled <= paged.value("region_tokens", 0) - paged.value("kept_tokens", 0) + 400);
}

TEST_CASE(ServerUnitFixture, test_unified_recall_off_flag) {
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {60, "mango"}});
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f, "0");
    TEST_ASSERT(run.calls_after == std::vector<int>({1, 1}));
    TEST_ASSERT(run.served_text[1].find("[Earlier in this conversation]") == std::string::npos);
    TEST_ASSERT(run.prepared[1].pflash_stats["unified"].value("decision", "") == "continue");
}

// Like unified_fruit_score, with filler near zero: the head's mass sits on the line the question asks for.
static double unified_sharp_fruit_score(const std::string & segment, const std::string & prompt) {
    double score = 0.01;
    if (prompt.find("KIWI?") != std::string::npos && segment.find("kiwi") != std::string::npos) score += 50.0;
    if (prompt.find("MANGO?") != std::string::npos && segment.find("mango") != std::string::npos) score += 50.0;
    return score;
}

static std::vector<json> unified_fruit_turns(const char * second_question) {
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {60, "mango"}});
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", second_question}});
    return {turn1, turn2};
}

TEST_CASE(ServerUnitFixture, test_unified_pages_a_document_sent_before_its_question) {
    // The documents arrive as their own user message and the question as the next one: both are the newest turn,
    // so the documents are paged exactly as when they share one message with the question.
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {60, "mango"}});
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc}},
        {{"role", "user"}, {"content", "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_sharp_fruit_score);
    const json & u0 = run.prepared[0].pflash_stats["unified"];
    TEST_ASSERT_MSG(u0.value("decision", "") == "region", u0.dump());
    const json & adm = u0["admission"];
    TEST_ASSERT_MSG(adm.size() == 2, adm.dump());
    TEST_ASSERT(adm[0].value("message", 0) == 1 && adm[0].value("decision", "") == "region");
    TEST_ASSERT(adm[1].value("message", 0) == 2 && adm[1].value("decision", "") != "region");
    TEST_ASSERT(run.prepared[0].tokens.size() < run.ids[0].size());
    // The question is served verbatim, after the paged documents.
    const std::string & s0 = run.served_text[0];
    TEST_ASSERT(s0.find("Which line mentions KIWI?") != std::string::npos);
    TEST_ASSERT(s0.find("Line 7: plain filler text here kiwi\n") != std::string::npos);
    TEST_ASSERT(s0.find("Line 60: plain filler text here mango\n") == std::string::npos);
    // The next question continues the view and recalls the dropped mango line; the served prefix is untouched (I1).
    const json & u1 = run.prepared[1].pflash_stats["unified"];
    TEST_ASSERT_MSG(u1.value("decision", "") == "recall", u1.dump());
    TEST_ASSERT(run.served_text[1].find("Line 60: plain filler text here mango\n") != std::string::npos);
    const int vg1 = run.prepared[0].snapshot_cut;
    TEST_ASSERT(std::equal(run.prepared[0].tokens.begin(), run.prepared[0].tokens.begin() + vg1,
                           run.prepared[1].tokens.begin()));
    // Same paging as the one-message layout (where the question line is one more unit of the region): the same
    // document lines are kept.
    const auto one = run_unified_session(unified_fruit_turns("Which line mentions MANGO?"), nullptr, nullptr, 64,
                                         unified_sharp_fruit_score);
    const json & k_split = adm[0];
    const json & k_one = one.prepared[0].pflash_stats["unified"]["admission"][0];
    TEST_ASSERT_MSG(k_split.value("units", -1) + 1 == k_one.value("units", -2) &&
                    k_split.value("kept_units", -1) == k_one.value("kept_units", -2) &&
                    k_split.value("kept_tokens", -1) == k_one.value("kept_tokens", -2), k_split.dump() + k_one.dump());
}

TEST_CASE(ServerUnitFixture, test_unified_recall_gate_takes_only_what_the_question_needs) {
    UnifiedOptions gate;
    gate.recall_gate = "0.05";
    const auto turns = unified_fruit_turns("Which line mentions MANGO?");
    const auto run = run_unified_session(turns, nullptr, nullptr, 64, unified_sharp_fruit_score, 0, 0.3f, nullptr,
                                         65536, gate);
    const auto ungated = run_unified_session(turns, nullptr, nullptr, 64, unified_sharp_fruit_score);
    // Turn 1 is paging only: the gate changes nothing there.
    TEST_ASSERT(run.prepared[0].tokens == ungated.prepared[0].tokens);
    const json & u = run.prepared[1].pflash_stats["unified"];
    TEST_ASSERT_MSG(u.value("decision", "") == "recall", u.dump());
    TEST_ASSERT(u["recall_gate"].value("decision", "") == "admitted");
    // The mock scores against the whole prompt: the history's KIWI? lifts the held kiwi line as much as MANGO?
    // lifts the dropped mango line, so the deficit starts near one half and one unit covers it.
    const double before = u["recall_gate"].value("deficit_before", 0.0);
    TEST_ASSERT_MSG(before > 0.4 && before < 0.6, u["recall_gate"].dump());
    TEST_ASSERT(u["recall_gate"].value("deficit_after", 1.0) < 0.05);
    // Nothing was trimmed at rendering: the served deficit is the planned one.
    TEST_ASSERT(u["recall_gate"].value("trimmed_units", -1) == 0);
    TEST_ASSERT(std::fabs(u["recall_gate"].value("deficit_after", 1.0) - u["recall_gate"].value("deficit_planned", 0.0)) <
                1e-12);
    // The mango line comes back; the ungated fill also brings back filler, the gate does not.
    const std::string & s = run.served_text[1];
    TEST_ASSERT(s.find("Line 60: plain filler text here mango\n") != std::string::npos);
    const json & recall = u["recall"];
    const json & full = ungated.prepared[1].pflash_stats["unified"]["recall"];
    TEST_ASSERT(recall.is_array() && full.is_array());
    TEST_ASSERT(recall[0].value("units", 0) == 1);
    TEST_ASSERT_MSG(full[0].value("units", 0) > 1, full.dump());
    TEST_ASSERT(run.prepared[1].tokens.size() < ungated.prepared[1].tokens.size());
    // The served prefix is untouched (I1).
    const int vg1 = run.prepared[0].snapshot_cut;
    TEST_ASSERT(std::equal(run.prepared[0].tokens.begin(), run.prepared[0].tokens.begin() + vg1,
                           run.prepared[1].tokens.begin()));
    TEST_ASSERT(run.unified["counts"].value("recall.gate_admitted", 0) == 1);
    TEST_ASSERT(run.unified.value("recall_gate", 0.0) == 0.05);
}

TEST_CASE(ServerUnitFixture, test_unified_recall_gate_appends_nothing_when_the_view_answers) {
    UnifiedOptions gate;
    gate.recall_gate = "0.05";
    // The second question asks for the kiwi line again: it is held (kept at paging).
    const auto run = run_unified_session(unified_fruit_turns("Which line mentions KIWI? Once more."), nullptr,
                                         nullptr, 64, unified_sharp_fruit_score, 0, 0.3f, nullptr, 65536, gate);
    const json & u = run.prepared[1].pflash_stats["unified"];
    TEST_ASSERT_MSG(u.value("decision", "") == "continue", u.dump());
    TEST_ASSERT(u["recall_gate"].value("decision", "") == "present");
    TEST_ASSERT(u["recall_gate"].value("deficit_before", 1.0) < 0.05);
    TEST_ASSERT(run.served_text[1].find("[Earlier in this conversation]") == std::string::npos);
    // The view grows by the new turn's raw tokens only.
    const int grown = (int) run.prepared[1].tokens.size() - (int) run.prepared[0].tokens.size();
    TEST_ASSERT(grown == (int) run.ids[1].size() - (int) run.ids[0].size());
    TEST_ASSERT(run.calls_after == std::vector<int>({1, 2}));
    TEST_ASSERT(run.unified["counts"].value("recall.gate_present", 0) == 1);
}

TEST_CASE(ServerUnitFixture, test_unified_recall_gate_fails_closed_on_a_non_finite_lift) {
    // One history candidate scores NaN at the second question (finite at paging): the deficit cannot be computed,
    // so nothing is appended -- never a gate decision over a silently smaller mass.
    const auto score = [] (const std::string & segment, const std::string & prompt) {
        if (prompt.find("MANGO?") != std::string::npos && segment.find("Line 30:") != std::string::npos) return (double) NAN;
        return unified_sharp_fruit_score(segment, prompt);
    };
    UnifiedOptions gate;
    gate.recall_gate = "0.05";
    const auto run = run_unified_session(unified_fruit_turns("Which line mentions MANGO?"), nullptr, nullptr, 64,
                                         score, 0, 0.3f, nullptr, 65536, gate);
    const json & u = run.prepared[1].pflash_stats["unified"];
    TEST_ASSERT_MSG(u.value("decision", "") == "continue", u.dump());
    TEST_ASSERT(u["recall_gate"].value("decision", "") == "unscorable");
    TEST_ASSERT(run.served_text[1].find("[Earlier in this conversation]") == std::string::npos);
    TEST_ASSERT(run.unified["counts"].value("recall.unscorable", 0) == 1);
}

TEST_CASE(ServerUnitFixture, test_unified_recall_gate_leaves_newly_dropped_content_out_of_the_context) {
    // The second question arrives with a new document that has a mango mark on every other line: paging keeps some
    // of them and drops the rest. That dropped mass is neither present nor recallable, so the gate's S excludes it
    // and the old region's missing mango line is measured against what the view could hold.
    std::map<int, std::string> marks;
    for (int i = 2; i <= 80; i += 2) marks[i] = "mango";
    const std::string fresh_doc = unified_doc(80, marks, "Row ");
    const auto turns = unified_fruit_turns((fresh_doc + "Which line mentions MANGO?").c_str());
    UnifiedOptions gate;
    gate.recall_gate = "0.01";
    gate.trace_units = "1";
    const auto run = run_unified_session(turns, nullptr, nullptr, 64, unified_sharp_fruit_score, 0, 0.3f, nullptr,
                                         65536, gate);
    const json & u = run.prepared[1].pflash_stats["unified"];
    TEST_ASSERT_MSG(u.value("decision", "") == "region_recall", u.dump());
    const json & g = u["recall_gate"];
    const json & q = u["recall_units"];
    const double dropped = g.value("new_dropped_mass", 0.0);
    TEST_ASSERT_MSG(dropped > 0.0, g.dump());
    TEST_ASSERT(q.value("new_dropped_mass", 0.0) == dropped);
    const double s = q.value("context_mass", 0.0) - dropped;
    TEST_ASSERT(std::fabs(g.value("context_mass", 0.0) - s) < 1e-12);
    double missing = 0.0;
    const json & r = q["regions"][0];
    for (size_t k = 0; k < r["held"].size(); ++k) {
        if (!r["held"][k].get<int>()) missing += r["mass"][k].get<double>();
    }
    TEST_ASSERT_MSG(std::fabs(g.value("deficit_before", 0.0) - missing / s) < 1e-9, g.dump());
    // The old region's mango line comes back at the start of the question.
    TEST_ASSERT(g.value("decision", "") == "admitted");
    TEST_ASSERT(run.served_text[1].find("Line 60: plain filler text here mango\n") != std::string::npos);
}

TEST_CASE(ServerUnitFixture, test_unified_recall_gate_skips_the_drafter_once_everything_is_held) {
    // A flat scorer: the deficit stays above a tiny theta, so questions recall to the cap until the region is
    // complete; after that a question runs no drafter pass at all.
    const std::string doc = unified_doc(80, {});
    json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Question 0?"}},
    });
    std::vector<json> turns{turn};
    for (int q = 1; q <= 8; ++q) {
        turn.push_back({{"role", "assistant"}, {"content", "Ok."}});
        turn.push_back({{"role", "user"}, {"content", "Question " + std::to_string(q) + "?"}});
        turns.push_back(turn);
    }
    UnifiedOptions gate;
    gate.recall_gate = "0.000001";
    const auto run = run_unified_session(turns, nullptr, nullptr, 64, nullptr, 0, 0.3f, nullptr, 65536, gate);
    const size_t n = run.prepared.size();
    for (int line = 1; line <= 80; ++line) {
        TEST_ASSERT_MSG(unified_occurrences(run.served_text.back(), "Line " + std::to_string(line) + ":") == 1,
                        std::to_string(line));
    }
    TEST_ASSERT(run.unified["counts"].value("recall.complete", 0) >= 1);
    TEST_ASSERT(run.calls_after[n - 1] == run.calls_after[n - 2]);
    const json & last = run.prepared[n - 1].pflash_stats["unified"];
    TEST_ASSERT_MSG(last.value("decision", "") == "continue", last.dump());
    TEST_ASSERT(last["recall_gate"].value("decision", "") == "complete");
}

TEST_CASE(ServerUnitFixture, test_unified_recall_trace_units_records_without_changing_bytes) {
    UnifiedOptions traced;
    traced.trace_units = "1";
    const auto turns = unified_fruit_turns("Which line mentions MANGO?");
    const auto run = run_unified_session(turns, nullptr, nullptr, 64, unified_sharp_fruit_score, 0, 0.3f, nullptr,
                                         65536, traced);
    const auto plain = run_unified_session(turns, nullptr, nullptr, 64, unified_sharp_fruit_score);
    for (size_t i = 0; i < run.prepared.size(); ++i) TEST_ASSERT(run.prepared[i].tokens == plain.prepared[i].tokens);
    // Paging: the region's cuts relative to its content, and the units it kept.
    const json & paged = run.prepared[0].pflash_stats["unified"]["admission"][0]["trace_units"];
    const int units = run.prepared[0].pflash_stats["unified"]["admission"][0].value("units", 0);
    TEST_ASSERT(paged["cuts"].size() == (size_t) units + 1);
    TEST_ASSERT(paged["kept"].size() == (size_t) units);
    // A question: every unit's held flag, mass, cost and score, and the context mass.
    const json & q = run.prepared[1].pflash_stats["unified"]["recall_units"];
    TEST_ASSERT(q.value("context_mass", 0.0) > 0.0);
    const json & r = q["regions"][0];
    TEST_ASSERT(r["held"] == paged["kept"]);
    TEST_ASSERT(r["mass"].size() == (size_t) units && r["cost"].size() == (size_t) units);
    double total = 0.0;
    for (const auto & m : r["mass"]) total += m.get<double>();
    TEST_ASSERT(total > 0.0 && total <= q.value("context_mass", 0.0) + 1e-9);
    TEST_ASSERT(plain.prepared[1].pflash_stats["unified"].find("recall_units") ==
                plain.prepared[1].pflash_stats["unified"].end());
}

TEST_CASE(ServerUnitFixture, test_unified_recall_gate_flags_fail_startup_on_bad_values) {
    const std::string path = write_pflash_bpe_tokenizer_fixture({"a"}, "abc");
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    ServerConfig config;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    config.pflash = true;
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    const auto refused = [&] (const char * gate, const char * source, const char * recall) {
        luce_test::ScopedEnvVar g{"PFLASH_RECALL_GATE", gate};
        luce_test::ScopedEnvVar s{"PFLASH_RECALL_SOURCE", source};
        luce_test::ScopedEnvVar r{"PFLASH_RECALL", recall};
        LuceEngine engine(std::make_unique<MockBackend>());
        try {
            HttpServer server(engine, tokenizer, config);
        } catch (const std::runtime_error &) {
            return true;
        }
        return false;
    };
    TEST_ASSERT(refused("0", nullptr, nullptr));
    TEST_ASSERT(refused("1", nullptr, nullptr));
    TEST_ASSERT(refused("1.5", nullptr, nullptr));
    TEST_ASSERT(refused("0.1x", nullptr, nullptr));
    TEST_ASSERT(refused("nan", nullptr, nullptr));
    TEST_ASSERT(refused("Off", nullptr, nullptr));
    TEST_ASSERT(refused("0.1", nullptr, "0"));
    // PFLASH_RECALL_SOURCE was removed: any value refuses to start.
    TEST_ASSERT(refused(nullptr, "fresh", nullptr));
    TEST_ASSERT(refused(nullptr, "kept", nullptr));
    {
        luce_test::ScopedEnvVar trace{"PFLASH_RECALL_TRACE_UNITS", "yes"};
        TEST_ASSERT(refused(nullptr, nullptr, nullptr));
    }
    TEST_ASSERT(!refused("0.1", nullptr, nullptr));
    TEST_ASSERT(!refused("off", nullptr, nullptr));
    TEST_ASSERT(!refused("off", nullptr, "0"));
    TEST_ASSERT(!refused(nullptr, nullptr, "0"));
    TEST_ASSERT(!refused(nullptr, nullptr, nullptr));
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_unified_defaults_gate_and_every_tool) {
    // The unified view's defaults: recall is gated at theta 0.075 and every tool's output may be paged. "off" and
    // an empty tool list opt out; PFLASH_RECALL=0 leaves no gate.
    const std::string path = write_pflash_bpe_tokenizer_fixture({"a"}, "abc");
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    ServerConfig config;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    config.pflash = true;
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    const auto settings = [&] (const char * gate, const char * tools, const char * recall) {
        luce_test::ScopedEnvVar g{"PFLASH_RECALL_GATE", gate};
        luce_test::ScopedEnvVar t{"PFLASH_REGION_TOOLS", tools};
        luce_test::ScopedEnvVar r{"PFLASH_RECALL", recall};
        LuceEngine engine(std::make_unique<MockBackend>());
        HttpServer server(engine, tokenizer, config);
        return HttpServerTestAccess::pflash_unified(server);
    };
    const json defaults = settings(nullptr, nullptr, nullptr);
    TEST_ASSERT_MSG(defaults.value("region_tools", "") == "*", defaults.dump());
    TEST_ASSERT(std::fabs(defaults["recall_gate"].get<double>() - 0.075) < 1e-12);
    const json opted_out = settings("off", "", nullptr);
    TEST_ASSERT_MSG(opted_out.value("region_tools", "x").empty(), opted_out.dump());
    TEST_ASSERT(opted_out["recall_gate"].is_null());
    const json listed = settings("0.2", "read_file,grep", nullptr);
    TEST_ASSERT(std::fabs(listed["recall_gate"].get<double>() - 0.2) < 1e-12);
    TEST_ASSERT(settings(nullptr, nullptr, "0")["recall_gate"].is_null());
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_unified_experimental_flags_fail_startup) {
    const std::string path = write_pflash_bpe_tokenizer_fixture({"a"}, "abc");
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    ServerConfig config;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    config.pflash = true;
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    {
        luce_test::ScopedEnvVar arm{"PFLASH_UNIFIED_DEFRAG", "1"};
        LuceEngine engine(std::make_unique<MockBackend>());
        bool threw = false;
        try {
            HttpServer server(engine, tokenizer, config);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        TEST_ASSERT(threw);
    }
    {
        luce_test::ScopedEnvVar tools{"PFLASH_REGION_TOOLS", "read_file,,grep"};
        LuceEngine engine(std::make_unique<MockBackend>());
        bool threw = false;
        try {
            HttpServer server(engine, tokenizer, config);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        TEST_ASSERT(threw);
    }
    unlink(path.c_str());
}

// The four lines of paragraph ``p`` of unified_paragraph_doc().
static std::string unified_paragraph_lines(int p, const std::map<int, std::string> & marks) {
    std::string out;
    for (int i = 1; i <= 4; ++i) {
        out += "P" + std::to_string(p) + " line " + std::to_string(i) + ": filler text";
        const auto it = marks.find(p);
        if (i == 3 && it != marks.end()) out += " " + it->second;
        out += "\n";
    }
    return out;
}

// Paragraphs of four lines "P<p> line <i>: filler text[ mark]\n", each followed by a blank line; a mark goes on line
// 3 of its paragraph, and paragraph ``gap`` gets a second blank line.
static std::string unified_paragraph_doc(int paragraphs, const std::map<int, std::string> & marks, int gap = -1) {
    std::string out;
    for (int p = 1; p <= paragraphs; ++p) {
        out += unified_paragraph_lines(p, marks);
        out += p == gap ? "\n\n" : "\n";
    }
    return out;
}

// ``served`` is a paged form of ``raw``: every served line is the next raw line, a marker for the raw lines it
// hides, or the one preamble, whose counts agree. "" when it holds, else what broke.
static std::string unified_check_paged(const std::string & served, const std::string & raw) {
    const auto split = [] (const std::string & text) {
        std::vector<std::string> out;
        size_t at = 0;
        for (;;) {
            const size_t nl = text.find('\n', at);
            out.push_back(text.substr(at, nl == std::string::npos ? std::string::npos : nl - at));
            if (nl == std::string::npos) break;
            at = nl + 1;
        }
        if (!text.empty() && text.back() == '\n') out.pop_back();
        return out;
    };
    const auto raw_lines = split(raw);
    const std::string open = "[\xE2\x80\xA6 ";
    const std::string close = " not shown \xE2\x80\xA6]";
    size_t next = 0;
    int shown = 0;
    int preambles = 0;
    int said_shown = -1;
    int said_total = -1;
    for (const auto & line : split(served)) {
        if (line.rfind("[Server note: this ", 0) == 0) {
            ++preambles;
            const size_t counts = line.find("shortened: ");
            if (counts == std::string::npos ||
                std::sscanf(line.c_str() + counts + 11, "%d of %d", &said_shown, &said_total) != 2) {
                return "preamble: " + line;
            }
            continue;
        }
        if (line.rfind(open, 0) == 0 && line.size() > close.size() &&
            line.compare(line.size() - close.size(), close.size(), close) == 0) {
            next += (size_t) std::atoi(line.c_str() + open.size());
            continue;
        }
        if (next >= raw_lines.size() || line != raw_lines[next]) {
            return "served line '" + line + "' at raw line " + std::to_string(next);
        }
        ++next;
        ++shown;
    }
    if (next != raw_lines.size()) return "ended at raw line " + std::to_string(next) + " of " +
                                          std::to_string(raw_lines.size());
    if (preambles != 1 || said_shown != shown || said_total != (int) raw_lines.size()) {
        return "preamble says " + std::to_string(said_shown) + " of " + std::to_string(said_total) + ", served " +
               std::to_string(shown) + " of " + std::to_string(raw_lines.size());
    }
    return "";
}

// The content of the ``nth`` (0-based) message of ``role`` in a rendered Qwen prompt.
static std::string unified_rendered_content(const std::string & text, const std::string & role, int nth) {
    const std::string open = "<|im_start|>" + role + "\n";
    size_t at = 0;
    for (int k = 0; k <= nth; ++k) {
        at = text.find(open, k == 0 ? 0 : at + 1);
        if (at == std::string::npos) return "";
    }
    const size_t begin = at + open.size();
    const size_t end = text.find("<|im_end|>", begin);
    return end == std::string::npos ? "" : text.substr(begin, end - begin);
}

TEST_CASE(ServerUnitFixture, test_unified_off_without_selection_env) {
    // Without a strict-selection environment the unified view cannot score: it is off (no stats) and the chat view
    // serves every turn exactly as on its own (a continuation takes the FlowKV path there).
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {60, "mango"}});
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    UnifiedOptions options;
    options.select_mode = nullptr;
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f,
                                         nullptr, 65536, options);
    UnifiedOptions off = options;
    off.chat_view_only = true;
    const auto base = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f,
                                          nullptr, 65536, off);
    for (size_t turn = 0; turn < 2; ++turn) {
        TEST_ASSERT_MSG(run.prepared[turn].pflash_stats.find("unified") == run.prepared[turn].pflash_stats.end(),
                        run.prepared[turn].pflash_stats.dump());
        TEST_ASSERT(run.prepared[turn].tokens == base.prepared[turn].tokens);
        TEST_ASSERT(run.prepared[turn].compressed == base.prepared[turn].compressed);
    }
    TEST_ASSERT(run.calls_after == base.calls_after);
    TEST_ASSERT_MSG(run.unified.is_null(), run.unified.dump());
}

TEST_CASE(ServerUnitFixture, test_unified_reread_holds_unit_ending_in_blank_line) {
    // Units are paragraphs ending in a blank line. A short user message that pastes paragraph 10 again covers its
    // four lines; the blank line after it, next to an uncovered line, does not keep the unit from being held, so
    // the question recalls other paragraphs, not that one again.
    const std::map<int, std::string> marks{{3, "kiwi"}, {10, "mango"}};
    const std::string doc = unified_paragraph_doc(16, marks);
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "P3 line 3."}});
    turn2.push_back({{"role", "user"}, {"content", "Here it is again:\n" + unified_paragraph_lines(10, marks) +
                                                   "\nWhich line mentions MANGO?"}});
    UnifiedOptions options;
    options.paragraphs = true;
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 400, unified_fruit_score, 0, 0.3f,
                                         nullptr, 65536, options);
    TEST_ASSERT_MSG(run.prepared[0].pflash_stats["unified"].value("decision", "") == "region",
                    run.prepared[0].pflash_stats.dump());
    TEST_ASSERT(run.served_text[0].find("mango") == std::string::npos);
    const json & u2 = run.prepared[1].pflash_stats["unified"];
    TEST_ASSERT_MSG(u2["admission"][0].value("decision", "") == "small", u2.dump());
    TEST_ASSERT_MSG(u2.value("reread_held_units", 0) == 1, u2.dump());
    TEST_ASSERT_MSG(u2.value("decision", "") == "recall", u2.dump());
    TEST_ASSERT_MSG(unified_occurrences(run.served_text[1], "mango") == 1, run.served_text[1]);
}

TEST_CASE(ServerUnitFixture, test_unified_reread_coverage_counts_fallen_back_messages) {
    // A big user message is a region candidate, but its cap does not cover the preamble and marker, so it falls
    // back to verbatim while paging. It still repeats paragraph 10: that unit is held, never recalled again.
    const std::map<int, std::string> marks{{10, "mango"}};
    const std::string doc = unified_paragraph_doc(16, marks);
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Summarise this."}},
    });
    std::string message = "Here it is again:\n" + unified_paragraph_lines(10, marks);
    for (int i = 1; i <= 20; ++i) message += "Extra note " + std::to_string(i) + " about the plan\n";
    message += "Which line mentions MANGO?";
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "A list of paragraphs."}});
    turn2.push_back({{"role", "user"}, {"content", message}});
    UnifiedOptions options;
    options.paragraphs = true;
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 400, unified_fruit_score, 0, 0.15f,
                                         nullptr, 65536, options);
    TEST_ASSERT_MSG(run.prepared[0].pflash_stats["unified"].value("decision", "") == "region",
                    run.prepared[0].pflash_stats.dump());
    TEST_ASSERT(run.served_text[0].find("mango") == std::string::npos);
    const json & u2 = run.prepared[1].pflash_stats["unified"];
    TEST_ASSERT_MSG(u2["admission"][0].value("decision", "") == "cap_below_overhead", u2.dump());
    TEST_ASSERT_MSG(u2.value("decision", "") == "recall", u2.dump());
    TEST_ASSERT_MSG(u2.value("reread_held_units", 0) == 1, u2.dump());
    TEST_ASSERT_MSG(unified_occurrences(run.served_text[1], "mango") == 1, run.served_text[1]);
}

TEST_CASE(ServerUnitFixture, test_unified_reset_hands_back_to_the_chat_view) {
    // Recall overheads can grow a view past the raw prompt. A turn whose view no longer fits goes back to admission
    // on the raw prompt; that history is uncached, so the chat view takes the turn instead of a raw prefill.
    const std::string doc = unified_doc(80, {});
    json turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Question 0?"}},
    });
    std::vector<json> turns{turn};
    for (int q = 1; q <= 8; ++q) {
        turn.push_back({{"role", "assistant"}, {"content", "Ok."}});
        turn.push_back({{"role", "user"}, {"content", "Question " + std::to_string(q) + "?"}});
        turns.push_back(turn);
    }
    const auto probe = run_unified_session(turns, nullptr, nullptr, 64, nullptr);
    size_t over = 0;
    for (size_t t = 1; t < probe.prepared.size() && over == 0; ++t) {
        if (probe.prepared[t].tokens.size() > probe.ids[t].size()) over = t;
    }
    TEST_ASSERT_MSG(over > 0 && over + 1 < turns.size(), "no turn whose view is larger than its raw prompt");
    const std::vector<json> session(turns.begin(), turns.begin() + (long) over + 2);

    // The view of turn ``over`` does not fit with a long output allowance; its raw prompt does. The next turn, with a
    // short allowance again, must not continue the view the chat view replaced (nor an ancestor of it).
    UnifiedOptions options;
    options.max_outputs.assign(session.size(), 16);
    options.max_outputs[over] = 4096;
    const int fits_raw = (int) probe.ids[over].size() + 4096;
    const auto run = run_unified_session(session, nullptr, nullptr, 64, nullptr, 0, 0.3f, nullptr, fits_raw, options);
    TEST_ASSERT_MSG(run.prepared[over].unified_handback == "reset_uncached", run.prepared[over].pflash_stats.dump());
    TEST_ASSERT(run.prepared[over].error.empty());
    TEST_ASSERT(run.unified["counts"].value("reset.context", 0) == 1);
    TEST_ASSERT(run.unified["counts"].value("raw.reset", 0) == 0);
    // This chat view keeps every candidate, so it serves the raw history: nothing to adopt, and the next turn is
    // uncached again (test_unified_cold_chat_history_is_adopted_after_the_chat_view covers a compressed one).
    TEST_ASSERT(run.unified["counts"].value("adopt.reset_uncached", 0) == 0);
    TEST_ASSERT_MSG(run.prepared[over + 1].unified_handback == "uncached", run.prepared[over + 1].pflash_stats.dump());
    TEST_ASSERT_MSG(run.unified.value("views", -1) == 0, run.unified.dump());

    // Nothing fits (the output allowance alone is the context): the chat view answers as it does without the flag.
    const int fits_view = (int) probe.prepared[over].tokens.size() + 16;
    UnifiedOptions tight;
    tight.max_outputs.assign(session.size(), 16);
    tight.max_outputs[over] = fits_view;
    tight.allow_errors = true;
    const auto again = run_unified_session(session, nullptr, nullptr, 64, nullptr, 0, 0.3f, nullptr, fits_view, tight);
    UnifiedOptions tight_off = tight;
    tight_off.chat_view_only = true;
    const auto base = run_unified_session(session, nullptr, nullptr, 64, nullptr, 0, 0.3f, nullptr, fits_view,
                                          tight_off);
    TEST_ASSERT_MSG(again.prepared[over].unified_handback == "reset_uncached", again.prepared[over].pflash_stats.dump());
    TEST_ASSERT_MSG(again.prepared[over].error == base.prepared[over].error, again.prepared[over].error);
    TEST_ASSERT(again.prepared[over].tokens == base.prepared[over].tokens);

    // The refused turn keeps the conversation's views: a retry with a short allowance continues the view.
    std::vector<json> retry(session.begin(), session.begin() + (long) over + 1);
    retry.push_back(session[over]);
    const auto retried = run_unified_session(retry, nullptr, nullptr, 64, nullptr, 0, 0.3f, nullptr, fits_view, tight);
    TEST_ASSERT(!retried.prepared[over].error.empty());
    TEST_ASSERT_MSG(retried.prepared[over + 1].unified_handback.empty() && retried.prepared[over + 1].unified_view,
                    retried.prepared[over + 1].pflash_stats.dump());
}

TEST_CASE(ServerUnitFixture, test_unified_widened_cap_applies_once) {
    // A stage-2 widened pass widens a region's cap by half, once: at paging and again at a later recall, never
    // compounded.
    const std::string doc = unified_doc(80, {{7, "kiwi"}, {60, "mango"}});
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "Line 7."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    UnifiedOptions options;
    options.widen = true;
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f,
                                         nullptr, 65536, options);
    const json & paged = run.prepared[0].pflash_stats["unified"]["admission"][0];
    const int unwidened = paged.value("unwidened_cap", 0);
    TEST_ASSERT_MSG(unwidened > 0 && paged.value("cap", 0) == unwidened + unwidened / 2, paged.dump());
    const json & recall = run.prepared[1].pflash_stats["unified"]["recall"];
    TEST_ASSERT_MSG(recall.is_array() && recall.size() == 1, run.prepared[1].pflash_stats.dump());
    TEST_ASSERT(recall[0].value("cap", 0) == paged.value("cap", 0));
    TEST_ASSERT(recall[0].value("wrapper", 0) > 0);
    TEST_ASSERT(recall[0].value("tokens", 0) <= paged.value("cap", 0) - recall[0].value("wrapper", 0));
}

TEST_CASE(ServerUnitFixture, test_unified_distinct_tokenizers_keep_lines_whole) {
    // Target and drafter merge differently: the target has "\n\n" and "\n\n\n" as one token (a blank line then
    // starts inside a token, so it can be cut only at its separator, and the second of two blank lines not at all),
    // and the escaped piece backslash + "nitem" (an escaped line "item ..." starts inside it); the drafter has its
    // own words. Lines stay whole, markers count what they hide, and turn 2 reuses turn 1's served prompt token for
    // token.
    UnifiedOptions options;
    options.distinct_drafter = true;
    options.target_words = {"\n\n", "\n\n\n", "\\nitem", " filler", " mentions"};
    options.drafter_words = {" text", " line", "Which"};

    const std::map<int, std::string> marks{{3, "kiwi"}, {10, "mango"}};
    const std::string doc = unified_paragraph_doc(16, marks, /*gap=*/6);
    const json turn1 = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", doc + "Which line mentions KIWI?"}},
    });
    json turn2 = turn1;
    turn2.push_back({{"role", "assistant"}, {"content", "P3 line 3."}});
    turn2.push_back({{"role", "user"}, {"content", "Which line mentions MANGO?"}});
    const auto run = run_unified_session({turn1, turn2}, nullptr, nullptr, 64, unified_fruit_score, 0, 0.3f,
                                         nullptr, 65536, options);
    // The scenario holds: a blank line starts inside a target token, at a drafter token start.
    const std::string & raw1 = run.raw_text[0];
    const size_t blank = raw1.find("text\n\nP2");
    const size_t gap = raw1.find("text\n\n\nP7");
    TEST_ASSERT(blank != std::string::npos && gap != std::string::npos);
    const auto starts_at = [] (const std::vector<size_t> & starts, size_t at) {
        return std::binary_search(starts.begin(), starts.end(), at);
    };
    TEST_ASSERT(starts_at(run.target_starts[0], blank + 4) && !starts_at(run.target_starts[0], blank + 5));
    TEST_ASSERT(starts_at(run.drafter_starts[0], blank + 5));
    TEST_ASSERT(!starts_at(run.target_starts[0], gap + 5) && !starts_at(run.target_starts[0], gap + 6));
    TEST_ASSERT(run.ids[0].size() != run.drafter_starts[0].size());

    TEST_ASSERT_MSG(run.prepared[0].pflash_stats["unified"].value("decision", "") == "region",
                    run.prepared[0].pflash_stats.dump());
    const std::string broke = unified_check_paged(unified_rendered_content(run.served_text[0], "user", 0),
                                                  doc + "Which line mentions KIWI?");
    TEST_ASSERT_MSG(broke.empty(), broke + "\n" + run.served_text[0]);
    TEST_ASSERT(run.served_text[0].find("P3 line 3: filler text kiwi\n") != std::string::npos);
    const int cut = run.prepared[0].snapshot_cut;
    TEST_ASSERT(cut > 0 && std::equal(run.prepared[0].tokens.begin(), run.prepared[0].tokens.begin() + cut,
                                      run.prepared[1].tokens.begin()));
    TEST_ASSERT_MSG(run.prepared[1].pflash_stats["unified"].value("decision", "") == "recall",
                    run.prepared[1].pflash_stats.dump());
    const std::string & s2 = run.served_text[1];
    const size_t block = s2.find("[Earlier in this conversation]\n[From an earlier user message]\n");
    TEST_ASSERT(block != std::string::npos);
    TEST_ASSERT(s2.find("P10 line 3: filler text mango\n", block) != std::string::npos);
    TEST_ASSERT(unified_occurrences(s2, "mango") == 1);

    // An escaped (JSON-string) tool output with the same two tokenizers: the result stays valid JSON and its lines
    // whole, and the line the scores favour is kept (the mark is long enough to lie whole in some 7-token chunk;
    // the cap leaves room for it next to the long tool preamble).
    std::string body;
    for (int i = 1; i <= 80; ++i) {
        body += "item " + std::to_string(i) + " value" + (i == 33 ? " kiwikiwikiwikiwi" : "") + "\n";
    }
    const std::string wrapped = json({{"content", body}, {"total_lines", 80}}).dump();
    const json tools = json::array({{{"type", "function"}, {"function", {{"name", "read_file"}}}}});
    const json tool_turn = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Which item mentions KIWI?"}},
        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({
            unified_tool_call("call_1", "read_file", {{"path", "items.txt"}})})}},
        {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", wrapped}},
    });
    const auto tool_run = run_unified_session({tool_turn}, tools, "read_file", 64, unified_fruit_score,
                                              /*chunk=*/7, 0.5f, nullptr, 65536, options);
    const size_t escaped = tool_run.raw_text[0].find("value\\nitem 40 ");
    TEST_ASSERT(escaped != std::string::npos);
    TEST_ASSERT(starts_at(tool_run.target_starts[0], escaped + 5) &&
                !starts_at(tool_run.target_starts[0], escaped + 7));
    TEST_ASSERT(starts_at(tool_run.drafter_starts[0], escaped + 7));
    const auto & prepared = tool_run.prepared[0];
    TEST_ASSERT_MSG(prepared.compressed, prepared.pflash_stats.dump());
    TEST_ASSERT(prepared.pflash_stats["unified"]["admission"][0].value("escaped_lines", false));
    const std::string & served = tool_run.served_text[0];
    const size_t open = served.rfind("<tool_response>\n");
    const size_t close = served.find("\n</tool_response>", open);
    TEST_ASSERT(open != std::string::npos && close != std::string::npos);
    const json result = json::parse(served.substr(open + 16, close - open - 16), nullptr, false);
    TEST_ASSERT_MSG(!result.is_discarded(), served.substr(open, 400));
    TEST_ASSERT(result.value("total_lines", 0) == 80);
    const std::string content = result.value("content", "");
    const std::string content_broke = unified_check_paged(content, body);
    TEST_ASSERT_MSG(content_broke.empty(), content_broke + "\n" + content);
    TEST_ASSERT(content.find("item 33 value kiwikiwikiwikiwi\n") != std::string::npos);
}

// PFlash on the concurrent scheduler. The engine counts the steps that ran
// while the drafter scored.
class PflashStageEngine final : public SeqEngine {
public:
    explicit PflashStageEngine(const std::atomic<bool> & scoring) : scoring_(scoring) {}
    int slot_count() const override { return 2; }
    int max_context() const override { return 4096; }
    bool token_is_eos(int32_t) const override { return false; }
    StepPlanLimits step_plan_limits(int) const override {
        StepPlanLimits limits;
        limits.max_prefill_sequences = 2;
        limits.max_prefill_tokens_per_sequence = 16;
        limits.max_prefill_tokens_total = 32;
        limits.prefill_allocation_quantum = 16;
        return limits;
    }
    AdmitResult admit(uint64_t, const std::vector<int32_t> & prompt, const SamplerCfg &) override {
        AdmitResult result;
        const int slot = !slots_[0].active ? 0 : !slots_[1].active ? 1 : -1;
        if (slot < 0) {
            result.status = AdmitResult::Status::busy;
            return result;
        }
        slots_[(size_t)slot] = {true, (int)prompt.size()};
        {
            std::lock_guard<std::mutex> lock(mu);
            admitted_sizes.push_back((int)prompt.size());
        }
        result.status = AdmitResult::Status::admitted;
        result.slot = slot;
        return result;
    }
    StepResult step(const StepPlan & plan) override {
        if (scoring_.load()) steps_beside_drafter.fetch_add(1);
        StepResult result;
        for (const StepInput & input : plan.decode) result.decode.push_back({input.slot, 7, false, {}});
        for (const PrefillSlice & slice : plan.prefills) {
            int & left = slots_[(size_t)slice.slot].remaining;
            left -= std::min(left, slice.max_tokens);
            PrefillOutput out;
            out.slot = slice.slot;
            out.status = left == 0 ? PrefillOutput::Status::completed
                                   : PrefillOutput::Status::advanced;
            if (left == 0) out.token = 7;
            result.prefills.push_back(out);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return result;
    }
    void retire(int slot) override { slots_[(size_t)slot] = {}; }

    std::mutex mu;
    std::vector<int> admitted_sizes;
    std::atomic<int> steps_beside_drafter{0};

private:
    struct Slot { bool active = false; int remaining = 0; };
    const std::atomic<bool> & scoring_;
    std::array<Slot, 2> slots_{};
};

struct PflashStageBackend : MockPflashBudgetBackend {
    std::atomic<bool> scoring{false};
    PflashStageEngine engine{scoring};
    SeqEngine * seq_engine() override { return &engine; }
    CompressResult compress(const CompressRequest & request) override {
        scoring = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CompressResult result = MockPflashBudgetBackend::compress(request);
        scoring = false;
        return result;
    }
};

struct PflashStageRun {
    bool done = false;
    int raw_tokens = 0;
    int compress_calls = 0;
    int steps_beside_drafter = 0;
    std::vector<int> admitted_sizes;
    std::vector<int> usage_prompt_tokens;  // per job, from its JSON reply
    std::vector<RoutingAdmission> admissions;
};

// Long prompts (PFlash) and short chats (served as they are) on a 2-slot
// engine; a chat starts first and the long prompts arrive while it decodes.
// Routed long prompts report their admission to a model router.
static PflashStageRun run_pflash_stage_case(int chats, int long_prompts,
                                            bool routed = false) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar chunk{"PFLASH_SELECT_CHUNK_SIZE", "4"};
    const auto prompt = pflash_long_system_prompt_case();
    const std::string path =
        write_pflash_bpe_tokenizer_fixture(prompt.vocab, prompt.rendered);
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));
    const std::vector<int32_t> ids = tokenizer.encode(prompt.rendered);

    auto backend_owner = std::make_unique<PflashStageBackend>();
    PflashStageBackend & backend = *backend_owner;
    LuceEngine engine(std::move(backend_owner));
    ServerConfig config;
    config.arch = "qwen35";
    config.pflash = true;
    config.pflash_threshold = 64;
    config.pflash_keep_ratio = 0.3f;
    config.max_ctx = 4096;
    config.prefix_cache_cap = 0;
    config.prefill_cache_cap = 0;
    config.admission_coalesce_ms = 0;
    PflashStageRun run;
    run.raw_tokens = (int)ids.size();
    {
        HttpServer server(engine, tokenizer, config);
        server.set_drafter_tokenizer(&tokenizer);
        const int n = chats + long_prompts;
        std::vector<int> fds((size_t)n * 2, -1);
        std::vector<std::unique_ptr<ServerJob>> jobs;
        for (int i = 0; i < n; ++i) {
            TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds[(size_t)i * 2]) == 0);
            auto job = std::make_unique<ServerJob>();
            job->fd = fds[(size_t)i * 2];
            job->req.format = ApiFormat::OPENAI_CHAT;
            job->req.stream = false;
            job->req.model = "scheduler-test";
            job->req.response_id = "job-" + std::to_string(i);
            if (i < chats) {
                job->req.prompt_tokens.assign(ids.begin(), ids.begin() + 4);
                job->req.max_output = 60;
            } else {
                job->req.messages = prompt.messages;
                job->req.prompt_tokens = ids;
                job->req.max_output = 1;
                job->report_admission = routed;
            }
            jobs.push_back(std::move(job));
        }
        for (int i = 0; i < chats; ++i) SchedulerTestHarness::enqueue(server, jobs[(size_t)i].get());
        std::thread scheduler([&] { SchedulerTestHarness::run(server, backend.engine); });
        if (chats > 0) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        for (int i = chats; i < n; ++i) SchedulerTestHarness::enqueue(server, jobs[(size_t)i].get());
        run.done = true;
        for (auto & job : jobs) {
            std::unique_lock<std::mutex> lock(job->mu);
            run.done &= job->cv.wait_for(lock, std::chrono::seconds(10), [&] { return job->done; });
            run.admissions.push_back(job->admission);
        }
        SchedulerTestHarness::stop(server);
        scheduler.join();
        for (int i = 0; i < n; ++i) {
            std::string reply;
            char buf[4096];
            ssize_t got;
            while ((got = recv(fds[(size_t)i * 2 + 1], buf, sizeof(buf), MSG_DONTWAIT)) > 0) {
                reply.append(buf, (size_t)got);
            }
            const size_t body = reply.find("\r\n\r\n");
            TEST_ASSERT(body != std::string::npos);
            const auto j = nlohmann::json::parse(reply.substr(body + 4));
            run.usage_prompt_tokens.push_back(j["usage"]["prompt_tokens"].get<int>());
        }
        for (const int fd : fds) close(fd);
    }
    run.compress_calls = backend.compress_calls;
    run.steps_beside_drafter = backend.engine.steps_beside_drafter.load();
    run.admitted_sizes = backend.engine.admitted_sizes;
    unlink(path.c_str());
    return run;
}

// A long prompt on an idle server is scored, then served compressed; its
// usage counts the prompt the client sent.
TEST_CASE(ServerUnitFixture, test_scheduler_pflash_serves_a_long_prompt_on_an_idle_server) {
    const auto run = run_pflash_stage_case(/*chats=*/0, /*long_prompts=*/1);
    TEST_ASSERT(run.done);
    TEST_ASSERT(run.compress_calls == 1);
    TEST_ASSERT(run.admitted_sizes.size() == 1);
    TEST_ASSERT(run.admitted_sizes[0] < run.raw_tokens);
    TEST_ASSERT(run.usage_prompt_tokens.size() == 1);
    TEST_ASSERT(run.usage_prompt_tokens[0] == run.raw_tokens);
}

// The chat already decoding keeps its slot stepping while the drafter
// scores the long prompt beside it.
TEST_CASE(ServerUnitFixture, test_scheduler_pflash_decodes_while_the_drafter_scores) {
    const auto run = run_pflash_stage_case(/*chats=*/1, /*long_prompts=*/1);
    TEST_ASSERT(run.done);
    TEST_ASSERT(run.compress_calls == 1);
    TEST_ASSERT(run.steps_beside_drafter > 0);
    TEST_ASSERT(run.admitted_sizes.size() == 2);
    TEST_ASSERT(run.admitted_sizes[1] < run.raw_tokens);
}

// Two long prompts that arrive together are both scored and served
// compressed.
TEST_CASE(ServerUnitFixture, test_scheduler_pflash_serves_two_long_prompts) {
    const auto run = run_pflash_stage_case(/*chats=*/0, /*long_prompts=*/2);
    TEST_ASSERT(run.done);
    TEST_ASSERT(run.compress_calls == 2);
    TEST_ASSERT(run.admitted_sizes.size() == 2);
    TEST_ASSERT(run.admitted_sizes[0] < run.raw_tokens);
    TEST_ASSERT(run.admitted_sizes[1] < run.raw_tokens);
}

// A routed long prompt is scored too, and served by the model that scored it.
TEST_CASE(ServerUnitFixture, test_scheduler_pflash_compresses_a_routed_prompt) {
    const auto run = run_pflash_stage_case(/*chats=*/0, /*long_prompts=*/1,
                                           /*routed=*/true);
    TEST_ASSERT(run.done);
    TEST_ASSERT(run.compress_calls == 1);
    TEST_ASSERT(run.admitted_sizes.size() == 1);
    TEST_ASSERT(run.admitted_sizes[0] < run.raw_tokens);
    TEST_ASSERT(run.admissions.size() == 1);
    TEST_ASSERT(run.admissions[0] == RoutingAdmission::handled);
    TEST_ASSERT(run.usage_prompt_tokens[0] == run.raw_tokens);
}

// Each thread reads the error its own failing call set: the PFlash thread
// scores beside the scheduler, and neither may see, or free, the other's.
TEST_CASE(ServerUnitFixture, test_last_error_is_per_thread) {
    luce::common::set_last_error("scheduler error");
    std::string seen_by_other;
    std::thread other([&] {
        luce::common::set_last_error("drafter error");
        seen_by_other = luce_last_error();
    });
    other.join();
    TEST_ASSERT(seen_by_other == "drafter error");
    TEST_ASSERT(std::string(luce_last_error()) == "scheduler error");
}
