#include "server/http_server.h"
#include "common/pflash_drafter_ipc.h"
#include "qwen3/qwen3_drafter.h"
#include "qwen3/qwen3_drafter_model.h"
#include "dflash27b.h"
#include "common/concurrency/seq_engine.h"
#include "support/environment.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cmath>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <fcntl.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

#if !defined(_WIN32)
#include <sys/socket.h>
#endif

namespace dflash::common {
std::vector<ChatMessage> normalize_chat_messages(
    const json & messages,
    ApiFormat format,
    ToolMemory & tool_memory);
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
        200, 201, 100, 101, 102, 103, 104, 105, 106, 107,
    };

    const auto window = http_detail::find_pflash_query_window(
        rendered, query, /*search_end=*/12);

    TEST_ASSERT(window.valid());
    TEST_ASSERT(window.tokens == 8);
    TEST_ASSERT(window.end == 10);
    TEST_ASSERT((int)rendered.size() - window.end == 10);
}

TEST_CASE(ServerUnitFixture, test_pflash_scorer_accepts_responses_string_input) {
    ToolMemory tool_memory;
    const auto messages = normalize_chat_messages(
        json("Which token is the answer?"), ApiFormat::RESPONSES, tool_memory);

    TEST_ASSERT(
        http_detail::pflash_user_query_text(messages) ==
        "Which token is the answer?");
}

TEST_CASE(ServerUnitFixture, test_pflash_query_mapping_tolerates_one_bpe_boundary_token) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<int32_t> rendered{
        1, 2, 999, 11, 12, 13, 14, 15, 16, 17, 200, 201,
    };

    const auto window = http_detail::find_pflash_query_window(
        rendered, query, /*search_end=*/10);

    TEST_ASSERT(window.valid());
    TEST_ASSERT(window.tokens == 7);
    TEST_ASSERT(window.end == 10);
}

TEST_CASE(ServerUnitFixture, test_pflash_query_mapping_rejects_weak_punctuation_match) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<int32_t> rendered{1, 2, 15, 16, 17, 200, 201};
    TEST_ASSERT(!http_detail::find_pflash_query_window(
        rendered, query, /*search_end=*/7).valid());

    const std::vector<int32_t> short_query{30, 31, 32};
    const std::vector<int32_t> short_rendered{1, 30, 31, 32, 200};
    const auto short_window =
        http_detail::find_pflash_query_window(
            short_rendered, short_query, /*search_end=*/4);
    TEST_ASSERT(short_window.valid());
    TEST_ASSERT(short_window.tokens == 3);
    TEST_ASSERT(short_window.end == 4);
}

TEST_CASE(ServerUnitFixture, test_pflash_query_mapping_rejects_earlier_turn_suffix) {
    const std::vector<int32_t> query{10, 11, 12, 13, 14, 15, 16, 17};
    // The generic four-token suffix occurs in an earlier turn. The real user
    // message diverges before its suffix, so it cannot safely supply a window.
    std::vector<int32_t> rendered{1, 14, 15, 16, 17};
    rendered.insert(rendered.end(), 24, 200);
    rendered.insert(rendered.end(), {10, 11, 12, 999});
    TEST_ASSERT(!http_detail::find_pflash_query_window(
        rendered, query, (int) rendered.size()).valid());

    // The same narrowed suffix at the actual query boundary remains usable.
    rendered.insert(rendered.end(), {14, 15, 16, 17, 201, 202});
    const auto window = http_detail::find_pflash_query_window(
        rendered, query, (int) rendered.size());
    TEST_ASSERT(window.valid());
    TEST_ASSERT(window.tokens == 4);
    TEST_ASSERT(window.end == (int) rendered.size() - 2);
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

TEST_CASE(ServerUnitFixture, test_qwen35_pflash_default_query_window_reaches_state_validation) {
    DrafterContext ctx;
    ctx.loaded = true;
    ctx.arch = DrafterArch::Qwen35_0p8b;
    const std::vector<int32_t> ids(16, 1);

    const auto compressed = drafter_score_and_compress(
        ctx, ids, 0.5f, /*chunk_size=*/32, /*n_lookahead=*/8,
        /*pool_kernel=*/13, /*score_query_end=*/-1);

    TEST_ASSERT(compressed.empty());
    TEST_ASSERT(std::string(dflash27b_last_error()) ==
                "qwen35 drafter state missing");
}

TEST_CASE(ServerUnitFixture, test_pflash_ipc_rejects_unsupported_query_widths) {
    TEST_ASSERT(valid_pflash_score_query_tokens(1));
    TEST_ASSERT(valid_pflash_score_query_tokens(8));
    TEST_ASSERT(!valid_pflash_score_query_tokens(0));
    TEST_ASSERT(!valid_pflash_score_query_tokens(9));
    TEST_ASSERT(!valid_pflash_score_query_tokens(
        (std::numeric_limits<int>::max)()));
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

// ═══════════════════════════════════════════════════════════════════════
// PFlash config tests (model-free)
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE(ServerUnitFixture, test_pflash_config_defaults) {
    ServerConfig cfg;
    TEST_ASSERT(cfg.pflash_mode == ServerConfig::PflashMode::OFF);
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

TEST_CASE(ServerUnitFixture, test_server_config_cache_defaults) {
    ServerConfig cfg;
    TEST_ASSERT(cfg.prefix_cache_cap == 32);
    TEST_ASSERT(cfg.prefill_cache_cap == 0);
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
