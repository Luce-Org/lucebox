#include "server/prefix_cache.h"
#include "server/pin_friendly_prompt.h"
#include "server/freeze_history.h"
#include "server/prompt_normalize.h"
#include "server/http_server.h"
#include "common/concurrency/seq_engine.h"
#include "gguf.h"
#include <nlohmann/json.hpp>
#include "support/environment.h"
#include "support/mock_backend.h"
#include "support/test_assert.h"
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

namespace dflash::common {

// Drives the scheduler loop and reaches into HttpServer's private state so
// restore/capture handling can be observed without a live socket pair.
struct SchedulerTestHarness {
    static PrefixCache & prefix_cache(HttpServer & server) {
        return server.prefix_cache_;
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

    static void finalize_inline_snapshot(
            HttpServer & server, const std::vector<int32_t> & prompt,
            PrefixCache::InlineReservation reservation,
            int slot, int requested_cut) {
        ParsedRequest req;
        req.prompt_tokens = prompt;
        HttpServer::PreparedPrompt prepared;
        prepared.tokens = prompt;
        HttpServer::GenerationCacheState cache;
        cache.snap_reservation = std::move(reservation);
        cache.snap_slot = slot;
        cache.snap_cut = requested_cut;
        cache.snap_prepared = true;
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
};

}

// ═══════════════════════════════════════════════════════════════════════
// Prefix cache hash tests (model-free)
// ═══════════════════════════════════════════════════════════════════════

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
                      token_types,
                      sizeof(token_types) / sizeof(token_types[0]));
    gguf_set_val_str(g, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen35");
    gguf_set_val_u32(g, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", 2);

    const std::string path = test_tmp_path("dflash_test_deepseek_markers.gguf").string();
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    return path;
}

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
    // The default snapshot cut stays before the current user turn.
    TEST_ASSERT(select_inline_snapshot_boundary(
                    find_all_boundaries(prompt, markers)) == 8);

    // A first turn snapshots its system text, so a new session on the same
    // system prompt restores it instead of prefilling it again.
    const std::vector<int32_t> first_turn = {1, 100, 3, 101, 4};
    TEST_ASSERT(find_all_boundaries(first_turn, markers) ==
                std::vector<int>({3, 5}));
    TEST_ASSERT(select_inline_snapshot_boundary(
                    find_all_boundaries(first_turn, markers)) == 3);
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

    remove_test_path(path);
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
    auto blocked = cache.reserve_inline_snap(
        next, /*restored_prefix_len=*/0,
        /*prefer_tools_boundary=*/false, /*forced_cut=*/2);
    TEST_ASSERT(!blocked.active());
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

    remove_test_path(path);
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

    remove_test_path(path);
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
    const int32_t unused_token = 0;
    auto h = hash_prefix(&unused_token, 0);
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
    // Second-to-last is the boundary before the current user turn.
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries) == 380);
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 240) == 380);
    // Do not reserve a snapshot when the restore already covers that point.
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 380) == 0);
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 500) == 0);
    TEST_ASSERT(select_inline_snapshot_boundary({}, 0) == 0);
    TEST_ASSERT(select_inline_snapshot_boundary({100}, 0) == 100);
}

TEST_CASE(ServerUnitFixture, test_inline_snapshot_prefers_tools_boundary_until_restored) {
    const std::vector<int> boundaries = {100, 240, 380, 520};
    // Cold tool-heavy: pin system+tools head (first marker), not deepen cut.
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 0, true) == 100);
    // After tools head is restored, deepen to second-to-last.
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 100, true) == 380);
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 380, true) == 0);
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
    TEST_ASSERT(select_inline_snapshot_boundary(boundaries, 100, true) == 380);

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
    // Oldest-first chain: [s] < [s,a] < [s,a,b]. Only the longest is a leaf, so
    // the short shared ancestors are kept and the victim is the deepest entry.
    std::vector<std::vector<int32_t>> ids = {{9}, {9, 1}, {9, 1, 2}};
    TEST_ASSERT(select_inline_evict_victim(ids) == 2);
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
    remove_test_path(path);
}

TEST_CASE(ServerUnitFixture, test_slide_restore_source_never_evicted) {
    std::vector<std::vector<int32_t>> ids = {
        {9}, {9, 1}, {9, 1, 2}, {9, 1, 2, 3},
    };
    for (int skip = 0; skip < 4; ++skip) {
        const int victim = select_inline_evict_victim(ids, nullptr, skip);
        TEST_ASSERT(victim >= 0 && victim != skip);
    }
    // Linear chain whose only leaf is the restore source: evict the
    // shallowest unprotected ancestor instead of cancelling everything.
    TEST_ASSERT(select_inline_evict_victim(ids, nullptr, 3) == 0);
    // With a free leaf the usual leaf preference still applies.
    TEST_ASSERT(select_inline_evict_victim(ids, nullptr, 0) == 3);
}

TEST_CASE(ServerUnitFixture, test_slide_protected_pin_never_evicted) {
    std::vector<std::vector<int32_t>> ids = {
        {9}, {9, 1}, {9, 1, 2}, {9, 1, 2, 3},
    };
    std::vector<bool> protect = {true, false, false, false};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect, 3) == 1);

    // Only the protected pin and the restore source remain: no safe victim,
    // so no snapshot is reserved instead of destroying the pin.
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
    remove_test_path(path);
}

TEST_CASE(ServerUnitFixture, test_slide_branching_oldest_leaf_unchanged) {
    // [9] is a shared root; leaves are idx 1 ([9,1]) and idx 2 ([9,2]).
    std::vector<std::vector<int32_t>> ids = {{9}, {9, 1}, {9, 2}};
    TEST_ASSERT(select_inline_evict_victim(ids) == 1);  // original behavior
    // Restore source is the newer leaf: the older leaf is still the victim.
    TEST_ASSERT(select_inline_evict_victim(ids, nullptr, 2) == 1);
    // Restore source is the older leaf: the remaining leaf is the victim.
    TEST_ASSERT(select_inline_evict_victim(ids, nullptr, 1) == 2);
    // A protected leaf is never evicted: with the restore source skipped and
    // the only remaining leaf protected, the shallowest unprotected ancestor
    // is the victim instead.
    std::vector<bool> protect = {false, true, false};
    TEST_ASSERT(select_inline_evict_victim(ids, &protect, 2) == 0);
}

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
    remove_test_path(path);
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
    std::string out = dflash::common::normalize_system_for_cache(system_blocks);
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
    std::string out = dflash::common::normalize_system_for_cache(messages);
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
    std::string out4 = dflash::common::normalize_system_for_cache(messages_turn4);
    std::string out5 = dflash::common::normalize_system_for_cache(messages_turn5);
    TEST_ASSERT(out4 == out5);
}

TEST_CASE(ServerUnitFixture, test_normalize_preserves_legit_system_content) {
    // A normal system prompt containing no billing header must pass through unchanged.
    json messages = json::array({
        {{"role", "system"},
         {"content", "You are an expert in C++ performance optimization."}},
        {{"role", "user"}, {"content", "Help me optimize this loop."}}
    });
    std::string out = dflash::common::normalize_system_for_cache(messages);
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
    std::string out = dflash::common::normalize_system_for_cache(system_blocks);
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
    std::string norm_a = dflash::common::normalize_system_for_cache(messages_a);
    std::string norm_b = dflash::common::normalize_system_for_cache(messages_b);
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

// T5: exercise the production FlowKV activation decision and defaults.
TEST_CASE(ServerUnitFixture, test_flowkv_T5_aggregate_activation_threshold) {
    ServerConfig config;
    config.pflash_mode = ServerConfig::PflashMode::AUTO;

    TEST_ASSERT(http_detail::flowkv_activation_threshold(config) == 32000);
    TEST_ASSERT(!http_detail::flowkv_should_activate(config, 13000));
    TEST_ASSERT(http_detail::flowkv_should_activate(config, 32000));

    config.pflash_threshold = 12000;
    TEST_ASSERT(!http_detail::flowkv_should_activate(config, 11999));
    TEST_ASSERT(http_detail::flowkv_should_activate(config, 13000));

    config.pflash_mode = ServerConfig::PflashMode::ALWAYS;
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
}

// ═══════════════════════════════════════════════════════════════════════
// Scheduler prefix-cache lifecycle tests
// ═══════════════════════════════════════════════════════════════════════

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

TEST_CASE(ServerUnitFixture,
          test_inline_snapshot_finalization_uses_actual_saved_position) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    ShortInlineSnapshotBackend backend;
    ServerConfig config;
    config.prefix_cache_cap = 2;
    HttpServer server(backend, tokenizer, config);
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
    TEST_ASSERT(hit.second == backend.saved_position);
    TEST_ASSERT(SchedulerTestHarness::slot_tokens(
                    server, backend.saved_slot) ==
                std::vector<int32_t>(prompt.begin(), prompt.begin() + 3));

    remove_test_path(path);
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

TEST_CASE(ServerUnitFixture,
          test_scheduler_counts_restore_only_after_engine_attempts_it) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    SchedulerPrefixBackend backend;
    backend.engine.defer_restore.store(true, std::memory_order_relaxed);
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;
    config.concurrent_prefix_cache_max_bytes = 1024;
    config.concurrent_paged_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(backend, tokenizer, config);
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
    remove_test_path(path);

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

    SchedulerPrefixBackend backend;
    backend.engine.malformed_restore.store(true, std::memory_order_relaxed);
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;
    config.concurrent_prefix_cache_max_bytes = 1024;
    config.concurrent_paged_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(backend, tokenizer, config);
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
    remove_test_path(path);

    TEST_ASSERT(done);
    TEST_ASSERT(cache.lookup_candidate({1, 100, 999}, 2).first == -1);
    TEST_ASSERT(backend.engine.discarded ==
                std::vector<PrefixStoreRef>({{1, 2}}));
    const auto stats = cache.stats();
    TEST_ASSERT(stats.restore_attempts == 1);
    TEST_ASSERT(stats.restore_invalidations == 1);
    TEST_ASSERT(stats.restore_stall_us_total == 2500);
}

TEST_CASE(ServerUnitFixture,
          test_scheduler_discards_unrequested_restored_checkpoint) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    SchedulerPrefixBackend backend;
    backend.engine.unrequested_restore.store(true, std::memory_order_relaxed);
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;
    config.concurrent_prefix_cache_max_bytes = 1024;
    config.concurrent_paged_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(backend, tokenizer, config);
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
    remove_test_path(path);

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

    SchedulerPrefixBackend backend;
    ServerConfig config;
    config.arch = "qwen35";
    config.max_ctx = 64;
    config.prefix_cache_cap = 2;
    config.concurrent_prefix_cache_max_bytes = 1024;
    config.concurrent_paged_prefix_cache = true;
    config.admission_coalesce_ms = 0;
    HttpServer server(backend, tokenizer, config);
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
    remove_test_path(path);

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
#endif  // !_WIN32
