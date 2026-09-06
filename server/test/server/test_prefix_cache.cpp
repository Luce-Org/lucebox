#include "server/prefix_cache.h"
#include "server/pin_friendly_prompt.h"
#include "server/freeze_history.h"
#include "server/prompt_normalize.h"
#include "server/http_server.h"
#include "gguf.h"
#include <nlohmann/json.hpp>
#include "support/test_assert.h"
#include <cmath>
#include <string>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
using namespace dflash::common;

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

    const std::string path = "/tmp/dflash_test_deepseek_markers.gguf";
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

    // Completed assistant turn followed by the next user marker. The reusable
    // boundary includes that role marker, matching the server's other chat
    // families and leaving only the new user content for suffix prefill.
    const std::vector<int32_t> prompt = {
        1, 100, 3, 101, 4, 102, 2, 3, 103, 4,
    };
    TEST_ASSERT(find_all_boundaries(prompt, markers) ==
                std::vector<int>({8}));
    unlink(path.c_str());
}

TEST_CASE(ServerUnitFixture, test_prefix_cache_reserves_disk_staging_slot) {
    const std::string path = write_deepseek_marker_tokenizer_fixture();
    Tokenizer tokenizer;
    TEST_ASSERT(tokenizer.load_from_gguf(path.c_str()));

    PrefixCache cache(PrefixCache::MAX_SLOTS, tokenizer);
    TEST_ASSERT(cache.stats().capacity == PrefixCache::MAX_CACHE_SLOTS);
    TEST_ASSERT(PrefixCache::MAX_CACHE_SLOTS == ModelBackend::kMaxSlots - 1);

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
