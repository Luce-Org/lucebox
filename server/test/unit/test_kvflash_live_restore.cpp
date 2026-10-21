#include "CppUnitTestFramework.hpp"
#include "../../src/qwen35/prefill_helpers.h"

namespace {
struct KvflashLiveRestoreFixture {};
using Toks = std::vector<int32_t>;
Toks seq(int n, int base = 0) {
    Toks t((size_t)n);
    for (int i = 0; i < n; i++) t[(size_t)i] = base + i;
    return t;
}
const std::vector<int> kRungs = {1024, 2048, 4096};
}

using namespace luce::common;

TEST_CASE(KvflashLiveRestoreFixture, identical_prompt_resumes_below_the_end) {
    // History = prompt + generated answer; same prompt again.
    Toks hist = seq(5000);
    Toks prompt = seq(4500);
    auto d = kvflash_live_restore_decide(prompt, hist, 5000, kRungs);
    CHECK(d.L == 4500);
    CHECK(d.C == 4096);
    CHECK(d.replay_from == 4096);
    CHECK(d.discard_above == 4500);
}

TEST_CASE(KvflashLiveRestoreFixture, rung_at_prompt_end_still_replays_last_token) {
    Toks hist = seq(4096);
    Toks prompt = seq(4096);
    auto d = kvflash_live_restore_decide(prompt, hist, 4096, kRungs);
    CHECK(d.L == 4096);
    CHECK(d.C == 2048);   // 4096 would leave nothing to prefill
    CHECK(d.replay_from == 2048);
}

TEST_CASE(KvflashLiveRestoreFixture, extended_prompt_new_turn) {
    Toks hist = seq(4300);              // prompt 4100 + 200 generated
    Toks prompt = seq(4300);
    for (int i = 0; i < 300; i++) prompt.push_back(90000 + i);   // new turn
    auto d = kvflash_live_restore_decide(prompt, hist, 4300, kRungs);
    CHECK(d.L == 4300);
    CHECK(d.C == 4096);
    CHECK(d.replay_from == 4096);
}

TEST_CASE(KvflashLiveRestoreFixture, divergence_in_the_middle) {
    Toks hist = seq(5000);
    Toks prompt = seq(5000);
    prompt[3000] = -7;
    auto d = kvflash_live_restore_decide(prompt, hist, 5000, kRungs);
    CHECK(d.L == 3000);
    CHECK(d.C == 2048);
    CHECK(d.discard_above == 3000);   // the 4096 rung is stale
}

TEST_CASE(KvflashLiveRestoreFixture, divergence_below_first_rung) {
    Toks hist = seq(5000);
    Toks prompt = seq(5000);
    prompt[500] = -7;
    auto d = kvflash_live_restore_decide(prompt, hist, 5000, kRungs);
    CHECK(d.L == 500);
    CHECK(d.C == -1);
    CHECK(d.replay_from == 0);
    CHECK(d.discard_above == 500);
}

TEST_CASE(KvflashLiveRestoreFixture, lcp_exactly_on_a_rung) {
    Toks hist = seq(5000);
    Toks prompt = seq(5000);
    prompt[2048] = -7;   // tokens [0,2048) match
    auto d = kvflash_live_restore_decide(prompt, hist, 5000, kRungs);
    CHECK(d.L == 2048);
    CHECK(d.C == 2048);
    CHECK(d.replay_from == 2048);
}

TEST_CASE(KvflashLiveRestoreFixture, empty_history_and_empty_ladder) {
    auto d = kvflash_live_restore_decide(seq(3000), {}, 0, kRungs);
    CHECK(d.L == 0);
    CHECK(d.C == -1);
    CHECK(d.replay_from == 0);
    auto e = kvflash_live_restore_decide(seq(3000), seq(3000), 3000, {});
    CHECK(e.L == 3000);
    CHECK(e.C == -1);
    auto f = kvflash_live_restore_decide({}, seq(10), 10, kRungs);
    CHECK(f.C == -1);
}

TEST_CASE(KvflashLiveRestoreFixture, history_capped_at_cur_pos) {
    // History longer than what the KV holds: tokens past cur_pos are unproven.
    Toks hist = seq(6000);
    Toks prompt = seq(6000);
    auto d = kvflash_live_restore_decide(prompt, hist, 3000, kRungs);
    CHECK(d.L == 3000);
    CHECK(d.C == 2048);
}

// ---- S9: generation-end checkpoint at an off-grid position ----------------

TEST_CASE(KvflashLiveRestoreFixture, gen_rung_off_grid_is_picked_when_prompt_extends_history) {
    // G = 4300 is not a multiple of 512; the next turn re-renders the whole
    // history and appends a new user turn.
    const std::vector<int> rungs = {1024, 2048, 4096, 4300};
    Toks hist = seq(4301);                       // 4300 in KV + 1 pending token
    Toks prompt = seq(4301);
    for (int i = 0; i < 80; i++) prompt.push_back(90000 + i);
    auto d = kvflash_live_restore_decide(prompt, hist, 4300, rungs);
    CHECK(d.L == 4300);
    CHECK(d.C == 4300);
    CHECK(d.replay_from == 4300);
    CHECK((int)prompt.size() - d.C == 81);       // only the new tokens replay
    CHECK(d.discard_above == 4300);              // the gen rung survives
}

TEST_CASE(KvflashLiveRestoreFixture, gen_rung_dropped_when_template_rewrites_the_answer) {
    const std::vector<int> rungs = {1024, 2048, 4096, 4300};
    Toks hist = seq(4300);
    Toks prompt = seq(4400);
    prompt[4250] = -5;                           // the re-rendered answer differs
    auto d = kvflash_live_restore_decide(prompt, hist, 4300, rungs);
    CHECK(d.L == 4250);
    CHECK(d.C == 4096);
    CHECK(d.discard_above == 4250);              // 4300 is stale and gets erased
}

TEST_CASE(KvflashLiveRestoreFixture, gen_ckpt_position_needs_history_to_cover_cur_pos) {
    CHECK(kvflash_gen_ckpt_pos(4300, 4301) == 4300);   // one pending token is fine
    CHECK(kvflash_gen_ckpt_pos(4300, 4300) == 4300);
    CHECK(kvflash_gen_ckpt_pos(4300, 4000) == -1);     // history lacks generated ids
    CHECK(kvflash_gen_ckpt_pos(0, 10) == -1);
    CHECK(kvflash_gen_ckpt_pos(-3, 10) == -1);
}

TEST_CASE(KvflashLiveRestoreFixture, early_stop_inside_a_verified_block_needs_a_state_repair) {
    CHECK(spec_early_stop_needs_repair(4, 2, false));    // EOS or cancel mid-block
    CHECK(spec_early_stop_needs_repair(4, 1, false));
    CHECK(!spec_early_stop_needs_repair(4, 4, false));   // whole block emitted (EOS last)
    CHECK(!spec_early_stop_needs_repair(1, 1, false));   // plain decode step
    CHECK(!spec_early_stop_needs_repair(4, 2, true));    // budget close rewinds on its own
}

// ---- spec decode: committed ids are appended to the history ---------------

TEST_CASE(KvflashLiveRestoreFixture, history_append_takes_generated_ids_after_entry) {
    Toks hist = seq(100);
    const Toks out = {7, 8, 9, 10};              // out_tokens, nothing before entry
    kvflash_history_set_generated(hist, 100, out, 0);
    CHECK(hist.size() == 104);
    CHECK(hist[100] == 7);
    CHECK(hist[103] == 10);
}

TEST_CASE(KvflashLiveRestoreFixture, history_append_is_idempotent_over_ar_fallback) {
    // The AR fallback inside spec decode already pushed some ids; resyncing
    // from out_tokens must not duplicate them.
    Toks hist = seq(100);
    hist.push_back(7); hist.push_back(8);
    const Toks out = {7, 8, 9};
    kvflash_history_set_generated(hist, 100, out, 0);
    CHECK(hist.size() == 103);
    CHECK(hist[102] == 9);
}

TEST_CASE(KvflashLiveRestoreFixture, history_append_skips_tokens_before_entry) {
    Toks hist = seq(50);
    const Toks out = {1, 2, 3, 4};               // 2 came from an earlier phase
    kvflash_history_set_generated(hist, 50, out, 2);
    CHECK(hist.size() == 52);
    CHECK(hist[50] == 3);
    CHECK(hist[51] == 4);
}

TEST_CASE(KvflashLiveRestoreFixture, history_append_nothing_generated_truncates_to_base) {
    Toks hist = seq(60);
    const Toks out = {};
    kvflash_history_set_generated(hist, 55, out, 0);
    CHECK(hist.size() == 55);
}

TEST_CASE(KvflashLiveRestoreFixture, turn_rung_catches_the_final_answer_rerender) {
    // Previous prompt Pend = 5000, turn rung at Pend-4 = 4996, gen ckpt G = 5300.
    // The new prompt re-renders the old assistant turn without the think block,
    // so it matches the history only up to 4996 and then diverges.
    Toks hist = seq(5300);
    Toks prompt = seq(4996);
    for (int i = 0; i < 400; i++) prompt.push_back(90000 + i);
    const std::vector<int> rungs = {1024, 4608, 4996, 5300};
    auto d = kvflash_live_restore_decide(prompt, hist, 5300, rungs);
    CHECK(d.L == 4996);
    CHECK(d.C == 4996);
    CHECK(d.replay_from == 4996);
    CHECK(d.discard_above == 4996);   // the gen rung 5300 is dropped
    // without the turn rung it falls back to the grid
    auto e = kvflash_live_restore_decide(prompt, hist, 5300, {1024, 4608, 5300});
    CHECK(e.C == 4608);
}
