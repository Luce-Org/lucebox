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

// ---- Stage S (PREREG_G3C): spec step commit, settle, gen rung, QK pooling ----

// S-U1: the budget clamp of one spec step (production: spec loop, step 5).
TEST_CASE(KvflashLiveRestoreFixture, spec_commit_n_bonus_is_clamped_by_the_budget) {
    CHECK(spec_commit_n(5, true, 100) == 6);   // 5 accepted + bonus
    CHECK(spec_commit_n(5, true, 5) == 5);     // bonus dropped
    CHECK(spec_commit_n(5, true, 3) == 3);     // accepted tokens beyond budget dropped
    CHECK(spec_commit_n(5, false, 100) == 5);  // fast rollback: no bonus
}

// Where a step leaves the target (production: spec loop, after the emit loop).
// committed = position at the step start. emitted = tokens the emit loop sent.
// A step that ends the request (EOS, cancel, stop string, last budget token)
// or fires a budget close leaves the last emitted token pending, like AR.
TEST_CASE(KvflashLiveRestoreFixture, spec_settle_whole_block_keeps_every_row) {
    auto s = spec_step_settle(1000, 6, 6, false, false);
    CHECK(!s.repair);
    CHECK(s.replay_n == 6);
    CHECK(s.new_committed == 1006);
}

TEST_CASE(KvflashLiveRestoreFixture, spec_settle_a_terminal_step_ends_in_ars_state) {
    // AR: start 190, stop after 2 emissions -> the second token is pending, cur_pos 191.
    const int ar_cur_pos = 190 + 2 - 1;
    auto eos = spec_step_settle(190, 6, 2, false, true);          // EOS or cancel at index 1 of 6
    CHECK(eos.repair);
    CHECK(eos.replay_n == 1);
    CHECK(eos.new_committed == ar_cur_pos);                       // not 192
    auto whole = spec_step_settle(190, 6, 6, false, true);        // last budget token or EOS last in the block
    CHECK(whole.repair);                                          // the whole block was forwarded: undo the last row
    CHECK(whole.new_committed == 190 + 6 - 1);
    auto one = spec_step_settle(190, 6, 1, false, true);          // stop on the first emitted token
    CHECK(one.repair);
    CHECK(one.replay_n == 0);
    CHECK(one.new_committed == 190);
}

TEST_CASE(KvflashLiveRestoreFixture, spec_settle_budget_close_leaves_the_close_token_pending) {
    auto s = spec_step_settle(1000, 6, 3, true, true);
    CHECK(s.repair);                                    // always restored: the block verified the original token
    CHECK(s.replay_n == 2);
    CHECK(s.new_committed == 1002);                     // committed + emitted - 1, not 1003
    auto only = spec_step_settle(1000, 6, 1, true, true);     // close on the very first emitted token
    CHECK(only.repair);
    CHECK(only.replay_n == 0);
    CHECK(only.new_committed == 1000);
}

// S-U2: generation-end rung G. A spec turn and an AR turn that emit the same
// tokens end in the same state: last token pending, history one longer than
// cur_pos. Their checkpoints must be equal.
TEST_CASE(KvflashLiveRestoreFixture, gen_rung_spec_equals_ar) {
    const int prompt = 4000, answer = 300;
    const int ar_cur_pos = prompt + answer - 1;                   // AR: the final token is pending
    const size_t hs = (size_t)(prompt + answer);                  // history holds every emitted id
    // Spec: the last step started 3 tokens before the end and EOS is its 3rd emission.
    auto st = spec_step_settle(prompt + answer - 3, 5, 3, false, true);
    CHECK(st.new_committed == ar_cur_pos);
    const int g_spec = kvflash_gen_ckpt_pos(st.new_committed, hs);
    const int g_ar = kvflash_gen_ckpt_pos(ar_cur_pos, hs);
    CHECK(g_spec == g_ar);
    CHECK(g_spec == 4299);
    // History one short of cur_pos (junk or missing id) refuses the checkpoint.
    CHECK(kvflash_gen_ckpt_pos(st.new_committed, (size_t)st.new_committed - 1) == -1);
}

TEST_CASE(KvflashLiveRestoreFixture, gen_rung_after_early_stop_repair_matches_emitted_ids) {
    auto s = spec_step_settle(4000, 6, 3, false, true);       // EOS at index 2
    Toks hist = seq(4000);
    const Toks out = {11, 12, 13};
    kvflash_history_set_generated(hist, 4000, out, 0);
    CHECK((int)hist.size() == s.new_committed + 1);           // pending token is in the history
    CHECK(kvflash_gen_ckpt_pos(s.new_committed, hist.size()) == s.new_committed);
}

// Decode-time reselection: AR calls it after the g-th generated token when
// g % tau(g) == 0 with tau(g) = max(tau0, (hist0 + g) / 45). A spec step
// advances several tokens, so it must find every boundary it crossed.
namespace {
std::vector<int> ar_reselect_points(size_t hist0, int tau0, int total) {
    std::vector<int> v;
    for (int g = 1; g <= total; g++) {
        const int tau = kvflash_reselect_tau(tau0, hist0 + (size_t)g);
        if (g % tau == 0) v.push_back(g);
    }
    return v;
}
}

TEST_CASE(KvflashLiveRestoreFixture, reselect_points_match_ar_for_any_step_sizes) {
    for (size_t hist0 : {(size_t)100, (size_t)3300, (size_t)30000}) {
        const auto want = ar_reselect_points(hist0, 64, 1500);
        for (int step : {1, 3, 7, 16}) {
            std::vector<int> got;
            for (int prev = 0; prev < 1500; prev += step) {
                const int cur = std::min(prev + step, 1500);
                const auto pts = kvflash_reselect_points(hist0, prev, cur, 64);
                got.insert(got.end(), pts.begin(), pts.end());
            }
            CHECK(got == want);
        }
    }
}

TEST_CASE(KvflashLiveRestoreFixture, reselect_points_a_step_crossing_a_boundary_reports_it) {
    CHECK(kvflash_reselect_points(100, 60, 70, 64) == std::vector<int>({64}));   // crossed, not equal
    CHECK(kvflash_reselect_points(100, 64, 70, 64).empty());                     // already done
    CHECK(kvflash_reselect_points(100, 0, 0, 64).empty());
    CHECK(kvflash_reselect_points(100, 10, 5, 64).empty());                       // no backwards range
    CHECK(kvflash_reselect_points(100, 0, 10, 0).empty());                        // tau <= 0: off
}

// S-U3: QK pooling happens at the settled position only.
TEST_CASE(KvflashLiveRestoreFixture, qk_pool_range_covers_sealed_chunks_only) {
    auto r = kvflash_qk_pool_range(3, 64 * 5 + 10, 64);
    CHECK(r.from == 3);
    CHECK(r.to == 5);                        // chunks 3 and 4 are sealed, 5 is open
    auto none = kvflash_qk_pool_range(5, 64 * 5 + 63, 64);
    CHECK(none.from == none.to);
    auto back = kvflash_qk_pool_range(7, 64 * 5, 64);   // pooled index never moves back
    CHECK(back.from == back.to);
    CHECK(back.to == 7);
}

TEST_CASE(KvflashLiveRestoreFixture, qk_pool_after_a_budget_close_does_not_pool_the_replaced_chunk) {
    // Step from 190, two tokens emitted, the second is replaced by the close
    // token: its row (191) is the last of chunk 2 ... here chunk 2 ends at 192.
    const int ct = 64;
    auto s = spec_step_settle(190, 6, 2, true, true);
    CHECK(s.new_committed == 191);
    auto settled = kvflash_qk_pool_range(2, s.new_committed, ct);
    CHECK(settled.to == 2);                  // chunk 2 is still open: nothing pooled
    // Pooling before the repair (committed + emitted = 192) would have sealed it.
    auto early = kvflash_qk_pool_range(2, 190 + 2, ct);
    CHECK(early.to == 3);
}

TEST_CASE(KvflashLiveRestoreFixture, qk_pool_spec_step_sealing_a_chunk_advances_the_index) {
    const int ct = 64;
    auto s = spec_step_settle(190, 6, 6, false, false);        // crosses the 192 boundary
    auto r = kvflash_qk_pool_range(2, s.new_committed, ct);
    CHECK(r.from == 2);
    CHECK(r.to == 3);
    auto s2 = spec_step_settle(s.new_committed, 6, 6, false, false);
    auto r2 = kvflash_qk_pool_range(r.to, s2.new_committed, ct);
    CHECK(r2.from == r2.to);
}

TEST_CASE(KvflashLiveRestoreFixture, qk_pool_a_16_token_block_seals_at_most_one_chunk) {
    // The pooled chunk must still be resident at seal time: with 64-token
    // chunks and blocks of at most 16 tokens a step never skips a chunk.
    for (int c = 0; c < 200; c++) {
        auto r = kvflash_qk_pool_range(c / 64, c + 16, 64);
        CHECK(r.to - r.from <= 1);
    }
}
