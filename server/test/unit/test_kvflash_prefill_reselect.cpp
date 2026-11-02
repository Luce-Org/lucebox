// Unit test for the LUCE_KVFLASH_PREFILL_RESELECT pure helpers (no GPU): the mode gate, the turn cut,
// the recall pick ranking and the recall budget.

#include "CppUnitTestFramework.hpp"

#include "kvflash_prefill_reselect.h"

#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPrefillReselectFixture {};
std::vector<float> ramp(int n) { std::vector<float> s((size_t)n); for (int i = 0; i < n; i++) s[(size_t)i] = (float)((i * 37) % 101); return s; }
}

// ---- the mode gate: only 0, 4 and 5 exist -----------------------------------------------------------------

TEST_CASE(KvflashPrefillReselectFixture, modes_one_to_three_are_off) {
    std::string why;
    for (int m : {1, 2, 3, 6, -1, 100}) {
        CHECK(kvflash_reselect_mode(m, true, "exact", &why) == 0);
        CHECK(kvflash_reselect_mode(m, true, "pflash2", &why) == 0);
    }
    CHECK(kvflash_reselect_mode(0, true, "exact", &why) == 0);
}
TEST_CASE(KvflashPrefillReselectFixture, modes_four_and_five_need_the_qk_policy_and_their_scorer) {
    std::string why;
    CHECK(kvflash_reselect_mode(4, true, "exact", &why) == 4);
    CHECK(why.empty());
    CHECK(kvflash_reselect_mode(5, true, "pflash2", &why) == 5);
    CHECK(why.empty());
    CHECK(kvflash_reselect_mode(4, false, "exact", &why) == 0);   // LRU policy: no recall, and nothing to complain about
    CHECK(why.empty());
    CHECK(kvflash_reselect_mode(4, true, "pflash2", &why) == 0);  // the scorer must match the mode
    CHECK(why == "LUCE_KVFLASH_PREFILL_RESELECT=4 needs LUCE_KVFLASH_RESELECT_SCORER=exact");
    CHECK(kvflash_reselect_mode(5, true, nullptr, &why) == 0);
    CHECK(why == "LUCE_KVFLASH_PREFILL_RESELECT=5 needs LUCE_KVFLASH_RESELECT_SCORER=pflash2");
    CHECK(kvflash_reselect_mode(4, true, "cosine", nullptr) == 0);   // the cosine scorer is gone
}

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_keeps_features_for_one_device_and_drops_them_under_tp) {
    const KvflashFeatureAsk profile{4, 64};
    const KvflashFeaturePlan one = kvflash_recall_features(1, profile);   // TP = 1: everything stays on
    CHECK(one.reselect_mode == 4 && one.uq_queries == 64 && one.why.empty());
    const KvflashFeaturePlan two = kvflash_recall_features(2, profile);   // TP > 1: recall and UQ off
    CHECK(two.reselect_mode == 0 && two.uq_queries == 0 && !two.why.empty());
    const KvflashFeaturePlan mode5 = kvflash_recall_features(4, {5, 0});
    CHECK(mode5.reselect_mode == 0 && !mode5.why.empty());
    const KvflashFeaturePlan uq_only = kvflash_recall_features(2, {0, 16});
    CHECK(uq_only.uq_queries == 0 && !uq_only.why.empty());
}

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_is_silent_when_nothing_was_asked) {
    CHECK(kvflash_recall_features(2, {0, 0}).why.empty());                  // no recall requested: nothing to report
    CHECK(kvflash_recall_features(1, {0, 0}).reselect_mode == 0);
}

// ---- turn cut -----------------------------------------------------------------------------------------------

TEST_CASE(KvflashPrefillReselectFixture, cuts_with_inserts_sorted) {
    CHECK((kvflash_cuts_with({100, 900}, 500) == std::vector<int>{100, 500, 900}));
    CHECK((kvflash_cuts_with({}, 7) == std::vector<int>{7}));
    CHECK((kvflash_cuts_with({100, 900}, 900) == std::vector<int>{100, 900}));
    CHECK((kvflash_cuts_with({100, 900}, -1) == std::vector<int>{100, 900}));
}

TEST_CASE(KvflashPrefillReselectFixture, last_user_turn_finds_the_last_im_start_user) {
    // 1=im_start 2=user 3=assistant 9=other
    CHECK(kvflash_last_user_turn({9, 1, 2, 9, 1, 3, 9, 1, 2, 9, 9}, 1, 2) == 7);
    CHECK(kvflash_last_user_turn({9, 1, 3, 9}, 1, 2) == -1);       // no user turn
    CHECK(kvflash_last_user_turn({1}, 1, 2) == -1);
    CHECK(kvflash_last_user_turn({1, 2}, -1, 2) == -1);            // unknown id
}

TEST_CASE(KvflashPrefillReselectFixture, mode4_plan_uses_last_user_turn_clamped_to_prefill_start) {
    const std::vector<int> toks = {9, 1, 2, 9, 1, 3, 9, 1, 2, 9, 9};
    KvFlashMode4Plan p = kvflash_mode4_plan(toks, 100, 1, 2);
    CHECK(p.run);
    CHECK(p.r == 107);                       // kv_offset + local index 7
    CHECK(p.reserve_chunks == -1);           // derived from the tokens left to prefill
    // No user turn in the (delta) tokens: fall back to the prefill start.
    p = kvflash_mode4_plan({9, 1, 3, 9}, 100, 1, 2);
    CHECK(p.run);
    CHECK(p.r == 100);
    // Cold prefill with the user turn at the very start: R4 = 0 (zero state).
    p = kvflash_mode4_plan({1, 2, 9}, 0, 1, 2);
    CHECK(p.run);
    CHECK(p.r == 0);
    CHECK(!kvflash_mode4_plan({}, 0, 1, 2).run);
}

TEST_CASE(KvflashPrefillReselectFixture, pool_fits_predicate_edges) {
    // 250 chunks + 6 reserve == 256 blocks: still fits (strict no-op).
    CHECK(kvflash_pool_fits(250 * 64, 64, 6, 256));
    // One more reserve chunk, or one more token past the chunk edge: the pool is full.
    CHECK(!kvflash_pool_fits(250 * 64, 64, 7, 256));
    CHECK(!kvflash_pool_fits(250 * 64 + 1, 64, 6, 256));
    CHECK(kvflash_pool_fits(10, 64, 1, 256));
    CHECK(kvflash_pool_fits(0, 64, 0, 256));
    CHECK(!kvflash_pool_fits(100, 0, 1, 256));   // unusable chunk size: never claim it fits
}

// ---- pick ranking -------------------------------------------------------------------------------------------

TEST_CASE(KvflashPrefillReselectFixture, pick_ranks_blocks_by_summed_score_best_first) {
    // 8 blocks of 2 chunks; block b scores 2 * (b == 5 ? 9 : b == 3 ? 7 : 1)
    std::vector<float> sc(16, 1.0f);
    sc[10] = sc[11] = 9.0f;
    sc[6] = sc[7] = 7.0f;
    CHECK((kvflash_recall_pick(sc, 16, 2, 2, 2) == std::vector<int>{5, 3}));
    CHECK((kvflash_recall_pick(sc, 16, 1, 2, 2) == std::vector<int>{5}));
    CHECK(kvflash_recall_pick(sc, 16, 0, 2, 2).empty());
}
TEST_CASE(KvflashPrefillReselectFixture, pick_ties_go_to_the_newer_block) {
    const std::vector<float> flat(40, 1.0f);
    CHECK((kvflash_recall_pick(flat, 40, 3, 2, 2) == std::vector<int>{19, 18, 17}));
}
TEST_CASE(KvflashPrefillReselectFixture, pick_only_sees_blocks_inside_the_floor_and_the_window) {
    std::vector<float> sc(60, 1.0f);
    for (int c = 0; c < 10; c++) sc[(size_t)c] = 100.0f;      // the sink block scores loudest and is never eligible
    for (int c = 50; c < 60; c++) sc[(size_t)c] = 100.0f;     // the live window is never eligible either
    const auto ids = kvflash_recall_pick(sc, 50, 20, 10);
    CHECK((ids == std::vector<int>{4, 3, 2, 1}));             // blocks 1..4 of 10 chunks; block 0 is the sink, 5 the window
    // a score vector shorter than the window bounds the candidates too
    CHECK((kvflash_recall_pick(std::vector<float>(30, 1.0f), 50, 20, 10) == std::vector<int>{2, 1}));
    CHECK(kvflash_recall_pick(sc, 50, 5, 0).empty());         // unusable block size
}
TEST_CASE(KvflashPrefillReselectFixture, pick_with_a_floor_skips_the_pinned_chunks) {
    std::vector<float> sc(400, 1.0f);
    for (int c = 0; c < 35; c++) sc[(size_t)c] = 100.0f;      // the pinned preamble scores loudest
    const int pinned = 35, unit = 10;
    const auto ids = kvflash_recall_pick(sc, 300, 50, unit, kvflash_recall_first_chunk(pinned));
    REQUIRE(!ids.empty());
    for (int b : ids) CHECK(b * unit >= pinned);
    // without the floor the loud preamble blocks are picked, which is what it prevents
    CHECK(kvflash_recall_pick(sc, 300, 50, unit).size() > ids.size());
    CHECK(kvflash_recall_first_chunk(0) == 10);
    CHECK(kvflash_recall_first_chunk(7) == 10);
}

// ---- recall budget ------------------------------------------------------------------------------------------

TEST_CASE(KvflashPrefillReselectFixture, tail_is_the_pool_less_recall_reserve_and_the_two_spares) {
    CHECK(kvflash_recall_tail_chunks(256, 60, 3, 0) == 256 - 1 - 60 - 3 - 1);
    CHECK(kvflash_recall_tail_chunks(20, 60, 5, 0) == 0);     // clamps at zero
}
TEST_CASE(KvflashPrefillReselectFixture, the_preamble_reserve_never_shrinks_the_history_budget) {
    for (int k : {1, 6, 10})
        for (int reserve : {2, 9, 30}) {
            const int off = kvflash_recall_tail_chunks(256, 10 * k, reserve, 0);
            for (int rc : {1, 112, 200}) CHECK(kvflash_recall_tail_chunks(256 + rc, 10 * k, reserve, rc) == off);
        }
}

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_derives_tail_window_and_picks) {
    const auto s = ramp(400);
    KvFlashRecallIn in;
    in.n_pool_blocks = 256; in.cur_chunk = 400; in.recall_chunks = 60; in.unit = 10; in.reserve = 3;
    const KvFlashRecallPlan p = kvflash_recall_plan(s, in);
    CHECK(p.k_units == 6);
    CHECK(p.first_chunk == 10);
    CHECK(p.tail == kvflash_recall_tail_chunks(256, 60, 3, 0));
    CHECK(p.first_window == 400 - p.tail);
    CHECK(p.picked == kvflash_recall_pick(s, p.first_window, 6, 10, 10));
    for (int b : p.picked) CHECK((b + 1) * 10 <= p.first_window);   // a picked block never reaches into the window
}
TEST_CASE(KvflashPrefillReselectFixture, recall_plan_with_the_preamble_pinned) {
    const auto s = ramp(400);
    KvFlashRecallIn in;
    in.n_pool_blocks = 256 + 96; in.cur_chunk = 400; in.recall_chunks = 60; in.unit = 10; in.reserve = 3;
    in.pin_reserve_chunks = 96; in.pinned_chunks = 96;
    const KvFlashRecallPlan p = kvflash_recall_plan(s, in);
    KvFlashRecallIn off = in;
    off.n_pool_blocks = 256; off.pin_reserve_chunks = 0; off.pinned_chunks = 0;
    CHECK(p.tail == kvflash_recall_plan(s, off).tail);        // the history budget is the feature-off one
    CHECK(p.first_chunk == 96);
    for (int b : p.picked) CHECK(b * 10 >= 96);              // no pinned chunk is a recall candidate
}
TEST_CASE(KvflashPrefillReselectFixture, recall_plan_without_recall_picks_nothing) {
    KvFlashRecallIn in;
    in.n_pool_blocks = 64; in.cur_chunk = 300; in.recall_chunks = 0; in.unit = 10; in.reserve = 3;
    const KvFlashRecallPlan p = kvflash_recall_plan(ramp(300), in);
    CHECK(p.k_units == 0);
    CHECK(p.picked.empty());
    CHECK(p.tail == 64 - 1 - 3 - 1);
}
TEST_CASE(KvflashPrefillReselectFixture, recall_plan_unit_other_than_ten) {
    KvFlashRecallIn in;
    in.n_pool_blocks = 256; in.cur_chunk = 600; in.recall_chunks = 60; in.unit = 2; in.reserve = 3;
    const KvFlashRecallPlan p = kvflash_recall_plan(ramp(600), in);
    CHECK(p.k_units == 30);
    CHECK((int)p.picked.size() == 30);
    for (int b : p.picked) CHECK(b * 2 >= 10);               // the 10-chunk sink floor holds at every unit
}
