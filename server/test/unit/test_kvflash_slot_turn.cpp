// Pure unit tests for the C1.4 per-slot turn bookkeeping (kvflash_slot_turn.h):
// mode-4 plan on/off, first-pass cuts, per-step snapshot/capture flags and the
// mode-5 apply point. No ggml, no GPU.
#include "CppUnitTestFramework.hpp"
#include "kvflash_test_seams.h"
#include "../../src/common/kvflash_slot_turn.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotTurnFixture {};

constexpr int32_t kImStart = 7, kUser = 9, kCt = 64, kQ = 32, kU = 512, kT = 64;

// A prompt of n tokens whose last user turn starts at `user_at` (-1: none).
std::vector<int32_t> prompt_with_user(int n, int user_at) {
    std::vector<int32_t> p((size_t)n, 1);
    if (user_at >= 0 && user_at + 1 < n) { p[(size_t)user_at] = kImStart; p[(size_t)user_at + 1] = kUser; }
    return p;
}
}  // namespace

TEST_CASE(KvflashSlotTurnFixture, cuts_force_r_then_u_cuts_and_a_last_cut_of_at_least_t_queries) {
    // r at 100 lands on a cut end; 100..2020 is 1920 tokens: 512,512,512,384 >= 64 untouched
    CHECK((kvflash_slot_turn_cuts(0, 100, 2020, kU, kT) == std::vector<int>{100, 612, 1124, 1636, 2020}));
    // 0..520 from r: the 8-token last cut would miss queries; its predecessor shrinks to end - 64
    CHECK((kvflash_slot_turn_cuts(0, 0, 520, kU, kT) == std::vector<int>{456, 520}));
    // a turn shorter than T is one cut and captures what there is
    CHECK((kvflash_slot_turn_cuts(1000, 1010, 1040, kU, kT) == std::vector<int>{1010, 1040}));
    // r == kv_offset: no cut before r
    CHECK((kvflash_slot_turn_cuts(1000, 1000, 1600, kU, kT) == std::vector<int>{1512, 1600}));
    // r before kv_offset clamps to it; r at end gives a first pass with no replay tokens
    CHECK((kvflash_slot_turn_cuts(1000, 10, 1600, kU, kT) == std::vector<int>{1512, 1600}));
    CHECK((kvflash_slot_turn_cuts(0, 600, 600, kU, kT) == std::vector<int>{512, 600}));
    CHECK(kvflash_slot_turn_cuts(600, 600, 600, kU, kT).empty());
}

TEST_CASE(KvflashSlotTurnFixture, plan_runs_mode4_only_with_the_profile_and_a_pool_that_does_not_fit) {
    const auto prompt = prompt_with_user(5000, 4800);          // 79 chunks > 32 blocks
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(t.first_pass());
    CHECK(t.r == 4800);
    CHECK(t.end == 5000);
    CHECK(t.reserve_chunks == (5000 - 4800 + kCt - 1) / kCt + 1);
    CHECK(t.snapshot_at_r());
    CHECK(!t.cuts.empty());
    CHECK(t.cuts.back() == 5000);
    CHECK(std::find(t.cuts.begin(), t.cuts.end(), 4800) != t.cuts.end());

    const KvFlashSlotTurn off = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false);
    CHECK(!off.recall());
    CHECK(off.end == 5000);
    CHECK(off.cuts.empty());

    // the user turn starts inside the first chunk: nothing sealed to score, LRU turn
    const KvFlashSlotTurn early = kvflash_slot_turn_plan(prompt_with_user(5000, 40), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(!early.recall());
    CHECK(early.r == -1);
    // 20 chunks + reserve fit 32 blocks: LRU turn
    const KvFlashSlotTurn fits = kvflash_slot_turn_plan(prompt_with_user(1280, 1100), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(!fits.recall());
    CHECK(fits.r == -1);
}

TEST_CASE(KvflashSlotTurnFixture, plan_on_a_resumed_session_rewinds_to_the_new_user_turn_only) {
    auto prompt = prompt_with_user(6000, 5900);
    prompt[(size_t)100] = kImStart; prompt[(size_t)101] = kUser;   // an older user turn inside the session
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 5880, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(t.first_pass());
    CHECK(t.kv_offset == 5880);
    CHECK(t.r == 5900);
    CHECK((t.cuts == std::vector<int>{5900, 6000}));
    // no user turn past the resume point: R is the resume point, nothing to snapshot
    const KvFlashSlotTurn none = kvflash_slot_turn_plan(prompt_with_user(6000, 100), 5880, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(none.first_pass());
    CHECK(none.r == 5880);
    CHECK(!none.snapshot_at_r());
    // nothing to prefill
    CHECK(!kvflash_slot_turn_plan(prompt, 6000, kImStart, kUser, kCt, kQ, kU, kT, true).recall());
}

TEST_CASE(KvflashSlotTurnFixture, next_cut_follows_the_plan_in_the_first_pass_and_u_cuts_in_the_replay) {
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    int pos = 0;
    std::vector<int> ends;
    while (int n = kvflash_slot_turn_next_cut(t, pos, kU)) { pos += n; ends.push_back(pos); }
    CHECK(ends == t.cuts);
    CHECK(kvflash_slot_turn_next_cut(t, 5000, kU) == 0);
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);          // 4800..5000 in one cut
    CHECK(kvflash_slot_turn_next_cut(t, 4000, kU) == std::min(5000, (4000 / kU + 1) * kU) - 4000);
    KvFlashSlotTurn lru;
    lru.end = 1300;
    CHECK(kvflash_slot_turn_next_cut(lru, 0, kU) == kU);
}

TEST_CASE(KvflashSlotTurnFixture, the_step_ending_at_r_snapshots_and_the_step_ending_the_first_pass_captures) {
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    int pos = 0;
    int snaps = 0, caps = 0;
    while (int n = kvflash_slot_turn_next_cut(t, pos, kU)) {
        const KvFlashTurnStep s = kvflash_slot_turn_step(t, pos, n);
        if (s.snapshot) { snaps++; CHECK(pos + n == 4800); }
        if (s.capture) { caps++; CHECK(pos + n == 5000); CHECK(n >= kT); }
        pos += n;
    }
    CHECK(snaps == 1);
    CHECK(caps == 1);
    KvFlashSlotTurn replay = t;
    replay.phase = KvFlashTurnPhase::Replay;
    CHECK(!kvflash_slot_turn_step(replay, 4800, 200).capture);
    CHECK(!kvflash_slot_turn_step(replay, 4288, 512).snapshot);
}

TEST_CASE(KvflashSlotTurnFixture, need_chunks_is_what_the_next_cut_adds_past_the_allocated_chunks) {
    CHECK(kvflash_slot_need_chunks(5022, 142, kCt, 79) == 2);        // chunk 78 is partial; 79 and 80 are new
    CHECK(kvflash_slot_need_chunks(5164, 16, kCt, 81) == 0);         // decode inside chunk 80
    CHECK(kvflash_slot_need_chunks(5180, 16, kCt, 81) == 1);         // the next 16 tokens cross into chunk 81
    CHECK(kvflash_slot_need_chunks(0, 512, kCt, 0) == 8);            // a full first cut
    CHECK(kvflash_slot_need_chunks(100, 0, kCt, 2) == 0);
    CHECK(kvflash_slot_need_chunks(100, 10, 0, 2) == 0);
}

TEST_CASE(KvflashSlotTurnFixture, mode5_plan_applies_the_drafters_picks_at_r_with_no_first_pass_and_no_r_snapshot) {
    const auto prompt = prompt_with_user(5000, 4800);          // 79 chunks > 32 blocks
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, true);
    CHECK(!t.recall());                                         // one prefill, no first pass
    CHECK(t.apply_at == 4800);
    CHECK(t.reserve_chunks == (5000 - 4800 + kCt - 1) / kCt + 1);
    CHECK(t.cuts.empty());
    CHECK(!t.snapshot_at_r());
    CHECK(!kvflash_slot_turn_step(t, 4500, 300).capture);
    CHECK(!kvflash_slot_turn_step(t, 4500, 300).snapshot);
    // the prefill stops at R, then runs plain cuts
    CHECK(kvflash_slot_turn_next_cut(t, 4700, kU) == 100);
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);
    CHECK(!kvflash_slot_turn_apply_due(t, 4799));
    CHECK(kvflash_slot_turn_apply_due(t, 4800));
    CHECK(kvflash_slot_turn_apply_due(t, 4900));

    // resumed at R: due at once; no apply when the profile is off, the pool fits or nothing is sealed before R
    CHECK(kvflash_slot_turn_apply_due(kvflash_slot_turn_plan(prompt, 4800, kImStart, kUser, kCt, kQ, kU, kT, true, true), 4800));
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false, true).apply_at == -1);
    CHECK(kvflash_slot_turn_plan(prompt_with_user(1000, 900), 0, kImStart, kUser, kCt, kQ, kU, kT, true, true).apply_at == -1);
    CHECK(kvflash_slot_turn_plan(prompt_with_user(5000, 40), 0, kImStart, kUser, kCt, kQ, kU, kT, true, true).apply_at == -1);
    // mode 4 plans never carry an apply point
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true).apply_at == -1);
    CHECK(!kvflash_slot_turn_apply_due(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true), 5000));
}

TEST_CASE(KvflashSlotTurnFixture, the_query_tail_accumulates_the_last_64_prompt_queries_across_cuts_like_the_serial_engine) {
    KvFlashSlotTurn t;
    t.phase = KvFlashTurnPhase::FirstPass; t.end = 74;
    // cuts of 40 / 30 / 4 tokens: every one holds some of the last 64 prompt tokens (positions 10..73)
    CHECK(kvflash_slot_turn_step(t, 0, 40, 64).q_rows == 30);
    CHECK(kvflash_slot_turn_step(t, 40, 30, 64).q_rows == 30);
    CHECK(kvflash_slot_turn_step(t, 70, 4, 64).q_rows == 4);
    CHECK(kvflash_slot_turn_step(t, 70, 4, 64).capture);
    CHECK(!kvflash_slot_turn_step(t, 40, 30, 64).capture);
    CHECK(kvflash_slot_turn_step(t, 0, 10, 64).q_rows == 0);          // wholly before the window
    CHECK(kvflash_slot_turn_step(t, 0, 40, 0).q_rows == 0);           // off
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_step(t, 70, 4, 64).q_rows == 0);          // only the first pass captures
    // one query per position (row_size 2: the position twice); the three cuts hand over their last q_rows rows
    std::vector<float> tail;
    auto cut = [&](int p, int n, int q_rows) {
        std::vector<float> rows;
        for (int i = n - q_rows; i < n; i++) { rows.push_back((float)(p + i)); rows.push_back((float)(p + i)); }
        kvflash_q_tail_push(tail, rows.data(), q_rows, 2, 64);
    };
    cut(0, 40, 30); cut(40, 30, 30); cut(70, 4, 4);
    CHECK(tail.size() == 64 * 2);
    for (int i = 0; i < 64; i++) { CHECK(tail[(size_t)2 * i] == (float)(10 + i)); CHECK(tail[(size_t)2 * i + 1] == (float)(10 + i)); }
    // a longer history keeps only the newest 64
    std::vector<float> one(2 * 100); for (int i = 0; i < 100; i++) one[2 * i] = one[2 * i + 1] = (float)i;
    tail.clear(); kvflash_q_tail_push(tail, one.data(), 100, 2, 64);
    CHECK(tail.size() == 128); CHECK(tail[0] == 36.0f); CHECK(tail.back() == 99.0f);
}
