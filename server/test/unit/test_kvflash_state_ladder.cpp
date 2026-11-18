#include "CppUnitTestFramework.hpp"
#include "../../src/qwen35/prefill_helpers.h"

namespace {
struct KvflashLadderFixture {};
using Pos = std::vector<int>;
}

using namespace luce::common;

TEST_CASE(KvflashLadderFixture, positions_head_stride_and_prompt_end) {
    // head 1300 -> 1024; every 8192; grid point <= end (20000 -> 19968).
    CHECK((kvflash_state_ladder_positions(1300, 20000, 512, 8192) == Pos{1024, 8192, 16384, 19968}));
    // prompt end exactly on the grid; no duplicate with the stride rung.
    CHECK((kvflash_state_ladder_positions(1024, 16384, 512, 8192) == Pos{1024, 8192, 16384}));
    // no head known: that rung is left out.
    CHECK((kvflash_state_ladder_positions(-1, 9000, 512, 8192) == Pos{8192, 8704}));
    // head below one grid cell rounds to 0: left out. Short prompt: nothing.
    CHECK((kvflash_state_ladder_positions(100, 300, 512, 8192) == Pos{}));
    // head beyond the prompt is dropped.
    CHECK((kvflash_state_ladder_positions(5000, 2000, 512, 8192) == Pos{1536}));
}

TEST_CASE(KvflashLadderFixture, pick_greatest_not_above_L) {
    CHECK(kvflash_ladder_pick({}, 5000) == -1);
    const Pos c{1024, 8192, 16384};
    CHECK(kvflash_ladder_pick(c, 1000) == -1);     // below every C
    CHECK(kvflash_ladder_pick(c, 1024) == 1024);   // exactly on C
    CHECK(kvflash_ladder_pick(c, 8191) == 1024);   // between
    CHECK(kvflash_ladder_pick(c, 8192) == 8192);
    CHECK(kvflash_ladder_pick(c, 99999) == 16384); // above all
    CHECK(kvflash_ladder_pick(c, 0) == -1);
}

TEST_CASE(KvflashLadderFixture, evict_lru_under_budget_protecting_head_and_newest) {
    // budget holds 3 of 5. Head (1024) and newest (tick 9) are kept;
    // LRU among the rest goes first.
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {8192, 5}, {16384, 2}, {24576, 4}, {32768, 9}};
    CHECK((kvflash_ladder_evict(e, 300, 100, 1024) == Pos{16384, 24576}));
}

TEST_CASE(KvflashLadderFixture, evict_keeps_the_newest_gen_rung) {
    // The gen rung (4300) is the newest; even with a budget of zero it stays,
    // and the older grid rungs go LRU first, the head last of all.
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {4096, 2}, {8192, 3}, {4300, 9}};
    CHECK((kvflash_ladder_evict(e, 200, 100, 1024) == Pos{4096, 8192}));
    CHECK((kvflash_ladder_evict(e, 0, 100, 1024) == Pos{4096, 8192}));
}

TEST_CASE(KvflashLadderFixture, evict_stops_when_only_protected_remain) {
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {8192, 2}, {16384, 3}};
    // Within budget nothing is evicted.
    CHECK(kvflash_ladder_evict(e, 300, 100, 1024).empty());
    // budget fits one; two protected (head, newest) survive regardless.
    CHECK((kvflash_ladder_evict(e, 100, 100, 1024) == Pos{8192}));
    CHECK((kvflash_ladder_evict(e, 0, 100, 1024) == Pos{8192}));
    // no head protection when head is unknown: only the newest is protected.
    CHECK((kvflash_ladder_evict(e, 0, 100, -1) == Pos{1024, 8192}));
}

// ---- S9b: turn rung at the end of the previous prompt minus the suffix ----

namespace {
constexpr int32_t kStart = 900, kAsst = 901, kNl = 902;
}

TEST_CASE(KvflashLadderFixture, positions_include_the_turn_rung) {
    // 18813-token prompt, turn rung at 18809: grid rungs plus the off-grid one.
    CHECK((kvflash_state_ladder_positions(1300, 18813, 512, 8192, 18809) == Pos{1024, 8192, 16384, 18432, 18809}));
    // absent / out of range: unchanged
    CHECK((kvflash_state_ladder_positions(1300, 18813, 512, 8192, -1) == kvflash_state_ladder_positions(1300, 18813, 512, 8192)));
    CHECK((kvflash_state_ladder_positions(1300, 18813, 512, 8192, 99999) == kvflash_state_ladder_positions(1300, 18813, 512, 8192)));
    // a turn rung on a grid point does not duplicate
    CHECK((kvflash_state_ladder_positions(-1, 9000, 512, 8192, 8704) == Pos{8192, 8704}));
}

TEST_CASE(KvflashLadderFixture, turn_cuts_all_headers_in_order) {
    std::vector<int32_t> p{kStart, kAsst, kNl, 1, 2, kStart, kAsst, kNl, 3, kStart, kAsst, kNl, 4, 5};
    CHECK((kvflash_turn_cuts(p, kStart, kAsst, kNl) == Pos{3, 8, 12}));
    // a user header is not a cut
    CHECK((kvflash_turn_cuts({1, 2, kStart, 5, 6, kStart, kAsst, kNl, 70, 71, 72, 73}, kStart, kAsst, kNl) == Pos{8}));
}

TEST_CASE(KvflashLadderFixture, turn_cuts_none_and_trailing_header) {
    CHECK((kvflash_turn_cuts({}, kStart, kAsst, kNl) == Pos{}));
    CHECK((kvflash_turn_cuts({1, 2, 3}, kStart, kAsst, kNl) == Pos{}));
    // user header only, or a broken header
    CHECK((kvflash_turn_cuts({kStart, 5, kNl, 7}, kStart, kAsst, kNl) == Pos{}));
    CHECK((kvflash_turn_cuts({kStart, kAsst, 5, 7}, kStart, kAsst, kNl) == Pos{}));
    CHECK((kvflash_turn_cuts({1, 2, 3}, -1, -1, -1) == Pos{}));
    // header ends the prompt: no cut at size
    CHECK((kvflash_turn_cuts({1, kStart, kAsst, kNl}, kStart, kAsst, kNl) == Pos{}));
    // an earlier header still counts when the last one ends the prompt
    CHECK((kvflash_turn_cuts({kStart, kAsst, kNl, 7, kStart, kAsst, kNl}, kStart, kAsst, kNl) == Pos{3}));
}

// LUCE_KVFLASH_SESSION: default on, only "0" turns the session stack off.
TEST_CASE(KvflashLadderFixture, session_switch_defaults_on_and_zero_disables) {
    CHECK(kvflash_session_flag(nullptr));
    CHECK(kvflash_session_flag("1"));
    CHECK(kvflash_session_flag(""));
    CHECK_FALSE(kvflash_session_flag("0"));
    CHECK(kvflash_session_flag("00"));

    unsetenv("LUCE_KVFLASH_SESSION");
    CHECK(kvflash_session_enabled());
    setenv("LUCE_KVFLASH_SESSION", "0", 1);
    CHECK_FALSE(kvflash_session_enabled());
    setenv("LUCE_KVFLASH_SESSION", "1", 1);
    CHECK(kvflash_session_enabled());
    unsetenv("LUCE_KVFLASH_SESSION");
}
