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

TEST_CASE(KvflashLadderFixture, evict_nothing_within_budget) {
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {8192, 2}, {16384, 3}};
    CHECK(kvflash_ladder_evict(e, 300, 100, 1024).empty());
    CHECK(kvflash_ladder_evict({}, 0, 100, -1).empty());
}

TEST_CASE(KvflashLadderFixture, evict_lru_under_budget_protecting_head_and_newest) {
    // budget holds 3 of 5. Head (1024) and newest (tick 9) are kept;
    // LRU among the rest goes first.
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {8192, 5}, {16384, 2}, {24576, 4}, {32768, 9}};
    CHECK((kvflash_ladder_evict(e, 300, 100, 1024) == Pos{16384, 24576}));
}

TEST_CASE(KvflashLadderFixture, evict_stops_when_only_protected_remain) {
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {8192, 2}, {16384, 3}};
    // budget fits one; two protected (head, newest) survive regardless.
    CHECK((kvflash_ladder_evict(e, 100, 100, 1024) == Pos{8192}));
    CHECK((kvflash_ladder_evict(e, 0, 100, 1024) == Pos{8192}));
    // no head protection when head is unknown: only the newest is protected.
    CHECK((kvflash_ladder_evict(e, 0, 100, -1) == Pos{1024, 8192}));
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
