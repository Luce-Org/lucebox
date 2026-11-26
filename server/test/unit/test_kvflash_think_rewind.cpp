// Unit test for Mode 5 Think-then-Rewind-to-R pure planning helpers (no GPU).

#include "CppUnitTestFramework.hpp"

#include "kvflash_think_rewind.h"

#include <set>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashThinkRewindFixture {};
}

TEST_CASE(KvflashThinkRewindFixture, half_budget_union_and_early_eos_gates) {
    KvFlashRecallIn rin;
    rin.n_pool_blocks = 256;
    rin.cur_chunk = 1000;
    rin.recall_chunks = 60;  // 30 units of 2
    rin.unit = 2;
    rin.reserve = 4;
    rin.pin_reserve_chunks = 18;
    rin.pinned_chunks = 17;

    std::vector<int> m5_picked;
    for (int i = 0; i < 30; ++i) m5_picked.push_back(50 + i);  // 50..79
    std::set<int> resident(m5_picked.begin(), m5_picked.end());

    std::vector<float> m4_scores(2000, 0.0f);
    for (int i = 0; i < 30; ++i) m4_scores[(200 + i) * 2] = 50.0f - (float)i;

    KvFlashThinkRewindCfg cfg;
    auto is_ev = [&](int u) { return resident.count(u) == 0; };

    // Half the recall budget (15) from M5, remaining 15 from exact-mass M4.
    const auto plan = kvflash_think_rewind_plan(
        m5_picked, m4_scores, rin, 2000, /*early_eos=*/false, cfg, is_ev);
    CHECK(plan.should_rewind);
    CHECK(plan.keep_m5 == 15);
    CHECK(plan.union_units.size() == 30);
    CHECK(plan.new_evicted == 15);
    CHECK(plan.top_evicted == 8);
    // Verify union_scores encodes union_units verbatim for kvflash_recall_plan.
    const auto check_plan = kvflash_recall_plan(plan.union_scores, rin);
    CHECK(check_plan.picked == plan.union_units);

    // Early-EOS gate: 1 top-8 evicted (< 2) does NOT rewind, while normal trigger DOES.
    std::set<int> res_mostly_hit(m5_picked.begin(), m5_picked.end());
    for (int i = 1; i < 30; ++i) res_mostly_hit.insert(plan.m4_top[(size_t)i]);
    auto is_ev_1 = [&](int u) { return res_mostly_hit.count(u) == 0; };

    const auto eos_skip = kvflash_think_rewind_plan(
        m5_picked, m4_scores, rin, 2000, /*early_eos=*/true, cfg, is_ev_1);
    CHECK(eos_skip.top_evicted == 1);
    CHECK(!eos_skip.should_rewind);

    const auto norm_go = kvflash_think_rewind_plan(
        m5_picked, m4_scores, rin, 2000, /*early_eos=*/false, cfg, is_ev_1);
    CHECK(norm_go.should_rewind);

    // Early-EOS with 2+ top-8 evicted DOES rewind (catches short hallucinations).
    res_mostly_hit.erase(plan.m4_top[1]);
    const auto eos_catch = kvflash_think_rewind_plan(
        m5_picked, m4_scores, rin, 2000, /*early_eos=*/true, cfg, is_ev_1);
    CHECK(eos_catch.top_evicted == 2);
    CHECK(eos_catch.should_rewind);

    // No rewind when every union unit is already resident.
    auto none_ev = [](int) { return false; };
    const auto all_res = kvflash_think_rewind_plan(
        m5_picked, m4_scores, rin, 2000, /*early_eos=*/false, cfg, none_ev);
    CHECK(all_res.new_evicted == 0);
    CHECK(!all_res.should_rewind);
    CHECK(all_res.union_scores.empty());

    // An evicted M5 unit retained in the union still counts toward new_evicted and triggers rewind.
    auto only_m5_0_ev = [&](int u) { return u == m5_picked[0]; };
    const auto m5_ev_plan = kvflash_think_rewind_plan(
        m5_picked, m4_scores, rin, 2000, /*early_eos=*/false, cfg, only_m5_0_ev);
    CHECK(m5_ev_plan.new_evicted == 1);
    CHECK(m5_ev_plan.should_rewind);
}
