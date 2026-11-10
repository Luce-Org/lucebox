// Stage C-4: the pure planning of a DFlash2 chain round over KVFlash slots. A spec lane plans
// `width` pager rows from its committed length (the candidates live in scratch rows, but the
// accepted prefix must find its rows already allocated); an AR peer plans one. The column the graph
// reads must cover every chunk the plan allocated, and a lane the pager could not fit sends the
// whole round to the AR path (never an error).
#include "CppUnitTestFramework.hpp"
#include "kvflash_slot_world.h"
#include "../../src/common/kvflash_step_driver.h"
#include "../../src/common/kvflash_chain_plan.h"
#include "../../src/common/kvflash_engine_step.h"

#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashChainPlanFixture {};

constexpr int kMaxCtx = 4096;

bool same_work(const KvFlashStepWork & w, int slot, int64_t start, int tokens) {
    return w.slot == slot && w.kv_start == start && w.tokens == tokens;
}
}  // namespace

TEST_CASE(KvflashChainPlanFixture, a_spec_lane_plans_its_chain_width_from_the_committed_length_and_an_ar_lane_plans_one_row) {
    const ChainTablePlan p = chain_table_plan({{0, 100, true, 16}, {1, 70, false, 1}}, kMaxCtx);
    REQUIRE(p.work.size() == 2);
    CHECK(same_work(p.work[0], 0, 100, 16));
    CHECK(same_work(p.work[1], 1, 70, 1));
    REQUIRE(p.cover_len.size() == 2);
    CHECK(p.cover_len[0] == std::make_pair(0, 116));     // the committed 100 plus the 16 rows allocated for the chain
    CHECK(p.cover_len[1] == std::make_pair(1, 71));      // an AR row: seq_len = position + 1
}

TEST_CASE(KvflashChainPlanFixture, the_planned_width_is_clamped_to_the_room_left_in_the_context_and_is_never_zero) {
    CHECK(same_work(chain_table_plan({{0, kMaxCtx - 3, true, 16}}, kMaxCtx).work[0], 0, kMaxCtx - 3, 3));
    CHECK(same_work(chain_table_plan({{0, kMaxCtx - 16, true, 16}}, kMaxCtx).work[0], 0, kMaxCtx - 16, 16));
    CHECK(chain_table_plan({{0, kMaxCtx, true, 16}}, kMaxCtx).work[0].tokens == 1);   // the engine's own `accepted == 0` error owns this case
    CHECK(chain_table_plan({{0, 5, true, 8}}, kMaxCtx).work[0].tokens == 8);          // a narrower verify width plans fewer rows
    CHECK(chain_table_plan({{0, 5, true, 16}}, kMaxCtx).cover_len[0].second == 21);
}

TEST_CASE(KvflashChainPlanFixture, a_duplicate_slot_or_a_bad_lane_makes_the_plan_invalid) {
    CHECK(!chain_table_plan({{0, 10, true, 16}, {0, 10, false, 1}}, kMaxCtx).valid);
    CHECK(!chain_table_plan({{-1, 10, true, 16}}, kMaxCtx).valid);
    CHECK(!chain_table_plan({{0, -1, true, 16}}, kMaxCtx).valid);
    CHECK(!chain_table_plan({{0, 10, true, 1}}, kMaxCtx).valid);       // a spec lane carries at least a root and one draft
    CHECK(chain_table_plan({{1, 10, true, 2}, {0, 3, false, 1}}, kMaxCtx).valid);
}

TEST_CASE(KvflashChainPlanFixture, a_chain_round_runs_only_when_the_pager_ran_every_planned_slot) {
    const ChainTablePlan p = chain_table_plan({{0, 100, true, 16}, {1, 70, false, 1}}, kMaxCtx);
    CHECK(chain_round_verdict(p, {0, 1}) == ChainRoundVerdict::Chain);
    CHECK(chain_round_verdict(p, {1, 0}) == ChainRoundVerdict::Chain);          // order is not the plan's
    CHECK(chain_round_verdict(p, {0}) == ChainRoundVerdict::FallbackAr);        // the AR lane did not fit
    CHECK(chain_round_verdict(p, {1}) == ChainRoundVerdict::FallbackAr);        // a spec lane did not fit: BlocksExhausted is an AR round
    CHECK(chain_round_verdict(p, {}) == ChainRoundVerdict::FallbackAr);
    CHECK(chain_round_verdict(chain_table_plan({{0, 1, true, 4}}, kMaxCtx), {0, 3}) == ChainRoundVerdict::Chain);   // extra ran slots are harmless
    ChainTablePlan bad;
    CHECK(bad.valid);                                                         // a default plan is the empty valid one
    CHECK(chain_round_verdict(chain_table_plan({{0, 1, true, 1}}, kMaxCtx), {0}) == ChainRoundVerdict::FallbackAr);   // an invalid plan never runs a chain
}

namespace {
constexpr int kN = 2, kQ = 12, kChunk = 64;
struct PagerWorld {
    KvFlashTestArena arena;
    KvFlashTestSlots s;
    std::unique_ptr<KvFlashStepDriver> drv;
    PagerWorld() {
        arena.set_host_wait([] {});
        s = kvflash_test_slots(kN, kQ, kvflash_test_config(2, kChunk), &arena);
        drv = std::make_unique<KvFlashStepDriver>(kvflash_test_raw(s), arena, 8);
    }
};
}  // namespace

TEST_CASE(KvflashChainPlanFixture, a_column_covering_the_plan_names_the_chunk_the_chain_rows_spill_into) {
    // Slot 0 holds 60 committed tokens (chunk 0). A width-16 chain at position 60 allocates rows 60..75,
    // which opens chunk 1. The graph reads only seq_len = 60, but the NEXT step (an AR row at 70, or the
    // accepted tokens) finds chunk 1 already resident, so the epoch does not move and table_stale stays
    // false: the column written now must already name chunk 1.
    PagerWorld w;
    REQUIRE(w.drv->run_step({{0, 0, 60}}).ok);
    w.drv->finish_step();
    int32_t col[kQ * 4];
    w.drv->write_table(0, col, 1, 16);

    const ChainTablePlan plan = chain_table_plan({{0, 60, true, 16}}, kMaxCtx);
    const auto ran = w.drv->run_step(plan.work);
    REQUIRE(ran.ok);
    CHECK(chain_round_verdict(plan, ran.ran) == ChainRoundVerdict::Chain);
    CHECK(w.drv->table_stale(0));
    const int cover = plan.cover_len[0].second;
    const int chunks = (cover + kChunk - 1) / kChunk;
    CHECK(chunks == 2);
    int32_t full[kQ * 4];
    std::fill(full, full + kQ * 4, -7);
    w.drv->write_table(0, full, chunks, 16);
    w.drv->finish_step();
    CHECK(full[4] == w.s[0]->block_of(1) * 4);           // chunk 1's first block is named, at the committed length 60
    CHECK(full[4] >= 0);

    // the accepted tokens now extend into chunk 1 and an AR round plans one row there: no new chunk, no stale table
    REQUIRE(w.drv->run_step({{0, 70, 1}}).ok);
    w.drv->finish_step();
    CHECK(!w.drv->table_stale(0));

    // a column written at the committed length alone would have left chunk 1 null for that next step
    int32_t narrow[kQ * 4];
    std::fill(narrow, narrow + kQ * 4, -7);
    PagerWorld v;
    REQUIRE(v.drv->run_step({{0, 0, 60}}).ok);
    v.drv->finish_step();
    REQUIRE(v.drv->run_step(plan.work).ok);
    v.drv->write_table(0, narrow, (60 + kChunk - 1) / kChunk, 16);   // cover = committed length: the bug this plan avoids
    v.drv->finish_step();
    CHECK(narrow[4] == -7);
}
