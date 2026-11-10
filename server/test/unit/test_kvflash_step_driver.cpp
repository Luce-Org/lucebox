// KvFlashStepDriver: preflight and the step's pins and allocations over real slot pagers
// (no-tensor path, CPU). The test plays the engine: per step it names the rows
// each slot appends, then reads the table columns the graph would be given.
#include "CppUnitTestFramework.hpp"
#include "kvflash_slot_world.h"
#include "../../src/common/kvflash_step_driver.h"
#include "../../src/common/kvflash_engine_step.h"
#include "../../src/common/concurrency/kvflash_slot_kv.h"
#include "../../src/common/concurrency/seq_slot_manager.h"

#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashStepDriverFixture {};

constexpr int kN = 2, kQ = 12, kChunk = 64;

struct World {
    KvFlashTestArena arena;
    KvFlashTestSlots s;
    std::unique_ptr<KvFlashStepDriver> drv;
    explicit World(int headroom = 8) {
        arena.set_host_wait([] {});
        s = kvflash_test_slots(kN, kQ, kvflash_test_config(2, kChunk), &arena);
        const auto raw = kvflash_test_raw(s);
        drv = std::make_unique<KvFlashStepDriver>(raw, arena, headroom);
    }
};

// The engine's step: pins and allocations, then (after the graph and readback) the pins end.
KvFlashStepDriver::Result run(World & w, const std::vector<KvFlashStepWork> & work) {
    auto r = w.drv->run_step(work);
    w.drv->finish_step();
    return r;
}
}  // namespace

TEST_CASE(KvflashStepDriverFixture, fitting_slots_run_pins_expire_and_the_table_is_stale_only_when_the_epoch_moved) {
    World w(0);                                                       // no headroom: nothing is over quota
    int32_t col[kQ * 4];
    auto r = w.drv->run_step({{0, 0, 8 * kChunk}, {1, 0, 8 * kChunk}});   // two aligned 512 cuts
    CHECK(w.s[0]->is_pinned(0));                                  // pins outlive the plan: the graph still reads these chunks
    w.drv->finish_step();
    CHECK(r.ok);
    CHECK((r.ran == std::vector<int>{0, 1}));
    for (int i = 0; i < kN; i++) {
        CHECK(w.s[i]->resident_blocks() == 8);
        CHECK(!w.s[i]->is_pinned(0));                             // finish_step expired the step pins
        CHECK(w.drv->table_stale(i));
        w.drv->write_table(i, col, 8, 16);
        CHECK(!w.drv->table_stale(i));
        CHECK(col[0] == w.s[i]->block_of(0) * 4);
    }
    r = run(w, {{0, 8 * kChunk, 1}});           // the first row of a new chunk
    CHECK(r.ok);
    CHECK(w.drv->table_stale(0));
    w.drv->write_table(0, col, 9, 16);
    r = run(w, {{0, 8 * kChunk + 1, 1}});       // inside the resident chunk: nothing to upload
    CHECK(r.ok);
    CHECK(!w.drv->table_stale(0));
    CHECK(!w.drv->table_stale(1));                                // the peer did not run
}

TEST_CASE(KvflashStepDriverFixture, a_slot_that_does_not_fit_idles_while_its_page_out_settles_and_peers_run) {
    World w;
    REQUIRE(run(w, {{0, 0, kQ * kChunk}, {1, 0, kChunk}}).ok);   // slot 0 holds its whole slice
    const int before = w.s[0]->resident_blocks();
    REQUIRE(before == kQ);
    auto r = run(w, {{0, kQ * kChunk, kChunk}, {1, kChunk, kChunk}});
    CHECK((r.ran == std::vector<int>{1}));                        // slot 0 needs a block that is not free: idle
    CHECK(w.s[0]->resident_blocks() == kQ - 1);                   // D2: it pages out the one block it lacks
    CHECK(w.s[1]->resident_blocks() == 2);                        // the peer was not held back
    r = run(w, {{0, kQ * kChunk, kChunk}});     // the recovery page-out settled at step start
    CHECK(r.ok);
    CHECK((r.ran == std::vector<int>{0}));
    CHECK(w.s[0]->block_of(kQ) >= 0);
    int holes = 0;
    for (int c = 0; c < kQ; c++) holes += w.s[0]->block_of(c) < 0;
    CHECK(holes == 1);                                            // LRU evicted the oldest evictable chunk only
}

TEST_CASE(KvflashStepDriverFixture, abort_expires_the_step_pins_of_every_slot) {
    World w;
    REQUIRE(run(w, {{0, 0, 3 * kChunk}}).ok);
    w.s[0]->pin_step_range(0, kChunk);
    REQUIRE(w.s[0]->is_pinned(0));
    w.drv->abort_step();
    CHECK(!w.s[0]->is_pinned(0));
}

TEST_CASE(KvflashStepDriverFixture, a_run_past_the_quota_evicts_and_decodes) {
    World w;
    int64_t pos[kN] = {0, 0};
    for (int step = 0; step < 40; step++) {
        std::vector<KvFlashStepWork> work;
        for (int i = 0; i < kN; i++) work.push_back({i, pos[i], kChunk});
        const auto r = run(w, work);
        REQUIRE(r.ok);
        for (int i : r.ran) pos[i] += work[(size_t)i].tokens;
    }
    CHECK(pos[0] > kQ * kChunk);                                  // it ran past the quota: eviction happened
}

// A step has no fence: a slot that runs takes free blocks whose page-out has completed, so its allocation queues no
// copy and no zeroing. The page-outs are queued by settle and by an idle slot, a step ahead.
TEST_CASE(KvflashStepDriverFixture, a_slot_that_runs_allocates_free_blocks_and_queues_no_page_work) {
    World w(4);
    int64_t pos = 0;
    int ran = 0, idled = 0;
    for (int step = 0; step < 60; step++) {
        const KvFlashStats before = w.s[0]->stats();
        const auto r = w.drv->run_step({{0, pos, 3 * kChunk}});
        REQUIRE(r.ok);
        if (r.ran.empty()) {
            idled++;
        } else {
            ran++;
            pos += 3 * kChunk;
            CHECK(w.s[0]->stats().page_outs == before.page_outs);
            CHECK(w.s[0]->stats().page_ins == before.page_ins);
            CHECK(w.s[0]->pending_free_blocks() == 0);
        }
        w.drv->finish_step();
        if (step % 2) REQUIRE(w.drv->settle({0}) >= 0);            // every other step skips settle: the slot must idle, not page in the call
    }
    CHECK(pos > 4 * kQ * kChunk);                                  // far past the quota
    CHECK(ran > 0);
    CHECK(idled > 0);
}

TEST_CASE(KvflashStepDriverFixture, preflight_keeps_slots_with_enough_completed_blocks_in_order) {
    const std::vector<KvFlashPreflightSlot> in = {{3, 8, 8}, {1, 7, 8}, {0, 0, 1}, {2, 5, 1}};
    CHECK((kvflash_preflight_step(in) == std::vector<int>{3, 2}));
    CHECK(kvflash_preflight_step({}).empty());
    CHECK((kvflash_preflight_step({{4, 0, 0}}) == std::vector<int>{4}));   // a slot that needs nothing always runs
}

TEST_CASE(KvflashStepDriverFixture, lru_victims_are_oldest_first_and_skip_sink_tail_and_pinned_chunks) {
    World w;
    REQUIRE(run(w, {{0, 0, 8 * kChunk}}).ok);       // chunks 0..7, last_use ascending
    auto v = w.s[0]->lru_victims(3);
    CHECK((v == std::vector<int>{1, 2, 3}));                          // 0 is the sink, 5..7 are the tail window
    w.s[0]->pin_step_range(kChunk, 2 * kChunk);                       // chunk 1 pinned
    CHECK((w.s[0]->lru_victims(2) == std::vector<int>{2, 3}));
    CHECK((w.s[0]->lru_victims(99) == std::vector<int>{2, 3, 4}));   // 5..7 are the tail window, 0 the sink, 1 pinned
    CHECK(w.s[0]->lru_victims(0).empty());
}

TEST_CASE(KvflashStepDriverFixture, settle_pages_a_full_slot_down_to_the_usable_quota_and_never_past_the_headroom) {
    World w(4);                                                       // usable quota 12 - 4 = 8
    REQUIRE(run(w, {{0, 0, kQ * kChunk}, {1, 0, 3 * kChunk}}).ok);
    CHECK(w.s[0]->resident_blocks() == kQ);
    CHECK(w.drv->settle({0, 1}) == 4);                                // slot 0: 12 -> 8; slot 1 is under quota
    CHECK(w.s[0]->resident_blocks() == 8);
    CHECK(w.s[1]->resident_blocks() == 3);
    CHECK(w.drv->settle({0, 1}) == 0);                                // idempotent
    auto r = run(w, {{0, kQ * kChunk, 4 * kChunk}});   // the settled page-outs complete at step start
    CHECK((r.ran == std::vector<int>{0}));                            // a 4-chunk cut fits at once
}

TEST_CASE(KvflashStepDriverFixture, a_cut_that_lacks_several_blocks_queues_them_all_in_one_idle_step) {
    World w(4);
    REQUIRE(run(w, {{0, 0, kQ * kChunk}}).ok);
    auto r = run(w, {{0, kQ * kChunk, 4 * kChunk}});
    CHECK(r.ran.empty());                                             // full slice: the cut idles
    CHECK(w.s[0]->resident_blocks() == kQ - 4);                       // but all four missing blocks (headroom 4) are on their way out
    r = run(w, {{0, kQ * kChunk, 4 * kChunk}});
    CHECK((r.ran == std::vector<int>{0}));                            // one step later it runs, not four
}

TEST_CASE(KvflashStepDriverFixture, kvflash_step_limits_grant_whole_cuts_to_each_slice_and_a_whole_number_of_cuts_per_step) {
    StepPlanLimits base{8, 2048, 4096, 64, 1.0, 8192};
    StepPlanLimits l = kvflash_step_limits(base);
    CHECK(l.max_prefill_tokens_per_sequence == 512);
    CHECK(l.prefill_allocation_quantum == 512);
    CHECK(l.max_prefill_tokens_total == 4096);
    base = {8, 512, 1300, 512, 0.5, 8192};                        // a total that is not a multiple of U
    l = kvflash_step_limits(base);
    CHECK(l.max_prefill_tokens_total == 1024);
    base = {8, 512, 100, 512, 0.5, 8192};
    CHECK(kvflash_step_limits(base).max_prefill_tokens_total == 512);   // never below one cut
    // the planner then gives every selected slice exactly U
    const auto slices = plan_prefill_slices({{0, 1}, {1, 2}, {2, 3}}, kvflash_step_limits({8, 2048, 4096, 64, 1.0, 8192}));
    CHECK(slices.size() == 3);
    for (const auto & sl : slices) CHECK(sl.max_tokens == 512);
}

TEST_CASE(KvflashStepDriverFixture, kvflash_column_entries_cover_the_kv_length_in_four_entries_per_chunk_and_clamp_to_the_table) {
    CHECK(kvflash_column_entries(1, 8192) == 4);
    CHECK(kvflash_column_entries(64, 8192) == 4);
    CHECK(kvflash_column_entries(65, 8192) == 8);
    CHECK(kvflash_column_entries(512, 8192) == 32);
    CHECK(kvflash_column_entries(200000, 8192) == 8192);
}

TEST_CASE(KvflashStepDriverFixture, page_out_checked_refuses_stale_generations_and_pinned_sink_and_tail_chunks) {
    World w;
    REQUIRE(run(w, {{0, 0, 8 * kChunk}}).ok);                      // chunks 0..7
    KvFlashSlotPager & p = *w.s[0];
    const uint64_t g = p.generation();
    CHECK(!p.page_out_checked(0, g));                              // sink
    CHECK(!p.page_out_checked(7, g));                              // tail window
    p.pin_step_range(2 * kChunk, 3 * kChunk);
    CHECK(!p.page_out_checked(2, g));                              // pinned this step
    p.end_step();
    CHECK(p.page_out_checked(2, g));                               // evictable once the pin expired
    CHECK(!p.page_out_checked(2, g));                              // already out
    p.truncate(4 * kChunk);                                        // a rewind makes ops planned against the longer sequence stale
    CHECK(p.generation() != g);
    CHECK(!p.page_out_checked(3, g));
    w.drv->abort_step();                                           // an abort invalidates what was planned before it too
    const uint64_t g2 = p.generation();
    w.drv->abort_step();
    CHECK(p.generation() != g2);
}

// The engine's step() on the CPU: SeqSlotManager over KvFlashSlotKv, the driver around it.
TEST_CASE(KvflashStepDriverFixture, engine_loop_prefills_in_whole_cuts_decodes_past_the_quota_without_idling_and_releases_everything) {
    World w(4);                                                       // quota 12, usable 8
    std::vector<KvFlashSlotPager *> raw{w.s[0].get(), w.s[1].get()};
    KvFlashHostLedger ledger(int64_t(1) << 30, 0);
    KvFlashSlotKvConfig kc; kc.quota_chunks = kQ; kc.headroom_blocks = 4; kc.max_ctx = 4096; kc.chunk_bytes = 1000;
    KvFlashSlotKv kv(raw, ledger, kc);
    SeqSlotManager slots(kv, 4096);
    const int prompts[2] = {1500, 700};
    const int gen = 300;
    for (int i = 0; i < 2; i++) {
        kv.set_next_total_tokens(prompts[i] + gen);
        const auto a = slots.admit(100 + i, std::vector<int32_t>(prompts[i], 7), SamplerCfg{});
        REQUIRE(a.status == SeqEngine::AdmitResult::Status::admitted);
    }
    std::vector<int> generated(2, 0), cut_ends[2];
    int idle_prefill_steps = 0;
    for (int step = 0; step < 400 && (generated[0] < gen || generated[1] < gen); step++) {
        std::vector<KvFlashStepWork> work;
        std::vector<int> cut(2, 0);
        for (int i = 0; i < 2; i++) {
            const SeqSlot & sq = slots.slot(i);
            if (sq.decoding() && generated[i] < gen) work.push_back({i, sq.cur_pos, 1});
            else if (sq.prefilling()) {
                cut[i] = std::min(sq.prompt_len, (sq.cur_pos / kKvFlashCutU + 1) * kKvFlashCutU) - sq.cur_pos;
                work.push_back({i, sq.cur_pos, cut[i]});
            }
        }
        std::vector<int> growth(2, 0);
        for (int i = 0; i < 2; i++) growth[i] = slots.slot(i).decoding() ? 1 : 0;
        if (growth[0] + growth[1] > 0) CHECK(slots.reserve_decode(growth));   // evicted history is still owned: no extra blocks
        const auto r = w.drv->run_step(work);
        REQUIRE(r.ok);
        for (const auto & wk : work) {
            const bool ran = std::find(r.ran.begin(), r.ran.end(), wk.slot) != r.ran.end();
            const int i = wk.slot;
            if (slots.slot(i).decoding()) {
                CHECK(ran);                                             // settle keeps the headroom: a decode row never idles
                if (!ran) continue;
                REQUIRE(slots.append_token(i, 7).ok);
                slots.commit_step(i);
                if (++generated[i] == gen) slots.retire(i);                // the scheduler's finish
            } else if (!ran) {
                idle_prefill_steps++;
            } else {
                REQUIRE(slots.append_prefill(i, cut[i]).ok);
                cut_ends[i].push_back(slots.slot(i).cur_pos);
                if (slots.slot(i).cur_pos == slots.slot(i).prompt_len) slots.commit_prefill(i);
            }
        }
        w.drv->finish_step();
        CHECK(w.drv->settle({0, 1}) >= 0);
        CHECK(w.s[0]->resident_blocks() <= kQ);
    }
    CHECK(generated[0] == gen);
    CHECK(generated[1] == gen);
    CHECK((cut_ends[0] == std::vector<int>{512, 1024, 1500}));        // cuts depend on position only
    CHECK((cut_ends[1] == std::vector<int>{512, 700}));
    slots.retire(0);
    slots.retire(1);
    CHECK(ledger.used_bytes() == 0);                                  // every exit releases the reservation
    CHECK(w.s[0]->resident_blocks() == 0);
}

TEST_CASE(KvflashStepDriverFixture, a_full_slot_runs_a_step_that_allocates_nothing_and_keeps_every_chunk) {
    World w(4);
    REQUIRE(run(w, {{0, 0, kQ * kChunk}}).ok);                        // all 12 resident, as after a recall apply
    auto r = run(w, {{0, 0, 1}});                                     // a row inside a resident chunk: needs 0 blocks
    CHECK((r.ran == std::vector<int>{0}));                            // full is the normal state, nothing is paged out
    CHECK(w.s[0]->resident_blocks() == kQ);
}

TEST_CASE(KvflashStepDriverFixture, complete_all_drains_the_deferred_frees_before_a_rewind) {
    World w;
    REQUIRE(run(w, {{0, 0, kQ * kChunk}}).ok);                     // slot 0 full
    REQUIRE(w.s[0]->page_out_checked(1, w.s[0]->generation()));    // deferred: the block waits for completion
    CHECK(w.s[0]->pending_free_blocks() == 1);
    w.drv->complete_all();
    CHECK(w.s[0]->pending_free_blocks() == 0);
    w.s[0]->set_deferred_free(false);                              // the apply's page-outs free at once
    REQUIRE(w.s[0]->page_out_checked(2, w.s[0]->generation()));
    CHECK(w.s[0]->pending_free_blocks() == 0);
    w.s[0]->set_deferred_free(true);
}
