// KvFlashSlotKv: the SlotKv over slot pagers, no-tensor path (CPU).
#include "CppUnitTestFramework.hpp"
#include "kvflash_slot_world.h"
#include "kvflash_test_seams.h"
#include "../../src/common/concurrency/kvflash_slot_kv.h"
#include "../../src/common/concurrency/seq_slot_manager.h"

#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotKvFixture {};

constexpr int kN = 2, kQ = 12, kChunk = 64;
constexpr int64_t kMaxCtx = 4096, kChunkBytes = 1000;

struct World {
    KvFlashTestArena arena;
    KvFlashTestSlots s;
    KvFlashTestLedger ledger;
    std::unique_ptr<KvFlashSlotKv> kv;
    explicit World(int64_t host_budget = 1000000) : ledger(host_budget) {
        arena.set_host_wait([] {});
        s = kvflash_test_slots(kN, kQ, kvflash_test_config(2, kChunk), &arena);
        const auto raw = kvflash_test_raw(s);
        KvFlashSlotKvConfig cfg; cfg.quota_chunks = kQ; cfg.headroom_blocks = 2; cfg.max_ctx = kMaxCtx; cfg.chunk_bytes = kChunkBytes;
        kv = std::make_unique<KvFlashSlotKv>(raw, ledger, cfg);
    }
};

SeqEngine::AdmitResult admit_tokens(SeqSlotManager & mgr, KvFlashSlotKv & kv, uint64_t id, int n, int gen = 60) {
    kv.set_next_total_tokens(n + gen);
    return mgr.admit(id, std::vector<int32_t>((size_t)n, 1), SamplerCfg{});
}
}  // namespace

TEST_CASE(KvflashSlotKvFixture, acquire_gives_distinct_slots_then_exhausts_and_release_recycles) {
    World w;
    PagedKvSequenceHandle a, b, c;
    CHECK(w.kv->acquire_reserved(1, 256, a) == PagedKvStatus::Ok);
    CHECK(w.kv->acquire_reserved(2, 256, b) == PagedKvStatus::Ok);
    CHECK(a.slot != b.slot);
    CHECK(w.kv->acquire_reserved(3, 256, c) == PagedKvStatus::SequenceSlotsExhausted);
    CHECK(w.kv->acquire_reserved(1, 256, c) == PagedKvStatus::DuplicateRequest);
    CHECK(w.kv->release(a) == PagedKvStatus::Ok);
    CHECK(w.kv->release(a) == PagedKvStatus::StaleHandle);       // a handle outlives nothing
    CHECK(w.kv->acquire_reserved(3, 256, c) == PagedKvStatus::Ok);
    CHECK(c.slot == a.slot);
    CHECK(c.generation != a.generation);
    CHECK(w.kv->owned_block_count(a, *std::make_unique<uint32_t>()) == PagedKvStatus::StaleHandle);
}

TEST_CASE(KvflashSlotKvFixture, host_ledger_decides_busy_vs_permanent_and_release_gives_it_back) {
    World w(/*budget=*/10 * kChunkBytes);                         // ten chunks of host backing in total
    PagedKvSequenceHandle a, b;
    CHECK(w.kv->acquire_reserved(1, 6 * kChunk, a) == PagedKvStatus::Ok);           // 6 chunks reserved
    CHECK(w.ledger.used_bytes() == 6 * kChunkBytes);
    CHECK(w.kv->acquire_reserved(2, 6 * kChunk, b) == PagedKvStatus::BlocksExhausted);   // transient: BusyHost
    CHECK(w.kv->acquire_reserved(2, 11 * kChunk, b) == PagedKvStatus::CapacityExceeded);  // permanent: HostTooSmall
    CHECK(w.kv->acquire_reserved(2, (uint32_t)kMaxCtx + kChunk + 1, b) == PagedKvStatus::InvalidArgument);   // PromptTooLong
    CHECK(w.kv->release(a) == PagedKvStatus::Ok);
    CHECK(w.ledger.used_bytes() == 0);
    CHECK(w.kv->acquire_reserved(2, 6 * kChunk, b) == PagedKvStatus::Ok);
}

TEST_CASE(KvflashSlotKvFixture, append_rows_stay_inside_the_slots_own_slice) {
    World w;
    PagedKvSequenceHandle a, b;
    REQUIRE(w.kv->acquire_reserved(1, 512, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->acquire_reserved(2, 512, b) == PagedKvStatus::Ok);
    for (PagedKvSequenceHandle h : {a, b}) {
        const PagedKvAppendResult r = w.kv->append(h, 200);
        REQUIRE(r);
        REQUIRE(r.write_slots.size() == 200);
        const KvFlashSlice sl = w.s[h.slot]->slice();
        for (uint32_t i = 0; i < 200; ++i) {
            CHECK(r.write_slots[i].logical_position == i);
            CHECK((int64_t)r.write_slots[i].physical_token_index >= sl.row_lo);
            CHECK((int64_t)r.write_slots[i].physical_token_index < sl.row_hi);
            CHECK((int)r.write_slots[i].physical_token_index == w.s[h.slot]->slot_of(i));
        }
    }
}

TEST_CASE(KvflashSlotKvFixture, full_slice_queues_a_page_out_and_succeeds_after_completion) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, (kQ + 4) * kChunk, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->append(a, kQ * kChunk));                        // slice full: Q chunks resident
    const PagedKvAppendResult blocked = w.kv->append(a, kChunk);
    CHECK(blocked.status == PagedKvStatus::BlocksExhausted);      // victim paged out, block in flight
    CHECK(blocked.write_slots.empty());
    CHECK(w.s[a.slot]->resident_blocks() <= kQ);
    CHECK(w.arena.complete_all());
    const PagedKvAppendResult ok = w.kv->append(a, kChunk);       // retry: same positions, nothing was consumed
    REQUIRE(ok);
    CHECK(ok.first.logical_position == (uint32_t)(kQ * kChunk));
    int null_entries = 0;
    for (int c = 0; c < kQ + 1; ++c) null_entries += w.s[a.slot]->block_of(c) < 0;
    CHECK(null_entries >= 1);                                     // the evicted chunk is a hole (-1) in the table column
    uint32_t owned = 0;
    CHECK(w.kv->owned_block_count(a, owned) == PagedKvStatus::Ok);
    CHECK(owned == (uint32_t)kQ + 4);                             // owns its reservation; evicted history is still owned
}

TEST_CASE(KvflashSlotKvFixture, append_beyond_the_reserved_capacity_is_refused) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 2 * kChunk, a) == PagedKvStatus::Ok);
    CHECK(w.kv->append(a, 2 * kChunk));
    CHECK(w.kv->append(a, 1).status == PagedKvStatus::BlocksExhausted);
    CHECK(w.kv->reserve_capacity(a, 2 * kChunk) == PagedKvStatus::Ok);
    CHECK(w.kv->reserve_capacity(a, 3 * kChunk) == PagedKvStatus::Ok);                // the manager's rolling decode page
    CHECK(w.kv->reserve_capacity(a, 3 * kChunk + 1) == PagedKvStatus::BlocksExhausted);
}

TEST_CASE(KvflashSlotKvFixture, append_room_is_the_reserved_rows_left_so_a_chain_never_overshoots_the_last_block) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 100, a) == PagedKvStatus::Ok);   // 2 blocks = 128 rows
    CHECK(w.kv->append_room((int)a.slot) == 2 * kChunk);
    CHECK(w.kv->append(a, 90));
    CHECK(w.kv->append_room((int)a.slot) == 2 * kChunk - 90);
    CHECK(w.kv->append_room(-1) == 0);                                // no such slot
    CHECK(w.kv->release(a) == PagedKvStatus::Ok);
    CHECK(w.kv->append_room((int)a.slot) == 0);                       // not live
}

// P1-2: a chain round plans rows for its whole width before verify; the rows it never accepts must not leave a chunk behind,
// or a park records more chunks than ceil(len/64) and a later adopt refuses the bundle.
TEST_CASE(KvflashSlotKvFixture, trim_to_len_drops_the_chunk_a_chain_round_planned_but_never_accepted) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 4 * kChunk, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->append(a, 60));
    KvFlashSlotPager & p = *w.s[a.slot];
    for (int row = 60; row < 76; ++row) REQUIRE(p.slot_for(row) >= 0);     // the plan: width 16 from committed 60 opens chunk 1
    REQUIRE(p.n_chunks() == 2);
    REQUIRE(w.kv->append(a, 1));                                          // the root alone is accepted: length 61
    CHECK(w.kv->trim_to_len((int)a.slot));
    CHECK(p.n_chunks() == 1);
    CHECK(!w.kv->trim_to_len((int)a.slot));                               // idempotent
    CHECK(!w.kv->trim_to_len(-1));
    // a park records exactly ceil(61/64) chunks (the bundle adopt that checks the count is PR 10's)
    REQUIRE(w.kv->park(a) == PagedKvStatus::Ok);
    CHECK(p.n_chunks() == 1);
}

TEST_CASE(KvflashSlotKvFixture, trim_to_len_keeps_a_chunk_the_accepted_rows_filled) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 4 * kChunk, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->append(a, 60));
    KvFlashSlotPager & p = *w.s[a.slot];
    for (int row = 60; row < 76; ++row) REQUIRE(p.slot_for(row) >= 0);
    REQUIRE(w.kv->append(a, 10));                                         // accepted through row 69: chunk 1 is in use
    CHECK(!w.kv->trim_to_len((int)a.slot));
    CHECK(p.n_chunks() == 2);
}

TEST_CASE(KvflashSlotKvFixture, rollback_and_clear_rewind_the_pager) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 6 * kChunk, a) == PagedKvStatus::Ok);
    const PagedKvAppendResult first = w.kv->append(a, 130);
    REQUIRE(first);
    CHECK(w.s[a.slot]->resident_blocks() == 3);
    CHECK(w.kv->rollback_append(a, 70) == PagedKvStatus::Ok);     // 60 tokens left: one chunk
    CHECK(w.s[a.slot]->resident_blocks() == 1);
    CHECK(w.kv->rollback_append(a, 61) == PagedKvStatus::InvalidArgument);
    const PagedKvAppendResult again = w.kv->append(a, 10);
    REQUIRE(again);
    CHECK(again.first.logical_position == 60);
    CHECK(w.kv->clear(a) == PagedKvStatus::Ok);
    CHECK(w.s[a.slot]->resident_blocks() == 0);
    CHECK(w.kv->append(a, 1).first.logical_position == 0);
}

TEST_CASE(KvflashSlotKvFixture, seq_slot_manager_runs_over_it_unchanged) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    CHECK(mgr.slot_count() == kN);
    const std::vector<int32_t> prompt(300, 1);
    const SeqEngine::AdmitResult r1 = mgr.admit(1, prompt, SamplerCfg{});
    const SeqEngine::AdmitResult r2 = mgr.admit(2, prompt, SamplerCfg{});
    REQUIRE(r1.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(r2.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(r1.slot != r2.slot);
    CHECK(mgr.admit(3, prompt, SamplerCfg{}).status == SeqEngine::AdmitResult::Status::busy);
    for (const auto & r : {r1, r2}) {
        const SeqSlotManager::PrefillChunk ch = mgr.append_prefill(r.slot, 300);
        REQUIRE(ch.ok);
        REQUIRE(ch.rows.size() == 300);
        const KvFlashSlice sl = w.s[(size_t)r.slot]->slice();      // manager slot == kv slot: same first-free order
        for (int64_t row : ch.rows) { CHECK(row >= sl.row_lo); CHECK(row < sl.row_hi); }
    }
    mgr.retire(r1.slot);
    CHECK(w.ledger.reservation_count() == 1);
    CHECK(mgr.admit(4, prompt, SamplerCfg{}).status == SeqEngine::AdmitResult::Status::admitted);
}

TEST_CASE(KvflashSlotKvFixture, failed_ubatch_rolls_back_to_the_last_committed_boundary) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, (kQ + 4) * kChunk, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->append(a, 130));                                // committed
    const PagedKvAppendResult ubatch = w.kv->append(a, 100);      // the ubatch whose graph then fails
    REQUIRE(ubatch);
    CHECK(w.s[a.slot]->resident_blocks() == 4);
    CHECK(w.kv->rollback_append(a, 100) == PagedKvStatus::Ok);
    CHECK(w.s[a.slot]->resident_blocks() == 3);
    CHECK(w.s[a.slot]->block_of(3) < 0);                          // its table entry is null again
    const PagedKvAppendResult again = w.kv->append(a, 100);
    REQUIRE(again);
    CHECK(again.first.logical_position == 130);
    CHECK(again.first.physical_token_index == ubatch.first.physical_token_index);   // same rows, nothing leaked
}

TEST_CASE(KvflashSlotKvFixture, retire_frees_the_slice_the_reservation_and_nulls_the_table_column) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 512, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->append(a, 300));
    int32_t col[5 * 4];
    w.s[a.slot]->fill_block_table(col, 5, 16);
    CHECK(col[0] >= 0);
    CHECK(w.kv->release(a) == PagedKvStatus::Ok);
    CHECK(w.s[a.slot]->resident_blocks() == 0);
    CHECK(w.s[a.slot]->stats().host_bytes == 0);
    CHECK(w.ledger.reservation_count() == 0);
    for (int32_t & e : col) e = 7;
    w.s[a.slot]->fill_block_table(col, 5, 16);
    for (int32_t e : col) CHECK(e == -1);                         // the column is all null, not stale
}

TEST_CASE(KvflashSlotKvFixture, kvflash_slots_are_not_parkable) {
    World w;
    CHECK(!w.kv->parkable());
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    CHECK(!mgr.parkable());
    const SeqEngine::AdmitResult r = mgr.admit(1, std::vector<int32_t>(300, 1), SamplerCfg{});
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 300).ok);
    CHECK(!mgr.detach_kv(r.slot));                                // parking would clear the slice
    CHECK(!mgr.evict_for_recompute(r.slot, 5));
    CHECK(mgr.is_active(r.slot));
}

TEST_CASE(KvflashSlotKvFixture, the_total_token_bound_sets_the_host_reservation_and_the_slot_capacity_for_one_acquire) {
    World w;
    PagedKvSequenceHandle a, b;
    w.kv->set_next_total_tokens(7 * kChunk);
    REQUIRE(w.kv->acquire_reserved(1, 2 * kChunk, a) == PagedKvStatus::Ok);
    CHECK(w.ledger.used_bytes() == 7 * kChunkBytes);              // prompt 2 + new 5 chunks of backing
    CHECK(w.kv->reserve_capacity(a, 7 * kChunk) == PagedKvStatus::Ok);       // decode may grow to the bound
    CHECK(w.kv->reserve_capacity(a, 8 * kChunk) == PagedKvStatus::Ok);       // plus the rolling decode page
    CHECK(w.kv->reserve_capacity(a, 8 * kChunk + 1) == PagedKvStatus::BlocksExhausted);
    REQUIRE(w.kv->acquire_reserved(2, 2 * kChunk, b) == PagedKvStatus::Ok);  // the bound was one-shot
    CHECK(w.ledger.used_bytes() == 9 * kChunkBytes);
    CHECK(w.kv->reserve_capacity(b, 3 * kChunk + 1) == PagedKvStatus::BlocksExhausted);
}

TEST_CASE(KvflashSlotKvFixture, a_decode_page_past_max_ctx_is_not_a_prompt_too_long) {
    World w;
    PagedKvSequenceHandle a;
    // prompt == max_ctx; the engine's capacity request adds the next decode page
    CHECK(w.kv->acquire_reserved(1, (uint32_t)kMaxCtx + kChunk, a) == PagedKvStatus::Ok);
}

TEST_CASE(KvflashSlotKvFixture, evicted_history_stays_owned_so_reserve_decode_needs_no_extra_blocks) {
    World w;
    SeqSlotManager slots(*w.kv, (int)kMaxCtx);
    w.kv->set_next_total_tokens(2048 + 1984);
    const auto a = slots.admit(1, std::vector<int32_t>(2048, 7), SamplerCfg{});
    REQUIRE(a.status == SeqEngine::AdmitResult::Status::admitted);
    PagedKvSequenceHandle h = slots.slot(a.slot).handle;
    uint32_t owned = 0;
    REQUIRE(w.kv->owned_block_count(h, owned) == PagedKvStatus::Ok);
    CHECK(owned == 63);                                           // 4032 tokens of reservation, not the resident count
    CHECK(w.s[a.slot]->resident_blocks() == 0);
    CHECK(w.kv->free_block_count() == w.kv->physical_block_count() - 63);
}

TEST_CASE(KvflashSlotKvFixture, a_bound_that_fills_max_ctx_fits_an_exact_host_budget_and_does_not_count_the_decode_page_twice) {
    World w(/*budget=*/64 * kChunkBytes);                         // exactly max_ctx worth of backing
    SeqSlotManager slots(*w.kv, (int)kMaxCtx);
    w.kv->set_next_total_tokens(300 + 3796);
    const auto a = slots.admit(1, std::vector<int32_t>(300, 7), SamplerCfg{});
    REQUIRE(a.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(w.ledger.used_bytes() == 64 * kChunkBytes);
    CHECK(w.kv->free_block_count() <= w.kv->physical_block_count());
}

TEST_CASE(KvflashSlotKvFixture, a_short_generation_is_charged_its_total_bound_and_a_decoder_at_its_reservation_does_not_block_peers) {
    World w(/*budget=*/5 * kChunkBytes);                          // exactly five chunks of backing
    SeqSlotManager slots(*w.kv, (int)kMaxCtx);
    w.kv->set_next_total_tokens(301);                             // prompt 300 + 1 generated token
    REQUIRE(slots.admit(1, std::vector<int32_t>(300, 7), SamplerCfg{}).status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(w.ledger.used_bytes() == 5 * kChunkBytes);              // not six: the manager's extra decode page is not charged
}

TEST_CASE(KvflashSlotKvFixture, release_and_acquire_clear_a_score_hook_left_by_a_recall_turn) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 256, a) == PagedKvStatus::Ok);
    w.s[(size_t)a.slot]->score_hook = [](int c) { return (float)c; };
    REQUIRE(w.kv->release(a) == PagedKvStatus::Ok);
    CHECK(!w.s[(size_t)a.slot]->score_hook);
    w.s[(size_t)a.slot]->score_hook = [](int c) { return (float)c; };
    REQUIRE(w.kv->acquire_reserved(2, 256, a) == PagedKvStatus::Ok);
    CHECK(!w.s[(size_t)a.slot]->score_hook);
}

TEST_CASE(KvflashSlotKvFixture, rewind_prefill_drops_the_rows_past_the_point_in_the_allocator_and_the_pager) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    w.kv->set_next_total_tokens(700);
    const auto r = mgr.admit(1, std::vector<int32_t>(640, 1), SamplerCfg{});
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 512).ok);
    CHECK(mgr.slot(r.slot).cur_pos == 512);
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 8);
    CHECK(!mgr.rewind_prefill(r.slot, 600));                       // past cur_pos
    CHECK(mgr.rewind_prefill(r.slot, 512));                        // no-op
    CHECK(mgr.rewind_prefill(r.slot, 300));
    CHECK(mgr.slot(r.slot).cur_pos == 300);
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 5);                   // ceil(300/64): the chunk holding 300 keeps rows < 300
    const auto again = mgr.append_prefill(r.slot, 340);            // the replay reaches the end
    REQUIRE(again.ok);
    CHECK(again.rows.size() == 340u);
    CHECK(mgr.slot(r.slot).cur_pos == 640);
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 10);
    mgr.commit_prefill(r.slot);
    CHECK(!mgr.rewind_prefill(r.slot, 100));                       // decoding: not a prefill any more
}

TEST_CASE(KvflashSlotKvFixture, park_keeps_the_pager_and_the_reservation_and_only_a_decoded_slot_parks) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    std::vector<int32_t> hist;
    CHECK(!mgr.park(r.slot, hist));                                   // still prefilling
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    CHECK(mgr.park(r.slot, hist));
    CHECK(hist.size() == 640u);
    CHECK(!mgr.is_active(r.slot));
    CHECK(w.kv->parked(r.slot));
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 10);                    // the KV stays
    CHECK(w.ledger.reservation_count() == 1);                        // host backing stays charged
    CHECK(!mgr.park(r.slot, hist));                                  // not active any more
    w.kv->drop_parked(r.slot);
    CHECK(!w.kv->parked(r.slot));
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 0);
    CHECK(w.ledger.reservation_count() == 0);
}

TEST_CASE(KvflashSlotKvFixture, resume_rebinds_the_parked_slot_at_its_length_for_the_new_total) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(r.slot, hist));
    const int64_t charged = w.ledger.used_bytes();
    // resume: the manager admits into the parked slot at cur_pos 640, the allocator keeps its rows
    w.kv->set_next_total_tokens(800 + 60);
    w.kv->set_next_resume(r.slot, 640);
    const auto a = mgr.admit_at(r.slot, 2, std::vector<int32_t>(760, 1), SamplerCfg{}, 640);
    REQUIRE(a.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(a.slot == r.slot);
    CHECK(mgr.slot(r.slot).cur_pos == 640);
    CHECK(mgr.slot(r.slot).prompt_len == 760);
    CHECK(!w.kv->parked(r.slot));
    CHECK(w.ledger.reservation_count() == 1);
    CHECK(w.ledger.used_bytes() > charged);                          // re-admitted for the larger total
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 10);
    const auto more = mgr.append_prefill(r.slot, 120);              // chunks 10 and 11: the two free blocks
    REQUIRE(more.ok);
    CHECK(more.rows.size() == 120u);
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 12);
    mgr.commit_prefill(r.slot);
    mgr.retire(r.slot);
    CHECK(w.ledger.reservation_count() == 0);
}

TEST_CASE(KvflashSlotKvFixture, a_resume_hint_that_does_not_match_the_parked_slot_is_refused_and_a_plain_acquire_skips_parked_slots) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640);
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(r.slot, hist));
    PagedKvSequenceHandle h;
    w.kv->set_next_resume(r.slot, 600);                              // wrong length
    CHECK(w.kv->acquire_reserved(2, 700, h) == PagedKvStatus::InvalidArgument);
    CHECK(w.kv->parked(r.slot));
    w.kv->set_next_resume(1 - r.slot, 640);                          // not parked
    CHECK(w.kv->acquire_reserved(2, 700, h) == PagedKvStatus::InvalidArgument);
    CHECK(w.kv->acquire_reserved(2, 256, h) == PagedKvStatus::Ok);   // no hint: the free slot, never the parked one
    CHECK((int)h.slot == 1 - r.slot);
    CHECK(w.kv->acquire_reserved(3, 256, h) == PagedKvStatus::SequenceSlotsExhausted);   // parked counts as taken
    w.kv->set_next_slot(r.slot);                                     // an explicit slot drops the parked session
    CHECK(w.kv->acquire_reserved(3, 256, h) == PagedKvStatus::Ok);
    CHECK((int)h.slot == r.slot);
    CHECK(!w.kv->parked(r.slot));
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 0);
}

TEST_CASE(KvflashSlotKvFixture, a_parked_session_truncates_to_a_checkpoint_and_resumes_there) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(r.slot, hist));
    CHECK(!w.kv->truncate_parked(r.slot, 641));                       // past the parked length
    CHECK(!w.kv->truncate_parked(1 - r.slot, 10));                    // not parked
    REQUIRE(w.kv->truncate_parked(r.slot, 500));
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 8);                      // chunks 8 and 9 dropped, 7 partial
    w.kv->set_next_resume(r.slot, 640);
    PagedKvSequenceHandle stale;
    CHECK(w.kv->acquire_reserved(2, 760, stale) == PagedKvStatus::InvalidArgument);   // the old length no longer matches
    w.kv->set_next_total_tokens(760 + 60);
    w.kv->set_next_resume(r.slot, 500);
    const auto a = mgr.admit_at(r.slot, 3, std::vector<int32_t>(760, 1), SamplerCfg{}, 500);
    REQUIRE(a.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(mgr.slot(r.slot).cur_pos == 500);
    REQUIRE(mgr.append_prefill(r.slot, 260).ok);                     // 500..760 refills chunks 7..11
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 12);
    mgr.commit_prefill(r.slot);
    mgr.retire(r.slot);
    CHECK(w.ledger.reservation_count() == 0);
}

TEST_CASE(KvflashSlotKvFixture, a_resume_the_host_budget_refuses_leaves_the_session_parked) {
    World w(/*host_budget=*/70000);                                   // chunks of backing at 1000 B
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640, 0);              // 10 chunks
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(r.slot, hist));
    w.kv->set_next_total_tokens(1280);                               // 20 chunks in the other slot (the allocator skips the parked one)
    const auto other = mgr.admit_at(1 - r.slot, 5, std::vector<int32_t>(1280, 1), SamplerCfg{}, 0);
    REQUIRE(other.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(other.slot == 1 - r.slot);
    const int64_t used = w.ledger.used_bytes();
    w.kv->set_next_total_tokens(4000);                               // 63 chunks: 20 + 63 > 70 even with the 10 released
    w.kv->set_next_resume(r.slot, 640);
    const auto a = mgr.admit_at(r.slot, 2, std::vector<int32_t>(3900, 1), SamplerCfg{}, 640);
    CHECK(a.status == SeqEngine::AdmitResult::Status::busy);
    CHECK(w.kv->parked(r.slot));
    CHECK(w.ledger.reservation_count() == 2);                        // the old reservation is back
    CHECK(w.ledger.used_bytes() == used);
}

// RC1 N=2 HTTP 500: a sticky session grew past what the whole host budget can ever hold (the budget is sized for fewer
// session-sized reservations than its context needs). The ledger's HostTooSmall is permanent, not busy: waiting or evicting
// a peer cannot help. The pool must report it as capacity_exceeded (HTTP 400), keep the session parked with its old
// reservation, and never let the engine evict the very session being admitted.
TEST_CASE(KvflashSlotKvFixture, a_resume_whose_total_exceeds_the_whole_host_budget_is_capacity_exceeded_and_leaves_the_session_parked) {
    World w(/*host_budget=*/50000);                                   // 50 chunks of backing at 1000 B in total
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640, 0);              // 10 chunks
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(r.slot, hist));
    const int64_t used = w.ledger.used_bytes();
    w.kv->set_next_total_tokens(4000);                               // 63 chunks: more than the whole 50-chunk budget, nothing else parked
    w.kv->set_next_resume(r.slot, 640);
    const auto a = mgr.admit_at(r.slot, 2, std::vector<int32_t>(3900, 1), SamplerCfg{}, 640);
    CHECK(a.status == SeqEngine::AdmitResult::Status::capacity_exceeded);
    CHECK(w.kv->parked(r.slot));                                     // the session is untouched
    CHECK(w.ledger.reservation_count() == 1);
    CHECK(w.ledger.used_bytes() == used);
}

// Codex P2: the engine asks before it rewinds the session; the question charges nothing and moves nothing.
TEST_CASE(KvflashSlotKvFixture, the_capacity_probe_names_a_total_above_the_whole_host_budget_and_changes_nothing) {
    World w(/*host_budget=*/50000);
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640, 0);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(r.slot, hist));
    const int64_t used = w.ledger.used_bytes();
    CHECK(w.kv->exceeds_host_budget(r.slot, 4000));          // 63 chunks > 50
    CHECK(!w.kv->exceeds_host_budget(r.slot, 3200));         // 50 chunks fit exactly
    CHECK(w.kv->parked(r.slot));
    CHECK(w.ledger.used_bytes() == used);
    CHECK(w.ledger.reservation_count() == 1);
    CHECK(w.s[(size_t)r.slot]->n_chunks() == 10);
}

// Codex re-review P2: the scheduler can hand a total of max_ctx + 1 (prompt + generation bound); admission clamps it to max_ctx and
// charges max_ctx. The probe must see the same charge, or the session is rewound and then destroyed by the refusal.
TEST_CASE(KvflashSlotKvFixture, the_capacity_probe_clamps_the_total_to_max_ctx_exactly_as_admission_does) {
    World w(/*host_budget=*/50000);                           // 50 chunks; max_ctx is 64 chunks
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640, 0);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(r.slot, hist));
    CHECK(w.kv->exceeds_host_budget(r.slot, kMaxCtx + 1));
    CHECK(w.kv->exceeds_host_budget(r.slot, kMaxCtx));
    w.kv->set_next_total_tokens(kMaxCtx + 1);                // admission agrees: the same refusal, not a different charge
    w.kv->set_next_resume(r.slot, 640);
    CHECK(mgr.admit_at(r.slot, 2, std::vector<int32_t>(1500, 1), SamplerCfg{}, 640).status == SeqEngine::AdmitResult::Status::capacity_exceeded);
}

// A prompt that really exceeds max_ctx never reaches the ledger: SeqSlotManager refuses it first, with its own message,
// and the session stays parked untouched.
TEST_CASE(KvflashSlotKvFixture, a_prompt_over_max_ctx_is_refused_by_the_manager_before_the_ledger_and_leaves_the_session_parked) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const auto r = admit_tokens(mgr, *w.kv, 1, 640, 0);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 640).ok);
    mgr.commit_prefill(r.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(r.slot, hist));
    const int64_t used = w.ledger.used_bytes();
    w.kv->set_next_total_tokens(kMaxCtx + 10);
    w.kv->set_next_resume(r.slot, 640);
    const auto a = mgr.admit_at(r.slot, 2, std::vector<int32_t>((size_t)kMaxCtx + 1, 1), SamplerCfg{}, 640);
    CHECK(a.status == SeqEngine::AdmitResult::Status::capacity_exceeded);
    CHECK(a.error == "prompt exceeds max_ctx");
    CHECK(w.kv->parked(r.slot));
    CHECK(w.ledger.used_bytes() == used);
}

// Codex pass 3 counterexamples: two parked 640-token peers, a new sid with a 3,900-token prompt and 100 generation tokens needs 63
// chunks of a 50-chunk budget. The probe says so from the config alone (no bundle, no slot choice), and asking it moves nothing, so
// the engine can refuse before evicting either peer. The disk-resume ladder bytes are the pool's configured ladder, not per bundle.
TEST_CASE(KvflashSlotKvFixture, the_probe_refuses_a_new_sid_above_the_budget_with_two_parked_peers_untouched) {
    World w(/*host_budget=*/50000);
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    for (uint64_t id : {1, 2}) {
        const auto r = admit_tokens(mgr, *w.kv, id, 640, 0);
        REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
        REQUIRE(mgr.append_prefill(r.slot, 640).ok);
        mgr.commit_prefill(r.slot);
        std::vector<int32_t> hist;
        REQUIRE(mgr.park(r.slot, hist));
    }
    const int64_t used = w.ledger.used_bytes();
    CHECK(w.kv->exceeds_host_budget(0, 3900 + 100));
    CHECK(w.kv->exceeds_host_budget(1, 3900 + 100));          // either slot: the pagers are alike
    CHECK(w.kv->parked(0)); CHECK(w.kv->parked(1));
    CHECK(w.ledger.used_bytes() == used);
    CHECK(w.ledger.reservation_count() == 2);
}
