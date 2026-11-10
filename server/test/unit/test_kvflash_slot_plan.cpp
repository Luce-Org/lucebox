// Pure unit tests for the slot-pager arithmetic (kvflash_slot_plan.h):
// slice geometry and host-budget admission. No ggml, no GPU.
#include "CppUnitTestFramework.hpp"
#include "kvflash_test_seams.h"
#include "../../src/common/kvflash_slot_plan.h"

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotPlanFixture {};

constexpr int64_t kMiB = 1024 * 1024;
constexpr int64_t kChunkBytes = 17 * kMiB / 8;   // 2.125 MiB, q8_0 qwen35

KvFlashAdmitInput base_admit() {
    KvFlashAdmitInput in;
    in.slot_free = true;
    in.quota_chunks = 352;
    in.headroom_blocks = 8;
    in.prompt_tokens = 1000;
    in.max_new_tokens = 24;               // 1024 tokens = 16 chunks
    in.max_ctx = 131072;
    in.chunk_tokens = 64;
    in.chunk_bytes = kChunkBytes;
    in.host_used_bytes = 0;
    in.host_budget_bytes = 16 * kChunkBytes;
    return in;
}
}  // namespace

TEST_CASE(KvflashSlotPlanFixture, arena_sizing_derives_the_pool_and_refuses_a_different_kv_pool_tokens) {
    const KvFlashArenaSizing a = kvflash_arena_sizing(4, 352, 0);
    CHECK(a.error.empty());
    CHECK(a.pool_tokens == 4 * 352 * 64);                          // N x Q x 64
    CHECK(a.rows == kvflash_arena_rows(4, 352));                   // ctx_alloc: the pool plus the scratch block
    const KvFlashArenaSizing same = kvflash_arena_sizing(4, 352, 4 * 352 * 64);
    CHECK(same.error.empty());
    CHECK(same.rows == a.rows);
    const KvFlashArenaSizing diff = kvflash_arena_sizing(4, 352, 4096);
    CHECK(diff.rows == 0);
    CHECK(diff.error.find("--kv-pool-tokens 4096") != std::string::npos);
    CHECK(diff.error.find("90112") != std::string::npos);          // names the derived value
    CHECK(!kvflash_arena_sizing(0, 352, 0).error.empty());
    CHECK(!kvflash_arena_sizing(4, 0, 0).error.empty());
    CHECK(!kvflash_arena_sizing(8, 1 << 24, 0).error.empty());    // rows do not fit the int ctx_alloc
}

TEST_CASE(KvflashSlotPlanFixture, arena_sizing_without_a_chain_keeps_the_one_scratch_block_layout) {
    const KvFlashArenaSizing a = kvflash_arena_sizing(2, 32, 0);
    CHECK(a.error.empty());
    CHECK(a.rows == 2 * 32 * 64 + 64);
    CHECK(a.scratch_base == a.pool_tokens);          // no chain rows: both sit where the slices end
    CHECK(a.dead_row == a.pool_tokens);
}

TEST_CASE(KvflashSlotPlanFixture, arena_sizing_with_a_chain_adds_one_scratch_stride_per_slot_between_the_slices_and_the_dead_block) {
    const int N = 2, Q = 32, stride = 16;            // a block-16 chain: paged_token_capacity(16) == 16
    const KvFlashArenaSizing a = kvflash_arena_sizing(N, Q, 0, 64, stride);
    REQUIRE(a.error.empty());
    CHECK(a.pool_tokens == N * Q * 64);              // the pool is unchanged
    CHECK(a.scratch_base == N * Q * 64);             // chain scratch starts at the end of the last slice
    CHECK(a.scratch_base % 16 == 0);
    CHECK(a.dead_row == a.scratch_base + (int64_t)N * stride);   // the dead-row block follows the scratch rows
    CHECK(a.rows == a.dead_row + 64);                // N*Q*64 + N*stride + 64
    CHECK(a.rows == kvflash_arena_rows(N, Q) + (int64_t)N * stride);
    // no slot owns a chain scratch row or the dead block
    for (int64_t r = a.scratch_base; r < a.rows; r++) CHECK(kvflash_slot_of_row(r, N, Q) == -1);
    // every slot's scratch window [base + s*stride, +stride) is disjoint and inside the arena
    for (int sl = 0; sl < N; sl++) CHECK(a.scratch_base + (int64_t)(sl + 1) * stride <= a.dead_row);
    // --kv-pool-tokens stays derived, chain or not
    CHECK(!kvflash_arena_sizing(N, Q, 2048, 64, stride).error.empty());
    CHECK(kvflash_arena_sizing(N, Q, N * Q * 64, 64, stride).error.empty());
}

TEST_CASE(KvflashSlotPlanFixture, arena_sizing_refuses_a_chain_stride_that_is_not_block_aligned_or_overflows_int32) {
    CHECK(!kvflash_arena_sizing(2, 32, 0, 64, 12).error.empty());      // scratch_base + slot*stride must stay 16-aligned
    CHECK(!kvflash_arena_sizing(2, 32, 0, 64, -16).error.empty());
    // N*Q*64 + N*stride + 64 > INT32_MAX though the slices and the dead block alone fit
    const int Q = (int)(((int64_t)INT32_MAX - 64) / 64 / 2);           // 2 slots fill the int range exactly
    CHECK(kvflash_arena_sizing(2, Q, 0).error.empty());
    CHECK(!kvflash_arena_sizing(2, Q, 0, 64, 1 << 20).error.empty());
    CHECK(kvflash_arena_sizing(2, 32, 0, 64, 16).rows == 2 * 32 * 64 + 2 * 16 + 64);
}

TEST_CASE(KvflashSlotPlanFixture, slices_tile_the_arena_exactly) {
    const int N = 4, Q = 352;                       // Q = 22,528 tokens
    REQUIRE(kvflash_arena_rows(N, Q) == (int64_t)N * Q * 64 + 64);   // N x Q x 64 + scratch block
    int64_t next = 0;
    for (int s = 0; s < N; s++) {
        const KvFlashSlice sl = kvflash_slice(s, Q);
        CHECK(sl.row_lo == next);                   // no gap, no overlap
        CHECK(sl.row_hi - sl.row_lo == 22528);
        CHECK(sl.block_hi - sl.block_lo == Q);
        CHECK(sl.row_lo == (int64_t)sl.block_lo * 64);
        next = sl.row_hi;
    }
    CHECK(next == (int64_t)N * Q * 64);
    CHECK(kvflash_scratch_row(N, Q) == next);       // the scratch block starts where the slices end
}

TEST_CASE(KvflashSlotPlanFixture, quota_floor_is_the_pager_minimum) {
    KvFlashConfig cfg;                              // sink 1, tail 4: 7 chunks
    CHECK(kvflash_quota_ok(7, cfg));
    CHECK(!kvflash_quota_ok(6, cfg));
    CHECK(!kvflash_quota_ok(0, cfg));
    CHECK(kvflash_quota_ok(352, cfg));
    CHECK(kvflash_quota_chunks(22528, 64) == 352);
    CHECK(kvflash_quota_chunks(22529, 64) == 0);    // not a whole number of chunks
}

TEST_CASE(KvflashSlotPlanFixture, admission_ok_when_everything_fits) {
    CHECK(kvflash_admit(base_admit()) == KvFlashAdmit::Ok);
}

TEST_CASE(KvflashSlotPlanFixture, admission_host_need_counts_prompt_plus_new_tokens_in_whole_chunks) {
    KvFlashAdmitInput in = base_admit();
    CHECK(kvflash_host_need_bytes(in) == 16 * kChunkBytes);          // 1024 tokens
    in.max_new_tokens = 25;                                          // 1025 tokens -> 17 chunks
    CHECK(kvflash_host_need_bytes(in) == 17 * kChunkBytes);
    in.ladder_bytes = 3 * 150 * kMiB;
    CHECK(kvflash_host_need_bytes(in) == 17 * kChunkBytes + 3 * 150 * kMiB);
    in.ladder_bytes = 0;
    // 128K prompt at q8_0: 2048 chunks = 4.25 GiB.
    in = base_admit();
    in.prompt_tokens = 131072; in.max_new_tokens = 0;
    CHECK(kvflash_host_need_bytes(in) == 2048 * kChunkBytes);
    CHECK(kvflash_host_need_bytes(in) == 4352 * kMiB);
}

TEST_CASE(KvflashSlotPlanFixture, admission_host_budget_boundary_is_exact) {
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 16 * kChunkBytes;                         // exactly the need
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    in.host_budget_bytes -= 1;                                       // the request alone cannot ever fit
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
    in = base_admit();
    in.host_used_bytes = 1;                                          // one byte already taken by others: transient
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);
}

TEST_CASE(KvflashSlotPlanFixture, the_ladder_is_part_of_what_the_request_alone_needs) {
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 16 * kChunkBytes + 100;
    in.ladder_bytes = 100;                                           // fits exactly
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    in.ladder_bytes = 101;
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);          // permanent: nothing else is reserved
    in.host_used_bytes = 2; in.ladder_bytes = 100; in.host_budget_bytes += 1;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);              // fits alone, waits for others
}

TEST_CASE(KvflashSlotPlanFixture, the_ladder_charge_is_one_slab_per_rung_and_one_for_a_recall_turns_snapshot) {
    constexpr int64_t kSlab = 150 * kMiB;
    CHECK(kvflash_ladder_bytes(kSlab, false, true) == 3 * kSlab);    // checkpoint, trunk and the third rung
    CHECK(kvflash_ladder_bytes(kSlab, true, true) == 4 * kSlab);     // the R snapshot beside the three rungs
    CHECK(kvflash_ladder_bytes(kSlab, true, false) == kSlab);        // not sticky: no rungs
    CHECK(kvflash_ladder_bytes(kSlab, false, false) == 0);
    KvFlashAdmitInput now = base_admit(), before = now;
    now.ladder_bytes = kvflash_ladder_bytes(kSlab, false, true);
    before.ladder_bytes = 2 * kSlab;                                 // the two-rung ladder before the third rung
    CHECK(kvflash_host_need_bytes(now) - kvflash_host_need_bytes(before) == kSlab);
}

TEST_CASE(KvflashSlotPlanFixture, admission_refuses_busy_slot_and_overlong_prompt_with_distinct_verdicts) {
    KvFlashAdmitInput in = base_admit();
    in.slot_free = false;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusySlot);
    in = base_admit();
    in.prompt_tokens = in.max_ctx + 1;                               // prompt <= max_ctx
    CHECK(kvflash_admit(in) == KvFlashAdmit::PromptTooLong);
    in.prompt_tokens = in.max_ctx;
    in.max_new_tokens = 0;
    in.host_budget_bytes = 1LL << 40;
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    // A permanent refusal outranks a transient one.
    in.prompt_tokens = in.max_ctx + 1;
    in.slot_free = false;
    CHECK(kvflash_admit(in) == KvFlashAdmit::PromptTooLong);
}

TEST_CASE(KvflashSlotPlanFixture, headroom_shrinks_the_usable_quota_and_keeps_the_pager_floor) {
    KvFlashConfig cfg;                                               // floor 7 chunks
    CHECK(kvflash_headroom_ok(352, 8, cfg));
    CHECK(kvflash_headroom_ok(15, 8, cfg));                          // usable 7: exactly the floor
    CHECK(!kvflash_headroom_ok(15, 9, cfg));                         // usable 6
    CHECK(!kvflash_headroom_ok(352, -1, cfg));
    CHECK(kvflash_headroom_ok(7, 0, cfg));
}

TEST_CASE(KvflashSlotPlanFixture, admission_with_headroom_refuses_a_quota_below_the_floor) {
    KvFlashAdmitInput in = base_admit();
    in.quota_chunks = 15; in.headroom_blocks = 8;                    // usable 7: ok
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    in.headroom_blocks = 9;                                          // usable 6
    CHECK(kvflash_admit(in) == KvFlashAdmit::BadQuota);
    in.slot_free = false; in.prompt_tokens = in.max_ctx + 1;         // a bad quota outranks everything
    CHECK(kvflash_admit(in) == KvFlashAdmit::BadQuota);
    // The host bound is the same with and without headroom.
    in = base_admit();
    const int64_t need = kvflash_host_need_bytes(in);
    in.headroom_blocks = 0;
    CHECK(kvflash_host_need_bytes(in) == need);
}

// ── the host reservation ledger and the permanent refusal ─────────

TEST_CASE(KvflashSlotPlanFixture, a_request_that_can_never_fit_is_refused_not_queued) {
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 16 * kChunkBytes - 1;                     // alone it exceeds the whole budget
    in.host_used_bytes = 0;
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
    in.host_used_bytes = 5 * kChunkBytes;                            // same with others present: still permanent
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
    in.host_budget_bytes = 100 * kChunkBytes;                        // fits alone, others hold it: transient
    in.host_used_bytes = 90 * kChunkBytes;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);
    // A permanent refusal outranks a busy slot; a bad quota and an overlong prompt outrank it.
    in.host_budget_bytes = 1; in.host_used_bytes = 0; in.slot_free = false;
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
    in.prompt_tokens = in.max_ctx + 1;
    CHECK(kvflash_admit(in) == KvFlashAdmit::PromptTooLong);
}

TEST_CASE(KvflashSlotPlanFixture, the_ledger_reserves_at_admit_and_releases_on_every_exit) {
    KvFlashTestLedger led(40 * kChunkBytes);
    KvFlashAdmitInput in = base_admit();                             // needs 16 chunks
    in.host_budget_bytes = 0;                                        // the ledger owns the budget and used bytes
    CHECK(led.admit(1, in) == KvFlashAdmit::Ok);
    CHECK(led.used_bytes() == 16 * kChunkBytes);
    CHECK(led.admit(2, in) == KvFlashAdmit::Ok);
    CHECK(led.used_bytes() == 32 * kChunkBytes);
    CHECK(led.admit(3, in) == KvFlashAdmit::BusyHost);               // 48 > 40, but it fits alone
    CHECK(led.used_bytes() == 32 * kChunkBytes);                     // a refusal reserves nothing
    CHECK(led.reservation_count() == 2);
    led.release(1);                                                  // finish
    CHECK(led.admit(3, in) == KvFlashAdmit::Ok);
    led.release(2);                                                  // cancel
    led.release(3);                                                  // error / abort_step
    CHECK(led.used_bytes() == 0);
    CHECK(led.reservation_count() == 0);
    led.release(3);                                                  // idempotent
    led.release(99);                                                 // unknown id
    CHECK(led.used_bytes() == 0);
}

TEST_CASE(KvflashSlotPlanFixture, the_ledger_never_double_counts_a_request_id) {
    KvFlashTestLedger led(40 * kChunkBytes);
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 0;
    CHECK(led.admit(1, in) == KvFlashAdmit::Ok);
    CHECK(led.admit(1, in) == KvFlashAdmit::BusySlot);               // same id twice: refused, not charged again
    CHECK(led.used_bytes() == 16 * kChunkBytes);
    CHECK(led.admit(2, in) == KvFlashAdmit::Ok);                     // 16 + 16 = 32
    KvFlashAdmitInput small = in; small.prompt_tokens = 500; small.max_new_tokens = 12;   // 8 chunks
    CHECK(led.admit(5, small) == KvFlashAdmit::Ok);                  // 32 + 8 = 40: exact fit
    CHECK(led.admit(3, small) == KvFlashAdmit::BusyHost);
    in.prompt_tokens = 100000;
    CHECK(led.admit(4, in) == KvFlashAdmit::HostTooSmall);           // bigger than the whole budget
    CHECK(led.reservation_count() == 3);
}

TEST_CASE(KvflashSlotPlanFixture, auto_budget_is_split_across_slots_and_the_reserve_is_held_once) {
    KvFlashAutoBudget b; b.free_bytes = 10000; b.reserve_bytes = 2000; b.bytes_per_token = 10;
    CHECK(kvflash_budget_per_slot(b, 1).free_bytes == 10000);
    CHECK(kvflash_budget_per_slot(b, 4).free_bytes == 2000 + 8000 / 4);
    KvFlashAutoBudget low; low.free_bytes = 1000; low.reserve_bytes = 2000;
    CHECK(kvflash_budget_per_slot(low, 4).free_bytes == 1000);     // nothing above the reserve: unchanged
}
