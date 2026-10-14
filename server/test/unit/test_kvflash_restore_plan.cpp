// Pure decision logic for restoring a KVFlash snapshot (patch 3/4):
// consume vs full re-prefill vs reject, and the token history a consuming
// restore hands the scorer. No GPU or filesystem state is involved.
#include "CppUnitTestFramework.hpp"
#include "../../src/qwen35/prefill_helpers.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
struct KvflashRestorePlanFixture {};
}

using luce::common::KvflashRestorePlan;
using luce::common::kvflash_restore_plan;
using luce::common::kvflash_restore_history;
using luce::common::kvflash_restore_consume_enabled;

TEST_CASE(KvflashRestorePlanFixture, dense_snapshot_is_never_touched) {
    CHECK(kvflash_restore_plan(false, false, 640, 64, true) == KvflashRestorePlan::Dense);
    CHECK(kvflash_restore_plan(true,  false, 641, 64, false) == KvflashRestorePlan::Dense);
}

TEST_CASE(KvflashRestorePlanFixture, pooled_with_consume_off_reprefills_from_zero) {
    CHECK(kvflash_restore_plan(false, true, 640, 64, true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(false, true, 640, 64, false) == KvflashRestorePlan::Reject);
}

TEST_CASE(KvflashRestorePlanFixture, pooled_aligned_consume_uses_the_restored_kv) {
    CHECK(kvflash_restore_plan(true, true, 640, 64, true) == KvflashRestorePlan::Consume);
    CHECK(kvflash_restore_plan(true, true, 640, 64, false) == KvflashRestorePlan::Consume);
}

TEST_CASE(KvflashRestorePlanFixture, misaligned_offset_never_prefills_a_bare_suffix) {
    // Donor bug: kv_offset reset to 0 but only the suffix prefilled. The plan
    // must be a re-prefill of the WHOLE prompt, or a clear failure.
    CHECK(kvflash_restore_plan(true, true, 641, 64, true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(true, true, 641, 64, false) == KvflashRestorePlan::Reject);
}

TEST_CASE(KvflashRestorePlanFixture, degenerate_geometry_falls_back) {
    CHECK(kvflash_restore_plan(true, true, 0, 64, true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(true, true, 640, 0, true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(true, true, 640, 0, false) == KvflashRestorePlan::Reject);
}

// A dense snapshot (restored prefix 7039) plus a prompt that crosses the pool
// (8192, chunk 64) cannot continue densely and the pooled prefill needs a fresh
// request: the plan must re-prefill the whole prompt, never fail.
TEST_CASE(KvflashRestorePlanFixture, dense_restore_that_outgrows_the_pool_reprefills) {
    CHECK(kvflash_restore_plan(false, false, 7039, 64, true, 7039 + 1354, 8192) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(false, false, 316, 64, true, 9000, 8192) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(true,  false, 316, 64, true, 9000, 8192) == KvflashRestorePlan::FullReprefill);
}

TEST_CASE(KvflashRestorePlanFixture, dense_restore_inside_the_pool_stays_dense) {
    CHECK(kvflash_restore_plan(false, false, 7039, 64, true, 8000, 8192) == KvflashRestorePlan::Dense);
    CHECK(kvflash_restore_plan(false, false, 7039, 64, true, 8192 - 64, 8192) == KvflashRestorePlan::Dense);
    CHECK(kvflash_restore_plan(false, false, 7039, 64, true, 8192 - 63, 8192) == KvflashRestorePlan::FullReprefill);
    // kvflash off (pool 0) or an exact hit: untouched.
    CHECK(kvflash_restore_plan(false, false, 7039, 64, true, 20000, 0) == KvflashRestorePlan::Dense);
    CHECK(kvflash_restore_plan(false, false, 8192, 64, true, 8192, 8192) == KvflashRestorePlan::Dense);
}

// A dense snapshot longer than the pool (a disk entry written by a server with
// no pool, or a larger one) cannot be written into the pool tensor, whatever
// the prompt length and whether or not the pager may adopt dense rows.
TEST_CASE(KvflashRestorePlanFixture, dense_snapshot_longer_than_the_pool_always_reprefills) {
    // Exact hit: nothing left to prefill, still too long for the pool.
    CHECK(kvflash_restore_plan(false, false, 9000, 64, true, 9000, 8192) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(false, false, 11600, 64, true, 11606, 8192, /*adopt_dense=*/true) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(false, false, 11600, 64, true, 11606, 8192, false) == KvflashRestorePlan::FullReprefill);
    CHECK(kvflash_restore_plan(false, false, 8193, 64, true, 8193, 8192, true) == KvflashRestorePlan::FullReprefill);
    // No prompt to re-prefill from: refuse rather than write out of bounds.
    CHECK(kvflash_restore_plan(false, false, 11600, 64, false, 0, 8192, true) == KvflashRestorePlan::Reject);
    // The server cut of a re-prefill is 0, so nothing is reported restorable.
    CHECK(luce::common::kvflash_restorable_tokens(true, false, false, 11600, 11600, 64, 11606, 8192,
                                    /*session=*/true, -1) == 0);
}

// ---- restorable prefix: the cut a restore would take ----------------------

using luce::common::kvflash_restorable_tokens;

namespace {
// Snapshot-side arguments are (has_snapshot, consume, pooled, snap_pos,
// slot_tokens); geometry is chunk 64, pool 65536, a 40000-token prompt.
int restorable(bool has, bool consume, bool pooled, int snap_pos, int slot_tokens,
               int live_cut, bool session = true) {
    return kvflash_restorable_tokens(has, consume, pooled, snap_pos, slot_tokens,
                                     64, 40000, 65536, session, live_cut);
}
} // namespace

TEST_CASE(KvflashRestorePlanFixture, restorable_live_session_alone_is_reported) {
    CHECK(restorable(false, false, false, 0, 0, 32000) == 32000);
}

TEST_CASE(KvflashRestorePlanFixture, restorable_nothing_is_zero_never_negative) {
    CHECK(restorable(false, false, false, 0, 0, -1) == 0);
    CHECK(restorable(false, true, true, 20032, 20032, -1) == 0);
}

TEST_CASE(KvflashRestorePlanFixture, restorable_dense_snapshot_loses_to_a_longer_live_cut) {
    CHECK(restorable(true, false, false, 20000, 20000, 32000) == 32000);
    CHECK(restorable(true, false, false, 20000, 20000, 10000) == 20000);
    CHECK(restorable(true, false, false, 20000, 20000, 20000) == 20000);
}

TEST_CASE(KvflashRestorePlanFixture, restorable_dense_snapshot_without_a_live_cut_is_trusted) {
    CHECK(restorable(true, false, false, 20000, 20000, -1) == 20000);
    CHECK(restorable(true, true, false, 20000, 20000, -1) == 20000);
}

TEST_CASE(KvflashRestorePlanFixture, restorable_misaligned_pooled_snapshot_is_thrown_away) {
    // The backend would re-prefill the whole prompt, so the cache restores 0.
    CHECK(restorable(true, true, true, 20001, 20001, -1) == 0);
    // ...unless the live session still covers a prefix.
    CHECK(restorable(true, true, true, 20001, 20001, 18000) == 18000);
}

TEST_CASE(KvflashRestorePlanFixture, restorable_aligned_pooled_snapshot_consumes) {
    CHECK(restorable(true, true, true, 20032, 20032, -1) == 20032);
    CHECK(restorable(true, true, true, 20032, 20032, 30000) == 30000);
}

TEST_CASE(KvflashRestorePlanFixture, restorable_pooled_snapshot_with_consume_off_is_zero) {
    CHECK(restorable(true, false, true, 20032, 20032, -1) == 0);
    CHECK(restorable(true, false, true, 20032, 20032, 9000) == 9000);
}

TEST_CASE(KvflashRestorePlanFixture, restorable_is_capped_by_the_tokens_the_slot_matches) {
    CHECK(restorable(true, false, false, 20000, 15000, -1) == 15000);
    CHECK(restorable(true, true, true, 20032, 12800, -1) == 12800);
    // The live cut is not capped by the slot.
    CHECK(restorable(true, false, false, 20000, 15000, 32000) == 32000);
}

TEST_CASE(KvflashRestorePlanFixture, restorable_dense_snapshot_outgrowing_the_pool) {
    // 40000-token prompt, pool 8192: a dense prefix cannot continue densely.
    // Without the live session it re-prefills whole; with it the pager adopts.
    CHECK(kvflash_restorable_tokens(true, false, false, 7039, 7039, 64, 40000, 8192,
                                    /*session=*/false, -1) == 0);
    CHECK(kvflash_restorable_tokens(true, false, false, 7039, 7039, 64, 40000, 8192,
                                    /*session=*/true, -1) == 7039);
}

TEST_CASE(KvflashRestorePlanFixture, history_cold_prefill_is_just_the_tokens) {
    const std::vector<int32_t> tok = {5, 6, 7};
    const auto h = kvflash_restore_history(tok, 0, nullptr);
    CHECK(h == tok);
}

TEST_CASE(KvflashRestorePlanFixture, history_uses_real_ids_for_the_restored_prefix) {
    const std::vector<int32_t> full = {1, 2, 3, 4, 5, 6};
    const std::vector<int32_t> suffix = {4, 5, 6};
    const auto h = kvflash_restore_history(suffix, 3, &full);
    CHECK(h == full);
}

TEST_CASE(KvflashRestorePlanFixture, history_without_full_prompt_zero_fills_prefix) {
    const std::vector<int32_t> suffix = {4, 5};
    const auto h = kvflash_restore_history(suffix, 3, nullptr);
    CHECK(h == (std::vector<int32_t>{0, 0, 0, 4, 5}));
    // A full prompt shorter than the offset cannot supply the prefix either.
    const std::vector<int32_t> short_full = {1, 2};
    CHECK(kvflash_restore_history(suffix, 3, &short_full) == h);
}

TEST_CASE(KvflashRestorePlanFixture, consume_flag_is_off_unless_exactly_one) {
    unsetenv("LUCE_KVFLASH_RESTORE_CONSUME");
    CHECK(!kvflash_restore_consume_enabled());
    setenv("LUCE_KVFLASH_RESTORE_CONSUME", "0", 1);
    CHECK(!kvflash_restore_consume_enabled());
    setenv("LUCE_KVFLASH_RESTORE_CONSUME", "", 1);
    CHECK(!kvflash_restore_consume_enabled());
    setenv("LUCE_KVFLASH_RESTORE_CONSUME", "1", 1);
    CHECK(kvflash_restore_consume_enabled());
    unsetenv("LUCE_KVFLASH_RESTORE_CONSUME");
}

// ---- S6: KVFlash prefill on an absolute grid ------------------------------

using luce::common::kvflash_prefill_batches;
using luce::common::kvflash_prefill_batch_end;
using Batches = std::vector<std::pair<int, int>>;

namespace {
constexpr int kU = 512;

// Invariants of every plan: contiguous cover of [start, end); every batch but
// the first starts on the grid; every batch is within max_batch; every batch
// but the last (the end-of-prompt remainder) ends on the grid.
void check_plan(const Batches & b, int start, int end, int grid, int max_batch) {
    if (start >= end) { if (!b.empty()) throw std::runtime_error("non-empty plan for empty range"); return; }
    if (b.empty() || b.front().first != start || b.back().second != end)
        throw std::runtime_error("plan does not cover the range");
    for (size_t i = 0; i < b.size(); i++) {
        if (b[i].second <= b[i].first) throw std::runtime_error("empty batch");
        if (b[i].second - b[i].first > max_batch) throw std::runtime_error("batch over max");
        if (i > 0 && b[i].first != b[i - 1].second) throw std::runtime_error("gap or overlap");
        if (i > 0 && b[i].first % grid != 0) throw std::runtime_error("batch starts off-grid");
        if (i + 1 < b.size() && b[i].second % grid != 0) throw std::runtime_error("inner batch ends off-grid");
    }
}
} // namespace

TEST_CASE(KvflashRestorePlanFixture, grid_plan_cold_prompt_lengths) {
    for (int n : {1, 511, 512, 513, 18813}) {
        check_plan(kvflash_prefill_batches(0, n, kU, kU), 0, n, kU, kU);
    }
    CHECK(kvflash_prefill_batches(0, 1, kU, kU) == (Batches{{0, 1}}));
    CHECK(kvflash_prefill_batches(0, 511, kU, kU) == (Batches{{0, 511}}));
    CHECK(kvflash_prefill_batches(0, 512, kU, kU) == (Batches{{0, 512}}));
    CHECK(kvflash_prefill_batches(0, 513, kU, kU) == (Batches{{0, 512}, {512, 513}}));
    const Batches big = kvflash_prefill_batches(0, 18813, kU, kU);
    CHECK(big.size() == 37);                       // 36 full cells + 381-token remainder
    CHECK(big.back() == (std::pair<int, int>{18432, 18813}));
}

TEST_CASE(KvflashRestorePlanFixture, grid_plan_off_grid_starts_run_to_the_grid_first) {
    // 18432 = 36 * 512 is on the grid.
    CHECK(kvflash_prefill_batches(18432, 18813, kU, kU) == (Batches{{18432, 18813}}));
    // 18813 is off-grid: the first batch ends at the next grid point 18944.
    CHECK(kvflash_prefill_batches(18813, 19000, kU, kU) == (Batches{{18813, 18944}, {18944, 19000}}));
    const Batches b = kvflash_prefill_batches(18813, 20000, kU, kU);
    CHECK(b.front() == (std::pair<int, int>{18813, 18944}));
    CHECK(b[1] == (std::pair<int, int>{18944, 19456}));
    check_plan(b, 18813, 20000, kU, kU);
    // A chunk-aligned off-grid start (what a consuming restore produces).
    const Batches r = kvflash_prefill_batches(18560, 19500, kU, kU);
    CHECK(r.front() == (std::pair<int, int>{18560, 18944}));
    check_plan(r, 18560, 19500, kU, kU);
}

TEST_CASE(KvflashRestorePlanFixture, grid_plan_max_batch_multiples_and_not) {
    for (int mb : {512, 1024, 1536, 640, 768, 1000, 4096}) {
        for (int start : {0, 18432, 18813, 18560}) {
            for (int n : {1, 511, 513, 18813}) {
                check_plan(kvflash_prefill_batches(start, start + n, kU, mb), start, start + n, kU, mb);
            }
        }
    }
    // 1024 packs two cells per batch; 640 (not a multiple) still packs one.
    CHECK(kvflash_prefill_batches(0, 2048, kU, 1024) == (Batches{{0, 1024}, {1024, 2048}}));
    CHECK(kvflash_prefill_batches(0, 2048, kU, 640) ==
          (Batches{{0, 512}, {512, 1024}, {1024, 1536}, {1536, 2048}}));
    // A start 1 token below a grid point folds the gap into a wider first batch
    // when max_batch allows it, so no 1-token mid-prompt batch exists.
    CHECK(kvflash_prefill_batches(1023, 3000, kU, 1024).front() == (std::pair<int, int>{1023, 1536}));
}

TEST_CASE(KvflashRestorePlanFixture, grid_plan_small_max_batch_shrinks_the_grid) {
    // max_batch below U: the grid becomes max_batch so the invariants hold.
    check_plan(kvflash_prefill_batches(0, 1000, kU, 192), 0, 1000, 192, 192);
    CHECK(kvflash_prefill_batches(5, 5, kU, kU).empty());
    CHECK(kvflash_prefill_batches(9, 3, kU, kU).empty());
}

TEST_CASE(KvflashRestorePlanFixture, grid_plan_batch_end_matches_the_list) {
    // The stateless per-position form (used by the prefill loop) and the list agree.
    for (int start : {0, 18813, 18560}) {
        const int end = start + 5000;
        for (const auto & p : kvflash_prefill_batches(start, end, kU, 1024)) {
            CHECK(kvflash_prefill_batch_end(p.first, end, kU, 1024) == p.second);
        }
    }
}

// ---- S9b: one extra cut point on the grid ---------------------------------

TEST_CASE(KvflashRestorePlanFixture, turn_cut_splits_one_batch_and_keeps_the_grid) {
    const Batches b = kvflash_prefill_batches(0, 2000, kU, kU, {1300});
    CHECK(b == (Batches{{0, 512}, {512, 1024}, {1024, 1300}, {1300, 1536}, {1536, 2000}}));
    // Per-position form agrees.
    for (const auto & p : b) CHECK(kvflash_prefill_batch_end(p.first, 2000, kU, kU, {1300}) == p.second);
}

TEST_CASE(KvflashRestorePlanFixture, turn_cut_near_the_end_leaves_a_short_final_batch) {
    // Pend = 1000, k = 4: the last batch is the 4 generation-suffix tokens.
    CHECK(kvflash_prefill_batches(0, 1000, kU, kU, {996}) == (Batches{{0, 512}, {512, 996}, {996, 1000}}));
}

TEST_CASE(KvflashRestorePlanFixture, turn_cut_outside_or_on_the_grid_changes_nothing) {
    for (int cut : {-1, 0, 512, 1024, 2000, 5000}) {
        CHECK(kvflash_prefill_batches(0, 2000, kU, kU, {cut}) == kvflash_prefill_batches(0, 2000, kU, kU));
    }
    // A cut at or below the start of the range is ignored (restore at C == cut).
    CHECK(kvflash_prefill_batches(1300, 2000, kU, kU, {1300}) == kvflash_prefill_batches(1300, 2000, kU, kU));
    check_plan(kvflash_prefill_batches(1300, 2000, kU, kU, {1300}), 1300, 2000, kU, kU);
}

TEST_CASE(KvflashRestorePlanFixture, turn_cut_with_wide_batches) {
    CHECK(kvflash_prefill_batches(0, 3000, kU, 1024, {1300}) ==
          (Batches{{0, 1024}, {1024, 1300}, {1300, 2048}, {2048, 3000}}));
}

TEST_CASE(KvflashRestorePlanFixture, several_cuts_split_in_order) {
    CHECK(kvflash_prefill_batches(0, 2000, kU, kU, {700, 1300}) ==
          (Batches{{0, 512}, {512, 700}, {700, 1024}, {1024, 1300}, {1300, 1536}, {1536, 2000}}));
    // two cuts in one grid cell
    CHECK(kvflash_prefill_batches(0, 1024, kU, kU, {100, 200}) ==
          (Batches{{0, 100}, {100, 200}, {200, 512}, {512, 1024}}));
}

TEST_CASE(KvflashRestorePlanFixture, replay_from_a_cut_matches_the_cold_batches) {
    // S9b: with the cut set a pure function of the prompt, a restore at C
    // (any cut) replays exactly the batches a cold prefill ran from C on.
    for (int mb : {512, 1024}) {
        const std::vector<int> cuts{100, 700, 1300, 1301, 4996};
        const Batches cold = kvflash_prefill_batches(0, 5000, kU, mb, cuts);
        for (int c : cuts) {
            Batches tail;
            for (const auto & p : cold) if (p.first >= c) tail.push_back(p);
            CHECK(kvflash_prefill_batches(c, 5000, kU, mb, cuts) == tail);
        }
    }
}
