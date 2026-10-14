// Pure decisions for resuming a KVFlash request across the pool boundary:
// server snapshot cut vs live-session cut, adopting dense rows as pager
// chunks, and the state-ladder rungs a dense prefill can capture.
// CPU only, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"
#include "../../src/qwen35/prefill_helpers.h"

#include <algorithm>
#include <vector>

namespace {
struct KvflashResumeFixture {};
}

using namespace luce::common;

// ── server cut vs live cut ──────────────────────────────────────────

TEST_CASE(KvflashResumeFixture, server_cut_is_zero_unless_the_plan_keeps_the_snapshot) {
    CHECK(kvflash_server_cut(KvflashRestorePlan::Dense, 8089) == 8089);
    CHECK(kvflash_server_cut(KvflashRestorePlan::Consume, 4096) == 4096);
    CHECK(kvflash_server_cut(KvflashRestorePlan::AdoptDense, 8089) == 8089);
    CHECK(kvflash_server_cut(KvflashRestorePlan::FullReprefill, 8089) == 0);
    CHECK(kvflash_server_cut(KvflashRestorePlan::Reject, 8089) == 0);
}

TEST_CASE(KvflashResumeFixture, the_larger_cut_wins) {
    CHECK(kvflash_pick_resume(8089, 9216) == KvflashResume::Live);
    CHECK(kvflash_pick_resume(8089, 4096) == KvflashResume::Server);
}

TEST_CASE(KvflashResumeFixture, ties_and_missing_live_cut_keep_the_server_snapshot) {
    CHECK(kvflash_pick_resume(4096, 4096) == KvflashResume::Server);
    CHECK(kvflash_pick_resume(4096, -1) == KvflashResume::Server);
    CHECK(kvflash_pick_resume(0, -1) == KvflashResume::Server);
}

TEST_CASE(KvflashResumeFixture, never_a_full_reprefill_while_the_live_session_matches) {
    // FullReprefill plan => server cut 0; any rung beats it.
    const int sc = kvflash_server_cut(KvflashRestorePlan::FullReprefill, 8089);
    CHECK(kvflash_pick_resume(sc, 512) == KvflashResume::Live);
}

// ── adopting a dense snapshot at the pool boundary ──────────────────

TEST_CASE(KvflashResumeFixture, dense_over_pool_adopts_only_when_enabled) {
    CHECK(kvflash_restore_plan(false, false, 8089, 64, true, 9435, 8192, /*adopt_dense=*/true)
          == KvflashRestorePlan::AdoptDense);
    CHECK(kvflash_restore_plan(false, false, 8089, 64, true, 9435, 8192, false)
          == KvflashRestorePlan::FullReprefill);
    // Inside the pool: still plain dense, adopt or not.
    CHECK(kvflash_restore_plan(false, false, 7039, 64, true, 8000, 8192, true)
          == KvflashRestorePlan::Dense);
    // Without the full prompt there is nothing to adopt into.
    CHECK(kvflash_restore_plan(false, false, 8089, 64, false, 9435, 8192, true)
          == KvflashRestorePlan::Dense);
}

// Dense prefill places chunk c in block c. A fresh pager hands blocks out in
// order, so registering [0, P) reproduces that layout with no data movement,
// partial last chunk included.
TEST_CASE(KvflashResumeFixture, adopting_dense_rows_is_an_identity_mapping) {
    KvFlashPager pager;
    KvFlashConfig c;
    c.pool_tokens = 512; c.chunk_tokens = 64; c.sink_chunks = 1; c.tail_window_chunks = 4;
    CHECK(pager.attach(c, {}, {}));
    for (int P : {64, 100, 320, 449}) {
        pager.reset();
        CHECK(pager.alloc_span(0, P));
        const int nc = (P + 63) / 64;
        CHECK(pager.n_chunks() == nc);
        CHECK(pager.is_identity());
        CHECK(pager.identity_prefix_covers(P));
        for (int ch = 0; ch < nc; ch++) CHECK(pager.block_of(ch) == ch);
        CHECK(pager.slot_of(P - 1) == P - 1);
        // The next token continues in the same (partial) chunk or the next block.
        CHECK(pager.slot_for(P) == P);
    }
}

// ── dense ladder rungs ──────────────────────────────────────────────

namespace {
// Positions where a dense prefill of [kv_offset, end) commits a chunk.
std::vector<int> dense_chunk_ends(int kv_offset, int end) {
    std::vector<int> out;
    for (int pos = kv_offset; pos < end;) {
        pos += qwen35_prefill_chunk_tokens(pos, end - pos, kQwen35PrefillUbatch, {},
                                           kQwen35MinChunkTokens);
        out.push_back(pos);
    }
    return out;
}
}

TEST_CASE(KvflashResumeFixture, dense_rungs_on_the_grid_are_reached_by_the_dense_chunker) {
    // A cold dense prefill and a restored one (offset = server cut) both cross
    // the 512 grid, so the end-of-prompt rung is captured either way.
    for (int off : {0, 2732, 6749, 8089}) {
        for (int len : {600, 1339, 2000}) {
            const int end = off + len;
            const auto rungs = kvflash_state_ladder_positions(off, end, kQwen35PrefillUbatch, 8192);
            const int tail = end / kQwen35PrefillUbatch * kQwen35PrefillUbatch;
            const auto ends = dense_chunk_ends(off, end);
            CHECK(std::binary_search(rungs.begin(), rungs.end(), tail));
            CHECK(std::find(ends.begin(), ends.end(), tail) != ends.end());
        }
    }
}

TEST_CASE(KvflashResumeFixture, dense_rung_bounds_replay_to_one_cell_plus_new_tokens) {
    // Turn k ends at 8095 (+ 13 answer tokens in the history). Turn k+1's
    // prompt shares the first 8108 tokens: the rung is 7680, so replay is
    // < 512 + answer + new tokens.
    const std::vector<int> rungs = kvflash_state_ladder_positions(8089, 8095, 512, 8192);
    CHECK(rungs.back() == 7680);
    CHECK(kvflash_ladder_pick(rungs, 8108) == 7680);
    CHECK(8108 - 7680 < 512 + 13);
}
