// Exact drafter-session resume (pflash_scoring_resume.h): the grid checkpoint, the exact-rows bound, the resume
// rule and the F32 query-set store. Model-free.

#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_scoring_resume.h"

#include <algorithm>
#include <vector>

using namespace luce::pflash;

namespace {

struct PFlashScoringResumeFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

// Ubatch boundaries of a call over S tokens resumed at r (the drafter's ubatch_at): 1024 steps from r, cut at S-64.
std::vector<int> ubatch_starts(int r, int S) {
    std::vector<int> out;
    const int cut = std::max(r, S - kScoringCutBack);
    for (int start = r; start < S;) {
        out.push_back(start);
        start += std::min(kScoringUbatch, (start < cut ? cut : S) - start);
    }
    return out;
}

} // namespace

TEST_CASE(PFlashScoringResumeFixture, checkpoint_is_the_last_grid_point_before_the_cut) {
    CHECK(pflash_scoring_checkpoint(0) == 0);
    CHECK(pflash_scoring_checkpoint(1024) == 0);                 // cut 960 < 1024
    CHECK(pflash_scoring_checkpoint(8192 + 64) == 8192);
    CHECK(pflash_scoring_checkpoint(8192 + 63) == 7168);
    CHECK(pflash_scoring_checkpoint(63565) == 63488);
    CHECK(pflash_scoring_checkpoint(65600) == 65536);
    CHECK(pflash_scoring_checkpoint(65599) == 64512);
}

TEST_CASE(PFlashScoringResumeFixture, exact_rows_end_is_the_last_full_ubatch_before_the_cut) {
    CHECK(pflash_scoring_exact_rows_end(1000) == 0);
    CHECK(pflash_scoring_exact_rows_end(1024 + 64) == 1024);
    CHECK(pflash_scoring_exact_rows_end(63565) == 63488);
    for (int S : {9000, 63565, 70000, 131072}) {
        CHECK(pflash_scoring_exact_rows_end(S) >= pflash_scoring_checkpoint(S));
        CHECK(pflash_scoring_exact_rows_end(S) <= S - kScoringCutBack);
    }
}

TEST_CASE(PFlashScoringResumeFixture, resume_only_at_a_grid_checkpoint_inside_the_shared_prefix) {
    // A follow-up: the previous prompt's checkpoint, all of it shared.
    CHECK(pflash_scoring_resume(63488, 63560, 63790, 1 << 30) == 63488);
    // Diverged before the checkpoint (an edit): fresh.
    CHECK(pflash_scoring_resume(57344, 57343, 63790, 1 << 30) == 0);
    // The checkpoint itself shared is enough.
    CHECK(pflash_scoring_resume(57344, 57344, 63790, 1 << 30) == 57344);
    // A shorter prompt whose S - 64 cut falls before the checkpoint: fresh.
    CHECK(pflash_scoring_resume(57344, 57380, 57380, 1 << 30) == 0);
    CHECK(pflash_scoring_resume(57344, 57408, 57408, 1 << 30) == 57344);
    // The cap (a redo) and an off-grid or empty checkpoint.
    CHECK(pflash_scoring_resume(57344, 63560, 63790, 0) == 0);
    CHECK(pflash_scoring_resume(57344, 63560, 63790, 57343) == 0);
    CHECK(pflash_scoring_resume(57345, 63560, 63790, 1 << 30) == 0);
    CHECK(pflash_scoring_resume(0, 63560, 63790, 1 << 30) == 0);
    // A retry (the same prompt): resumes at its own checkpoint.
    CHECK(pflash_scoring_resume(57344, 63565, 63565, 1 << 30) == 57344);
}

TEST_CASE(PFlashScoringResumeFixture, a_resumed_call_lands_on_the_fresh_partition) {
    // For every resume the rule allows, the ubatches from the resume point are exactly the
    // fresh call's from that point on.
    for (int S : {16500, 63565, 63790, 65600, 74200, 98304 + 64}) {
        const std::vector<int> fresh = ubatch_starts(0, S);
        for (int prev : {16400, 57400, 63565, 65599, 65600, 74100}) {
            const int r = pflash_scoring_resume(pflash_scoring_checkpoint(prev), std::min(prev, S), S, 1 << 30);
            if (r == 0) continue;
            CHECK(r % kScoringResumeGrid == 0);
            std::vector<int> tail;
            for (const int b : fresh) {
                if (b >= r) tail.push_back(b);
            }
            CHECK(ubatch_starts(r, S) == tail);
            // The fresh call has a ubatch boundary at r (so the state there is the one the snapshot holds).
            bool boundary = false;
            for (const int b : fresh) boundary = boundary || b == r;
            CHECK(boundary);
        }
    }
}

// The 1024 grid: a call resumes close to where the previous one ended; the 8192 projection chunk that holds the
// resume point is re-projected whole from the rows the session kept.
TEST_CASE(PFlashScoringResumeFixture, checkpoint_is_on_the_ubatch_grid) {
    CHECK(kScoringResumeGrid == kScoringUbatch);
    CHECK(kScoringProjChunk == 8192);
    CHECK(pflash_scoring_checkpoint(1024 + 63) == 0);
    CHECK(pflash_scoring_checkpoint(1024 + 64) == 1024);
    CHECK(pflash_scoring_checkpoint(21673) == 21504);
    CHECK(pflash_scoring_checkpoint(32334) == 31744);
    for (int S : {1100, 9000, 21673, 32334, 63565, 131072}) {
        CHECK(pflash_scoring_checkpoint(S) == pflash_scoring_exact_rows_end(S));
    }
}

TEST_CASE(PFlashScoringResumeFixture, a_longer_prompt_resumes_at_the_previous_calls_checkpoint) {
    // S=21673 then S=32334 (the lucebox4 night run): resume at 21504, not 16384.
    const int cp = pflash_scoring_checkpoint(21673);
    CHECK(cp == 21504);
    CHECK(pflash_scoring_resume(cp, 21673, 32334, 1 << 30) == 21504);
    CHECK(pflash_scoring_resume(cp, 21503, 32334, 1 << 30) == 0);       // diverged before it
    CHECK(pflash_scoring_resume(21505, 32000, 32334, 1 << 30) == 0);    // off the grid
}

TEST_CASE(PFlashScoringResumeFixture, projection_chunk_start_and_stored_rows) {
    CHECK(pflash_scoring_proj_begin(0) == 0);
    CHECK(pflash_scoring_proj_begin(8191) == 0);
    CHECK(pflash_scoring_proj_begin(8192) == 8192);
    CHECK(pflash_scoring_proj_begin(21504) == 16384);
    // Rows [16384, 21504) cover a resume at 21504; fewer, a wrong start, or none do not.
    CHECK(pflash_scoring_proj_rows_cover(21504, 16384, 21504 - 16384));
    CHECK(pflash_scoring_proj_rows_cover(21504, 16384, 21504 - 16384 + 1024));
    CHECK(!pflash_scoring_proj_rows_cover(21504, 16384, 21504 - 16384 - 1));
    CHECK(!pflash_scoring_proj_rows_cover(21504, 8192, 21504 - 8192));
    // A resume on a chunk start needs no rows.
    CHECK(pflash_scoring_proj_rows_cover(16384, 16384, 0));
    CHECK(pflash_scoring_proj_rows_cover(16384, 0, 0));
    CHECK(pflash_scoring_proj_rows_cover(0, 0, 0));
}

TEST_CASE(PFlashScoringResumeFixture, resumed_projection_chunks_are_the_fresh_chunks) {
    // From proj_begin(r) in steps of 8192 (the last cut at S) the chunk starts are the fresh call's from there on.
    for (int S : {21673, 32334, 63565, 65600}) {
        std::vector<int> fresh;
        for (int b = 0; b < S; b += kScoringProjChunk) fresh.push_back(b);
        for (int prev : {9000, 21673, 40000, 63565}) {
            const int r = pflash_scoring_resume(pflash_scoring_checkpoint(prev), std::min(prev, S), S, 1 << 30);
            if (r == 0) continue;
            const int pb = pflash_scoring_proj_begin(r);
            CHECK(pb % kScoringProjChunk == 0);
            CHECK(pb <= r);
            CHECK(r - pb < kScoringProjChunk);
            std::vector<int> mine;
            for (int b = pflash_scoring_proj_begin(r); b < S; b += kScoringProjChunk) mine.push_back(b);
            CHECK(mine == std::vector<int>(std::find(fresh.begin(), fresh.end(), pflash_scoring_proj_begin(r)),
                                           fresh.end()));
        }
    }
}

TEST_CASE(PFlashScoringResumeFixture, query_sets_store_exact_rows_only) {
    PFlashExactRowSets sets(2);
    const int hidden = 2;
    const float rows3[] = {1, 2, 3, 4, 5, 6};
    std::vector<float> out;
    // Below the exact end: stored and found by the exact token list.
    CHECK(sets.put({10, 11, 12}, rows3, hidden, 13));
    CHECK(sets.lookup({10, 11, 12}, hidden, out));
    CHECK(out == std::vector<float>(rows3, rows3 + 6));
    CHECK(!sets.lookup({10, 11}, hidden, out));
    // Reaching the exact end, unsorted or empty: refused.
    CHECK(!sets.put({20, 21, 22}, rows3, hidden, 22));
    CHECK(!sets.put({12, 11, 10}, rows3, hidden, 100));
    CHECK(!sets.put({}, rows3, hidden, 100));
    CHECK(!sets.lookup({20, 21, 22}, hidden, out));
    // Bounded: the oldest goes first; an equal set is replaced, not duplicated.
    const float rows1a[] = {7, 8};
    const float rows1b[] = {9, 9};
    CHECK(sets.put({30}, rows1a, hidden, 100));
    CHECK(sets.put({30}, rows1b, hidden, 100));
    CHECK(sets.size() == 2);
    CHECK(sets.lookup({30}, hidden, out));
    CHECK(out == std::vector<float>({9, 9}));
    CHECK(sets.put({40}, rows1a, hidden, 100));
    CHECK(sets.size() == 2);
    CHECK(!sets.lookup({10, 11, 12}, hidden, out));
    // A divergence drops every set that reaches it.
    CHECK(sets.invalidate_from(40) == 1);
    CHECK(sets.lookup({30}, hidden, out));
    CHECK(sets.invalidate_from(30) == 1);
    CHECK(sets.size() == 0);
}

// Query windows are row sets keyed by their positions: a window stored before
// a divergence point stays, one that reaches it is dropped.
TEST_CASE(PFlashScoringResumeFixture, query_windows_are_position_sets) {
    CHECK(pflash_span_tokens(5, 8) == std::vector<int>({5, 6, 7}));
    CHECK(pflash_span_tokens(8, 8).empty());
    PFlashExactRowSets windows;
    const int hidden = 1;
    const float row[] = {1};
    std::vector<float> out;
    CHECK(windows.put(pflash_span_tokens(447, 448), row, hidden, 65536));
    CHECK(windows.put(pflash_span_tokens(12000, 12001), row, hidden, 65536));
    CHECK(windows.invalidate_from(11000) == 1);
    CHECK(windows.lookup(pflash_span_tokens(447, 448), hidden, out));
    CHECK(!windows.lookup(pflash_span_tokens(12000, 12001), hidden, out));
}

// Session capacity: headroom S + S/2 + 4096, clamped to the context limit, never below S padded to 256.
TEST_CASE(PFlashScoringResumeFixture, session_capacity_is_headroom_clamped_to_the_context_limit) {
    using luce::pflash::pflash_session_capacity;
    CHECK(pflash_session_capacity(100000, 0) == 100000 + 50000 + 4096);
    CHECK(pflash_session_capacity(100000, 262144) == 100000 + 50000 + 4096);
    CHECK(pflash_session_capacity(200000, 262144) == 262144);
    CHECK(pflash_session_capacity(217177, 225280) == 225280);
    CHECK(pflash_session_capacity(250000, 225280) == 250112);   // the prompt itself always fits (padded to 256)
    CHECK(pflash_session_capacity(200000, 225000) == 225024);   // limit rounded up to 256
    for (int S : {1, 255, 256, 8191, 65536, 131071, 225279, 262144}) {
        for (int limit : {0, 1, 4096, 131072, 225280, 262144}) {
            const int c = pflash_session_capacity(S, limit);
            CHECK(c >= (S + 255) / 256 * 256);
            CHECK(c <= std::max((S + 255) / 256 * 256, limit > 0 ? (limit + 255) / 256 * 256 : S + S / 2 + 4096));
        }
    }
}

TEST_CASE(PFlashScoringResumeFixture, session_ctx_limit_is_the_smaller_known_context) {
    using luce::pflash::pflash_session_ctx_limit;
    CHECK(pflash_session_ctx_limit(262144, 225280) == 225280);
    CHECK(pflash_session_ctx_limit(262144, 0) == 262144);
    CHECK(pflash_session_ctx_limit(0, 139264) == 139264);
    CHECK(pflash_session_ctx_limit(0, 0) == 0);
    CHECK(pflash_session_ctx_limit(-1, 8192) == 8192);
}

namespace {

std::vector<int32_t> iota_ids(int n) {
    std::vector<int32_t> ids((size_t) n);
    for (int i = 0; i < n; ++i) ids[(size_t) i] = i;
    return ids;
}

} // namespace

TEST_CASE(PFlashScoringResumeFixture, an_undersized_session_still_resumes_and_needs_to_grow) {
    // The measured case: a session of capacity 36605 holds a 21673-token prompt (checkpoint 21504); the next turn has
    // 46124 tokens, all of the old prompt shared.
    const std::vector<int32_t> old_ids = iota_ids(21673);
    const std::vector<int32_t> ids = iota_ids(46124);
    PFlashResumeCandidate c;
    c.ids = &old_ids;
    c.checkpoint = pflash_scoring_checkpoint(21673);
    c.capacity = 36605;
    const PFlashResumePick pick = pflash_pick_resume({c}, ids, 1 << 30);
    CHECK(pick.index == 0);
    CHECK(pick.resume == 21504);
    CHECK(pick.shared == 21673);
    CHECK(pick.needs_grow);
}

TEST_CASE(PFlashScoringResumeFixture, a_fitting_session_needs_no_grow) {
    const std::vector<int32_t> old_ids = iota_ids(21673);
    const std::vector<int32_t> ids = iota_ids(30000);
    PFlashResumeCandidate c;
    c.ids = &old_ids;
    c.checkpoint = 16384;
    c.capacity = 36605;
    const PFlashResumePick pick = pflash_pick_resume({c}, ids, 1 << 30);
    CHECK(pick.index == 0);
    CHECK(pick.resume == 16384);
    CHECK(!pick.needs_grow);
    // Exactly S rounded up to 256 fits.
    c.capacity = 30000 / 256 * 256 + 256;
    CHECK(!pflash_pick_resume({c}, ids, 1 << 30).needs_grow);
    c.capacity -= 256;
    CHECK(pflash_pick_resume({c}, ids, 1 << 30).needs_grow);
}

TEST_CASE(PFlashScoringResumeFixture, a_mismatched_prefix_or_unusable_session_gives_fresh) {
    std::vector<int32_t> old_ids = iota_ids(21673);
    old_ids[100] = -1;   // diverges before the checkpoint
    const std::vector<int32_t> ids = iota_ids(46124);
    PFlashResumeCandidate c;
    c.ids = &old_ids;
    c.checkpoint = 16384;
    c.capacity = 36605;
    PFlashResumePick pick = pflash_pick_resume({c}, ids, 1 << 30);
    CHECK(pick.index == -1);
    CHECK(pick.resume == 0);
    CHECK(!pick.needs_grow);
    const std::vector<int32_t> good = iota_ids(21673);
    c.ids = &good;
    c.usable = false;
    CHECK(pflash_pick_resume({c}, ids, 1 << 30).index == -1);
}

TEST_CASE(PFlashScoringResumeFixture, a_checkpoint_past_the_cut_gives_fresh_even_when_undersized) {
    // S = 16384 + 63: the S - 64 cut falls before the checkpoint 16384.
    const std::vector<int32_t> old_ids = iota_ids(20000);
    const std::vector<int32_t> ids = iota_ids(16384 + 63);
    PFlashResumeCandidate c;
    c.ids = &old_ids;
    c.checkpoint = 16384;
    c.capacity = 100;
    CHECK(pflash_pick_resume({c}, ids, 1 << 30).index == -1);
    CHECK(pflash_pick_resume({c}, iota_ids(16384 + 64), 1 << 30).resume == 16384);
    // The redo cap still applies.
    CHECK(pflash_pick_resume({c}, iota_ids(16384 + 64), 0).index == -1);
}

TEST_CASE(PFlashScoringResumeFixture, the_best_resume_wins_and_a_fitting_session_wins_a_tie) {
    const std::vector<int32_t> a_ids = iota_ids(30000), b_ids = iota_ids(50000);
    const std::vector<int32_t> ids = iota_ids(60000);
    PFlashResumeCandidate a, b;
    a.ids = &a_ids; a.checkpoint = pflash_scoring_checkpoint(30000); a.capacity = 70000;
    b.ids = &b_ids; b.checkpoint = pflash_scoring_checkpoint(50000); b.capacity = 51000;
    PFlashResumePick pick = pflash_pick_resume({a, b}, ids, 1 << 30);
    CHECK(pick.index == 1);
    CHECK(pick.resume == 49152);
    CHECK(pick.needs_grow);
    b.checkpoint = a.checkpoint;   // equal resume points: the one that fits
    pick = pflash_pick_resume({b, a}, ids, 1 << 30);
    CHECK(pick.index == 1);
    CHECK(!pick.needs_grow);
}

TEST_CASE(PFlashScoringResumeFixture, the_row_check_can_send_a_session_back_to_fresh) {
    const std::vector<int32_t> old_ids = iota_ids(21673);
    const std::vector<int32_t> ids = iota_ids(46124);
    PFlashResumeCandidate c;
    c.ids = &old_ids;
    c.checkpoint = 16384;
    c.capacity = 36605;
    c.rows_cover = [](int, int) { return 0; };   // the stored rows do not cover the resume point
    CHECK(pflash_pick_resume({c}, ids, 1 << 30).index == -1);
    c.rows_cover = [](int r, int) { return r; };
    CHECK(pflash_pick_resume({c}, ids, 1 << 30).resume == 16384);
}
