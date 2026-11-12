// Unit test for the mode-4 same-question decision (no GPU): when a turn runs once from the last exact scores
// (reuse), once after a scoring from the held tail queries (prescore), or twice with the rewind (full).

#include "CppUnitTestFramework.hpp"

#include "kvflash_m4_reuse.h"
#include "kvflash_prefill_reselect.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashM4ReuseFixture {};
constexpr int CT = 64;
std::vector<int32_t> seq(int n) { std::vector<int32_t> v((size_t)n); for (int i = 0; i < n; i++) v[(size_t)i] = i; return v; }
std::vector<int32_t> extend(std::vector<int32_t> v, int n) { for (int i = 0; i < n; i++) v.push_back(1000000 + i); return v; }
// The last request: a 6400-token prompt, question key {900, 7}, scores over its first 90 chunks, tail queries held.
KvFlashM4Last last_scored() {
    KvFlashM4Last l;
    l.valid = true; l.uq_hi = 900; l.uq_hash = 7; l.ids = seq(6400);
    l.scores.assign(90, 0.5f); l.q_tail.assign(8, 1.0f);
    return l;
}
// The next request: same question, restored at 6336 (>= the 90 scored chunks = 5760 tokens), window starting at chunk 80.
KvFlashM4Ask ask() { KvFlashM4Ask a; a.uq_hi = 900; a.uq_hash = 7; a.uq_stored = true; a.kv_offset = 6336; a.chunk_tokens = CT; a.first_window = 80; return a; }
}

TEST_CASE(KvflashM4ReuseFixture, no_previous_request_runs_the_full_path) {
    CHECK(kvflash_m4_path(KvFlashM4Last{}, ask(), extend(seq(6400), 300)) == KvFlashM4Path::Full);
}
TEST_CASE(KvflashM4ReuseFixture, same_question_with_every_candidate_scored_reuses) {
    CHECK(kvflash_m4_path(last_scored(), ask(), extend(seq(6400), 300)) == KvFlashM4Path::Reuse);
    KvFlashM4Ask a = ask(); a.first_window = 90;   // the window starts right after the last scored chunk
    CHECK(kvflash_m4_path(last_scored(), a, extend(seq(6400), 300)) == KvFlashM4Path::Reuse);
}
TEST_CASE(KvflashM4ReuseFixture, an_unscored_chunk_left_of_the_window_never_reuses) {
    // mode 5's failure: a chunk the last scoring never saw must not be ranked by a made-up score
    KvFlashM4Ask a = ask(); a.first_window = 91;
    CHECK(kvflash_m4_path(last_scored(), a, extend(seq(6400), 300)) == KvFlashM4Path::Prescore);
    KvFlashM4Last l = last_scored(); l.q_tail.clear();   // nothing to score with before the pass
    CHECK(kvflash_m4_path(l, a, extend(seq(6400), 300)) == KvFlashM4Path::Full);
}
TEST_CASE(KvflashM4ReuseFixture, a_new_question_runs_the_full_path) {
    KvFlashM4Ask a = ask(); a.uq_hash = 8;
    CHECK(kvflash_m4_path(last_scored(), a, extend(seq(6400), 300)) == KvFlashM4Path::Full);
    a = ask(); a.uq_hi = 950;
    CHECK(kvflash_m4_path(last_scored(), a, extend(seq(6400), 300)) == KvFlashM4Path::Full);
    a = ask(); a.uq_hi = -1;   // no human turn found
    KvFlashM4Last l = last_scored(); l.uq_hi = -1;
    CHECK(kvflash_m4_path(l, a, extend(seq(6400), 300)) == KvFlashM4Path::Full);
}
TEST_CASE(KvflashM4ReuseFixture, question_rows_not_stored_run_the_full_path) {
    KvFlashM4Ask a = ask(); a.uq_stored = false;
    CHECK(kvflash_m4_path(last_scored(), a, extend(seq(6400), 300)) == KvFlashM4Path::Full);
}
TEST_CASE(KvflashM4ReuseFixture, another_conversation_runs_the_full_path) {
    std::vector<int32_t> other = seq(6400);
    other[(size_t)(6400 - kKvFlashM4TailSlack - 1)] = -1;
    CHECK(kvflash_m4_path(last_scored(), ask(), extend(other, 300)) == KvFlashM4Path::Full);
    CHECK(kvflash_m4_path(last_scored(), ask(), seq(6000)) == KvFlashM4Path::Full);   // shorter prompt
}
TEST_CASE(KvflashM4ReuseFixture, a_rerendered_generation_prompt_still_continues) {
    std::vector<int32_t> next = seq(6400);
    for (int i = 6400 - kKvFlashM4TailSlack; i < 6400; i++) next[(size_t)i] = -1;
    CHECK(kvflash_m4_path(last_scored(), ask(), extend(next, 300)) == KvFlashM4Path::Reuse);
}
TEST_CASE(KvflashM4ReuseFixture, scores_of_rewritten_or_changed_chunks_are_not_reused) {
    KvFlashM4Ask a = ask(); a.kv_offset = 5759;   // the restore re-prefills a row of scored chunk 89
    CHECK(kvflash_m4_path(last_scored(), a, extend(seq(6400), 300)) == KvFlashM4Path::Prescore);
    // scores reach into the slack and the prompt differs there
    KvFlashM4Last l = last_scored(); l.scores.assign(100, 0.5f);   // 6400 tokens scored
    std::vector<int32_t> next = seq(6400); next[6399] = -1;
    a = ask(); a.kv_offset = 6400; a.first_window = 80;
    CHECK(kvflash_m4_path(l, a, extend(next, 300)) == KvFlashM4Path::Prescore);
}
TEST_CASE(KvflashM4ReuseFixture, no_scores_yet_prescores_from_the_held_queries) {
    // the pool held the whole prompt until now: queries were kept, nothing was scored
    KvFlashM4Last l = last_scored(); l.scores.clear();
    CHECK(kvflash_m4_path(l, ask(), extend(seq(6400), 300)) == KvFlashM4Path::Prescore);
}
TEST_CASE(KvflashM4ReuseFixture, first_window_is_the_one_the_recall_plan_uses) {
    KvFlashRecallIn in;
    in.n_pool_blocks = 256; in.cur_chunk = 1400; in.recall_chunks = 60; in.unit = 2; in.reserve = 5; in.pin_reserve_chunks = 18; in.pinned_chunks = 17;
    const std::vector<float> scores(1400, 0.0f);
    CHECK_EQUAL(kvflash_recall_first_window(in), kvflash_recall_plan(scores, in).first_window);
    CHECK_EQUAL(kvflash_recall_first_window(in), 1400 - (256 - 18 - 1 - 60 - 5 - 1));
}
// The property the rule exists for: on a reuse turn the pick over the stale scores equals the pick a scorer that
// returned the same values for the scored chunks would make, whatever it returned for the chunks inside the window.
TEST_CASE(KvflashM4ReuseFixture, reuse_pick_ignores_whatever_the_window_chunks_would_score) {
    KvFlashRecallIn in;
    in.n_pool_blocks = 64; in.cur_chunk = 120; in.recall_chunks = 10; in.unit = 2; in.reserve = 3; in.pin_reserve_chunks = 0; in.pinned_chunks = 0;
    const int fw = kvflash_recall_first_window(in);   // 120 - (64 - 1 - 10 - 3 - 1) = 71
    std::vector<float> stale((size_t)fw + 4);          // the last scoring covered 4 chunks past the window start
    for (size_t c = 0; c < stale.size(); c++) stale[c] = (float)((c * 37) % 23);
    std::vector<float> fresh = stale;
    fresh.resize(120, 1e9f);                           // window chunks with huge scores
    CHECK(kvflash_recall_plan(stale, in).picked == kvflash_recall_plan(fresh, in).picked);
    CHECK_EQUAL((int)kvflash_recall_plan(stale, in).picked.size(), 5);
}
