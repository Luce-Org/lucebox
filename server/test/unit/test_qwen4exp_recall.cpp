// Pure host checks of the QSA-indexer KVFlash recall scorer (LUCE_Q4_INDEXER_RECALL): vote aggregation, newer-chunk
// tie-break, top list, the two-pass tail's start and knob. No model, backend or GPU.
#include "qwen4exp/qwen4exp_recall.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
using namespace luce::common;
int main() {
    // 2 layers x 2 queries x budget 3, 4 blocks per chunk, 3 chunks (blocks 0..11).
    const std::vector<int32_t> ids = {
        0, 1, 5,    // L0 q0: chunk 0, 0, 1
        5, 6, 11,   // L0 q1: chunk 1, 1, 2
        4, 4, 4,    // L1 q0: chunk 1 x3 (duplicates count)
        2, 9, 10,   // L1 q1: chunk 0, 2, 2
    };
    auto v = q4_recall_chunk_votes(ids, 2, 2, 3, 4, 3);
    assert(v.size() == 3);
    assert(v[0] == 3.f && v[1] == 6.f && v[2] == 3.f);
    float total = 0; for (float x : v) total += x;
    assert(total == 12.f);  // every id lands in exactly one chunk

    // Out-of-range ids (negative, past the last chunk) are dropped.
    auto o = q4_recall_chunk_votes({-1, 12, 99, 3}, 1, 1, 4, 4, 3);
    assert(o[0] == 1.f && o[1] == 0.f && o[2] == 0.f);

    // Per-query visibility: query 0 sees blocks < 2, query 1 sees blocks < 12; invisible picks (top-k padding) drop.
    const std::vector<int> visible = {2, 12};
    auto m = q4_recall_chunk_votes(ids, 2, 2, 3, 4, 3, &visible);
    // L0 q0 keeps 0,1 (chunk 0); L1 q0 keeps none (ids 4 >= 2); q1 rows keep all.
    assert(m[0] == 2.f + 1.f && m[1] == 2.f && m[2] == 3.f);

    // Scores: integer votes, ties broken to the newer chunk, never overturning a vote difference.
    auto s = q4_recall_chunk_scores({3.f, 3.f, 0.f, 0.f, 4.f});
    assert(s[1] > s[0] && s[3] > s[2]);
    assert(s[4] > s[1] && s[0] > s[3]);
    for (size_t i = 0; i < s.size(); ++i) assert(std::floor(s[i]) == std::floor(std::vector<float>{3, 3, 0, 0, 4}[i]));

    // Top list.
    assert((q4_recall_top(s, 3) == std::vector<int>{4, 1, 0}));
    assert(q4_recall_top(s, 99).size() == 5);

    // Two-pass tail: starts `tail` tokens before the prompt's end, never before the restored prefix, and only past the pool.
    assert(q4_recall_tail_pos(0, 10000, 2048, 4096) == 10000 - 2048);
    assert(q4_recall_tail_pos(9500, 10000, 2048, 4096) == 9500);     // a restored prefix inside the tail: start there
    assert(q4_recall_tail_pos(0, 6000, 2048, 4096) == -1);           // tail starts at 3952 <= pool: single pass
    assert(q4_recall_tail_pos(0, 6144, 2048, 4096) == -1);           // tail starts exactly at the pool: nothing paged before it
    assert(q4_recall_tail_pos(0, 6145, 2048, 4096) == 4097);
    assert(q4_recall_tail_pos(0, 10000, 0, 4096) == -1);             // tail off
    assert(q4_recall_tail_pos(0, 100, 2048, 4096) == -1);            // tail longer than the prompt
    // The path a mode-2 request takes, as the one [q4-tail] line names it.
    using P = Q4TailPath;
    assert(q4_recall_tail_path(0, 10000, 0, 4096) == P::Off);
    assert(q4_recall_tail_path(0, 3000, 2048, 4096) == P::FitsPool);       // nothing is paged out
    assert(q4_recall_tail_path(0, 4096, 2048, 4096) == P::FitsPool);
    assert(q4_recall_tail_path(0, 6000, 2048, 4096) == P::TailInPool);     // pool < P <= pool + tail: recall after prefill
    assert(q4_recall_tail_path(0, 10000, 2048, 4096) == P::TwoPass);
    assert(q4_recall_tail_path(9990, 10000, 2048, 4096) == P::TwoPass);    // a restored prefix shortens the tail
    assert(q4_recall_tail_path(9999, 10000, 2048, 4096) == P::TwoPass);     // a one-token tail still recalls from the earlier rows
    assert(q4_recall_tail_path(10000, 10000, 2048, 4096) == P::NoForward);   // exact repeat: no forward at all
    // The [q4-tail] line names what ran: a planned two-pass run reports how far it got, every other path is itself.
    using S = Q4TailStage;
    assert(std::string(q4_tail_outcome(P::FitsPool, S::Planned)) == q4_tail_outcome(P::FitsPool, S::Rewound));
    const std::string two[] = {q4_tail_outcome(P::TwoPass, S::Planned), q4_tail_outcome(P::TwoPass, S::SnapFailed),
                               q4_tail_outcome(P::TwoPass, S::RewindFailed), q4_tail_outcome(P::TwoPass, S::Rewound)};
    for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) assert(two[i] != two[j]);
    assert(two[1].find("single pass") == 0);                              // a failed tail snapshot runs a single pass
    assert(two[2].find("prefilled twice") == std::string::npos);          // a failed rewind never replayed the tail
    assert(two[3].find("rewound") != std::string::npos);
    // The mode after context fitting settled the layout: attention on the expert device refuses recall, no pool left
    // downgrades recall to score only, and each change carries the one startup line's reason.
    auto st = q4_recall_settle(2, 0, true);
    assert(st.mode == 2 && st.why == nullptr);
    st = q4_recall_settle(2, 0, false);
    assert(st.mode == 1 && st.why != nullptr);
    st = q4_recall_settle(2, 3, false);
    assert(st.mode == 0 && st.why != nullptr);
    st = q4_recall_settle(1, 3, false);
    assert(st.mode == 0 && st.why != nullptr);
    st = q4_recall_settle(1, 0, false);                         // score only needs no pool
    assert(st.mode == 1 && st.why == nullptr);
    st = q4_recall_settle(0, 3, false);                         // off stays off, silently
    assert(st.mode == 0 && st.why == nullptr);
    // Env integers: the whole string, in range; unset or empty is the default without complaint, anything else the
    // default and a refusal for the caller to name.
    auto e = q4_parse_env_int(nullptr, 0, 2, 0);
    assert(e.value == 0 && !e.refused);
    e = q4_parse_env_int("", 2, 256, 64);
    assert(e.value == 64 && !e.refused);
    e = q4_parse_env_int("128", 2, 256, 64);
    assert(e.value == 128 && !e.refused);
    e = q4_parse_env_int("2", 2, 256, 64);                      // both bounds are in range
    assert(e.value == 2 && !e.refused);
    e = q4_parse_env_int("256", 2, 256, 64);
    assert(e.value == 256 && !e.refused);
    for (const char * bad : {"2048junk", " 5", "5 ", "x", "-", "+", "1.5", "0x10", "1", "257", "-5", "99999999999999999999",
                             "2147483648", "-99999999999999999999"}) {
        e = q4_parse_env_int(bad, 2, 256, 64);
        assert(e.value == 64 && e.refused);
    }
    e = q4_parse_env_int("-5", -10, 10, 0);                     // a sign is fine when the range allows it
    assert(e.value == -5 && !e.refused);
    // LUCE_Q4_RECALL_TAIL: 0 = off, up to the largest context (2^24 - 1) the cache accepts.
    assert(q4_parse_env_int("0", 0, kQ4RecallTailMax, kQ4RecallTailDefault).value == 0);
    assert(q4_parse_env_int("16777216", 0, kQ4RecallTailMax, kQ4RecallTailDefault).refused);

    // Empty inputs.
    assert(q4_recall_chunk_votes({}, 0, 0, 0, 4, 0).empty());
    assert(q4_recall_chunk_votes({}, 1, 1, 1, 0, 3) == std::vector<float>(3, 0.f));  // bad blocks_per_chunk: no votes
    std::printf("test_qwen4exp_recall OK\n");
    return 0;
}
