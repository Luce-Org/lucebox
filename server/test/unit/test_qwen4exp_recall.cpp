// Pure host checks of the QSA-indexer KVFlash recall scorer (LUCE_Q4_INDEXER_RECALL): vote aggregation, newer-chunk
// tie-break, top list. No model, backend or GPU.
#include "qwen4exp/qwen4exp_recall.h"
#include <cassert>
#include <cmath>
#include <cstdio>
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

    // Empty inputs.
    assert(q4_recall_chunk_votes({}, 0, 0, 0, 4, 0).empty());
    assert(q4_recall_chunk_votes({}, 1, 1, 1, 0, 3) == std::vector<float>(3, 0.f));  // bad blocks_per_chunk: no votes
    std::printf("test_qwen4exp_recall OK\n");
    return 0;
}
