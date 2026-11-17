// QSA-indexer KVFlash recall scorer (LUCE_Q4_INDEXER_RECALL): pure host aggregation of the indexer's top-k block picks
// into one score per KVFlash chunk. Header-only so the unit test builds without a model, backend or GPU.
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace luce::common {

// Votes per chunk. `ids` is [layer][query][budget] block ids (the indexer's ggml_top_k output, layer-major); block b
// lies in chunk b / blocks_per_chunk. A chunk's vote is the picks inside it summed over every layer and query.
// `visible` (n_queries entries, optional) is each query's complete-block count: a pick at or past it is top-k padding
// of an invisible (-inf) block and does not vote. Ids outside [0, n_chunks * blocks_per_chunk) are dropped.
inline std::vector<float> q4_recall_chunk_votes(const std::vector<int32_t> & ids, int n_layers, int n_queries, int budget,
                                                int blocks_per_chunk, int n_chunks,
                                                const std::vector<int> * visible = nullptr) {
    std::vector<float> votes((size_t) std::max(0, n_chunks), 0.f);
    if (blocks_per_chunk <= 0 || n_chunks <= 0) return votes;
    for (int l = 0; l < n_layers; ++l) {
        for (int q = 0; q < n_queries; ++q) {
            const size_t row = ((size_t) l * n_queries + q) * budget;
            const int64_t vis = visible ? (*visible)[(size_t) q] : INT64_MAX;
            for (int i = 0; i < budget && row + i < ids.size(); ++i) {
                const int64_t b = ids[row + i];
                if (b < 0 || b >= vis || b / blocks_per_chunk >= n_chunks) continue;
                votes[(size_t) (b / blocks_per_chunk)] += 1.f;
            }
        }
    }
    return votes;
}

// Votes plus a tie-break below one vote: equal votes go to the newer chunk.
inline std::vector<float> q4_recall_chunk_scores(const std::vector<float> & votes) {
    std::vector<float> s(votes.size());
    for (size_t c = 0; c < votes.size(); ++c) s[c] = votes[c] + 0.5f * (float) (c + 1) / (float) (votes.size() + 1);
    return s;
}

// The n best chunks, best first.
inline std::vector<int> q4_recall_top(const std::vector<float> & scores, int n) {
    std::vector<int> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    n = std::min<int>(n, (int) idx.size());
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(), [&](int a, int b) { return scores[(size_t) a] > scores[(size_t) b]; });
    idx.resize((size_t) n);
    return idx;
}

}  // namespace luce::common
