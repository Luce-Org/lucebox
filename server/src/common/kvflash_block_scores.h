// Per-block chunk-score sums (pure, no ggml): R-LRU recall picks whole blocks of chunks.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace luce::common {

// Sum of chunk scores over each block of `cpb` chunks, block b = chunks [b*cpb, (b+1)*cpb); only blocks that lie
// entirely inside the first `n_chunks` scores are returned.
inline std::vector<double> kvflash_block_scores(const std::vector<float> & chunk_scores, int cpb,
                                                size_t n_chunks = (size_t)-1) {
    std::vector<double> bs;
    if (cpb <= 0) return bs;
    const size_t n = std::min(n_chunks, chunk_scores.size());
    for (size_t b = 0; (b + 1) * cpb <= n; b++) {
        double s = 0.0;
        for (size_t c = b * cpb; c < (b + 1) * cpb; c++) s += chunk_scores[c];
        bs.push_back(s);
    }
    return bs;
}

}  // namespace luce::common
