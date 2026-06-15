// Compute [score_layer_start, score_layer_end) for tail-attention scoring.
// SCORE_LAYERS counts from the END of [0, fwd_layer_limit); -1 = all computed layers.
#pragma once

#include <algorithm>

namespace dflash::common {

struct ScoreRange {
    int start; // inclusive
    int end;   // exclusive
    int count() const { return end - start; }
    bool empty() const { return start >= end; }
};

// Returns scoring layer range within [0, fwd_layer_limit).
inline ScoreRange compute_score_range(int n_layer, int score_layers, int fwd_layer_limit) {
    const int effective_n = fwd_layer_limit;
    int start;
    if (score_layers > 0 && score_layers < n_layer) {
        int want = std::min(score_layers, effective_n);
        start = effective_n - want;
    } else {
        start = 0;
    }
    int end = fwd_layer_limit;
    if (start > end) start = end;
    return { start, end };
}

} // namespace dflash::common
