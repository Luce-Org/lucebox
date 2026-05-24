// Pure helper: compute the [score_layer_start, score_layer_end) range for
// tail-attention scoring given the forward-pass layer limit and the optional
// SCORE_LAYERS count.
//
// Parameters:
//   n_layer        - total number of layers in the model (e.g. 28)
//   score_layers   - value of PFLASH_DRAFTER_SCORE_LAYERS (-1 = all)
//   fwd_layer_limit - number of layers actually computed (== early_exit_n when
//                    early-exit is active, else n_layer)
//
// Semantics: SCORE_LAYERS is interpreted as "how many of the computed layers
// to score", counted from the END of the forward range [0, fwd_layer_limit).
// This way SCORE_LAYERS=7 with early_exit_n=7 scores layers [0,7) instead of
// producing the empty interval [7,7) that the old code yielded.
#pragma once

#include <algorithm>

namespace dflash::common {

struct ScoreRange {
    int start; // inclusive
    int end;   // exclusive
    int count() const { return end - start; }
    bool empty() const { return start >= end; }
};

// Compute the scoring layer range.
// When early-exit is active, SCORE_LAYERS counts from 0 upward within the
// computed range [0, fwd_layer_limit), not from the end of the full model.
inline ScoreRange compute_score_range(int n_layer, int score_layers, int fwd_layer_limit) {
    // score_layers <= 0 means "use all computed layers"
    const int effective_n = fwd_layer_limit;
    int start;
    if (score_layers > 0 && score_layers < n_layer) {
        // Clamp: can't request more layers than were computed.
        int want = std::min(score_layers, effective_n);
        start = effective_n - want;
    } else {
        start = 0;
    }
    int end = fwd_layer_limit;
    // Clamp start to never exceed end.
    if (start > end) start = end;
    return { start, end };
}

} // namespace dflash::common
