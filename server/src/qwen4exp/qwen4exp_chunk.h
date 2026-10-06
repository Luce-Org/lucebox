#pragma once

#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace luce::common {

// Largest 256-row tile multiple within a measured workspace budget. Above
// 512, the dense/QSA peak envelope grows with T. Small contexts / low-memory
// fallbacks also try 256, 128, ... 1. Keep 10% of available memory for runtime,
// driver and OS growth; private backend scratch is an estimate, not gallocr.
template<class Measure>
int qwen4exp_fit_chunk(int max_ctx, size_t available, size_t fixed, Measure measure) {
    const size_t budget = available - available / 10;
    if (max_ctx <= 0 || fixed >= budget) return 0;
    int best = 0, lo = 2, hi = max_ctx / 256;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (measure(mid * 256) <= budget - fixed) { best = mid * 256; lo = mid + 1; }
        else hi = mid - 1;
    }
    if (best) return best;
    int chunk = 256;
    while (chunk > max_ctx) chunk /= 2;
    for (; chunk; chunk /= 2) if (measure(chunk) <= budget - fixed) return chunk;
    return 0;
}

struct Qwen4ExpWeights;
struct Qwen4ExpCache;
// Call after the weights and the cache are allocated, before prefill.
int qwen4exp_select_chunk(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache);

}  // namespace luce::common
