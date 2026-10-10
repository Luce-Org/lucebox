#include "token_logprobs.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace luce::common {

TokenLogprobs compute_token_logprobs(const float * logits, int vocab,
                                     int32_t chosen, int top_n) {
    TokenLogprobs out;
    out.chosen.token = chosen;
    if (!logits || vocab <= 0) {
        out.chosen.logprob = -std::numeric_limits<float>::infinity();
        return out;
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int v = 0; v < vocab; ++v) {
        if (logits[v] > max_logit) max_logit = logits[v];
    }
    // Accumulate in double: a 250k-entry vocab loses precision in float.
    double sum = 0.0;
    for (int v = 0; v < vocab; ++v) {
        sum += std::exp((double)logits[v] - (double)max_logit);
    }
    const double log_z = (double)max_logit + std::log(sum);
    auto logprob_of = [&](int v) {
        return (float)((double)logits[v] - log_z);
    };

    out.chosen.logprob = chosen >= 0 && chosen < vocab
        ? logprob_of(chosen)
        : -std::numeric_limits<float>::infinity();

    const int n = std::min({std::max(top_n, 0), kMaxTopLogprobs, vocab});
    if (n == 0) return out;
    // Insertion into a short sorted list: O(vocab * n) worst case, no
    // vocab-sized allocation. Ascending scan plus upper_bound keeps the lower
    // id first among equal logits.
    std::vector<int> best;
    best.reserve((size_t)n + 1);
    auto higher = [&](float x, int v) { return x > logits[v]; };
    for (int v = 0; v < vocab; ++v) {
        const float x = logits[v];
        if ((int)best.size() == n && !(x > logits[best.back()])) continue;
        best.insert(std::upper_bound(best.begin(), best.end(), x, higher), v);
        if ((int)best.size() > n) best.pop_back();
    }
    out.top.reserve(best.size());
    for (int v : best) out.top.push_back({(int32_t)v, logprob_of(v)});
    return out;
}

}  // namespace luce::common
