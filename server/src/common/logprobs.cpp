#include "logprobs.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace dflash::common {

namespace {

// Map a computed logprob onto the finite (-inf, 0] range the wire format
// requires: JSON cannot represent -inf/NaN, and float rounding can push a
// zero-mass result a hair above 0.
float finite_logprob(double lp) {
    if (std::isnan(lp)) return -std::numeric_limits<float>::max();
    if (lp > 0.0) return 0.0f;
    if (lp < -(double) std::numeric_limits<float>::max()) {
        return -std::numeric_limits<float>::max();
    }
    return (float) lp;
}

}  // namespace

TokenLogprob compute_token_logprob(const float * logits, int vocab,
                                   int32_t token, int top_k) {
    TokenLogprob out;
    out.token = token;
    if (!logits || vocab <= 0 || token < 0 || token >= vocab) return out;

    // Log-sum-exp in double so rows in the ±1e4 range stay finite.
    double max_logit = (double) logits[0];
    for (int i = 1; i < vocab; ++i) {
        if ((double) logits[i] > max_logit) max_logit = logits[i];
    }
    double sum_exp = 0.0;
    for (int i = 0; i < vocab; ++i) {
        sum_exp += std::exp((double) logits[i] - max_logit);
    }
    const double log_denom = max_logit + std::log(sum_exp);
    out.logprob = finite_logprob((double) logits[token] - log_denom);

    const int k = std::min(std::max(top_k, 0), vocab);
    if (k == 0) return out;

    std::vector<int32_t> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    const auto better = [&](int32_t a, int32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;  // equal logits: lower token id wins (argmax rule)
    };
    std::partial_sort(order.begin(), order.begin() + k, order.end(), better);
    out.top.reserve(k);
    for (int i = 0; i < k; ++i) {
        const int32_t t = order[i];
        out.top.emplace_back(
            t, finite_logprob((double) logits[t] - log_denom));
    }
    return out;
}

} // namespace dflash::common
