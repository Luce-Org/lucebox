// The fast sampler over the candidates the Inkling 2 graph extracts on the GPU
// (inkling2_top_cands_stage{1,2}_kernel): each row holds the count of tokens
// whose logit is at least max - 12 * temp (stored when there are at most 256),
// the row's maximum logit, the exact full-vocabulary softmax denominator Z, and
// the (logit, token) pairs in no particular order.

#pragma once

#include "common/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace luce::common {

// One row: count, max (float bits), Z (float bits), unused, then 256 (logit
// bits, token) pairs.
inline constexpr int kInkling2TopCandsRow = 516;
inline constexpr int kInkling2TopCandsMax = 256;

// The tokens the request samples from and their weights, in rank order, when
// the candidates settle them; false when they do not, and the caller samples
// the full row. Only for temp > 0 without penalties or a grammar.
inline bool inkling2_top_cands_weights(const int32_t * row, int n_vocab, const SamplerCfg & cfg,
                                       std::vector<std::pair<double, int32_t>> & out) {
    (void) n_vocab;
    out.clear();
    const int cnt = row[0];
    float gmax = 0.0f, Z = 0.0f;
    std::memcpy(&gmax, &row[1], sizeof(float));
    std::memcpy(&Z, &row[2], sizeof(float));
    if (cnt <= 0 || cnt > kInkling2TopCandsMax || !(Z > 0.0f) || !std::isfinite(Z)) return false;

    struct Cand { float v; int32_t id; };
    Cand cands[kInkling2TopCandsMax];
    for (int j = 0; j < cnt; ++j) {
        std::memcpy(&cands[j].v, &row[4 + 2 * j], sizeof(float));
        cands[j].id = row[4 + 2 * j + 1];
    }
    std::sort(cands, cands + cnt, [](const Cand & a, const Cand & b) {
        return a.v > b.v || (a.v == b.v && a.id < b.id);
    });
    const int k_lim = (cfg.top_k > 0 && cfg.top_k < cnt) ? cfg.top_k : cnt;
    const float inv_t = 1.0f / std::max(cfg.temp, 1e-6f);
    float probs[kInkling2TopCandsMax];
    double cum = 0.0;
    int keep = 0;
    for (int j = 0; j < k_lim; ++j) {
        const float p = std::exp((cands[j].v - gmax) * inv_t) / Z;
        probs[j] = p;
        cum += p;
        keep = j + 1;
        if (cfg.top_p > 0.0f && cfg.top_p < 1.0f && cum >= cfg.top_p) {
            break;
        }
    }
    const bool exact_ok =
        (cfg.top_k > 0 && cnt >= cfg.top_k) ||
        (cfg.top_p > 0.0f && cfg.top_p < 1.0f && cum >= cfg.top_p) ||
        (cum >= 0.9995);
    if (!exact_ok || keep <= 0 || !(cum > 0.0)) return false;
    for (int j = 0; j < keep; ++j) out.push_back({(double) probs[j], cands[j].id});
    return true;
}

}  // namespace luce::common
