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
// The kernel keeps the logits at least max - 12 * temp: every other token
// weighs under e^-12 against the maximum.
inline constexpr float kInkling2TopCandsSpan = 12.0f;

// The tokens the request samples from and their weights, in rank order, when
// the candidates settle them; false when they do not, and the caller samples
// the full row. Only for temp > 0 without penalties or a grammar. The shared
// chain (common/sampler.h) keeps the top_k tokens, takes their softmax, and
// cuts top_p of the mass of those tokens, not of the whole vocabulary.
inline bool inkling2_top_cands_weights(const int32_t * row, int n_vocab, const SamplerCfg & cfg,
                                       std::vector<std::pair<double, int32_t>> & out) {
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
    const bool top_k = cfg.top_k > 0 && cfg.top_k < n_vocab;
    const bool top_p = cfg.top_p > 0.0f && cfg.top_p < 1.0f;
    // Every token outside the candidates has a logit below the kernel's cut:
    // the candidates hold the top_k tokens when top_k <= cnt.
    const bool top_k_known = top_k && cfg.top_k <= cnt;
    const int n = top_k ? std::min(cfg.top_k, cnt) : cnt;
    const float inv_t = 1.0f / std::max(cfg.temp, 1e-6f);   // the kernel's floor
    double w[kInkling2TopCandsMax];
    double S = 0.0;
    for (int j = 0; j < n; ++j) {
        w[j] = std::exp((double) ((cands[j].v - gmax) * inv_t));
        S += w[j];
    }
    // The mass top_p is a share of: the top_k tokens' when top_k applies, else
    // the vocabulary's (Z). The top_k tokens outside the candidates weigh under
    // e^-kInkling2TopCandsSpan each, so with top_k > cnt their mass is only bounded.
    double mass_lo = Z, mass_hi = Z;
    if (top_k_known) {
        mass_lo = mass_hi = S;
    } else if (top_k) {
        mass_lo = S;
        mass_hi = std::min((double) Z,
                           S + (double) (cfg.top_k - cnt) * std::exp(-(double) kInkling2TopCandsSpan));
    }
    // The candidates hold 99.95% of the vocabulary's mass: close enough to
    // draw a reply that reaches past them from them alone.
    const bool most_mass = S >= 0.9995 * (double) Z;
    int keep = n;
    if (top_p) {
        // The smallest prefix holding top_p of the mass, when both bounds give it.
        const auto nucleus = [&](double mass) {
            const double target = (double) cfg.top_p * mass;
            double cum = 0.0;
            for (int j = 0; j < n; ++j) {
                cum += w[j];
                if (cum >= target) return j + 1;
            }
            return 0;
        };
        const int lo = nucleus(mass_lo), hi = nucleus(mass_hi);
        if (lo > 0 && lo == hi) {
            keep = lo;
        } else if (lo > 0 || !most_mass) {
            return false;
        }
    } else if (!top_k_known && !most_mass) {
        return false;
    }
    if (keep <= 0 || !(S > 0.0)) return false;
    for (int j = 0; j < keep; ++j) out.push_back({w[j], cands[j].id});
    return true;
}

}  // namespace luce::common
