// Host-side helpers for statistical sampler tests: seeded random logits and
// a frequency check of observed draws against a distribution.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace sampler_test {

inline std::vector<float> random_logits(int vocab, uint64_t seed, float scale) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n(0.0f, scale);
    std::vector<float> logits((size_t) vocab);
    for (float & l : logits) l = n(rng);
    return logits;
}

inline std::map<int, double> as_map(const std::vector<std::pair<float, int>> & dist) {
    std::map<int, double> out;
    for (auto & d : dist) if (d.first > 0.0f) out[d.second] += d.first;
    return out;
}

// Largest |p - q| over the union of supports.
inline double max_abs_diff(const std::map<int, double> & p, const std::map<int, double> & q) {
    double worst = 0.0;
    for (auto & kv : p) {
        auto it = q.find(kv.first);
        worst = std::max(worst, std::fabs(kv.second - (it == q.end() ? 0.0 : it->second)));
    }
    for (auto & kv : q) if (!p.count(kv.first)) worst = std::max(worst, kv.second);
    return worst;
}

// Empirical frequencies over n draws agree with p within 5 standard errors.
inline bool frequencies_match(const std::map<int, long> & counts, long n, const std::map<int, double> & p) {
    for (auto & kv : counts) if (!p.count(kv.first)) return false;
    for (auto & kv : p) {
        auto it = counts.find(kv.first);
        const double f = it == counts.end() ? 0.0 : (double) it->second / (double) n;
        const double se = std::sqrt(std::max(kv.second * (1.0 - kv.second), 1e-12) / (double) n);
        if (std::fabs(f - kv.second) > 5.0 * se + 1e-9) return false;
    }
    return true;
}

}  // namespace sampler_test
