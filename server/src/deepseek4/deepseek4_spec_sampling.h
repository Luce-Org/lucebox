// Speculative sampling for DSpark's greedy drafts.
//
// The DSpark drafter proposes deterministic (greedy) candidates. For a
// deterministic draft q(x) = [x == d], the speculative-sampling rule reduces
// to: keep candidate d with the target's probability p(d); on rejection draw
// from p with d removed, renormalized (max(0, p - q) / (1 - p(d))). Each
// emitted token then follows the target sampler's distribution exactly, so a
// sampled request keeps its sampler contract while decoding speculatively.

#pragma once

#include "common/sampler.h"

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace luce::common {

// Per-request state: the request's sampler chain, the token history its
// penalties read (prompt + every emitted token, seed included) and its RNG.
struct DSparkSpecSampling {
    SamplerCfg cfg;
    std::vector<int32_t> history;
    std::mt19937_64 * rng = nullptr;
};

struct DSparkSampleStep {
    int accept = 1;   // seed + kept candidates, as in the greedy loop
    int bonus = -1;   // token emitted after the last kept candidate
};

// rows[i] is the target sampler's distribution (sampler_distribution) after
// draft[0..i], i.e. with draft[1..i] appended to the history; draft[0] is the
// step's seed. Row i is only consulted when draft[1..i] were all kept, so the
// rows can be built up front. Rows are modified (a rejected candidate's
// weight is zeroed). Uniforms are drawn in walk order: one per candidate
// test, one for the final draw.
inline DSparkSampleStep dspark_spec_sample_accept(
        std::vector<std::vector<std::pair<float, int>>> & rows,
        const int32_t * draft, int q, std::mt19937_64 & rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    DSparkSampleStep step;
    for (int i = 0; i < q; i++) {
        auto & dist = rows[(size_t) i];
        if (i == q - 1) {                       // every candidate kept
            step.bonus = sampler_draw(dist, unif(rng));
            return step;
        }
        const int cand = draft[i + 1];
        float p_cand = 0.0f;
        for (auto & d : dist) {
            if (d.second == cand) { p_cand = d.first; d.first = 0.0f; break; }
        }
        if (unif(rng) < (double) p_cand) {
            step.accept++;
            continue;
        }
        step.bonus = sampler_draw(dist, unif(rng));   // p without the candidate
        return step;
    }
    return step;
}

}  // namespace luce::common
