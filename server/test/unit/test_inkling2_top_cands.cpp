#include "CppUnitTestFramework.hpp"
#include "common/sampler.h"
#include "inkling2/inkling2_top_cands.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace luce::common;

namespace {
struct Inkling2TopCandsFixture {};

// The row inkling2_top_cands_stage2_kernel writes for `logits`: every token whose logit is at
// least max - 12 * temp, in shuffled order (the kernel's atomic slots), the maximum, and
// Z = sum exp((x - max) / temp) over the vocabulary. False when more than 256 tokens qualify:
// the kernel stores none of them then, and the backend samples the full row.
bool kernel_row(const std::vector<float> & logits, float temp, std::mt19937_64 & rng,
                std::vector<int32_t> & row) {
    const float t = std::max(temp, 1e-6f);
    const float inv_t = 1.0f / t;
    const float gmax = *std::max_element(logits.begin(), logits.end());
    const float cut = gmax - kInkling2TopCandsSpan * t;
    double z = 0.0;
    std::vector<std::pair<float, int32_t>> cands;
    for (size_t v = 0; v < logits.size(); ++v) {
        const float scaled = (logits[v] - gmax) * inv_t;
        if (scaled > -40.0f) z += std::exp(scaled);
        if (logits[v] >= cut) cands.push_back({logits[v], (int32_t) v});
    }
    if ((int) cands.size() > kInkling2TopCandsMax) return false;
    std::shuffle(cands.begin(), cands.end(), rng);
    row.assign(kInkling2TopCandsRow, 0);
    row[0] = (int32_t) cands.size();
    const float zf = (float) z;
    std::memcpy(&row[1], &gmax, sizeof(float));
    std::memcpy(&row[2], &zf, sizeof(float));
    for (size_t j = 0; j < cands.size(); ++j) {
        std::memcpy(&row[4 + 2 * j], &cands[j].first, sizeof(float));
        row[4 + 2 * j + 1] = cands[j].second;
    }
    return true;
}

// Logit rows shaped like a language model's: a few leading tokens over a long tail. `head`
// sets how many lead and how far, so some rows keep fewer candidates than top_k.
std::vector<float> lm_row(int vocab, int head, float lead, std::mt19937_64 & rng) {
    std::normal_distribution<float> tail(0.0f, 2.0f);
    std::uniform_real_distribution<float> up(0.0f, lead);
    std::vector<float> logits((size_t) vocab);
    for (auto & x : logits) x = tail(rng);
    for (int i = 0; i < head; ++i) logits[(size_t) (rng() % (uint64_t) vocab)] += 5.0f + up(rng);
    return logits;
}
}  // namespace

// The fast sampler draws from the distribution the shared chain gives the full row:
// top_k, softmax(temp) over those, then top_p within them (common/sampler.h). With top_k and
// top_p together the nucleus is a share of the top_k tokens' mass, not of the vocabulary's.
TEST_CASE(Inkling2TopCandsFixture, weights_match_the_sampler_chain) {
    std::mt19937_64 rng(20261124);
    const int vocab = 2048;
    int checked = 0, settled = 0, settled_past_cands = 0;
    std::string failures;
    for (int r = 0; r < 60; ++r) {
        const int head = r % 3 == 0 ? 3 : r % 3 == 1 ? 30 : 120;
        const float lead = r % 3 == 0 ? 15.0f : 10.0f;
        const std::vector<float> logits = lm_row(vocab, head, lead, rng);
        for (const float temp : {0.5f, 0.7f, 1.0f}) {
            std::vector<int32_t> row;
            if (!kernel_row(logits, temp, rng, row)) continue;
            for (const int top_k : {0, 1, 5, 20, 40, 200}) {
                for (const float top_p : {0.2f, 0.5f, 0.8f, 0.9f, 0.95f}) {
                    SamplerCfg cfg;
                    cfg.temp = temp;
                    cfg.top_k = top_k;
                    cfg.top_p = top_p;
                    ++checked;
                    std::vector<std::pair<double, int32_t>> fast;
                    if (!inkling2_top_cands_weights(row.data(), vocab, cfg, fast)) continue;
                    ++settled;
                    if (top_k > row[0]) ++settled_past_cands;
                    std::vector<std::pair<float, int>> ref;
                    sampler_distribution(logits.data(), vocab, cfg, {}, ref);
                    double total = 0.0;
                    for (const auto & c : fast) total += c.first;
                    std::map<int, double> diff;
                    for (const auto & c : fast) diff[c.second] += c.first / total;
                    for (const auto & c : ref) diff[c.second] -= c.first;
                    double tv = 0.0;
                    for (const auto & d : diff) tv += std::fabs(d.second);
                    tv *= 0.5;
                    if (tv > 1e-5) {
                        failures += "row " + std::to_string(r) + " temp " + std::to_string(temp) +
                                    " top_k " + std::to_string(top_k) + " top_p " + std::to_string(top_p) +
                                    ": " + std::to_string(fast.size()) + " tokens vs " +
                                    std::to_string(ref.size()) + ", total variation " +
                                    std::to_string(tv) + "\n";
                    }
                }
            }
        }
    }
    std::fprintf(stderr, "%d checked, %d settled by the candidates (%d with top_k beyond them)\n%s",
                 checked, settled, settled_past_cands, failures.c_str());
    CHECK(failures.empty());
    // The candidates settle nearly every draw; the full row is the rare fallback.
    CHECK(checked > 1000);
    CHECK(settled * 100 >= checked * 99);
    // Rows with fewer candidates than top_k take the bounded path.
    CHECK(settled_past_cands > 100);
}
