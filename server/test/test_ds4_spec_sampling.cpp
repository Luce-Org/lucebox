#include "CppUnitTestFramework.hpp"
#include "common/sampler.h"
#include "deepseek4/deepseek4_spec_sampling.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace luce::common;

namespace {
struct Ds4SpecSamplingFixture {};

std::vector<float> random_logits(int vocab, uint64_t seed, float scale) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n(0.0f, scale);
    std::vector<float> logits((size_t) vocab);
    for (float & l : logits) l = n(rng);
    return logits;
}

// Independent double-precision reference of the sampler chain: penalties,
// top_k, temperature softmax, top_p cut (inclusive), renormalize.
std::map<int, double> reference_distribution(const std::vector<float> & logits,
                                             const SamplerCfg & cfg,
                                             const std::vector<int32_t> & history) {
    std::vector<double> z(logits.begin(), logits.end());
    const int win = std::min((int) history.size(), cfg.rep_window);
    const int from = (int) history.size() - win;
    if (cfg.rep_pen > 1.0f) {
        std::unordered_set<int> seen(history.begin() + from, history.end());
        for (int t : seen) z[(size_t) t] = z[(size_t) t] > 0.0 ? z[(size_t) t] / cfg.rep_pen : z[(size_t) t] * cfg.rep_pen;
    }
    if (cfg.freq_pen != 0.0f || cfg.pres_pen != 0.0f) {
        std::unordered_map<int, int> counts;
        for (int i = from; i < (int) history.size(); i++) counts[history[(size_t) i]]++;
        for (auto & kv : counts) z[(size_t) kv.first] -= (double) cfg.freq_pen * kv.second + cfg.pres_pen;
    }
    std::vector<std::pair<double, int>> c;
    for (size_t i = 0; i < z.size(); i++) c.push_back({z[i], (int) i});
    std::sort(c.begin(), c.end(), [](auto & a, auto & b) { return a.first > b.first; });
    if (cfg.top_k > 0 && cfg.top_k < (int) c.size()) c.resize((size_t) cfg.top_k);
    const double m = c.front().first / cfg.temp;
    double Z = 0.0;
    for (auto & e : c) { e.first = std::exp(e.first / cfg.temp - m); Z += e.first; }
    for (auto & e : c) e.first /= Z;
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        double cum = 0.0;
        size_t cut = c.size();
        for (size_t i = 0; i < c.size(); i++) {
            cum += c[i].first;
            if (cum >= cfg.top_p) { cut = i + 1; break; }
        }
        c.resize(cut);
        double Zc = 0.0;
        for (auto & e : c) Zc += e.first;
        for (auto & e : c) e.first /= Zc;
    }
    std::map<int, double> out;
    for (auto & e : c) out[e.second] = e.first;
    return out;
}

std::map<int, double> as_map(const std::vector<std::pair<float, int>> & dist) {
    std::map<int, double> out;
    for (auto & d : dist) if (d.first > 0.0f) out[d.second] += d.first;
    return out;
}

// Largest |p - q| over the union of supports.
double max_abs_diff(const std::map<int, double> & p, const std::map<int, double> & q) {
    double worst = 0.0;
    for (auto & kv : p) {
        auto it = q.find(kv.first);
        worst = std::max(worst, std::fabs(kv.second - (it == q.end() ? 0.0 : it->second)));
    }
    for (auto & kv : q) if (!p.count(kv.first)) worst = std::max(worst, kv.second);
    return worst;
}

// Empirical frequencies over n draws agree with p within 5 standard errors.
bool frequencies_match(const std::map<int, long> & counts, long n, const std::map<int, double> & p) {
    for (auto & kv : counts) if (!p.count(kv.first)) return false;
    for (auto & kv : p) {
        auto it = counts.find(kv.first);
        const double f = it == counts.end() ? 0.0 : (double) it->second / (double) n;
        const double se = std::sqrt(std::max(kv.second * (1.0 - kv.second), 1e-12) / (double) n);
        if (std::fabs(f - kv.second) > 5.0 * se + 1e-9) return false;
    }
    return true;
}
} // namespace

TEST_CASE(Ds4SpecSamplingFixture, distribution_matches_reference_chain) {
    const int vocab = 50000;
    const auto logits = random_logits(vocab, 7, 3.0f);
    std::vector<int32_t> history;
    for (int i = 0; i < 300; i++) history.push_back((i * 37) % 2000);

    std::vector<SamplerCfg> cfgs(5);
    cfgs[0].temp = 0.7f; cfgs[0].top_p = 0.95f;                     // nucleus over the full vocab
    cfgs[1].temp = 1.0f;                                             // plain softmax
    cfgs[2].temp = 0.8f; cfgs[2].top_k = 40; cfgs[2].top_p = 0.9f;   // top_k then top_p
    cfgs[3].temp = 0.6f; cfgs[3].top_p = 0.5f; cfgs[3].rep_pen = 1.15f;
    cfgs[4].temp = 1.0f; cfgs[4].top_p = 0.99f; cfgs[4].freq_pen = 0.4f; cfgs[4].pres_pen = 0.3f;

    std::vector<std::pair<float, int>> dist;
    for (const auto & cfg : cfgs) {
        sampler_distribution(logits.data(), vocab, cfg, history, dist);
        const auto got = as_map(dist);
        const auto want = reference_distribution(logits, cfg, history);
        CHECK(got.size() == want.size());
        CHECK(max_abs_diff(got, want) < 2e-5);
    }
}

TEST_CASE(Ds4SpecSamplingFixture, distribution_matches_sample_logits_draws) {
    const int vocab = 24;
    const auto logits = random_logits(vocab, 11, 1.5f);
    const std::vector<int32_t> history = {3, 5, 5, 9};
    SamplerCfg cfg;
    cfg.temp = 0.9f; cfg.top_p = 0.9f; cfg.rep_pen = 1.2f;

    std::vector<std::pair<float, int>> dist;
    sampler_distribution(logits.data(), vocab, cfg, history, dist);
    std::mt19937_64 rng(1234);
    std::map<int, long> counts;
    const long n = 200000;
    for (long i = 0; i < n; i++) counts[sample_logits(logits.data(), vocab, cfg, history, rng)]++;
    CHECK(frequencies_match(counts, n, as_map(dist)));
}

TEST_CASE(Ds4SpecSamplingFixture, greedy_draft_acceptance_keeps_target_distribution) {
    // Three verify rows for a seed + two greedy candidates.
    const int vocab = 12;
    SamplerCfg cfg;
    cfg.temp = 1.0f;
    std::vector<std::vector<std::pair<float, int>>> base(3);
    for (int r = 0; r < 3; r++) {
        const auto logits = random_logits(vocab, 100 + (uint64_t) r, 1.2f);
        sampler_distribution(logits.data(), vocab, cfg, {}, base[(size_t) r]);
    }
    // Like the greedy drafter: propose each row's most likely token (the
    // rows still reject it often, since none of them is close to one-hot).
    const auto top = [](const std::vector<std::pair<float, int>> & d) {
        return std::max_element(d.begin(), d.end())->second;
    };
    const int32_t draft[3] = {0, top(base[0]), top(base[1])};

    std::mt19937_64 rng(99);
    const long n = 400000;
    std::map<int, long> first;          // first emitted token
    std::map<int, long> second_after;   // second token, when the first was draft[1]
    std::map<int, long> third_after;    // third token, when the first two were the drafts
    long n_second = 0, n_third = 0;
    for (long t = 0; t < n; t++) {
        auto rows = base;
        const DSparkSampleStep s = dspark_spec_sample_accept(rows, draft, 3, rng);
        // Emitted tokens: accepted candidates draft[1..accept-1], then the bonus.
        std::vector<int> out;
        for (int i = 1; i < s.accept; i++) out.push_back(draft[i]);
        out.push_back(s.bonus);
        first[out[0]]++;
        if (out[0] == draft[1] && out.size() >= 2) { second_after[out[1]]++; n_second++; }
        if (out.size() >= 3 && out[0] == draft[1] && out[1] == draft[2]) { third_after[out[2]]++; n_third++; }
    }
    CHECK(frequencies_match(first, n, as_map(base[0])));
    CHECK(n_second > 1000);
    CHECK(frequencies_match(second_after, n_second, as_map(base[1])));
    CHECK(n_third > 1000);
    CHECK(frequencies_match(third_after, n_third, as_map(base[2])));
}
