#include "CppUnitTestFramework.hpp"
#include "common/sampler.h"
#include "deepseek4/deepseek4_spec_sampling.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
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

    std::vector<SamplerCfg> cfgs(6);
    cfgs[0].temp = 0.7f; cfgs[0].top_p = 0.95f;                     // nucleus over the full vocab
    cfgs[1].temp = 1.0f;                                             // plain softmax
    cfgs[2].temp = 0.8f; cfgs[2].top_k = 40; cfgs[2].top_p = 0.9f;   // top_k then top_p
    cfgs[3].temp = 0.6f; cfgs[3].top_p = 0.5f; cfgs[3].rep_pen = 1.15f;
    cfgs[4].temp = 1.0f; cfgs[4].top_p = 0.99f; cfgs[4].freq_pen = 0.4f; cfgs[4].pres_pen = 0.3f;
    cfgs[5].temp = 0.8f; cfgs[5].top_k = 50;                         // top_k only

    std::vector<std::pair<float, int>> dist;
    for (const auto & cfg : cfgs) {
        sampler_distribution(logits.data(), vocab, cfg, history, dist);
        const auto got = as_map(dist);
        const auto want = reference_distribution(logits, cfg, history);
        // Support sizes can differ by a boundary token whose float and double
        // masses round to opposite sides of the cut; the mass check bounds it.
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
    // 20K draws: GPU sampler builds run each through a device round trip.
    const long n = 20000;
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

TEST_CASE(Ds4SpecSamplingFixture, draw_never_returns_zeroed_entry) {
    // A rejected candidate is zeroed in place; a uniform of exactly 0 must
    // still land on the first positive entry.
    const std::vector<std::pair<float, int>> dist = {{0.0f, 7}, {0.25f, 3}, {0.0f, 9}, {0.75f, 5}};
    CHECK(sampler_draw(dist, 0.0) == 3);
    CHECK(sampler_draw(dist, 1.0) == 5);
    CHECK(sampler_draw(dist, 0.5) == 5);
}

TEST_CASE(Ds4SpecSamplingFixture, row_history_is_the_penalty_window) {
    std::vector<int32_t> history;
    for (int i = 0; i < 1000; i++) history.push_back(i);
    const int32_t draft[4] = {-1, 2001, 2002, 2003};
    std::vector<int32_t> out;

    SamplerCfg plain;
    plain.temp = 0.7f;
    dspark_row_history(plain, history, draft, 3, out);
    CHECK(out.empty());                      // no penalty reads history

    SamplerCfg pen;
    pen.temp = 0.7f; pen.rep_pen = 1.1f; pen.rep_window = 256;
    dspark_row_history(pen, history, draft, 2, out);
    CHECK(out.size() == 258);                // window + two drafts
    CHECK(out.front() == 744 && out[255] == 999);
    CHECK(out[256] == 2001 && out[257] == 2002);
}

TEST_CASE(Ds4SpecSamplingFixture, row_pool_runs_every_row_once) {
    DSparkRowPool pool(4);
    for (int round = 0; round < 200; round++) {
        const int n = 1 + round % 7;         // also more rows than workers
        std::vector<int> hits((size_t) n, 0);
        const std::function<void(int)> fn = [&](int i) { hits[(size_t) i]++; };
        pool.run(n, fn);
        for (int h : hits) CHECK(h == 1);
    }
}

TEST_CASE(Ds4SpecSamplingFixture, greedy_penalties_match_the_cpu_chain_bit_for_bit) {
    // temp 0 with penalties is an argmax over penalized logits, so the rows
    // must apply the penalties with the CPU chain's float operations in its
    // order (repetition, then frequency, then presence), or a near tie can
    // pick a different token than the AR draw. Logits on a coarse grid make
    // ties and last-bit differences common.
    const int vocab = 64;
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<int> level(0, 20), tok(0, vocab - 1);
    SamplerCfg cfg;
    cfg.temp = 0.0f; cfg.rep_pen = 1.1f; cfg.freq_pen = 0.1f; cfg.pres_pen = 0.3f;
    std::vector<std::pair<float, int>> dist;
    int mismatches = 0;
    for (int trial = 0; trial < 20000; trial++) {
        std::vector<float> logits((size_t) vocab);
        for (float & l : logits) l = 0.1f * (float) level(rng) - 0.7f;
        std::vector<int32_t> history(30);
        for (auto & t : history) t = tok(rng);
        std::vector<float> z = logits;
        std::unordered_map<int, int> counts;
        for (int t : history) counts[t]++;
        for (auto & kv : counts) {
            float & l = z[(size_t) kv.first];
            l = l > 0.0f ? l / cfg.rep_pen : l * cfg.rep_pen;
            l -= cfg.freq_pen * kv.second;
            l -= cfg.pres_pen;
        }
        const int want = (int) (std::max_element(z.begin(), z.end()) - z.begin());
        sampler_distribution(logits.data(), vocab, cfg, history, dist);
        mismatches += dist.size() != 1 || dist[0].second != want;
    }
    CHECK(mismatches == 0);
}

TEST_CASE(Ds4SpecSamplingFixture, candidate_outside_the_nucleus_is_never_kept) {
    // p(d) = 0 for a candidate the nucleus cuts: it must be rejected every
    // time, and the redraw must come from the nucleus.
    const int vocab = 32;
    const auto logits = random_logits(vocab, 21, 2.0f);
    SamplerCfg cfg;
    cfg.temp = 0.8f; cfg.top_p = 0.5f;
    std::vector<std::pair<float, int>> row0, row1;
    sampler_distribution(logits.data(), vocab, cfg, {}, row0);
    sampler_distribution(logits.data(), vocab, cfg, {}, row1);
    const int outside = (int) (std::min_element(logits.begin(), logits.end()) - logits.begin());
    std::unordered_set<int> nucleus;
    for (auto & e : row0) nucleus.insert(e.second);
    REQUIRE_TRUE(!nucleus.count(outside));
    const int32_t draft[2] = {0, outside};
    std::mt19937_64 rng(3);
    for (int t = 0; t < 10000; t++) {
        std::vector<std::vector<std::pair<float, int>>> rows = {row0, row1};
        const DSparkSampleStep s = dspark_spec_sample_accept(rows, draft, 2, rng);
        CHECK(s.accept == 1);
        CHECK(s.bonus != outside && nucleus.count(s.bonus) == 1);
    }
}

TEST_CASE(Ds4SpecSamplingFixture, row_pool_propagates_a_failing_row) {
    DSparkRowPool pool(3);
    for (int bad = 0; bad < 5; bad++) {      // the caller's rows and the workers'
        const std::function<void(int)> fn = [&](int i) {
            if (i == bad) throw std::runtime_error("row failed");
        };
        bool threw = false;
        try {
            pool.run(5, fn);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        std::vector<int> hits(4, 0);           // and the pool still works
        const std::function<void(int)> ok = [&](int i) { hits[(size_t) i]++; };
        pool.run(4, ok);
        for (int h : hits) CHECK(h == 1);
    }
}
