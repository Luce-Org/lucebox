// Single-chain sampled-verify (Qwen35Backend::do_spec_decode with
// LUCE_SAMPLED_VERIFY=1): the committed tokens must be distributed exactly as
// AR sampling of the target. A fake target and drafter run the backend's step:
// sampled_verify_walk over the verify rows, the budget clamp, the fast-rollback
// bonus drop, then the next seed from sampled_verify_carried_seed or a fresh
// draw of the tip row.

#include "CppUnitTestFramework.hpp"
#include "common/sampled_verify.h"
#include "common/sampler.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

using namespace luce::common;

namespace {
struct SampledVerifyFixture {};

constexpr int kVocab = 6;

// Next-token logits are a fixed random function of the last two tokens.
class FakeModel {
public:
    FakeModel(uint64_t seed, float scale)
        : table_((size_t)kVocab * kVocab * kVocab) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> normal(0.0f, scale);
        for (float & logit : table_) logit = normal(rng);
    }
    const float * logits(const std::vector<int32_t> & seq) const {
        const int32_t a = seq.size() >= 2 ? seq[seq.size() - 2] : 0;
        const int32_t b = seq.empty() ? 0 : seq.back();
        return table_.data() + (size_t)(a * kVocab + b) * kVocab;
    }
    int32_t argmax(const std::vector<int32_t> & seq) const {
        const float * row = logits(seq);
        int32_t best = 0;
        for (int32_t v = 1; v < kVocab; ++v) {
            if (row[v] > row[best]) best = v;
        }
        return best;
    }

private:
    std::vector<float> table_;
};

using Drafter = std::function<std::vector<int32_t>(
    const std::vector<int32_t> & prefix, int count)>;

// Proposes the target's argmax chain: most drafts accepted, misses still
// happen whenever the sample leaves the mode.
Drafter greedy_drafter(const FakeModel & model) {
    return [&model](const std::vector<int32_t> & prefix, int count) {
        std::vector<int32_t> seq = prefix, out;
        for (int i = 0; i < count; ++i) {
            out.push_back(model.argmax(seq));
            seq.push_back(out.back());
        }
        return out;
    };
}

// Uniform noise: right only by chance.
Drafter noise_drafter(uint64_t seed) {
    auto rng = std::make_shared<std::mt19937_64>(seed);
    return [rng](const std::vector<int32_t> &, int count) {
        std::vector<int32_t> out;
        for (int i = 0; i < count; ++i) {
            out.push_back((int32_t)((*rng)() % kVocab));
        }
        return out;
    };
}

constexpr int kNeverFastRollback = INT_MAX;

enum class Seed { kBackend, kAlwaysRedraw };

// One generation of `length` tokens through the backend's chain step.
// `fast_rollback_threshold` plays rollback_policy.fast_rollback_threshold on a
// target that supports fast rollback; kNeverFastRollback is legacy replay.
// The target sees the prompt; the penalty history is the generated tokens
// only, as the backend's out_tokens.
std::vector<int32_t> generate(const FakeModel & target, const Drafter & drafter,
                              const SamplerCfg & sampler,
                              const std::vector<int32_t> & prompt, int width,
                              int length, int fast_rollback_threshold,
                              Seed seed_rule, std::mt19937_64 & rng) {
    std::vector<int32_t> context = prompt;
    std::vector<int32_t> out;
    // The prefill's first token is sampled like any other.
    int32_t last_tok = sample_logits(target.logits(context), kVocab, sampler,
                                     out, rng);
    std::vector<float> rows;
    std::vector<int32_t> verify_history;
    while ((int)out.size() < length) {
        const int need_commit_budget = length - (int)out.size();
        std::vector<int32_t> draft_tok = {last_tok};
        std::vector<int32_t> seq = context;
        seq.push_back(last_tok);
        for (int32_t token : drafter(seq, width - 1)) draft_tok.push_back(token);
        const int v_len = (int)draft_tok.size();

        // Verify rows: row i is the target after context + draft_tok[0..i].
        rows.clear();
        for (int i = 0; i < v_len; ++i) {
            if (i > 0) seq.push_back(draft_tok[(size_t)i]);
            const float * row = target.logits(seq);
            rows.insert(rows.end(), row, row + kVocab);
        }
        verify_history = out;
        const SampledVerifyWalk walk = sampled_verify_walk(
            rows.data(), kVocab, draft_tok, v_len, sampler, verify_history, rng);
        const int accept_n = walk.accept_n;
        const int32_t miss_draw = walk.miss_draw;
        int bonus_tok = miss_draw;
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }
        if (accept_n >= fast_rollback_threshold) {
            bonus_tok = -1;
            commit_n = std::min(accept_n, need_commit_budget);
        }
        for (int i = 0; i < commit_n; ++i) {
            const int32_t token = (i < accept_n) ? draft_tok[(size_t)i] : bonus_tok;
            context.push_back(token);
            out.push_back(token);
        }

        const int32_t carried = seed_rule == Seed::kBackend
            ? sampled_verify_carried_seed(miss_draw, accept_n, commit_n)
            : -1;
        // The tip row (commit_n - 1) is the target after the committed tokens.
        last_tok = carried >= 0
            ? carried
            : sample_logits(target.logits(context), kVocab, sampler, out, rng);
    }
    return out;
}

// P(tokens) under AR sampling, for every sequence of `length` tokens.
std::vector<double> exact_distribution(const FakeModel & target,
                                       const SamplerCfg & sampler,
                                       const std::vector<int32_t> & prompt,
                                       int length) {
    size_t cells = 1;
    for (int i = 0; i < length; ++i) cells *= kVocab;
    std::vector<double> probs(cells, 0.0);
    std::vector<std::pair<float, int>> dist;
    for (size_t cell = 0; cell < cells; ++cell) {
        std::vector<int32_t> seq = prompt, generated;
        double p = 1.0;
        size_t code = cell;
        for (int i = 0; i < length && p > 0.0; ++i) {
            const int32_t token = (int32_t)(code % kVocab);
            code /= kVocab;
            sampler_distribution(target.logits(seq), kVocab, sampler, generated,
                                 dist);
            double q = 0.0;
            for (const auto & entry : dist) {
                if (entry.second == token) q += entry.first;
            }
            p *= q;
            seq.push_back(token);
            generated.push_back(token);
        }
        probs[cell] = p;
    }
    return probs;
}

size_t cell_of(const std::vector<int32_t> & tokens) {
    size_t cell = 0, scale = 1;
    for (int32_t token : tokens) {
        cell += (size_t)token * scale;
        scale *= kVocab;
    }
    return cell;
}

// 200K generations of 4 tokens over 6^4 cells, fixed seeds. The target is
// peaked (logit scale 2), so sampling noise alone gives a total variation of a
// few thousandths here; the bound is 0.025. A flat target would need a looser
// bound: 200K draws over 1296 equal cells give about 0.03.
constexpr int kRuns = 200000;
constexpr int kLength = 4;
constexpr int kWidth = 3;  // seed + 2 drafts: 4 tokens span step boundaries
constexpr double kTvBound = 0.025;

double total_variation(const FakeModel & target, const Drafter & drafter,
                       const SamplerCfg & sampler, int fast_rollback_threshold,
                       Seed seed_rule, uint64_t seed) {
    const std::vector<int32_t> prompt = {1, 2};
    const std::vector<double> exact =
        exact_distribution(target, sampler, prompt, kLength);
    std::vector<double> counts(exact.size(), 0.0);
    std::mt19937_64 rng(seed);
    for (int run = 0; run < kRuns; ++run) {
        counts[cell_of(generate(target, drafter, sampler, prompt, kWidth,
                                kLength, fast_rollback_threshold, seed_rule,
                                rng))] += 1.0;
    }
    double tv = 0.0;
    for (size_t cell = 0; cell < exact.size(); ++cell) {
        tv += std::fabs(counts[cell] / kRuns - exact[cell]);
    }
    tv *= 0.5;
    std::fprintf(stderr, "  threshold=%d tv=%.4f\n",
                 fast_rollback_threshold == kNeverFastRollback
                     ? -1 : fast_rollback_threshold,
                 tv);
    return tv;
}

SamplerCfg thinking_sampler() {
    SamplerCfg sampler;
    sampler.temp = 1.0f;
    sampler.top_p = 0.95f;
    sampler.top_k = 20;
    return sampler;
}

// top_k below the vocab and a repetition penalty: the carried draw must have
// been made with the same history a fresh draw would use.
SamplerCfg penalized_sampler() {
    SamplerCfg sampler;
    sampler.temp = 0.8f;
    sampler.top_p = 0.9f;
    sampler.top_k = 4;
    sampler.rep_pen = 1.3f;
    return sampler;
}
}  // namespace

// Fast rollback from one accepted token (the default policy), fast rollback
// from two, and legacy replay that commits the bonus: all exact.
TEST_CASE(SampledVerifyFixture, committed_tokens_match_ar_sampling) {
    const FakeModel target(11, 2.0f);
    const Drafter drafters[] = {greedy_drafter(target), noise_drafter(5)};
    for (const SamplerCfg & sampler : {thinking_sampler(), penalized_sampler()}) {
        for (const Drafter & drafter : drafters) {
            for (int threshold : {1, 2, kNeverFastRollback}) {
                CHECK(total_variation(target, drafter, sampler, threshold,
                                      Seed::kBackend, 7) < kTvBound);
            }
        }
    }
}

// The bound has power: redrawing the tip row after a miss on fast rollback
// (the old seed) offers the rejected draft a second chance.
TEST_CASE(SampledVerifyFixture, redrawing_the_miss_row_is_caught) {
    const FakeModel target(11, 2.0f);
    CHECK(total_variation(target, greedy_drafter(target), thinking_sampler(), 1,
                          Seed::kAlwaysRedraw, 7) > 2 * kTvBound);
}

// The miss draw seeds the next step only when the tip row is the miss row.
TEST_CASE(SampledVerifyFixture, carries_only_the_tip_rows_own_draw) {
    // Fast rollback after a miss at row 2: commit_n == accept_n.
    CHECK(sampled_verify_carried_seed(4, 3, 3) == 4);
    // Every draft accepted: no draw at the tip row.
    CHECK(sampled_verify_carried_seed(-1, 3, 3) == -1);
    // Legacy replay committed the bonus: the tip row follows it, undrawn.
    CHECK(sampled_verify_carried_seed(4, 3, 4) == -1);
    // Budget clamp before the miss row: row commit_n - 1 drew an accepted
    // draft, not the miss.
    CHECK(sampled_verify_carried_seed(4, 3, 2) == -1);
    CHECK(sampled_verify_carried_seed(0, 1, 1) == 0);
}
