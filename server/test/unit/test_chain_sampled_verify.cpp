// Sampled verify for batched chain rounds (sampled_verify.h): the
// committed tokens, the pending token included, follow the target's sampler
// exactly, and a round spends the RNG and grows the history as
// autoregressive sampling does, with and without the min-token floor. Pure
// host logic: every sampler config has top_k set, so no GPU sampler path
// runs.
#include "CppUnitTestFramework.hpp"
#include "common/sampled_verify.h"
#include "common/gbnf.h"
#include "common/output_constraint.h"
#include "sampler_test_stats.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace luce::common;
using namespace sampler_test;

namespace {
struct ChainSampledVerifyFixture {};

int top_token(const std::vector<std::pair<float, int>> & dist) {
    return std::max_element(dist.begin(), dist.end())->second;
}

// The min-token floor, as Qwen35Backend::apply_min_tokens_floor: an EOS
// before min_tokens generated tokens becomes the row's best other token.
struct MinTokens {
    int32_t eos = -1;
    int     min_tokens = 0;
};

int32_t apply_floor(int32_t token, int generated, const float * row, int vocab,
                    const MinTokens & floor) {
    if (token != floor.eos || generated >= floor.min_tokens) return token;
    int32_t alt = -1;
    float best = -1e30f;
    for (int v = 0; v < vocab; v++) {
        if (v != floor.eos && row[v] > best) { best = row[v]; alt = v; }
    }
    return alt;
}

// One round as the engine runs it: the walk, a draw the floor would replace
// ending it, then the pending token from sampled_verify_next with the
// committed history (the walk returns the history with the kept prefix
// appended, as commit_step leaves it) and the floor.
// generated: tokens generated before the root.
std::vector<int32_t> committed_tokens(
        const std::vector<float> & rows, int vocab, int width,
        const std::vector<int32_t> & tokens, const SamplerCfg & cfg,
        std::vector<int32_t> & history, std::mt19937_64 & rng,
        const MinTokens & floor = {}, int generated = 0) {
    const SampledVerifyWalk walk = sampled_verify_walk(
        rows.data(), vocab, width, tokens.data(), cfg, history, rng,
        [&](int32_t token, int position) {
            return token == floor.eos && generated + position < floor.min_tokens;
        });
    const int kept = walk.accepted;
    std::vector<int32_t> out(tokens.begin() + 1, tokens.begin() + kept);
    const int32_t next = sampled_verify_next(
        walk, tokens.data(), kept, [&](int row) {
            return sample_logits(rows.data() + (size_t) row * (size_t) vocab,
                                 vocab, cfg, history, rng);
        });
    const float * tip = rows.data() + (size_t) (kept - 1) * (size_t) vocab;
    out.push_back(apply_floor(next, generated + kept, tip, vocab, floor));
    return out;
}

// Runs n rounds over fixed rows and checks the first `positions` committed
// tokens against the (floored) sampler distribution at their positions,
// each conditional on the drafted prefix before it; position width - 1 is
// the pending draw after every candidate was accepted. (Free helpers
// cannot CHECK: the caller checks the result.)
bool committed_distribution_matches(const std::vector<float> & rows, int vocab,
                                    const std::vector<int32_t> & tokens,
                                    const SamplerCfg & cfg,
                                    const std::vector<int32_t> & history,
                                    size_t positions,
                                    const MinTokens & floor = {}) {
    const int width = (int) tokens.size();
    std::vector<std::map<int, double>> want;
    std::vector<int32_t> h = history;
    h.push_back(tokens[0]);
    std::vector<std::pair<float, int>> dist;
    for (size_t i = 0; i < positions; i++) {
        const float * row = rows.data() + i * (size_t) vocab;
        sampler_distribution(row, vocab, cfg, h, dist);
        for (auto & d : dist) {
            d.second = apply_floor(d.second, (int) i + 1, row, vocab, floor);
        }
        want.push_back(as_map(dist));
        if (i + 1 < tokens.size()) h.push_back(tokens[i + 1]);
    }
    std::mt19937_64 rng(99);
    const long n = 400000;
    std::vector<std::map<int, long>> got(positions);
    std::vector<long> n_got(positions, 0);
    for (long t = 0; t < n; t++) {
        std::vector<int32_t> round_history = history;
        const auto out = committed_tokens(rows, vocab, width, tokens, cfg,
                                          round_history, rng, floor);
        for (size_t i = 0; i < positions && i < out.size(); i++) {
            got[i][out[i]]++;
            n_got[i]++;
            if (i + 1 >= (size_t) width || out[i] != tokens[i + 1]) break;
        }
    }
    bool ok = n_got[0] == n;
    for (size_t i = 0; i < positions; i++) {
        ok = ok && n_got[i] > 1000 &&
             frequencies_match(got[i], n_got[i], want[i]);
    }
    return ok;
}

// Autoregressive sampling over the same rows, one token per row: draw,
// apply the floor, feed the result. rng_after[i]: the state after i + 1
// draws; draws: the raw draws before the floor.
struct ArRun {
    std::vector<int32_t> tokens;
    std::vector<int32_t> draws;
    std::vector<std::mt19937_64> rng_after;
};

ArRun run_ar(const std::vector<float> & rows, int vocab, int width,
             int32_t root, const SamplerCfg & cfg,
             const std::vector<int32_t> & history, uint64_t seed,
             const MinTokens & floor = {}) {
    ArRun ar;
    std::mt19937_64 rng(seed);
    std::vector<int32_t> h = history;
    h.push_back(root);
    for (int i = 0; i < width; i++) {
        const float * row = rows.data() + (size_t) i * vocab;
        ar.draws.push_back(sample_logits(row, vocab, cfg, h, rng));
        ar.tokens.push_back(apply_floor(ar.draws.back(), i + 1, row, vocab, floor));
        h.push_back(ar.tokens.back());
        ar.rng_after.push_back(rng);
    }
    return ar;
}
} // namespace

TEST_CASE(ChainSampledVerifyFixture, committed_tokens_follow_the_target_sampler) {
    // Four verify rows for a root + three greedy candidates, the card's
    // thinking sampler with top_k below the vocabulary. All four committed
    // positions are checked, the last one drawn after every candidate was
    // accepted.
    const int vocab = 24;
    SamplerCfg cfg;
    cfg.temp = 1.0f; cfg.top_k = 20; cfg.top_p = 0.95f;
    std::vector<float> rows;
    std::vector<int32_t> tokens = {5};
    std::vector<std::pair<float, int>> dist;
    for (int r = 0; r < 4; r++) {
        const auto logits = random_logits(vocab, 100 + (uint64_t) r, 1.2f);
        rows.insert(rows.end(), logits.begin(), logits.end());
        sampler_distribution(logits.data(), vocab, cfg, {}, dist);
        if (r < 3) tokens.push_back(top_token(dist));
    }
    CHECK(committed_distribution_matches(rows, vocab, tokens, cfg, {1, 2, 3}, 4));
}

TEST_CASE(ChainSampledVerifyFixture, penalties_see_the_root_and_accepted_candidates) {
    // A repeated candidate: row 1 must penalize the candidate row 0
    // accepted, and row 0 the root, so a walk that leaves either out of
    // the history draws from the wrong distribution.
    const int vocab = 16;
    SamplerCfg cfg;
    cfg.temp = 0.9f; cfg.top_k = 12; cfg.top_p = 0.95f; cfg.rep_pen = 1.5f;
    std::vector<float> rows;
    for (int r = 0; r < 4; r++) {
        const auto logits = random_logits(vocab, 200 + (uint64_t) r, 1.0f);
        rows.insert(rows.end(), logits.begin(), logits.end());
    }
    std::vector<std::pair<float, int>> dist;
    sampler_distribution(rows.data(), vocab, cfg, {}, dist);
    const int32_t a = top_token(dist);
    rows[(size_t) vocab + (size_t) a] = 3.0f;      // row 1 favours a again
    rows[(size_t) a] += 0.5f;                       // and row 0 still does
    const int32_t root = (a + 1) % vocab;
    rows[(size_t) root] = 2.5f;                     // row 0 also likes the root
    const std::vector<int32_t> history = {7, 8};
    std::vector<int32_t> h = history;
    h.insert(h.end(), {root, a, a});
    sampler_distribution(rows.data() + 2 * vocab, vocab, cfg, h, dist);
    const std::vector<int32_t> tokens = {root, a, a, top_token(dist)};

    // The penalty moves the distributions this case checks.
    std::vector<std::pair<float, int>> with, without;
    sampler_distribution(rows.data() + vocab, vocab, cfg, {root, a}, with);
    sampler_distribution(rows.data() + vocab, vocab, cfg, {root}, without);
    CHECK(max_abs_diff(as_map(with), as_map(without)) > 0.05);
    sampler_distribution(rows.data(), vocab, cfg, {root}, with);
    sampler_distribution(rows.data(), vocab, cfg, {}, without);
    CHECK(max_abs_diff(as_map(with), as_map(without)) > 0.05);

    CHECK(committed_distribution_matches(rows, vocab, tokens, cfg, history, 4));
}

TEST_CASE(ChainSampledVerifyFixture, min_token_floor_keeps_the_target_distribution) {
    // Each row draws EOS/A/B with 0.5/0.3/0.2 and the floor turns an early
    // EOS into A, so autoregressive decode emits A 0.8 and B 0.2. A drafted
    // EOS the walk draws must not be accepted and redrawn after the clamp
    // (that would give B 0.2 + 0.5 x 0.2).
    const int vocab = 8;
    const int32_t eos = 7, A = 2, B = 4;
    SamplerCfg cfg;
    cfg.temp = 1.0f; cfg.top_k = 4;
    std::vector<float> row((size_t) vocab, -30.0f);
    row[(size_t) eos] = std::log(0.5f);
    row[(size_t) A] = std::log(0.3f);
    row[(size_t) B] = std::log(0.2f);
    std::vector<float> rows;
    for (int r = 0; r < 4; r++) rows.insert(rows.end(), row.begin(), row.end());
    const MinTokens floor{eos, 100};

    std::vector<std::pair<float, int>> dist;
    sampler_distribution(row.data(), vocab, cfg, {}, dist);
    auto floored = as_map(dist);
    CHECK(std::fabs(floored[eos] - 0.5) < 1e-4);

    CHECK(committed_distribution_matches(rows, vocab, {1, eos, A, B}, cfg, {}, 1, floor));
    CHECK(committed_distribution_matches(rows, vocab, {1, A, A, A}, cfg, {}, 4, floor));
}

TEST_CASE(ChainSampledVerifyFixture, round_matches_autoregressive_draws) {
    const int vocab = 32;
    const int width = 6;
    SamplerCfg cfg;
    cfg.temp = 0.8f; cfg.top_k = 20; cfg.top_p = 0.95f; cfg.rep_pen = 1.1f;
    std::vector<float> rows;
    for (int r = 0; r < width; r++) {
        const auto logits = random_logits(vocab, 300 + (uint64_t) r, 1.5f);
        rows.insert(rows.end(), logits.begin(), logits.end());
    }
    const std::vector<int32_t> base = {4, 9, 9, 17};
    const int32_t root = 11;
    const uint64_t seed = 1234;
    const ArRun ar = run_ar(rows, vocab, width, root, cfg, base, seed);

    // A draft equal to the AR draws is accepted whole; the pending token is
    // the AR draw from the last row, and the round spent width draws.
    std::vector<int32_t> tokens = {root};
    tokens.insert(tokens.end(), ar.tokens.begin(), ar.tokens.end() - 1);
    {
        std::mt19937_64 rng(seed);
        std::vector<int32_t> history = base;
        const auto out = committed_tokens(rows, vocab, width, tokens, cfg, history, rng);
        CHECK(out == ar.tokens);
        CHECK(rng == ar.rng_after.back());
        std::vector<int32_t> want = base;
        want.push_back(root);
        want.insert(want.end(), ar.tokens.begin(), ar.tokens.end() - 1);
        CHECK(history == want);
    }

    // A draft that leaves the AR draws at position j stops there: the
    // pending token is the AR draw at j, with no second draw for it.
    for (int j = 0; j + 1 < width; j++) {
        std::vector<int32_t> perturbed = tokens;
        perturbed[(size_t) j + 1] = (ar.tokens[(size_t) j] + 1) % vocab;
        std::mt19937_64 rng(seed);
        std::vector<int32_t> history = base;
        const auto out = committed_tokens(rows, vocab, width, perturbed, cfg, history, rng);
        CHECK(out == std::vector<int32_t>(ar.tokens.begin(), ar.tokens.begin() + j + 1));
        CHECK(rng == ar.rng_after[(size_t) j]);
        std::vector<int32_t> want = base;
        want.push_back(root);
        want.insert(want.end(), ar.tokens.begin(), ar.tokens.begin() + j);
        CHECK(history == want);
    }
}

TEST_CASE(ChainSampledVerifyFixture, floored_round_matches_autoregressive_draws) {
    // EOS is likely on every row and the floor is active: AR replaces each
    // drawn EOS. A draft of the AR tokens, or of the raw draws (EOS where
    // the floor fired), commits the AR prefix up to and including the first
    // floored position, with exactly that many draws.
    const int vocab = 12;
    const int width = 5;
    const int32_t eos = 3;
    SamplerCfg cfg;
    cfg.temp = 1.0f; cfg.top_k = 8;
    std::vector<float> rows;
    for (int r = 0; r < width; r++) {
        auto logits = random_logits(vocab, 400 + (uint64_t) r, 1.0f);
        logits[(size_t) eos] = 2.0f;
        rows.insert(rows.end(), logits.begin(), logits.end());
    }
    const MinTokens floor{eos, 100};
    const int32_t root = 6;
    int floored_rounds = 0;
    for (uint64_t seed = 1; seed <= 200; seed++) {
        const ArRun ar = run_ar(rows, vocab, width, root, cfg, {}, seed, floor);
        size_t first_floored = (size_t) width;
        for (size_t i = 0; i < ar.draws.size(); i++) {
            if (ar.draws[i] != ar.tokens[i]) { first_floored = i; break; }
        }
        // The round ends at the first floored draw, or after the last row.
        const size_t emitted = std::min(first_floored + 1, (size_t) width);
        floored_rounds += first_floored + 1 < (size_t) width;
        for (const auto * draft : {&ar.tokens, &ar.draws}) {
            std::vector<int32_t> tokens = {root};
            tokens.insert(tokens.end(), draft->begin(), draft->end() - 1);
            std::mt19937_64 rng(seed);
            std::vector<int32_t> history;
            const auto out = committed_tokens(rows, vocab, width, tokens, cfg,
                                              history, rng, floor);
            CHECK(out == std::vector<int32_t>(ar.tokens.begin(),
                                              ar.tokens.begin() + (long) emitted));
            CHECK(rng == ar.rng_after[emitted - 1]);
        }
    }
    CHECK(floored_rounds > 20);
}

TEST_CASE(ChainSampledVerifyFixture, clamped_round_keeps_the_walks_draw) {
    // A clamp keeps fewer rows than the walk accepted: the walk's draw at
    // the kept tip's row matched the next draft, so that draft is the
    // pending token and nothing is drawn again.
    const int vocab = 16;
    SamplerCfg cfg;
    cfg.temp = 0.7f; cfg.top_k = 10; cfg.rep_pen = 1.2f;
    std::vector<float> rows;
    for (int r = 0; r < 4; r++) {
        const auto logits = random_logits(vocab, 500 + (uint64_t) r, 1.0f);
        rows.insert(rows.end(), logits.begin(), logits.end());
    }
    const ArRun ar = run_ar(rows, vocab, 4, 2, cfg, {9, 1, 5}, 77);
    std::vector<int32_t> tokens = {2};
    tokens.insert(tokens.end(), ar.tokens.begin(), ar.tokens.end() - 1);
    std::vector<int32_t> history = {9, 1, 5};
    std::mt19937_64 rng(77);
    const SampledVerifyWalk walk = sampled_verify_walk(
        rows.data(), vocab, 4, tokens.data(), cfg, history, rng);
    CHECK(walk.accepted == 4);
    const std::mt19937_64 after_walk = rng;
    const int32_t pending = sampled_verify_next(
        walk, tokens.data(), 2, [&](int row) {
            return sample_logits(rows.data() + (size_t) row * vocab, vocab,
                                 cfg, history, rng);
        });
    CHECK(pending == ar.tokens[1]);
    CHECK(rng == after_walk);
}

TEST_CASE(ChainSampledVerifyFixture, lane_gate_by_sampler_and_flag) {
    SamplerCfg greedy;
    CHECK(chain_sampler_spec_capable(greedy, false));
    CHECK(chain_sampler_spec_capable(greedy, true));

    SamplerCfg temperature;
    temperature.temp = 1.0f; temperature.top_k = 20; temperature.top_p = 0.95f;
    CHECK(!chain_sampler_spec_capable(temperature, false));
    CHECK(chain_sampler_spec_capable(temperature, true));

    SamplerCfg presence;
    presence.temp = 0.7f; presence.pres_pen = 1.5f;
    CHECK(!chain_sampler_spec_capable(presence, false));
    CHECK(chain_sampler_spec_capable(presence, true));

    SamplerCfg repetition;
    repetition.rep_pen = 1.1f;   // greedy over penalized logits
    CHECK(!chain_sampler_spec_capable(repetition, false));
    CHECK(chain_sampler_spec_capable(repetition, true));

    std::string error;
    auto matcher = gbnf::Matcher::compile("root ::= \"a\"", "root", error);
    CHECK(matcher != nullptr);
    SamplerCfg grammar;
    grammar.constraint = std::make_shared<OutputConstraint>(
        ConstraintVocab::build({"a", "<end>"}, {1}), std::move(matcher),
        OutputConstraint::Options{});
    CHECK(!chain_sampler_spec_capable(grammar, false));
    CHECK(!chain_sampler_spec_capable(grammar, true));
    grammar.temp = 1.0f;
    CHECK(!chain_sampler_spec_capable(grammar, true));
}

TEST_CASE(ChainSampledVerifyFixture, host_only_sampler_configs) {
    // top_k below the vocabulary never takes a GPU sampler path, so such a
    // lane's walk may run on a pool worker.
    SamplerCfg card;
    card.temp = 1.0f; card.top_k = 20; card.top_p = 0.95f;
    CHECK(sample_logits_host_only(card, 248320));
    // Without top_k, a build with the GPU sampler may sample on the GPU
    // (unless LUCE_GPU_SAMPLE=0); the engine keeps such lanes on its own
    // thread. Without the GPU sampler every config is host-only.
#ifndef LUCE_HAVE_GPU_SAMPLER
    SamplerCfg plain;
    plain.temp = 1.0f;
    SamplerCfg nucleus;
    nucleus.temp = 0.7f; nucleus.top_p = 0.9f;
    CHECK(sample_logits_host_only(plain, 248320));
    CHECK(sample_logits_host_only(nucleus, 248320));
#endif
}
