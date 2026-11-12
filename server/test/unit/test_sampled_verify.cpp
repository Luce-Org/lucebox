// Single-chain sampled verify (Qwen35Backend::do_spec_decode, sampled verify
// on): the committed tokens must be distributed exactly as AR sampling of
// the target. A fake target and drafter run the backend's
// step: sampled_verify_walk over the verify rows, the budget clamp, the
// fast-rollback bonus drop, then the next seed from sampled_verify_next.
// Under a grammar, sampled and greedy verify must also match constrained AR
// decoding, the grammar advanced over each committed token.

#include "CppUnitTestFramework.hpp"
#include "common/gbnf.h"
#include "common/output_constraint.h"
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
#include <string>
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

// kBackend: sampled_verify_next. The others spend a second draw on a row the
// walk already drew: kAlwaysRedraw draws the tip row fresh every step;
// kRedrawAfterClamp keeps an uncommitted bonus but draws fresh when a clamp
// cut the walk's prefix.
enum class Seed { kBackend, kAlwaysRedraw, kRedrawAfterClamp };

// Letters a..f, one per token, under a grammar that alternates [abc] and
// [def]: the tokens it allows flip at every position, so a verify that checks
// a row against the grammar one position off is caught.
std::shared_ptr<OutputConstraint> alternating_grammar() {
    std::vector<std::string> pieces;
    for (int t = 0; t < kVocab; ++t) pieces.push_back(std::string(1, (char)('a' + t)));
    std::string error;
    auto matcher = gbnf::Matcher::compile("root ::= ([abc] [def])*", "root", error);
    if (!matcher) std::fprintf(stderr, "grammar error: %s\n", error.c_str());
    return std::make_shared<OutputConstraint>(ConstraintVocab::build(std::move(pieces), {}),
                                              std::move(matcher),
                                              OutputConstraint::Options{});
}

struct StepPolicy {
    // rollback_policy.fast_rollback_threshold on a target that supports fast
    // rollback; kNeverFastRollback is legacy replay, which commits the bonus.
    int fast_rollback_threshold = 1;
    // A per-step cap on committed tokens that does not depend on the draws,
    // like the budget clamp but without ending the generation.
    int max_commit = INT_MAX;
    Seed seed = Seed::kBackend;
    // Hand the walk a grammar one token ahead of the root's position.
    bool grammar_off_by_one = false;
};

// One generation of `length` tokens through the backend's chain step. The
// target sees the prompt; the penalty history is the generated tokens only,
// as the backend's out_tokens.
std::vector<int32_t> generate(const FakeModel & target, const Drafter & drafter,
                              const SamplerCfg & sampler,
                              const std::vector<int32_t> & prompt, int width,
                              int length, const StepPolicy & policy,
                              std::mt19937_64 & rng) {
    std::vector<int32_t> context = prompt;
    std::vector<int32_t> out;
    // The commit point advances the grammar over every kept token.
    if (sampler.constraint) sampler.constraint->reset();
    // The prefill's first token is sampled like any other.
    int32_t last_tok = sample_logits(target.logits(context), kVocab, sampler,
                                     out, rng);
    std::vector<float> rows;
    std::vector<int32_t> verify_history;
    while ((int)out.size() < length) {
        const int need_commit_budget =
            std::min(length - (int)out.size(), policy.max_commit);
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
        SamplerCfg walk_sampler = sampler;
        if (policy.grammar_off_by_one && sampler.constraint) {
            walk_sampler.constraint = sampler.constraint->snapshot();
            walk_sampler.constraint->accept(last_tok);
        }
        const SampledVerifyWalk walk = sampled_verify_walk(
            rows.data(), kVocab, v_len, draft_tok.data(), walk_sampler,
            verify_history, rng);
        const int accept_n = walk.accepted;
        int bonus_tok = walk.bonus;
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }
        if (accept_n >= policy.fast_rollback_threshold) {
            bonus_tok = -1;
            commit_n = std::min(accept_n, need_commit_budget);
        }
        for (int i = 0; i < commit_n; ++i) {
            const int32_t token = (i < accept_n) ? draft_tok[(size_t)i] : bonus_tok;
            context.push_back(token);
            out.push_back(token);
            if (sampler.constraint) sampler.constraint->accept(token);
        }

        // The tip row (commit_n - 1) is the target after the committed tokens.
        auto draw = [&](int) {
            return sample_logits(target.logits(context), kVocab, sampler, out,
                                 rng);
        };
        switch (policy.seed) {
        case Seed::kBackend:
            last_tok = sampled_verify_next(walk, draft_tok.data(), commit_n, draw);
            break;
        case Seed::kAlwaysRedraw:
            last_tok = draw(commit_n - 1);
            break;
        case Seed::kRedrawAfterClamp:
            last_tok = commit_n == accept_n && walk.bonus >= 0
                ? walk.bonus : draw(commit_n - 1);
            break;
        }
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
    SamplerCfg position = sampler;
    for (size_t cell = 0; cell < cells; ++cell) {
        std::vector<int32_t> seq = prompt, generated;
        double p = 1.0;
        size_t code = cell;
        if (sampler.constraint) {
            sampler.constraint->reset();
            position.constraint = sampler.constraint->snapshot();
        }
        for (int i = 0; i < length && p > 0.0; ++i) {
            const int32_t token = (int32_t)(code % kVocab);
            code /= kVocab;
            sampler_distribution(target.logits(seq), kVocab, position, generated,
                                 dist);
            double q = 0.0;
            for (const auto & entry : dist) {
                if (entry.second == token) q += entry.first;
            }
            p *= q;
            seq.push_back(token);
            generated.push_back(token);
            if (position.constraint) position.constraint->accept(token);
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
                       const SamplerCfg & sampler, const StepPolicy & policy,
                       uint64_t seed) {
    const std::vector<int32_t> prompt = {1, 2};
    const std::vector<double> exact =
        exact_distribution(target, sampler, prompt, kLength);
    std::vector<double> counts(exact.size(), 0.0);
    std::mt19937_64 rng(seed);
    for (int run = 0; run < kRuns; ++run) {
        counts[cell_of(generate(target, drafter, sampler, prompt, kWidth,
                                kLength, policy, rng))] += 1.0;
    }
    double tv = 0.0;
    for (size_t cell = 0; cell < exact.size(); ++cell) {
        tv += std::fabs(counts[cell] / kRuns - exact[cell]);
    }
    tv *= 0.5;
    std::fprintf(stderr, "  threshold=%d max_commit=%d tv=%.4f\n",
                 policy.fast_rollback_threshold == kNeverFastRollback
                     ? -1 : policy.fast_rollback_threshold,
                 policy.max_commit == INT_MAX ? -1 : policy.max_commit, tv);
    return tv;
}

SamplerCfg thinking_sampler() {
    SamplerCfg sampler;
    sampler.temp = 1.0f;
    sampler.top_p = 0.95f;
    sampler.top_k = 20;
    return sampler;
}

// top_k below the vocab and a repetition penalty: the seed must be drawn
// with the same history a fresh AR draw would use.
SamplerCfg penalized_sampler() {
    SamplerCfg sampler;
    sampler.temp = 0.8f;
    sampler.top_p = 0.9f;
    sampler.top_k = 4;
    sampler.rep_pen = 1.3f;
    return sampler;
}

// Temperature alone: under a grammar, rejection sampling then draws from the
// masked distribution exactly, which is what exact_distribution computes.
SamplerCfg constrained_sampler() {
    SamplerCfg sampler;
    sampler.temp = 0.9f;
    sampler.constraint = alternating_grammar();
    return sampler;
}

// Greedy decoding under the sampler's grammar, as the backend's chain step
// runs it: the target's argmax rows, constrained_greedy_walk, the budget
// clamp, fast rollback or replay, and the next root checked against the
// grammar at the commit point. With `walk_grammar_off_by_one` the walk is
// handed a grammar a token ahead.
std::vector<int32_t> generate_greedy(const FakeModel & target, const Drafter & drafter,
                                     const SamplerCfg & sampler,
                                     const std::vector<int32_t> & prompt, int width,
                                     int length, int fast_rollback_threshold,
                                     bool walk_grammar_off_by_one = false) {
    std::vector<int32_t> context = prompt;
    std::vector<int32_t> out;
    std::mt19937_64 rng(0);
    sampler.constraint->reset();
    auto constrained_argmax = [&](const std::vector<int32_t> & seq) {
        const int32_t best = target.argmax(seq);
        if (sampler.constraint->allows(best)) return best;
        return (int32_t)sample_logits(target.logits(seq), kVocab, sampler, out, rng);
    };
    int32_t last_tok = constrained_argmax(context);
    while ((int)out.size() < length) {
        const int need_commit_budget = length - (int)out.size();
        std::vector<int32_t> draft_tok = {last_tok};
        std::vector<int32_t> seq = context;
        seq.push_back(last_tok);
        for (int32_t token : drafter(seq, width - 1)) draft_tok.push_back(token);
        const int v_len = (int)draft_tok.size();
        std::vector<int32_t> target_tok;
        std::vector<std::vector<int32_t>> row_seq;
        for (int i = 0; i < v_len; ++i) {
            if (i > 0) seq.push_back(draft_tok[(size_t)i]);
            target_tok.push_back(target.argmax(seq));
            row_seq.push_back(seq);
        }
        SamplerCfg walk_sampler = sampler;
        if (walk_grammar_off_by_one) {
            walk_sampler.constraint = sampler.constraint->snapshot();
            walk_sampler.constraint->accept(last_tok);
        }
        const int accept_n = constrained_greedy_walk(
            target_tok.data(), v_len, draft_tok.data(), walk_sampler,
            [&](int row, int & vocab) {
                vocab = kVocab;
                return target.logits(row_seq[(size_t)row]);
            });
        int bonus_tok = accept_n < v_len ? target_tok[(size_t)accept_n - 1] : -1;
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }
        int32_t replay_last_tok = -1;
        if (accept_n >= fast_rollback_threshold) {
            bonus_tok = -1;
            commit_n = std::min(accept_n, need_commit_budget);
            replay_last_tok = target_tok[(size_t)commit_n - 1];
        }
        for (int i = 0; i < commit_n; ++i) {
            const int32_t token = (i < accept_n) ? draft_tok[(size_t)i] : bonus_tok;
            context.push_back(token);
            out.push_back(token);
            sampler.constraint->accept(token);
        }
        // Replay: the target's plain argmax after the committed tokens.
        if (replay_last_tok < 0) replay_last_tok = target.argmax(context);
        last_tok = sampler.constraint->allows(replay_last_tok)
            ? replay_last_tok : constrained_argmax(context);
    }
    return out;
}

std::vector<int32_t> generate_greedy_ar(const FakeModel & target, const SamplerCfg & sampler,
                                        const std::vector<int32_t> & prompt, int length) {
    std::vector<int32_t> context = prompt;
    std::vector<int32_t> out;
    std::mt19937_64 rng(0);
    sampler.constraint->reset();
    while ((int)out.size() < length) {
        const int32_t token = sample_logits(target.logits(context), kVocab, sampler, out, rng);
        context.push_back(token);
        out.push_back(token);
        sampler.constraint->accept(token);
    }
    return out;
}

// Each draw's row index, returned as 1000 + row.
struct DrawRecorder {
    std::vector<int> rows;
    int32_t operator()(int row) {
        rows.push_back(row);
        return 1000 + row;
    }
};
}  // namespace

// Fast rollback from one accepted token (the default policy), fast rollback
// from two, and legacy replay that commits the bonus: all exact.
TEST_CASE(SampledVerifyFixture, committed_tokens_match_ar_sampling) {
    const FakeModel target(11, 2.0f);
    const Drafter drafters[] = {greedy_drafter(target), noise_drafter(5)};
    for (const SamplerCfg & sampler : {thinking_sampler(), penalized_sampler()}) {
        for (const Drafter & drafter : drafters) {
            for (int threshold : {1, 2, kNeverFastRollback}) {
                StepPolicy policy;
                policy.fast_rollback_threshold = threshold;
                CHECK(total_variation(target, drafter, sampler, policy, 7) <
                      kTvBound);
            }
        }
    }
}

// A clamp that keeps fewer tokens than the walk accepted, and lets the
// generation go on, still commits exact samples: the seed is the walk's draw
// at the kept tip's row, which matched the next draft.
TEST_CASE(SampledVerifyFixture, clamped_steps_match_ar_sampling) {
    const FakeModel target(11, 2.0f);
    StepPolicy policy;
    policy.max_commit = 1;
    CHECK(total_variation(target, greedy_drafter(target), thinking_sampler(),
                          policy, 7) < kTvBound);
}

// The bound has power: redrawing the tip row after the walk drew it gives
// the draft token a second chance after a miss (fast rollback), and takes
// one away after a clamp.
TEST_CASE(SampledVerifyFixture, redrawing_a_drawn_row_is_caught) {
    const FakeModel target(11, 2.0f);
    StepPolicy miss;
    miss.seed = Seed::kAlwaysRedraw;
    CHECK(total_variation(target, greedy_drafter(target), thinking_sampler(),
                          miss, 7) > 2 * kTvBound);
    StepPolicy clamp;
    clamp.seed = Seed::kRedrawAfterClamp;
    clamp.max_commit = 1;
    CHECK(total_variation(target, greedy_drafter(target), thinking_sampler(),
                          clamp, 7) > 2 * kTvBound);
}

TEST_CASE(SampledVerifyFixture, walk_keeps_drafts_while_they_match) {
    // Rows peaked on 3, then 4, then 5 (temperature 0: post-penalty argmax).
    std::vector<float> rows(3 * kVocab, 0.0f);
    rows[0 * kVocab + 3] = 10.0f;
    rows[1 * kVocab + 4] = 10.0f;
    rows[2 * kVocab + 5] = 10.0f;
    const SamplerCfg greedy;
    std::mt19937_64 rng(1);
    std::vector<int32_t> history = {9};
    const int32_t whole[] = {2, 3, 4};
    SampledVerifyWalk walk = sampled_verify_walk(rows.data(), kVocab, 3, whole,
                                                 greedy, history, rng);
    CHECK(walk.accepted == 3);
    CHECK(walk.bonus == -1);
    CHECK((history == std::vector<int32_t>{9, 2, 3, 4}));

    history = {9};
    const int32_t miss[] = {2, 3, 1};
    walk = sampled_verify_walk(rows.data(), kVocab, 3, miss, greedy, history,
                               rng);
    CHECK(walk.accepted == 2);
    CHECK(walk.bonus == 4);
    CHECK((history == std::vector<int32_t>{9, 2, 3}));

    // A draw the caller rewrites ends the walk even when it matches.
    history = {9};
    walk = sampled_verify_walk(rows.data(), kVocab, 3, whole, greedy, history,
                               rng, [](int32_t token, int) { return token == 4; });
    CHECK(walk.accepted == 2);
    CHECK(walk.bonus == 4);
}

TEST_CASE(SampledVerifyFixture, walk_draws_rows_only_up_to_the_first_mismatch) {
    // Sampled rows (thinking sampler) drawn row by row give the same walk, RNG
    // stream and history as the walk over the whole block, and the rows after
    // the mismatch are never asked for.
    const int width = 6;
    std::vector<float> rows(width * kVocab);
    std::mt19937_64 fill(5);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    for (float & x : rows) x = u(fill);
    const SamplerCfg cfg = thinking_sampler();
    for (uint64_t seed = 1; seed <= 64; ++seed) {
        int32_t tokens[width] = {7};
        std::mt19937_64 probe(seed);
        std::vector<int32_t> probe_history = {9, 7};
        for (int i = 0; i + 1 < width; ++i) {   // drafts equal to the draws up to row 2, then a miss
            const int32_t t = sample_logits(rows.data() + (size_t)i * kVocab, kVocab, cfg,
                                            probe_history, probe);
            tokens[i + 1] = i < 2 ? t : (t + 1) % kVocab;
            probe_history.push_back(tokens[i + 1]);
        }
        std::mt19937_64 rng_all(seed), rng_rows(seed);
        std::vector<int32_t> hist_all = {9}, hist_rows = {9};
        const SampledVerifyWalk all = sampled_verify_walk(rows.data(), kVocab, width, tokens,
                                                          cfg, hist_all, rng_all);
        std::vector<int> asked;
        const SampledVerifyWalk lazy = sampled_verify_walk_draws(
            [&](int i, const SamplerCfg & round, const std::vector<int32_t> & h,
                std::mt19937_64 & r) {
                asked.push_back(i);
                return sample_logits(rows.data() + (size_t)i * kVocab, kVocab, round, h, r);
            },
            width, tokens, cfg, hist_rows, rng_rows, [](int32_t, int) { return false; });
        CHECK(lazy.accepted == all.accepted);
        CHECK(lazy.bonus == all.bonus);
        CHECK(hist_rows == hist_all);
        CHECK(rng_rows() == rng_all());
        CHECK(all.accepted == 3);
        CHECK(all.bonus >= 0);
        CHECK((asked == std::vector<int>{0, 1, 2}));
    }
    // A row that cannot be read ends the walk with nothing kept.
    std::mt19937_64 rng(1);
    std::vector<int32_t> history = {9};
    const int32_t tokens[] = {2, 3, 4};
    const SampledVerifyWalk failed = sampled_verify_walk_draws(
        [](int, const SamplerCfg &, const std::vector<int32_t> &, std::mt19937_64 &) {
            return -1;
        },
        3, tokens, cfg, history, rng, [](int32_t, int) { return false; });
    CHECK(failed.accepted == 0);
    CHECK(failed.bonus == -1);
}

TEST_CASE(SampledVerifyFixture, next_reuses_the_draw_at_the_kept_tip) {
    const int32_t tokens[] = {10, 11, 12, 13};
    SampledVerifyWalk walk;
    walk.accepted = 3;
    walk.bonus = 77;
    DrawRecorder rec;
    // Fast rollback: the bonus ended the walk at row 2 and was not committed.
    CHECK(sampled_verify_next(walk, tokens, 3, std::ref(rec)) == 77);
    // A clamp kept 2 of 3: row 1 drew tokens[2].
    CHECK(sampled_verify_next(walk, tokens, 2, std::ref(rec)) == 12);
    CHECK(rec.rows.empty());
    // The replay committed the bonus: one draw from the row after it.
    CHECK(sampled_verify_next(walk, tokens, 4, std::ref(rec)) == 1003);
    // Every draft matched: one draw from the last row.
    walk.bonus = -1;
    CHECK(sampled_verify_next(walk, tokens, 3, std::ref(rec)) == 1002);
    CHECK((rec.rows == std::vector<int>{3, 2}));
    CHECK(sampled_verify_next(walk, tokens, 0, std::ref(rec)) == -1);
}

// Under a grammar the walk draws from a snapshot advanced over the root and
// each kept draft: still exact against constrained AR sampling, and a grammar
// one token off is caught.
TEST_CASE(SampledVerifyFixture, constrained_tokens_match_constrained_ar_sampling) {
    const FakeModel target(11, 2.0f);
    const SamplerCfg sampler = constrained_sampler();
    const Drafter drafters[] = {greedy_drafter(target), noise_drafter(5)};
    for (const Drafter & drafter : drafters) {
        for (int threshold : {1, kNeverFastRollback}) {
            StepPolicy policy;
            policy.fast_rollback_threshold = threshold;
            CHECK(total_variation(target, drafter, sampler, policy, 7) < kTvBound);
        }
    }
    StepPolicy clamp;
    clamp.max_commit = 1;
    CHECK(total_variation(target, greedy_drafter(target), sampler, clamp, 7) < kTvBound);
    StepPolicy off;
    off.grammar_off_by_one = true;
    CHECK(total_variation(target, greedy_drafter(target), sampler, off, 7) > 2 * kTvBound);
}

// Greedy verify under a grammar commits exactly the tokens constrained greedy
// AR decoding does, whichever drafter, width and rollback policy.
TEST_CASE(SampledVerifyFixture, constrained_greedy_matches_constrained_ar) {
    SamplerCfg sampler;
    sampler.constraint = alternating_grammar();
    const std::vector<int32_t> prompt = {1, 2};
    constexpr int kTokens = 24;
    int rewritten_runs = 0;
    for (uint64_t model_seed = 1; model_seed <= 40; ++model_seed) {
        const FakeModel target(model_seed, 2.0f);
        const std::vector<int32_t> ar = generate_greedy_ar(target, sampler, prompt, kTokens);
        // The grammar changes the argmax path for most models.
        std::vector<int32_t> context = prompt, plain;
        for (int i = 0; i < kTokens; ++i) {
            plain.push_back(target.argmax(context));
            context.push_back(plain.back());
        }
        if (plain != ar) ++rewritten_runs;
        const Drafter drafters[] = {greedy_drafter(target), noise_drafter(model_seed)};
        for (const Drafter & drafter : drafters) {
            for (int width : {1, 2, 4}) {
                for (int threshold : {1, 3, kNeverFastRollback}) {
                    CHECK((generate_greedy(target, drafter, sampler, prompt, width, kTokens,
                                           threshold) == ar));
                }
            }
        }
    }
    CHECK(rewritten_runs >= 30);
    // A grammar one token off commits other tokens.
    const FakeModel target(3, 2.0f);
    CHECK((generate_greedy(target, greedy_drafter(target), sampler, prompt, 4, kTokens, 1,
                           /*walk_grammar_off_by_one=*/true) !=
           generate_greedy_ar(target, sampler, prompt, kTokens)));
}

// The AR hop commits the root, forwards it alone, and seeds the next root from
// that forward's argmax. Under a grammar the seed goes through
// constrained_next_root; `checked` false is the argmax taken as is.
namespace {
std::vector<int32_t> generate_hops(const FakeModel & target, const SamplerCfg & sampler,
                                   const std::vector<int32_t> & prompt, int length,
                                   bool checked) {
    std::vector<int32_t> context = prompt, out;
    std::mt19937_64 rng(0);
    sampler.constraint->reset();
    int32_t last_tok = constrained_next_root(target.argmax(context), sampler, out, rng,
                                             [&](int & v) { v = kVocab; return target.logits(context); });
    while ((int)out.size() < length) {
        context.push_back(last_tok);
        out.push_back(last_tok);
        sampler.constraint->accept(last_tok);
        const int32_t argmax = target.argmax(context);
        last_tok = checked
            ? constrained_next_root(argmax, sampler, out, rng,
                                    [&](int & v) { v = kVocab; return target.logits(context); })
            : argmax;
    }
    return out;
}
}  // namespace

// Regression: the hop's next root was the raw argmax, so a grammar-rejected
// token became the root, was committed, and killed the grammar.
TEST_CASE(SampledVerifyFixture, ar_hop_root_stays_in_grammar) {
    SamplerCfg sampler;
    sampler.constraint = alternating_grammar();
    const std::vector<int32_t> prompt = {1, 2};
    int unchecked_differs = 0;
    for (uint64_t model_seed = 1; model_seed <= 40; ++model_seed) {
        const FakeModel target(model_seed, 2.0f);
        const auto ar = generate_greedy_ar(target, sampler, prompt, 24);
        CHECK((generate_hops(target, sampler, prompt, 24, true) == ar));
        if (generate_hops(target, sampler, prompt, 24, false) != ar) ++unchecked_differs;
    }
    CHECK(unchecked_differs >= 30);
}

TEST_CASE(SampledVerifyFixture, next_root_fails_when_the_row_is_unreadable) {
    SamplerCfg sampler;
    sampler.constraint = alternating_grammar();
    sampler.constraint->reset();
    std::vector<int32_t> history;
    std::mt19937_64 rng(0);
    const int32_t rejected = 3;  // 'd' at a position that allows [abc]
    CHECK(constrained_next_root(rejected, sampler, history, rng,
                                [](int &) -> const float * { return nullptr; }) == -1);
    CHECK(constrained_next_root(0, sampler, history, rng,
                                [](int &) -> const float * { return nullptr; }) == 0);
}
