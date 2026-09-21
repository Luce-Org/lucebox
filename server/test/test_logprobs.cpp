// Unit tests for common/logprobs — the raw-logit log-softmax reducer that
// backs `logprobs`/`top_logprobs` on /v1/chat/completions.
//
// Contract under test: logprobs form a proper distribution (sum of exp = 1,
// every value <= 0, finite even at ±1e4 logits), top-K is descending with
// lower-id tie-breaks matching argmax, K=0 yields an empty list, K>vocab
// clamps, and the committed token's own logprob is reported even when it is
// not among the top-K.

#include "CppUnitTestFramework.hpp"

#include "common/logprobs.h"

#include <cmath>
#include <vector>

namespace {
struct LogprobsFixture {};

using dflash::common::TokenLogprob;
using dflash::common::compute_token_logprob;

std::vector<float> logits(std::initializer_list<float> vals) {
    return std::vector<float>(vals);
}
}  // namespace

TEST_CASE(LogprobsFixture, log_softmax_sums_to_one) {
    auto row = logits({0.5f, -1.0f, 2.0f, 0.0f, -0.25f});
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 2,
                              (int) row.size());
    double sum = 0.0;
    for (const auto & [tid, lp] : rec.top) sum += std::exp((double) lp);
    CHECK(std::fabs(sum - 1.0) < 1e-5);
}

TEST_CASE(LogprobsFixture, argmax_token_has_max_logprob) {
    auto row = logits({0.5f, -1.0f, 2.0f, 0.0f, -0.25f});
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 2, 5);
    CHECK(rec.top[0].first == 2);
    CHECK(rec.top[0].second == rec.logprob);
    for (const auto & [tid, lp] : rec.top) CHECK(lp <= rec.top[0].second);
}

TEST_CASE(LogprobsFixture, top_k_descending_and_lower_id_tiebreak) {
    auto row = logits({1.0f, 1.0f, 0.5f, 1.0f, -2.0f});
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 0, 3);
    CHECK(rec.top.size() == 3);
    CHECK(rec.top[0].first == 0);  // equal logits: lower id first
    CHECK(rec.top[1].first == 1);
    CHECK(rec.top[2].first == 3);
    CHECK(rec.top[0].second >= rec.top[1].second);
    CHECK(rec.top[1].second >= rec.top[2].second);
    CHECK(rec.top[0].second == rec.top[1].second);
}

TEST_CASE(LogprobsFixture, top_k_zero_returns_empty_list) {
    auto row = logits({0.1f, 0.9f, -0.4f});
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 1, 0);
    CHECK(rec.top.empty());
    CHECK(rec.token == 1);
    CHECK(std::isfinite(rec.logprob));
    CHECK(rec.logprob <= 0.0f);
}

TEST_CASE(LogprobsFixture, top_k_above_vocab_clamps_to_vocab) {
    auto row = logits({0.1f, 0.9f, -0.4f});
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 1, 999);
    CHECK(rec.top.size() == 3);
}

TEST_CASE(LogprobsFixture, chosen_token_outside_top_k_still_reported) {
    auto row = logits({3.0f, 2.0f, 1.0f, 0.0f});
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 3, 2);
    CHECK(rec.top.size() == 2);
    CHECK(rec.top[0].first == 0);
    CHECK(rec.top[1].first == 1);
    CHECK(rec.token == 3);
    CHECK(rec.logprob < rec.top[1].second);
    CHECK(rec.logprob <= 0.0f);
    // Chosen-token logprob still comes from the same distribution.
    double sum = std::exp((double) rec.logprob);
    for (const auto & [tid, lp] : rec.top) sum += std::exp((double) lp);
    const TokenLogprob full =
        compute_token_logprob(row.data(), (int) row.size(), 3, 4);
    double full_sum = 0.0;
    for (const auto & [tid, lp] : full.top) full_sum += std::exp((double) lp);
    CHECK(std::fabs(full_sum - 1.0) < 1e-5);
    CHECK(sum < full_sum);  // two entries missing from the partial sum
}

TEST_CASE(LogprobsFixture, huge_logits_stay_finite) {
    std::vector<float> row(64, -1e4f);
    row[5] = 1e4f;
    row[17] = 9999.0f;
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 17, 20);
    CHECK(std::isfinite(rec.logprob));
    CHECK(rec.logprob <= 0.0f);
    CHECK(rec.top.size() == 20);
    for (const auto & [tid, lp] : rec.top) {
        CHECK(std::isfinite(lp));
        CHECK(lp <= 0.0f);
    }
    CHECK(rec.top[0].first == 5);   // max logit wins
    CHECK(rec.top[1].first == 17);  // second largest
    // 9999 is one unit below the 1e4 max: -log(1 + e^-1) ~ -0.3133.
    CHECK(std::fabs(rec.top[0].second - (-0.3133f)) < 1e-3f);
}

TEST_CASE(LogprobsFixture, every_logprob_at_most_zero) {
    auto row = logits({0.0f, 0.0f, 0.0f, 0.0f});  // uniform
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 2, 4);
    CHECK(rec.logprob <= 0.0f);
    for (const auto & [tid, lp] : rec.top) CHECK(lp <= 0.0f);
    // Uniform vocab: logprob == -log(4).
    CHECK(std::fabs(rec.logprob - (-std::log(4.0f))) < 1e-5f);
}

TEST_CASE(LogprobsFixture, out_of_range_token_returns_default) {
    auto row = logits({0.1f, 0.9f});
    const TokenLogprob rec =
        compute_token_logprob(row.data(), (int) row.size(), 7, 2);
    CHECK(rec.token == 7);
    CHECK(rec.top.empty());
}
