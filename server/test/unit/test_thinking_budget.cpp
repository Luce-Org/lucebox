// Generation-owned thinking-budget behavior. Pure logic: no model or GPU.
#include "CppUnitTestFramework.hpp"

#include "common/thinking_budget.h"
#include "qwen4exp/qwen4exp_mtp.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace {

using luce::common::BudgetHook;
using luce::common::ThinkingBudget;

struct BudgetHookFixture : CppUnitTestFramework::CommonFixture {
    using CommonFixture::CommonFixture;

    void expect_step(ThinkingBudget & budget, int32_t candidate, int32_t token, bool cut);
    std::vector<int32_t> speculative_decode(ThinkingBudget & budget, int q_cap,
                                          std::mt19937_64 & rng);
    std::vector<int32_t> spec_then_ar(ThinkingBudget & budget, int q_cap,
                                    std::mt19937_64 & rng);
    void expect_same_state(const ThinkingBudget & got, const ThinkingBudget & want);
};

BudgetHook make_hook(std::vector<int32_t> close, std::vector<int32_t> marker, int hard) {
    BudgetHook h;
    h.close_token_ids = std::move(close);
    h.marker_token_ids = std::move(marker);
    h.hard_limit_remaining = hard;
    return h;
}

const std::vector<int32_t> kClose3 = {101, 102, 103};
const std::vector<int32_t> kMarker = {103};

void BudgetHookFixture::expect_step(ThinkingBudget & budget, int32_t candidate, int32_t token, bool cut) {
    const int before = budget.remaining();
    const auto decision = budget.apply(candidate);
    REQUIRE(decision.token == token);
    REQUIRE(decision.cut == cut);
    REQUIRE(budget.remaining() == before - 1);
}

}  // namespace

namespace BudgetHookTests {

TEST_CASE(BudgetHookFixture, disabled_budgets_preserve_output_and_accounting) {
    const auto hook = make_hook({}, kMarker, 4096);
    ThinkingBudget empty(hook, 3);
    ThinkingBudget absent(nullptr, 3);
    for (ThinkingBudget * budget : {&empty, &absent}) {
        REQUIRE(budget->settled());
        REQUIRE(!budget->needs_ar_tail(3));
        expect_step(*budget, 77, 77, false);
        std::vector<int32_t> block = {103, 78};
        REQUIRE(budget->apply_block(block.data(), 2) == 2);
        REQUIRE(block == std::vector<int32_t>({103, 78}));
        REQUIRE(budget->remaining() == 0);
        REQUIRE(!budget->forced_close());
    }
}

TEST_CASE(BudgetHookFixture, boundary_uses_emitted_token_count) {
    const auto hook = make_hook(kClose3, kMarker, 4);
    ThinkingBudget budget(hook, 6);
    expect_step(budget, 40, 40, false);
    expect_step(budget, 41, 41, false);
    REQUIRE(!budget.forced_close());
    expect_step(budget, 42, 101, true);
    REQUIRE(budget.forced_close());
    REQUIRE(!budget.naturally_closed());
    REQUIRE(budget.injecting());
    REQUIRE(budget.pending_close_tokens() == 2);
    REQUIRE(budget.pending_close_token(0) == 102);
    REQUIRE(budget.pending_close_token(1) == 103);
    expect_step(budget, 43, 102, true);
    expect_step(budget, 44, 103, true);
    REQUIRE(budget.settled());
    REQUIRE(!budget.injecting());
    REQUIRE(budget.pending_close_tokens() == 0);
    expect_step(budget, 900, 900, false);
    REQUIRE(budget.remaining() == 0);
}

TEST_CASE(BudgetHookFixture, force_on_seed_continues_into_reserved_answer) {
    const auto hook = make_hook(kClose3, kMarker, 6);
    ThinkingBudget budget(hook, 6);
    expect_step(budget, 42, 101, true);
    expect_step(budget, 43, 102, true);
    expect_step(budget, 44, 103, true);
    std::vector<int32_t> answer = {900, 901, 902};
    REQUIRE(budget.apply_block(answer.data(), 3) == 3);
    REQUIRE(answer == std::vector<int32_t>({900, 901, 902}));
    REQUIRE(budget.remaining() == 0);
    REQUIRE(budget.forced_close());
    REQUIRE(budget.settled());
}

TEST_CASE(BudgetHookFixture, hint_token_at_boundary_is_not_a_natural_close) {
    const auto hook = make_hook(kClose3, kMarker, 4);
    ThinkingBudget budget(hook, 4);
    expect_step(budget, 101, 101, true);
    REQUIRE(budget.forced_close());
    REQUIRE(!budget.naturally_closed());
    expect_step(budget, 55, 102, true);
}

TEST_CASE(BudgetHookFixture, natural_close_seed_at_boundary_preserves_answer) {
    const auto hook = make_hook({7}, {7}, 4);
    ThinkingBudget budget(hook, 4);
    expect_step(budget, 7, 7, false);
    REQUIRE(budget.naturally_closed());
    REQUIRE(!budget.forced_close());
    REQUIRE(budget.settled());
    REQUIRE(!budget.needs_ar_tail(4));
    std::vector<int32_t> answer = {500, 501, 502};
    REQUIRE(budget.apply_block(answer.data(), 3) == 3);
    REQUIRE(answer == std::vector<int32_t>({500, 501, 502}));
    REQUIRE(budget.remaining() == 0);
}

TEST_CASE(BudgetHookFixture, single_token_close_then_free) {
    const auto hook = make_hook({7}, {7}, 3);
    ThinkingBudget budget(hook, 3);
    expect_step(budget, 42, 7, true);
    REQUIRE(budget.settled());
    expect_step(budget, 500, 500, false);
    expect_step(budget, 501, 501, false);
    REQUIRE(budget.remaining() == 0);
}

TEST_CASE(BudgetHookFixture, never_fires_with_zero_reserve) {
    const auto hook = make_hook(kClose3, kMarker, 0);
    ThinkingBudget budget(hook, 3);
    for (int32_t token : {40, 41, 42}) expect_step(budget, token, token, false);
    REQUIRE(budget.remaining() == 0);
    REQUIRE(!budget.forced_close());
}

TEST_CASE(BudgetHookFixture, natural_close_then_long_answer_is_untouched) {
    const auto hook = make_hook(kClose3, kMarker, 4);
    ThinkingBudget budget(hook, 12);
    const std::vector<int32_t> model = {10, 11, 103, 20, 21, 22, 23, 24, 25, 26, 27, 28};
    for (int32_t token : model) expect_step(budget, token, token, false);
    REQUIRE(budget.remaining() == 0);
    REQUIRE(budget.naturally_closed());
    REQUIRE(!budget.forced_close());
}

TEST_CASE(BudgetHookFixture, eos_at_threshold_after_natural_close) {
    const auto hook = make_hook(kClose3, kMarker, 4);
    ThinkingBudget budget(hook, 12);
    const int32_t eos = 2;
    for (int32_t token : {10, 103, 20, 21, 22, 23, 24, 25, eos}) {
        expect_step(budget, token, token, false);
    }
    REQUIRE(budget.remaining() == 3);
    REQUIRE(budget.naturally_closed());
    REQUIRE(!budget.forced_close());
}

TEST_CASE(BudgetHookFixture, natural_close_mid_block) {
    const auto hook = make_hook(kClose3, kMarker, 4);
    ThinkingBudget budget(hook, 6);
    std::vector<int32_t> block = {30, 103, 31, 32};
    const auto want = block;
    REQUIRE(budget.apply_block(block.data(), 4) == 4);
    REQUIRE(block == want);
    REQUIRE(budget.remaining() == 2);
    REQUIRE(budget.naturally_closed());
    REQUIRE(!budget.forced_close());
}

// Qwen4Exp samples verify rows lazily, then compares the substituted token
// with the draft. Exercise that acceptance contract with the real controller:
// a retained forced draft is safe, but a replaced mismatch must stop the walk.
TEST_CASE(BudgetHookFixture, qwen4exp_mtp_accepts_only_budget_substituted_prefix) {
    for (bool matching_drafts : {false, true}) {
        const auto hook = make_hook(kClose3, kMarker, 5);
        ThinkingBudget budget(hook, 6);
        expect_step(budget, 40, 40, false);  // prefill seed
        const std::vector<int32_t> drafts = matching_drafts
            ? std::vector<int32_t>{101, 102} : std::vector<int32_t>{41, 42};
        std::vector<int32_t> samples;
        luce::common::Qwen4ExpMtpAcceptance accepted;
        for (int i = 0; i <= 2; ++i) {
            samples.push_back(budget.apply(41 + i).token);
            accepted = luce::common::qwen4exp_mtp_accept(
                drafts.data(), 2, samples.data(), (int)samples.size());
            if (accepted.n_accepted != i + 1) break;
        }
        REQUIRE(accepted.n_emitted == (matching_drafts ? 3 : 1));
        REQUIRE(accepted.n_accepted == (matching_drafts ? 2 : 0));
        REQUIRE(accepted.emitted[0] == 101);
        REQUIRE(budget.remaining() == 5 - accepted.n_emitted);
        REQUIRE(budget.forced_close());
        if (matching_drafts) {
            REQUIRE(accepted.emitted[1] == 102);
            REQUIRE(accepted.emitted[2] == 103);
        } else {
            // Rejected verify rows did not consume the remaining close tokens.
            expect_step(budget, 80, 102, true);
            expect_step(budget, 81, 103, true);
        }
        REQUIRE(budget.settled());
        expect_step(budget, 900, 900, false);
        expect_step(budget, 901, 901, false);
        REQUIRE(budget.remaining() == 0);
    }
}

TEST_CASE(BudgetHookFixture, qwen4exp_mtp_natural_close_across_seed_preserves_answer) {
    const auto hook = make_hook({101, 5, 6}, {5, 6}, 5);
    ThinkingBudget budget(hook, 6);
    expect_step(budget, 5, 5, false);  // first marker token is the prefill seed
    const std::vector<int32_t> drafts = {6, 50};
    std::vector<int32_t> samples;
    for (int32_t token : {6, 50, 51}) samples.push_back(budget.apply(token).token);
    const auto accepted = luce::common::qwen4exp_mtp_accept(
        drafts.data(), 2, samples.data(), (int)samples.size());
    REQUIRE(accepted.n_accepted == 2);
    REQUIRE(accepted.n_emitted == 3);
    REQUIRE(accepted.emitted[0] == 6);
    REQUIRE(accepted.emitted[1] == 50);
    REQUIRE(accepted.emitted[2] == 51);
    REQUIRE(budget.naturally_closed());
    REQUIRE(!budget.forced_close());
    expect_step(budget, 52, 52, false);
    expect_step(budget, 53, 53, false);
    REQUIRE(budget.remaining() == 0);
}

TEST_CASE(BudgetHookFixture, rejected_draft_marker_does_not_count) {
    const auto hook = make_hook(kClose3, kMarker, 4);
    ThinkingBudget budget(hook, 8);
    // Draft [40, 41, 103] was rejected at the marker; only accepted output is supplied.
    std::vector<int32_t> block = {40, 41, 42};
    REQUIRE(budget.apply_block(block.data(), 3) == 3);
    REQUIRE(!budget.naturally_closed());
    std::vector<int32_t> next = {43, 44, 45};
    REQUIRE(budget.apply_block(next.data(), 3) == 2);
    REQUIRE(next[0] == 43);
    REQUIRE(next[1] == 101);
    REQUIRE(budget.remaining() == 3);
    REQUIRE(budget.forced_close());
}

TEST_CASE(BudgetHookFixture, marker_split_between_seed_block_and_ar_tail) {
    const auto hook = make_hook({101, 5, 6, 7}, {5, 6, 7}, 4);
    ThinkingBudget budget(hook, 6);
    expect_step(budget, 5, 5, false);
    std::vector<int32_t> block = {6};
    REQUIRE(budget.apply_block(block.data(), 1) == 1);
    REQUIRE(block[0] == 6);
    REQUIRE(!budget.naturally_closed());
    REQUIRE(budget.needs_ar_tail(2));
    // Completion at the reserve boundary takes priority over forcing.
    expect_step(budget, 7, 7, false);
    REQUIRE(budget.naturally_closed());
    REQUIRE(!budget.needs_ar_tail(2));
    std::vector<int32_t> answer = {50, 51, 52};
    REQUIRE(budget.apply_block(answer.data(), 3) == 3);
    REQUIRE(answer == std::vector<int32_t>({50, 51, 52}));
    REQUIRE(budget.remaining() == 0);
    REQUIRE(!budget.forced_close());
}

TEST_CASE(BudgetHookFixture, split_marker_across_blocks) {
    const auto hook = make_hook({101, 102, 5, 6}, {5, 6}, 4);
    ThinkingBudget budget(hook, 10);
    std::vector<int32_t> a = {40, 5};
    REQUIRE(budget.apply_block(a.data(), 2) == 2);
    REQUIRE(!budget.naturally_closed());
    std::vector<int32_t> b = {6, 50, 51, 52, 53, 54};
    const auto want = b;
    REQUIRE(budget.apply_block(b.data(), 6) == 6);
    REQUIRE(b == want);
    REQUIRE(budget.remaining() == 2);
    REQUIRE(budget.naturally_closed());
    REQUIRE(!budget.forced_close());
}

TEST_CASE(BudgetHookFixture, broken_marker_prefix_is_not_a_close) {
    const auto hook = make_hook({101, 102, 5, 6}, {5, 6}, 2);
    ThinkingBudget budget(hook, 5);
    for (int32_t token : {5, 101, 6}) expect_step(budget, token, token, false);
    REQUIRE(!budget.naturally_closed());
    expect_step(budget, 6, 101, true);
    REQUIRE(budget.forced_close());
}

TEST_CASE(BudgetHookFixture, overlapping_marker_prefix_survives_mismatch) {
    const auto hook = make_hook({101, 5, 6, 5, 7}, {5, 6, 5, 7}, 2);
    ThinkingBudget budget(hook, 7);
    // The second 6 mismatches the first prefix, but [5,6,5] is still a live suffix.
    for (int32_t token : {5, 6, 5, 6, 5}) {
        expect_step(budget, token, token, false);
        REQUIRE(!budget.naturally_closed());
    }
    expect_step(budget, 7, 7, false);
    REQUIRE(budget.naturally_closed());
    REQUIRE(!budget.forced_close());
    expect_step(budget, 900, 900, false);
}

TEST_CASE(BudgetHookFixture, marker_before_transition_hint_is_natural_without_hint) {
    // Channel-end marker followed by an answer-transition hint (Gemma-style layout).
    const auto hook = make_hook({7, 101, 102}, {7}, 4);
    ThinkingBudget budget(hook, 4);
    expect_step(budget, 7, 7, false);
    REQUIRE(budget.naturally_closed());
    REQUIRE(budget.settled());
    for (int32_t token : {900, 901, 902}) expect_step(budget, token, token, false);
    REQUIRE(!budget.forced_close());
}

TEST_CASE(BudgetHookFixture, forced_marker_does_not_skip_trailing_transition) {
    const auto hook = make_hook({7, 101, 102}, {7}, 5);
    ThinkingBudget budget(hook, 5);
    expect_step(budget, 42, 7, true);
    REQUIRE(!budget.naturally_closed());
    REQUIRE(!budget.settled());
    std::vector<int32_t> block = {101, 102, 900, 901};
    const auto want = block;
    REQUIRE(budget.apply_block(block.data(), 4) == 4);
    REQUIRE(block == want);
    REQUIRE(budget.remaining() == 0);
    REQUIRE(budget.settled());
    REQUIRE(budget.forced_close());
}

TEST_CASE(BudgetHookFixture, empty_marker_disables_only_natural_detection) {
    const auto hook = make_hook(kClose3, {}, 4);
    ThinkingBudget budget(hook, 5);
    expect_step(budget, 103, 103, false);
    REQUIRE(!budget.naturally_closed());
    expect_step(budget, 103, 101, true);
    REQUIRE(budget.forced_close());
}

TEST_CASE(BudgetHookFixture, first_equal_forced_token_cuts_later_equal_drafts_survive) {
    const auto hook = make_hook(kClose3, kMarker, 5);
    ThinkingBudget budget(hook, 5);
    std::vector<int32_t> first = {101, 102, 103, 800};
    REQUIRE(budget.apply_block(first.data(), 4) == 1);
    REQUIRE(first[0] == 101);
    REQUIRE(budget.remaining() == 4);
    REQUIRE(budget.pending_close_tokens() == 2);
    std::vector<int32_t> next = {102, 103, 900, 901};
    REQUIRE(budget.apply_block(next.data(), 4) == 4);
    REQUIRE(next == std::vector<int32_t>({102, 103, 900, 901}));
    REQUIRE(budget.remaining() == 0);
    REQUIRE(budget.settled());
}

TEST_CASE(BudgetHookFixture, forcing_mid_block_discards_suffix_without_consuming_budget) {
    const auto hook = make_hook(kClose3, kMarker, 4);
    ThinkingBudget budget(hook, 12);
    std::vector<int32_t> out;
    for (int32_t token : {1, 2, 3, 4, 5, 6}) out.push_back(budget.apply(token).token);
    std::vector<int32_t> block = {7, 8, 9, 103};
    const int kept = budget.apply_block(block.data(), 4);
    REQUIRE(kept == 3);
    REQUIRE(std::vector<int32_t>(block.begin(), block.begin() + kept) ==
            std::vector<int32_t>({7, 8, 101}));
    REQUIRE(budget.remaining() == 3);
    REQUIRE(budget.injecting());
    REQUIRE(!budget.naturally_closed());
    out.insert(out.end(), block.begin(), block.begin() + kept);
    for (int32_t model : {60, 61, 62}) out.push_back(budget.apply(model).token);
    REQUIRE(out == std::vector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 101, 102, 103, 62}));
    REQUIRE(budget.remaining() == 0);
    REQUIRE(budget.forced_close());
    REQUIRE(budget.settled());
}

TEST_CASE(BudgetHookFixture, mismatching_pending_draft_cuts_only_at_replacement) {
    const auto hook = make_hook({101, 102, 103, 104}, {104}, 6);
    ThinkingBudget budget(hook, 6);
    expect_step(budget, 42, 101, true);
    std::vector<int32_t> block = {102, 99, 800, 801};
    REQUIRE(budget.apply_block(block.data(), 4) == 2);
    REQUIRE(block[0] == 102);
    REQUIRE(block[1] == 103);
    REQUIRE(budget.remaining() == 3);
    REQUIRE(budget.pending_close_tokens() == 1);
    expect_step(budget, 104, 104, false);
    std::vector<int32_t> answer = {900, 901};
    REQUIRE(budget.apply_block(answer.data(), 2) == 2);
    REQUIRE(answer == std::vector<int32_t>({900, 901}));
    REQUIRE(budget.remaining() == 0);
}

}  // namespace BudgetHookTests

namespace {

int32_t toy_model(const std::vector<int32_t> & seq) {
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : seq) h = (h ^ (uint64_t) t) * 1099511628211ull;
    return 1 + (int32_t) ((h >> 32) % 40);
}

std::vector<int32_t> ar_reference(ThinkingBudget & budget) {
    std::vector<int32_t> seq;
    while (budget.remaining() > 0) seq.push_back(budget.apply(toy_model(seq)).token);
    return seq;
}

// Seed, verified drafts and bonus; pending close tokens can themselves be drafted.
std::vector<int32_t> BudgetHookFixture::speculative_decode(ThinkingBudget & budget, int q_cap,
                                                        std::mt19937_64 & rng) {
    std::uniform_int_distribution<int32_t> wrong(1, 40);
    std::vector<int32_t> seq;
    seq.push_back(budget.apply(toy_model(seq)).token);
    while (budget.remaining() > 0) {
        const int width = std::min(q_cap, budget.remaining());
        std::vector<int32_t> ctx = seq;
        std::vector<int32_t> block;
        for (int i = 0; i < width; ++i) {
            const bool forced_draft = (size_t) i < budget.pending_close_tokens();
            const int32_t right = forced_draft ? budget.pending_close_token((size_t) i) : toy_model(ctx);
            const int32_t draft = rng() % 10 < 7 ? right : wrong(rng);
            block.push_back(right);
            ctx.push_back(right);
            // Rejected draft is not output; the target's token is the final bonus.
            if (draft != right) break;
        }
        const int before = budget.remaining();
        const int kept = budget.apply_block(block.data(), (int) block.size());
        REQUIRE(kept > 0);
        REQUIRE(kept <= (int) block.size());
        REQUIRE(budget.remaining() == before - kept);
        seq.insert(seq.end(), block.begin(), block.begin() + kept);
    }
    return seq;
}

// Block decoding hands its existing controller to AR before reaching the reserve.
std::vector<int32_t> BudgetHookFixture::spec_then_ar(ThinkingBudget & budget, int q_cap,
                                                  std::mt19937_64 & rng) {
    std::vector<int32_t> seq;
    while (budget.remaining() > 0 && !budget.needs_ar_tail(q_cap)) {
        std::vector<int32_t> block;
        std::vector<int32_t> ctx = seq;
        const int width = std::min(q_cap, budget.remaining());
        for (int i = 0; i < width; i++) {
            const int32_t right = toy_model(ctx);
            block.push_back(right);
            ctx.push_back(right);
            if (i > 0 && rng() % 10 >= 7) break;
        }
        const int before = budget.remaining();
        const int kept = budget.apply_block(block.data(), (int) block.size());
        REQUIRE(budget.remaining() == before - kept);
        seq.insert(seq.end(), block.begin(), block.begin() + kept);
        if (budget.forced_close()) break;
    }
    while (budget.remaining() > 0) seq.push_back(budget.apply(toy_model(seq)).token);
    return seq;
}

void BudgetHookFixture::expect_same_state(const ThinkingBudget & got, const ThinkingBudget & want) {
    REQUIRE(got.remaining() == want.remaining());
    REQUIRE(got.forced_close() == want.forced_close());
    REQUIRE(got.naturally_closed() == want.naturally_closed());
    REQUIRE(got.settled() == want.settled());
    REQUIRE(got.injecting() == want.injecting());
    REQUIRE(got.pending_close_tokens() == want.pending_close_tokens());
    for (size_t i = 0; i < want.pending_close_tokens(); ++i) {
        REQUIRE(got.pending_close_token(i) == want.pending_close_token(i));
    }
}

}  // namespace

namespace BudgetHookTests {

TEST_CASE(BudgetHookFixture, spec_matches_ar_rule) {
    std::mt19937_64 rng(2026);
    const std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>> configs = {
        {{7}, {7}},
        {{101, 102, 7}, {7}},
        {{101, 102, 103, 104, 105, 106, 7}, {7}},
        {{101, 102, 5, 6}, {5, 6}},
        {{7, 101, 102}, {7}},
        {{101, 7, 102}, {7}},
        {{101, 5, 6, 5, 7}, {5, 6, 5, 7}},
    };
    int forced = 0;
    int natural_then_answer_past_threshold = 0;
    int forced_seeds = 0;
    for (int trial = 0; trial < 4000; trial++) {
        const auto & cfg = configs[(size_t) trial % configs.size()];
        const int n_gen = 1 + (int) (rng() % 87);
        // Include seed forcing, zero reserve and windows shorter than the close sequence.
        const int hard = (int) (rng() % (uint64_t) (n_gen + 1));
        const auto hook = make_hook(cfg.first, cfg.second, hard);
        const int q_cap = 1 + (int) (rng() % 6);
        ThinkingBudget want_budget(hook, n_gen);
        const auto want = ar_reference(want_budget);
        REQUIRE((int) want.size() == n_gen);
        REQUIRE(want_budget.remaining() == 0);
        ThinkingBudget spec_budget(hook, n_gen);
        REQUIRE(speculative_decode(spec_budget, q_cap, rng) == want);
        expect_same_state(spec_budget, want_budget);
        ThinkingBudget tail_budget(hook, n_gen);
        REQUIRE(spec_then_ar(tail_budget, q_cap, rng) == want);
        expect_same_state(tail_budget, want_budget);
        REQUIRE(!(want_budget.naturally_closed() && want_budget.forced_close()));
        forced += want_budget.forced_close();
        natural_then_answer_past_threshold += want_budget.naturally_closed() && hard > 0;
        forced_seeds += want_budget.forced_close() && hard == n_gen;
    }
    REQUIRE(forced > 0);
    REQUIRE(natural_then_answer_past_threshold > 0);
    REQUIRE(forced_seeds > 0);
}

}  // namespace BudgetHookTests

namespace {

// Distinct token ids [first, first + n): text with no period of its own.
std::vector<int32_t> distinct_tokens(int32_t first, int n) {
    std::vector<int32_t> out((size_t) n);
    for (int i = 0; i < n; ++i) out[(size_t) i] = first + i;
    return out;
}

void append(std::vector<int32_t> & out, const std::vector<int32_t> & part) {
    out.insert(out.end(), part.begin(), part.end());
}

// Feeds `reply` one token at a time, as the decoders call the watchdog.
// Returns {tokens emitted when it fired, period}, or {0, 0} if it never did.
std::pair<size_t, int> first_loop(const std::vector<int32_t> & reply) {
    std::vector<int32_t> emitted;
    for (int32_t token : reply) {
        emitted.push_back(token);
        if (const int period = luce::common::post_close_loop_period(emitted)) {
            return {emitted.size(), period};
        }
    }
    return {0, 0};
}

}  // namespace

namespace PostCloseLoopTests {

// A test file that reads the same range again to check a cache repeats a line
// on purpose. Cutting there truncated a tool call mid-argument.
TEST_CASE(BudgetHookFixture, repeated_code_line_is_not_a_loop) {
    const auto line = distinct_tokens(500, 21);
    std::vector<int32_t> reply = distinct_tokens(1000, 400);
    for (int i = 0; i < 3; ++i) append(reply, line);
    append(reply, distinct_tokens(2000, 200));
    REQUIRE(first_loop(reply) == std::make_pair(size_t{0}, 0));
}

// The aime2025-02 shape: a ~50-token phrase repeated verbatim until max_tokens.
TEST_CASE(BudgetHookFixture, verbatim_runaway_is_cut_within_a_few_periods) {
    const int content = 400, period = 50;
    const auto phrase = distinct_tokens(500, period);
    std::vector<int32_t> reply = distinct_tokens(1000, content);
    for (int i = 0; i < 80; ++i) append(reply, phrase);  // until the reply reserve runs out
    const auto [emitted, found] = first_loop(reply);
    REQUIRE(found == period);
    REQUIRE(emitted > (size_t) content);
    REQUIRE(emitted - (size_t) content <= (size_t) (6 * period));
}

}  // namespace PostCloseLoopTests
