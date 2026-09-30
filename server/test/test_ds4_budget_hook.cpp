// Level-2 thinking-budget force-close rule. Pure logic: no model, no GPU.
//
// The case that matters most is CONTINUES_AFTER_SEQUENCE. The previous DeepSeek4
// implementation emitted the close sequence and then broke out of the decode loop, so the
// reserved reply budget was never spent and the answer was always empty -- while finish_reason
// still said "stop". Measured on DeepSeek-V4-Flash-0731: completions landed on exactly
// thinking_ceiling + close_len for close sequences of 1, 3 and 23 tokens, and one item returned
// 22,706 characters of reasoning against 1 character of answer.

#include "CppUnitTestFramework.hpp"

#include "deepseek4/deepseek4_budget_hook.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct BudgetHookFixture {};

using luce::deepseek4::budget_hook_apply;

struct HookState {
    bool        started = false;
    std::size_t pos     = 0;
    bool        forced  = false;

    int32_t step(const std::vector<int32_t> & close, int remaining, int hard, int32_t sampled) {
        return budget_hook_apply(close, remaining, hard, sampled, started, pos, forced);
    }
};

const std::vector<int32_t> kClose3 = {101, 102, 103};

}  // namespace

namespace BudgetHookTests {

// Empty close sequence must be a complete no-op: the hook is how thinking budgets are
// enforced, and a model without a close token should never have its stream altered.
TEST_CASE(BudgetHookFixture, disabled_when_no_close_tokens) {
    HookState st;
    const std::vector<int32_t> none;
    REQUIRE(st.step(none, /*remaining=*/1, /*hard=*/4096, /*sampled=*/77) == 77);
    REQUIRE(!st.started);
    REQUIRE(!st.forced);
}

// Inside the thinking window, tokens pass through untouched.
TEST_CASE(BudgetHookFixture, passthrough_before_threshold) {
    HookState st;
    REQUIRE(st.step(kClose3, /*remaining=*/5000, /*hard=*/4096, /*sampled=*/42) == 42);
    REQUIRE(!st.started);
    REQUIRE(!st.forced);
}

// At the boundary the sampled token is replaced by close[0] and forced_close is reported.
TEST_CASE(BudgetHookFixture, fires_at_threshold) {
    HookState st;
    REQUIRE(st.step(kClose3, /*remaining=*/4096, /*hard=*/4096, /*sampled=*/42) == 101);
    REQUIRE(st.started);
    REQUIRE(st.forced);
    REQUIRE(st.pos == 1);
}

// A multi-token close sequence is emitted one token per step, so each one goes through the
// normal decode path and the next forward pass sees it.
TEST_CASE(BudgetHookFixture, injects_sequence_one_token_per_step) {
    HookState st;
    REQUIRE(st.step(kClose3, 4096, 4096, 42) == 101);
    REQUIRE(st.step(kClose3, 4095, 4096, 43) == 102);
    REQUIRE(st.step(kClose3, 4094, 4096, 44) == 103);
    REQUIRE(st.pos == 3);
}

// THE REGRESSION GUARD. Once the close sequence is complete the hook must stop intervening so
// the model can write a visible answer in the reserved budget. The old implementation ended
// generation here, which is why every capped item scored zero.
TEST_CASE(BudgetHookFixture, continues_after_sequence) {
    HookState st;
    st.step(kClose3, 4096, 4096, 42);
    st.step(kClose3, 4095, 4096, 43);
    st.step(kClose3, 4094, 4096, 44);

    // Real answer tokens now flow through unmodified, for the rest of the window.
    REQUIRE(st.step(kClose3, 4093, 4096, 900) == 900);
    REQUIRE(st.step(kClose3, 4092, 4096, 901) == 901);
    REQUIRE(st.step(kClose3, 1, 4096, 902) == 902);
    REQUIRE(st.pos == 3);
}

// If the model reaches the boundary already emitting close[0], consume it as the start of the
// sequence rather than overriding it with the same value.
TEST_CASE(BudgetHookFixture, consumes_model_self_close) {
    HookState st;
    REQUIRE(st.step(kClose3, 4096, 4096, /*sampled=*/101) == 101);
    REQUIRE(st.started);
    REQUIRE(st.pos == 1);
    // The remainder of the sequence still follows.
    REQUIRE(st.step(kClose3, 4095, 4096, 55) == 102);
}

// A single-token close sequence is the common case (a bare `</think>`): fire once, then hand
// the stream straight back to the model.
TEST_CASE(BudgetHookFixture, single_token_close_then_free) {
    HookState st;
    const std::vector<int32_t> one = {7};
    REQUIRE(st.step(one, 4096, 4096, 42) == 7);
    REQUIRE(st.step(one, 4095, 4096, 500) == 500);
    REQUIRE(st.step(one, 4094, 4096, 501) == 501);
}

// hard_limit == 0 means no reply reserve, so the hook should never fire while the window lasts.
TEST_CASE(BudgetHookFixture, never_fires_with_zero_reserve) {
    HookState st;
    REQUIRE(st.step(kClose3, /*remaining=*/1, /*hard=*/0, /*sampled=*/42) == 42);
    REQUIRE(!st.started);
    REQUIRE(!st.forced);
}

// The DSpark form of the rule (spec_budget_hook_step) must emit exactly what the AR rule emits
// token by token, whatever the drafts, their acceptance and the verify width. A deterministic
// toy model makes both paths comparable: its next token is a hash of everything emitted so far.
namespace {

int32_t toy_model(const std::vector<int32_t> & seq) {
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : seq) h = (h ^ (uint64_t) t) * 1099511628211ull;
    return 1 + (int32_t) (h % 40);
}

std::vector<int32_t> ar_reference(const std::vector<int32_t> & close, int n_gen, int hard) {
    std::vector<int32_t> seq;
    HookState st;
    for (int g = 0; g < n_gen; g++) seq.push_back(st.step(close, n_gen - g, hard, toy_model(seq)));
    return seq;
}

std::vector<int32_t> spec_decode(const std::vector<int32_t> & close, int n_gen, int hard,
                                 int q_cap, std::mt19937_64 & rng) {
    using namespace luce::deepseek4;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_int_distribution<int32_t> wrong(1, 40);
    std::vector<int32_t> seq = {toy_model({})};   // the seed, emitted before speculation
    SpecBudgetHookState st;
    while ((int) seq.size() < n_gen) {
        const bool forcing = spec_budget_hook_forcing(close, st);
        std::vector<int32_t> draft = {seq.back()};
        if (forcing) {
            const int k = std::min((int) (close.size() - st.inject_pos), q_cap - 1);
            for (int i = 0; i < k; i++) draft.push_back(close[st.inject_pos + (size_t) i]);
        } else {
            std::vector<int32_t> ctx = seq;
            for (int i = 1; i < q_cap; i++) {
                const int32_t right = toy_model(ctx);
                const int32_t d = unif(rng) < 0.7 ? right : wrong(rng);
                draft.push_back(d);
                ctx.push_back(d);
            }
        }
        const int q = (int) draft.size();
        int accept = 1;
        std::vector<int32_t> ctx = seq;
        if (forcing) {
            accept = q;
            for (int i = 1; i < q; i++) ctx.push_back(draft[(size_t) i]);
        } else {
            for (int i = 1; i < q && draft[(size_t) i] == toy_model(ctx); i++) {
                ctx.push_back(draft[(size_t) i]);
                accept++;
            }
        }
        int32_t bonus = toy_model(ctx);
        spec_budget_hook_step(close, n_gen - (int) seq.size(), hard, forcing,
                              forcing ? q - 1 : 0, accept, bonus, st);
        for (int i = 1; i < accept && (int) seq.size() < n_gen; i++) seq.push_back(draft[(size_t) i]);
        if ((int) seq.size() < n_gen) seq.push_back(bonus);
    }
    return seq;
}

}  // namespace

TEST_CASE(BudgetHookFixture, spec_step_matches_ar_rule) {
    std::mt19937_64 rng(2026);
    const std::vector<std::vector<int32_t>> closes = {{101}, {101, 102, 103}, {101, 102, 103, 104, 105, 106, 107}};
    int fired = 0;
    for (int trial = 0; trial < 3000; trial++) {
        const auto & close = closes[(size_t) trial % closes.size()];
        const int n_gen = 8 + (int) (rng() % 80);
        const int hard = (int) (rng() % (uint64_t) (n_gen - 1));   // never fires on the seed
        const int q_cap = 2 + (int) (rng() % 5);
        const auto want = ar_reference(close, n_gen, hard);
        const auto got = spec_decode(close, n_gen, hard, q_cap, rng);
        REQUIRE(got == want);
        fired += std::find(want.begin(), want.end(), 101) != want.end();
    }
    REQUIRE(fired > 2000);   // the close actually happened in most trials
}

}  // namespace BudgetHookTests
