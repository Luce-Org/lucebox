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

#include <cstdint>
#include <vector>

namespace {

struct BudgetHookFixture {};

using dflash::deepseek4::budget_hook_apply;

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

}  // namespace BudgetHookTests
