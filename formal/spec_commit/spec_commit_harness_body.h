// Shared body for the generated and legacy spec-commit exactness harnesses.

#include "common/spec_commit.h"

#include <cassert>
#include <cstdint>

#ifndef LUCEBOX_FORMAL_CONTRACT_SYMBOL
#error "LUCEBOX_FORMAL_CONTRACT_SYMBOL must name the production decision type"
#endif

#ifndef LUCEBOX_FORMAL_MAX_WIDTH
#define LUCEBOX_FORMAL_MAX_WIDTH 4
#endif

extern "C" unsigned int nondet_uint();
extern "C" int32_t nondet_int();
extern "C" void __ESBMC_assume(bool);

using ContractDecision = LUCEBOX_FORMAL_CONTRACT_SYMBOL;

static unsigned int bounded(unsigned int upper_exclusive) {
    const unsigned int value = nondet_uint();
    __ESBMC_assume(value < upper_exclusive);
    return value;
}

static void assert_safe_tokens(
        const ContractDecision & decision,
        const int32_t * draft,
        int verify_count,
        int32_t expected_bonus) {
    for (int index = 0; index < decision.commit_count(); ++index) {
        int32_t token = nondet_int();
        assert(decision.token_at(index, draft, verify_count, token));
        if (index < decision.accepted_count()) {
            assert(token == draft[index]);
        } else {
            assert(index == decision.accepted_count());
            assert(decision.has_bonus());
            assert(decision.commits_bonus());
            assert(token == expected_bonus);
        }
    }

    int32_t ignored = nondet_int();
    assert(!decision.token_at(-1, draft, verify_count, ignored));
    assert(!decision.token_at(
        decision.commit_count(), draft, verify_count, ignored));
}

int main() {
    static_assert(LUCEBOX_FORMAL_MAX_WIDTH > 0);
    const int verify_count =
        static_cast<int>(bounded(LUCEBOX_FORMAL_MAX_WIDTH)) + 1;
    const int commit_budget =
        static_cast<int>(bounded(LUCEBOX_FORMAL_MAX_WIDTH + 1));

    int32_t draft[LUCEBOX_FORMAL_MAX_WIDTH];
    int32_t target[LUCEBOX_FORMAL_MAX_WIDTH];
    for (int index = 0; index < verify_count; ++index) {
        draft[index] = nondet_int();
        target[index] = nondet_int();
        __ESBMC_assume(draft[index] >= 0);
        __ESBMC_assume(target[index] >= 0);
    }

    const ContractDecision greedy = ContractDecision::greedy(
        draft, verify_count, target, verify_count,
        verify_count, commit_budget);
    assert(greedy.valid());
    assert(greedy.accepted_count() >= 1);
    assert(greedy.accepted_count() <= verify_count);
    assert(greedy.commit_count() >= 0);
    assert(greedy.commit_count() <= commit_budget);
    assert(greedy.commit_count() <= greedy.accepted_count() + 1);

    for (int index = 1; index < greedy.accepted_count(); ++index) {
        assert(draft[index] == target[index - 1]);
    }
    if (greedy.accepted_count() < verify_count) {
        assert(
            draft[greedy.accepted_count()] !=
            target[greedy.accepted_count() - 1]);
        assert(
            greedy.bonus_token() ==
            target[greedy.accepted_count() - 1]);
    }
    const bool greedy_bonus_available =
        greedy.accepted_count() < verify_count;
    const bool greedy_bonus_committed =
        greedy_bonus_available &&
        commit_budget > greedy.accepted_count();
    assert(greedy.has_bonus() == greedy_bonus_available);
    assert(greedy.commits_bonus() == greedy_bonus_committed);
    assert_safe_tokens(
        greedy, draft, verify_count,
        target[greedy.accepted_count() - 1]);

    // Exercise the shared finalizer used by sampled/model-specific acceptance.
    const int accepted =
        static_cast<int>(bounded(static_cast<unsigned int>(verify_count))) + 1;
    const bool bonus_available = accepted < verify_count;
    const int32_t bonus = nondet_int();
    __ESBMC_assume(bonus >= 0);
    const ContractDecision precomputed = ContractDecision::precomputed(
        accepted, verify_count, bonus_available, bonus, commit_budget);
    assert(precomputed.valid());
    assert(precomputed.accepted_count() == accepted);
    assert(precomputed.commit_count() >= 0);
    assert(precomputed.commit_count() <= commit_budget);
    assert(precomputed.commit_count() <= accepted + 1);
    assert(precomputed.has_bonus() == bonus_available);
    assert(
        precomputed.commits_bonus() ==
        (bonus_available && commit_budget > accepted));
    assert_safe_tokens(precomputed, draft, verify_count, bonus);

    assert(!ContractDecision::precomputed(
        0, verify_count, true, bonus, commit_budget).valid());
    assert(!ContractDecision::precomputed(
        verify_count + 1, verify_count, false, bonus, commit_budget).valid());
    assert(!ContractDecision::precomputed(
        verify_count, verify_count, true, bonus, commit_budget).valid());
    if (verify_count > 1) {
        const ContractDecision sentinel_prefix =
            ContractDecision::precomputed(
                1, verify_count, false, -1, commit_budget);
        assert(sentinel_prefix.valid());
        assert(sentinel_prefix.accepted_count() == 1);
        assert(!sentinel_prefix.has_bonus());
        assert(!sentinel_prefix.commits_bonus());
        assert_safe_tokens(
            sentinel_prefix, draft, verify_count, bonus);
    }
    return 0;
}
