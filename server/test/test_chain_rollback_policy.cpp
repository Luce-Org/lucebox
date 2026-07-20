#include "CppUnitTestFramework.hpp"
#include "chain_rollback_policy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using dflash::common::resolve_chain_rollback_policy;
using dflash::common::RollbackDiag;

namespace {
struct ChainRollbackPolicyFixture {};
}

static void clear_policy_env() {
    unsetenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32");
    unsetenv("DFLASH_FAST_ROLLBACK_THRESHOLD");
    unsetenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG");
}

TEST_CASE(ChainRollbackPolicyFixture, policy_defaults_and_env_parsing) {
    clear_policy_env();
    auto policy = resolve_chain_rollback_policy();
    CHECK(!policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 5);
    CHECK(!policy.diagnostics);

    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "2", 1);
    policy = resolve_chain_rollback_policy();
    CHECK(!policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 5);

    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "1", 1);
    policy = resolve_chain_rollback_policy();
    CHECK(policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 2);

    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "true", 1);
    CHECK(resolve_chain_rollback_policy().checkpoint_f32);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "yes", 1);
    CHECK(resolve_chain_rollback_policy().checkpoint_f32);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "on", 1);
    CHECK(resolve_chain_rollback_policy().checkpoint_f32);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "0", 1);
    CHECK(!resolve_chain_rollback_policy().checkpoint_f32);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "1", 1);

    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "0", 1);
    CHECK(resolve_chain_rollback_policy().fast_rollback_threshold == 5);
    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "6", 1);
    CHECK(resolve_chain_rollback_policy().fast_rollback_threshold == 5);
    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "garbage", 1);
    CHECK(resolve_chain_rollback_policy().fast_rollback_threshold == 5);

    setenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG", "1", 1);
    CHECK(resolve_chain_rollback_policy().diagnostics);
    clear_policy_env();
}

TEST_CASE(ChainRollbackPolicyFixture, diagnostics_accumulator_and_print_contract) {
    clear_policy_env();
    setenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG", "1", 1);

    RollbackDiag diag;
    diag.record_accept(1);
    diag.record_accept(3);
    diag.record_accept(7);
    diag.record_accept(40);
    diag.record_fast_rollback(3);
    diag.record_fast_rollback(7);
    diag.record_legacy_replay();
    diag.record_failed_fallback();
    CHECK(diag.accept_hist[1] == 1);
    CHECK(diag.accept_hist[3] == 1);
    CHECK(diag.accept_hist[7] == 1);
    CHECK(diag.accept_hist[16] == 1);
    CHECK(diag.fast_low == 1);
    CHECK(diag.fast_high == 1);
    CHECK(diag.legacy_replay == 1);
    CHECK(diag.failed_fallback == 1);

    auto print_to_string = [](const RollbackDiag & d) {
        std::string text;
        std::FILE * f = tmpfile();
        if (!f) {
            return text;
        }
        const auto policy = resolve_chain_rollback_policy();
        d.print(policy, f);
        long n = std::ftell(f);
        std::rewind(f);
        text.resize(n > 0 ? (size_t) n : 0);
        if (n > 0 && std::fread(&text[0], 1, (size_t) n, f) != (size_t) n) {
            text.clear();
        }
        std::fclose(f);
        return text;
    };

    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "1", 1);
    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "1", 1);
    const std::string line = print_to_string(diag);
    CHECK(line ==
        "[chain-rollback-policy] checkpoint=F32 threshold=1 fast_low=1 fast_high=1 "
        "legacy_replay=1 failed_fallback=1 "
        "accept_hist=1:1,2:0,3:1,4:0,5:0,6:0,7:1,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16+:1\n");

    diag.print(resolve_chain_rollback_policy(), nullptr);

    unsetenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG");
    CHECK(print_to_string(diag).empty());
    clear_policy_env();
}
