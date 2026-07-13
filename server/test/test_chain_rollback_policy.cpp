#include "chain_rollback_policy.h"

#include <cstdio>
#include <cstdlib>

using dflash::common::resolve_chain_rollback_policy;

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static void clear_policy_env() {
    unsetenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32");
    unsetenv("DFLASH_FAST_ROLLBACK_THRESHOLD");
    unsetenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG");
}

int main() {
    clear_policy_env();
    auto policy = resolve_chain_rollback_policy();
    CHECK(!policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 5);
    CHECK(!policy.diagnostics);

    // A threshold flag alone must not alter the established F16 policy.
    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "2", 1);
    policy = resolve_chain_rollback_policy();
    CHECK(!policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 5);

    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "1", 1);
    policy = resolve_chain_rollback_policy();
    CHECK(policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 2);

    // Boolean flags follow the project's non-empty, non-"0" convention.
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "true", 1);
    CHECK(resolve_chain_rollback_policy().checkpoint_f32);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "yes", 1);
    CHECK(resolve_chain_rollback_policy().checkpoint_f32);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "on", 1);
    CHECK(resolve_chain_rollback_policy().checkpoint_f32);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "0", 1);
    CHECK(!resolve_chain_rollback_policy().checkpoint_f32);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "1", 1);

    // Invalid values degrade safely to the default threshold.
    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "0", 1);
    CHECK(resolve_chain_rollback_policy().fast_rollback_threshold == 5);
    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "6", 1);
    CHECK(resolve_chain_rollback_policy().fast_rollback_threshold == 5);
    setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "garbage", 1);
    CHECK(resolve_chain_rollback_policy().fast_rollback_threshold == 5);

    setenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG", "1", 1);
    CHECK(resolve_chain_rollback_policy().diagnostics);

    clear_policy_env();
    if (failures != 0) return 1;
    std::printf("chain rollback policy tests passed\n");
    return 0;
}
