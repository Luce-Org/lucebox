#include "chain_rollback_policy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using dflash::common::resolve_chain_rollback_policy;
using dflash::common::RollbackDiag;

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

    // Shared diagnostics accumulator used by both spec-decode loops.
    {
        RollbackDiag diag;
        diag.record_accept(1);
        diag.record_accept(3);
        diag.record_accept(7);
        diag.record_accept(40);          // clamps into the 16+ bucket
        diag.record_fast_rollback(3);    // below the F16 breakeven
        diag.record_fast_rollback(7);    // at/above the breakeven
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
            if (!f) return text;
            const auto policy = resolve_chain_rollback_policy();
            d.print(policy, f);
            long n = std::ftell(f);
            std::rewind(f);
            text.resize(n > 0 ? (size_t)n : 0);
            if (n > 0 && std::fread(&text[0], 1, (size_t)n, f) != (size_t)n) text.clear();
            std::fclose(f);
            return text;
        };

        // Diagnostics enabled above: the shared print emits the exact line
        // format the validation harness greps for.
        setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "1", 1);
        setenv("DFLASH_FAST_ROLLBACK_THRESHOLD", "1", 1);
        const std::string line = print_to_string(diag);
        CHECK(line ==
            "[chain-rollback-policy] checkpoint=F32 threshold=1 fast_low=1 fast_high=1 "
            "legacy_replay=1 failed_fallback=1 "
            "accept_hist=1:1,2:0,3:1,4:0,5:0,6:0,7:1,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16+:1\n");

        // A failed fopen/tmpfile may provide a null stream. Diagnostics must
        // remain a no-op rather than forwarding nullptr to std::fprintf.
        diag.print(resolve_chain_rollback_policy(), nullptr);

        // print() is a no-op when diagnostics are disabled.
        unsetenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG");
        CHECK(print_to_string(diag).empty());
        setenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG", "1", 1);
    }

    clear_policy_env();
    if (failures != 0) return 1;
    std::printf("chain rollback policy tests passed\n");
    return 0;
}
