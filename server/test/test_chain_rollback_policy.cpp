#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "chain_rollback_policy.h"
#include "internal.h"

#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using dflash::common::resolve_chain_rollback_policy;
using dflash::common::RollbackDiag;
using dflash::common::split_chain_fast_rollback_enabled;

namespace {
struct ChainRollbackPolicyFixture {};
}

static void clear_policy_env() {
    unsetenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32");
    unsetenv("DFLASH_FAST_ROLLBACK_THRESHOLD");
    unsetenv("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG");
}

TEST_CASE(ChainRollbackPolicyFixture, policy_defaults_and_env_parsing) {
    const luce_test::ScopedEnvVar checkpoint("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", nullptr);
    const luce_test::ScopedEnvVar threshold("DFLASH_FAST_ROLLBACK_THRESHOLD", nullptr);
    const luce_test::ScopedEnvVar diagnostics("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG", nullptr);
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

    // TP restores rank-local recurrent state directly and always rolls back
    // from the first accepted token, independent of checkpoint precision.
    policy = resolve_chain_rollback_policy(true);
    CHECK(policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 1);
    unsetenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32");
    policy = resolve_chain_rollback_policy(true);
    CHECK(!policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 1);
    policy = resolve_chain_rollback_policy(false, true);
    CHECK(!policy.checkpoint_f32);
    CHECK(policy.fast_rollback_threshold == 1);
    setenv("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", "1", 1);

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

TEST_CASE(ChainRollbackPolicyFixture, split_fast_rollback_is_explicitly_opt_in) {
    const luce_test::ScopedEnvVar split_fast("DFLASH_SPLIT_FAST_ROLLBACK", nullptr);
    unsetenv("DFLASH_SPLIT_FAST_ROLLBACK");
    CHECK(!split_chain_fast_rollback_enabled());

    setenv("DFLASH_SPLIT_FAST_ROLLBACK", "1", 1);
    CHECK(split_chain_fast_rollback_enabled());
    setenv("DFLASH_SPLIT_FAST_ROLLBACK", "true", 1);
    CHECK(split_chain_fast_rollback_enabled());
    setenv("DFLASH_SPLIT_FAST_ROLLBACK", "0", 1);
    CHECK(!split_chain_fast_rollback_enabled());
    setenv("DFLASH_SPLIT_FAST_ROLLBACK", "", 1);
    CHECK(!split_chain_fast_rollback_enabled());
    unsetenv("DFLASH_SPLIT_FAST_ROLLBACK");
}

TEST_CASE(ChainRollbackPolicyFixture, split_checkpoint_dtype_is_gated_at_allocation) {
    const luce_test::ScopedEnvVar kv_f16("DFLASH27B_KV_F16", nullptr);
    const luce_test::ScopedEnvVar kv_q4("DFLASH27B_KV_Q4", nullptr);
    const luce_test::ScopedEnvVar kv_tq3("DFLASH27B_KV_TQ3", nullptr);
    const luce_test::ScopedEnvVar kv_k("DFLASH27B_KV_K", nullptr);
    const luce_test::ScopedEnvVar kv_v("DFLASH27B_KV_V", nullptr);

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) return;

    dflash::common::TargetWeights weights;
    weights.n_layer = 2;
    weights.full_attention_interval = 2;
    weights.n_embd_head_k = 32;
    weights.n_embd_head_v = 32;
    weights.n_head = 1;
    weights.n_head_kv = 1;
    weights.n_embd = 32;
    weights.n_capture_layers = 0;
    weights.ssm_d_inner = 32;
    weights.ssm_d_state = 1;
    weights.ssm_dt_rank = 1;
    weights.ssm_n_group = 1;
    weights.ssm_d_conv = 2;

    auto check_type = [&](bool f32_checkpoints, ggml_type expected) {
        dflash::common::TargetCache cache;
        const bool ok = dflash::common::create_target_cache_partial(
            weights, /*max_ctx=*/1, /*max_verify_tokens=*/2, backend, cache,
            /*prefill_only=*/false, /*layer_begin=*/0, /*layer_end=*/2,
            /*allocate_target_feat=*/false, /*ctx_alloc=*/0,
            /*f32_ssm_intermediates=*/f32_checkpoints);
        CHECK(ok);
        if (ok) {
            CHECK(cache.ssm_intermediate.size() == 1);
            CHECK(cache.ssm_intermediate[0] != nullptr);
            if (cache.ssm_intermediate[0]) {
                CHECK(cache.ssm_intermediate[0]->type == expected);
            }
        }
        dflash::common::free_target_cache(cache);
    };

    check_type(false, GGML_TYPE_Q8_0);
    check_type(true, GGML_TYPE_F32);
    ggml_backend_free(backend);
}

TEST_CASE(ChainRollbackPolicyFixture, diagnostics_accumulator_and_print_contract) {
    const luce_test::ScopedEnvVar checkpoint("DFLASH_SINGLE_CHAIN_CHECKPOINT_F32", nullptr);
    const luce_test::ScopedEnvVar threshold("DFLASH_FAST_ROLLBACK_THRESHOLD", nullptr);
    const luce_test::ScopedEnvVar diagnostics("DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG", nullptr);
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
