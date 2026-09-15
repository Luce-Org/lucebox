#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "../deps/llama.cpp/ggml/src/ggml-cuda/ds4-env.cuh"

namespace {
struct Ds4EnvFixture {};
}

TEST_CASE(Ds4EnvFixture, split_kv_aliases_values_and_kill_switch_precedence) {
    const char * values[] = {nullptr, "", "0", "1"};
    for (bool device_default : {false, true}) {
        for (int force = 0; force < 4; ++force) {
            luce_test::ScopedEnvVar primary("GGML_CUDA_MLA_SPLIT_KV", values[force]);
            for (int alias = 0; alias < 4; ++alias) {
                luce_test::ScopedEnvVar legacy("GGML_DS4_FA_SPLIT_KV", values[alias]);
                for (int disable = 0; disable < 4; ++disable) {
                    luce_test::ScopedEnvVar kill("GGML_CUDA_MLA_NO_SPLIT_KV", values[disable]);
                    for (int disable_alias = 0; disable_alias < 4; ++disable_alias) {
                        luce_test::ScopedEnvVar legacy_kill(
                            "GGML_DS4_FA_NO_SPLIT_KV", values[disable_alias]);
                        const bool expected = disable != 3 && disable_alias != 3 &&
                            (device_default || force == 3 || alias == 3);
                        CHECK(ds4_mla_split_kv_enabled(device_default) == expected);
                    }
                }
            }
        }
    }
}
