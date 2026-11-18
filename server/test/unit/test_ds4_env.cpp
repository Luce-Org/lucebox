#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "../../deps/llama.cpp/ggml/src/ggml-cuda/ds4-env.cuh"

namespace {
struct Ds4EnvFixture {};
}

TEST_CASE(Ds4EnvFixture, split_kv_aliases_values_and_kill_switch_precedence) {
    struct Row {
        const char * force;
        const char * alias;
        const char * kill;
        const char * kill_alias;
        bool device_default;
        bool want;
    };
    const Row rows[] = {
        {nullptr, nullptr, nullptr, nullptr, false, false},  // nothing set: the device default
        {nullptr, nullptr, nullptr, nullptr, true, true},
        {"1", nullptr, nullptr, nullptr, false, true},       // either force name turns it on
        {nullptr, "1", nullptr, nullptr, false, true},
        {"0", "", nullptr, nullptr, false, false},           // "0" and "" do not force
        {"1", nullptr, "1", nullptr, true, false},           // either kill switch beats force and default
        {nullptr, "1", nullptr, "1", true, false},
        {nullptr, nullptr, "0", "", true, true},             // "0" and "" do not kill
    };
    for (const Row & r : rows) {
        luce_test::ScopedEnvVar force("GGML_CUDA_MLA_SPLIT_KV", r.force);
        luce_test::ScopedEnvVar alias("GGML_DS4_FA_SPLIT_KV", r.alias);
        luce_test::ScopedEnvVar kill("GGML_CUDA_MLA_NO_SPLIT_KV", r.kill);
        luce_test::ScopedEnvVar kill_alias("GGML_DS4_FA_NO_SPLIT_KV", r.kill_alias);
        CHECK(ds4_mla_split_kv_enabled(r.device_default) == r.want);
    }
}
