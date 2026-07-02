#include "../src/common/moe_expert_compute.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace dflash::common;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void expect_close(float got, float want, const char * msg) {
    if (std::fabs(got - want) > 1e-5f) {
        std::fprintf(stderr, "FAIL: %s got=%f want=%f\n", msg, got, want);
        std::exit(1);
    }
}

struct FakeMoeExpertCompute final : MoeExpertCompute {
    explicit FakeMoeExpertCompute(int target_id_) : target_id(target_id_) {}

    bool prefers_padded_batch() const override {
        return padded_preferred;
    }

    bool compute(const MoeExpertLayer &,
                 const float *,
                 const int32_t *,
                 const float *,
                 int,
                 int,
                 int,
                 float *) override {
        ++single_calls;
        return false;
    }

    bool compute_batch(const MoeExpertLayer &,
                       const float * input,
                       const int32_t * ids,
                       const float * weights,
                       int n_tokens,
                       int n_selected,
                       int n_embd,
                       int,
                       float * output) override {
        ++batch_calls;
        batch_tokens.push_back(n_tokens);
        batch_selected.push_back(n_selected);
        for (int t = 0; t < n_tokens; ++t) {
            float acc = 0.0f;
            for (int i = 0; i < n_selected; ++i) {
                const size_t idx = (size_t)t * (size_t)n_selected + (size_t)i;
                acc += weights[idx] * (float)(target_id * 100 + ids[idx]);
            }
            for (int e = 0; e < n_embd; ++e) {
                output[(size_t)t * (size_t)n_embd + (size_t)e] =
                    input[(size_t)t * (size_t)n_embd + (size_t)e] + acc + (float)e;
            }
        }
        return true;
    }

    int target_id = 0;
    int single_calls = 0;
    int batch_calls = 0;
    bool padded_preferred = false;
    std::vector<int> batch_tokens;
    std::vector<int> batch_selected;
};

static void test_multi_target_compute_batch_groups_by_target_and_count() {
    MoeMultiTargetExpertRuntime runtime;
    runtime.targets.resize(2);
    runtime.layer_routes.resize(1);
    runtime.layers.resize(1);
    runtime.layers[0].layer_idx = 0;
    runtime.layers[0].cold_global_by_local = {0, 1, 2, 3};
    runtime.enabled = true;

    auto target0_compute = std::make_unique<FakeMoeExpertCompute>(1);
    auto target1_compute = std::make_unique<FakeMoeExpertCompute>(2);
    FakeMoeExpertCompute * target0 = target0_compute.get();
    FakeMoeExpertCompute * target1 = target1_compute.get();

    runtime.targets[0].target_index = 0;
    runtime.targets[0].placement.total_hot = 2;
    runtime.targets[0].compute_active = true;
    runtime.targets[0].runtime.compute = std::move(target0_compute);
    runtime.targets[0].runtime.layers.resize(1);
    runtime.targets[0].runtime.layers[0].layer_idx = 0;
    runtime.targets[0].runtime.layers[0].cold_global_by_local = {0, 1};

    runtime.targets[1].target_index = 1;
    runtime.targets[1].placement.total_hot = 2;
    runtime.targets[1].compute_active = true;
    runtime.targets[1].runtime.compute = std::move(target1_compute);
    runtime.targets[1].runtime.layers.resize(1);
    runtime.targets[1].runtime.layers[0].layer_idx = 0;
    runtime.targets[1].runtime.layers[0].cold_global_by_local = {2, 3};

    auto & routes = runtime.layer_routes[0].route_by_union_local;
    routes.resize(4);
    routes[0] = {0, 0, 0};
    routes[1] = {0, 1, 1};
    routes[2] = {1, 0, 2};
    routes[3] = {1, 1, 3};

    runtime.compute = make_multi_target_moe_expert_compute(&runtime);

    const int n_tokens = 3;
    const int n_selected = 3;
    const int n_embd = 2;
    const float input[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const int32_t ids[] = {
        0, 2, 3,
        1, 0, 2,
        2, 3, 1,
    };
    const float weights[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    float output[n_tokens * n_embd] = {};

    expect(runtime.compute->compute_batch(runtime.layers[0], input, ids, weights,
                                          n_tokens, n_selected, n_embd,
                                          /*n_ff=*/4, output),
           "multi-target compute_batch");

    expect(target0->single_calls == 0, "target0 did not use single compute");
    expect(target1->single_calls == 0, "target1 did not use single compute");
    expect(target0->batch_calls == 2, "target0 grouped into two batch calls");
    expect(target1->batch_calls == 2, "target1 grouped into two batch calls");

    expect(target0->batch_selected[0] == 1, "target0 first selected count");
    expect(target0->batch_selected[1] == 2, "target0 second selected count");
    expect(target1->batch_selected[0] == 1, "target1 first selected count");
    expect(target1->batch_selected[1] == 2, "target1 second selected count");
    expect(target0->batch_tokens[0] + target0->batch_tokens[1] == n_tokens,
           "target0 covers all tokens");
    expect(target1->batch_tokens[0] + target1->batch_tokens[1] == n_tokens,
           "target1 covers all tokens");

    const float expected[] = {
        1105.0f, 1109.0f,
        2110.0f, 2114.0f,
        3927.0f, 3931.0f,
    };
    for (int i = 0; i < n_tokens * n_embd; ++i) {
        expect_close(output[i], expected[i], "accumulated output");
    }
}

static void test_multi_target_compute_batch_accumulates_after_direct_write() {
    MoeMultiTargetExpertRuntime runtime;
    runtime.targets.resize(2);
    runtime.layer_routes.resize(1);
    runtime.layers.resize(1);
    runtime.layers[0].layer_idx = 0;
    runtime.layers[0].cold_global_by_local = {0, 1};
    runtime.enabled = true;

    auto target0_compute = std::make_unique<FakeMoeExpertCompute>(1);
    auto target1_compute = std::make_unique<FakeMoeExpertCompute>(2);

    runtime.targets[0].target_index = 0;
    runtime.targets[0].placement.total_hot = 1;
    runtime.targets[0].compute_active = true;
    runtime.targets[0].runtime.compute = std::move(target0_compute);
    runtime.targets[0].runtime.layers.resize(1);
    runtime.targets[0].runtime.layers[0].layer_idx = 0;
    runtime.targets[0].runtime.layers[0].cold_global_by_local = {0};

    runtime.targets[1].target_index = 1;
    runtime.targets[1].placement.total_hot = 1;
    runtime.targets[1].compute_active = true;
    runtime.targets[1].runtime.compute = std::move(target1_compute);
    runtime.targets[1].runtime.layers.resize(1);
    runtime.targets[1].runtime.layers[0].layer_idx = 0;
    runtime.targets[1].runtime.layers[0].cold_global_by_local = {1};

    auto & routes = runtime.layer_routes[0].route_by_union_local;
    routes.resize(2);
    routes[0] = {0, 0, 0};
    routes[1] = {1, 0, 1};

    runtime.compute = make_multi_target_moe_expert_compute(&runtime);

    const int n_tokens = 2;
    const int n_selected = 2;
    const int n_embd = 2;
    const float input[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const int32_t ids[] = {
        0, 1,
        0, 1,
    };
    const float weights[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    float output[n_tokens * n_embd] = {};

    expect(runtime.compute->compute_batch(runtime.layers[0], input, ids, weights,
                                          n_tokens, n_selected, n_embd,
                                          /*n_ff=*/4, output),
           "multi-target direct-write accumulation");

    const float expected[] = {
        502.0f, 506.0f,
        1106.0f, 1110.0f,
    };
    for (int i = 0; i < n_tokens * n_embd; ++i) {
        expect_close(output[i], expected[i], "direct write then add output");
    }
}

static void test_multi_target_compute_batch_reuses_scratch_without_stale_groups() {
    MoeMultiTargetExpertRuntime runtime;
    runtime.targets.resize(2);
    runtime.layer_routes.resize(1);
    runtime.layers.resize(1);
    runtime.layers[0].layer_idx = 0;
    runtime.layers[0].cold_global_by_local = {0, 1, 2};
    runtime.enabled = true;

    auto target0_compute = std::make_unique<FakeMoeExpertCompute>(1);
    auto target1_compute = std::make_unique<FakeMoeExpertCompute>(2);
    FakeMoeExpertCompute * target0 = target0_compute.get();
    FakeMoeExpertCompute * target1 = target1_compute.get();

    runtime.targets[0].target_index = 0;
    runtime.targets[0].placement.total_hot = 2;
    runtime.targets[0].compute_active = true;
    runtime.targets[0].runtime.compute = std::move(target0_compute);
    runtime.targets[0].runtime.layers.resize(1);
    runtime.targets[0].runtime.layers[0].layer_idx = 0;
    runtime.targets[0].runtime.layers[0].cold_global_by_local = {0, 1};

    runtime.targets[1].target_index = 1;
    runtime.targets[1].placement.total_hot = 1;
    runtime.targets[1].compute_active = true;
    runtime.targets[1].runtime.compute = std::move(target1_compute);
    runtime.targets[1].runtime.layers.resize(1);
    runtime.targets[1].runtime.layers[0].layer_idx = 0;
    runtime.targets[1].runtime.layers[0].cold_global_by_local = {2};

    auto & routes = runtime.layer_routes[0].route_by_union_local;
    routes.resize(3);
    routes[0] = {0, 0, 0};
    routes[1] = {0, 1, 1};
    routes[2] = {1, 0, 2};

    runtime.compute = make_multi_target_moe_expert_compute(&runtime);

    const int n_embd = 1;
    const float input1[] = {1.0f, 2.0f};
    const int32_t ids1[] = {
        0, 1, 2,
        0, 1, 2,
    };
    const float weights1[] = {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
    };
    float output1[2] = {};
    expect(runtime.compute->compute_batch(runtime.layers[0], input1, ids1, weights1,
                                          /*n_tokens=*/2, /*n_selected=*/3, n_embd,
                                          /*n_ff=*/4, output1),
           "multi-target first scratch reuse batch");

    const float input2[] = {3.0f};
    const int32_t ids2[] = {2, 0};
    const float weights2[] = {4.0f, 5.0f};
    float output2[1] = {};
    expect(runtime.compute->compute_batch(runtime.layers[0], input2, ids2, weights2,
                                          /*n_tokens=*/1, /*n_selected=*/2, n_embd,
                                          /*n_ff=*/4, output2),
           "multi-target second scratch reuse batch");

    expect(target0->batch_calls == 2, "target0 called once per batch");
    expect(target1->batch_calls == 2, "target1 called once per batch");
    expect(target0->batch_selected.back() == 1, "target0 second batch selected");
    expect(target1->batch_selected.back() == 1, "target1 second batch selected");
    expect(target0->batch_tokens.back() == 1, "target0 second batch tokens");
    expect(target1->batch_tokens.back() == 1, "target1 second batch tokens");
    expect_close(output2[0], 1306.0f, "second batch output excludes stale buckets");
}

static void test_multi_target_prefers_grouped_batches_with_multiple_active_targets() {
    MoeMultiTargetExpertRuntime runtime;
    runtime.targets.resize(2);
    runtime.enabled = true;

    auto target0_compute = std::make_unique<FakeMoeExpertCompute>(1);
    auto target1_compute = std::make_unique<FakeMoeExpertCompute>(2);
    target1_compute->padded_preferred = true;

    runtime.targets[0].compute_active = true;
    runtime.targets[0].runtime.compute = std::move(target0_compute);
    runtime.targets[1].compute_active = true;
    runtime.targets[1].runtime.compute = std::move(target1_compute);
    runtime.compute = make_multi_target_moe_expert_compute(&runtime);

    expect(!runtime.compute->prefers_padded_batch(),
           "multi-target disables padded batch preference across multiple active targets");
}

static void test_multi_target_inactive_primary_does_not_require_compute() {
    MoeMultiTargetExpertRuntime runtime;
    runtime.targets.resize(2);
    runtime.enabled = true;

    runtime.targets[0].target_index = 0;
    runtime.targets[0].placement.total_hot = 4;
    runtime.targets[0].compute_active = false;

    auto target1_compute = std::make_unique<FakeMoeExpertCompute>(2);
    target1_compute->padded_preferred = true;
    runtime.targets[1].target_index = 1;
    runtime.targets[1].placement.total_hot = 2;
    runtime.targets[1].compute_active = true;
    runtime.targets[1].runtime.compute = std::move(target1_compute);

    runtime.compute = make_multi_target_moe_expert_compute(&runtime);
    expect(runtime.compute->healthy(),
           "inactive primary target does not make runtime unhealthy");
    expect(runtime.compute->prefers_padded_batch(),
           "inactive primary target is skipped for preferences");
}

int main() {
    test_multi_target_compute_batch_groups_by_target_and_count();
    test_multi_target_compute_batch_accumulates_after_direct_write();
    test_multi_target_compute_batch_reuses_scratch_without_stale_groups();
    test_multi_target_prefers_grouped_batches_with_multiple_active_targets();
    test_multi_target_inactive_primary_does_not_require_compute();
    std::printf("OK\n");
    return 0;
}
