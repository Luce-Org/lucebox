#include "qwen35/qwen35_tensor_parallel.h"

#include "ggml.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

#define CHECK(condition) do {                                               \
    if (!(condition)) {                                                     \
        std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n",                 \
                     #condition, __FILE__, __LINE__);                       \
        std::abort();                                                       \
    }                                                                       \
} while (0)

static void expect_split(const TargetWeights & weights,
                         ggml_tensor * tensor,
                         const char * name,
                         ggml_backend_meta_split_axis axis,
                         int64_t per_device,
                         uint32_t repeat) {
    ggml_set_name(tensor, name);
    const auto state = qwen35_tensor_parallel_split_state(tensor, weights, 2);
    CHECK(state.axis == axis);
    CHECK(state.n_segments == 1);
    CHECK(state.ne[0] == per_device);
    CHECK(state.ne[1] == per_device);
    CHECK(state.nr[0] == repeat);
    CHECK((state.ne[0] + state.ne[1]) * state.nr[0] == tensor->ne[axis]);
}

int main() {
    ggml_init_params params{};
    params.mem_size = 32 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx);

    TargetWeights weights;

    expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 10240),
        "blk.0.attn_qkv.weight", GGML_BACKEND_SPLIT_AXIS_1, 1024, 5);
    expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10240),
        "blk.0.ssm_conv1d.weight", GGML_BACKEND_SPLIT_AXIS_1, 1024, 5);
    expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6144, 5120),
        "blk.0.ssm_out.weight", GGML_BACKEND_SPLIT_AXIS_0, 1024, 3);
    expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 12288),
        "blk.3.attn_q.weight", GGML_BACKEND_SPLIT_AXIS_1, 6144, 1);
    expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6144, 5120),
        "blk.3.attn_output.weight", GGML_BACKEND_SPLIT_AXIS_0, 3072, 1);
    expect_split(weights,
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 4096, 4),
        "cache_k_3", GGML_BACKEND_SPLIT_AXIS_2, 2, 1);
    expect_split(weights,
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 128, 48),
        "ssm_state_0", GGML_BACKEND_SPLIT_AXIS_2, 8, 3);
    expect_split(weights,
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 10240),
        "conv_state_0", GGML_BACKEND_SPLIT_AXIS_1, 1024, 5);

    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
    ggml_set_name(norm, "blk.0.attn_norm.weight");
    const auto mirrored = qwen35_tensor_parallel_split_state(norm, weights, 2);
    CHECK(mirrored.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 248320);
    ggml_set_name(output, "output.weight");
    const auto output_state =
        qwen35_tensor_parallel_split_state(output, weights, 2);
    CHECK(output_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    DevicePlacement placement;
    placement.layer_split_gpus = {1, 2};
    placement.layer_split_backends = {PlacementBackend::Cuda, PlacementBackend::Cuda};
    CHECK(placement.is_layer_split());
    placement.split_mode = TargetSplitMode::Tensor;
    CHECK(placement.is_tensor_parallel());
    CHECK(validate_device_placement(placement, 3).empty());
    placement.layer_split_weights = {1.0, 1.0};
    CHECK(!validate_device_placement(placement, 3).empty());

    ggml_free(ctx);
    std::puts("qwen35 tensor-parallel split tests passed");
    return 0;
}
