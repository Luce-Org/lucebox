#include "CppUnitTestFramework.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "common/platform_env.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#include <hip/hip_runtime_api.h>
#endif

namespace {
struct MixedMmqPolicy : CppUnitTestFramework::CommonFixture {
    using CommonFixture::CommonFixture;
#if defined(GGML_USE_HIP)
    void check_mix_graph_isolation(ggml_type type);
#endif
};

struct SavedMixEnvironment {
    static constexpr const char * name = "DFLASH_DS4_MIX_MMQ_PREFILL";
    const bool present = std::getenv(name) != nullptr;
    const std::string value = present ? std::getenv(name) : "";
    ~SavedMixEnvironment() {
        if (present) dflash::common::set_environment_variable(name, value.c_str(), true);
        else dflash::common::unset_environment_variable(name);
    }
};
}

TEST_CASE(MixedMmqPolicy, operation_metadata_is_independent) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({1u << 20, nullptr, true}), ggml_free);
    CHECK(ctx != nullptr);
    auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 64);
    auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 16);
    auto * approximate = ggml_mul_mat(ctx.get(), a, b);
    auto * exact = ggml_mul_mat(ctx.get(), a, b);
    ggml_mul_mat_set_prec(approximate, GGML_PREC_F32);
    ggml_mul_mat_set_mixed_mmq(approximate, GGML_MIXED_MMQ_ENABLED);
    CHECK(ggml_mul_mat_get_mixed_mmq(exact) == GGML_MIXED_MMQ_DEFAULT);
    CHECK(ggml_mul_mat_get_mixed_mmq(approximate) == GGML_MIXED_MMQ_ENABLED);
    CHECK(approximate->op_params[0] == GGML_PREC_F32);

    auto * physical = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 16, 2);
    auto * grouped = ggml_mul_mat_grouped_src(ctx.get(), a, physical);
    ggml_mul_mat_set_mixed_mmq(grouped, GGML_MIXED_MMQ_DISABLED);
    CHECK(ggml_mul_mat_grouped_src_groups(grouped) == 2);
    CHECK(ggml_mul_mat_get_mixed_mmq(grouped) == GGML_MIXED_MMQ_DISABLED);
    // Scheduler copies preserve op_params, not any ambient backend setting.
    ggml_tensor copied = *approximate;
    CHECK(ggml_mul_mat_get_mixed_mmq(&copied) == GGML_MIXED_MMQ_ENABLED);
}

#if defined(GGML_USE_HIP)
namespace {
struct MixGpuRun {
    ggml_backend_t fast_backend = ggml_backend_cuda_init(0);
    ggml_backend_t exact_backend = ggml_backend_cuda_init(0);
    ggml_context * ctx = ggml_init({4u << 20, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    void * registered = nullptr;
    ggml_type type;
    explicit MixGpuRun(ggml_type type) : type(type) {}
    ~MixGpuRun() {
        if (fast_backend) ggml_backend_synchronize(fast_backend);
        if (exact_backend) ggml_backend_synchronize(exact_backend);
        if (registered) {
            if (type == GGML_TYPE_Q2_1_ROCMFP2_MIX) ggml_cuda_rocmfp2_mix_unregister(registered);
            else ggml_cuda_rocmfp3_mix_unregister(registered);
        }
        if (fast_backend) ggml_backend_free(fast_backend);
        if (exact_backend) ggml_backend_free(exact_backend);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

struct GraphCaptureOverride {
    bool previous = ggml_backend_cuda_set_graphs_disabled_override(true);
    ~GraphCaptureOverride() { ggml_backend_cuda_set_graphs_disabled_override(previous); }
};

void MixedMmqPolicy::check_mix_graph_isolation(ggml_type type) {
    MixGpuRun run(type);
    CHECK(run.fast_backend && run.exact_backend && run.ctx);
    constexpr int k = 256, rows = 64, experts = 4, used = 2, tokens = 16;
    auto * weights = ggml_new_tensor_3d(run.ctx, type, k, rows, experts);
    auto * input = ggml_new_tensor_3d(run.ctx, GGML_TYPE_F32, k, 1, tokens);
    auto * ids = ggml_new_tensor_2d(run.ctx, GGML_TYPE_I32, used, tokens);
    ggml_set_input(input);
    ggml_set_input(ids);
    auto * fast = ggml_mul_mat_id(run.ctx, weights, input, ids);
    auto * exact = ggml_mul_mat_id(run.ctx, weights, input, ids);
    ggml_mul_mat_set_mixed_mmq(fast, GGML_MIXED_MMQ_ENABLED);
    auto * fast_graph = ggml_new_graph_custom(run.ctx, 16, false);
    auto * exact_graph = ggml_new_graph_custom(run.ctx, 16, false);
    ggml_build_forward_expand(fast_graph, fast);
    ggml_build_forward_expand(exact_graph, exact);
    run.buffer = ggml_backend_alloc_ctx_tensors(run.ctx, run.fast_backend);
    CHECK(run.buffer != nullptr);

    const size_t block_bytes = ggml_type_size(type);
    std::vector<uint8_t> bytes(ggml_nbytes(weights));
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i * 37 + i / 17);
    for (size_t i = 0; i < bytes.size(); i += block_bytes) {
        bytes[i + block_bytes - 2] = 0x30;
        bytes[i + block_bytes - 1] = 0xb0;
    }
    const int levels = type == GGML_TYPE_Q2_1_ROCMFP2_MIX ? 4 : 8;
    std::vector<uint16_t> books(experts * 2 * levels);
    for (size_t i = 0; i < books.size(); ++i) {
        const float value = float(int(i % levels) - levels / 2) * 0.25f;
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        books[i] = uint16_t(bits >> 16);
    }
    std::vector<uint8_t> modes(experts, 1);
    ggml_backend_tensor_set(weights, bytes.data(), 0, bytes.size());
    const bool registered = type == GGML_TYPE_Q2_1_ROCMFP2_MIX
        ? ggml_cuda_rocmfp2_mix_register_host(weights->data, weights->nb[2], experts, rows, k, books.data(), modes.data())
        : ggml_cuda_rocmfp3_mix_register_host(weights->data, weights->nb[2], experts, rows, k, books.data(), modes.data());
    CHECK(registered);
    run.registered = weights->data;
    std::vector<float> x(k * tokens);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 23) - 11) / 32.0f;
    std::vector<int32_t> routes(used * tokens);
    for (size_t i = 0; i < routes.size(); ++i) routes[i] = int32_t(i % experts);
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));

    const auto compute = [&](bool use_fast, bool expect_mmq) {
        const size_t before = ggml_backend_cuda_get_mmq_launch_count();
        CHECK(ggml_backend_graph_compute(
            use_fast ? run.fast_backend : run.exact_backend,
            use_fast ? fast_graph : exact_graph) == GGML_STATUS_SUCCESS);
        const size_t delta = ggml_backend_cuda_get_mmq_launch_count() - before;
        CHECK(expect_mmq ? delta > 0 : delta == 0);
        std::vector<float> output(rows * used * tokens);
        ggml_backend_tensor_get(use_fast ? fast : exact, output.data(), 0, output.size() * sizeof(float));
        return output;
    };
    GraphCaptureOverride capture;
    const auto reference = compute(false, false);
    for (int round = 0; round < 3; ++round) {
        const auto candidate = compute(true, true);
        CHECK(compute(false, false) == reference);
        double error = 0.0, power = 0.0;
        for (size_t i = 0; i < candidate.size(); ++i) {
            CHECK(std::isfinite(candidate[i]));
            const double diff = candidate[i] - reference[i];
            error += diff * diff;
            power += double(reference[i]) * reference[i];
        }
        CHECK(error / std::max(power, 1e-30) < 5e-4);
        CHECK(std::getenv(SavedMixEnvironment::name) == nullptr);
    }

    // DEFAULT keeps the explicit legacy override, without latching the
    // first model's value in a function-static dispatch decision.
    REQUIRE(dflash::common::set_environment_variable(
        SavedMixEnvironment::name, "1", true) == 0);
    compute(false, true);
    REQUIRE(dflash::common::set_environment_variable(
        SavedMixEnvironment::name, "0", true) == 0);
    CHECK(compute(false, false) == reference);
    compute(true, true); // model-local selection is already resolved
    REQUIRE(dflash::common::unset_environment_variable(SavedMixEnvironment::name) == 0);

    // Capture/replay must also honor graph-local policy and detect changes.
    ggml_backend_cuda_set_graphs_disabled_override(false);
    for (int round = 0; round < 3; ++round) {
        CHECK(ggml_backend_graph_compute(run.fast_backend, fast_graph) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_graph_compute(run.exact_backend, exact_graph) == GGML_STATUS_SUCCESS);
    }
    ggml_mul_mat_set_mixed_mmq(fast, GGML_MIXED_MMQ_DISABLED);
    CHECK(ggml_backend_graph_compute(run.fast_backend, fast_graph) == GGML_STATUS_SUCCESS);
    std::vector<float> disabled(reference.size());
    ggml_backend_tensor_get(fast, disabled.data(), 0, disabled.size() * sizeof(float));
    CHECK(disabled == reference);
}
}
#endif

TEST_CASE(MixedMmqPolicy, interleaved_backends_and_replay_preserve_policy) {
#if defined(GGML_USE_HIP)
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess ||
        (std::strncmp(props.gcnArchName, "gfx1151", 7) != 0 &&
         std::strncmp(props.gcnArchName, "gfx12", 5) != 0)) {
        SKIP("requires gfx1151/gfx12 mixed MMQ");
    }
    SavedMixEnvironment saved;
    CHECK(dflash::common::unset_environment_variable(saved.name) == 0);
    for (auto type : {GGML_TYPE_Q2_1_ROCMFP2_MIX, GGML_TYPE_Q3_1_ROCMFP3_MIX}) {
        check_mix_graph_isolation(type);
    }
#else
    SKIP("HIP mixed-MMQ dispatch qualification");
#endif
}

TEST_CASE(MixedMmqPolicy, dense_glu_fusion_preserves_matvec_dispatch) {
#if defined(GGML_USE_HIP)
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess ||
        (std::strncmp(props.gcnArchName, "gfx1151", 7) != 0 &&
         std::strncmp(props.gcnArchName, "gfx12", 5) != 0)) {
        SKIP("requires RDNA mixed-precision kernels");
    }
    struct Overrides {
        bool graphs = ggml_backend_cuda_set_graphs_disabled_override(true);
        int ceiling = ggml_backend_cuda_set_mmvq_max_ncols_override(0);
        ~Overrides() {
            ggml_backend_cuda_set_mmvq_max_ncols_override(ceiling);
            ggml_backend_cuda_set_graphs_disabled_override(graphs);
        }
    } saved;
    // Distinct graph outputs prevent fusion in the reference without changing
    // matmul precision, kernel policy or the activation tensor's layout.
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_0_ROCMFP4_FAST}) {
        for (int ceiling : {0, 4}) {
            ggml_backend_cuda_set_mmvq_max_ncols_override(ceiling);
            for (int width : {2, 3, 4, 8}) {
                auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
                    ggml_init({4u << 20, nullptr, true}), ggml_free);
                auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
                    ggml_backend_cuda_init(0), ggml_backend_free);
                REQUIRE(ctx && backend);
                constexpr int k = 256, rows = 64;
                auto * wg = ggml_new_tensor_2d(ctx.get(), type, k, rows);
                auto * wu = ggml_new_tensor_2d(ctx.get(), type, k, rows);
                auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, width);
                ggml_set_input(x);
                ggml_tensor * outputs[2];
                ggml_cgraph * graphs[2];
                for (int ref = 0; ref < 2; ++ref) {
                    auto * gate = ggml_mul_mat(ctx.get(), wg, x);
                    auto * up = ggml_mul_mat(ctx.get(), wu, x);
                    if (ref) {
                        ggml_set_output(gate);
                        ggml_set_output(up);
                    }
                    outputs[ref] = ggml_swiglu_ds4_split(ctx.get(), gate, up, 7.0f);
                    ggml_set_output(outputs[ref]);
                    graphs[ref] = ggml_new_graph_custom(ctx.get(), 16, false);
                    ggml_build_forward_expand(graphs[ref], outputs[ref]);
                }
                auto buffer = std::unique_ptr<ggml_backend_buffer,
                    decltype(&ggml_backend_buffer_free)>(
                        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()),
                        ggml_backend_buffer_free);
                REQUIRE(buffer != nullptr);
                std::vector<float> weights(k * rows), input(k * width);
                std::vector<uint8_t> quantized(ggml_nbytes(wg));
                int half = 0;
                for (auto * w : {wg, wu}) {
                    ++half;
                    for (size_t i = 0; i < weights.size(); ++i) {
                        weights[i] = std::sin(float(i * 13 + half) * 0.071f) * 0.17f;
                    }
                    ggml_get_type_traits(type)->from_float_ref(
                        weights.data(), quantized.data(), weights.size());
                    ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
                }
                for (int replay = 0; replay < 4; ++replay) {
                    ggml_backend_cuda_set_graphs_disabled_override(replay == 0);
                    for (size_t i = 0; i < input.size(); ++i) {
                        input[i] = std::cos(float(i * 7 + replay * 11) * 0.013f);
                    }
                    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                    std::vector<float> result[2];
                    size_t launches[2]{};
                    for (int ref = 0; ref < 2; ++ref) {
                        const size_t before = ggml_backend_cuda_get_mmq_launch_count();
                        REQUIRE(ggml_backend_graph_compute(backend.get(), graphs[ref]) == GGML_STATUS_SUCCESS);
                        launches[ref] = ggml_backend_cuda_get_mmq_launch_count() - before;
                        result[ref].resize(rows * width);
                        ggml_backend_tensor_get(outputs[ref], result[ref].data(), 0,
                            result[ref].size() * sizeof(float));
                    }
                    if (replay == 0) {
                        CHECK((launches[0] == 0) == (launches[1] == 0));
                    }
                    for (float v : result[0]) CHECK(std::isfinite(v));
                    CHECK(std::memcmp(result[0].data(), result[1].data(),
                        result[0].size() * sizeof(float)) == 0);
                }
            }
        }
    }
#else
    SKIP("HIP dense-GLU dispatch qualification");
#endif
}
