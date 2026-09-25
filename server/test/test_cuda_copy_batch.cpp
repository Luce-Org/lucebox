#include "CppUnitTestFramework.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {
struct CudaCopyBatchFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};
}  // namespace

// ggml_backend_cuda_copy_batch_async: more descriptors than one launch holds,
// aligned and unaligned ends, sizes from 1 byte to several KB, and an empty
// descriptor. Every destination byte must hold its source byte and every
// byte between destinations must stay untouched.
TEST_CASE(CudaCopyBatchFixture, descriptor_batch) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);

    constexpr int64_t kBytes = 1 << 20;
    ggml_init_params ip{};
    ip.mem_size = 2 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);
    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, kBytes);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, kBytes);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    REQUIRE(buf != nullptr);

    std::vector<uint8_t> src_host(kBytes);
    for (int64_t i = 0; i < kBytes; ++i) {
        src_host[i] = (uint8_t) ((i * 131 + 7) & 0xff);
    }
    std::vector<uint8_t> expected(kBytes, 0xEE);
    ggml_backend_tensor_set(src, src_host.data(), 0, kBytes);
    ggml_backend_tensor_set(dst, expected.data(), 0, kBytes);

    std::vector<ggml_cuda_copy_desc> descs;
    int64_t src_off = 3;
    int64_t dst_off = 5;
    for (int i = 0; i < 101; ++i) {
        const int64_t n = i == 50 ? 0 : 1 + (int64_t) (i * 97) % 9000;
        // Every fourth copy starts 16-byte aligned on both ends.
        if (i % 4 == 0) {
            src_off = (src_off + 15) & ~int64_t(15);
            dst_off = (dst_off + 15) & ~int64_t(15);
        }
        REQUIRE(src_off + n <= kBytes);
        REQUIRE(dst_off + n <= kBytes);
        descs.push_back({(const uint8_t *) src->data + src_off,
                         (uint8_t *) dst->data + dst_off, (size_t) n});
        for (int64_t j = 0; j < n; ++j) {
            expected[dst_off + j] = src_host[src_off + j];
        }
        src_off += n + 1;
        dst_off += n + 3;   // gaps between destinations must stay 0xEE
    }

    ggml_backend_cuda_copy_batch_async(gpu, descs.data(), (int) descs.size());
    ggml_backend_synchronize(gpu);

    std::vector<uint8_t> got(kBytes);
    ggml_backend_tensor_get(dst, got.data(), 0, kBytes);
    int64_t first_bad = -1;
    for (int64_t i = 0; i < kBytes && first_bad < 0; ++i) {
        if (got[i] != expected[i]) first_bad = i;
    }
    if (first_bad >= 0) {
        std::fprintf(stderr, "[cuda-copy-batch] byte %lld: got %u want %u\n",
                     (long long) first_bad, got[first_bad], expected[first_bad]);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    REQUIRE(first_bad == -1);
}

namespace {
constexpr int64_t kGraphN = 4099;

std::vector<float> graph_pattern(float base, float step) {
    std::vector<float> v(kGraphN);
    for (int64_t i = 0; i < kGraphN; ++i) v[i] = base + step * (float) i;
    return v;
}

int64_t graph_mismatches(ggml_tensor * t, const std::vector<float> & want) {
    std::vector<float> got(want.size());
    ggml_backend_tensor_get(t, got.data(), 0, sizeof(float) * got.size());
    int64_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
    return bad;
}
}  // namespace

// Graph pass: every dependency must end a run so copies keep their order.
//   1 a -> x   2 b -> y   3 x -> z   read after write (3 reads 1's dst)
//   4 c -> w   5 d -> c              write after read (5 writes 4's src)
//   6 e -> u   7 f -> u              write after write (7 writes 6's dst)
//   8 h -> t
// The pass issues exactly four runs: [1 2] [3 4] [5 6] [7 8]. Dropping any
// one of the three independence checks merges two of them (three runs), so
// the count pins each check even when a race would not show in the data.
TEST_CASE(CudaCopyBatchFixture, graph_copy_dependencies) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);
    ggml_init_params gp{};
    gp.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead();
    gp.no_alloc = true;
    ggml_context * gctx = ggml_init(gp);
    REQUIRE(gctx != nullptr);
    auto tensor = [&] { return ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kGraphN); };
    ggml_tensor * a = tensor(); ggml_tensor * b = tensor(); ggml_tensor * c = tensor();
    ggml_tensor * d = tensor(); ggml_tensor * e = tensor(); ggml_tensor * f = tensor();
    ggml_tensor * h = tensor();
    ggml_tensor * x = tensor(); ggml_tensor * y = tensor(); ggml_tensor * z = tensor();
    ggml_tensor * w = tensor(); ggml_tensor * u = tensor(); ggml_tensor * t = tensor();
    ggml_cgraph * gf = ggml_new_graph(gctx);
    const std::pair<ggml_tensor *, ggml_tensor *> order[] = {
        {a, x}, {b, y}, {x, z}, {c, w}, {d, c}, {e, u}, {f, u}, {h, t}};
    for (const auto & [src, dst] : order) {
        ggml_build_forward_expand(gf, ggml_cpy(gctx, src, dst));
    }
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
    REQUIRE(gbuf != nullptr);
    const auto va = graph_pattern(1.0f, 1.0f), vb = graph_pattern(0.0f, -2.0f);
    const auto vc = graph_pattern(0.0f, 0.5f), vd = graph_pattern(7.0f, -1.0f);
    const auto ve = graph_pattern(2.0f, 0.25f), vf = graph_pattern(9.0f, 3.0f);
    const auto vh = graph_pattern(-4.0f, 0.75f), zero = graph_pattern(0.0f, 0.0f);
    const std::pair<ggml_tensor *, const std::vector<float> *> inputs[] = {
        {a, &va}, {b, &vb}, {c, &vc}, {d, &vd}, {e, &ve}, {f, &vf}, {h, &vh},
        {x, &zero}, {y, &zero}, {z, &zero}, {w, &zero}, {u, &zero}, {t, &zero}};
    for (const auto & [tensor_in, values] : inputs) {
        ggml_backend_tensor_set(tensor_in, values->data(), 0, sizeof(float) * kGraphN);
    }
    const size_t runs_before = ggml_backend_cuda_get_copy_batch_run_count();
    const bool computed = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    const size_t runs = ggml_backend_cuda_get_copy_batch_run_count() - runs_before;
    const std::pair<ggml_tensor *, const std::vector<float> *> checks[] = {
        {x, &va}, {y, &vb}, {z, &va}, {w, &vc}, {c, &vd}, {u, &vf}, {t, &vh}};
    int64_t bad = 0;
    for (const auto & [tensor_out, want] : checks) bad += graph_mismatches(tensor_out, *want);
    ggml_backend_buffer_free(gbuf);
    ggml_free(gctx);
    ggml_backend_free(gpu);
    const bool disabled = std::getenv("GGML_CUDA_DISABLE_COPY_BATCH") != nullptr;
    std::printf("[cuda-copy-batch] dependency graph: %lld mismatched values, %zu batched runs%s\n",
                (long long) bad, runs, disabled ? " (batching disabled)" : "");
    REQUIRE(computed);
    REQUIRE(bad == 0);
    REQUIRE(runs == (disabled ? 0u : 4u));
}

// A run of 60 independent small copies is longer than one launch.
TEST_CASE(CudaCopyBatchFixture, graph_copy_long_run) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);
    constexpr int kRun = 60;
    ggml_init_params gp{};
    gp.mem_size = (4 * kRun) * ggml_tensor_overhead() + ggml_graph_overhead();
    gp.no_alloc = true;
    ggml_context * gctx = ggml_init(gp);
    REQUIRE(gctx != nullptr);
    ggml_tensor * run_src[kRun];
    ggml_tensor * run_dst[kRun];
    ggml_cgraph * gf = ggml_new_graph(gctx);
    for (int r = 0; r < kRun; ++r) {
        run_src[r] = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 33 + r);
        run_dst[r] = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 33 + r);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, run_src[r], run_dst[r]));
    }
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
    REQUIRE(gbuf != nullptr);
    for (int r = 0; r < kRun; ++r) {
        std::vector<float> v(33 + r, 3.0f + (float) r);
        std::vector<float> zero(33 + r, 0.0f);
        ggml_backend_tensor_set(run_src[r], v.data(), 0, sizeof(float) * v.size());
        ggml_backend_tensor_set(run_dst[r], zero.data(), 0, sizeof(float) * zero.size());
    }
    const size_t runs_before = ggml_backend_cuda_get_copy_batch_run_count();
    const bool computed = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    const size_t runs = ggml_backend_cuda_get_copy_batch_run_count() - runs_before;
    int64_t bad = 0;
    for (int r = 0; r < kRun; ++r) {
        std::vector<float> v(33 + r);
        ggml_backend_tensor_get(run_dst[r], v.data(), 0, sizeof(float) * v.size());
        for (float value : v) bad += value != 3.0f + (float) r;
    }
    ggml_backend_buffer_free(gbuf);
    ggml_free(gctx);
    ggml_backend_free(gpu);
    const bool disabled = std::getenv("GGML_CUDA_DISABLE_COPY_BATCH") != nullptr;
    std::printf("[cuda-copy-batch] long run: %lld mismatched values, %zu batched runs%s\n",
                (long long) bad, runs, disabled ? " (batching disabled)" : "");
    REQUIRE(computed);
    REQUIRE(bad == 0);
    if (disabled) {
        REQUIRE(runs == 0);
    } else {
        REQUIRE(runs >= 2);
    }
}
