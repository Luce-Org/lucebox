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

// Graph pass: a run of plain CPY nodes is batched, but a copy that reads an
// earlier copy's destination must still see the copied bytes.
// a -> x, b -> y, x -> z (depends on the first), c -> w, then a run of 60
// independent small copies (longer than one launch). The pass must fire
// unless GGML_CUDA_DISABLE_COPY_BATCH is set.
TEST_CASE(CudaCopyBatchFixture, graph_copy_runs) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    REQUIRE(gpu != nullptr);

    constexpr int64_t kN = 4099;
    ggml_init_params gp{};
    gp.mem_size = 256 * ggml_tensor_overhead() + ggml_graph_overhead();
    gp.no_alloc = true;
    ggml_context * gctx = ggml_init(gp);
    REQUIRE(gctx != nullptr);
    ggml_tensor * a = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kN);
    ggml_tensor * b = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kN);
    ggml_tensor * c = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kN);
    ggml_tensor * x = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kN);
    ggml_tensor * y = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kN);
    ggml_tensor * z = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kN);
    ggml_tensor * w = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, kN);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, ggml_cpy(gctx, a, x));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, b, y));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, x, z));
    ggml_build_forward_expand(gf, ggml_cpy(gctx, c, w));
    constexpr int kRun = 60;
    ggml_tensor * run_src[kRun];
    ggml_tensor * run_dst[kRun];
    for (int r = 0; r < kRun; ++r) {
        run_src[r] = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 33 + r);
        run_dst[r] = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 33 + r);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, run_src[r], run_dst[r]));
    }
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, gpu);
    REQUIRE(gbuf != nullptr);
    std::vector<float> va(kN), vb(kN), vc(kN), zero(kN, 0.0f);
    for (int64_t i = 0; i < kN; ++i) {
        va[i] = 1.0f + (float) i;
        vb[i] = -2.0f * (float) i;
        vc[i] = 0.5f * (float) i;
    }
    ggml_backend_tensor_set(a, va.data(), 0, sizeof(float) * kN);
    ggml_backend_tensor_set(b, vb.data(), 0, sizeof(float) * kN);
    ggml_backend_tensor_set(c, vc.data(), 0, sizeof(float) * kN);
    for (ggml_tensor * t : {x, y, z, w}) {
        ggml_backend_tensor_set(t, zero.data(), 0, sizeof(float) * kN);
    }
    for (int r = 0; r < kRun; ++r) {
        std::vector<float> v(33 + r, 3.0f + (float) r);
        ggml_backend_tensor_set(run_src[r], v.data(), 0, sizeof(float) * v.size());
        ggml_backend_tensor_set(run_dst[r], zero.data(), 0, sizeof(float) * v.size());
    }
    const size_t runs_before = ggml_backend_cuda_get_copy_batch_run_count();
    const bool computed = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    const size_t runs = ggml_backend_cuda_get_copy_batch_run_count() - runs_before;
    std::vector<float> out(kN);
    int64_t graph_bad = 0;
    const std::pair<ggml_tensor *, const std::vector<float> *> checks[] = {
        {x, &va}, {y, &vb}, {z, &va}, {w, &vc}};
    for (const auto & [t, want] : checks) {
        ggml_backend_tensor_get(t, out.data(), 0, sizeof(float) * kN);
        for (int64_t i = 0; i < kN; ++i) {
            graph_bad += out[i] != (*want)[i];
        }
    }
    for (int r = 0; r < kRun; ++r) {
        std::vector<float> v(33 + r);
        ggml_backend_tensor_get(run_dst[r], v.data(), 0, sizeof(float) * v.size());
        for (float f : v) {
            graph_bad += f != 3.0f + (float) r;
        }
    }
    ggml_backend_buffer_free(gbuf);
    ggml_free(gctx);
    ggml_backend_free(gpu);
    const bool disabled = std::getenv("GGML_CUDA_DISABLE_COPY_BATCH") != nullptr;
    std::printf("[cuda-copy-batch] graph copies: %lld mismatched values, %zu batched runs%s\n",
                (long long) graph_bad, runs, disabled ? " (batching disabled)" : "");
    REQUIRE(computed);
    REQUIRE(graph_bad == 0);
    // The first run ends at the dependent copy; the 61 later copies (c->w
    // plus the 60-copy run) span at least two launches.
    if (disabled) {
        REQUIRE(runs == 0);
    } else {
        REQUIRE(runs >= 2);
    }
}
