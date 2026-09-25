#include "CppUnitTestFramework.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
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
