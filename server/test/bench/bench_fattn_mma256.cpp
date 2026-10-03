// DKQ=256 fattn kernel throughput bench (RDNA4, gfx1201).
//
// Times ggml_flash_attn_ext at chunked-prefill shapes (nq=512 query rows
// over a growing KV) with q8_0 K/V, the Qwen3.8-27B serving config. The
// default run takes the tensor-core route (LUCE_FA256_MMA defaults
// to 1 on RDNA4); LUCE_FA256_MMA=0 A/B's the tile kernel,
// LUCE_FA256_WMMA=1 forces the rocWMMA kernel on flag builds.
// Throughput only; no correctness checks.
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int D  = 256; // head dim
constexpr int Hq = 24;  // query heads
constexpr int Hk = 4;   // KV heads (gqa ratio 6)
constexpr int nq = 512; // prefill chunk rows

uint32_t g_rng = 0x9e3779b9u;
float next_float() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((int32_t)(g_rng >> 8) - 8388608) / 8388608.0f; // [-1, 1)
}

} // namespace

int main() {
#if defined(GGML_USE_HIP)
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess ||
        std::strncmp(props.gcnArchName, "gfx12", 5) != 0) {
        std::printf("[bench-fattn-mma256] SKIP: requires gfx12 (RDNA4)\n");
        return 77;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) return 1;

    const int iters = 10;
    std::printf("[bench-fattn-mma256] D=%d Hq=%d Hk=%d nq=%d iters=%d\n",
                D, Hq, Hk, nq, iters);

    for (int S : { 8192, 16384, 32768, 65536, 131072 }) {
        ggml_init_params params = { 4u << 20, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return 1;

        ggml_tensor * Q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  D, nq, Hq);
        ggml_tensor * K    = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, S,  Hk);
        ggml_tensor * V    = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, S,  Hk);
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,  S, nq);
        for (ggml_tensor * t : { Q, K, V, mask }) {
            ggml_set_input(t);
        }
        ggml_tensor * out = ggml_flash_attn_ext(ctx, Q, K, V, mask,
                                                1.0f / std::sqrt((float) D), 0.0f, 0.0f);
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            std::printf("[bench-fattn-mma256] S=%d alloc failed\n", S);
            ggml_free(ctx);
            continue;
        }

        std::vector<float> qv(ggml_nelements(Q));
        for (float & x : qv) x = next_float();
        ggml_backend_tensor_set(Q, qv.data(), 0, ggml_nbytes(Q));

        const size_t n_rows = (size_t) S * Hk;
        std::vector<float> kvf(n_rows * D);
        for (float & x : kvf) x = next_float();
        std::vector<uint8_t> kvq(ggml_nbytes(K));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, kvf.data(), kvq.data(), 0,
                            (int64_t) n_rows, D, nullptr);
        ggml_backend_tensor_set(K, kvq.data(), 0, kvq.size());
        ggml_backend_tensor_set(V, kvq.data(), 0, kvq.size());

        std::vector<ggml_fp16_t> mv((size_t) S * nq);
        for (int q = 0; q < nq; ++q) {
            for (int kv = 0; kv < S; ++kv) {
                mv[(size_t) q * S + kv] = ggml_fp32_to_fp16(kv <= q ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(mask, mv.data(), 0, mv.size() * sizeof(ggml_fp16_t));

        for (int i = 0; i < 3; ++i) {
            ggml_backend_graph_compute(gpu, gf); // warmup
        }
        hipDeviceSynchronize();
        const size_t mma256_before  = ggml_backend_cuda_get_fattn_mma256_launch_count();
        const size_t wmma256_before = ggml_backend_cuda_get_fattn_wmma256_launch_count();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(gpu, gf);
        }
        hipDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const size_t mma256_launches =
            ggml_backend_cuda_get_fattn_mma256_launch_count() - mma256_before;
        const size_t wmma256_launches =
            ggml_backend_cuda_get_fattn_wmma256_launch_count() - wmma256_before;

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        std::printf("[bench-fattn-mma256] S=%6d kv=q8_0 mma256=%zu wmma256=%zu %8.2f ms/iter %7.1f tok/s\n",
                    S, mma256_launches / iters, wmma256_launches / iters, ms, nq * 1000.0 / ms);

        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }
    ggml_backend_free(gpu);
    return 0;
#else
    std::printf("[bench-fattn-mma256] SKIP: HIP-only kernel path\n");
    return 77;
#endif
}
