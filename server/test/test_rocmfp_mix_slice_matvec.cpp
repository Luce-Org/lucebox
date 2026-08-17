// Numerical correctness for the fused DENSE 3-D-slice matvec added to
// ggml-cuda/rocmfp{3,2}_mix.cu (ggml_cuda_rocmfp*_mix_mul_mat_vec_3d).
//
// Why this exists. The target reshapes attn_output_a to
// [group_dim, n_lora_o, n_out_group] and mul_mats it against a matching 3-D src1
// (deepseek4_graph.cpp:2122), so src1->ne[2] > 1 and the 2-D fused hooks reject it on
// `src1->ne[2] == 1`. The generic fallback for a mix qtype is dequantize->cuBLAS, which
// reads the blocks, writes a full f16 copy and reads it back -- MORE bytes than the f16
// weight it replaced, on ~31% of the attention weight read. The slice kernel handles it.
//
// The claim under test is not "it runs" but **bit-identity with the already-validated
// per-slice path**. The slice kernel is the MoE kernel with `expert = blockIdx.y` instead
// of an ids[] lookup, so for the same weights each output element must match what the
// 2-D entry point produces for that slice, exactly -- not to a tolerance. Anything else
// means the slice striding or the accumulation order drifted, and a reassociated fold
// flips greedy tokens (every kernel comment in those files says the correctness gate
// hashes the greedy output).
//
// Also covers the guard that matters more than speed: a 3-D tensor registered with fewer
// slices than the grid will index must be REFUSED, not silently decoded against a
// neighbouring tensor's codebook. That failure mode is wrong numbers, not a crash.

#include "ds4_test_gpu_runtime.h"
#include "CppUnitTestFramework.hpp"
#include "ggml-cuda.h"
using CppUnitTestFramework::CommonFixture;
#undef CHECK

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

bool ggml_cuda_rocmfp3_mix_mul_mat_vec(
    const void * vx, const float * x, float * y,
    int in, int out, int ncols,
    int64_t x_col_stride, int64_t y_col_stride, cudaStream_t stream);

bool ggml_cuda_rocmfp3_mix_mul_mat_vec_3d(
    const void * vx, const float * src1, float * dst,
    int in, int out, int nslices, int ntokens,
    int64_t src1_token_stride, int64_t src1_slice_stride,
    int64_t dst_token_stride,  int64_t dst_slice_stride,
    cudaStream_t stream);

static int g_fails = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fails; } \
    } while (0)

#define HIP_OK(expr)                                                           \
    do {                                                                       \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "FAIL: %s -> %s\n", #expr,                    \
                         cudaGetErrorString(_e));                               \
            ++g_fails;                                                         \
        }                                                                      \
    } while (0)

// qtype-105 wire: 32 weights per block, 14 bytes = 12 code bytes (3 bits each) + 2
// metadata bytes (one per 16-weight half-block: 7-bit UE4M3 scale index + 1-bit
// codebook select). Only the byte layout matters here -- the kernel decodes it and we
// compare kernel-vs-kernel, so this fills plausible blocks rather than reimplementing
// the encoder.
static constexpr int QK = 32;
static constexpr int BLOCK_BYTES = 14;

static uint32_t xs = 0x2545F491u;
static uint32_t rnd() { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs; }

namespace {
struct RocmfpMixSliceMatvecFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

TEST_CASE(RocmfpMixSliceMatvecFixture, slice_matvec_matches_reference) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        SKIP("no HIP device available");
    }

    // Shapes echo the real case in miniature: attn_output_a is [group_dim, n_lora_o]
    // viewed as [group_dim, n_lora_o/nslices, nslices]. `in` must be a multiple of 128
    // so the wide load-from-floor staging in mix_block_accum stays in bounds on the
    // final block (the in % 128 guard in register_host enforces this).
    const int in = 128, out = 16, nslices = 4, ntokens = 3;
    const int nb = in / QK;
    const size_t slice_bytes = (size_t) out * nb * BLOCK_BYTES;

    std::vector<uint8_t> blocks(slice_bytes * nslices);
    for (auto & b : blocks) b = (uint8_t) (rnd() & 0xFF);
    // Keep the scale indices in the finite range of UE4M3 (>0x7E decodes to 0.0, which
    // would make whole half-blocks vanish and weaken the comparison).
    for (size_t blk = 0; blk < blocks.size() / BLOCK_BYTES; ++blk) {
        for (int h = 0; h < 2; ++h) {
            uint8_t & m = blocks[blk * BLOCK_BYTES + 12 + h];
            m = (uint8_t) (m & 0x7F);
            if (m > 0x7E) m = 0x40;
        }
    }

    // One codebook PAIR per slice. For a real dense tensor the codebook is per-tensor
    // and would be replicated across slices; here they are deliberately DIFFERENT per
    // slice, so a kernel that ignored the slice stride on `codebooks` would produce
    // wrong values and fail the comparison rather than passing by luck.
    const int K = 8;
    std::vector<uint16_t> books((size_t) nslices * 2 * K);
    for (size_t i = 0; i < books.size(); ++i) {
        // plausible bf16 magnitudes around +/-1, varying per slice
        const float v = 0.25f * (float) ((int) (i % 7) - 3) + 0.05f * (float) (i / 16);
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        books[i] = (uint16_t) (bits >> 16);   // fp32 -> bf16 (truncate)
    }
    std::vector<uint8_t> modes(nslices, 1);   // 1 = adaptive: exercises the codebook path

    uint8_t * d_w = nullptr;
    float   * d_x = nullptr;
    float   * d_y3 = nullptr;
    float   * d_y2 = nullptr;
    HIP_OK(cudaMalloc(&d_w, blocks.size()));
    HIP_OK(cudaMemcpy(d_w, blocks.data(), blocks.size(), cudaMemcpyHostToDevice));
    HIP_OK(cudaMalloc(&d_x, sizeof(float) * (size_t) in * ntokens * nslices));
    HIP_OK(cudaMalloc(&d_y3, sizeof(float) * (size_t) out * ntokens * nslices));
    HIP_OK(cudaMalloc(&d_y2, sizeof(float) * (size_t) out * ntokens * nslices));

    std::vector<float> xh((size_t) in * ntokens * nslices);
    for (size_t i = 0; i < xh.size(); ++i) {
        xh[i] = 0.5f - (float) (rnd() % 1000) / 1000.0f;
    }
    HIP_OK(cudaMemcpy(d_x, xh.data(), sizeof(float) * xh.size(), cudaMemcpyHostToDevice));
    HIP_OK(cudaMemset(d_y3, 0, sizeof(float) * (size_t) out * ntokens * nslices));
    HIP_OK(cudaMemset(d_y2, 0, sizeof(float) * (size_t) out * ntokens * nslices));

    // Contiguous [in, ntokens, nslices] src1 and [out, ntokens, nslices] dst.
    const int64_t s1_tok = in,  s1_sl = (int64_t) in  * ntokens;
    const int64_t d_tok  = out, d_sl  = (int64_t) out * ntokens;

    // --- 1. the guard: too few registered slices must be REFUSED, not decoded ---
    CHECK(ggml_cuda_rocmfp3_mix_register_host(
              d_w, slice_bytes, /*n_experts=*/nslices - 1,
              out, in, books.data(), modes.data()),
          "partial slice registration succeeds");
    CHECK(!ggml_cuda_rocmfp3_mix_mul_mat_vec_3d(
              d_w, d_x, d_y3, in, out, nslices, ntokens,
              s1_tok, s1_sl, d_tok, d_sl, nullptr),
          "3d matvec refuses a tensor registered with fewer slices than the grid indexes");
    ggml_cuda_rocmfp3_mix_unregister(d_w);

    // --- 2. registered metadata must match the requested launch shape -----------
    CHECK(ggml_cuda_rocmfp3_mix_register_host(
              d_w, slice_bytes, nslices, out, in,
              books.data(), modes.data()),
          "full slice registration succeeds");
    CHECK(!ggml_cuda_rocmfp3_mix_mul_mat_vec_3d(
              d_w, d_x, d_y3, in, out + 1, nslices, ntokens,
              s1_tok, s1_sl, d_tok, d_sl, nullptr),
          "3d matvec refuses a shape that differs from its registration");
    CHECK(!ggml_cuda_rocmfp3_mix_mul_mat_vec(
              d_w + BLOCK_BYTES, d_x, d_y2, in, out, 1,
              s1_tok, d_tok, nullptr),
          "2d matvec refuses an interior block pointer as a full slice");

    // --- 3. correctness: 3-D launch vs the validated 2-D entry point, per slice ---
    CHECK(ggml_cuda_rocmfp3_mix_mul_mat_vec_3d(
              d_w, d_x, d_y3, in, out, nslices, ntokens,
              s1_tok, s1_sl, d_tok, d_sl, nullptr),
          "3d matvec handles a fully registered tensor");
    HIP_OK(cudaDeviceSynchronize());

    // Reference: call the 2-D path once per slice. Its base pointer lands inside the
    // registered range, so mix_lookup resolves that slice's own codebook/mode -- the
    // same side-data the 3-D kernel selects via blockIdx.y.
    for (int s = 0; s < nslices; ++s) {
        const bool ok = ggml_cuda_rocmfp3_mix_mul_mat_vec(
            d_w + (size_t) s * slice_bytes,
            d_x + (size_t) s * s1_sl, d_y2 + (size_t) s * d_sl,
            in, out, ntokens, s1_tok, d_tok, nullptr);
        CHECK(ok, "2d reference path handles each slice");
    }
    HIP_OK(cudaDeviceSynchronize());

    std::vector<float> y3((size_t) out * ntokens * nslices);
    std::vector<float> y2(y3.size());
    HIP_OK(cudaMemcpy(y3.data(), d_y3, sizeof(float) * y3.size(), cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(y2.data(), d_y2, sizeof(float) * y2.size(), cudaMemcpyDeviceToHost));

    // BIT-identical, not close: same decode, same ascending-j fold, same acc chain.
    size_t mismatches = 0, first = 0;
    bool all_zero = true;
    for (size_t i = 0; i < y3.size(); ++i) {
        if (y3[i] != 0.0f) all_zero = false;
        if (std::memcmp(&y3[i], &y2[i], sizeof(float)) != 0) {
            if (mismatches == 0) first = i;
            ++mismatches;
        }
    }
    CHECK(!all_zero, "output is not all zeros (a null kernel would trivially 'match')");
    if (mismatches) {
        std::fprintf(stderr,
                     "FAIL: %zu/%zu elements differ from the per-slice reference; "
                     "first at %zu (3d=%.9g 2d=%.9g)\n",
                     mismatches, y3.size(), first, (double) y3[first], (double) y2[first]);
        ++g_fails;
    }

    ggml_cuda_rocmfp3_mix_unregister(d_w);
    HIP_OK(cudaFree(d_w));
    HIP_OK(cudaFree(d_x));
    HIP_OK(cudaFree(d_y3));
    HIP_OK(cudaFree(d_y2));

    if (g_fails == 0) {
        std::printf("PASS: dense 3-D-slice matvec is bit-identical to the per-slice path "
                    "(%d slices x %d tokens x %d rows) and refuses under-registration\n",
                    nslices, ntokens, out);
    }
    REQUIRE_TRUE(g_fails == 0);
}
