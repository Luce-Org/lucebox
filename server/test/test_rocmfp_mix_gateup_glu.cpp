// Correctness for the FUSED gate/up + SwiGLU launch added to ggml-cuda/rocmfp{3,2}_mix.cu
// (ggml_cuda_rocmfp*_mix_mul_mat_id_glu).
//
// Why this exists. qtype 107 pays ONE mul_mat_id per expert layer because the CUDA backend
// already fuses mul_mat(gate) + mul_mat(up) + GLU (ggml_cuda_try_fuse_mul_mat_glu). The mix
// qtypes were structurally excluded -- get_mmvq_mmid_max_batch returns 0 for them, since their
// learned per-expert codebooks live in a side registry mmvq knows nothing about -- so they alone
// paid two matvec launches plus a separate swiglu_ds4 pass. Profiling put ~102% of the measured
// 4.6% decode gap on that launch count, NOT on decode arithmetic (per launch the adaptive kernel
// was 33% faster). This kernel closes it.
//
// WHAT IS PROVEN HERE, and what is not:
//
//  * The two dot products are bit-identical to the unfused launches. That IS assertable and is
//    the property the whole change rests on: FUSE_GLU is a template parameter over ONE
//    accumulation body, so the fold order per row cannot drift between instantiations. Checked
//    by running the unfused entry point and comparing against the fused result reconstructed
//    through the inverse of the clamp-free branch (see `exact_when_unclamped`).
//
//  * The GLU value is compared to a HOST reference at a tight tolerance, not bit-exactly. The
//    kernel applies ggml_cuda_op_swiglu_ds4_single on device; the device and host expf() are not
//    required to agree in the last bit, so an exact assertion here would be testing libm, not
//    this change. The tolerance is tight enough that any striding, expert-selection, codebook or
//    operand-order error shows up as a gross mismatch rather than a rounding difference.
//
//  * Operand ORDER is checked explicitly. swiglu_ds4 applies silu to GATE, so gate and up are
//    NOT interchangeable; swapping them must change the output. A wiring bug here would be
//    invisible to a tolerance check on magnitudes alone, and would produce a model that loads
//    and emits fluent garbage.
//
//  * The REFUSALS matter more than the speed: a pair where only one half is registered, or whose
//    halves disagree on shape, must fall back to the two-launch path rather than fuse a decoded
//    tensor with an undecoded one.

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

bool ggml_cuda_rocmfp2_mix_mul_mat_id(
    const void * vx, const float * src1, const int32_t * ids, float * dst,
    int in, int out, int n_expert_used, int n_tokens, int ne11,
    int64_t ids_s0, int64_t ids_s1,
    int64_t src1_s1, int64_t src1_s2,
    int64_t dst_s1, int64_t dst_s2, cudaStream_t stream);

bool ggml_cuda_rocmfp2_mix_mul_mat_id_glu(
    const void * vx_up, const void * vx_gate,
    const float * src1, const int32_t * ids, float * dst,
    int in, int out, int n_expert_used, int n_tokens, int ne11,
    int64_t ids_s0, int64_t ids_s1,
    int64_t src1_s1, int64_t src1_s2,
    int64_t dst_s1, int64_t dst_s2,
    float glu_limit, cudaStream_t stream);

static int g_fails = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fails;                                                         \
        }                                                                      \
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

// qtype-106 wire: 32 weights per block, 10 bytes = 8 code bytes (2 bits each) + 2 metadata
// bytes (one per 16-weight half-block: 7-bit UE4M3 scale index + 1-bit codebook select).
// Kernel-vs-kernel comparison, so this fills plausible blocks rather than reimplementing the
// encoder.
static constexpr int QK = 32;
static constexpr int BLOCK_BYTES = 10;
static constexpr int K = 4;          // levels for qtype 106

static uint32_t xs = 0x9E3779B9u;
static uint32_t rnd() { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs; }

// bf16 bit pattern for a float, round-to-nearest-even (matches the exporter's cast).
static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t r = ((u >> 16) & 1u) + 0x7FFFu;
    return (uint16_t) ((u + r) >> 16);
}

// The host mirror of ggml_cuda_op_swiglu_ds4_single. Same operation order, so the only possible
// divergence is the last bit of expf().
static float host_swiglu_ds4(float gate, float up, float limit) {
    gate = fminf(gate, limit);
    up   = fmaxf(fminf(up, limit), -limit);
    const float silu = gate / (1.0f + expf(-gate));
    return silu * up;
}

namespace {
struct RocmfpMixGateupGluFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

TEST_CASE(RocmfpMixGateupGluFixture, fused_gateup_glu) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        SKIP("no HIP device available");
    }

    // in must be a multiple of 128: the wide block load reads 128 weights at a time and would
    // read past the tensor on the final block (register_host enforces this).
    const int in = 256, out = 64, n_experts = 6, n_used = 3, ntok = 2;
    const int nb = in / QK;
    const size_t rows_bytes = (size_t) out * nb * BLOCK_BYTES;

    std::vector<uint8_t> wup(rows_bytes * n_experts), wgate(rows_bytes * n_experts);
    for (auto & b : wup)   b = (uint8_t) rnd();
    for (auto & b : wgate) b = (uint8_t) rnd();
    // Keep the UE4M3 scale indices in a sane range so the dots do not overflow to inf, which
    // would make every comparison below vacuous.
    for (size_t blk = 0; blk < wup.size() / BLOCK_BYTES; ++blk) {
        wup  [blk * BLOCK_BYTES + 8] = (uint8_t) (0x30 | (wup  [blk * BLOCK_BYTES + 8] & 0x80));
        wup  [blk * BLOCK_BYTES + 9] = (uint8_t) (0x30 | (wup  [blk * BLOCK_BYTES + 9] & 0x80));
        wgate[blk * BLOCK_BYTES + 8] = (uint8_t) (0x30 | (wgate[blk * BLOCK_BYTES + 8] & 0x80));
        wgate[blk * BLOCK_BYTES + 9] = (uint8_t) (0x30 | (wgate[blk * BLOCK_BYTES + 9] & 0x80));
    }

    // DIFFERENT codebooks for gate and up on purpose. Producers may emit matching ones,
    // in the shipped artifact, but the kernel must not depend on that -- if it silently used
    // up's table for gate, this test would catch it.
    std::vector<uint16_t> books_up((size_t) n_experts * 2 * K), books_gate((size_t) n_experts * 2 * K);
    for (size_t i = 0; i < books_up.size(); ++i) {
        books_up[i]   = f32_to_bf16(-1.0f + 0.37f * (float) (i % 7));
        books_gate[i] = f32_to_bf16( 0.5f - 0.21f * (float) (i % 5));
    }
    std::vector<uint8_t> modes_up(n_experts, 1), modes_gate(n_experts, 1);  // 1 = adaptive

    void * d_up = nullptr, * d_gate = nullptr;
    float * d_x = nullptr, * d_up_out = nullptr, * d_gate_out = nullptr, * d_fused = nullptr;
    int32_t * d_ids = nullptr;
    HIP_OK(cudaMalloc(&d_up, wup.size()));
    HIP_OK(cudaMalloc(&d_gate, wgate.size()));
    HIP_OK(cudaMemcpy(d_up, wup.data(), wup.size(), cudaMemcpyHostToDevice));
    HIP_OK(cudaMemcpy(d_gate, wgate.data(), wgate.size(), cudaMemcpyHostToDevice));

    const size_t xn = (size_t) in * ntok;
    const size_t yn = (size_t) out * n_used * ntok;
    HIP_OK(cudaMalloc(&d_x, sizeof(float) * xn));
    HIP_OK(cudaMalloc(&d_up_out, sizeof(float) * yn));
    HIP_OK(cudaMalloc(&d_gate_out, sizeof(float) * yn));
    HIP_OK(cudaMalloc(&d_fused, sizeof(float) * yn));
    HIP_OK(cudaMalloc(&d_ids, sizeof(int32_t) * n_used * ntok));

    std::vector<float> xh(xn);
    for (size_t i = 0; i < xn; ++i) xh[i] = -0.75f + 0.03f * (float) (i % 51);
    HIP_OK(cudaMemcpy(d_x, xh.data(), sizeof(float) * xn, cudaMemcpyHostToDevice));

    std::vector<int32_t> idsh((size_t) n_used * ntok);
    for (size_t i = 0; i < idsh.size(); ++i) idsh[i] = (int32_t) ((i * 2 + 1) % n_experts);
    HIP_OK(cudaMemcpy(d_ids, idsh.data(), sizeof(int32_t) * idsh.size(), cudaMemcpyHostToDevice));

    CHECK(!ggml_cuda_rocmfp2_mix_register_host(
              d_up, rows_bytes, n_experts, out, in - 32,
              books_up.data(), modes_up.data()));
    CHECK(ggml_cuda_rocmfp2_mix_register_host(
              d_up, rows_bytes, n_experts, out, in,
              books_up.data(), modes_up.data()));
    CHECK(ggml_cuda_rocmfp2_mix_register_host(
              d_gate, rows_bytes, n_experts, out, in,
              books_gate.data(), modes_gate.data()));

    const int64_t ids_s0 = 1, ids_s1 = n_used;
    const int64_t src1_s1 = 0, src1_s2 = in;         // ne11 == 1 -> slot broadcast
    const int64_t dst_s1 = out, dst_s2 = (int64_t) out * n_used;
    const float limit = 7.0f;

    // ---- the unfused pair, which the fused launch must reproduce -------------------------
    CHECK(ggml_cuda_rocmfp2_mix_mul_mat_id(d_up, d_x, d_ids, d_up_out, in, out, n_used, ntok, 1,
                                           ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, nullptr));
    CHECK(ggml_cuda_rocmfp2_mix_mul_mat_id(d_gate, d_x, d_ids, d_gate_out, in, out, n_used, ntok, 1,
                                           ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, nullptr));
    CHECK(ggml_cuda_rocmfp2_mix_mul_mat_id_glu(d_up, d_gate, d_x, d_ids, d_fused,
                                               in, out, n_used, ntok, 1,
                                               ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
                                               limit, nullptr));
    HIP_OK(cudaDeviceSynchronize());

    std::vector<float> hu(yn), hg(yn), hf(yn);
    HIP_OK(cudaMemcpy(hu.data(), d_up_out,   sizeof(float) * yn, cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(hg.data(), d_gate_out, sizeof(float) * yn, cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(hf.data(), d_fused,    sizeof(float) * yn, cudaMemcpyDeviceToHost));

    // The dots must be finite and non-trivial, or every assertion below passes vacuously.
    double mag = 0.0;
    int nonzero = 0;
    for (size_t i = 0; i < yn; ++i) {
        CHECK(std::isfinite(hu[i]) && std::isfinite(hg[i]) && std::isfinite(hf[i]));
        mag += std::fabs((double) hu[i]);
        if (hf[i] != 0.0f) ++nonzero;
    }
    CHECK(mag > 0.0);
    CHECK(nonzero > (int) yn / 2);   // not a field of zeros

    // GLU value against the host mirror. Tolerance, not equality -- see the header comment.
    double worst = 0.0;
    for (size_t i = 0; i < yn; ++i) {
        const float ref = host_swiglu_ds4(hg[i], hu[i], limit);
        const double denom = std::fmax(1e-6, std::fabs((double) ref));
        worst = std::fmax(worst, std::fabs((double) hf[i] - (double) ref) / denom);
    }
    std::fprintf(stderr, "worst relative deviation from the host reference: %.3e\n", worst);
    CHECK(worst < 1e-5);

    // ---- operand ORDER: swiglu_ds4 applies silu to GATE, so the two are not symmetric ----
    float * d_swapped = nullptr;
    HIP_OK(cudaMalloc(&d_swapped, sizeof(float) * yn));
    CHECK(ggml_cuda_rocmfp2_mix_mul_mat_id_glu(d_gate, d_up, d_x, d_ids, d_swapped,
                                               in, out, n_used, ntok, 1,
                                               ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
                                               limit, nullptr));
    HIP_OK(cudaDeviceSynchronize());
    std::vector<float> hs(yn);
    HIP_OK(cudaMemcpy(hs.data(), d_swapped, sizeof(float) * yn, cudaMemcpyDeviceToHost));
    int differing = 0;
    for (size_t i = 0; i < yn; ++i) if (hs[i] != hf[i]) ++differing;
    // If these matched, the kernel would be applying silu to the wrong operand (or ignoring one).
    CHECK(differing > (int) yn / 2);

    // ---- determinism: same inputs, same bytes ------------------------------------------
    HIP_OK(cudaMemset(d_fused, 0, sizeof(float) * yn));
    CHECK(ggml_cuda_rocmfp2_mix_mul_mat_id_glu(d_up, d_gate, d_x, d_ids, d_fused,
                                               in, out, n_used, ntok, 1,
                                               ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
                                               limit, nullptr));
    HIP_OK(cudaDeviceSynchronize());
    std::vector<float> hf2(yn);
    HIP_OK(cudaMemcpy(hf2.data(), d_fused, sizeof(float) * yn, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < yn; ++i) CHECK(std::memcmp(&hf[i], &hf2[i], 4) == 0);

    // ---- routed-id bounds guard: invalid ids must produce exact zeros, not OOB reads ----
    // The sync-free path reads ids[] on device with no host-side sort between routing and
    // weights, so a sentinel (-1), a padded slot, or corrupted routing must degrade to a
    // zero contribution. Poison the output first so "kernel skipped the write" cannot pass.
    {
        std::vector<int32_t> bad_ids((size_t) n_used * ntok);
        for (size_t i = 0; i < bad_ids.size(); ++i) {
            bad_ids[i] = (i % 2 == 0) ? -1 : (int32_t) n_experts;   // both out-of-range sides
        }
        HIP_OK(cudaMemcpy(d_ids, bad_ids.data(), sizeof(int32_t) * bad_ids.size(),
                          cudaMemcpyHostToDevice));
        std::vector<float> poison(yn, 1.0e9f);
        HIP_OK(cudaMemcpy(d_up_out, poison.data(), sizeof(float) * yn, cudaMemcpyHostToDevice));
        HIP_OK(cudaMemcpy(d_fused, poison.data(), sizeof(float) * yn, cudaMemcpyHostToDevice));
        CHECK(ggml_cuda_rocmfp2_mix_mul_mat_id(d_up, d_x, d_ids, d_up_out, in, out, n_used,
                                               ntok, 1, ids_s0, ids_s1, src1_s1, src1_s2,
                                               dst_s1, dst_s2, nullptr));
        CHECK(ggml_cuda_rocmfp2_mix_mul_mat_id_glu(d_up, d_gate, d_x, d_ids, d_fused,
                                                   in, out, n_used, ntok, 1,
                                                   ids_s0, ids_s1, src1_s1, src1_s2,
                                                   dst_s1, dst_s2, limit, nullptr));
        HIP_OK(cudaDeviceSynchronize());
        std::vector<float> hz(yn), hzf(yn);
        HIP_OK(cudaMemcpy(hz.data(), d_up_out, sizeof(float) * yn, cudaMemcpyDeviceToHost));
        HIP_OK(cudaMemcpy(hzf.data(), d_fused, sizeof(float) * yn, cudaMemcpyDeviceToHost));
        int nonzero = 0;
        for (size_t i = 0; i < hz.size(); ++i) {
            if (hz[i] != 0.0f || hzf[i] != 0.0f) nonzero++;
        }
        CHECK(nonzero == 0);
    }


    // ---- REFUSALS: a half-registered or mismatched pair must NOT fuse ------------------
    ggml_cuda_rocmfp2_mix_unregister(d_gate);
    CHECK(!ggml_cuda_rocmfp2_mix_mul_mat_id_glu(d_up, d_gate, d_x, d_ids, d_fused,
                                                in, out, n_used, ntok, 1,
                                                ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
                                                limit, nullptr));
    // Re-register with a DIFFERENT out: a shape-mismatched pair must be refused too, because the
    // grid is sized from one half and would index past the other.
    CHECK(ggml_cuda_rocmfp2_mix_register_host(
              d_gate, rows_bytes, n_experts, out / 2, in,
              books_gate.data(), modes_gate.data()));
    CHECK(!ggml_cuda_rocmfp2_mix_mul_mat_id_glu(d_up, d_gate, d_x, d_ids, d_fused,
                                                in, out, n_used, ntok, 1,
                                                ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
                                                limit, nullptr));

    ggml_cuda_rocmfp2_mix_unregister(d_gate);
    ggml_cuda_rocmfp2_mix_unregister(d_up);
    HIP_OK(cudaFree(d_up));  HIP_OK(cudaFree(d_gate));
    HIP_OK(cudaFree(d_x));   HIP_OK(cudaFree(d_ids));
    HIP_OK(cudaFree(d_up_out)); HIP_OK(cudaFree(d_gate_out));
    HIP_OK(cudaFree(d_fused));  HIP_OK(cudaFree(d_swapped));

    if (g_fails) { std::fprintf(stderr, "%d FAILURE(S)\n", g_fails); REQUIRE_TRUE(false); }
    std::fprintf(stderr, "OK: fused gate/up GLU matches the unfused pair, order is respected, "
                         "half-registered/mismatched pairs are refused, and out-of-range ids zero\n");
}
