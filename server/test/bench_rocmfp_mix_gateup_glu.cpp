// Microbenchmark: the dynamic fused gate/up+SwiGLU path vs TWO plain matvec launches.
//
// Tests the claim the whole fusion rests on. Profiling attributed ~102% of the measured 4.6%
// adaptive decode penalty to LAUNCH COUNT (30100 vs qtype 107's 15050) rather than to decode
// arithmetic -- per launch the adaptive kernel was already 33% faster. If that attribution is
// right, halving the launches at decode shapes should recover most of the gap. This measures it
// directly, at the real decode geometry, without needing the ~100 GB artifact that was deleted.
//
// Deliberately does NOT include the separate swiglu_ds4 pass in the unfused arm: that pass needs
// a ggml graph to invoke, so the unfused number here is an UNDER-estimate of what the unfused
// shape actually costs. The fused arm therefore looks *worse* than reality -- a conservative
// comparison, which is the direction to err in.
//
// The fused wrapper uses one dual-projection launch through q=2 and two
// lower-register launches (with SwiGLU folded into the second) for wider verification. Reports
// per-iteration wall time over many iterations on one stream, after a warmup,
// plus the relative delta. Single stream and back-to-back launches on purpose:
// that is the decode dependency chain, where launch overhead cannot be hidden.

#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

bool ggml_cuda_rocmfp2_mix_mul_mat_id(
    const void * vx, const float * src1, const int32_t * ids, float * dst,
    int in, int out, int n_expert_used, int n_tokens, int ne11,
    int64_t ids_s0, int64_t ids_s1, int64_t src1_s1, int64_t src1_s2,
    int64_t dst_s1, int64_t dst_s2, hipStream_t stream);

bool ggml_cuda_rocmfp2_mix_mul_mat_id_glu(
    const void * vx_up, const void * vx_gate,
    const float * src1, const int32_t * ids, float * dst,
    int in, int out, int n_expert_used, int n_tokens, int ne11,
    int64_t ids_s0, int64_t ids_s1, int64_t src1_s1, int64_t src1_s2,
    int64_t dst_s1, int64_t dst_s2, float glu_limit, hipStream_t stream);

bool ggml_cuda_rocmfp3_mix_mul_mat_id(
    const void * vx, const float * src1, const int32_t * ids, float * dst,
    int in, int out, int n_expert_used, int n_tokens, int ne11,
    int64_t ids_s0, int64_t ids_s1, int64_t src1_s1, int64_t src1_s2,
    int64_t dst_s1, int64_t dst_s2, hipStream_t stream);

bool ggml_cuda_rocmfp3_mix_mul_mat_id_glu(
    const void * vx_up, const void * vx_gate,
    const float * src1, const int32_t * ids, float * dst,
    int in, int out, int n_expert_used, int n_tokens, int ne11,
    int64_t ids_s0, int64_t ids_s1, int64_t src1_s1, int64_t src1_s2,
    int64_t dst_s1, int64_t dst_s2, float glu_limit, hipStream_t stream);

static constexpr int QK = 32;

static uint32_t xs = 0x1234567u;
static uint32_t rnd() { xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs; }
static uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
}

#define HIP_OK(e) do { hipError_t _e = (e); if (_e != hipSuccess) { \
    std::fprintf(stderr, "FAIL %s: %s\n", #e, hipGetErrorString(_e)); return 1; } } while (0)

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static uint64_t fnv1a64(const std::vector<float> & values) {
    uint64_t hash = 1469598103934665603ull;
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(values.data());
    for (size_t i = 0; i < sizeof(float) * values.size(); ++i) {
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    }
    return hash;
}

int main(int argc, char ** argv) {
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        std::fprintf(stderr, "SKIP: no HIP device\n");
        return 0;
    }

    // DeepSeek-V4-Flash verifier geometry: hidden 4096, n_ff_exp 2048,
    // model-default top-k 6. The optional first argument selects the verifier
    // width so q=1 and q=4 can be compared without rebuilding.
    const int in = 4096, out = 2048, n_experts = 8, n_used = 6;
    const int ntok = argc > 1 ? std::atoi(argv[1]) : 4;
    const bool fp3 = argc > 2 && std::strcmp(argv[2], "q3") == 0;
    const bool fixed_levels = argc > 3 && std::strcmp(argv[3], "fixed") == 0;
    if (ntok <= 0 || ntok > 16 ||
        (argc > 2 && !fp3 && std::strcmp(argv[2], "q2") != 0) ||
        (argc > 3 && !fixed_levels && std::strcmp(argv[3], "learned") != 0)) {
        std::fprintf(stderr, "usage: %s [tokens:1..16] [q2|q3] [learned|fixed]\n", argv[0]);
        return 2;
    }
    const int block_bytes = fp3 ? 14 : 10;
    const int levels = fp3 ? 8 : 4;
    const int nb = in / QK;
    const size_t rows_bytes = (size_t) out * nb * block_bytes;

    std::vector<uint8_t> w(rows_bytes * n_experts);
    for (auto & b : w) b = (uint8_t) rnd();
    for (size_t blk = 0; blk < w.size() / block_bytes; ++blk) {
        w[blk * block_bytes + block_bytes - 2] = 0x30;
        w[blk * block_bytes + block_bytes - 1] = 0x30;
    }
    std::vector<uint16_t> books((size_t) n_experts * 2 * levels);
    for (size_t i = 0; i < books.size(); ++i) books[i] = f32_to_bf16(-0.5f + 0.2f * (float) (i % 5));
    std::vector<uint8_t> modes(n_experts, fixed_levels ? 0 : 1);

    void * d_up = nullptr, * d_gate = nullptr;
    float * d_x = nullptr, * d_a = nullptr, * d_b = nullptr;
    int32_t * d_ids = nullptr;
    HIP_OK(hipMalloc(&d_up, w.size()));
    HIP_OK(hipMalloc(&d_gate, w.size()));
    HIP_OK(hipMemcpy(d_up, w.data(), w.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_gate, w.data(), w.size(), hipMemcpyHostToDevice));
    const size_t yn = (size_t) out * n_used * ntok;
    HIP_OK(hipMalloc(&d_x, sizeof(float) * (size_t) in * ntok));
    HIP_OK(hipMalloc(&d_a, sizeof(float) * yn));
    HIP_OK(hipMalloc(&d_b, sizeof(float) * yn));
    HIP_OK(hipMalloc(&d_ids, sizeof(int32_t) * n_used * ntok));
    std::vector<float> xh((size_t) in * ntok);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = -0.4f + 0.01f * (float) (i % 71);
    HIP_OK(hipMemcpy(d_x, xh.data(), sizeof(float) * xh.size(), hipMemcpyHostToDevice));
    std::vector<int32_t> idsh(n_used * ntok);
    for (int i = 0; i < n_used * ntok; ++i) idsh[i] = i % n_experts;
    HIP_OK(hipMemcpy(d_ids, idsh.data(), sizeof(int32_t) * idsh.size(), hipMemcpyHostToDevice));

    const auto register_mix = fp3 ? ggml_cuda_rocmfp3_mix_register_host
                                  : ggml_cuda_rocmfp2_mix_register_host;
    const auto unregister_mix = fp3 ? ggml_cuda_rocmfp3_mix_unregister
                                    : ggml_cuda_rocmfp2_mix_unregister;
    const auto mul_mat_id = fp3 ? ggml_cuda_rocmfp3_mix_mul_mat_id
                                : ggml_cuda_rocmfp2_mix_mul_mat_id;
    const auto mul_mat_id_glu = fp3 ? ggml_cuda_rocmfp3_mix_mul_mat_id_glu
                                    : ggml_cuda_rocmfp2_mix_mul_mat_id_glu;

    if (!register_mix(
            d_up, rows_bytes, n_experts, out, in,
            books.data(), modes.data())) {
        std::fprintf(stderr, "FAIL: mixed-tensor registration failed\n");
        return 1;
    }
    if (!register_mix(
            d_gate, rows_bytes, n_experts, out, in,
            books.data(), modes.data())) {
        unregister_mix(d_up);
        std::fprintf(stderr, "FAIL: mixed-tensor registration failed\n");
        return 1;
    }

    const int64_t ids_s0 = 1, ids_s1 = n_used;
    const int64_t src1_s1 = 0, src1_s2 = in;
    const int64_t dst_s1 = out, dst_s2 = (int64_t) out * n_used;

    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));

    const int WARM = 50, ITERS = 300, REPS = 7;

    auto time_unfused = [&]() {
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        for (int i = 0; i < WARM; ++i) {
            mul_mat_id(d_up, d_x, d_ids, d_a, in, out, n_used, ntok, 1,
                ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, stream);
            mul_mat_id(d_gate, d_x, d_ids, d_b, in, out, n_used, ntok, 1,
                ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, stream);
        }
        hipStreamSynchronize(stream);
        hipEventRecord(a, stream);
        for (int i = 0; i < ITERS; ++i) {
            mul_mat_id(d_up, d_x, d_ids, d_a, in, out, n_used, ntok, 1,
                ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, stream);
            mul_mat_id(d_gate, d_x, d_ids, d_b, in, out, n_used, ntok, 1,
                ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, stream);
        }
        hipEventRecord(b, stream);
        hipEventSynchronize(b);
        float ms = 0.0f; hipEventElapsedTime(&ms, a, b);
        hipEventDestroy(a); hipEventDestroy(b);
        return (double) ms / ITERS;
    };
    auto time_fused = [&]() {
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        for (int i = 0; i < WARM; ++i) {
            mul_mat_id_glu(d_up, d_gate, d_x, d_ids, d_a, in, out, n_used,
                ntok, 1, ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, 7.0f, stream);
        }
        hipStreamSynchronize(stream);
        hipEventRecord(a, stream);
        for (int i = 0; i < ITERS; ++i) {
            mul_mat_id_glu(d_up, d_gate, d_x, d_ids, d_a, in, out, n_used,
                ntok, 1, ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, 7.0f, stream);
        }
        hipEventRecord(b, stream);
        hipEventSynchronize(b);
        float ms = 0.0f; hipEventElapsedTime(&ms, a, b);
        hipEventDestroy(a); hipEventDestroy(b);
        return (double) ms / ITERS;
    };

    // INTERLEAVED reps, not blocked: this box drifts ~1% over a sequential run, and a blocked
    // design confounds that drift with the effect (a null control proved it on the tok/s A/Bs).
    std::vector<double> u, f;
    for (int r = 0; r < REPS; ++r) { u.push_back(time_unfused()); f.push_back(time_fused()); }

    const double mu = median(u), mf = median(f);
    std::fprintf(stderr, "geometry: qtype=%s levels=%s in=%d out=%d top_k=%d ntok=%d  "
                         "(%d iters x %d interleaved reps)\n",
                 fp3 ? "q3-mix" : "q2-mix", fixed_levels ? "fixed" : "learned",
                 in, out, n_used, ntok, ITERS, REPS);
    std::fprintf(stderr, "  unfused (2 launches, swiglu NOT counted): %8.4f ms/step  [",
                 mu);
    for (double v : u) std::fprintf(stderr, " %.4f", v);
    const int fused_launches = fp3 || ntok <= 2 ? 1 : 2;
    std::fprintf(stderr, " ]\n  fused   (%d launch%s, swiglu included): %8.4f ms/step  [",
                 fused_launches, fused_launches == 1 ? " " : "es", mf);
    for (double v : f) std::fprintf(stderr, " %.4f", v);
    std::fprintf(stderr, " ]\n");
    std::fprintf(stderr, "  fused is %+.2f%% vs unfused (negative = faster)\n", 100.0 * (mf / mu - 1.0));
    std::fprintf(stderr, "  per-layer saving %.4f ms -> over 43 layers %.3f ms/step\n",
                 mu - mf, (mu - mf) * 43.0);

    // Hash both raw projections and the fused output after timing. This catches
    // mode-specific numerical drift even when the generated-token hash happens
    // to remain unchanged.
    if (!mul_mat_id(d_up, d_x, d_ids, d_a, in, out, n_used, ntok, 1,
            ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, stream) ||
        !mul_mat_id(d_gate, d_x, d_ids, d_b, in, out, n_used, ntok, 1,
            ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, stream)) {
        std::fprintf(stderr, "FAIL: raw projection dispatch rejected\n");
        return 1;
    }
    HIP_OK(hipStreamSynchronize(stream));
    std::vector<float> up_result(yn), gate_result(yn), fused_result(yn);
    HIP_OK(hipMemcpy(up_result.data(), d_a, sizeof(float) * yn, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(gate_result.data(), d_b, sizeof(float) * yn, hipMemcpyDeviceToHost));
    if (!mul_mat_id_glu(d_up, d_gate, d_x, d_ids, d_a, in, out, n_used,
            ntok, 1, ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2, 7.0f, stream)) {
        std::fprintf(stderr, "FAIL: fused projection dispatch rejected\n");
        return 1;
    }
    HIP_OK(hipStreamSynchronize(stream));
    HIP_OK(hipMemcpy(fused_result.data(), d_a, sizeof(float) * yn, hipMemcpyDeviceToHost));
    std::fprintf(stderr, "  raw up/gate fnv1a64: %016llx %016llx\n",
                 (unsigned long long) fnv1a64(up_result),
                 (unsigned long long) fnv1a64(gate_result));
    std::fprintf(stderr, "  result fnv1a64: %016llx\n",
                 (unsigned long long) fnv1a64(fused_result));

    unregister_mix(d_gate);
    unregister_mix(d_up);
    hipStreamDestroy(stream);
    HIP_OK(hipFree(d_up)); HIP_OK(hipFree(d_gate)); HIP_OK(hipFree(d_x));
    HIP_OK(hipFree(d_a));  HIP_OK(hipFree(d_b));    HIP_OK(hipFree(d_ids));
    return 0;
}
