// Correctness test for geometric_extract_draft_topk_cuda (GPU) vs extract_draft_topk (CPU).
//
// The GPU kernel in src/common/geometric_draft_topk_cuda.cu is a drop-in
// replacement for the CPU top-K + online-logsumexp path in ddtree.cpp. This test
// feeds the same random logits to both and asserts the GPU results match the CPU
// reference:
//   - token ids identical (rank by rank, per position)
//   - log-probs within a small bf16/float tolerance
//
// Exact float ties (where two vocab entries share a logit) are vanishingly
// unlikely with random normal logits, but if one does occur the two paths may
// order the tied ids differently; we treat an id swap as OK when the matching
// log-probs are equal within tolerance.
//
// Build: registered in server/CMakeLists.txt under DFLASH27B_TESTS for both the
//        CUDA and HIP backends (the HIP build compiles this via the hip_compat
//        <cuda_runtime.h> shim). Run: ./test_draft_topk_cuda (0 = pass).

#include "../src/common/geometric_draft_topk_cuda.h"
#include "../src/common/ddtree.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using dflash::common::extract_draft_topk;
using dflash::common::geometric_extract_draft_topk_cuda;

namespace {

// Tolerance on log-prob magnitude. CPU uses double-free float exp/log; the GPU
// kernel does the same in f32 with a different reduction order, so small drift
// is expected — 2e-3 is comfortably above observed error and well below the gap
// between distinct top-K logits.
constexpr float kLogProbTol = 2e-3f;

struct Case {
    int   n;
    int   vocab;
    int   K;
    float temp;
};

bool run_case(const Case & c, unsigned seed) {
    const size_t n_logits = (size_t)c.n * c.vocab;
    const size_t n_out    = (size_t)c.n * c.K;

    std::vector<float> h_logits(n_logits);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 4.f);
    for (auto & x : h_logits) x = dist(rng);

    // CPU reference (the production path).
    std::vector<float>   cpu_lp(n_out);
    std::vector<int32_t> cpu_ids(n_out);
    extract_draft_topk(h_logits.data(), c.n, c.vocab, c.K,
                       cpu_lp.data(), cpu_ids.data(), c.temp);

    // GPU kernel: logits must live in device memory.
    float * d_logits = nullptr;
    cudaError_t err = cudaMalloc(&d_logits, n_logits * sizeof(float));
    if (err != cudaSuccess) {
        printf("  cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaMemcpy(d_logits, h_logits.data(), n_logits * sizeof(float),
               cudaMemcpyHostToDevice);

    std::vector<float>   gpu_lp(n_out);
    std::vector<int32_t> gpu_ids(n_out);
    bool ok = geometric_extract_draft_topk_cuda(d_logits, c.n, c.vocab, c.K,
                                      gpu_lp.data(), gpu_ids.data(), c.temp);
    cudaFree(d_logits);

    if (!ok) {
        printf("  FAIL: geometric_extract_draft_topk_cuda returned false\n");
        return false;
    }

    int   id_mismatch = 0;
    int   tie_swap    = 0;
    float max_lp_err  = 0.f;
    for (int r = 0; r < c.n; r++) {
        for (int k = 0; k < c.K; k++) {
            const size_t i = (size_t)r * c.K + k;
            const float lp_err = std::fabs(gpu_lp[i] - cpu_lp[i]);
            max_lp_err = std::fmax(max_lp_err, lp_err);

            if (gpu_ids[i] != cpu_ids[i]) {
                // Accept as a tie reorder only if the log-prob at this rank is
                // identical within tolerance (both paths picked equal-logit
                // entries, just in a different order).
                if (lp_err <= kLogProbTol) {
                    tie_swap++;
                } else {
                    id_mismatch++;
                    if (id_mismatch <= 8) {
                        printf("    id mismatch pos=%d rank=%d gpu=%d(lp=%.5f) "
                               "cpu=%d(lp=%.5f)\n",
                               r, k, gpu_ids[i], gpu_lp[i],
                               cpu_ids[i], cpu_lp[i]);
                    }
                }
            }
        }
    }

    const bool pass = (id_mismatch == 0) && (max_lp_err <= kLogProbTol);
    printf("  [%s] n=%d vocab=%d K=%d temp=%.2f  id_mismatch=%d tie_swap=%d "
           "max_lp_err=%.3e\n",
           pass ? "PASS" : "FAIL", c.n, c.vocab, c.K, c.temp,
           id_mismatch, tie_swap, max_lp_err);
    return pass;
}

}  // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }

    // The kernel supports K up to kMaxK (=8 in geometric_draft_topk_cuda.cu); larger K is
    // handled by a documented CPU fallback (returns false), checked separately.
    const Case cases[] = {
        // Realistic decode shape: Qwen3.5 vocab, small position batch.
        {15,  151936, 8,  1.0f},
        {1,   151936, 8,  1.0f},
        {15,  151936, 8,  0.7f},   // temperature scaling
        {15,  151936, 8,  2.0f},
        // Small/edge shapes to stress the kernel's split-K / tail handling.
        {7,   1024,   8,  1.0f},
        {32,  4096,   8,  1.0f},
        {3,   257,    8,  1.0f},    // vocab barely above K, non-power-of-two
        {1,   151936, 1,  1.0f},    // K=1 (argmax + log_z)
        {15,  151936, 4,  1.0f},
    };

    int failures = 0;
    int idx = 0;
    for (const Case & c : cases) {
        if (!run_case(c, /*seed=*/1234u + idx)) failures++;
        idx++;
    }

    // Fallback contract: K beyond the kernel's supported range must return false
    // (not silently produce wrong output) so the caller can use the CPU path.
    {
        const int n = 4, vocab = 4096, big_K = 64;
        std::vector<float> h(n * vocab, 0.f);
        float * d = nullptr;
        if (cudaMalloc(&d, h.size() * sizeof(float)) == cudaSuccess) {
            cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
            std::vector<float>   lp(n * big_K);
            std::vector<int32_t> ids(n * big_K);
            bool ret = geometric_extract_draft_topk_cuda(d, n, vocab, big_K,
                                               lp.data(), ids.data(), 1.0f);
            cudaFree(d);
            const bool pass = !ret;  // expect false
            printf("  [%s] fallback contract: K=%d (>kMaxK) returned %s\n",
                   pass ? "PASS" : "FAIL", big_K, ret ? "true" : "false");
            if (!pass) failures++;
            idx++;
        }
    }

    if (failures) {
        printf("\nFAILED: %d/%d cases\n", failures, idx);
        return 1;
    }
    printf("\nALL PASS: %d/%d cases\n", idx, idx);
    return 0;
}
