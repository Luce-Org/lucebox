// Correctness tests for geometric_sample_logits_cuda (src/common/geometric_sampler_cuda.cu)
// vs the CPU sample_logits chain (src/common/sampler.h/.cpp). CUDA only — the GPU
// sampler is compiled into dflash_common solely on the cuda backend (DFLASH_GPU_SAMPLER,
// default ON). All tests self-skip at runtime when no CUDA device is present.
//
// Build: registered in server/CMakeLists.txt under DFLASH27B_TESTS (CUDA only).
// Run:   ./test_gpu_sampler_cuda   (exit 0 = pass, non-zero = fail)

#include "common/sampler.h"
#include "common/geometric_sampler_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>

using namespace dflash::common;

// ─── Test framework (ds4 style) ────────────────────────────────────────

static int test_failures = 0;
static int test_count = 0;

#define TEST_ASSERT_MSG(expr, msg) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s — %s\n", __FILE__, __LINE__, #expr, msg); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::fprintf(stderr, "  %s ...", #fn); \
    int before = test_failures; \
    fn(); \
    if (test_failures == before) std::fprintf(stderr, " ok\n"); \
    else std::fprintf(stderr, "\n"); \
} while (0)

// ═══════════════════════════════════════════════════════════════════════
// GPU sampler (geometric_sampler_cuda.cu) — compared against the CPU chain.
// All tests self-skip when the build has no GPU sampler or no CUDA device.
// ═══════════════════════════════════════════════════════════════════════

static bool gpu_sampler_test_available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return false; }
    return n > 0;
}

// Deterministic logits: a smooth but non-monotonic spread over the vocab so the
// argmax is unambiguous and the softmax has spread-out mass.
static std::vector<float> gpu_test_logits(int vocab, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-6.0f, 6.0f);
    std::vector<float> v(vocab);
    for (auto & x : v) x = u(rng);
    return v;
}

// Analytic softmax over `logits` at temperature `temp`.
static std::vector<double> cpu_softmax(const std::vector<float> & logits, float temp) {
    const double inv_t = 1.0 / std::max(1e-3f, temp);
    double mx = -1e300;
    for (float l : logits) mx = std::max(mx, (double)l * inv_t);
    std::vector<double> p(logits.size());
    double z = 0.0;
    for (size_t i = 0; i < logits.size(); i++) { p[i] = std::exp((double)logits[i] * inv_t - mx); z += p[i]; }
    for (auto & x : p) x /= z;
    return p;
}

// Analytic nucleus (top_p) distribution: the full softmax, restricted to the
// smallest descending-probability prefix whose cumulative mass reaches `top_p`,
// then renormalized within that nucleus (mass 0 elsewhere). This is the exact
// distribution the CPU nucleus_cutoff + draw is supposed to reproduce.
static std::vector<double> cpu_top_p_dist(const std::vector<float> & logits, float temp, double top_p) {
    auto p = cpu_softmax(logits, temp);
    std::vector<int> idx(p.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return p[a] > p[b]; });
    std::vector<double> q(p.size(), 0.0);
    double cum = 0.0, znuc = 0.0;
    std::vector<int> keep;
    for (int i : idx) { keep.push_back(i); cum += p[i]; if (cum >= top_p) break; }
    for (int i : keep) znuc += p[i];
    for (int i : keep) q[i] = p[i] / znuc;
    return q;
}

// Greedy (temp=0): GPU must pick exactly the same token as the CPU chain, with
// and without penalties active.
static void test_gpu_sampler_greedy_matches_cpu() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, " (skip: no CUDA) "); return; }
    const int vocab = 257;  // not a multiple of the block size on purpose
    for (uint64_t seed = 1; seed <= 6; seed++) {
        auto logits = gpu_test_logits(vocab, seed);
        SamplerCfg cfg;  // temp=0 → greedy
        std::vector<int32_t> history;
        std::mt19937_64 rng(seed);
        const int cpu_tok = sample_logits(logits.data(), vocab, cfg, history, rng);
        const int gpu_tok = geometric_sample_logits_cuda(logits.data(), vocab, cfg, history,
                                               0.0, /*on_device=*/false);
        TEST_ASSERT_MSG(gpu_tok == cpu_tok, "greedy GPU token must equal CPU argmax");
    }
}

static void test_gpu_sampler_greedy_penalties_match_cpu() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, " (skip: no CUDA) "); return; }
    const int vocab = 200;
    auto logits = gpu_test_logits(vocab, 99);
    // A history that makes penalties bite on what would otherwise be the argmax.
    std::vector<int32_t> history = {3, 3, 7, 12, 7, 50, 50, 50, 100};
    struct { float rep, freq, pres; } cases[] = {
        {1.5f, 0.0f, 0.0f},
        {1.0f, 0.8f, 0.0f},
        {1.0f, 0.0f, 1.2f},
        {1.3f, 0.5f, 0.7f},
    };
    for (auto c : cases) {
        SamplerCfg cfg;  // temp=0 → greedy after penalties
        cfg.rep_pen = c.rep; cfg.freq_pen = c.freq; cfg.pres_pen = c.pres;
        std::mt19937_64 rng(7);
        const int cpu_tok = sample_logits(logits.data(), vocab, cfg, history, rng);
        const int gpu_tok = geometric_sample_logits_cuda(logits.data(), vocab, cfg, history,
                                               0.0, /*on_device=*/false);
        TEST_ASSERT_MSG(gpu_tok == cpu_tok, "greedy+penalties GPU token must equal CPU");
    }
}

// top_k>0 is intentionally CPU-only: the GPU entry must signal fallback (-1).
static void test_gpu_sampler_top_k_falls_back() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, " (skip: no CUDA) "); return; }
    const int vocab = 128;
    auto logits = gpu_test_logits(vocab, 5);
    SamplerCfg cfg; cfg.temp = 1.0f; cfg.top_k = 10;
    std::vector<int32_t> history;
    const int gpu_tok = geometric_sample_logits_cuda(logits.data(), vocab, cfg, history,
                                           0.3, /*on_device=*/false);
    TEST_ASSERT_MSG(gpu_tok == -1, "top_k>0 must return -1 (CPU fallback)");
}

// L1 distance between an empirical histogram of GPU draws and the analytic
// distribution. Pulls the uniform from the same RNG family the CPU path uses.
static double gpu_empirical_l1(const std::vector<float> & logits, const SamplerCfg & cfg,
                               const std::vector<double> & analytic, int n_draws) {
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int> hist(logits.size(), 0);
    std::vector<int32_t> history;
    int valid = 0;
    for (int i = 0; i < n_draws; i++) {
        const int tok = geometric_sample_logits_cuda(logits.data(), (int)logits.size(), cfg,
                                           history, u(rng), /*on_device=*/false);
        if (tok >= 0 && tok < (int)logits.size()) { hist[tok]++; valid++; }
    }
    if (valid == 0) return 1e9;
    double l1 = 0.0;
    for (size_t k = 0; k < logits.size(); k++)
        l1 += std::fabs((double)hist[k] / valid - analytic[k]);
    return l1;
}

// Temperature sampling (no truncation): the GPU draw distribution must match the
// analytic softmax — i.e. the same distribution the CPU multinomial samples.
static void test_gpu_sampler_temperature_distribution() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, " (skip: no CUDA) "); return; }
    const int vocab = 24;
    auto logits = gpu_test_logits(vocab, 42);
    SamplerCfg cfg; cfg.temp = 1.0f;  // full vocab
    auto analytic = cpu_softmax(logits, cfg.temp);
    const double l1 = gpu_empirical_l1(logits, cfg, analytic, 120000);
    TEST_ASSERT_MSG(l1 < 0.03, "GPU temp-sample dist must match analytic softmax (L1<0.03)");
}

// top_p in (0,1) is intentionally unimplemented on the GPU (removed; see
// geometric_sampler_cuda.h): the GPU entry must signal fallback (-1), same
// contract as top_k>0.
static void test_gpu_sampler_top_p_falls_back() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, " (skip: no CUDA) "); return; }
    const int vocab = 128;
    auto logits = gpu_test_logits(vocab, 5);
    SamplerCfg cfg; cfg.temp = 1.0f; cfg.top_p = 0.8f;
    std::vector<int32_t> history;
    const int gpu_tok = geometric_sample_logits_cuda(logits.data(), vocab, cfg, history,
                                           0.3, /*on_device=*/false);
    TEST_ASSERT_MSG(gpu_tok == -1, "top_p in (0,1) must return -1 (CPU fallback)");
}

// geometric_compute_probs_cuda is the shared prefix behind the GPU-assisted
// top_p path: it must return the same penalty+softmax(temp) probability vector
// the CPU chain computes. Checks a plain-temp config and one with penalties.
static void test_gpu_compute_probs_matches_softmax() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, " (skip: no CUDA) "); return; }
    const int vocab = 300;  // not a multiple of the block size on purpose
    auto logits = gpu_test_logits(vocab, 21);

    struct { float temp, rep, freq, pres; std::vector<int32_t> hist; } cases[] = {
        {0.8f, 1.0f, 0.0f, 0.0f, {}},
        {1.0f, 1.4f, 0.3f, 0.5f, {3, 3, 7, 12, 7, 50, 50, 100, 100, 100}},
    };
    for (auto & c : cases) {
        SamplerCfg cfg; cfg.temp = c.temp; cfg.rep_pen = c.rep;
        cfg.freq_pen = c.freq; cfg.pres_pen = c.pres;
        std::vector<float> probs(vocab);
        const bool ok = geometric_compute_probs_cuda(logits.data(), vocab, cfg, c.hist,
                                           probs.data(), /*on_device=*/false);
        TEST_ASSERT_MSG(ok, "geometric_compute_probs_cuda must succeed for temp>0");
        if (!ok) continue;
        // Reference: apply the same penalties on the CPU, then softmax(temp).
        std::vector<float> penalized = logits;
        if (!c.hist.empty()) {
            std::unordered_map<int, int> counts;
            for (int id : c.hist) counts[id]++;
            for (auto & kv : counts) {
                if (kv.first < 0 || kv.first >= vocab) continue;
                float & v = penalized[kv.first];
                if (cfg.rep_pen > 1.0f) v = (v > 0.0f) ? v / cfg.rep_pen : v * cfg.rep_pen;
                v -= cfg.freq_pen * kv.second + cfg.pres_pen;
            }
        }
        auto analytic = cpu_softmax(penalized, cfg.temp);
        double l1 = 0.0, sum = 0.0;
        for (int i = 0; i < vocab; i++) { l1 += std::fabs((double)probs[i] - analytic[i]); sum += probs[i]; }
        TEST_ASSERT_MSG(l1 < 1e-3, "GPU probs must match CPU penalty+softmax (L1<1e-3)");
        TEST_ASSERT_MSG(std::fabs(sum - 1.0) < 1e-3, "GPU probs must sum to ~1");
    }
}

// GPU-assisted top_p: sample_logits routes pure top_p (no top_k) through
// geometric_compute_probs_cuda + the CPU nucleus_cutoff/draw. The empirical
// draw distribution must match the analytic nucleus distribution — this is the
// only end-to-end check of nucleus_cutoff and sample_from_gpu_probs.
static void test_gpu_sampler_top_p_distribution() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, " (skip: no CUDA) "); return; }
    const int vocab = 48;
    auto logits = gpu_test_logits(vocab, 7);
    SamplerCfg cfg; cfg.temp = 1.0f; cfg.top_p = 0.9f;
    std::vector<int32_t> history;
    auto analytic = cpu_top_p_dist(logits, cfg.temp, cfg.top_p);

    std::mt19937_64 rng(2024);
    std::vector<int> hist(vocab, 0);
    const int n = 120000;
    int valid = 0;
    for (int i = 0; i < n; i++) {
        const int tok = sample_logits(logits.data(), vocab, cfg, history, rng);
        if (tok >= 0 && tok < vocab) { hist[tok]++; valid++; }
    }
    TEST_ASSERT_MSG(valid > 0, "top_p sampling must produce valid draws");
    double l1 = 0.0;
    for (int k = 0; valid > 0 && k < vocab; k++)
        l1 += std::fabs((double)hist[k] / valid - analytic[k]);
    TEST_ASSERT_MSG(l1 < 0.03, "GPU-assisted top_p dist must match analytic nucleus (L1<0.03)");
}

// Direct CPU-vs-GPU agreement on the same draws is not bit-exact (different
// inverse-CDF orderings), but both must agree on the *argmax* of their empirical
// histograms — the most-likely token — under temperature sampling.
static void test_gpu_sampler_modal_token_matches_cpu() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, " (skip: no CUDA) "); return; }
    const int vocab = 32;
    auto logits = gpu_test_logits(vocab, 11);
    SamplerCfg cfg; cfg.temp = 0.9f;
    std::vector<int32_t> history;

    auto modal = [&](bool gpu) {
        std::mt19937_64 rng(555);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<int> hist(vocab, 0);
        for (int i = 0; i < 60000; i++) {
            const int tok = gpu
                ? geometric_sample_logits_cuda(logits.data(), vocab, cfg, history, u(rng), false)
                : sample_logits(logits.data(), vocab, cfg, history, rng);
            if (tok >= 0 && tok < vocab) hist[tok]++;
        }
        return (int)(std::max_element(hist.begin(), hist.end()) - hist.begin());
    };
    TEST_ASSERT_MSG(modal(true) == modal(false), "GPU and CPU agree on the modal token");
}

// Per-call latency microbench (gated by env DFLASH_SAMPLER_BENCH=1). Isolates
// the three regimes that explain the end-to-end numbers: the CPU chain, the GPU
// path fed host logits (pays a full-vocab H2D every call), and the GPU path fed
// a device pointer (the integrated path that skips the copy).
static void gpu_sampler_microbench() {
    if (!gpu_sampler_test_available()) { std::fprintf(stderr, "[microbench] no CUDA device\n"); return; }
    const int vocab = 151936;          // Qwen3 vocab
    const int iters = 1000;
    auto logits = gpu_test_logits(vocab, 1);
    std::vector<int32_t> history;

    auto now = []{ return std::chrono::steady_clock::now(); };
    auto us  = [](auto a, auto b){ return std::chrono::duration<double, std::micro>(b - a).count(); };

    // Persistent device copy for the device-pointer regime (no per-call H2D).
    float * d_logits = nullptr;
    bool have_dev = cudaMalloc(&d_logits, (size_t)vocab * sizeof(float)) == cudaSuccess &&
                    cudaMemcpy(d_logits, logits.data(), (size_t)vocab * sizeof(float),
                               cudaMemcpyHostToDevice) == cudaSuccess;

    volatile long sink = 0;
    auto bench = [&](const char * label, const SamplerCfg & cfg) {
        std::mt19937_64 rng(1);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int i = 0; i < 30; i++) sink += sample_logits(logits.data(), vocab, cfg, history, rng);
        for (int i = 0; i < 30; i++) sink += geometric_sample_logits_cuda(logits.data(), vocab, cfg, history, u(rng), false);
        auto t0 = now();
        for (int i = 0; i < iters; i++) sink += sample_logits(logits.data(), vocab, cfg, history, rng);
        auto t1 = now();
        for (int i = 0; i < iters; i++) sink += geometric_sample_logits_cuda(logits.data(), vocab, cfg, history, u(rng), false);
        auto t2 = now();
        double dev_us = -1.0;
        if (have_dev) {
            for (int i = 0; i < 30; i++) sink += geometric_sample_logits_cuda(d_logits, vocab, cfg, history, u(rng), true);
            auto t3 = now();
            for (int i = 0; i < iters; i++) sink += geometric_sample_logits_cuda(d_logits, vocab, cfg, history, u(rng), true);
            auto t4 = now();
            dev_us = us(t3, t4) / iters;
        }
        const double cpu_us  = us(t0, t1) / iters;
        const double gpuh_us = us(t1, t2) / iters;
        std::fprintf(stderr,
            "  %-22s CPU %8.2f us | GPU+H2D %8.2f us (%.2fx) | GPU devptr %8.2f us (%.2fx)\n",
            label, cpu_us, gpuh_us, cpu_us / gpuh_us,
            dev_us, dev_us > 0 ? cpu_us / dev_us : 0.0);
    };

    std::fprintf(stderr, "[microbench] vocab=%d iters=%d (per call; >1.0x = GPU faster)\n", vocab, iters);
    { SamplerCfg c;                                   bench("greedy (temp=0)", c); }
    { SamplerCfg c; c.temp = 0.8f;                    bench("temp=0.8 (full vocab)", c); }
    // top_p is unimplemented on the GPU (always falls back to CPU), so it's not benched here.
    { SamplerCfg c; c.temp = 0.8f; c.rep_pen = 1.2f;
      std::vector<int32_t> h; for (int i = 0; i < 200; i++) h.push_back(i * 7 % vocab);
      history = h; bench("temp=0.8 rep_pen=1.2", c); history.clear(); }

    if (have_dev) cudaFree(d_logits);
    (void)sink;
}

// ─── main ────────────────────────────────────────────────────────────

int main() {
    if (const char * b = std::getenv("DFLASH_SAMPLER_BENCH"); b && b[0] == '1') {
        gpu_sampler_microbench();
        return 0;
    }

    std::fprintf(stderr, "=== test_gpu_sampler_cuda ===\n");

    RUN_TEST(test_gpu_sampler_greedy_matches_cpu);
    RUN_TEST(test_gpu_sampler_greedy_penalties_match_cpu);
    RUN_TEST(test_gpu_sampler_top_k_falls_back);
    RUN_TEST(test_gpu_sampler_temperature_distribution);
    RUN_TEST(test_gpu_sampler_top_p_falls_back);
    RUN_TEST(test_gpu_compute_probs_matches_softmax);
    RUN_TEST(test_gpu_sampler_top_p_distribution);
    RUN_TEST(test_gpu_sampler_modal_token_matches_cpu);

    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
