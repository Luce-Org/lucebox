// GGML_OP_DS4_INDEXER_SCORE at Qwen3.8-Flash-Next's QSA shape (4 indexer heads of 128, unit head weights, block
// ratio 4) against the CPU reference, for every GPU specialization the op dispatches on token count (1, 2..8,
// 9..255, >= 256) with a chunk that starts mid-sequence. Visibility (block c < (kv_start + t + 1) / ratio) must
// match exactly; visible scores within F16 rounding.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static std::vector<float> run(ggml_backend_t be, int T, int n_comp, int kv_start, const std::vector<float> & qv,
                              const std::vector<ggml_fp16_t> & kv, bool masked = false, bool tokenwise = false) {
    ggml_init_params ip{64 * 1024 * 1024, nullptr, true};
    ggml_context * c = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 4, T);
    ggml_tensor * w = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, T);
    ggml_tensor * k = ggml_new_tensor_2d(c, GGML_TYPE_F16, 128, n_comp);
    ggml_tensor * mask = masked ? ggml_new_tensor_2d(c, GGML_TYPE_F32, n_comp, T) : nullptr;
    ggml_tensor * s = tokenwise ? ggml_ds4_indexer_score_tokenwise(c, q, w, k, mask, 4)
                    : mask      ? ggml_ds4_indexer_score_masked(c, q, w, k, mask, 0, 4)
                                : ggml_ds4_indexer_score(c, q, w, k, kv_start, 4);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, s);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    std::vector<float> ones((size_t) 4 * T, 1.0f);
    ggml_backend_tensor_set(w, ones.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(k, kv.data(), 0, ggml_nbytes(k));
    if (mask) {
        std::vector<float> mv((size_t) n_comp * T, -INFINITY);
        for (int t = 0; t < T; ++t) {
            std::fill_n(mv.begin() + (size_t) t * n_comp, std::min(n_comp, (kv_start + t + 1) / 4), 0.0f);
        }
        ggml_backend_tensor_set(mask, mv.data(), 0, ggml_nbytes(mask));
    }
    std::vector<float> out;
    if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
        out.resize((size_t) n_comp * T);
        ggml_backend_tensor_get(s, out.data(), 0, ggml_nbytes(s));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return out;
}

int main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) return 77;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    const int n_comp = 1500, kv_start = 5900;
    bool ok = true;
    for (int T : {1, 2, 4, 8, 16, 100, 128, 256, 512}) {
        std::vector<float> qv((size_t) 128 * 4 * T);
        std::vector<ggml_fp16_t> kv((size_t) 128 * n_comp);
        for (size_t i = 0; i < qv.size(); ++i) qv[i] = std::sin(0.37f * (float) i) * 0.5f;
        for (size_t i = 0; i < kv.size(); ++i) kv[i] = ggml_fp32_to_fp16(std::cos(0.11f * (float) i) * 0.5f);
        const std::vector<float> want = run(cpu, T, n_comp, kv_start, qv, kv);
        const std::vector<float> got = run(gpu, T, n_comp, kv_start, qv, kv);
        int vis_mismatch = 0;
        double max_rel = 0.0;
        for (size_t i = 0; i < want.size() && got.size() == want.size(); ++i) {
            const bool vw = want[i] > -1.0e20f, vg = got[i] > -1.0e20f;
            if (vw != vg) { ++vis_mismatch; continue; }
            if (vw) max_rel = std::max(max_rel, std::fabs((double) got[i] - want[i]) / (std::fabs((double) want[i]) + 1.0));
        }
        const bool pass = got.size() == want.size() && vis_mismatch == 0 && max_rel < 2e-2;
        std::printf("indexer-score T=%-4d visibility_mismatches=%d max_rel_err=%.2e %s\n", T, vis_mismatch, max_rel,
                    pass ? "OK" : "FAIL");
        ok = ok && pass;
    }
    // Fixed bucket vs exact width on the SAME backend: require score bits,
    // including at partial WMMA tiles, and poison all invisible key columns.
    for (int capacity : {576, 1088, 2112, 4160, 8256, 65536}) {
        std::vector<float> qv(128 * 4);
        for (size_t i = 0; i < qv.size(); ++i) qv[i] = std::sin(0.37f * (float) i) * 0.5f;
        for (int n : {capacity - 64, capacity - 63, capacity - 1, capacity}) {
            std::vector<ggml_fp16_t> kv((size_t) 128 * capacity);
            for (size_t i = 0; i < kv.size(); ++i) kv[i] = ggml_fp32_to_fp16(i < (size_t) 128 * n
                ? std::cos(0.11f * (float) i) * 0.5f : std::numeric_limits<float>::quiet_NaN());
            const auto want = run(gpu, 1, n, 4 * n - 1, qv, kv);
            const auto got = run(gpu, 1, capacity, 4 * n - 1, qv, kv, true);
            bool pass = want.size() == (size_t) n && got.size() == (size_t) capacity &&
                std::memcmp(want.data(), got.data(), n * sizeof(float)) == 0;
            for (int b = n; pass && b < capacity; ++b) pass = got[b] == -1.0e30f;
            std::printf("indexer-score masked capacity=%d valid=%d bits %s\n", capacity, n, pass ? "OK" : "FAIL");
            ok &= pass;
        }
    }
    // Tokenwise rows (stable verify scoring every row at once) against one-token masked calls on the
    // SAME backend: score bits per row, at each multi-token width the plain op would specialize.
    for (int T : {2, 3, 4, 5, 6, 8}) {
        const int n = 4160, start = 4 * n - 4 * T - 9;
        std::vector<float> qv((size_t) 128 * 4 * T);
        std::vector<ggml_fp16_t> kv((size_t) 128 * n);
        for (size_t i = 0; i < qv.size(); ++i) qv[i] = std::sin(0.37f * (float) i) * 0.5f;
        for (size_t i = 0; i < kv.size(); ++i) kv[i] = ggml_fp32_to_fp16(std::cos(0.11f * (float) i) * 0.5f);
        const auto rows = run(gpu, T, n, start, qv, kv, true, true);
        bool pass = rows.size() == (size_t) n * T;
        for (int t = 0; pass && t < T; ++t) {
            const std::vector<float> qt(qv.begin() + (size_t) t * 512, qv.begin() + (size_t) (t + 1) * 512);
            const auto one = run(gpu, 1, n, start + t, qt, kv, true);
            pass = one.size() == (size_t) n && std::memcmp(one.data(), rows.data() + (size_t) t * n, n * sizeof(float)) == 0;
        }
        std::printf("indexer-score tokenwise T=%d vs one-token bits %s\n", T, pass ? "OK" : "FAIL");
        ok &= pass;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
