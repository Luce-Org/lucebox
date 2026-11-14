// qwen35 applies the batch-invariant matmul switch (LUCE_BATCH_INVARIANT=1) through
// luce::common::target_graph_compute, the call every qwen35 target graph (plain decode, chain/tree verify,
// concurrent step) goes through. With the variable set, each column of a T-column product must be
// byte-identical to the one-column product of that column, for every verify width up to the draft width
// (16 for DFlash2, 22 for the DDTree budget), not only up to MMVQ_MAX_BATCH_SIZE (8). The one-column product
// is plain decode, so it goes through target_graph_compute as well: with the variable set, a decode row runs
// on MMQ like every other width. SKIP (77) without a GPU.
#include "common/cuda_graph_overrides.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using luce::common::batch_invariant_from_env;

std::vector<float> product(ggml_backend_t gpu, ggml_type type, const std::vector<uint8_t> & wq,
                           const std::vector<float> & x, int64_t K, int64_t M, int T) {
    ggml_init_params ip{ 64 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(c, type, K, M);
    ggml_tensor * src1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, T);
    ggml_set_input(w); ggml_set_input(src1);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_tensor * o = ggml_mul_mat(c, w, src1);
    ggml_set_output(o); ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(src1, x.data(), 0, x.size() * sizeof(float));
    const ggml_status st = luce::common::target_graph_compute(gpu, gf);
    std::vector<float> out((size_t) T * M);
    if (st == GGML_STATUS_SUCCESS) ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
    else out.clear();
    ggml_backend_buffer_free(buf); ggml_free(c);
    return out;
}

}  // namespace

int main() {
    if (batch_invariant_from_env(nullptr) || batch_invariant_from_env("") || batch_invariant_from_env("0") ||
        batch_invariant_from_env("10") || !batch_invariant_from_env("1")) {
        std::printf("[qwen35-batch-invariant] FAIL: env parse\n");
        return 1;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::printf("[qwen35-batch-invariant] SKIP: no GPU\n"); return 77; }
    // "off" leaves the variable unset: the same widths then take MMQ and the test shows the switch matters.
    const bool on = !(std::getenv("LUCE_TEST_INVARIANT_OFF") != nullptr);
    if (on) setenv("LUCE_BATCH_INVARIANT", "1", 1); else unsetenv("LUCE_BATCH_INVARIANT");
    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0 };
    const int widths[] = { 2, 4, 8, 9, 12, 16, 22 };
    const int64_t K = 2560, M = 1024;
    uint32_t r = 7u;
    auto rnd = [&r]() { r = r * 1664525u + 1013904223u; return (float) (r >> 8) / (float) (1u << 24) * 2.0f - 1.0f; };
    int bad = 0, cases = 0;
    for (ggml_type type : types) {
        std::vector<float> wf((size_t) K * M);
        for (float & v : wf) v = rnd();
        std::vector<uint8_t> wq(ggml_row_size(type, K) * (size_t) M);
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, M, K, nullptr);
        for (int T : widths) {
            std::vector<float> x((size_t) K * T);
            for (float & v : x) v = rnd();
            const std::vector<float> got = product(gpu, type, wq, x, K, M, T);
            bool same = got.size() == (size_t) T * M;
            for (int j = 0; same && j < T; ++j) {
                const std::vector<float> col(x.begin() + (size_t) j * K, x.begin() + (size_t) (j + 1) * K);
                const std::vector<float> want = product(gpu, type, wq, col, K, M, 1);
                same = want.size() == (size_t) M && std::memcmp(want.data(), got.data() + (size_t) j * M, M * sizeof(float)) == 0;
            }
            ++cases;
            if (!same) { ++bad; std::printf("[qwen35-batch-invariant] DIFF %s T=%d\n", ggml_type_name(type), T); }
        }
    }
    std::printf("[qwen35-batch-invariant] invariant=%d %d/%d cases bit-equal to the one-column product\n",
                on ? 1 : 0, cases - bad, cases);
    ggml_backend_free(gpu);
    return bad == 0 ? 0 : 1;
}
