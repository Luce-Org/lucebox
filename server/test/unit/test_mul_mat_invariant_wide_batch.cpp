// Batch-invariant mode (ggml_backend_cuda_set_mmvq_batch_invariant, LUCE_BATCH_INVARIANT=1): a packed engine step
// puts prefill cuts (512 tokens) and the other lanes' decode rows in ONE MUL_MAT, and how many rows share a step
// depends on wall-clock scheduling. Contract: every column of an N-column product is byte-identical to that column
// in a narrow product (at most 32 columns), for every N, so a token's bits never depend on what was packed with it.
// The printed hash (column 0) groups the widths that share bits.
// What broke it: MMVQ up to 32 columns and MMQ above give different bits on every device; on RDNA3.5 (gfx1151)
// Q4_K / Q6_K above 256 columns went to dequantize + BLAS GEMM, which picks its kernel by N (the Strix Halo
// "spec decode != AR" divergence of rc3-det); on NVIDIA MMQ stream-k splits the k range by the tile count.
// Shapes: a small one, and the qwen35 27B row lengths (5120, 17408) with row counts that are not a tile multiple.
// Device: LUCE_TEST_DEVICE (default 0). SKIP (77) without a GPU.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

const int k_widths[] = { 1, 2, 3, 8, 9, 16, 17, 31, 32, 33, 48, 49, 64, 65, 128, 255, 256, 257, 320, 384, 512, 513, 514, 1024 };
const int64_t k_shapes[][2] = { { 2560, 1024 }, { 5120, 1000 }, { 17408, 328 } };   // K (row length), M (rows)
// Every weight type of Qwen3.8-27B-UD-IQ4_XS.gguf (its only 2-D F32 weights are ssm_conv1d, not a matmul).
const ggml_type k_types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q3_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_Q8_0,
                              GGML_TYPE_Q2_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL };

// The N-column product w * x[:, :N] in batch-invariant mode.
std::vector<float> product(ggml_backend_t gpu, ggml_type type, const std::vector<uint8_t> & wq,
                           const float * x, int64_t K, int64_t M, int N) {
    ggml_init_params ip{ 64 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(c, type, K, M);
    ggml_tensor * src1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, N);
    ggml_set_input(w); ggml_set_input(src1);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_tensor * o = ggml_mul_mat(c, w, src1);
    ggml_set_output(o); ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(src1, x, 0, (size_t) K * N * sizeof(float));
    const bool prev = ggml_backend_cuda_set_mmvq_batch_invariant(true);
    const bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    ggml_backend_cuda_set_mmvq_batch_invariant(prev);
    std::vector<float> out((size_t) N * M);
    if (ok) ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
    else out.clear();
    ggml_backend_buffer_free(buf); ggml_free(c);
    return out;
}

}  // namespace

int main() {
    const char * dev = std::getenv("LUCE_TEST_DEVICE");
    ggml_backend_t gpu = ggml_backend_cuda_init(dev ? std::atoi(dev) : 0);
    if (!gpu) { std::printf("[invariant-wide-batch] SKIP: no GPU\n"); return 77; }
    const int n_max = k_widths[sizeof(k_widths) / sizeof(k_widths[0]) - 1];
    uint32_t r = 7u;
    auto rnd = [&r]() { r = r * 1664525u + 1013904223u; return (float) (r >> 8) / (float) (1u << 24) * 2.0f - 1.0f; };
    bool ok = true;
    for (const auto & shape : k_shapes) {
    const int64_t K = shape[0], M = shape[1];
    std::vector<float> wf((size_t) K * M), x((size_t) K * n_max);
    for (float & v : wf) v = rnd();
    for (float & v : x) v = rnd();
    for (ggml_type type : k_types) {
        std::vector<uint8_t> wq(ggml_row_size(type, K) * (size_t) M);
        ggml_quantize_init(type);   // IQ2 / IQ3 tables
        const std::vector<float> imatrix((size_t) K, 1.0f);   // IQ2 quantizers require one
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, M, K, imatrix.data());
        std::vector<float> want;   // every column from a product of at most 32 columns
        for (int col = 0; col < n_max; col += 32) {
            const std::vector<float> part = product(gpu, type, wq, x.data() + (size_t) col * K, K, M, std::min(32, n_max - col));
            want.insert(want.end(), part.begin(), part.end());
        }
        for (int N : k_widths) {
            const std::vector<float> got = product(gpu, type, wq, x.data(), K, M, N);
            const bool same = !got.empty() && std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0;
            size_t bad = 0;
            for (size_t i = 0; i < got.size(); ++i) bad += std::memcmp(&got[i], &want[i], sizeof(float)) != 0;
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < (size_t) M && i < got.size(); ++i) { uint32_t b; std::memcpy(&b, &got[i], 4); h = (h ^ b) * 1099511628211ull; }
            std::printf("[invariant-wide-batch] %-6s %5dx%-4d N=%-4d: %s (%zu of %zu floats differ from the narrow products) hash %016llx\n",
                        ggml_type_name(type), (int) K, (int) M, N, same ? "PASS" : "FAIL", bad, got.size(), (unsigned long long) h);
            ok = ok && same;
        }
    }
    }
    ggml_backend_free(gpu);
    std::printf("[invariant-wide-batch] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
