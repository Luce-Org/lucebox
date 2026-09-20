// Bitwise differential against the unfused GPU RMS_NORM -> SCALE sequence.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool run_case(ggml_backend_t gpu, int tokens, bool strided, int heads = 16) {
    constexpr int width = 128;
    const int channels = heads * (strided ? 5 : 1);
    ggml_init_params ip{16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, channels, tokens);
    ggml_set_input(storage);
    ggml_tensor * x = ggml_view_3d(ctx, storage, width, heads, tokens,
        width * sizeof(float), width * channels * sizeof(float), 0);
    ggml_tensor * reference_norm = ggml_rms_norm(ctx, x, 1e-6f / width);
    ggml_set_output(reference_norm); // Prevent the control branch from fusing.
    ggml_tensor * reference = ggml_scale(ctx, reference_norm, 1.0f / sqrtf(width));
    ggml_tensor * candidate = ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f / width), 1.0f / sqrtf(width));
    ggml_set_output(reference);
    ggml_set_output(candidate);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, candidate);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buf) { ggml_free(ctx); return false; }
    std::vector<float> input(ggml_nelements(storage));
    for (size_t i = 0; i < input.size(); ++i) {
        const int row = i / width;
        const float magnitude[] = {0.0f, 1e-12f, 1e-4f, 1.0f, 1e8f};
        input[i] = sinf(float(i % 7919)) * magnitude[row % 5];
    }
    ggml_backend_tensor_set(storage, input.data(), 0, ggml_nbytes(storage));
    bool ok = ggml_backend_graph_compute(gpu, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> a(ggml_nelements(reference)), b(a.size());
    if (ok) {
        ggml_backend_tensor_get(reference, a.data(), 0, ggml_nbytes(reference));
        ggml_backend_tensor_get(candidate, b.data(), 0, ggml_nbytes(candidate));
        ok = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        if (!ok) {
            size_t mismatches = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                    if (!mismatches) std::printf("first mismatch index=%zu reference=%a candidate=%a\n", i, a[i], b[i]);
                    ++mismatches;
                }
            }
            std::printf("differing elements=%zu\n", mismatches);
        }
    }
    std::printf("rms-scale tokens=%d heads=%d strided=%d elements=%zu %s\n", tokens, heads, strided, a.size(), ok ? "EXACT" : "FAIL");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}
int main() {
#ifdef _WIN32
    _putenv_s("QWEN4EXP_RMS_SCALE_FUSED", "1");
#else
    setenv("QWEN4EXP_RMS_SCALE_FUSED", "1", 1);
#endif
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) return 77;
    bool ok = true;
    for (int tokens : {1, 17, 1024, 16366}) {
        for (bool strided : {false, true}) ok = run_case(gpu, tokens, strided) && ok;
    }
    ok = run_case(gpu, 17, true, 3) && ok; // Nonstandard head count.
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
