// UMA regression: fallback GEMM must read CUDA_Host inputs, not uninitialized
// device staging. Width 17 crosses RDNA3.5's custom-MMF width-16 boundary.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) return 1;
    const auto dev = ggml_backend_get_device(gpu);
    const auto host_type = ggml_backend_dev_host_buffer_type(dev);
    if (!host_type || !ggml_backend_dev_supports_buft(dev, host_type)) {
        std::puts("SKIP: device does not support pinned host inputs");
        ggml_backend_free(gpu);
        return 0;
    }
    bool ok = true;
    for (int n : {16, 17, 32}) {
        constexpr int k = 256, m = 256;
        auto ctx = ggml_init({ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true});
        auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, k, m);
        auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        auto h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        auto host = ggml_backend_buft_alloc_buffer(host_type, ggml_nbytes(h));
        ggml_backend_tensor_alloc(host, h, ggml_backend_buffer_get_base(host));
        auto y = ggml_mul_mat(ctx, w, x);
        auto z = ggml_mul_mat(ctx, w, h);
        ggml_set_output(y);
        ggml_set_output(z);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        ggml_build_forward_expand(graph, z);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        if (!ggml_gallocr_alloc_graph(alloc, graph)) return 1;
        std::vector<ggml_bf16_t> weights(k*m);
        std::vector<float> input(k*n), a(m*n), b(m*n);
        for (int i = 0; i < k*m; ++i) weights[i] = ggml_fp32_to_bf16((i%31 - 15)*0.03125f);
        for (int i = 0; i < k*n; ++i) input[i] = (i%29 - 14)*0.0625f;
        ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(h, input.data(), 0, ggml_nbytes(h));
        if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS) return 1;
        ggml_backend_tensor_get(y, a.data(), 0, ggml_nbytes(y));
        ggml_backend_tensor_get(z, b.data(), 0, ggml_nbytes(z));
        const bool equal = std::memcmp(a.data(), b.data(), ggml_nbytes(y)) == 0;
        std::printf("BF16 host/device n=%d: %s\n", n, equal ? "PASS" : "FAIL");
        ok &= equal;
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(host);
        ggml_free(ctx);
    }
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
