// gfx1151 width-4 dispatch regression test.
//
// Monolithic paged DeepSeek4 decode batches up to six lanes through plain
// mul_mat. Two dispatch policies keep the four-lane arithmetic identical to the
// single-lane arithmetic:
//
//   1. Narrow F16 weights (the hyper-connection mix projection, ne01 = 24)
//      stay on MMVF for ne11 <= 8 instead of falling into the rocBLAS F16
//      Tensile kernel, whose only gfx1151 macro-tile is 128x128 and which
//      accumulates in F16.
//   2. ROCmFP4 dense projections at ne11 == 4 and ne11 == 5 use the four- and
//      five-column MMVQ specializations instead of the 16-wide MMQ tile.
//
// For every shape this test compares the N-column result bit for bit with N
// independent single-column computes. Any change that alters the per-column
// accumulation order fails here before it reaches a service measurement.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

uint32_t lcg_state = 0x12345678u;
float lcg_uniform() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float) ((lcg_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

bool compute(ggml_backend_t backend, ggml_type wtype, int64_t k, int64_t m, int64_t n,
             const std::vector<uint8_t> & weight_bytes, const float * input,
             std::vector<float> & out) {
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, k, m);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_input(w);
    ggml_set_input(x);
    ggml_set_output(y);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(w, weight_bytes.data(), 0, weight_bytes.size());
    ggml_backend_tensor_set(x, input, 0, (size_t) k * n * sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    if (ok) {
        out.resize((size_t) m * n);
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

size_t bit_mismatches(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return a.size() + b.size();
    }
    size_t mm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            ++mm;
        }
    }
    return mm;
}

// Compares one N-column compute against N single-column computes.
bool check_columns(ggml_backend_t backend, const char * label, ggml_type wtype,
                   int64_t k, int64_t m, int n, const std::vector<uint8_t> & wbytes,
                   const std::vector<float> & input) {
    std::vector<float> wide;
    if (!compute(backend, wtype, k, m, n, wbytes, input.data(), wide)) {
        std::printf("FAIL %s N=%d: compute failed\n", label, n);
        return false;
    }
    std::vector<float> joined;
    for (int c = 0; c < n; ++c) {
        std::vector<float> one;
        if (!compute(backend, wtype, k, m, 1, wbytes, input.data() + (size_t) k * c, one)) {
            std::printf("FAIL %s N=%d: single-column compute failed\n", label, n);
            return false;
        }
        joined.insert(joined.end(), one.begin(), one.end());
    }
    const size_t mm = bit_mismatches(wide, joined);
    std::printf("%s %s N=%d: %zu values, %zu bit mismatches vs single-column\n",
                mm == 0 ? "PASS" : "FAIL", label, n, wide.size(), mm);
    return mm == 0;
}

bool test_narrow_f16(ggml_backend_t backend) {
    const int64_t K = 16384, M = 24;
    std::vector<uint8_t> wbytes((size_t) K * M * sizeof(ggml_fp16_t));
    ggml_fp16_t * wh = (ggml_fp16_t *) wbytes.data();
    lcg_state = 0x1234567u;
    for (size_t i = 0; i < (size_t) K * M; ++i) {
        wh[i] = ggml_fp32_to_fp16(lcg_uniform() * 0.05f);
    }
    std::vector<float> input((size_t) K * 8);
    for (float & v : input) {
        v = lcg_uniform() * 2.0f;
    }
    bool ok = true;
    for (int n : {2, 3, 4, 5, 6, 8}) {
        ok = check_columns(backend, "f16[16384,24]", GGML_TYPE_F16, K, M, n, wbytes, input) && ok;
    }
    return ok;
}

bool test_rocmfp4_four_columns(ggml_backend_t backend) {
    const ggml_type T = GGML_TYPE_Q4_0_ROCMFP4_FAST;
    const ggml_type_traits * tr = ggml_get_type_traits(T);
    struct Shape { int64_t k, m; const char * label; };
    const Shape shapes[] = {
        {4096, 1024, "rocmfp4[4096,1024]"},   // attn_q_a, attn_output_a, compressor width
        {1024, 4096, "rocmfp4[1024,4096]"},   // attn_output_b
        {4096, 512, "rocmfp4[4096,512]"},     // attn_kv
    };
    bool ok = true;
    for (const Shape & shape : shapes) {
        const size_t row_bytes = ggml_row_size(T, shape.k);
        std::vector<uint8_t> wbytes(row_bytes * shape.m);
        std::vector<float> row(shape.k);
        lcg_state = 0x9e3779b9u ^ (uint32_t) (shape.k * 31 + shape.m);
        for (int64_t r = 0; r < shape.m; ++r) {
            for (float & v : row) {
                v = lcg_uniform() * 0.02f;
            }
            tr->from_float_ref(row.data(), wbytes.data() + row_bytes * r, shape.k);
        }
        std::vector<float> input((size_t) shape.k * 5);
        for (float & v : input) {
            v = lcg_uniform() * 3.0f;
        }
        // The paged backend raises the MMVQ crossover to five columns; mirror it.
        const int previous = ggml_backend_cuda_set_mmvq_max_ncols_override(5);
        ok = check_columns(backend, shape.label, T, shape.k, shape.m, 4, wbytes, input) && ok;
        ok = check_columns(backend, shape.label, T, shape.k, shape.m, 5, wbytes, input) && ok;
        ggml_backend_cuda_set_mmvq_max_ncols_override(previous);
    }
    return ok;
}

} // namespace

int main() {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fprintf(stderr, "failed to query HIP device 0\n");
        return 1;
    }
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7) != 0) {
        std::printf("SKIP: gfx1151 width-4 dispatch test expects gfx1151 (found %s)\n",
                    properties.gcnArchName);
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize HIP backend\n");
        return 1;
    }
    const bool previous_graphs = ggml_backend_cuda_set_graphs_disabled_override(true);
    bool ok = test_narrow_f16(backend);
    ok = test_rocmfp4_four_columns(backend) && ok;
    ggml_backend_cuda_set_graphs_disabled_override(previous_graphs);
    ggml_backend_free(backend);
    std::printf("%s\n", ok ? "PASS gfx1151 width-4 dispatch" : "FAIL gfx1151 width-4 dispatch");
    return ok ? 0 : 1;
}
