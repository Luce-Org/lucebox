// Differential for the ne00==1 scalar get_rows kernel (k_get_rows_scalar).
//
// ggml_get_rows with a single-element row takes the scalar gather path. It is
// what the QSA top-k block sort and the MoE softmax selection use, so compare
// it against a reference gather over a range of sizes that cross the 256-thread
// block boundary and for both I32 and F32 payloads. On gfx1151 the F32 cases
// keep ne10 < 128 so the RDNA3.5 packed fast path (which needs ne10 >= 128)
// does not shadow the kernel under test.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t g_state = 0xabcdef01u;
static uint32_t rnd_u32() {
    g_state = g_state * 1664525u + 1013904223u;
    return g_state;
}

static bool run_case(ggml_type wtype, int64_t nrows, int64_t ne10, int64_t ne2, int64_t ne3) {
    const size_t n_idx = (size_t) ne10 * ne2 * ne3;
    const size_t n_src = (size_t) nrows * ne2 * ne3;

    std::vector<int32_t> idx(n_idx);
    for (size_t i = 0; i < n_idx; ++i) idx[i] = (int32_t) (rnd_u32() % (uint32_t) nrows);

    std::vector<float>   src_f(wtype == GGML_TYPE_F32 ? n_src : 0);
    std::vector<int32_t> src_i(wtype == GGML_TYPE_I32 ? n_src : 0);
    for (auto & v : src_f) v = (float) (int32_t) rnd_u32();
    for (auto & v : src_i) v = (int32_t) rnd_u32();

    const uint8_t * src_bytes = wtype == GGML_TYPE_F32
        ? (const uint8_t *) src_f.data() : (const uint8_t *) src_i.data();
    const size_t esize = wtype == GGML_TYPE_F32 ? sizeof(float) : sizeof(int32_t);

    ggml_init_params ip{};
    ip.mem_size = 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor * a = ggml_new_tensor_4d(ctx, wtype, 1, nrows, ne2, ne3);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, ne10, ne2, ne3);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_tensor * d = ggml_get_rows(ctx, a, b);
    ggml_set_output(d);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);

    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { ggml_free(ctx); return false; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    bool ok = buf != nullptr;
    if (ok) {
        ggml_backend_tensor_set(a, src_bytes, 0, n_src * esize);
        ggml_backend_tensor_set(b, idx.data(), 0, n_idx * sizeof(int32_t));
        ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    }

    std::vector<int32_t> out_i(n_idx);
    std::vector<float>   out_f(n_idx);
    if (ok) {
        ggml_backend_tensor_get(d, wtype == GGML_TYPE_F32 ? (void *) out_f.data() : (void *) out_i.data(),
                                0, n_idx * esize);
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(gpu);
    ggml_free(ctx);
    if (!ok) { std::fprintf(stderr, "  FAIL: setup/compute\n"); return false; }

    for (int64_t i12 = 0; i12 < ne3; ++i12) {
        for (int64_t i11 = 0; i11 < ne2; ++i11) {
            for (int64_t i10 = 0; i10 < ne10; ++i10) {
                const size_t o = (size_t) (i10 + ne10 * (i11 + ne2 * i12));
                const size_t s = (size_t) (idx[o] + nrows * (i11 + ne2 * i12));
                if (wtype == GGML_TYPE_F32) {
                    if (out_f[o] != src_f[s]) {
                        std::fprintf(stderr, "  FAIL: F32 %s at (%lld,%lld,%lld) got %g want %g (row %d)\n",
                            ggml_type_name(wtype), (long long) i10, (long long) i11, (long long) i12,
                            out_f[o], src_f[s], idx[o]);
                        return false;
                    }
                } else {
                    if (out_i[o] != src_i[s]) {
                        std::fprintf(stderr, "  FAIL: I32 at (%lld,%lld,%lld) got %d want %d (row %d)\n",
                            (long long) i10, (long long) i11, (long long) i12, out_i[o], src_i[s], idx[o]);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
#if defined(GGML_USE_HIP)
    if (ggml_backend_cuda_get_device_count() <= 0) {
        std::printf("[getrows-scalar] SKIP: no HIP device\n");
        return 77;
    }

    bool ok = true;
    const int64_t spans[] = { 1, 2, 7, 63, 127, 255, 256, 257, 511, 512, 513, 1000, 4096, 20000 };
    for (int64_t ne10 : spans) {
        const bool c = run_case(GGML_TYPE_I32, 1024, ne10, 1, 1);
        std::printf("[getrows-scalar] I32 ne10=%lld %s\n", (long long) ne10, c ? "ok" : "FAIL");
        ok = c && ok;
    }
    const int64_t fspans[] = { 1, 3, 31, 100, 127 };
    for (int64_t ne10 : fspans) {
        const bool c = run_case(GGML_TYPE_F32, 1024, ne10, 1, 1);
        std::printf("[getrows-scalar] F32 ne10=%lld %s\n", (long long) ne10, c ? "ok" : "FAIL");
        ok = c && ok;
    }
    // Batched gather exercises the i11/i12 strides of the scalar kernel.
    for (int64_t ne2 : { 3, 4 }) {
        const bool c = run_case(GGML_TYPE_I32, 1024, 37, ne2, 4);
        std::printf("[getrows-scalar] I32 batched ne10=37 ne2=%lld ne3=4 %s\n", (long long) ne2, c ? "ok" : "FAIL");
        ok = c && ok;
    }
    std::printf("[getrows-scalar] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[getrows-scalar] SKIP: HIP-only\n");
    return 77;
#endif
}
