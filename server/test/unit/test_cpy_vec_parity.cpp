// Parity test for the 16-byte same-type strided CPY (cpy.cu cpy_vec16; LUCE_CPY_VEC_LEGACY=1 keeps the scalar
// kernel). Cases: the qwen4exp prefill K/V cache append (f16 [D, Hk, T] permuted to [D, T, Hk] copied into a view
// of the cache at pos0; D = 256 so every row is 16-byte aligned: the vector path), the same with D = 24 and with a
// 24-byte view offset (fall back to the scalar kernel), and an f32 state copy. Every case is compared byte for byte
// with a host reference of the whole destination buffer. SKIP (77) without a GPU.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// src [D, H, T] (type) permuted to [D, T, H] -> cpy into view [D, T, H] of cache [D, ctx, H] at row pos0, byte offset
// `extra` on top of the row offset (0 or a misaligning 24 bytes with the view narrowed to keep it in bounds).
bool run(ggml_backend_t gpu, ggml_type type, int64_t D, int64_t H, int64_t T, int64_t ctx, int64_t pos0, size_t extra) {
    const size_t es = ggml_type_size(type);
    ggml_init_params ip{ 16 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * src = ggml_new_tensor_3d(c, type, D, H, T);
    ggml_tensor * cache = ggml_new_tensor_3d(c, type, D, ctx, H);
    ggml_set_input(src); ggml_set_input(cache);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_tensor * dst = ggml_view_3d(c, cache, D, T, H, cache->nb[1], cache->nb[2], cache->nb[1] * (size_t) pos0 + extra);
    ggml_tensor * cpy = ggml_cpy(c, ggml_permute(c, src, 0, 2, 1, 3), dst);
    ggml_build_forward_expand(gf, cpy);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    if (!buf) { ggml_free(c); return false; }
    std::vector<uint8_t> s(ggml_nbytes(src)), k(ggml_nbytes(cache));
    uint32_t r = 3u;
    for (uint8_t & b : s) { r = r * 1664525u + 1013904223u; b = (uint8_t) (r >> 24); }
    for (uint8_t & b : k) { r = r * 1664525u + 1013904223u; b = (uint8_t) (r >> 24); }
    ggml_backend_tensor_set(src, s.data(), 0, s.size());
    ggml_backend_tensor_set(cache, k.data(), 0, k.size());
    // host reference: element (d, t, h) of the permuted source lands at view offset d*es + t*nb1 + h*nb2
    std::vector<uint8_t> want = k;
    for (int64_t h = 0; h < H; ++h) for (int64_t t = 0; t < T; ++t)
        std::memcpy(want.data() + dst->view_offs + t * cache->nb[1] + h * cache->nb[2], s.data() + (t * H + h) * D * es, D * es);
    bool ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    std::vector<uint8_t> got(k.size());
    ggml_backend_tensor_get(cache, got.data(), 0, got.size());
    ok = ok && got == want;
    std::printf("[cpy-vec] %s D=%lld H=%lld T=%lld pos0=%lld extra=%zu %s\n", ggml_type_name(type), (long long) D, (long long) H,
                (long long) T, (long long) pos0, extra, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return ok;
}

}  // namespace

int main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::printf("[cpy-vec] SKIP: no GPU\n"); return 77; }
    bool ok = true;
    ok = run(gpu, GGML_TYPE_F16, 256, 2, 4136, 8192, 7, 0) && ok;     // K/V cache append at p4096 (vector path)
    ok = run(gpu, GGML_TYPE_F16, 256, 2, 1, 512, 3, 0) && ok;         // one row
    ok = run(gpu, GGML_TYPE_F16, 24, 2, 300, 512, 5, 0) && ok;        // D % 8 != 0: scalar fallback
    ok = run(gpu, GGML_TYPE_F16, 8, 2, 300, 512, 5, 24) && ok;        // misaligned view base: scalar fallback
    ok = run(gpu, GGML_TYPE_F32, 128, 48, 128, 256, 0, 0) && ok;      // f32 (VEC = 4)
    ok = run(gpu, GGML_TYPE_F32, 6, 3, 100, 200, 1, 0) && ok;         // f32, D % 4 != 0: scalar fallback
    ggml_backend_free(gpu);
    std::printf("[cpy-vec] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
