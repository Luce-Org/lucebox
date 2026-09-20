// Differential for the vectorized bf16<->f32 cast kernels.
//
// convert_unary_cont_cuda takes the 8-wide vector kernel only when both
// pointers are 16-byte aligned and falls back to the scalar convert_unary
// kernel otherwise. Offsetting the destination by one element forces the
// scalar path, so the same input runs through both and the outputs must be
// bit-identical.
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

typedef void (*cast_fn)(const void *, void *, int64_t, hipStream_t);
cast_fn ggml_get_to_bf16_cuda(ggml_type type);
cast_fn ggml_get_to_fp32_cuda(ggml_type type);

static uint32_t g_state = 0x9e3779b9u;
static uint32_t rnd_u32() {
    g_state = g_state * 1664525u + 1013904223u;
    return g_state;
}
static float rnd_f32() {
    return (float) ((rnd_u32() >> 8) & 0xFFFFFF) / (float) 0x1000000 * 4.0f - 2.0f;
}

struct dev {
    void * p = nullptr;
    explicit dev(size_t bytes) { if (hipMalloc(&p, bytes) != hipSuccess) p = nullptr; }
    ~dev() { if (p) (void) hipFree(p); }
    dev(const dev &) = delete;
    dev & operator=(const dev &) = delete;
};

static bool check(const char * what, int64_t k) {
    std::fprintf(stderr, "  FAIL: %s (k=%lld)\n", what, (long long) k);
    return false;
}

static bool run_bf16_to_f32(cast_fn to_f32, int64_t k) {
    const size_t n = (size_t) k;
    std::vector<uint16_t> h((size_t) k + 8);
    for (int64_t i = 0; i < k; ++i) h[(size_t) i] = (uint16_t) rnd_u32();

    dev src(n * sizeof(uint16_t) + 16);
    dev vdst((n + 8) * sizeof(float));
    dev sdst((n + 8) * sizeof(float));
    if (!src.p || !vdst.p || !sdst.p) return check("hipMalloc", k);
    if (hipMemcpy(src.p, h.data(), n * sizeof(uint16_t), hipMemcpyHostToDevice) != hipSuccess) return check("H2D", k);
    if (hipMemset(vdst.p, 0xCD, (n + 8) * sizeof(float)) != hipSuccess) return check("memset", k);
    if (hipMemset(sdst.p, 0xCD, (n + 8) * sizeof(float)) != hipSuccess) return check("memset", k);

    to_f32(src.p, vdst.p, k, (hipStream_t) 0);
    to_f32(src.p, (float *) sdst.p + 1, k, (hipStream_t) 0);
    if (hipDeviceSynchronize() != hipSuccess) return check("sync", k);

    std::vector<float> vout(n), sout(n);
    if (hipMemcpy(vout.data(), vdst.p, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return check("D2H", k);
    if (hipMemcpy(sout.data(), (float *) sdst.p + 1, n * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return check("D2H", k);
    if (std::memcmp(vout.data(), sout.data(), n * sizeof(float)) != 0) return check("bf16->f32 vec != scalar", k);
    return true;
}

static bool run_f32_to_bf16(cast_fn to_bf16, int64_t k) {
    const size_t n = (size_t) k;
    std::vector<float> h((size_t) k);
    for (int64_t i = 0; i < k; ++i) h[(size_t) i] = rnd_f32();

    dev src(n * sizeof(float) + 16);
    dev vdst((n + 8) * sizeof(uint16_t));
    dev sdst((n + 8) * sizeof(uint16_t));
    if (!src.p || !vdst.p || !sdst.p) return check("hipMalloc", k);
    if (hipMemcpy(src.p, h.data(), n * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return check("H2D", k);
    if (hipMemset(vdst.p, 0xCD, (n + 8) * sizeof(uint16_t)) != hipSuccess) return check("memset", k);
    if (hipMemset(sdst.p, 0xCD, (n + 8) * sizeof(uint16_t)) != hipSuccess) return check("memset", k);

    to_bf16(src.p, vdst.p, k, (hipStream_t) 0);
    to_bf16(src.p, (uint16_t *) sdst.p + 1, k, (hipStream_t) 0);
    if (hipDeviceSynchronize() != hipSuccess) return check("sync", k);

    std::vector<uint16_t> vout(n), sout(n);
    if (hipMemcpy(vout.data(), vdst.p, n * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess) return check("D2H", k);
    if (hipMemcpy(sout.data(), (uint16_t *) sdst.p + 1, n * sizeof(uint16_t), hipMemcpyDeviceToHost) != hipSuccess) return check("D2H", k);
    if (std::memcmp(vout.data(), sout.data(), n * sizeof(uint16_t)) != 0) return check("f32->bf16 vec != scalar", k);
    return true;
}

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[convert-vec] SKIP: no HIP device\n");
        return 77;
    }

    const cast_fn to_f32  = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
    const cast_fn to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    if (!to_f32 || !to_bf16) {
        std::printf("[convert-vec] FAIL: cast lookup\n");
        return 1;
    }

    const int64_t ks[] = { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65,
                           127, 255, 256, 257, 511, 1000, 1023, 1024, 4097, 65536, 99999 };
    bool ok = true;
    for (int64_t k : ks) {
        const bool a = run_bf16_to_f32(to_f32, k);
        const bool b = run_f32_to_bf16(to_bf16, k);
        std::printf("[convert-vec] k=%lld %s\n", (long long) k, (a && b) ? "ok" : "FAIL");
        ok = a && b && ok;
    }
    std::printf("[convert-vec] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[convert-vec] SKIP: HIP-only\n");
    return 77;
#endif
}
