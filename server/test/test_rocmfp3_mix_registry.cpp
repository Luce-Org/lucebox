// Regression test for the qtype-105 (Q3_1_ROCMFP3_MIX) decode registry in
// ggml-cuda/rocmfp3_mix.cu. Covers the ownership / cleanup contract added for
// the PR review:
//   - register_host makes a resolvable entry; range lookup is correct;
//   - unregister removes it (no stale base range survives an "unload");
//   - update-in-place and repeated register/unregister cycles do not leak the
//     device side-data buffers (codebooks/modes/rotations).
// The pre-fix code (unregister only erased the vector entry, never cudaFree'd
// the register_host allocations) fails the leak assertion below.

#include "ds4_test_gpu_runtime.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// Registry entry points (register_host/unregister are extern "C"; registered is
// C++ linkage — declared to match the definitions in rocmfp3_mix.cu).
extern "C" void ggml_cuda_rocmfp3_mix_register_host(
    const void * base, size_t nb02, int n_experts, int out, int in,
    const void * codebooks_bf16_host, const uint8_t * modes_host,
    const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp3_mix_unregister(const void * base);
bool ggml_cuda_rocmfp3_mix_registered(const void * vx);

static int g_fails = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fails; }  \
    } while (0)

int main() {
    const int    E    = 8, out = 64, in = 64;
    const size_t nb02 = 4096;  // per-expert byte stride (opaque to the registry)
    std::vector<uint16_t> books((size_t) E * 2 * 8, 0x3f80);  // bf16 ~1.0
    std::vector<uint8_t>  modes(E, 1), rots(E, 0);

    // registered() only does pointer-range arithmetic on the base key — it never
    // dereferences it — so opaque, distinct, aligned values stand in for two
    // model tensors' device bases.
    const void * b0 = reinterpret_cast<const void *>(0x100000000ull);
    const void * b1 = reinterpret_cast<const void *>(0x200000000ull);

    // 1. register + range lookup
    ggml_cuda_rocmfp3_mix_register_host(b0, nb02, E, out, in,
                                        books.data(), modes.data(), rots.data());
    CHECK(ggml_cuda_rocmfp3_mix_registered(b0), "b0 resolves after register");
    CHECK(ggml_cuda_rocmfp3_mix_registered(static_cast<const char *>(b0) + nb02),
          "expert-1 slice resolves (in range)");
    CHECK(!ggml_cuda_rocmfp3_mix_registered(static_cast<const char *>(b0) + (size_t) E * nb02),
          "just past the last expert does not resolve");
    CHECK(!ggml_cuda_rocmfp3_mix_registered(b1), "unrelated base does not resolve");

    // 2. unregister leaves no stale range (the reload/address-reuse hazard)
    ggml_cuda_rocmfp3_mix_unregister(b0);
    CHECK(!ggml_cuda_rocmfp3_mix_registered(b0), "b0 gone after unregister");

    // 3. update-in-place then unregister
    ggml_cuda_rocmfp3_mix_register_host(b0, nb02, E, out, in,
                                        books.data(), modes.data(), rots.data());
    ggml_cuda_rocmfp3_mix_register_host(b0, nb02, E, out, in,  // update same base
                                        books.data(), modes.data(), rots.data());
    CHECK(ggml_cuda_rocmfp3_mix_registered(b0), "b0 resolves after in-place update");
    ggml_cuda_rocmfp3_mix_unregister(b0);
    CHECK(!ggml_cuda_rocmfp3_mix_registered(b0), "b0 gone after update+unregister");

    // 4. no device-memory leak across many register/unregister cycles. A missing
    //    cudaFree in unregister (or on update) leaks ~E*(2*8*2 + 1 + 1) bytes per
    //    cycle; 4000 cycles would drop free VRAM by tens of MB.
    cudaDeviceSynchronize();
    size_t free_warm = 0, total = 0;
    // warm the allocator first so pool growth isn't counted as a leak
    for (int i = 0; i < 64; ++i) {
        ggml_cuda_rocmfp3_mix_register_host(b0, nb02, E, out, in,
                                            books.data(), modes.data(), rots.data());
        ggml_cuda_rocmfp3_mix_unregister(b0);
    }
    cudaDeviceSynchronize();
    (void) cudaMemGetInfo(&free_warm, &total);
    for (int i = 0; i < 4000; ++i) {
        ggml_cuda_rocmfp3_mix_register_host(b0, nb02, E, out, in,
                                            books.data(), modes.data(), rots.data());
        ggml_cuda_rocmfp3_mix_unregister(b0);
    }
    cudaDeviceSynchronize();
    size_t free_end = 0;
    (void) cudaMemGetInfo(&free_end, &total);
    const long long delta = (long long) free_warm - (long long) free_end;
    std::fprintf(stderr, "[registry] free VRAM delta over 4000 cycles: %lld bytes\n", delta);
    CHECK(delta < 8 * 1024 * 1024, "no device leak across register/unregister cycles");

    std::fprintf(stderr, g_fails ? "REGISTRY TEST FAILED (%d)\n"
                                 : "REGISTRY TEST OK\n", g_fails);
    return g_fails ? 1 : 0;
}
