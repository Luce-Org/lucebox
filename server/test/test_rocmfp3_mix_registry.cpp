// Regression test for the qtype-105 (Q3_1_ROCMFP3_MIX) decode registry in
// ggml-cuda/rocmfp3_mix.cu. Covers the ownership / cleanup contract added for
// the PR review:
//   - register_host makes a resolvable entry; range lookup is correct;
//   - unregister removes it (no stale base range survives an "unload");
//   - update-in-place and repeated register/unregister cycles do not leak the
//     device side-data buffers (codebooks/modes).
// The pre-fix code (unregister only erased the vector entry, never cudaFree'd
// the register_host allocations) fails the leak assertion below.

#include "ds4_test_gpu_runtime.h"
#include "CppUnitTestFramework.hpp"
#include "ggml-cuda.h"
#include "rocmfp3_mix.cuh"
using CppUnitTestFramework::CommonFixture;
#undef CHECK

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fails; }  \
    } while (0)

namespace {
struct Rocmfp3MixRegistryFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

TEST_CASE(Rocmfp3MixRegistryFixture, registry_lifecycle) {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice || device_count == 0) {
        SKIP("no CUDA/HIP device available");
    }
    REQUIRE_TRUE(device_status == cudaSuccess);

    const int    E    = 8, out = 64, in = 64;
    const size_t expert_bytes = (size_t) out * (in / 32) * 14;
    const size_t nb02 = 4096;  // includes alignment padding after each payload
    std::vector<uint16_t> books((size_t) E * 2 * 8, 0x3f80);  // bf16 ~1.0
    std::vector<uint8_t>  modes(E, 1);

    // registered() only does pointer-range arithmetic on the base key — it never
    // dereferences it — so opaque, distinct, aligned values stand in for two
    // model tensors' device bases.
    const void * b0 = reinterpret_cast<const void *>(0x100000000ull);
    const void * b1 = reinterpret_cast<const void *>(0x200000000ull);

    // 1. Invalid metadata fails without aborting or leaving a registry entry.
    CHECK(!ggml_cuda_rocmfp3_mix_register_host(
              nullptr, nb02, E, out, in, books.data(), modes.data()),
          "null registration base is rejected");
    CHECK(!ggml_cuda_rocmfp3_mix_register_host(
              b1, expert_bytes - 1, E, out, in, books.data(), modes.data()),
          "undersized expert stride is rejected");
    std::vector<uint8_t> invalid_modes = modes;
    invalid_modes[0] = 2;
    CHECK(!ggml_cuda_rocmfp3_mix_register_host(
              b1, nb02, E, out, in, books.data(), invalid_modes.data()),
          "unsupported mode is rejected");
    CHECK(!ggml_cuda_rocmfp3_mix_registered(b1),
          "invalid registration leaves no entry");

    // 2. register + range lookup
    CHECK(ggml_cuda_rocmfp3_mix_register_host(
              b0, nb02, E, out, in, books.data(), modes.data()),
          "valid registration succeeds");
    CHECK(ggml_cuda_rocmfp3_mix_registered(b0), "b0 resolves after register");
    CHECK(ggml_cuda_rocmfp3_mix_registered(static_cast<const char *>(b0) + nb02),
          "expert-1 slice resolves (in range)");
    CHECK(!ggml_cuda_rocmfp3_mix_registered(static_cast<const char *>(b0) + 14),
          "an interior block is not a registered tensor base");
    CHECK(!ggml_cuda_rocmfp3_mix_registered(
              static_cast<const char *>(b0) + expert_bytes),
          "padding after an expert payload does not resolve");
    CHECK(!ggml_cuda_rocmfp3_mix_registered(static_cast<const char *>(b0) + (size_t) E * nb02),
          "just past the last expert does not resolve");
    CHECK(!ggml_cuda_rocmfp3_mix_registered(b1), "unrelated base does not resolve");
    const void * codebooks = nullptr;
    const uint8_t * registered_modes = nullptr;
    CHECK(ggml_cuda_rocmfp3_mix_mmq_info(
              static_cast<const char *>(b0) + 14,
              &codebooks, &registered_modes),
          "MMQ accepts a block-aligned offset inside an expert");
    CHECK(codebooks != nullptr && registered_modes != nullptr,
          "MMQ returns the registered side data");
    CHECK(!ggml_cuda_rocmfp3_mix_mmq_info(
              static_cast<const char *>(b0) + 1,
              &codebooks, &registered_modes),
          "MMQ rejects an unaligned offset inside an expert");

    // 3. unregister leaves no stale range (the reload/address-reuse hazard)
    ggml_cuda_rocmfp3_mix_unregister(b0);
    CHECK(!ggml_cuda_rocmfp3_mix_registered(b0), "b0 gone after unregister");

    // 4. update-in-place then unregister
    CHECK(ggml_cuda_rocmfp3_mix_register_host(
              b0, nb02, E, out, in, books.data(), modes.data()),
          "registration before update succeeds");
    CHECK(ggml_cuda_rocmfp3_mix_register_host(
              b0, nb02, E, out, in, books.data(), modes.data()),
          "in-place registration update succeeds");
    CHECK(ggml_cuda_rocmfp3_mix_registered(b0), "b0 resolves after in-place update");
    ggml_cuda_rocmfp3_mix_unregister(b0);
    CHECK(!ggml_cuda_rocmfp3_mix_registered(b0), "b0 gone after update+unregister");

    // 5. Teardown cannot free side data between lookup and asynchronous kernel
    //    enqueue. Dispatchers hold this lock across both operations; unregister
    //    must wait until the launch has been handed to the device.
    CHECK(ggml_cuda_rocmfp3_mix_register_host(
              b0, nb02, E, out, in, books.data(), modes.data()),
          "registration before dispatch lock test succeeds");
    std::atomic<bool> teardown_started{false};
    std::atomic<bool> teardown_finished{false};
    ggml_cuda_rocmfp3_mix_registry_lock();
    std::thread teardown([&] {
        teardown_started.store(true, std::memory_order_release);
        ggml_cuda_rocmfp3_mix_unregister(b0);
        teardown_finished.store(true, std::memory_order_release);
    });
    while (!teardown_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(!teardown_finished.load(std::memory_order_acquire),
          "unregister waits for an in-flight dispatch");
    ggml_cuda_rocmfp3_mix_registry_unlock();
    teardown.join();
    CHECK(teardown_finished.load(std::memory_order_acquire),
          "unregister completes after dispatch releases the registry");
    CHECK(!ggml_cuda_rocmfp3_mix_registered(b0),
          "dispatch lock test leaves no registry entry");

    // 6. no device-memory leak across many register/unregister cycles. A missing
    //    cudaFree in unregister (or on update) leaks ~E*(2*8*2 + 1) bytes per
    //    cycle; 4000 cycles would produce a measurable free-VRAM drop.
    cudaDeviceSynchronize();
    size_t free_warm = 0, total = 0;
    // warm the allocator first so pool growth isn't counted as a leak
    for (int i = 0; i < 64; ++i) {
        CHECK(ggml_cuda_rocmfp3_mix_register_host(
                  b0, nb02, E, out, in, books.data(), modes.data()),
              "warmup registration succeeds");
        ggml_cuda_rocmfp3_mix_unregister(b0);
    }
    cudaDeviceSynchronize();
    (void) cudaMemGetInfo(&free_warm, &total);
    for (int i = 0; i < 4000; ++i) {
        CHECK(ggml_cuda_rocmfp3_mix_register_host(
                  b0, nb02, E, out, in, books.data(), modes.data()),
              "cycle registration succeeds");
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
    REQUIRE_TRUE(g_fails == 0);
}
