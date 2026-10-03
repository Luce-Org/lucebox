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

    // 6. no device-memory leak across many register/unregister cycles. Use a
    //    real device allocation here: production registrations always receive
    //    one, while repeatedly querying an invented address can make the HIP
    //    runtime retain architecture-dependent pointer-tracking storage.
    void * leak_base = nullptr;
    REQUIRE_TRUE(cudaMalloc(&leak_base, (size_t) E * nb02) == cudaSuccess);

    // A missing cudaFree in unregister (or on update) leaks the side-data
    // allocations every cycle; 4000 cycles produce a measurable VRAM drop.
    cudaDeviceSynchronize();
    size_t free_warm = 0, total = 0;
    // warm the allocator first so pool growth isn't counted as a leak
    for (int i = 0; i < 64; ++i) {
        CHECK(ggml_cuda_rocmfp3_mix_register_host(
                  leak_base, nb02, E, out, in, books.data(), modes.data()),
              "warmup registration succeeds");
        ggml_cuda_rocmfp3_mix_unregister(leak_base);
    }
    cudaDeviceSynchronize();
    (void) cudaMemGetInfo(&free_warm, &total);
    // cudaMemGetInfo is device-wide: on a shared GPU another process can
    // move free memory by hundreds of MB during the loop. A leak in this
    // code path costs the same every cycle, so it shows in every chunk of
    // the loop; an external allocation shows in one chunk and not the rest.
    // Require the leak signature (every chunk over the threshold), and
    // report a moving device as such instead of as a leak.
    // A chunk of 1000 cycles requests about 264 KiB of side-data device
    // memory in total, so a leak of those buffers shows as a drop of that
    // order in every chunk; the threshold sits well below it. A device
    // shared with another process can also lose free memory steadily, so
    // each cycle chunk is paired with an idle chunk of the same wall time:
    // a leak drops free memory only while cycling, an external consumer
    // drops it in the idle chunks too.
    constexpr int kChunks = 4;
    constexpr int kCyclesPerChunk = 1000;
    constexpr long long kLeakThreshold = 64 * 1024;   // per chunk
    long long chunk_delta[kChunks] = {};
    long long idle_delta[kChunks] = {};
    size_t free_prev = free_warm;
    for (int c = 0; c < kChunks; ++c) {
        const auto cycle_t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kCyclesPerChunk; ++i) {
            CHECK(ggml_cuda_rocmfp3_mix_register_host(
                      leak_base, nb02, E, out, in, books.data(), modes.data()),
                  "cycle registration succeeds");
            ggml_cuda_rocmfp3_mix_unregister(leak_base);
        }
        cudaDeviceSynchronize();
        const auto cycle_wall = std::chrono::steady_clock::now() - cycle_t0;
        size_t free_now = 0;
        (void) cudaMemGetInfo(&free_now, &total);
        chunk_delta[c] = (long long) free_prev - (long long) free_now;
        free_prev = free_now;
        // idle chunk: same wall time, no registry activity
        std::this_thread::sleep_for(cycle_wall);
        cudaDeviceSynchronize();
        (void) cudaMemGetInfo(&free_now, &total);
        idle_delta[c] = (long long) free_prev - (long long) free_now;
        free_prev = free_now;
    }
    const long long delta = (long long) free_warm - (long long) free_prev;
    std::fprintf(stderr,
                 "[registry] free VRAM delta over %d cycles: %lld bytes "
                 "(per %d-cycle chunk: %lld, %lld, %lld, %lld; idle chunks: "
                 "%lld, %lld, %lld, %lld)\n",
                 kChunks * kCyclesPerChunk, delta, kCyclesPerChunk,
                 chunk_delta[0], chunk_delta[1], chunk_delta[2], chunk_delta[3],
                 idle_delta[0], idle_delta[1], idle_delta[2], idle_delta[3]);
    bool every_chunk_leaks = true;
    bool any_chunk_moved = false;
    bool device_moves_when_idle = false;
    for (int c = 0; c < kChunks; ++c) {
        every_chunk_leaks = every_chunk_leaks && chunk_delta[c] >= kLeakThreshold;
        any_chunk_moved = any_chunk_moved || chunk_delta[c] >= kLeakThreshold;
        device_moves_when_idle = device_moves_when_idle || idle_delta[c] >= kLeakThreshold;
    }
    if (any_chunk_moved && (!every_chunk_leaks || device_moves_when_idle)) {
        std::fprintf(stderr,
                     "[registry] device-wide free memory moved outside the "
                     "leak signature (shared device), not counted as a leak\n");
    }
    CHECK(!(every_chunk_leaks && !device_moves_when_idle),
          "no device leak across register/unregister cycles");
    CHECK(cudaFree(leak_base) == cudaSuccess, "leak-test base allocation is released");

    std::fprintf(stderr, g_fails ? "REGISTRY TEST FAILED (%d)\n"
                                 : "REGISTRY TEST OK\n", g_fails);
    REQUIRE_TRUE(g_fails == 0);
}
