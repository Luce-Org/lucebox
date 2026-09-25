// Helper copies must keep working while another thread captures a graph.
//
// A process serving several models (--load-balancing) captures each model's
// graphs on its own stream while other workers copy draft features. The
// helpers used to fall back to the legacy default stream and
// cudaDeviceSynchronize(), which fail on HIP ("operation not permitted when
// stream is capturing") and invalidate the other thread's capture.

#include "CppUnitTestFramework.hpp"
#include "common/peer_access.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using luce::common::copy_peer_async;

namespace {
struct CaptureSafeCopiesFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

// Frees `ptr` with its owning device current.
void free_on(int device, void * ptr) {
    if (ptr && cudaSetDevice(device) == cudaSuccess) cudaFree(ptr);
}

// A listed device can still be unusable (visibility masks, exclusive mode).
bool device_usable(int device) {
    void * probe = nullptr;
    const bool ok = cudaSetDevice(device) == cudaSuccess &&
                    cudaMalloc(&probe, 256) == cudaSuccess;
    if (probe) cudaFree(probe);
    (void) cudaGetLastError();
    return ok;
}
}  // namespace

TEST_CASE(CaptureSafeCopiesFixture, copies_during_relaxed_capture) {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices <= 0) {
        SKIP("CUDA/HIP device unavailable");
    }
    constexpr size_t kBytes = 1 << 16;
    // With a second usable GPU, also copy device 1 -> device 0 (the
    // target/draft split), which goes through the peer or pinned-staging path.
    const bool cross = n_devices > 1 && device_usable(1);
    if (n_devices > 1 && !cross) {
        std::puts("[capture-safe-copies] device 1 unusable: same-device leg only");
    }
    void * src = nullptr;
    void * dst = nullptr;
    void * src_peer = nullptr;
    void * dst_peer = nullptr;
    cudaStream_t capture_stream = nullptr;
    std::vector<unsigned char> host(kBytes);
    for (size_t i = 0; i < kBytes; ++i) host[i] = (unsigned char) (i * 7 + 3);
    std::atomic<bool> copy_ok{false};
    std::atomic<bool> cross_ok{!cross};
    bool setup_ok = false;
    bool capture_ok = false;
    bool read_ok = false;

    // Every failure falls through to the cleanup below.
    do {
        if (cudaSetDevice(0) != cudaSuccess ||
            cudaMalloc(&src, kBytes) != cudaSuccess || cudaMalloc(&dst, kBytes) != cudaSuccess ||
            cudaMemcpy(src, host.data(), kBytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking) != cudaSuccess) {
            break;
        }
        if (cross) {
            if (cudaMalloc(&dst_peer, kBytes) != cudaSuccess || cudaSetDevice(1) != cudaSuccess ||
                cudaMalloc(&src_peer, kBytes) != cudaSuccess ||
                cudaMemcpy(src_peer, host.data(), kBytes, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaSetDevice(0) != cudaSuccess) {
                break;
            }
        }
        setup_ok = true;
        // Hold a relaxed capture open (as ggml-cuda does) while another
        // thread runs the helper copies.
        if (cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            break;
        }
        std::thread worker([&] {
            copy_ok = copy_peer_async(dst, 0, src, 0, kBytes);
            if (cross) cross_ok = copy_peer_async(dst_peer, 0, src_peer, 1, kBytes);
        });
        worker.join();
        cudaGraph_t graph = nullptr;
        capture_ok = cudaStreamEndCapture(capture_stream, &graph) == cudaSuccess;
        if (graph) cudaGraphDestroy(graph);

        std::vector<unsigned char> back(kBytes, 0);
        read_ok = cudaSetDevice(0) == cudaSuccess &&
                  cudaMemcpy(back.data(), dst, kBytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  back == host;
        if (cross) {
            std::vector<unsigned char> peer_back(kBytes, 0);
            read_ok = read_ok &&
                cudaMemcpy(peer_back.data(), dst_peer, kBytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                peer_back == host;
        }
    } while (false);

    if (capture_stream && cudaSetDevice(0) == cudaSuccess) cudaStreamDestroy(capture_stream);
    free_on(0, src);
    free_on(0, dst);
    free_on(0, dst_peer);
    free_on(1, src_peer);
    std::printf("[capture-safe-copies] copy=%d cross=%s capture=%d data=%d\n",
                (int) copy_ok.load(), cross ? (cross_ok ? "1" : "0") : "skip",
                (int) capture_ok, (int) read_ok);
    REQUIRE(setup_ok);
    CHECK(copy_ok.load());
    CHECK(cross_ok.load());
    CHECK(capture_ok);
    REQUIRE(read_ok);
}
