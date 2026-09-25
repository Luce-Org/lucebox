// Helper copies must keep working while another thread captures a graph.
//
// A process serving several models (--load-balancing) captures each model's
// graphs on its own stream while other workers copy draft features. The
// helpers used to fall back to the legacy default stream and
// cudaDeviceSynchronize(), which fail on HIP ("operation not permitted when
// stream is capturing") and invalidate the other thread's capture.

#include "common/peer_access.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using luce::common::copy_peer_async;

int main() {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices <= 0) {
        std::puts("[capture-safe-copies] SKIP: GPU device unavailable");
        return 77;
    }
    constexpr size_t kBytes = 1 << 16;
    if (cudaSetDevice(0) != cudaSuccess) return 1;
    void * src = nullptr;
    void * dst = nullptr;
    // With a second GPU, also copy device 1 -> device 0 (the target/draft
    // split), which goes through the peer or pinned-staging path.
    const bool cross = n_devices > 1;
    void * src_peer = nullptr;
    void * dst_peer = nullptr;
    cudaStream_t capture_stream = nullptr;
    bool ok = false;
    std::vector<unsigned char> host(kBytes);
    for (size_t i = 0; i < kBytes; ++i) host[i] = (unsigned char) (i * 7 + 3);
    std::atomic<bool> copy_ok{false};
    std::atomic<bool> cross_ok{!cross};
    bool capture_ok = false;
    bool read_ok = false;

    // Every failure falls through to the cleanup below.
    do {
        if (cudaMalloc(&src, kBytes) != cudaSuccess || cudaMalloc(&dst, kBytes) != cudaSuccess ||
            cudaMemcpy(src, host.data(), kBytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::fprintf(stderr, "[capture-safe-copies] setup failed\n");
            break;
        }
        if (cross) {
            if (cudaMalloc(&dst_peer, kBytes) != cudaSuccess || cudaSetDevice(1) != cudaSuccess ||
                cudaMalloc(&src_peer, kBytes) != cudaSuccess ||
                cudaMemcpy(src_peer, host.data(), kBytes, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaSetDevice(0) != cudaSuccess) {
                std::fprintf(stderr, "[capture-safe-copies] cross-device setup failed\n");
                break;
            }
        }
        // Hold a relaxed capture open (as ggml-cuda does) while another
        // thread runs a default-stream helper copy.
        if (cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            std::fprintf(stderr, "[capture-safe-copies] begin capture failed\n");
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
        read_ok = cudaMemcpy(back.data(), dst, kBytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  back == host;
        if (cross) {
            std::vector<unsigned char> peer_back(kBytes, 0);
            read_ok = read_ok &&
                cudaMemcpy(peer_back.data(), dst_peer, kBytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                peer_back == host;
        }
        ok = copy_ok && cross_ok && capture_ok && read_ok;
    } while (false);

    if (capture_stream) cudaStreamDestroy(capture_stream);
    if (src) cudaFree(src);
    if (dst) cudaFree(dst);
    if (dst_peer) cudaFree(dst_peer);
    if (src_peer && cudaSetDevice(1) == cudaSuccess) cudaFree(src_peer);
    std::printf("[capture-safe-copies] copy=%d cross=%s capture=%d data=%d\n",
                (int) copy_ok.load(), cross ? (cross_ok ? "1" : "0") : "skip",
                (int) capture_ok, (int) read_ok);
    return ok ? 0 : 1;
}
