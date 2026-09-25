#include "peer_access.h"
#include "internal.h"   // dflash_cuda_copy_between_devices

#include <cstdio>

namespace luce::common {

// ── global state ────────────────────────────────────────────────
bool g_peer_access_opt_in = false;
std::unordered_map<std::uint64_t, bool> g_peer_pair_ok_cache;

// ── helpers ─────────────────────────────────────────────────────
bool enable_peer_access_one_way(int device, int peer) {
    if (device == peer) return true;
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, device, peer);
    if (err != cudaSuccess || !can_access) return false;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) return false;
    err = cudaDeviceEnablePeerAccess(peer, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return true;
    }
    return err == cudaSuccess;
}

bool enable_peer_access_pair(int a, int b) {
    if (a == b) return true;
    const bool ab = enable_peer_access_one_way(a, b);
    const bool ba = enable_peer_access_one_way(b, a);
    return ab && ba;
}

static std::uint64_t peer_pair_key(int a, int b) {
    const int lo = std::min(a, b);
    const int hi = std::max(a, b);
    return (std::uint64_t)(unsigned)lo << 32 | (unsigned)hi;
}

static void log_staged_cross_gpu_once() {
    static bool logged = false;
    if (logged) return;
    logged = true;
    std::fprintf(stderr,
                 "[luce] Using safe (slower) cross-GPU copy via host staging "
                 "(--peer-access not set or P2P unavailable for this device pair).\n");
}

bool cross_device_peer_memcpy_ok(int src_device, int dst_device) {
    if (src_device == dst_device) return true;
    if (!g_peer_access_opt_in) return false;
    const std::uint64_t k = peer_pair_key(src_device, dst_device);
    const auto it = g_peer_pair_ok_cache.find(k);
    if (it != g_peer_pair_ok_cache.end()) return it->second;
    const bool ok = enable_peer_access_pair(src_device, dst_device);
    g_peer_pair_ok_cache[k] = ok;
    return ok;
}

namespace {
// Destroys a thread's copy streams when the thread exits.
struct ThreadCopyStreams {
    static constexpr int kMaxDevices = 16;
    cudaStream_t streams[kMaxDevices] = {};
    ~ThreadCopyStreams() {
        for (int device = 0; device < kMaxDevices; ++device) {
            if (streams[device] && cudaSetDevice(device) == cudaSuccess) {
                (void) cudaStreamDestroy(streams[device]);
            }
        }
    }
};
}  // namespace

cudaStream_t luce_copy_stream(int device) {
    thread_local ThreadCopyStreams local;
    if (device < 0 || device >= ThreadCopyStreams::kMaxDevices) return nullptr;
    cudaStream_t & stream = local.streams[device];
    if (!stream) {
        if (cudaSetDevice(device) != cudaSuccess ||
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            stream = nullptr;
        }
    }
    return stream;
}

bool luce_copy_stream_sync(int device) {
    cudaStream_t stream = luce_copy_stream(device);
    if (!stream || cudaSetDevice(device) != cudaSuccess) return false;
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

bool copy_peer_async(void * dst, int dst_device,
                     const void * src, int src_device,
                     size_t bytes,
                     cudaStream_t stream) {
    if (bytes == 0) return true;
    if (!stream) stream = luce_copy_stream(dst_device);
    if (!stream) return false;
    cudaError_t err = cudaSuccess;
    if (dst_device == src_device) {
        err = cudaSetDevice(dst_device);
        if (err != cudaSuccess) return false;
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) return false;
        return cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (cross_device_peer_memcpy_ok(src_device, dst_device)) {
        err = cudaSetDevice(dst_device);
        if (err != cudaSuccess) return false;
        err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream);
        if (err != cudaSuccess) return false;
        return cudaStreamSynchronize(stream) == cudaSuccess;
    }
    log_staged_cross_gpu_once();
#if defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP)
    err = cudaSetDevice(dst_device);
    if (err != cudaSuccess) return false;
    err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream);
    if (err != cudaSuccess) return false;
    return cudaStreamSynchronize(stream) == cudaSuccess;
#else
    return dflash_cuda_copy_between_devices(src_device, src, dst_device, dst, bytes,
                                            luce_copy_stream(src_device), stream);
#endif
}

}  // namespace luce::common
