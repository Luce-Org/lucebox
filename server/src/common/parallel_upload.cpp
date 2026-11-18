#include "parallel_upload.h"

#include "common/gpu_runtime_compat.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <thread>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace luce::common {

void append_upload_pieces(std::vector<UploadPiece> & pieces, ggml_tensor * tensor,
                          const uint8_t * src, size_t size, size_t piece) {
    if (!tensor || !src || size == 0) return;
    piece = std::max<size_t>(piece, 1);
    for (size_t at = 0; at < size; at += piece) {
        pieces.push_back(UploadPiece{tensor, src + at, at, std::min(piece, size - at)});
    }
}

static int env_threads(const char * name, int fallback) {
    if (const char * env = std::getenv(name)) {
        const int n = std::atoi(env);
        if (n > 0) return std::min(n, 64);
    }
    return fallback;
}

int upload_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return env_threads("LUCE_LOAD_THREADS", (int) std::clamp<unsigned>(hw ? hw : 4, 1, 16));
}

// Threads for one destination. An integrated GPU's copies are host memcpys
// that scale with threads. A discrete GPU behind a narrow link loses
// bandwidth to many interleaved copies: an R9700 on PCIe Gen3 x4 took 3.45 GB/s
// from 1-4 threads of staged copies and 2.35 GB/s from 8-16.
static int threads_for(ggml_backend_buffer_type_t buft) {
    int device = -1;
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        for (int d = 0; d < count && device < 0; ++d) {
            if (ggml_backend_cuda_buffer_type(d) == buft) device = d;
        }
    } else {
        (void) cudaGetLastError();
    }
    cudaDeviceProp prop{};
    const bool integrated = device >= 0 && cudaGetDeviceProperties(&prop, device) == cudaSuccess && prop.integrated;
    if (device >= 0 && !integrated) return env_threads("LUCE_LOAD_THREADS_DISCRETE", std::min(4, upload_threads()));
    return upload_threads();
}

// Starts the kernel's read-ahead of the source bytes, so a few upload threads
// do not wait on page faults one at a time when the file is not cached.
// Windows: no read-ahead hint; the copies fault the pages in.
static void prefetch(const std::vector<const UploadPiece *> & pieces) {
#if defined(_WIN32)
    (void) pieces;
#else
    static const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    for (const UploadPiece * p : pieces) {
        const uintptr_t lo = (uintptr_t) p->src & ~(uintptr_t) (page - 1);
        const uintptr_t hi = (uintptr_t) p->src + p->size;
        (void) madvise((void *) lo, hi - lo, MADV_WILLNEED);
    }
#endif
}

static void run_group(std::vector<const UploadPiece *> order, int threads) {
    // Largest first, so the long copies start early and the tail is short.
    std::stable_sort(order.begin(), order.end(),
                     [](const UploadPiece * a, const UploadPiece * b) { return a->size > b->size; });
    std::atomic<size_t> next{0};
    auto work = [&]() {
        for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < order.size();) {
            const UploadPiece & p = *order[i];
            ggml_backend_tensor_set(p.tensor, p.src, p.offset, p.size);
        }
    };
    threads = std::max(1, std::min<int>(threads, (int) order.size()));
    std::vector<std::thread> pool;
    pool.reserve((size_t) threads - 1);
    for (int t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (std::thread & t : pool) t.join();
}

size_t upload_pieces(const std::vector<UploadPiece> & pieces, int threads) {
    if (pieces.empty()) return 0;
    // One group per destination buffer type, each with its own threads, all at once.
    std::map<ggml_backend_buffer_type_t, std::vector<const UploadPiece *>> groups;
    size_t total = 0;
    for (const UploadPiece & p : pieces) {
        if (!p.tensor || !p.tensor->buffer || p.size == 0) continue;
        groups[ggml_backend_buffer_get_type(p.tensor->buffer)].push_back(&p);
        total += p.size;
    }
    std::vector<std::thread> runners;
    for (auto & [buft, group] : groups) {
        const int n = threads > 0 ? threads : threads_for(buft);
        if (n < 8) prefetch(group);
        runners.emplace_back(run_group, std::move(group), n);
    }
    for (std::thread & t : runners) t.join();
    return total;
}

}  // namespace luce::common
