// Page-out / slot-reuse ordering on a real GPU. page_out() queues the device
// to host copy and the block zeroing on the pager's own stream and hands the
// block back at once; the next forward step then writes new K/V into the
// reused slot on the backend's compute stream. ggml-cuda streams are created
// non-blocking, so nothing orders them against the pager's stream. Without an
// ordering the new write can land before the queued copy (the host archive
// receives the NEW bytes) or before the queued zeroing (the new row is wiped).
//
// The stress appends chunk after chunk through a full pool and writes each new
// chunk on a separate non-blocking stream right after slot_for(), as a backend
// does. The interval under test, from the page-out inside slot_for() to the
// replacement write, is never synchronized. The opposite dependency is
// established explicitly: before each slot_for() (which may evict an earlier
// chunk) the producer's previous write is waited for, so the test does not rely
// on the implicit sync of a pinned allocation. Afterwards every chunk is checked:
// resident chunks straight from the pool, paged-out chunks by recalling them.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerCudaRaceFixture {};

constexpr int kRHeadDim = 128, kRPool = 512, kRNHeadKv = 8, kRNLayer = 8, kRChunk = 64;

uint8_t rpat(int c, size_t i) { return (uint8_t)(c * 53 + i * 31 + 7 + (i >> 8) * 13); }

// A failed setup step must fail the test with a message, never continue on a
// null handle.
void cu(cudaError_t e, const char * what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void need(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(what);
}

struct CudaHarness {
    ggml_context *        ctx = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    CudaHarness() {
        try {
            backend = ggml_backend_cuda_init(0);
            need(backend != nullptr, "ggml_backend_cuda_init(0) failed");
            ggml_init_params ip{};
            ip.mem_size = (size_t)(kRNLayer * 2 + 8) * ggml_tensor_overhead();
            ip.no_alloc = true;
            ctx = ggml_init(ip);
            need(ctx != nullptr, "ggml_init failed");
            for (int l = 0; l < kRNLayer; l++) {
                k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kRHeadDim, kRPool, kRNHeadKv));
                v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kRHeadDim, kRPool, kRNHeadKv));
                need(k.back() && v.back(), "ggml_new_tensor_3d failed");
            }
            buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            need(buf != nullptr, "ggml_backend_alloc_ctx_tensors failed");
        } catch (...) {
            release();
            throw;
        }
    }
    ~CudaHarness() { release(); }
    CudaHarness(const CudaHarness &) = delete;
    CudaHarness & operator=(const CudaHarness &) = delete;

    void release() {
        if (buf)     ggml_backend_buffer_free(buf);
        if (ctx)     ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
        buf = nullptr; ctx = nullptr; backend = nullptr;
    }
    size_t chunk_bytes() const {
        return (size_t)kRNLayer * 2 * kRNHeadKv * kRChunk * k[0]->nb[1];
    }
};

// Visit the per-(layer, K/V, head) segments of `block` in the pager's order.
template <class F>
void for_segments(CudaHarness & h, int block, F && fn) {
    size_t n = 0;
    for (int l = 0; l < kRNLayer; l++)
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * t = kv == 0 ? h.k[l] : h.v[l];
            for (int hh = 0; hh < kRNHeadKv; hh++) {
                const size_t off = (size_t)block * kRChunk * t->nb[1] + (size_t)hh * t->nb[2];
                const size_t len = (size_t)kRChunk * t->nb[1];
                fn((uint8_t *)t->data + off, n, len);
                n += len;
            }
        }
}

KvFlashConfig race_cfg() {
    KvFlashConfig cfg;
    cfg.pool_tokens = kRPool;
    cfg.chunk_tokens = kRChunk;
    cfg.sink_chunks = 1;
    cfg.tail_window_chunks = 4;
    return cfg;
}

// Returns the number of chunks whose bytes differ from rpat after the run;
// throws on any setup or CUDA failure.
// `wire_stream`: register the compute stream with the pager (device-side wait,
// what Qwen35Backend does) instead of relying on the default host wait.
int run_round(int n_chunks, bool wire_stream) {
    CudaHarness h;
    KvFlashPager p;
    need(p.attach(race_cfg(), h.k, h.v), "pager attach failed");

    cudaStream_t cs = nullptr;                       // the backend's compute stream is non-blocking
    cu(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    std::vector<uint8_t *> staging;
    int bad = 0;
    try {
        if (wire_stream) p.set_compute_stream(cs);
        for (int c = 0; c < n_chunks; c++) {         // all host work up front
            uint8_t * st = nullptr;
            cu(cudaMallocHost(&st, h.chunk_bytes()), "cudaMallocHost(staging)");
            staging.push_back(st);
            for (size_t i = 0; i < h.chunk_bytes(); i++) st[i] = rpat(c, i);
        }
        for (int c = 0; c < n_chunks; c++) {
            // Producer -> eviction: the previous chunk's write is complete before
            // this slot_for() can choose it as a victim.
            cu(cudaStreamSynchronize(cs), "cudaStreamSynchronize(compute)");
            need(p.slot_for((int64_t)c * kRChunk) >= 0, "slot_for failed");   // may page a victim out
            // Page-out -> replacement write: deliberately NOT synchronized.
            for_segments(h, p.block_of(c), [&](uint8_t * dev, size_t n, size_t len) {
                cu(cudaMemcpyAsync(dev, staging[(size_t)c] + n, len, cudaMemcpyHostToDevice, cs),
                   "cudaMemcpyAsync(new chunk)");
            });
        }
        cu(cudaStreamSynchronize(cs), "cudaStreamSynchronize(compute)");
        p.synchronize_paging();
        cu(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        std::vector<uint8_t> got(h.chunk_bytes());
        auto check = [&](int c) {       // chunk c is resident: compare its block with rpat
            for_segments(h, p.block_of(c), [&](uint8_t * dev, size_t n, size_t len) {
                cu(cudaMemcpy(got.data() + n, dev, len, cudaMemcpyDeviceToHost), "cudaMemcpy(readback)");
            });
            for (size_t i = 0; i < got.size(); i++) if (got[i] != rpat(c, i)) return false;
            return true;
        };
        std::vector<int> resident;
        for (int c = 0; c < n_chunks; c++) {
            if (!p.is_resident(c)) continue;
            resident.push_back(c);
            bad += !check(c);
        }
        for (int c : resident) need(p.page_out(c), "page_out failed");   // archive them, freeing every block
        p.synchronize_paging();
        for (int c = 0; c < n_chunks; c++) {         // recall each chunk and read it back
            need(p.page_in(c), "page_in failed");
            p.synchronize_paging();
            bad += !check(c);
            need(p.page_out(c), "page_out failed");
            p.synchronize_paging();
        }
    } catch (...) {
        cudaDeviceSynchronize();
        for (uint8_t * s : staging) cudaFreeHost(s);
        cudaStreamDestroy(cs);
        throw;
    }
    for (uint8_t * s : staging) cu(cudaFreeHost(s), "cudaFreeHost");
    cu(cudaStreamDestroy(cs), "cudaStreamDestroy");
    return bad;
}

int race_rounds() {
    const char * env = std::getenv("LUCE_KVFLASH_RACE_ROUNDS");
    if (!env) return 40;
    char * end = nullptr;
    errno = 0;
    const long v = std::strtol(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || v <= 0 || v > 100000)
        throw std::runtime_error(std::string("invalid LUCE_KVFLASH_RACE_ROUNDS='") + env + "' (want 1..100000)");
    return (int)v;
}

// Corrupted chunks over all rounds. Returns "" or the failure message.
std::string stress(bool wire_stream, int & corrupted_chunks) {
    try {
        const int rounds = race_rounds();
        int corrupted_rounds = 0;
        corrupted_chunks = 0;
        for (int r = 0; r < rounds; r++) {
            const int bad = run_round(24, wire_stream);
            corrupted_rounds += bad > 0;
            corrupted_chunks += bad;
        }
        std::fprintf(stderr, "[race] wire_stream=%d rounds=%d corrupted_rounds=%d corrupted_chunks=%d\n",
                     (int)wire_stream, rounds, corrupted_rounds, corrupted_chunks);
    } catch (const std::exception & e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST_CASE(KvflashPagerCudaRaceFixture, reused_slot_write_is_ordered_after_queued_page_out) {
    if (ggml_backend_cuda_get_device_count() < 1) SKIP("no CUDA device");
    int bad = -1;
    CHECK_EQUAL(stress(/*wire_stream=*/false, bad), std::string());
    CHECK_EQUAL(bad, 0);
}

TEST_CASE(KvflashPagerCudaRaceFixture, reused_slot_write_is_ordered_with_registered_compute_stream) {
    if (ggml_backend_cuda_get_device_count() < 1) SKIP("no CUDA device");
    int bad = -1;
    CHECK_EQUAL(stress(/*wire_stream=*/true, bad), std::string());
    CHECK_EQUAL(bad, 0);
}

// attach() must drop a registered compute stream: the caller may destroy it
// before reattaching, and a stale handle must never be waited on. A failed
// wait on a destroyed stream aborts the process, so a regression here shows up
// as a crashed test run, not a failed check; the test detects only what the
// CUDA runtime reports for a destroyed handle (it may not report it, and on
// HIP builds the registered stream is ignored altogether). It needs no GPU fault.
TEST_CASE(KvflashPagerCudaRaceFixture, reattach_forgets_a_destroyed_compute_stream) {
    if (ggml_backend_cuda_get_device_count() < 1) SKIP("no CUDA device");
    std::string err;
    try {
        CudaHarness h;
        KvFlashPager p;
        need(p.attach(race_cfg(), h.k, h.v), "pager attach failed");
        cudaStream_t cs = nullptr;
        cu(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        p.set_compute_stream(cs);
        cu(cudaStreamDestroy(cs), "cudaStreamDestroy");
        need(p.attach(race_cfg(), h.k, h.v), "pager reattach failed");
        for (int c = 0; c < 24; c++)                 // fills the pool and pages out
            need(p.slot_for((int64_t)c * kRChunk) >= 0, "slot_for failed after reattach");
        p.synchronize_paging();
        cu(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    } catch (const std::exception & e) {
        err = e.what();
    }
    CHECK_EQUAL(err, std::string());
}
