// Synthetic microbenchmark: MoE prefill streaming vs CPU cold path.
//
// This benchmark creates fake expert data in a temporary mmap'd file,
// builds layer regions, and measures DMA + GPU compute latency at various
// batch sizes (T = number of tokens).
//
// Usage: bench_moe_stream [--n-expert 128] [--n-cold 4] [--hidden 5120] [--ffn 3584]
//
// Requires: CUDA GPU.

#include "../src/common/moe_hybrid_stream.h"
#include "../src/common/moe_hybrid_storage.h"
#include "../src/common/gpu_runtime_compat.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace dflash::common;
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Q4_K_M bytes per row: approximate as 0.5625 bytes/element (4.5 bits)
static size_t q4km_bytes(int rows, int cols) {
    // ggml block_q4_K: 144 bytes per 256 elements → 0.5625 B/elem
    return (size_t)rows * (size_t)cols * 9 / 16;
}

int main(int argc, char ** argv) {
    int n_expert = 128;
    int n_cold = 4;
    int hidden = 5120;
    int ffn = 3584;
    int n_expert_used = 8;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n-expert") == 0 && i + 1 < argc) n_expert = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--n-cold") == 0 && i + 1 < argc) n_cold = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ffn") == 0 && i + 1 < argc) ffn = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--n-expert-used") == 0 && i + 1 < argc) n_expert_used = std::atoi(argv[++i]);
    }

    std::printf("bench_moe_stream: n_expert=%d n_cold=%d hidden=%d ffn=%d n_expert_used=%d\n",
                n_expert, n_cold, hidden, ffn, n_expert_used);

    // Compute per-expert sizes (gate_up fused + down)
    const size_t gate_up_bytes = q4km_bytes(ffn * 2, hidden);  // fused gate+up
    const size_t down_bytes = q4km_bytes(hidden, ffn);
    const size_t expert_total_bytes = gate_up_bytes + down_bytes;
    const size_t file_size = expert_total_bytes * (size_t)n_expert;

    std::printf("  per_expert: gate_up=%.2f MiB  down=%.2f MiB  total=%.2f MiB\n",
                gate_up_bytes / 1024.0 / 1024.0,
                down_bytes / 1024.0 / 1024.0,
                expert_total_bytes / 1024.0 / 1024.0);
    std::printf("  file_size=%.2f MiB\n", file_size / 1024.0 / 1024.0);

#if defined(_WIN32)
    std::fprintf(stderr, "bench_moe_stream: Windows not yet supported in this benchmark\n");
    return 1;
#else
    // Create a temporary file with random data
    char tmppath[] = "/tmp/bench_moe_stream_XXXXXX";
    int fd = mkstemp(tmppath);
    if (fd < 0) { perror("mkstemp"); return 1; }
    unlink(tmppath);  // auto-delete on close

    if (ftruncate(fd, (off_t)file_size) != 0) { perror("ftruncate"); close(fd); return 1; }

    void * mmap_addr = ::mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_addr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    // Fill with random data to ensure pages are faulted in
    std::printf("  filling temp file with random data...\n");
    std::mt19937 rng(42);
    auto * ptr = static_cast<uint8_t *>(mmap_addr);
    for (size_t off = 0; off < file_size; off += 4096) {
        uint32_t val = rng();
        std::memcpy(ptr + off, &val, sizeof(val));
    }

    // Build a fake LayerExpertRegions
    LayerExpertRegions regions{};
    regions.fused_gate_up = true;
    regions.gate_up_exps.offset = 0;
    regions.gate_up_exps.size = gate_up_bytes * (size_t)n_expert;
    regions.expert_bytes_gate_up = gate_up_bytes;
    regions.down_exps.offset = regions.gate_up_exps.size;
    regions.down_exps.size = down_bytes * (size_t)n_expert;
    regions.expert_bytes_down = down_bytes;

    // Init CUDA backend
    ggml_backend_t gpu_backend = ggml_backend_cuda_init(0);
    if (!gpu_backend) {
        std::fprintf(stderr, "Failed to init CUDA backend\n");
        munmap(mmap_addr, file_size);
        close(fd);
        return 1;
    }

    // Init stream engine
    MoeHybridStreamEngine engine;
    std::string err;
    if (!engine.init(gpu_backend, expert_total_bytes, &err)) {
        std::fprintf(stderr, "Stream engine init failed: %s\n", err.c_str());
        ggml_backend_free(gpu_backend);
        munmap(mmap_addr, file_size);
        close(fd);
        return 1;
    }
    std::printf("  stream engine ready: pinned=%.1f MiB scratch=%.1f MiB\n",
                engine.pinned_bytes() / 1024.0 / 1024.0,
                engine.scratch_bytes() / 1024.0 / 1024.0);

    // Benchmark: measure DMA time for streaming cold experts
    std::vector<int> test_T = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<int32_t> cold_ids(n_cold);
    for (int i = 0; i < n_cold; ++i) cold_ids[i] = i;  // first n_cold experts are "cold"

    std::printf("\n%-8s  %-12s  %-12s  %-12s\n", "T", "prefetch_ms", "stream_ms", "total_ms");
    std::printf("--------  ------------  ------------  ------------\n");

    for (int T : test_T) {
        // Warm up
        engine.prefetch_cold_experts(mmap_addr, file_size, regions, cold_ids.data(), n_cold);

        const int n_iter = 5;
        double prefetch_total = 0, stream_total = 0;

        for (int iter = 0; iter < n_iter; ++iter) {
            auto t0 = Clock::now();
            engine.prefetch_cold_experts(mmap_addr, file_size, regions, cold_ids.data(), n_cold);
            auto t1 = Clock::now();

            // Simulate streaming all cold experts
            for (int ci = 0; ci < n_cold; ++ci) {
                engine.stream_expert_sync(mmap_addr, file_size, regions, cold_ids[ci], gpu_backend, nullptr);
            }
            auto t2 = Clock::now();

            prefetch_total += elapsed_ms(t0, t1);
            stream_total += elapsed_ms(t1, t2);
        }

        double avg_prefetch = prefetch_total / n_iter;
        double avg_stream = stream_total / n_iter;
        std::printf("%-8d  %-12.3f  %-12.3f  %-12.3f\n", T, avg_prefetch, avg_stream, avg_prefetch + avg_stream);
    }

    // Cleanup
    engine.destroy();
    ggml_backend_free(gpu_backend);
    munmap(mmap_addr, file_size);
    close(fd);

    std::printf("\nDone.\n");
    return 0;
#endif
}
