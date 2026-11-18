// MoeHybridStreamEngine: a streamed cold expert's bytes land in GPU scratch
// exactly as the mmap holds them. Synthetic data only (no model files).

#include "CppUnitTestFramework.hpp"
#include "../../src/common/moe_hybrid_stream.h"
#include "../../src/common/moe_hybrid_storage.h"
#include "../../src/common/gpu_runtime_compat.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace luce::common;

namespace {
struct MoeStreamFixture {};
}

// Q4_K_M bytes per row: approximate as 0.5625 bytes/element (4.5 bits)
static size_t q4km_bytes(int rows, int cols) {
    // ggml block_q4_K: 144 bytes per 256 elements → 0.5625 B/elem
    return (size_t)rows * (size_t)cols * 9 / 16;
}

TEST_CASE(MoeStreamFixture, streamed_expert_bytes_land_in_scratch) {
#if defined(_WIN32)
    SKIP("the synthetic expert file uses POSIX mmap");
#else
    const int n_expert = 8;
    const int hidden = 256;
    const int ffn = 128;

    // Per-expert sizes (gate_up fused + down)
    const size_t gate_up_bytes = q4km_bytes(ffn * 2, hidden);
    const size_t down_bytes = q4km_bytes(hidden, ffn);
    const size_t expert_total_bytes = gate_up_bytes + down_bytes;
    const size_t file_size = expert_total_bytes * (size_t)n_expert;

    struct FdGuard {
        int fd = -1;
        ~FdGuard() { if (fd >= 0) close(fd); }
    } fd_guard;
    struct MmapGuard {
        void * addr = MAP_FAILED;
        size_t size = 0;
        ~MmapGuard() { if (addr != MAP_FAILED) munmap(addr, size); }
    } map_guard;
    struct BackendGuard {
        ggml_backend_t backend = nullptr;
        ~BackendGuard() { if (backend) ggml_backend_free(backend); }
    } backend_guard;
    MoeHybridStreamEngine engine;

    char tmppath[] = "/tmp/test_moe_stream_XXXXXX";
    fd_guard.fd = mkstemp(tmppath);
    REQUIRE(fd_guard.fd >= 0);
    unlink(tmppath);  // auto-delete on close
    REQUIRE(ftruncate(fd_guard.fd, (off_t)file_size) == 0);

    map_guard.addr = ::mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_guard.fd, 0);
    map_guard.size = file_size;
    REQUIRE(map_guard.addr != MAP_FAILED);

    // The pattern's period (251, prime) divides no expert size or region start,
    // so every expert's bytes differ and a wrong offset shows.
    auto * ptr = static_cast<uint8_t *>(map_guard.addr);
    for (size_t i = 0; i < file_size; ++i) {
        ptr[i] = (uint8_t)(i % 251);
    }
    REQUIRE(std::memcmp(ptr, ptr + 3 * gate_up_bytes, gate_up_bytes) != 0);

    LayerExpertRegions regions{};
    regions.fused_gate_up = true;
    regions.gate_up_exps.offset = 0;
    regions.gate_up_exps.size = gate_up_bytes * (size_t)n_expert;
    regions.expert_bytes_gate_up = gate_up_bytes;
    regions.down_exps.offset = regions.gate_up_exps.size;
    regions.down_exps.size = down_bytes * (size_t)n_expert;
    regions.expert_bytes_down = down_bytes;

    backend_guard.backend = ggml_backend_cuda_init(0);
    if (!backend_guard.backend) SKIP("no CUDA device");

    std::string err;
    REQUIRE(engine.init(backend_guard.backend, expert_total_bytes, &err));

    for (int id : {0, 3, n_expert - 1}) {
        REQUIRE(engine.stream_expert_sync(map_guard.addr, file_size, regions, id,
                                          backend_guard.backend, &err));
        std::vector<uint8_t> gate_up(gate_up_bytes), down(down_bytes);
        REQUIRE(cudaMemcpy(gate_up.data(), engine.scratch_gate_data(), gate_up_bytes,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(down.data(), engine.scratch_down_data(), down_bytes,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(std::memcmp(gate_up.data(),
                          ptr + regions.gate_up_exps.offset + (size_t)id * gate_up_bytes,
                          gate_up_bytes) == 0);
        CHECK(std::memcmp(down.data(),
                          ptr + regions.down_exps.offset + (size_t)id * down_bytes,
                          down_bytes) == 0);
        CHECK(engine.scratch_up_data() == nullptr);  // fused gate_up
    }

    // An expert that runs past the mapping is refused, not read out of bounds.
    CHECK(!engine.stream_expert_sync(map_guard.addr, file_size - 1, regions, n_expert - 1,
                                     backend_guard.backend, &err));
    CHECK(err.find("out of file bounds") != std::string::npos);

    engine.destroy();
#endif
}
