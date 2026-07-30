// Read-only end-to-end NVMe -> pinned host -> HIP/CUDA device benchmark.
// It exercises MoeHybridStreamEngine itself on either Lucebox GPU.

#include "common/moe_hybrid_stream.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace dflash::common;

namespace {

bool parse_positive(const char * text, uint64_t & out) {
    if (!text || !text[0] || text[0] == '-') return false;
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) return false;
    out = (uint64_t) value;
    return true;
}

size_t align_down(size_t value, size_t alignment) {
    return value & ~(alignment - 1);
}

double gib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0 * 1024.0);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 7) {
        std::fprintf(stderr,
            "usage: %s MODEL_FILE [device=0] [expert_mib=24] "
            "[working_set=64] [rounds=4] [batch=8]\n", argv[0]);
        return 2;
    }

    uint64_t device_arg = 0;
    uint64_t expert_mib = 24;
    uint64_t requested_working_set = 64;
    uint64_t rounds = 4;
    uint64_t requested_batch = 8;
    uint64_t * values[] = {&device_arg, &expert_mib, &requested_working_set,
                           &rounds, &requested_batch};
    for (int i = 2; i < argc; ++i) {
        // Device zero is valid; other arguments must be positive.
        if (i == 2 && std::strcmp(argv[i], "0") == 0) continue;
        if (!parse_positive(argv[i], *values[i - 2])) {
            std::fprintf(stderr, "invalid integer: %s\n", argv[i]);
            return 2;
        }
    }
    if (device_arg >= (uint64_t) ggml_backend_cuda_get_device_count()) {
        std::fprintf(stderr, "GPU device is out of range\n");
        return 2;
    }
    if (expert_mib > (uint64_t) std::numeric_limits<size_t>::max() / (1024 * 1024)) {
        std::fprintf(stderr, "expert size is too large\n");
        return 2;
    }

#if defined(_WIN32)
    const int fd = ::_open(argv[1], _O_RDONLY | _O_BINARY);
    struct _stat64 st{};
    if (fd < 0 || ::_fstat64(fd, &st) != 0 || st.st_size <= 0) {
#else
    const int fd = ::open(argv[1], O_RDONLY | O_CLOEXEC);
    struct stat st{};
    if (fd < 0 || ::fstat(fd, &st) != 0 || st.st_size <= 0) {
#endif
        std::fprintf(stderr, "cannot open input file: %s\n", std::strerror(errno));
        return 1;
    }

    const size_t file_bytes = (size_t) st.st_size;
    const size_t alignment = 4096;
    const size_t requested_payload = (size_t) expert_mib * 1024 * 1024;
    const size_t gate_up_bytes = align_down(requested_payload * 2 / 3, alignment);
    const size_t down_bytes = align_down(requested_payload - gate_up_bytes, alignment);
    const size_t payload_bytes = gate_up_bytes + down_bytes;
    const size_t possible_experts = payload_bytes ? file_bytes / payload_bytes : 0;
    const size_t working_set = std::min<size_t>(
        (size_t) requested_working_set, possible_experts);
    if (gate_up_bytes == 0 || down_bytes == 0 || working_set < 2) {
        std::fprintf(stderr, "file or expert payload is too small\n");
        return 2;
    }

    LayerExpertRegions region;
    region.fused_gate_up = true;
    region.expert_bytes_gate_up = gate_up_bytes;
    region.expert_bytes_down = down_bytes;
    region.gate_up_exps = {0, gate_up_bytes * working_set};
    region.down_exps = {region.gate_up_exps.size, down_bytes * working_set};

    ggml_backend_t backend = ggml_backend_cuda_init((int) device_arg);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize GPU backend\n");
        return 1;
    }

    MoeHybridStorage storage;
    storage.mmap_size = file_bytes;
#if defined(_WIN32)
    storage.mmap_fd = -1;
#else
    storage.mmap_fd = ::dup(fd);
#endif
    storage.layer_regions = {region};
    MoeHybridStreamEngine engine;
    std::string error;
    if (!engine.init(backend, payload_bytes, storage, &error)) {
        std::fprintf(stderr, "stream engine initialization failed: %s\n", error.c_str());
        ggml_backend_free(backend);
        return 1;
    }

    const MoeNvmeConfig io_config = MoeNvmeConfig::from_env();
    const size_t batch = std::max<size_t>(1, std::min<size_t>(
        {(size_t) requested_batch, working_set, (size_t) io_config.host_slots}));
    const uint64_t total_requests = rounds * working_set;
    uint64_t completed = 0;
    std::vector<double> batch_ms;
    batch_ms.reserve((size_t) ((total_requests + batch - 1) / batch));
    const auto wall_begin = std::chrono::steady_clock::now();
    while (completed < total_requests) {
        const size_t count = (size_t) std::min<uint64_t>(batch, total_requests - completed);
        std::vector<int32_t> experts(count);
        for (size_t i = 0; i < count; ++i) {
            const uint64_t ordinal = completed + i;
            const uint64_t round = ordinal / working_set;
            const uint64_t within = ordinal % working_set;
            experts[i] = (int32_t) ((within + round * (working_set / 2 + 1)) % working_set);
        }
        const auto batch_begin = std::chrono::steady_clock::now();
        engine.request_experts(0, experts.data(), (int) experts.size(),
                               MoeNvmePriority::Demand);
        int staged_slot = -1;
        if (!engine.stage_expert_cached_async(0, experts[0], &staged_slot, &error)) {
            std::fprintf(stderr, "initial stage failed: %s\n", error.c_str());
            return 1;
        }
        for (size_t i = 0; i < count; ++i) {
            const int current_slot = staged_slot;
            if (i + 1 < count) {
                int next_slot = -1;
                if (!engine.stage_expert_cached_async(
                        0, experts[i + 1], &next_slot, &error)) {
                    std::fprintf(stderr, "pipelined stage failed: %s\n", error.c_str());
                    return 1;
                }
                staged_slot = next_slot;
            }
            if (!engine.activate_device_slot(current_slot, &error)) {
                std::fprintf(stderr, "activation failed: %s\n", error.c_str());
                return 1;
            }
            engine.release_device_slot(current_slot);
        }
        const auto batch_end = std::chrono::steady_clock::now();
        batch_ms.push_back(std::chrono::duration<double, std::milli>(
            batch_end - batch_begin).count());
        completed += count;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(wall_end - wall_begin).count();
    const MoeNvmeStats stats = engine.io_stats();
    std::sort(batch_ms.begin(), batch_ms.end());
    const size_t p50_index = (batch_ms.size() - 1) / 2;
    const size_t p95_index = (batch_ms.size() - 1) * 95 / 100;
    char description[256] = {};
    ggml_backend_cuda_get_device_description((int) device_arg, description,
                                              sizeof(description));

    std::printf("device=%" PRIu64 " description=%s backend=%s "
                "expert_mib=%.2f working_set=%zu rounds=%" PRIu64
                " batch=%zu host_mib=%.1f device_mib=%.1f\n",
                device_arg, description, engine.io_backend_name(),
                payload_bytes / 1024.0 / 1024.0, working_set, rounds, batch,
                engine.pinned_bytes() / 1024.0 / 1024.0,
                engine.scratch_bytes() / 1024.0 / 1024.0);
    std::printf("elapsed_s=%.6f transferred_gib=%.3f pipeline_gib_s=%.3f "
                "experts_s=%.2f batch_ms_p50=%.3f batch_ms_p95=%.3f "
                "physical_gib=%.3f errors=%" PRIu64 "\n",
                seconds, gib(stats.payload_bytes),
                seconds > 0 ? gib(stats.payload_bytes) / seconds : 0.0,
                seconds > 0 ? total_requests / seconds : 0.0,
                batch_ms[p50_index], batch_ms[p95_index],
                gib(stats.physical_bytes), stats.errors);

    engine.destroy();
    ggml_backend_free(backend);
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
    return stats.errors == 0 ? 0 : 1;
}
