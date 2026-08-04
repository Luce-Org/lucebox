// Read-only SSD benchmark for the routed-MoE scheduler.
//
// Usage:
//   bench_moe_nvme_io MODEL_FILE [expert_mib] [working_set] [rounds] [batch]
//
// DFLASH_MOE_NVME_BACKEND and DFLASH_MOE_NVME_DIRECT select the same path as
// production. The benchmark treats two large, disjoint ranges of the input as
// fused gate/up and down expert tensors. It never modifies the input file.

#include "common/moe_nvme_scheduler.h"

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
#include <malloc.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace dflash::common;

namespace {

bool aligned_allocate(void ** ptr, size_t bytes, void *) {
#if defined(_WIN32)
    *ptr = _aligned_malloc(bytes, 4096);
    return *ptr != nullptr;
#else
    return ::posix_memalign(ptr, 4096, bytes) == 0;
#endif
}

void aligned_free(void * ptr, void *) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

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
    if (argc < 2 || argc > 6) {
        std::fprintf(stderr,
            "usage: %s MODEL_FILE [expert_mib=24] [working_set=64] "
            "[rounds=4] [batch=8]\n", argv[0]);
        return 2;
    }

    uint64_t expert_mib = 24;
    uint64_t requested_working_set = 64;
    uint64_t rounds = 4;
    uint64_t requested_batch = 8;
    uint64_t * values[] = {&expert_mib, &requested_working_set, &rounds,
                           &requested_batch};
    for (int i = 2; i < argc; ++i) {
        if (!parse_positive(argv[i], *values[i - 2])) {
            std::fprintf(stderr, "invalid positive integer: %s\n", argv[i]);
            return 2;
        }
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
        if (fd >= 0) {
#if defined(_WIN32)
            ::_close(fd);
#else
            ::close(fd);
#endif
        }
        return 1;
    }

    const size_t file_bytes = (size_t) st.st_size;
    const size_t alignment = 4096;
    const size_t requested_payload = (size_t) expert_mib * 1024 * 1024;
    // Keep both tensor slices naturally aligned so direct-I/O throughput is
    // measured rather than an artificial alignment-amplification worst case.
    const size_t gate_up_bytes = align_down(requested_payload * 2 / 3, alignment);
    const size_t down_bytes = align_down(requested_payload - gate_up_bytes, alignment);
    const size_t payload_bytes = gate_up_bytes + down_bytes;
    if (gate_up_bytes == 0 || down_bytes == 0 || payload_bytes == 0) {
        std::fprintf(stderr, "expert payload is too small\n");
        return 2;
    }

    const size_t possible_experts = file_bytes / payload_bytes;
    const size_t working_set = std::min<size_t>(
        (size_t) requested_working_set, possible_experts);
    if (working_set < 2) {
        std::fprintf(stderr, "file is too small for two %.1f MiB experts\n",
                     payload_bytes / 1024.0 / 1024.0);
        return 2;
    }

    LayerExpertRegions region;
    region.fused_gate_up = true;
    region.expert_bytes_gate_up = gate_up_bytes;
    region.expert_bytes_down = down_bytes;
    region.gate_up_exps = {0, gate_up_bytes * working_set};
    region.down_exps = {region.gate_up_exps.size, down_bytes * working_set};

    MoeNvmeConfig config = MoeNvmeConfig::from_env();
    const size_t batch = std::max<size_t>(1, std::min<size_t>(
        {(size_t) requested_batch, working_set, (size_t) config.host_slots}));
    MoeNvmeScheduler scheduler;
    std::string error;
    if (!scheduler.init(config, payload_bytes, aligned_allocate, aligned_free,
                        nullptr, &error) ||
        !scheduler.bind_source({nullptr, file_bytes, fd}, {region}, &error)) {
        std::fprintf(stderr, "scheduler initialization failed: %s\n", error.c_str());
        return 1;
    }

#if defined(__linux__)
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif

    const uint64_t total_requests = rounds * working_set;
    uint64_t completed = 0;
    uint64_t checksum = 0;
    std::vector<double> batch_ms;
    batch_ms.reserve((size_t) ((total_requests + batch - 1) / batch));
    const auto wall_begin = std::chrono::steady_clock::now();
    while (completed < total_requests) {
        const size_t count = (size_t) std::min<uint64_t>(batch, total_requests - completed);
        std::vector<int> experts(count);
        for (size_t i = 0; i < count; ++i) {
            // Rotate every round and walk the full working set. This avoids
            // measuring a tiny cache-resident subset while remaining repeatable.
            const uint64_t ordinal = completed + i;
            const uint64_t round = ordinal / working_set;
            const uint64_t within = ordinal % working_set;
            experts[i] = (int) ((within + round * (working_set / 2 + 1)) % working_set);
        }

        const auto batch_begin = std::chrono::steady_clock::now();
        for (int expert : experts) {
            if (!scheduler.request(0, expert, MoeNvmePriority::Demand, &error)) {
                std::fprintf(stderr, "request failed: %s\n", error.c_str());
                return 1;
            }
        }
        for (int expert : experts) {
            MoeNvmeLease lease;
            if (!scheduler.acquire(0, expert, lease, &error)) {
                std::fprintf(stderr, "acquire failed: %s\n", error.c_str());
                return 1;
            }
            for (int span = 0; span < lease.layout().span_count; ++span) {
                const MoeExpertIoSpan & io = lease.layout().spans[span];
                const uint8_t * data = lease.data() + io.buffer_offset;
                checksum += data[0];
                checksum += data[io.bytes - 1];
            }
        }
        const auto batch_end = std::chrono::steady_clock::now();
        batch_ms.push_back(std::chrono::duration<double, std::milli>(
            batch_end - batch_begin).count());
        completed += count;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(wall_end - wall_begin).count();
    const MoeNvmeStats stats = scheduler.stats();
    std::sort(batch_ms.begin(), batch_ms.end());
    const size_t p50_index = (batch_ms.size() - 1) / 2;
    const size_t p95_index = (batch_ms.size() - 1) * 95 / 100;

    std::printf("backend=%s direct=%s file_gib=%.2f expert_mib=%.2f "
                "working_set=%zu rounds=%" PRIu64 " batch=%zu slots=%d\n",
                scheduler.effective_backend_name(),
                scheduler.direct_io_active() ? "yes" : "no", gib(file_bytes),
                payload_bytes / 1024.0 / 1024.0, working_set, rounds, batch,
                scheduler.slot_count());
    std::printf("elapsed_s=%.6f payload_gib=%.3f physical_gib=%.3f "
                "payload_gib_s=%.3f physical_gib_s=%.3f experts_s=%.2f\n",
                seconds, gib(stats.payload_bytes), gib(stats.physical_bytes),
                seconds > 0 ? gib(stats.payload_bytes) / seconds : 0.0,
                seconds > 0 ? gib(stats.physical_bytes) / seconds : 0.0,
                seconds > 0 ? stats.payload_bytes / (double) payload_bytes / seconds : 0.0);
    std::printf("batch_ms_p50=%.3f batch_ms_p95=%.3f cache_hits=%" PRIu64
                " dedupe=%" PRIu64 " evictions=%" PRIu64
                " errors=%" PRIu64 " checksum=%" PRIu64 "\n",
                batch_ms[p50_index], batch_ms[p95_index], stats.cache_hits,
                stats.inflight_deduplications, stats.evictions, stats.errors,
                checksum);

    scheduler.destroy();
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
    return stats.errors == 0 ? 0 : 1;
}
