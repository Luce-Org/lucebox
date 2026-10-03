// The estimate of GPU driver pages that MemAvailable does not count.
#include "common/gpu_page_pool.h"

#include <cstdio>

using luce::common::reclaimable_gpu_page_pool_bytes;

static int failures = 0;
static void check(bool condition, const char * message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

int main() {
    constexpr uint64_t GiB = 1ULL << 30;
    // 100,000,000 kB total, 31,500,000 kB attributed -> 68,500,000 kB unattributed.
    const char * meminfo =
        "MemTotal:       100000000 kB\n"
        "MemFree:         10000000 kB\n"
        "MemAvailable:    28000000 kB\n"
        "Buffers:                0 kB\n"
        "Cached:          20000000 kB\n"
        "AnonPages:        1000000 kB\n"
        "Shmem:             300000 kB\n"   // already inside Cached, must not be counted twice
        "Slab:              500000 kB\n"
        "HugePages_Total:        0\n"
        "Hugepagesize:        2048 kB\n";
    const uint64_t unattributed = 68500000ULL * 1024;

    check(reclaimable_gpu_page_pool_bytes(meminfo, 0) == unattributed - GiB,
          "idle GPU: everything unattributed, less the margin for other drivers");
    check(reclaimable_gpu_page_pool_bytes(meminfo, 5 * GiB) == unattributed - 6 * GiB,
          "live GPU buffers are in use, not reclaimable");
    check(reclaimable_gpu_page_pool_bytes(meminfo, unattributed) == 0,
          "all unattributed memory is live GPU buffers");
    check(reclaimable_gpu_page_pool_bytes(meminfo, unattributed + 9 * GiB) == 0, "never negative");

    const char * huge =
        "MemTotal:       100000000 kB\n"
        "MemFree:         10000000 kB\n"
        "HugePages_Total:    20000\n"
        "Hugepagesize:        2048 kB\n";  // 40,960,000 kB of huge pages are attributed
    check(reclaimable_gpu_page_pool_bytes(huge, 0) == (100000000ULL - 10000000 - 40960000) * 1024 - GiB,
          "huge pages count as attributed");

    const char * mixed_huge =
        "MemTotal:       100000000 kB\n"
        "MemFree:         10000000 kB\n"
        "HugePages_Total:    20000\n"
        "Hugepagesize:        2048 kB\n"
        "Hugetlb:         50000000 kB\n";  // includes a pool of another page size
    check(reclaimable_gpu_page_pool_bytes(mixed_huge, 0) == (100000000ULL - 10000000 - 50000000) * 1024 - GiB,
          "Hugetlb covers huge page pools of every size");

    check(reclaimable_gpu_page_pool_bytes("MemFree: 5 kB\n", 0) == 0, "no MemTotal, no estimate");
    check(reclaimable_gpu_page_pool_bytes("", 0) == 0 && reclaimable_gpu_page_pool_bytes(nullptr, 0) == 0,
          "empty input");

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
