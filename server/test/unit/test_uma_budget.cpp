#include "CppUnitTestFramework.hpp"
#include "../../deps/llama.cpp/ggml/src/ggml-cuda/uma-budget.h"

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <filesystem>
#include <fstream>
#include <unistd.h>
#endif

namespace {
struct UmaBudget {};
constexpr size_t GiB = size_t(1) << 30;
// A 125 GiB integrated-GPU box, as /proc/meminfo reports MemTotal.
constexpr size_t kStrixRam = 125 * GiB + GiB / 10;
}

// The overcommit this check exists for: a 103.9 GiB managed buffer while
// MemAvailable says ~92 GiB. It must fail up front instead of ending in the
// OOM killer.
TEST_CASE(UmaBudget, rejects_allocation_larger_than_available_ram) {
    const size_t size = 103 * GiB + 9 * GiB / 10;
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(size, 92 * GiB, kStrixRam));
}

// The case the managed path is for: a large model on a box whose RAM is free.
TEST_CASE(UmaBudget, admits_large_model_when_ram_is_free) {
    CHECK_TRUE(ggml_cuda_uma_alloc_fits(86 * GiB, 118 * GiB, kStrixRam));
}

// The reserve kept free is 1/16 of RAM, at least 1 GiB and at most 8 GiB:
// 1, 4 and 8 GiB on 8, 64 and 256 GiB boxes with half their RAM available.
TEST_CASE(UmaBudget, keeps_reserve_free) {
    struct Box { size_t ram; size_t reserve; };
    for (const Box box : {Box{8 * GiB, 1 * GiB}, Box{64 * GiB, 4 * GiB}, Box{256 * GiB, 8 * GiB}}) {
        const size_t available = box.ram / 2;
        CHECK_TRUE(ggml_cuda_uma_alloc_fits(available - box.reserve, available, box.ram));
        CHECK_FALSE(ggml_cuda_uma_alloc_fits(available - box.reserve + 1, available, box.ram));
    }
}

// Less available than the reserve, including none at all, rejects any
// allocation without the subtraction wrapping around; a huge size does not
// overflow either.
TEST_CASE(UmaBudget, near_empty_ram_and_huge_sizes_reject) {
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(1, GiB / 4, kStrixRam));
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(1, 0, kStrixRam));
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(SIZE_MAX, 100 * GiB, kStrixRam));
}

// After a large run the GPU driver keeps the freed pages in its pool, which
// MemAvailable leaves out: 58 GiB available, 61 GiB held by the driver. An
// 86 GiB model still fits, because the kernel hands the pool back under
// pressure.
TEST_CASE(UmaBudget, counts_the_driver_page_pool_after_a_run) {
    // 125 GiB total; 64 GiB attributed (free, cache, anon, slab), so 61 GiB
    // belong to nothing /proc/meminfo lists.
    const char * meminfo =
        "MemTotal:       131072000 kB\n"
        "MemFree:         52428800 kB\n"
        "MemAvailable:    60817408 kB\n"
        "Cached:           8388608 kB\n"
        "AnonPages:        5242880 kB\n"
        "Slab:             1048576 kB\n"
        "HugePages_Total:        0\n"
        "Hugepagesize:        2048 kB\n";
    const uint64_t no_live_gpu_buffers = 0;
    uint64_t available = 0;
    CHECK_TRUE(ggml_cuda_uma_available_bytes(meminfo, &no_live_gpu_buffers, nullptr, &available));
    CHECK_TRUE(ggml_cuda_uma_alloc_fits(86 * GiB, (size_t) available, kStrixRam));

    // Without knowing which GPU buffers are live, the pool cannot be told
    // apart from them and is not counted.
    CHECK_TRUE(ggml_cuda_uma_available_bytes(meminfo, nullptr, nullptr, &available));
    CHECK_TRUE(available == 60817408ull * 1024);
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(86 * GiB, (size_t) available, kStrixRam));
}

// A missing MemAvailable is unknown (the caller keeps the old behaviour and
// admits); a MemAvailable of zero is a reading, and with no driver pool to
// add, nothing fits in it.
TEST_CASE(UmaBudget, unknown_ram_differs_from_no_ram) {
    uint64_t available = 1;
    CHECK_FALSE(ggml_cuda_uma_available_bytes("MemTotal: 131072000 kB\n", nullptr, nullptr, &available));
    CHECK_FALSE(ggml_cuda_uma_available_bytes(nullptr, nullptr, nullptr, &available));
    CHECK_TRUE(ggml_cuda_uma_available_bytes("MemTotal: 131072000 kB\nMemAvailable: 0 kB\n", nullptr, nullptr, &available));
    CHECK_TRUE(available == 0);
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(GiB, (size_t) available, kStrixRam));
}

// A container sees the host's MemAvailable, but the pages are charged to its
// cgroup as they are mapped: a 60 GiB managed buffer in a 4 GiB cgroup on a
// free 125 GiB box was OOM-killed during the allocation. The room under the
// limit bounds the budget.
TEST_CASE(UmaBudget, cgroup_limit_bounds_the_budget) {
    const char * meminfo =
        "MemTotal:       131072000 kB\n"
        "MemFree:        110000000 kB\n"
        "MemAvailable:   123731968 kB\n";
    const uint64_t no_live_gpu_buffers = 0;
    const uint64_t room = 4 * (uint64_t) GiB;
    uint64_t available = 0;
    CHECK_TRUE(ggml_cuda_uma_available_bytes(meminfo, &no_live_gpu_buffers, &room, &available));
    CHECK_TRUE(available == room);
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(60 * GiB, (size_t) available, kStrixRam));

    const uint64_t roomy = 64 * (uint64_t) GiB;
    CHECK_TRUE(ggml_cuda_uma_available_bytes(meminfo, &no_live_gpu_buffers, &roomy, &available));
    CHECK_TRUE(ggml_cuda_uma_alloc_fits(40 * GiB, (size_t) available, kStrixRam));
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(60 * GiB, (size_t) available, kStrixRam));
}

// Strix Halo under 6.17.0-1033-oem reported 121.6 of 1280 GiB free however
// much was allocated. With 101.8 GiB of experts in place, 18.7 GiB of RAM
// available and 365 MiB of its 512 MiB carve left, the device can take the
// carve left and that RAM: 19.1 GiB, not 121.6. DeepSeek's sizing keeps the
// managed-buffer reserve (7.8 GiB) out of the RAM as well: 11.2 GiB.
TEST_CASE(UmaBudget, bounds_a_free_that_does_not_follow_allocations) {
    const uint64_t carve = 512ULL << 20, carve_used = 147ULL << 20;
    const uint64_t ram_available = 18 * (uint64_t) GiB + 7 * GiB / 10;
    size_t free = 121 * GiB + 6 * GiB / 10, total = 1280 * GiB;
    ggml_cuda_uma_bound_device_memory(&free, &total, carve, carve_used, ram_available, kStrixRam);
    CHECK_TRUE(free == (carve - carve_used) + ram_available);
    CHECK_TRUE(total == carve + kStrixRam);

    const uint64_t ram_room = ram_available - ggml_cuda_uma_reserve_bytes(kStrixRam);
    free = 121 * GiB + 6 * GiB / 10;
    ggml_cuda_uma_bound_device_memory(&free, &total, carve, carve_used, ram_room, kStrixRam);
    CHECK_TRUE(free == (carve - carve_used) + ram_room);
}

// Under 6.17.0-1032-oem the same box reported 125.09 GiB in all and a free
// that followed allocations (11.40 GiB with the experts in place), under what
// the RAM allows: both figures stay as they are.
TEST_CASE(UmaBudget, keeps_a_free_that_follows_allocations) {
    size_t free = 11 * GiB + 4 * GiB / 10, total = kStrixRam;
    ggml_cuda_uma_bound_device_memory(&free, &total, 512ULL << 20, 147ULL << 20, 20 * (uint64_t) GiB, kStrixRam);
    CHECK_TRUE(free == 11 * GiB + 4 * GiB / 10);
    CHECK_TRUE(total == kStrixRam);
}

// A runtime that reports less than RAM allows (a GTT limit below RAM, or DGX
// Spark's, which leaves out reclaimable page cache) keeps its figure.
TEST_CASE(UmaBudget, keeps_a_free_below_what_ram_allows) {
    size_t free = 50 * GiB, total = 96 * GiB;
    ggml_cuda_uma_bound_device_memory(&free, &total, 0, 0, 80 * (uint64_t) GiB, kStrixRam);
    CHECK_TRUE(free == 50 * GiB);
    CHECK_TRUE(total == 96 * GiB);
}

// A large carve is the device's own memory, outside MemAvailable: with 60 of
// 64 GiB of it free and only 10 GiB of RAM available, the device can still
// take its 60 GiB.
TEST_CASE(UmaBudget, counts_the_carve_beside_ram) {
    size_t free = 60 * GiB, total = 64 * GiB;
    ggml_cuda_uma_bound_device_memory(&free, &total, 64 * (uint64_t) GiB, 4 * (uint64_t) GiB,
                                      10 * (uint64_t) GiB, 64 * GiB);
    CHECK_TRUE(free == 60 * GiB);
    CHECK_TRUE(total == 64 * GiB);
}

// No RAM to take leaves only the carve; no carve either leaves nothing.
TEST_CASE(UmaBudget, no_ram_leaves_the_carve) {
    size_t free = 100 * GiB, total = 1280 * GiB;
    ggml_cuda_uma_bound_device_memory(&free, &total, GiB, GiB / 2, 0, kStrixRam);
    CHECK_TRUE(free == GiB / 2);
    free = 100 * GiB;
    ggml_cuda_uma_bound_device_memory(&free, &total, 0, 0, 0, kStrixRam);
    CHECK_TRUE(free == 0);
}

// The device interface's RAM is MemAvailable, as llama.cpp reports it, unless
// the driver's page pool holds more than the managed-buffer reserve: right
// after DeepSeek stopped on a 125 GiB box MemAvailable read 59 GiB with 60 GiB
// in the pool, and the figure is what one managed buffer could take, 119 GiB
// less the 7.8 GiB reserve. A buffer sized from all of it is still admitted;
// one sized from MemAvailable plus the whole pool was not. An idle box, or a
// pool smaller than the reserve, reads MemAvailable; a cgroup bounds it.
TEST_CASE(UmaBudget, shared_ram_counts_the_pool_beyond_the_reserve) {
    const size_t reserve = ggml_cuda_uma_reserve_bytes(kStrixRam);
    const uint64_t after_swap = ggml_cuda_uma_shared_ram_bytes(59 * (uint64_t) GiB, 119 * (uint64_t) GiB, kStrixRam);
    CHECK_TRUE(after_swap == 119 * (uint64_t) GiB - reserve);
    CHECK_TRUE(ggml_cuda_uma_alloc_fits((size_t) after_swap, 119 * GiB, kStrixRam));
    CHECK_FALSE(ggml_cuda_uma_alloc_fits(119 * GiB - GiB, 119 * GiB, kStrixRam));
    CHECK_TRUE(ggml_cuda_uma_shared_ram_bytes(122 * (uint64_t) GiB, 122 * (uint64_t) GiB, kStrixRam) ==
               122 * (uint64_t) GiB);
    CHECK_TRUE(ggml_cuda_uma_shared_ram_bytes(100 * (uint64_t) GiB, 103 * (uint64_t) GiB, kStrixRam) ==
               100 * (uint64_t) GiB);
    CHECK_TRUE(ggml_cuda_uma_shared_ram_bytes(120 * (uint64_t) GiB, 4 * (uint64_t) GiB, kStrixRam) ==
               4 * (uint64_t) GiB);
    CHECK_TRUE(ggml_cuda_uma_shared_ram_bytes(0, 0, kStrixRam) == 0);
}

// The device interface reports that RAM, not an integrated GPU's carve
// besides: on a 64 GiB APU with 60 GiB of carve free, 16 GiB of MemAvailable
// and 8 GiB in the pool, 20 GiB, which one managed buffer can take, not 80. A
// buffer sized from 80 would go to managed memory (above 45% of RAM), which
// takes RAM alone.
TEST_CASE(UmaBudget, shared_free_leaves_out_the_carve) {
    const uint64_t shared_ram = ggml_cuda_uma_shared_ram_bytes(16 * (uint64_t) GiB, 24 * (uint64_t) GiB, 64 * GiB);
    CHECK_TRUE(shared_ram == 20 * (uint64_t) GiB);
    CHECK_TRUE(ggml_cuda_uma_shared_free_bytes(60 * GiB, shared_ram, false) == 20 * GiB);
    CHECK_TRUE(ggml_cuda_uma_alloc_fits(20 * GiB, 24 * GiB, 64 * GiB));
    CHECK_TRUE(ggml_cuda_uma_shared_free_bytes(121 * GiB, 0, false) == 0);
}

// Allocations stay in the device heap without managed memory on the device
// (hipMallocManaged unsupported, even with GGML_CUDA_ENABLE_UNIFIED_MEMORY), or
// with automatic managed memory turned off and not forced on.
TEST_CASE(UmaBudget, heap_only_without_managed_memory) {
    CHECK_TRUE(ggml_cuda_uma_heap_only(false, false, false));
    CHECK_TRUE(ggml_cuda_uma_heap_only(false, true, false));
    CHECK_TRUE(ggml_cuda_uma_heap_only(true, false, true));
    CHECK_FALSE(ggml_cuda_uma_heap_only(true, true, true));
    CHECK_FALSE(ggml_cuda_uma_heap_only(true, false, false));
}

// There the heap, which a GTT limit bounds, is the limit: with 30 GiB free in
// it and 60 GiB of RAM to share, 30 GiB, not 60. With managed memory, 60.
TEST_CASE(UmaBudget, shared_free_keeps_the_heap_limit_without_managed_memory) {
    CHECK_TRUE(ggml_cuda_uma_shared_free_bytes(30 * GiB, 60 * (uint64_t) GiB, true) == 30 * GiB);
    CHECK_TRUE(ggml_cuda_uma_shared_free_bytes(30 * GiB, 60 * (uint64_t) GiB, false) == 60 * GiB);
    // A heap limit above what RAM allows is no limit.
    CHECK_TRUE(ggml_cuda_uma_shared_free_bytes(121 * GiB, 60 * (uint64_t) GiB, true) == 60 * GiB);
}

// A discrete GPU with GGML_CUDA_ENABLE_UNIFIED_MEMORY: on a 64 GiB host with 12 GiB of MemAvailable and
// 24 GiB more in the pool, 36 GiB less the 4 GiB reserve, where llama.cpp's
// query gave 12.
TEST_CASE(UmaBudget, shared_free_of_a_discrete_gpu_in_unified_memory_counts_the_pool) {
    const uint64_t shared_ram = ggml_cuda_uma_shared_ram_bytes(12 * (uint64_t) GiB, 36 * (uint64_t) GiB, 64 * GiB);
    CHECK_TRUE(shared_ram == 32 * (uint64_t) GiB);
    CHECK_TRUE(ggml_cuda_uma_shared_free_bytes(32 * GiB, shared_ram, false) == 32 * GiB);
}

// DeepSeek's Strix stack: the device heap (what the runtime allows of the room)
// beside the locked host tier, at most the room, each less the 9.5 GiB carve
// reserve. On an APU with a 64 GiB carve whose runtime reports the carve alone
// (60 GiB free), 60 GiB of RAM (56.25 after the reserve) and 54 GiB lockable:
// 50.5 GiB in the heap and 54 on the host, 104.5 GiB, within the 106.75 GiB
// room; capped at the heap alone it was 50.5. More host tier than RAM is left
// stops at the room. Without a carve the heap is the room already, and a host
// tier adds nothing beyond it.
TEST_CASE(UmaBudget, tiered_capacity_keeps_the_host_tier_beside_a_large_carve) {
    constexpr uint64_t carve_reserve = 9 * (uint64_t) GiB + GiB / 2;
    const uint64_t room = ggml_cuda_uma_room_bytes(64 * (uint64_t) GiB, 4 * (uint64_t) GiB,
                                                   60 * (uint64_t) GiB - ggml_cuda_uma_reserve_bytes(60 * GiB));
    CHECK_TRUE(room == 116 * (uint64_t) GiB + GiB / 4);
    const uint64_t heap = 60 * (uint64_t) GiB - carve_reserve;
    CHECK_TRUE(ggml_cuda_uma_tiered_capacity(heap, 54 * (uint64_t) GiB, room - carve_reserve) ==
               104 * (uint64_t) GiB + GiB / 2);
    CHECK_TRUE(ggml_cuda_uma_tiered_capacity(heap, 80 * (uint64_t) GiB, room - carve_reserve) == room - carve_reserve);
    const uint64_t strix = 113 * (uint64_t) GiB - carve_reserve;
    CHECK_TRUE(ggml_cuda_uma_tiered_capacity(strix, 0, strix) == strix);
    CHECK_TRUE(ggml_cuda_uma_tiered_capacity(strix, 100 * (uint64_t) GiB, strix) == strix);
    CHECK_TRUE(ggml_cuda_uma_tiered_capacity(UINT64_MAX, 1, strix) == strix);
}

#if defined(__linux__)
// The carve comes from amdgpu's mem_info_vram_total and mem_info_vram_used
// under the device's PCI directory; a device without them reports none.
TEST_CASE(UmaBudget, reads_the_carve_from_sysfs) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("luce_test_pci_" + std::to_string((long long) getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "0000:c5:00.0");
    fs::create_directories(root / "0000:01:00.0");
    std::ofstream(root / "0000:c5:00.0" / "mem_info_vram_total") << "536870912\n";
    std::ofstream(root / "0000:c5:00.0" / "mem_info_vram_used") << "154140672\n";

    uint64_t total = 0, used = 0;
    CHECK_TRUE(ggml_cuda_uma_carve_bytes_at(root.string(), "0000:c5:00.0", &total, &used));
    CHECK_TRUE(total == 536870912ULL);
    CHECK_TRUE(used == 154140672ULL);
    CHECK_FALSE(ggml_cuda_uma_carve_bytes_at(root.string(), "0000:01:00.0", &total, &used));
    CHECK_FALSE(ggml_cuda_uma_carve_bytes_at(root.string(), "0000:ff:00.0", &total, &used));
    fs::remove_all(root);
}

// The walk from the process's group to the root takes the smallest room. A
// group whose memory.current cannot be read keeps its limit as the room
// instead of dropping it: there the child's 4 GiB limit binds, not the
// parent's 6 GiB left.
TEST_CASE(UmaBudget, cgroup_limit_without_usage_still_bounds_the_room) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("luce_test_cgroup_" + std::to_string((long long) getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "app" / "job");
    const auto write = [](const fs::path & path, const char * text) { std::ofstream(path) << text; };
    write(root / "memory.max", "max\n");
    write(root / "app" / "memory.max", "8589934592\n");      // 8 GiB
    write(root / "app" / "memory.current", "2147483648\n");  // 2 GiB used
    write(root / "app" / "job" / "memory.max", "4294967296\n");  // 4 GiB, no memory.current

    uint64_t room = 0;
    CHECK_TRUE(ggml_cuda_uma_cgroup_room_bytes_at(root.string(), "/app/job", &room));
    CHECK_TRUE(room == 4 * (uint64_t) GiB);
    fs::remove_all(root);
}
#endif // defined(__linux__)
