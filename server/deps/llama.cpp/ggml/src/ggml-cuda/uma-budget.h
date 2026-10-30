#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#include <fstream>
#endif

// RAM check for managed (unified) allocations on integrated GPUs (Linux), shared
// by ggml-cuda.cu, server/src/common (gpu_page_pool, platform_io) and the unit
// tests.
//
// cudaMallocManaged/hipMallocManaged can succeed for more memory than is free:
// the pages are taken as they are mapped (on Strix Halo, without XNACK, during
// the call itself), and a shortfall ends in reclaim and the OOM killer instead
// of an allocation error. So a managed allocation must fit in the RAM that is
// available now (MemAvailable plus the GPU driver's reclaimable page pool,
// swap excluded: GPU pages pushed to swap stall the device; at most the room
// under this process's cgroup memory limits) minus a reserve for the kernel,
// the driver and the rest of the process: 1/16 of RAM, between 1 GiB and
// 8 GiB (7.8 GiB on a 125 GiB box).

static inline size_t ggml_cuda_uma_reserve_bytes(size_t total_ram) {
    constexpr size_t min_reserve = (size_t) 1 << 30;
    constexpr size_t max_reserve = (size_t) 8 << 30;
    const size_t reserve = total_ram / 16;
    return reserve < min_reserve ? min_reserve : reserve > max_reserve ? max_reserve : reserve;
}

static inline bool ggml_cuda_uma_alloc_fits(size_t size, size_t available, size_t total_ram) {
    const size_t reserve = ggml_cuda_uma_reserve_bytes(total_ram);
    return available > reserve && size <= available - reserve;
}

// Memory the GPU driver is holding for reuse (see gpu_page_pool.h). Freed GPU
// buffers in system RAM stay in the kernel's TTM page pool, which serves the
// next GPU allocation first and shrinks under memory pressure, but
// MemAvailable does not count it: after one large run it can read tens of GiB
// low. The pool is only visible to root (debugfs), so this estimates it as the
// RAM /proc/meminfo attributes to nothing, minus the GPU buffers live now
// (amdgpu sysfs) and a 1 GiB margin for pages owned by other drivers.
static inline uint64_t ggml_cuda_uma_page_pool_bytes(const char * meminfo_text, uint64_t live_gpu_host_bytes) {
    // Fields that together describe every page /proc/meminfo can attribute.
    static const char * const accounted_fields[] = {
        "MemFree", "Buffers", "Cached", "SwapCached", "AnonPages", "Slab",
        "KernelStack", "PageTables", "SecPageTables", "Percpu", "Bounce",
    };
    constexpr uint64_t other_drivers_margin = 1ULL << 30;
    if (!meminfo_text) return 0;
    uint64_t total = 0, accounted = 0, huge_pages = 0, huge_page_kb = 0, hugetlb_kb = 0;
    bool has_hugetlb = false;
    std::istringstream lines(meminfo_text);
    std::string line;
    while (std::getline(lines, line)) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = line.substr(0, colon);
        const uint64_t value = std::strtoull(line.c_str() + colon + 1, nullptr, 10);
        if (key == "MemTotal") total = value;
        else if (key == "HugePages_Total") huge_pages = value;
        else if (key == "Hugepagesize") huge_page_kb = value;
        else if (key == "Hugetlb") { hugetlb_kb = value; has_hugetlb = true; }
        else for (const char * field : accounted_fields) if (key == field) accounted += value;
    }
    // Hugetlb covers pools of every page size; older kernels only report the
    // default pool's page count.
    accounted += has_hugetlb ? hugetlb_kb : huge_pages * huge_page_kb;
    if (total <= accounted) return 0;
    const uint64_t unattributed = (total - accounted) * 1024;
    const uint64_t in_use = live_gpu_host_bytes + other_drivers_margin;
    return unattributed > in_use ? unattributed - in_use : 0;
}

// RAM a managed allocation can take now: MemAvailable plus the driver's page
// pool, which is credited only when the live GPU buffers are known
// (live_gpu_host_bytes not null), at most the room under the cgroup limits
// (cgroup_room, null without a limit). A MemAvailable of 0 is a reading like
// any other. False when MemAvailable is missing: the caller cannot tell, which
// is not the same as no RAM.
static inline bool ggml_cuda_uma_available_bytes(const char * meminfo_text, const uint64_t * live_gpu_host_bytes,
                                                 const uint64_t * cgroup_room, uint64_t * available) {
    if (!meminfo_text) return false;
    std::istringstream lines(meminfo_text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.compare(0, 13, "MemAvailable:") != 0) continue;
        char * end = nullptr;
        const uint64_t kb = std::strtoull(line.c_str() + 13, &end, 10);
        if (end == line.c_str() + 13) return false;
        *available = kb * 1024;
        if (live_gpu_host_bytes) *available += ggml_cuda_uma_page_pool_bytes(meminfo_text, *live_gpu_host_bytes);
        if (cgroup_room && *cgroup_room < *available) *available = *cgroup_room;
        return true;
    }
    return false;
}

// What an integrated GPU's allocations can take, those that spill to host
// memory included: its own VRAM (the BIOS carve) left plus `ram_room`, the RAM
// they can take now (from ggml_cuda_uma_available_bytes, less the reserve
// where the caller keeps it), whatever its runtime reports.
static inline uint64_t ggml_cuda_uma_room_bytes(uint64_t carve_total, uint64_t carve_used, uint64_t ram_room) {
    const uint64_t carve_free = carve_total > carve_used ? carve_total - carve_used : 0;
    return carve_free + ram_room;
}

// Free and total memory of an integrated GPU, from its runtime's figures
// (cudaMemGetInfo), bounded by that room, and at most the carve plus RAM in
// all. The runtime's free need not follow what is allocated: on a Strix Halo
// box (amdgpu.no_system_mem_limit=1, ttm.pages_limit at 1.25 TiB) under
// 6.17.0-1033-oem it reported 121.6 of 1280 GiB free however much was
// allocated, and an expert cache sized from it took 117.5 GiB beside 101.8 GiB
// of experts on a 125 GiB box. The same box under 6.17.0-1032-oem reported
// 125.09 GiB in all and a free that followed what was allocated, which the
// bound leaves as it is. A runtime that reports less than RAM allows (a GTT
// limit below RAM, a large carve it reports alone, or DGX Spark's figure,
// which leaves out reclaimable page cache) keeps its figure too.
static inline void ggml_cuda_uma_bound_device_memory(size_t * free, size_t * total, uint64_t carve_total,
                                                     uint64_t carve_used, uint64_t ram_room, size_t total_ram) {
    const uint64_t room = ggml_cuda_uma_room_bytes(carve_total, carve_used, ram_room);
    const uint64_t capacity = carve_total + total_ram;
    if (room < *free) *free = (size_t) room;
    if (capacity < *total) *total = (size_t) capacity;
}

// What a device heap of `heap` bytes beside a locked host tier of `host` bytes
// can hold on an integrated GPU, both taken from `room`
// (ggml_cuda_uma_room_bytes): at most the room. On an APU with a large carve
// the runtime may report the carve alone (the heap) and the host tier is the
// RAM beside it; on one without, heap and host tier are the same RAM.
static inline uint64_t ggml_cuda_uma_tiered_capacity(uint64_t heap, uint64_t host, uint64_t room) {
    const uint64_t both = heap + host < heap ? UINT64_MAX : heap + host;
    return both < room ? both : room;
}

// The RAM a GPU whose memory is host RAM is sized from: MemAvailable, as
// llama.cpp's UMA query reports it, or, when the driver's page pool holds more
// than the managed-buffer reserve, what one managed buffer could take
// (`ram_available`, MemAvailable plus the pool, less that reserve), so that a
// buffer sized from it all is still admitted (ggml_cuda_uma_alloc_fits); at
// most `ram_available`, which the cgroup limits bound. Right after DeepSeek
// stopped on a 125 GiB box MemAvailable read 59 GiB, with 60 GiB more in the
// pool: 111.2 GiB. An idle box reads MemAvailable, as before.
static inline uint64_t ggml_cuda_uma_shared_ram_bytes(uint64_t mem_available, uint64_t ram_available,
                                                      size_t total_ram) {
    const uint64_t reserve = ggml_cuda_uma_reserve_bytes(total_ram);
    const uint64_t with_pool = ram_available > reserve ? ram_available - reserve : 0;
    const uint64_t ram = mem_available > with_pool ? mem_available : with_pool;
    return ram < ram_available ? ram : ram_available;
}

// Whether a device's allocations stay in its heap: without managed memory on
// the device, or with automatic managed memory turned off (LUCE_HIP_NO_AUTO_UMA,
// qwen4exp) and not forced on (GGML_CUDA_ENABLE_UNIFIED_MEMORY); see
// ggml_cuda_device_malloc, which falls back to the heap in both cases.
static inline bool ggml_cuda_uma_heap_only(bool has_managed_memory, bool unified_env, bool auto_uma_off) {
    return !has_managed_memory || (!unified_env && auto_uma_off);
}

// What the device interface (ggml_backend_dev_memory) reports free on a GPU
// whose memory is host RAM (integrated, or GGML_CUDA_ENABLE_UNIFIED_MEMORY),
// when the amdgpu driver's page pool can be estimated: `shared_ram`
// (ggml_cuda_uma_shared_ram_bytes), and when its allocations stay in the
// device heap (ggml_cuda_uma_heap_only), at most the runtime's free. Not an
// integrated GPU's own VRAM besides, as llama.cpp's figure never counted it: a
// buffer sized from both on an APU with a large carve goes to managed memory,
// which takes RAM alone, and the managed-buffer check refuses it.
static inline size_t ggml_cuda_uma_shared_free_bytes(size_t runtime_free, uint64_t shared_ram, bool heap_only) {
    return heap_only && runtime_free < shared_ram ? runtime_free : (size_t) shared_ram;
}

#if defined(__linux__)
static inline bool ggml_cuda_uma_read_meminfo(std::string * text) {
    std::ifstream input("/proc/meminfo");
    if (!input) return false;
    std::stringstream buffer;
    buffer << input.rdbuf();
    *text = buffer.str();
    return !text->empty();
}

// GTT bytes in use on every amdgpu device. False when none reports it, or when
// one has the counter but it cannot be read: the pool cannot then be told
// apart from pages that are in use.
static inline bool ggml_cuda_uma_live_gpu_host_bytes(uint64_t * total) {
    *total = 0;
    bool found = false;
    DIR * dir = opendir("/sys/class/drm");
    if (!dir) return false;
    while (const dirent * entry = readdir(dir)) {
        const std::string name = entry->d_name;
        // card0, card1, ... but not the connector entries such as card0-DP-1.
        if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;
        std::ifstream used("/sys/class/drm/" + name + "/device/mem_info_gtt_used");
        if (!used.is_open()) continue;  // not an amdgpu device
        uint64_t bytes = 0;
        if (!(used >> bytes)) { closedir(dir); return false; }
        *total += bytes;
        found = true;
    }
    closedir(dir);
    return found;
}

// Own VRAM (the BIOS carve) of the amdgpu device at `pci_bus_id`, such as
// "0000:c5:00.0", under the PCI devices directory `devices`. False when the
// device does not report it (not amdgpu).
static inline bool ggml_cuda_uma_carve_bytes_at(const std::string & devices, const std::string & pci_bus_id,
                                                uint64_t * total, uint64_t * used) {
    const std::string dir = devices + "/" + pci_bus_id + "/";
    std::ifstream total_file(dir + "mem_info_vram_total");
    std::ifstream used_file(dir + "mem_info_vram_used");
    unsigned long long total_value = 0, used_value = 0;
    if (!(total_file >> total_value) || !(used_file >> used_value)) return false;
    *total = (uint64_t) total_value;
    *used = (uint64_t) used_value;
    return true;
}

static inline bool ggml_cuda_uma_carve_bytes(const std::string & pci_bus_id, uint64_t * total, uint64_t * used) {
    return ggml_cuda_uma_carve_bytes_at("/sys/bus/pci/devices", pci_bus_id, total, used);
}

// Room left under the cgroup v2 memory limits from `group` (a path such as
// "/user.slice/app.scope") up to the root of the hierarchy mounted at `root`.
// False when no group on the way has a limit.
static inline bool ggml_cuda_uma_cgroup_room_bytes_at(const std::string & root, std::string group,
                                                      uint64_t * room) {
    const auto read_u64 = [](const std::string & path, uint64_t * out) {
        std::ifstream file(path);
        unsigned long long value = 0;
        if (!(file >> value)) return false; // also memory.max "max": no limit
        *out = (uint64_t) value;
        return true;
    };
    bool limited = false;
    // The walk ends at the mount root "/" (checked once): a container with
    // its own cgroup namespace sees its group as "/", its limit at the root.
    while (!group.empty()) {
        uint64_t max = 0, cur = 0;
        const std::string dir = root + (group == "/" ? std::string() : group);
        if (read_u64(dir + "/memory.max", &max)) {
            // A limit whose usage cannot be read still bounds the room.
            if (!read_u64(dir + "/memory.current", &cur)) cur = 0;
            const uint64_t left = max > cur ? max - cur : 0;
            if (!limited || left < *room) *room = left;
            limited = true;
        }
        if (group == "/") break;
        group.resize(group.rfind('/') == 0 ? 1 : group.rfind('/'));
    }
    return limited;
}

// The same for this process's group. A container sees the host's
// MemAvailable, but a managed buffer's pages are charged to its group as they
// are allocated.
static inline bool ggml_cuda_uma_cgroup_room_bytes(uint64_t * room) {
    std::string group;
    std::ifstream cgroup("/proc/self/cgroup");
    std::string line;
    while (std::getline(cgroup, line)) {
        if (line.compare(0, 3, "0::") == 0) {
            group = line.substr(3);
            while (!group.empty() && (group.back() == '\n' || group.back() == '\r')) group.pop_back();
            break;
        }
    }
    return ggml_cuda_uma_cgroup_room_bytes_at("/sys/fs/cgroup", group, room);
}
#endif // defined(__linux__)
