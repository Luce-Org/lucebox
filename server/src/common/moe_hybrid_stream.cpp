#include "moe_hybrid_stream.h"
#include "gpu_runtime_compat.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_map>
#include <utility>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace dflash::common {
namespace {

bool pinned_allocate(void ** ptr, size_t bytes, void *) {
    return cudaMallocHost(ptr, bytes) == cudaSuccess;
}

void pinned_free(void * ptr, void *) {
    if (ptr) (void) cudaFreeHost(ptr);
}

int env_bounded_int(const char * name, int fallback, int lo, int hi) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < lo || parsed > hi) return fallback;
    return (int) parsed;
}

size_t env_mib(const char * name, size_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0] || value[0] == '-') return fallback;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    constexpr size_t kMiB = 1024 * 1024;
    if (end == value || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max() / kMiB) {
        return fallback;
    }
    return (size_t) parsed * kMiB;
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t device_key(int layer, int expert) {
    return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
}

int backend_device_index(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return -1;
    ggml_backend_dev_t wanted = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_cuda_reg();
    const int count = ggml_backend_cuda_get_device_count();
    for (int device = 0; device < count; ++device) {
        if (ggml_backend_reg_dev_get(reg, (size_t) device) == wanted) return device;
    }
    return -1;
}

// HIP/CUDA streams, events, and allocations belong to the current device.
// The heterogeneous engine alternates R9700 and Strix backends on one host
// thread, so relying on whichever backend ran last is a cross-device bug.
class ScopedGpuDevice {
public:
    explicit ScopedGpuDevice(int target) : target_(target) {
        if (target_ < 0 || cudaGetDevice(&previous_) != cudaSuccess) return;
        valid_ = true;
        if (previous_ != target_) switched_ = cudaSetDevice(target_) == cudaSuccess;
    }

    ~ScopedGpuDevice() {
        if (valid_ && switched_) (void) cudaSetDevice(previous_);
    }

    bool ready() const { return valid_ && (previous_ == target_ || switched_); }

private:
    int target_ = -1;
    int previous_ = -1;
    bool valid_ = false;
    bool switched_ = false;
};

} // namespace

MoeStreamConfig MoeStreamConfig::from_env() {
    MoeStreamConfig config;
    config.nvme = MoeNvmeConfig::from_env(config.nvme);
    config.device_slots = env_bounded_int(
        "DFLASH_MOE_NVME_DEVICE_SLOTS", config.device_slots, 2, 8);
    config.device_cache_bytes = env_mib(
        "DFLASH_MOE_NVME_DEVICE_CACHE_MB", config.device_cache_bytes);
    config.prefill_threshold = env_bounded_int(
        "DFLASH_MOE_NVME_PREFILL_THRESHOLD", config.prefill_threshold, 1, 4096);
    return config;
}

struct MoeHybridStreamEngine::Runtime {
    struct DeviceSlot {
        void * data = nullptr;
        cudaEvent_t ready = nullptr;
        bool pending = false;
        bool valid = false;
        bool cache_managed = false;
        int compute_users = 0;
        MoeExpertKey key{};
        uint64_t frequency = 0;
        uint64_t last_touch = 0;
        MoeNvmeLease host_lease;
        MoeExpertIoLayout layout{};
    };

    ggml_backend_t backend = nullptr;
    int device = -1;
    size_t max_expert_bytes = 0;
    MoeStreamConfig config{};
    std::unique_ptr<MoeNvmeScheduler> io;
    cudaStream_t transfer_stream = nullptr;
    void * device_pool = nullptr;
    size_t device_stride = 0;
    size_t device_pool_bytes = 0;
    std::vector<DeviceSlot> device_slots;
    std::unordered_map<uint64_t, int> device_index;
    uint64_t device_clock = 0;
    uint64_t device_cache_hits = 0;
    uint64_t device_cache_misses = 0;
    uint64_t device_cache_evictions = 0;
    int active_slot = -1;
};

template <typename RuntimeT>
bool allocate_device_cache(RuntimeT & runtime, std::string * err) {
    runtime.device_stride = align_up(runtime.max_expert_bytes, 256);
    if (runtime.device_stride == 0) {
        if (err) *err = "SSD device-cache stride overflow";
        return false;
    }

    size_t desired_slots = (size_t) std::max(2, runtime.config.device_slots);
    if (runtime.config.device_cache_bytes > 0) {
        desired_slots = std::max(
            desired_slots, runtime.config.device_cache_bytes / runtime.device_stride);
    }
    constexpr size_t kMaxDeviceSlots = 65536;
    desired_slots = std::min(desired_slots, kMaxDeviceSlots);
    desired_slots = std::min(
        desired_slots, std::numeric_limits<size_t>::max() / runtime.device_stride);

    // A large contiguous allocation keeps address arithmetic cheap and avoids
    // thousands of allocator objects. If the planner's free-memory snapshot
    // raced another allocation, converge to a smaller usable cache instead of
    // failing model startup.
    size_t attempt_slots = desired_slots;
    cudaError_t gpu_err = cudaSuccess;
    while (attempt_slots >= 2) {
        const size_t bytes = attempt_slots * runtime.device_stride;
        gpu_err = cudaMalloc(&runtime.device_pool, bytes);
        if (gpu_err == cudaSuccess) {
            runtime.device_pool_bytes = bytes;
            break;
        }
        if (attempt_slots == 2) break;
        attempt_slots = std::max<size_t>(2, attempt_slots * 3 / 4);
    }
    if (!runtime.device_pool) {
        if (err) {
            *err = std::string("failed to allocate SSD GPU expert cache: ") +
                   cudaGetErrorString(gpu_err);
        }
        return false;
    }

    try {
        runtime.device_slots.resize(attempt_slots);
    } catch (const std::bad_alloc &) {
        (void) cudaFree(runtime.device_pool);
        runtime.device_pool = nullptr;
        runtime.device_pool_bytes = 0;
        if (err) *err = "failed to allocate SSD GPU cache metadata";
        return false;
    }
    auto * base = static_cast<uint8_t *>(runtime.device_pool);
    for (size_t i = 0; i < runtime.device_slots.size(); ++i) {
        runtime.device_slots[i].data = base + i * runtime.device_stride;
    }
    runtime.config.device_slots = (int) attempt_slots;
    return true;
}

MoeHybridStreamEngine::MoeHybridStreamEngine() = default;
MoeHybridStreamEngine::~MoeHybridStreamEngine() { destroy(); }
MoeHybridStreamEngine::MoeHybridStreamEngine(MoeHybridStreamEngine &&) noexcept = default;
MoeHybridStreamEngine & MoeHybridStreamEngine::operator=(MoeHybridStreamEngine &&) noexcept = default;

bool MoeHybridStreamEngine::init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
                                 std::string * err) {
    destroy();
    if (!gpu_backend || max_expert_bytes == 0) {
        if (err) *err = "invalid arguments to stream engine init";
        return false;
    }

    std::unique_ptr<Runtime> runtime(new (std::nothrow) Runtime);
    if (!runtime) {
        if (err) *err = "failed to allocate stream runtime";
        return false;
    }
    runtime->backend = gpu_backend;
    runtime->device = backend_device_index(gpu_backend);
    ScopedGpuDevice device_scope(runtime->device);
    if (!device_scope.ready()) {
        if (err) *err = "failed to resolve/select SSD stream GPU";
        return false;
    }
    runtime->max_expert_bytes = max_expert_bytes;
    runtime->config = MoeStreamConfig::from_env();
    runtime->io.reset(new (std::nothrow) MoeNvmeScheduler);
    if (!runtime->io) {
        if (err) *err = "failed to allocate SSD scheduler";
        return false;
    }
    if (!runtime->io->init(runtime->config.nvme, max_expert_bytes,
                           pinned_allocate, pinned_free, nullptr, err)) {
        return false;
    }

    cudaError_t gpu_err = cudaStreamCreate(&runtime->transfer_stream);
    if (gpu_err != cudaSuccess) {
        if (err) *err = std::string("failed to create SSD transfer stream: ") +
                        cudaGetErrorString(gpu_err);
        return false;
    }
    if (!allocate_device_cache(*runtime, err)) {
        // Install the partial runtime so destroy() releases every resource.
        runtime_ = std::move(runtime);
        destroy();
        return false;
    }
    runtime_ = std::move(runtime);
    return true;
}

bool MoeHybridStreamEngine::init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
                                 const MoeHybridStorage & storage,
                                 std::string * err) {
    return init(gpu_backend, max_expert_bytes, storage, MoeStreamConfig::from_env(), err);
}

bool MoeHybridStreamEngine::init(ggml_backend_t gpu_backend, size_t max_expert_bytes,
                                 const MoeHybridStorage & storage,
                                 const MoeStreamConfig & config,
                                 std::string * err) {
    destroy();
    if (!gpu_backend || max_expert_bytes == 0) {
        if (err) *err = "invalid arguments to stream engine init";
        return false;
    }

    std::unique_ptr<Runtime> runtime(new (std::nothrow) Runtime);
    if (!runtime) {
        if (err) *err = "failed to allocate stream runtime";
        return false;
    }
    runtime->backend = gpu_backend;
    runtime->device = backend_device_index(gpu_backend);
    ScopedGpuDevice device_scope(runtime->device);
    if (!device_scope.ready()) {
        if (err) *err = "failed to resolve/select SSD stream GPU";
        return false;
    }
    runtime->max_expert_bytes = max_expert_bytes;
    runtime->config = config;
    runtime->config.device_slots = std::max(2, runtime->config.device_slots);
    runtime->io.reset(new (std::nothrow) MoeNvmeScheduler);
    if (!runtime->io ||
        !runtime->io->init(runtime->config.nvme, max_expert_bytes,
                           pinned_allocate, pinned_free, nullptr, err)) {
        return false;
    }

    cudaError_t gpu_err = cudaStreamCreate(&runtime->transfer_stream);
    if (gpu_err != cudaSuccess) {
        if (err) *err = std::string("failed to create SSD transfer stream: ") +
                        cudaGetErrorString(gpu_err);
        return false;
    }
    if (!allocate_device_cache(*runtime, err)) {
        runtime_ = std::move(runtime);
        destroy();
        return false;
    }
    runtime_ = std::move(runtime);
    if (!bind_storage(storage, err)) {
        destroy();
        return false;
    }
    return true;
}

bool MoeHybridStreamEngine::bind_storage(const MoeHybridStorage & storage,
                                         std::string * err) {
    if (!runtime_ || !runtime_->io || !runtime_->io->is_initialized()) {
        if (err) *err = "stream engine is not initialized";
        return false;
    }
    return runtime_->io->bind_source(
        {storage.mmap_data, storage.mmap_size, storage.mmap_fd},
        storage.layer_regions, err);
}

bool MoeHybridStreamEngine::bind_sources(
        const std::vector<MoeNvmeSource> & sources,
        const std::vector<LayerExpertRegions> & layer_regions,
        std::string * err) {
    if (!runtime_ || !runtime_->io || !runtime_->io->is_initialized()) {
        if (err) *err = "stream engine is not initialized";
        return false;
    }
    return runtime_->io->bind_sources(sources, layer_regions, err);
}

bool MoeHybridStreamEngine::is_ready() const {
    return runtime_ && runtime_->backend && runtime_->io &&
           runtime_->io->is_initialized() && runtime_->transfer_stream &&
           !runtime_->device_slots.empty();
}

bool MoeHybridStreamEngine::is_bound() const {
    return is_ready() && runtime_->io->is_bound();
}

void MoeHybridStreamEngine::destroy() {
    if (!runtime_) return;
    const size_t device_cache_slot_count = runtime_->device_slots.size();
    const size_t device_cache_byte_count = runtime_->device_pool_bytes;
    ScopedGpuDevice device_scope(runtime_->device);
    if (runtime_->transfer_stream) {
        (void) cudaStreamSynchronize(runtime_->transfer_stream);
    }
    for (Runtime::DeviceSlot & slot : runtime_->device_slots) {
        slot.host_lease.reset();
        if (slot.ready) (void) cudaEventDestroy(slot.ready);
        slot.ready = nullptr;
        slot.data = nullptr;
        slot.pending = false;
    }
    runtime_->device_slots.clear();
    runtime_->device_index.clear();
    if (runtime_->device_pool) (void) cudaFree(runtime_->device_pool);
    runtime_->device_pool = nullptr;
    if (runtime_->transfer_stream) (void) cudaStreamDestroy(runtime_->transfer_stream);
    runtime_->transfer_stream = nullptr;
    if (runtime_->io) {
        const MoeNvmeStats stats = runtime_->io->stats();
        if (stats.requests != 0 || stats.read_ops != 0 || stats.errors != 0) {
            const double payload_gib = (double) stats.payload_bytes /
                                       (1024.0 * 1024.0 * 1024.0);
            const double physical_gib = (double) stats.physical_bytes /
                                        (1024.0 * 1024.0 * 1024.0);
            const double read_seconds = (double) stats.active_io_ns / 1.0e9;
            const double read_gib_s = read_seconds > 0.0
                ? physical_gib / read_seconds : 0.0;
            const double hit_rate = stats.requests > 0
                ? 100.0 * (double) stats.cache_hits / (double) stats.requests : 0.0;
            const double mean_wait_ms = stats.demand_requests > 0
                ? ((double) stats.wait_ns / 1.0e6) /
                  (double) stats.demand_requests : 0.0;
            std::fprintf(stderr,
                "[moe-nvme] io=%s requests=%llu reads=%llu "
                "payload=%.3f GiB physical=%.3f GiB active-io-rate=%.3f GiB/s "
                "cache-hit=%.1f%% mean-demand-wait=%.3f ms "
                "dedupe=%llu upgrades=%llu dropped-prefetch=%llu errors=%llu "
                "strix-cache=%.1f MiB slots=%zu hits=%llu misses=%llu evictions=%llu\n",
                runtime_->io->effective_backend_name(),
                (unsigned long long) stats.requests,
                (unsigned long long) stats.read_ops,
                payload_gib, physical_gib, read_gib_s, hit_rate, mean_wait_ms,
                (unsigned long long) stats.inflight_deduplications,
                (unsigned long long) stats.demand_upgrades,
                (unsigned long long) stats.prefetch_drops,
                (unsigned long long) stats.errors,
                device_cache_byte_count / 1024.0 / 1024.0,
                device_cache_slot_count,
                (unsigned long long) runtime_->device_cache_hits,
                (unsigned long long) runtime_->device_cache_misses,
                (unsigned long long) runtime_->device_cache_evictions);
        }
        runtime_->io->destroy();
    }
    runtime_.reset();
}

void MoeHybridStreamEngine::request_experts(int layer, const int32_t * expert_ids,
                                             int count, MoeNvmePriority priority) {
    if (!is_bound() || !expert_ids || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        if (expert_ids[i] < 0) continue;
        const uint64_t key = device_key(layer, expert_ids[i]);
        const auto cached = runtime_->device_index.find(key);
        if (cached != runtime_->device_index.end()) {
            const int slot_index = cached->second;
            if (slot_index >= 0 && slot_index < (int) runtime_->device_slots.size()) {
                const Runtime::DeviceSlot & slot =
                    runtime_->device_slots[(size_t) slot_index];
                if (slot.valid && slot.key.layer == layer &&
                    slot.key.expert == expert_ids[i]) {
                    continue;
                }
            }
            runtime_->device_index.erase(cached);
        }
        (void) runtime_->io->request(layer, expert_ids[i], priority, nullptr);
    }
}

void MoeHybridStreamEngine::prefetch_cold_experts(
    const void * mmap_data, size_t mmap_size,
    const LayerExpertRegions & regions,
    const int32_t * cold_expert_ids, int n_cold) {
    if (!mmap_data || mmap_size == 0 || !cold_expert_ids || n_cold <= 0) return;

#if !defined(_WIN32)
    const size_t page_size = (size_t) std::max<long>(1, ::sysconf(_SC_PAGESIZE));
    auto advise = [&](size_t offset, size_t bytes) {
        if (offset > mmap_size || bytes > mmap_size - offset || bytes == 0) return;
        const size_t aligned = (offset / page_size) * page_size;
        const size_t length = bytes + (offset - aligned);
        (void) ::madvise(
            const_cast<uint8_t *>(static_cast<const uint8_t *>(mmap_data)) + aligned,
            length, MADV_WILLNEED);
    };
    for (int i = 0; i < n_cold; ++i) {
        const int expert = cold_expert_ids[i];
        if (expert < 0) continue;
        if (regions.fused_gate_up) {
            advise(regions.gate_up_exps.offset + (size_t) expert * regions.expert_bytes_gate_up,
                   regions.expert_bytes_gate_up);
        } else {
            advise(regions.gate_exps.offset + (size_t) expert * regions.expert_bytes_gate,
                   regions.expert_bytes_gate);
            advise(regions.up_exps.offset + (size_t) expert * regions.expert_bytes_up,
                   regions.expert_bytes_up);
        }
        advise(regions.down_exps.offset + (size_t) expert * regions.expert_bytes_down,
               regions.expert_bytes_down);
    }
#else
    (void) regions;
#endif
}

bool MoeHybridStreamEngine::stage_expert_async(int layer, int expert_id,
                                                int device_slot,
                                                std::string * err) {
    if (!is_bound()) {
        if (err) *err = "stream engine has no bound SSD model source";
        return false;
    }
    ScopedGpuDevice device_scope(runtime_->device);
    if (!device_scope.ready()) {
        if (err) *err = "failed to select SSD stream GPU";
        return false;
    }
    if (device_slot < 0 || device_slot >= (int) runtime_->device_slots.size()) {
        if (err) *err = "SSD device slot is out of range";
        return false;
    }
    Runtime::DeviceSlot & dst = runtime_->device_slots[(size_t) device_slot];
    if (dst.compute_users != 0) {
        if (err) *err = "SSD device slot is still in use by expert compute";
        return false;
    }
    if (dst.pending) {
        const cudaError_t wait_err = cudaEventSynchronize(dst.ready);
        if (wait_err != cudaSuccess) {
            if (err) *err = std::string("failed waiting for prior SSD upload: ") +
                            cudaGetErrorString(wait_err);
            return false;
        }
        dst.pending = false;
        dst.host_lease.reset();
    }
    if (dst.cache_managed && dst.valid) {
        runtime_->device_index.erase(device_key(dst.key.layer, dst.key.expert));
    }
    dst.valid = false;
    dst.cache_managed = false;
    dst.key = {};

    if (!dst.ready) {
        const cudaError_t event_create_err =
            cudaEventCreateWithFlags(&dst.ready, cudaEventDisableTiming);
        if (event_create_err != cudaSuccess) {
            if (err) *err = std::string("failed to create expert upload event: ") +
                            cudaGetErrorString(event_create_err);
            return false;
        }
    }

    MoeNvmeLease lease;
    if (!runtime_->io->acquire(layer, expert_id, lease, err)) return false;
    if (lease.layout().payload_bytes > runtime_->max_expert_bytes) {
        if (err) *err = "streamed expert exceeds GPU device slot";
        return false;
    }
    for (int i = 0; i < lease.layout().span_count; ++i) {
        const MoeExpertIoSpan & span = lease.layout().spans[i];
        cudaError_t gpu_err = cudaMemcpyAsync(
            static_cast<uint8_t *>(dst.data) + span.device_offset,
            lease.data() + span.buffer_offset,
            span.bytes, cudaMemcpyHostToDevice, runtime_->transfer_stream);
        if (gpu_err != cudaSuccess) {
            (void) cudaStreamSynchronize(runtime_->transfer_stream);
            if (err) *err = std::string("asynchronous expert H2D failed: ") +
                            cudaGetErrorString(gpu_err);
            return false;
        }
    }
    const cudaError_t event_err = cudaEventRecord(dst.ready, runtime_->transfer_stream);
    if (event_err != cudaSuccess) {
        (void) cudaStreamSynchronize(runtime_->transfer_stream);
        if (err) *err = std::string("failed to record expert upload event: ") +
                        cudaGetErrorString(event_err);
        return false;
    }
    dst.layout = lease.layout();
    dst.host_lease = std::move(lease);
    dst.pending = true;
    return true;
}

bool MoeHybridStreamEngine::stage_expert_cached_async(
        int layer, int expert_id, int * device_slot, std::string * err) {
    if (!device_slot) {
        if (err) *err = "SSD cache stage requires an output slot";
        return false;
    }
    *device_slot = -1;
    if (!is_bound()) {
        if (err) *err = "stream engine has no bound SSD model source";
        return false;
    }

    const uint64_t key = device_key(layer, expert_id);
    auto cached = runtime_->device_index.find(key);
    if (cached != runtime_->device_index.end()) {
        const int index = cached->second;
        if (index >= 0 && index < (int) runtime_->device_slots.size()) {
            Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) index];
            if (slot.valid && slot.cache_managed &&
                slot.key.layer == layer && slot.key.expert == expert_id) {
                ++runtime_->device_cache_hits;
                ++slot.frequency;
                slot.last_touch = ++runtime_->device_clock;
                *device_slot = index;
                return true;
            }
        }
        runtime_->device_index.erase(cached);
    }

    ++runtime_->device_cache_misses;
    int victim = -1;
    for (size_t i = 0; i < runtime_->device_slots.size(); ++i) {
        const Runtime::DeviceSlot & slot = runtime_->device_slots[i];
        if (!slot.valid && !slot.pending && slot.compute_users == 0) {
            victim = (int) i;
            break;
        }
    }
    if (victim < 0) {
        uint64_t best_score = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < runtime_->device_slots.size(); ++i) {
            const Runtime::DeviceSlot & slot = runtime_->device_slots[i];
            if (!slot.valid || slot.pending || slot.compute_users != 0) continue;
            const uint64_t age = runtime_->device_clock >= slot.last_touch
                ? runtime_->device_clock - slot.last_touch : 0;
            const uint64_t recency = age < 65535 ? 65535 - age : 0;
            const uint64_t score = (slot.frequency << 16) | recency;
            if (score < best_score) {
                best_score = score;
                victim = (int) i;
            }
        }
    }
    if (victim < 0) {
        if (err) *err = "all SSD GPU expert-cache slots are busy";
        return false;
    }

    const bool evicting = runtime_->device_slots[(size_t) victim].valid;
    if (!stage_expert_async(layer, expert_id, victim, err)) return false;
    Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) victim];
    slot.valid = true;
    slot.cache_managed = true;
    slot.key = {(int32_t) layer, (int32_t) expert_id};
    slot.frequency = 1;
    slot.last_touch = ++runtime_->device_clock;
    runtime_->device_index[key] = victim;
    if (evicting) ++runtime_->device_cache_evictions;
    *device_slot = victim;
    return true;
}

bool MoeHybridStreamEngine::activate_device_slot(int device_slot,
                                                  std::string * err) {
    if (!is_ready() || device_slot < 0 ||
        device_slot >= (int) runtime_->device_slots.size()) {
        if (err) *err = "SSD device slot is out of range";
        return false;
    }
    ScopedGpuDevice device_scope(runtime_->device);
    if (!device_scope.ready()) {
        if (err) *err = "failed to select SSD stream GPU";
        return false;
    }
    Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) device_slot];
    if (slot.pending) {
        const cudaError_t gpu_err = cudaEventSynchronize(slot.ready);
        if (gpu_err != cudaSuccess) {
            if (err) *err = std::string("expert H2D synchronization failed: ") +
                            cudaGetErrorString(gpu_err);
            return false;
        }
        slot.pending = false;
        slot.host_lease.reset();
    }
    if (slot.layout.span_count < 2) {
        if (err) *err = "SSD device slot has no complete expert";
        return false;
    }
    if (slot.cache_managed) {
        if (slot.compute_users != 0) {
            if (err) *err = "cached expert slot is already executing";
            return false;
        }
        ++slot.compute_users;
        ++slot.frequency;
        slot.last_touch = ++runtime_->device_clock;
    }
    runtime_->active_slot = device_slot;
    return true;
}

void MoeHybridStreamEngine::release_device_slot(int device_slot) {
    if (!runtime_ || device_slot < 0 ||
        device_slot >= (int) runtime_->device_slots.size()) {
        return;
    }
    Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) device_slot];
    if (slot.cache_managed && slot.compute_users > 0) --slot.compute_users;
    if (runtime_->active_slot == device_slot) runtime_->active_slot = -1;
}

int MoeHybridStreamEngine::device_slot_count() const {
    return runtime_ ? (int) runtime_->device_slots.size() : 0;
}

size_t MoeHybridStreamEngine::device_cache_bytes() const {
    return runtime_ ? runtime_->device_pool_bytes : 0;
}

ggml_backend_t MoeHybridStreamEngine::compute_backend() const {
    return runtime_ ? runtime_->backend : nullptr;
}

bool MoeHybridStreamEngine::stream_expert_sync(int layer, int expert_id,
                                                std::string * err) {
    if (!stage_expert_async(layer, expert_id, 0, err)) return false;
    return activate_device_slot(0, err);
}

bool MoeHybridStreamEngine::stream_expert_sync(
    const void * mmap_data, size_t mmap_size,
    const LayerExpertRegions & regions, int expert_id,
    ggml_backend_t gpu_backend, std::string * err) {
    (void) gpu_backend;
    if (!is_ready()) {
        if (err) *err = "stream engine is not initialized";
        return false;
    }
    if (!is_bound()) {
        std::vector<LayerExpertRegions> one_layer{regions};
        if (!runtime_->io->bind_source({mmap_data, mmap_size, -1}, one_layer, err)) return false;
    }
    return stream_expert_sync(0, expert_id, err);
}

const void * MoeHybridStreamEngine::scratch_gate_data() const {
    if (!runtime_ || runtime_->active_slot < 0) return nullptr;
    const Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) runtime_->active_slot];
    return static_cast<const uint8_t *>(slot.data) + slot.layout.spans[0].device_offset;
}

const void * MoeHybridStreamEngine::scratch_up_data() const {
    if (!runtime_ || runtime_->active_slot < 0) return nullptr;
    const Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) runtime_->active_slot];
    if (slot.layout.fused_gate_up) return nullptr;
    return static_cast<const uint8_t *>(slot.data) + slot.layout.spans[1].device_offset;
}

const void * MoeHybridStreamEngine::scratch_down_data() const {
    if (!runtime_ || runtime_->active_slot < 0) return nullptr;
    const Runtime::DeviceSlot & slot = runtime_->device_slots[(size_t) runtime_->active_slot];
    const int index = slot.layout.fused_gate_up ? 1 : 2;
    return static_cast<const uint8_t *>(slot.data) + slot.layout.spans[index].device_offset;
}

size_t MoeHybridStreamEngine::scratch_gate_bytes() const {
    if (!runtime_ || runtime_->active_slot < 0) return 0;
    return runtime_->device_slots[(size_t) runtime_->active_slot].layout.spans[0].bytes;
}

size_t MoeHybridStreamEngine::scratch_up_bytes() const {
    if (!runtime_ || runtime_->active_slot < 0) return 0;
    const MoeExpertIoLayout & layout =
        runtime_->device_slots[(size_t) runtime_->active_slot].layout;
    return layout.fused_gate_up ? 0 : layout.spans[1].bytes;
}

size_t MoeHybridStreamEngine::scratch_down_bytes() const {
    if (!runtime_ || runtime_->active_slot < 0) return 0;
    const MoeExpertIoLayout & layout =
        runtime_->device_slots[(size_t) runtime_->active_slot].layout;
    return layout.spans[layout.fused_gate_up ? 1 : 2].bytes;
}

size_t MoeHybridStreamEngine::pinned_bytes() const {
    return runtime_ && runtime_->io ? runtime_->io->total_host_bytes() : 0;
}

size_t MoeHybridStreamEngine::scratch_bytes() const {
    return runtime_ ? runtime_->device_pool_bytes : 0;
}

const char * MoeHybridStreamEngine::io_backend_name() const {
    return runtime_ && runtime_->io ? runtime_->io->effective_backend_name() : "uninitialized";
}

MoeNvmeStats MoeHybridStreamEngine::io_stats() const {
    return runtime_ && runtime_->io ? runtime_->io->stats() : MoeNvmeStats{};
}

bool eval_moe_cold_experts_streaming(
    MoeHybridStreamEngine &         engine,
    ggml_backend_t                  gpu_backend,
    const void *                    mmap_data,
    size_t                          mmap_size,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    const LayerExpertRegions &      regions,
    const MoeHybridLayerStorage &   storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    int                             layer) {

    // The streamed tier can intentionally target a different owner from the
    // caller (Strix for Lucebox, while hot/dense work stays on the R9700).
    // Always build and launch the expert graph on the device that owns the
    // stream slots.
    if (engine.compute_backend()) gpu_backend = engine.compute_backend();

    const int n_embd = cfg.expert_embd();
    const int n_ff_exp = cfg.n_ff_exp;
    const int n_used = cfg.n_expert_used;
    const int total_slots = n_used * n_tokens;

    if (cfg.gated_activation == MoeGatedActivation::Situ &&
        (cfg.situ_beta <= 0.0f || cfg.situ_linear_beta <= 0.0f)) {
        if (err) *err = "SiTU activation scales must be positive";
        return false;
    }

    out.assign((size_t) n_embd * (size_t) n_tokens, 0.0f);
    if (!engine.is_ready()) {
        if (err) *err = "stream engine is not ready";
        return false;
    }
    if (!engine.is_bound() && (!mmap_data || mmap_size == 0)) {
        if (err) *err = "mmap is not available";
        return false;
    }

    std::vector<bool> cold_needed((size_t) cfg.n_expert, false);
    for (int i = 0; i < total_slots; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= cfg.n_expert) continue;
        if (selected_weights[i] == 0.0f) continue;
        if (storage.hot_local_by_global[(size_t) gid] < 0) cold_needed[(size_t) gid] = true;
    }

    std::vector<int32_t> unique_cold;
    for (int expert = 0; expert < cfg.n_expert; ++expert) {
        if (cold_needed[(size_t) expert]) unique_cold.push_back((int32_t) expert);
    }
    if (unique_cold.empty()) return true;

    const bool cache_pipeline = engine.is_bound();

    // Admit every actual route before compute. io_uring sees the whole batch;
    // the thread fallback obtains enough outstanding reads to saturate NVMe.
    if (cache_pipeline) {
        engine.request_experts(layer, unique_cold.data(), (int) unique_cold.size(),
                               MoeNvmePriority::Demand);
    }

    int staged_device_slot = 0;
    if (cache_pipeline) {
        if (!engine.stage_expert_cached_async(
                layer, unique_cold[0], &staged_device_slot, err)) return false;
    } else {
        if (!engine.stream_expert_sync(mmap_data, mmap_size, regions,
                                       unique_cold[0], gpu_backend, err)) return false;
    }

    for (size_t cold_index = 0; cold_index < unique_cold.size(); ++cold_index) {
        const int32_t cold_eid = unique_cold[cold_index];
        const int current_device_slot = staged_device_slot;
        if (cache_pipeline &&
            !engine.activate_device_slot(current_device_slot, err)) return false;
        auto release_current = [&]() {
            if (cache_pipeline) engine.release_device_slot(current_device_slot);
        };

        struct TokenHit { int token; float weight; };
        std::vector<TokenHit> hits;
        hits.reserve((size_t) n_tokens);
        for (int token = 0; token < n_tokens; ++token) {
            for (int k = 0; k < n_used; ++k) {
                const int slot = token * n_used + k;
                if (selected_ids[slot] != cold_eid) continue;
                if (selected_weights[slot] != 0.0f) {
                    hits.push_back({token, selected_weights[slot]});
                }
                break;
            }
        }
        if (hits.empty()) {
            release_current();
            continue;
        }

        const int batch = (int) hits.size();
        std::vector<float> batch_input((size_t) n_embd * (size_t) batch);
        for (int i = 0; i < batch; ++i) {
            const float * src = cur_host + (size_t) hits[(size_t) i].token * (size_t) n_embd;
            std::memcpy(batch_input.data() + (size_t) i * (size_t) n_embd,
                        src, sizeof(float) * (size_t) n_embd);
        }

        ggml_init_params ip{};
        ip.mem_size = 32 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            if (err) *err = "ggml_init failed in SSD streaming eval";
            release_current();
            return false;
        }

        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch);
        ggml_set_input(inp);
        ggml_tensor * gate_t = nullptr;
        ggml_tensor * up_t = nullptr;
        ggml_tensor * down_t = nullptr;
        ggml_tensor * gate_up_t = nullptr;
        if (regions.fused_gate_up) {
            gate_up_t = ggml_new_tensor_2d(ctx, desc.ffn_gate_up_exps->type,
                                            n_embd, 2 * n_ff_exp);
            down_t = ggml_new_tensor_2d(ctx, desc.ffn_down_exps->type,
                                        n_ff_exp, n_embd);
            ggml_set_input(gate_up_t);
            ggml_set_input(down_t);
        } else {
            gate_t = ggml_new_tensor_2d(ctx, desc.ffn_gate_exps->type,
                                        n_embd, n_ff_exp);
            up_t = ggml_new_tensor_2d(ctx, desc.ffn_up_exps->type,
                                      n_embd, n_ff_exp);
            down_t = ggml_new_tensor_2d(ctx, desc.ffn_down_exps->type,
                                        n_ff_exp, n_embd);
            ggml_set_input(gate_t);
            ggml_set_input(up_t);
            ggml_set_input(down_t);
        }

        auto apply_gated_activation = [&](ggml_tensor * gate,
                                          ggml_tensor * up) -> ggml_tensor * {
            if (cfg.gated_activation == MoeGatedActivation::Situ) {
                ggml_tensor * nonlinear = ggml_scale(ctx, gate, 1.0f / cfg.situ_beta);
                nonlinear = ggml_tanh(ctx, nonlinear);
                nonlinear = ggml_scale(ctx, nonlinear, cfg.situ_beta);
                nonlinear = ggml_mul(ctx, nonlinear, ggml_sigmoid(ctx, gate));
                ggml_tensor * linear = ggml_scale(
                    ctx, up, 1.0f / cfg.situ_linear_beta);
                linear = ggml_tanh(ctx, linear);
                linear = ggml_scale(ctx, linear, cfg.situ_linear_beta);
                return ggml_mul(ctx, nonlinear, linear);
            }
            if (cfg.swiglu_clamp > 0.0f) {
                return ggml_swiglu_ds4_split(ctx, gate, up, cfg.swiglu_clamp);
            }
            return ggml_swiglu_split(ctx, gate, up);
        };

        ggml_tensor * gated = nullptr;
        if (gate_up_t) {
            ggml_tensor * gate_up_out = ggml_mul_mat(ctx, gate_up_t, inp);
            if (desc.ffn_gate_up_exps_s != 1.0f) {
                gate_up_out = ggml_scale(ctx, gate_up_out, desc.ffn_gate_up_exps_s);
            }
            ggml_tensor * gate_part = ggml_view_2d(
                ctx, gate_up_out, n_ff_exp, batch, gate_up_out->nb[1], 0);
            ggml_tensor * up_part = ggml_view_2d(
                ctx, gate_up_out, n_ff_exp, batch, gate_up_out->nb[1],
                (size_t) n_ff_exp * sizeof(float));
            gate_part = ggml_cont(ctx, gate_part);
            up_part = ggml_cont(ctx, up_part);
            gated = apply_gated_activation(gate_part, up_part);
        } else {
            ggml_tensor * gate = ggml_mul_mat(ctx, gate_t, inp);
            if (desc.ffn_gate_exps_s != 1.0f) gate = ggml_scale(ctx, gate, desc.ffn_gate_exps_s);
            ggml_tensor * up = ggml_mul_mat(ctx, up_t, inp);
            if (desc.ffn_up_exps_s != 1.0f) up = ggml_scale(ctx, up, desc.ffn_up_exps_s);
            gated = apply_gated_activation(gate, up);
        }
        ggml_tensor * expert_out = ggml_mul_mat(ctx, down_t, gated);
        if (desc.ffn_down_exps_s != 1.0f) {
            expert_out = ggml_scale(ctx, expert_out, desc.ffn_down_exps_s);
        }

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
        ggml_set_output(expert_out);
        ggml_build_forward_expand(graph, expert_out);
        ggml_gallocr_t alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(gpu_backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
            if (err) *err = "SSD streaming eval graph allocation failed";
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            release_current();
            return false;
        }

        ggml_backend_tensor_set(inp, batch_input.data(), 0,
                                sizeof(float) * (size_t) n_embd * (size_t) batch);
        if (gate_up_t) {
            gate_up_t->data = const_cast<void *>(engine.scratch_gate_data());
            down_t->data = const_cast<void *>(engine.scratch_down_data());
        } else {
            gate_t->data = const_cast<void *>(engine.scratch_gate_data());
            up_t->data = const_cast<void *>(engine.scratch_up_data());
            down_t->data = const_cast<void *>(engine.scratch_down_data());
        }

        const ggml_status status = ggml_backend_graph_compute_async(gpu_backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            if (err) *err = "SSD streaming expert compute launch failed";
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            release_current();
            return false;
        }

        // Compute N is now running. Wait for the already-issued disk read of
        // N+1 and enqueue its H2D into a different device slot.
        if (cold_index + 1 < unique_cold.size() && cache_pipeline) {
            int next_slot = -1;
            if (!engine.stage_expert_cached_async(
                    layer, unique_cold[cold_index + 1], &next_slot, err)) {
                ggml_backend_synchronize(gpu_backend);
                release_current();
                ggml_gallocr_free(alloc);
                ggml_free(ctx);
                return false;
            }
            staged_device_slot = next_slot;
        }

        ggml_backend_synchronize(gpu_backend);
        std::vector<float> batch_result((size_t) n_embd * (size_t) batch);
        ggml_backend_tensor_get(expert_out, batch_result.data(), 0,
                                sizeof(float) * (size_t) n_embd * (size_t) batch);
        for (int i = 0; i < batch; ++i) {
            const float weight = hits[(size_t) i].weight;
            float * dst = out.data() + (size_t) hits[(size_t) i].token * (size_t) n_embd;
            const float * src = batch_result.data() + (size_t) i * (size_t) n_embd;
            for (int j = 0; j < n_embd; ++j) dst[j] += weight * src[(size_t) j];
        }
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        release_current();

        if (cold_index + 1 < unique_cold.size() && !cache_pipeline) {
            if (!engine.stream_expert_sync(mmap_data, mmap_size, regions,
                                           unique_cold[cold_index + 1],
                                           gpu_backend, err)) return false;
        }
    }
    return true;
}

} // namespace dflash::common
