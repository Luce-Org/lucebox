// Model-neutral asynchronous SSD scheduler for routed MoE weights.
//
// This layer deliberately knows nothing about a model architecture or ggml.
// Callers describe an expert as two or three byte ranges in a model file. The
// scheduler owns a bounded set of page-locked host slots, merges duplicate
// requests, gives demand reads strict priority over speculation, and retains
// completed demand reads as a small protected cache.

#pragma once

#include "moe_hybrid_storage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

enum class MoeNvmeBackend {
    Auto,
    ThreadPool,
    IoUring,
    Mmap,
};

enum class MoeNvmeDirectMode {
    Auto,
    Disabled,
    Enabled,
};

enum class MoeNvmePriority : uint8_t {
    Prefetch = 0,
    Demand   = 1,
};

struct MoeNvmeConfig {
    // Host slots are both the in-flight queue bound and the small L2 cache.
    // Eight slots give one modern NVMe enough queue depth without consuming a
    // model-sized amount of pinned memory.
    int host_slots = 8;
    int io_threads = 4;

    MoeNvmeBackend backend = MoeNvmeBackend::Auto;
    MoeNvmeDirectMode direct_io = MoeNvmeDirectMode::Auto;

    // At least this many slots cannot be occupied by speculative requests.
    int demand_reserve = 2;

    // Limit speculative work admitted in one io_uring batch. This bounds the
    // latency of a demand arriving immediately after speculation was issued.
    int max_prefetch_batch = 2;

    // Bound a demand wait so a failed drive cannot hang an inference worker
    // forever. Zero disables the timeout for diagnostic use.
    int demand_timeout_ms = 30000;

    size_t direct_alignment = 4096;

    // Environment overrides use the DFLASH_MOE_NVME_* prefix. Invalid values
    // leave the supplied/default value unchanged.
    static MoeNvmeConfig from_env();
    static MoeNvmeConfig from_env(MoeNvmeConfig base);
};

struct MoeNvmeSource {
    const void * mmap_data = nullptr;
    size_t mmap_size = 0;
    int fd = -1; // borrowed; bind_source(s) duplicates it on POSIX
};

struct MoeExpertKey {
    int32_t layer = -1;
    int32_t expert = -1;

    bool operator==(const MoeExpertKey & other) const {
        return layer == other.layer && expert == other.expert;
    }
};

// One logical tensor slice and its direct-I/O envelope.
struct MoeExpertIoSpan {
    size_t file_offset = 0;       // first payload byte in the model file
    uint32_t source_index = 0;    // model shard containing this span
    size_t bytes = 0;             // payload bytes
    size_t buffer_offset = 0;     // first payload byte in the host slot
    size_t device_offset = 0;     // packed destination offset on the GPU

    size_t io_file_offset = 0;    // aligned direct-I/O start
    size_t io_buffer_offset = 0;  // aligned direct-I/O destination
    size_t io_bytes = 0;          // aligned direct-I/O length
    // Bytes that must be returned to cover the logical payload. This may be
    // smaller than io_bytes for the final unaligned page of a model shard.
    size_t io_required_bytes = 0;
};

enum class MoeExpertComponentKind : uint8_t {
    Gate,
    Up,
    Down,
    FusedGateUp,
};

// Compute-facing view of one tensor inside the packed device slot. Keeping
// components separate from I/O spans lets tensor-major storage use three reads
// while an expert-major representation uses one read with identical compute
// pointers and numerical behavior.
struct MoeExpertComponentLayout {
    MoeExpertComponentKind kind = MoeExpertComponentKind::Gate;
    size_t device_offset = 0;
    size_t bytes = 0;
};

struct MoeExpertIoLayout {
    MoeExpertKey key;
    MoeExpertIoSpan spans[3]{};
    int span_count = 0;
    MoeExpertComponentLayout components[3]{};
    int component_count = 0;
    size_t payload_bytes = 0;
    size_t host_bytes = 0;
    bool fused_gate_up = false;

    const MoeExpertComponentLayout * component(MoeExpertComponentKind kind) const {
        for (int i = 0; i < component_count; ++i) {
            if (components[i].kind == kind) return &components[i];
        }
        return nullptr;
    }
};

// Convert the common LayerExpertRegions descriptor into an exact read plan.
// The plan is usable by mmap, buffered pread, and aligned direct I/O.
bool make_moe_expert_io_layout(
    int layer,
    int expert,
    const LayerExpertRegions & regions,
    size_t source_size,
    size_t direct_alignment,
    MoeExpertIoLayout & out,
    std::string * err = nullptr);

// Split-model variant. Each ExpertFileRegion selects one entry by
// source_index; a tensor itself is never split across files.
bool make_moe_expert_io_layout(
    int layer,
    int expert,
    const LayerExpertRegions & regions,
    const std::vector<size_t> & source_sizes,
    size_t direct_alignment,
    MoeExpertIoLayout & out,
    std::string * err = nullptr);

struct MoeNvmeStats {
    uint64_t requests = 0;
    uint64_t demand_requests = 0;
    uint64_t prefetch_requests = 0;
    uint64_t cache_hits = 0;
    uint64_t inflight_deduplications = 0;
    uint64_t demand_upgrades = 0;
    uint64_t prefetch_drops = 0;
    uint64_t evictions = 0;
    uint64_t read_ops = 0;
    uint64_t payload_bytes = 0;
    uint64_t physical_bytes = 0;
    // Union of intervals in which at least one storage operation was active.
    // Unlike read_ns, this does not double-count concurrent reads.
    uint64_t active_io_ns = 0;
    uint64_t read_ns = 0;
    uint64_t wait_ns = 0;
    uint64_t demand_timeouts = 0;
    uint64_t errors = 0;
};

class MoeNvmeScheduler;

// A lease pins one host slot against eviction while an H2D transfer consumes
// it. It is intentionally move-only.
class MoeNvmeLease {
public:
    MoeNvmeLease() = default;
    ~MoeNvmeLease();

    MoeNvmeLease(const MoeNvmeLease &) = delete;
    MoeNvmeLease & operator=(const MoeNvmeLease &) = delete;
    MoeNvmeLease(MoeNvmeLease && other) noexcept;
    MoeNvmeLease & operator=(MoeNvmeLease && other) noexcept;

    explicit operator bool() const { return scheduler_ != nullptr; }
    const uint8_t * data() const { return data_; }
    const MoeExpertIoLayout & layout() const { return layout_; }
    int slot_index() const { return slot_; }
    void reset();

private:
    friend class MoeNvmeScheduler;
    MoeNvmeScheduler * scheduler_ = nullptr;
    const uint8_t * data_ = nullptr;
    MoeExpertIoLayout layout_{};
    int slot_ = -1;
    uint64_t generation_ = 0;
};

class MoeNvmeScheduler {
public:
    using AllocateFn = bool (*)(void ** ptr, size_t bytes, void * opaque);
    using FreeFn = void (*)(void * ptr, void * opaque);

    MoeNvmeScheduler();
    ~MoeNvmeScheduler();

    MoeNvmeScheduler(const MoeNvmeScheduler &) = delete;
    MoeNvmeScheduler & operator=(const MoeNvmeScheduler &) = delete;

    // Allocate the bounded slot pool. bind_source() starts I/O workers after
    // the model file and per-layer layouts are available.
    bool init(const MoeNvmeConfig & config,
              size_t max_expert_payload_bytes,
              AllocateFn allocate,
              FreeFn free_fn,
              void * allocator_opaque,
              std::string * err = nullptr);

    bool bind_source(const MoeNvmeSource & source,
                     const std::vector<LayerExpertRegions> & layer_regions,
                     std::string * err = nullptr);

    bool bind_sources(const std::vector<MoeNvmeSource> & sources,
                      const std::vector<LayerExpertRegions> & layer_regions,
                      std::string * err = nullptr);

    bool is_initialized() const;
    bool is_bound() const;
    void destroy();

    // Non-blocking admission. A false prefetch result simply means the bounded
    // speculative budget was full; a false demand result includes an error.
    bool request(int layer, int expert, MoeNvmePriority priority,
                 std::string * err = nullptr);

    // Request (or upgrade) and wait for an exact expert. On success the lease
    // protects the host slot until reset/destruction.
    bool acquire(int layer, int expert, MoeNvmeLease & out,
                 std::string * err = nullptr);

    MoeNvmeStats stats() const;
    void reset_stats();

    size_t slot_bytes() const;
    size_t total_host_bytes() const;
    int slot_count() const;
    const char * effective_backend_name() const;
    bool direct_io_active() const;

private:
    friend class MoeNvmeLease;
    void release_lease(int slot, uint64_t generation);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dflash::common
