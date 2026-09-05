#pragma once

#include <cstdint>
#include <string>

struct ggml_backend;
namespace dflash::common {
struct DeepSeek4Weights;
struct MoeHybridPlacement;
struct MoeHybridConfig;
}

namespace dflash::vision {

enum class ImageMemoryDomain { Unknown, Dedicated, HostShared };

struct ImageAdmissionReserves {
    // GGML currently reports integrated HIP devices as GPU. The caller must
    // classify the actual devices using HIP properties and unified-memory mode.
    ImageMemoryDomain primary_domain = ImageMemoryDomain::Unknown;
    ImageMemoryDomain cold_domain = ImageMemoryDomain::Unknown;
    bool duplicate_hot_on_cold = false;

    // Only allocations not already reflected by the free-memory snapshot:
    // KV, the remaining vision reservation, runtime arenas/pools, and existing
    // allocator/safety margins. Do not charge preloaded core/projector twice.
    uint64_t primary_future_bytes = 0;
    uint64_t cold_future_bytes = 0;
    // Named cold-runtime arena/pool reservation, additional to cold_future_bytes.
    // This is a conservative headroom policy tested by the resource guard, not
    // a claim that every backend temporary has a formal upper bound.
    uint64_t cold_runtime_reservation_bytes = 0;
    int max_chunk_tokens = 1024;

    // Full request/decode/preprocess host peak for the configured request limits
    // and concurrency, plus any future host-only runtime/staging allocations.
    uint64_t host_request_bytes = 0;
    // GGUF metadata parser copies, embedded-table backing, allocator overhead,
    // and the existing host safety reserve. Owner table payloads and expert
    // copy staging are computed separately below.
    uint64_t host_loader_overhead_bytes = 0;
    uint64_t host_runtime_bytes = 0;
};

struct ImageStorageEstimate {
    uint64_t hot_payload_bytes = 0;
    uint64_t cold_payload_bytes = 0;
    uint64_t hot_allocation_bytes = 0;
    uint64_t cold_allocation_bytes = 0;
    uint64_t hot_mix_table_bytes = 0;
    uint64_t cold_mix_table_bytes = 0;
    uint64_t largest_copy_bytes = 0;
    uint64_t host_copy_peak_bytes = 0;
    uint64_t host_mix_payload_peak_bytes = 0;
    uint64_t hot_buffer_count = 0;
    uint64_t cold_buffer_count = 0;
    uint64_t mix_device_allocation_count = 0;
};

struct ImageAdmissionReport {
    ImageStorageEstimate storage;
    uint64_t primary_free_bytes = 0;
    uint64_t cold_free_bytes = 0;
    uint64_t host_available_bytes = 0;
    uint64_t host_gpu_reclaim_credit_bytes = 0;
    std::string host_capacity_policy = "raw";
    uint64_t primary_required_bytes = 0;
    uint64_t cold_required_bytes = 0;
    uint64_t host_required_bytes = 0;
    bool storage_estimated = false;
    bool known_charges_fit = false;
    uint64_t cold_activation_estimate_bytes = 0;
    uint64_t cold_runtime_reservation_bytes = 0;
};

struct ImageMemorySnapshot {
    uint64_t primary_free_bytes = 0;
    uint64_t cold_free_bytes = 0;
    uint64_t host_available_bytes = 0;
    uint64_t host_gpu_reclaim_credit_bytes = 0;
    std::string host_capacity_policy = "raw";
};

// Pure startup-only snapshot transform. The live caller supplies one complete
// /proc/meminfo read and a Linux/HIP kernel release. Credit is qualified only for
// 7.1.3-070103-generic: upstream v7.1.3 mm/show_mem.c si_mem_available excludes
// NR_GPU_RECLAIM, while Documentation/filesystems/proc.rst defines reclaimable
// GPU pools separately from GPUActive. Kernel build provenance must be retained
// during qualification; neither newer versions nor field presence imply support.
// Unknown kernels/missing optional fields retain raw accounting. Malformed
// provided fields on a qualified kernel fail closed. No reserved huge pages are
// supported for credit. This is capacity accounting, not a zero-swap guarantee.
// Shared owners use the same capacity as the combined host gate, never separate
// additive credits. Runtime and preparation deliberately do not call this helper.
bool prepare_deepseek4_image_startup_snapshot(
    const std::string & meminfo, const std::string & kernel_release,
    const ImageAdmissionReserves & reserves, ImageMemorySnapshot & snapshot,
    std::string & error);

// Pure assessment shared by live admission and deterministic CPU tests. All
// figures describe allocations still to come relative to this one snapshot.
bool assess_deepseek4_image_admission(
    const ImageStorageEstimate & storage, uint64_t cold_activation_estimate_bytes,
    const ImageAdmissionReserves & reserves, const ImageMemorySnapshot & snapshot,
    ImageAdmissionReport & result, std::string & error);

// Reads metadata and backend allocation-size functions only; no device buffers,
// expert copies, graph execution, file mapping, or table registration occur.
// Supported scope matches DS4's materialized same-runtime two-GPU owner path
// with zero cache slots. Duplicate mode must match the storage environment.
bool estimate_deepseek4_image_storage(
    const common::DeepSeek4Weights & weights,
    const common::MoeHybridPlacement & placement,
    const common::MoeHybridConfig & config,
    ggml_backend * primary, ggml_backend * cold,
    bool duplicate_hot_on_cold, ImageStorageEstimate & result, std::string & error);

// Invoke after owner initialization and final placement, immediately before
// build_deepseek4_moe_hybrid_storage_from_file_with_mmap. Queries current device
// free memory and Linux MemAvailable, combining new UMA and host charges.
// Returns false on missing reservations or insufficient capacity. The report
// separates the metadata-derived activation estimate from reserved headroom.
// A successful snapshot is admission, not an allocation reservation or a proof
// against concurrent external allocations; guarded runtime verification remains
// necessary. Host cgroup/process limits must be included by caller policy.
bool check_deepseek4_image_admission(
    const common::DeepSeek4Weights & weights,
    const common::MoeHybridPlacement & placement,
    const common::MoeHybridConfig & config,
    ggml_backend * primary, ggml_backend * cold,
    const ImageAdmissionReserves & reserves,
    ImageAdmissionReport & result, std::string & error);

// CPU-only predecode check for the client thread after taking the image lease.
// Does not touch backend state or query a GPU.
bool check_deepseek4_image_host_preparation(uint64_t required_bytes, std::string & error);

// Recheck after synchronizing owners and releasing disposable graph caches.
// Existing experts/tables/core/KV/snapshots are represented only by live free
// memory. No model-loader or expert-copy charge is added a second time.
bool check_deepseek4_image_runtime_admission(
    const common::MoeHybridConfig & config, ggml_backend * primary, ggml_backend * cold,
    const ImageAdmissionReserves & reserves, ImageAdmissionReport & result, std::string & error);

} // namespace dflash::vision
