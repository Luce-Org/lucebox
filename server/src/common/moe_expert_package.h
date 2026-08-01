// Model-neutral expert-major package for SSD-backed MoE inference.
//
// The package changes only physical byte ordering: every routed expert's
// gate/up/down components become one aligned record. Quantized bytes and the
// model's numerical contract remain unchanged. A small self-describing header
// lets any model adapter replace tensor-major shard regions with a single
// package source after validating the original layout fingerprint.

#pragma once

#include "moe_nvme_scheduler.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeExpertPackageOptions {
    // Align each complete expert record for one exact O_DIRECT request.
    size_t record_alignment = 4096;
    // Keep component starts suitable for backend tensor bindings/copies.
    size_t component_alignment = 256;
    // Publish the header only after data is durable. Disable only for disposable
    // benchmark artifacts where the caller accepts crash-corrupted output.
    bool sync_on_finish = true;
    // Optional coarse progress hook, called after each complete layer record
    // range has been written. It never runs from a background thread.
    void (*progress)(size_t completed_layers, size_t total_layers,
                     void * opaque) = nullptr;
    void * progress_opaque = nullptr;
};

struct MoeExpertPackageManifest {
    uint32_t version = 0;
    uint64_t source_layout_hash = 0;
    uint64_t file_bytes = 0;
    size_t record_alignment = 0;
    size_t component_alignment = 0;
    size_t max_record_bytes = 0;
    std::vector<uint32_t> expert_counts;
    std::vector<LayerExpertRegions> layer_regions;
};

// Stable fingerprint of source sizes, tensor regions, component sizes, expert
// counts, and small deterministic samples from every expert stack. It avoids
// hashing hundreds of GiB while rejecting same-shape packages from a different
// checkpoint or quantization artifact.
uint64_t moe_expert_source_layout_hash(
    const std::vector<MoeNvmeSource> & sources,
    const std::vector<LayerExpertRegions> & layers,
    const std::vector<uint32_t> & expert_counts);

// Validate and plan a package without reading or writing weight data.
bool plan_moe_expert_package(
    const std::vector<MoeNvmeSource> & sources,
    const std::vector<LayerExpertRegions> & layers,
    const std::vector<uint32_t> & expert_counts,
    const MoeExpertPackageOptions & options,
    MoeExpertPackageManifest & out,
    std::string * err = nullptr);

// Compile tensor-major source regions into one expert-major output file.
// output_fd is borrowed and must be open for read/write. The file is truncated;
// its magic is published last so interrupted builds cannot look valid.
bool write_moe_expert_package(
    int output_fd,
    const std::vector<MoeNvmeSource> & sources,
    const std::vector<LayerExpertRegions> & layers,
    const std::vector<uint32_t> & expert_counts,
    const MoeExpertPackageOptions & options,
    MoeExpertPackageManifest * out = nullptr,
    std::string * err = nullptr);

// Read and validate a package header. The returned regions all select source
// zero and can be passed directly to MoeHybridStreamEngine::bind_sources with
// the package as its only MoeNvmeSource.
bool read_moe_expert_package(
    const MoeNvmeSource & package,
    MoeExpertPackageManifest & out,
    std::string * err = nullptr);

} // namespace dflash::common
