#include "moe_expert_package.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <new>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace dflash::common {
namespace {

constexpr uint8_t kMagic[8] = {'L', 'B', 'M', 'O', 'E', 'P', 'K', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kFixedHeaderBytes = 64;
constexpr size_t kLayerEntryBytes = 128;
constexpr size_t kCopyChunkBytes = 8 * 1024 * 1024;

bool checked_add(size_t a, size_t b, size_t & out) {
    if (a > std::numeric_limits<size_t>::max() - b) return false;
    out = a + b;
    return true;
}

bool checked_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool align_up(size_t value, size_t alignment, size_t & out) {
    if (!is_power_of_two(alignment)) return false;
    const size_t mask = alignment - 1;
    if (value > std::numeric_limits<size_t>::max() - mask) return false;
    out = (value + mask) & ~mask;
    return true;
}

void put_u32(uint8_t * dst, uint32_t value) {
    for (int i = 0; i < 4; ++i) dst[i] = (uint8_t) (value >> (8 * i));
}

void put_u64(uint8_t * dst, uint64_t value) {
    for (int i = 0; i < 8; ++i) dst[i] = (uint8_t) (value >> (8 * i));
}

uint32_t get_u32(const uint8_t * src) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= (uint32_t) src[i] << (8 * i);
    return value;
}

uint64_t get_u64(const uint8_t * src) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= (uint64_t) src[i] << (8 * i);
    return value;
}

void hash_u64(uint64_t & hash, uint64_t value) {
    constexpr uint64_t kPrime = 1099511628211ULL;
    for (int i = 0; i < 8; ++i) {
        hash ^= (uint8_t) (value >> (8 * i));
        hash *= kPrime;
    }
}

void hash_bytes(uint64_t & hash, const uint8_t * data, size_t bytes) {
    constexpr uint64_t kPrime = 1099511628211ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= data[i];
        hash *= kPrime;
    }
}

bool read_at(const MoeNvmeSource & source, size_t offset,
             void * dst, size_t bytes, std::string * err) {
    if (offset > source.mmap_size || bytes > source.mmap_size - offset) {
        if (err) *err = "expert package source read is out of bounds";
        return false;
    }
    if (bytes == 0) return true;
    if (source.mmap_data) {
        std::memcpy(dst,
                    static_cast<const uint8_t *>(source.mmap_data) + offset,
                    bytes);
        return true;
    }
    if (source.fd < 0) {
        if (err) *err = "expert package source has neither mmap data nor fd";
        return false;
    }

    size_t done = 0;
    while (done < bytes) {
        const size_t chunk = std::min(bytes - done, kCopyChunkBytes);
#if defined(_WIN32)
        if (::_lseeki64(source.fd, (__int64) (offset + done), SEEK_SET) < 0) {
            if (err) *err = "expert package source seek failed";
            return false;
        }
        const int result = ::_read(source.fd,
                                   static_cast<uint8_t *>(dst) + done,
                                   (unsigned int) chunk);
#else
        const ssize_t result = ::pread(source.fd,
                                       static_cast<uint8_t *>(dst) + done,
                                       chunk, (off_t) (offset + done));
#endif
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) {
            if (err) *err = std::string("expert package source read failed: ") +
                            std::strerror(errno);
            return false;
        }
        done += (size_t) result;
    }
    return true;
}

bool write_at(int fd, size_t offset, const void * src,
              size_t bytes, std::string * err) {
    size_t done = 0;
    while (done < bytes) {
        const size_t chunk = std::min(bytes - done, kCopyChunkBytes);
#if defined(_WIN32)
        if (::_lseeki64(fd, (__int64) (offset + done), SEEK_SET) < 0) {
            if (err) *err = "expert package output seek failed";
            return false;
        }
        const int result = ::_write(fd,
                                    static_cast<const uint8_t *>(src) + done,
                                    (unsigned int) chunk);
#else
        const ssize_t result = ::pwrite(fd,
                                        static_cast<const uint8_t *>(src) + done,
                                        chunk, (off_t) (offset + done));
#endif
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) {
            if (err) *err = std::string("expert package output write failed: ") +
                            std::strerror(errno);
            return false;
        }
        done += (size_t) result;
    }
    return true;
}

bool resize_file(int fd, size_t bytes, std::string * err) {
#if defined(_WIN32)
    if (::_chsize_s(fd, bytes) == 0) return true;
#else
    if (::ftruncate(fd, (off_t) bytes) == 0) return true;
#endif
    if (err) *err = std::string("failed to size expert package: ") +
                    std::strerror(errno);
    return false;
}

bool sync_file(int fd, std::string * err) {
#if defined(_WIN32)
    if (::_commit(fd) == 0) return true;
#else
    if (::fsync(fd) == 0) return true;
#endif
    if (err) *err = std::string("failed to sync expert package: ") +
                    std::strerror(errno);
    return false;
}

bool validate_component(const ExpertFileRegion & region,
                        size_t expert_bytes, uint32_t experts,
                        const std::vector<MoeNvmeSource> & sources,
                        const char * label, std::string * err) {
    if (expert_bytes == 0 || experts == 0 ||
        region.source_index >= sources.size()) {
        if (err) *err = std::string("invalid expert package ") + label +
                        " component";
        return false;
    }
    size_t required = 0;
    if (!checked_mul(expert_bytes, (size_t) experts, required) ||
        region.size < required ||
        region.offset > sources[region.source_index].mmap_size ||
        required > sources[region.source_index].mmap_size - region.offset) {
        if (err) *err = std::string("expert package ") + label +
                        " component exceeds its source";
        return false;
    }
    return true;
}

struct ComponentRef {
    const ExpertFileRegion * source = nullptr;
    size_t bytes = 0;
    size_t destination_offset = 0;
};

int layer_components(const LayerExpertRegions & layer,
                     ComponentRef (&components)[3]) {
    if (layer.fused_gate_up) {
        components[0] = {&layer.gate_up_exps,
                         layer.expert_bytes_gate_up,
                         layer.expert_major.gate_up_offset};
        components[1] = {&layer.down_exps,
                         layer.expert_bytes_down,
                         layer.expert_major.down_offset};
        return 2;
    }
    components[0] = {&layer.gate_exps,
                     layer.expert_bytes_gate,
                     layer.expert_major.gate_offset};
    components[1] = {&layer.up_exps,
                     layer.expert_bytes_up,
                     layer.expert_major.up_offset};
    components[2] = {&layer.down_exps,
                     layer.expert_bytes_down,
                     layer.expert_major.down_offset};
    return 3;
}

bool encode_header(const MoeExpertPackageManifest & manifest,
                   std::vector<uint8_t> & header, std::string * err) {
    size_t table_bytes = 0;
    if (!checked_mul(manifest.layer_regions.size(), kLayerEntryBytes,
                     table_bytes)) {
        if (err) *err = "expert package header size overflow";
        return false;
    }
    size_t raw_header_bytes = 0;
    if (!checked_add(kFixedHeaderBytes, table_bytes, raw_header_bytes)) {
        if (err) *err = "expert package header size overflow";
        return false;
    }
    size_t header_bytes = 0;
    if (!align_up(raw_header_bytes, manifest.record_alignment, header_bytes)) {
        if (err) *err = "expert package header alignment overflow";
        return false;
    }
    if (header_bytes > std::numeric_limits<uint32_t>::max()) {
        if (err) *err = "expert package header exceeds its on-disk field";
        return false;
    }
    try {
        header.assign(header_bytes, 0);
    } catch (const std::bad_alloc &) {
        if (err) *err = "failed to allocate expert package header";
        return false;
    }
    std::memcpy(header.data(), kMagic, sizeof(kMagic));
    put_u32(header.data() + 8, manifest.version);
    put_u32(header.data() + 12, (uint32_t) header_bytes);
    put_u32(header.data() + 16, (uint32_t) manifest.layer_regions.size());
    put_u32(header.data() + 20, (uint32_t) kLayerEntryBytes);
    put_u64(header.data() + 24, manifest.file_bytes);
    put_u64(header.data() + 32, manifest.source_layout_hash);
    put_u64(header.data() + 40, manifest.record_alignment);
    put_u64(header.data() + 48, manifest.component_alignment);

    for (size_t i = 0; i < manifest.layer_regions.size(); ++i) {
        const LayerExpertRegions & layer = manifest.layer_regions[i];
        uint8_t * entry = header.data() + kFixedHeaderBytes +
                          i * kLayerEntryBytes;
        put_u32(entry + 0, manifest.expert_counts[i]);
        put_u32(entry + 4, layer.fused_gate_up ? 1U : 0U);
        put_u64(entry + 8, layer.expert_major.experts.offset);
        put_u64(entry + 16, layer.expert_major.experts.size);
        put_u64(entry + 24, layer.expert_major.expert_stride);
        put_u64(entry + 32, layer.expert_major.gate_offset);
        put_u64(entry + 40, layer.expert_bytes_gate);
        put_u64(entry + 48, layer.expert_major.up_offset);
        put_u64(entry + 56, layer.expert_bytes_up);
        put_u64(entry + 64, layer.expert_major.down_offset);
        put_u64(entry + 72, layer.expert_bytes_down);
        put_u64(entry + 80, layer.expert_major.gate_up_offset);
        put_u64(entry + 88, layer.expert_bytes_gate_up);
    }
    return true;
}

} // namespace

uint64_t moe_expert_source_layout_hash(
        const std::vector<MoeNvmeSource> & sources,
        const std::vector<LayerExpertRegions> & layers,
        const std::vector<uint32_t> & expert_counts) {
    if (layers.size() != expert_counts.size()) return 0;
    uint64_t hash = 1469598103934665603ULL;
    hash_u64(hash, sources.size());
    for (const MoeNvmeSource & source : sources) hash_u64(hash, source.mmap_size);
    hash_u64(hash, layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerExpertRegions & layer = layers[i];
        hash_u64(hash, expert_counts[i]);
        hash_u64(hash, layer.fused_gate_up ? 1 : 0);
        auto add_region = [&](const ExpertFileRegion & region,
                              size_t expert_bytes) -> bool {
            hash_u64(hash, region.source_index);
            hash_u64(hash, region.offset);
            hash_u64(hash, region.size);
            hash_u64(hash, expert_bytes);
            if (region.source_index >= sources.size()) return false;
            size_t logical_bytes = 0;
            if (!checked_mul(expert_bytes, (size_t) expert_counts[i],
                             logical_bytes) ||
                logical_bytes == 0 || region.size < logical_bytes) {
                return false;
            }
            // Layout alone cannot distinguish two checkpoints with identical
            // tensor shapes. Sample the start, middle and end of every expert
            // stack so attaching another model's package fails closed without
            // hashing hundreds of GiB at startup.
            constexpr size_t kSampleBytes = 64;
            uint8_t sample[kSampleBytes]{};
            const size_t sample_bytes = std::min(kSampleBytes, logical_bytes);
            const size_t positions[3] = {
                0,
                (logical_bytes - sample_bytes) / 2,
                logical_bytes - sample_bytes,
            };
            for (size_t position : positions) {
                size_t offset = 0;
                if (!checked_add(region.offset, position, offset) ||
                    !read_at(sources[region.source_index], offset,
                             sample, sample_bytes, nullptr)) {
                    return false;
                }
                hash_u64(hash, position);
                hash_bytes(hash, sample, sample_bytes);
            }
            return true;
        };
        if (layer.fused_gate_up) {
            if (!add_region(layer.gate_up_exps,
                            layer.expert_bytes_gate_up)) return 0;
        } else {
            if (!add_region(layer.gate_exps, layer.expert_bytes_gate) ||
                !add_region(layer.up_exps, layer.expert_bytes_up)) return 0;
        }
        if (!add_region(layer.down_exps, layer.expert_bytes_down)) return 0;
    }
    return hash;
}

bool plan_moe_expert_package(
        const std::vector<MoeNvmeSource> & sources,
        const std::vector<LayerExpertRegions> & layers,
        const std::vector<uint32_t> & expert_counts,
        const MoeExpertPackageOptions & options,
        MoeExpertPackageManifest & out,
        std::string * err) {
    out = {};
    if (sources.empty() || layers.empty() ||
        layers.size() != expert_counts.size()) {
        if (err) *err = "expert package source/layer dimensions are invalid";
        return false;
    }
    if (!is_power_of_two(options.record_alignment) ||
        !is_power_of_two(options.component_alignment) ||
        options.record_alignment < 512 ||
        options.component_alignment > options.record_alignment) {
        if (err) *err = "expert package alignments must be powers of two";
        return false;
    }
    if (layers.size() > std::numeric_limits<uint32_t>::max()) {
        if (err) *err = "expert package has too many layers";
        return false;
    }
    for (const MoeNvmeSource & source : sources) {
        if (source.mmap_size == 0 || (!source.mmap_data && source.fd < 0)) {
            if (err) *err = "expert package source is unavailable";
            return false;
        }
    }

    size_t table_bytes = 0;
    size_t raw_header_bytes = 0;
    size_t cursor = 0;
    if (!checked_mul(layers.size(), kLayerEntryBytes, table_bytes) ||
        !checked_add(kFixedHeaderBytes, table_bytes, raw_header_bytes) ||
        !align_up(raw_header_bytes, options.record_alignment, cursor)) {
        if (err) *err = "expert package header size overflow";
        return false;
    }

    MoeExpertPackageManifest manifest;
    manifest.version = kVersion;
    manifest.record_alignment = options.record_alignment;
    manifest.component_alignment = options.component_alignment;
    manifest.expert_counts = expert_counts;
    manifest.layer_regions.resize(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerExpertRegions & source = layers[i];
        const uint32_t experts = expert_counts[i];
        if (experts == 0) {
            if (err) *err = "expert package layer has zero experts";
            return false;
        }
        if (source.fused_gate_up) {
            if (!validate_component(source.gate_up_exps,
                                    source.expert_bytes_gate_up, experts,
                                    sources, "gate_up", err) ||
                !validate_component(source.down_exps,
                                    source.expert_bytes_down, experts,
                                    sources, "down", err)) {
                return false;
            }
        } else if (!validate_component(source.gate_exps,
                                       source.expert_bytes_gate, experts,
                                       sources, "gate", err) ||
                   !validate_component(source.up_exps,
                                       source.expert_bytes_up, experts,
                                       sources, "up", err) ||
                   !validate_component(source.down_exps,
                                       source.expert_bytes_down, experts,
                                       sources, "down", err)) {
            return false;
        }

        LayerExpertRegions packed;
        packed.fused_gate_up = source.fused_gate_up;
        packed.expert_bytes_gate = source.expert_bytes_gate;
        packed.expert_bytes_up = source.expert_bytes_up;
        packed.expert_bytes_down = source.expert_bytes_down;
        packed.expert_bytes_gate_up = source.expert_bytes_gate_up;
        packed.expert_major.enabled = true;

        size_t record_cursor = 0;
        auto place = [&](size_t bytes, size_t & offset) -> bool {
            if (!align_up(record_cursor, options.component_alignment, offset) ||
                !checked_add(offset, bytes, record_cursor)) {
                return false;
            }
            return true;
        };
        if (source.fused_gate_up) {
            if (!place(source.expert_bytes_gate_up,
                       packed.expert_major.gate_up_offset) ||
                !place(source.expert_bytes_down,
                       packed.expert_major.down_offset)) {
                if (err) *err = "expert package fused record size overflow";
                return false;
            }
        } else if (!place(source.expert_bytes_gate,
                          packed.expert_major.gate_offset) ||
                   !place(source.expert_bytes_up,
                          packed.expert_major.up_offset) ||
                   !place(source.expert_bytes_down,
                          packed.expert_major.down_offset)) {
            if (err) *err = "expert package record size overflow";
            return false;
        }
        if (!align_up(record_cursor, options.record_alignment,
                      packed.expert_major.expert_stride)) {
            if (err) *err = "expert package record alignment overflow";
            return false;
        }

        size_t layer_bytes = 0;
        if (!checked_mul(packed.expert_major.expert_stride,
                         (size_t) experts, layer_bytes)) {
            if (err) *err = "expert package layer size overflow";
            return false;
        }
        packed.expert_major.experts = {cursor, layer_bytes, 0};
        if (!checked_add(cursor, layer_bytes, cursor)) {
            if (err) *err = "expert package file size overflow";
            return false;
        }
        manifest.max_record_bytes = std::max(
            manifest.max_record_bytes, packed.expert_major.expert_stride);
        manifest.layer_regions[i] = packed;
    }
    manifest.file_bytes = cursor;
    manifest.source_layout_hash = moe_expert_source_layout_hash(
        sources, layers, expert_counts);
    if (manifest.source_layout_hash == 0) {
        if (err) *err = "failed to fingerprint expert package sources";
        return false;
    }
    out = std::move(manifest);
    return true;
}

bool write_moe_expert_package(
        int output_fd,
        const std::vector<MoeNvmeSource> & sources,
        const std::vector<LayerExpertRegions> & layers,
        const std::vector<uint32_t> & expert_counts,
        const MoeExpertPackageOptions & options,
        MoeExpertPackageManifest * out,
        std::string * err) {
    if (output_fd < 0) {
        if (err) *err = "expert package output fd is invalid";
        return false;
    }
    MoeExpertPackageManifest manifest;
    if (!plan_moe_expert_package(sources, layers, expert_counts,
                                 options, manifest, err)) {
        return false;
    }

    // Invalidate any previous package before resizing or copying data.
    const uint8_t invalid_magic[8]{};
    if (!write_at(output_fd, 0, invalid_magic, sizeof(invalid_magic), err) ||
        !resize_file(output_fd, (size_t) manifest.file_bytes, err)) {
        return false;
    }

    std::vector<uint8_t> record;
    try {
        record.resize(manifest.max_record_bytes);
    } catch (const std::bad_alloc &) {
        if (err) *err = "failed to allocate expert package record buffer";
        return false;
    }

    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const LayerExpertRegions & source_layer = layers[layer_index];
        const LayerExpertRegions & packed_layer = manifest.layer_regions[layer_index];
        ComponentRef source_components[3]{};
        ComponentRef packed_components[3]{};
        const int source_count = layer_components(source_layer, source_components);
        const int packed_count = layer_components(packed_layer, packed_components);
        if (source_count != packed_count) {
            if (err) *err = "expert package internal component mismatch";
            return false;
        }
        for (uint32_t expert = 0;
             expert < expert_counts[layer_index]; ++expert) {
            const size_t stride = packed_layer.expert_major.expert_stride;
            std::memset(record.data(), 0, stride);
            for (int component = 0; component < source_count; ++component) {
                const ComponentRef & src = source_components[component];
                const ComponentRef & dst = packed_components[component];
                size_t expert_delta = 0;
                size_t source_offset = 0;
                if (!checked_mul((size_t) expert, src.bytes, expert_delta) ||
                    !checked_add(src.source->offset, expert_delta,
                                 source_offset) ||
                    dst.destination_offset > stride ||
                    dst.bytes > stride - dst.destination_offset) {
                    if (err) *err = "expert package component offset overflow";
                    return false;
                }
                if (!read_at(sources[src.source->source_index], source_offset,
                             record.data() + dst.destination_offset,
                             src.bytes, err)) {
                    return false;
                }
            }
            size_t record_delta = 0;
            size_t output_offset = 0;
            if (!checked_mul((size_t) expert, stride, record_delta) ||
                !checked_add(packed_layer.expert_major.experts.offset,
                             record_delta, output_offset) ||
                !write_at(output_fd, output_offset, record.data(),
                          stride, err)) {
                return false;
            }
        }
        if (options.progress) {
            options.progress(layer_index + 1, layers.size(),
                             options.progress_opaque);
        }
    }

    if (options.sync_on_finish && !sync_file(output_fd, err)) return false;
    std::vector<uint8_t> header;
    if (!encode_header(manifest, header, err) ||
        !write_at(output_fd, 0, header.data(), header.size(), err) ||
        (options.sync_on_finish && !sync_file(output_fd, err))) {
        return false;
    }
    if (out) *out = std::move(manifest);
    return true;
}

bool read_moe_expert_package(
        const MoeNvmeSource & package,
        MoeExpertPackageManifest & out,
        std::string * err) {
    out = {};
    if (package.mmap_size < kFixedHeaderBytes ||
        (!package.mmap_data && package.fd < 0)) {
        if (err) *err = "expert package source is unavailable or truncated";
        return false;
    }
    uint8_t fixed[kFixedHeaderBytes]{};
    if (!read_at(package, 0, fixed, sizeof(fixed), err)) return false;
    if (std::memcmp(fixed, kMagic, sizeof(kMagic)) != 0) {
        if (err) *err = "expert package magic is missing (incomplete or wrong file)";
        return false;
    }
    const uint32_t version = get_u32(fixed + 8);
    const uint32_t header_bytes = get_u32(fixed + 12);
    const uint32_t layer_count = get_u32(fixed + 16);
    const uint32_t entry_bytes = get_u32(fixed + 20);
    const uint64_t file_bytes = get_u64(fixed + 24);
    const uint64_t source_hash = get_u64(fixed + 32);
    const uint64_t record_alignment = get_u64(fixed + 40);
    const uint64_t component_alignment = get_u64(fixed + 48);
    if (version != kVersion || layer_count == 0 ||
        entry_bytes != kLayerEntryBytes ||
        source_hash == 0 ||
        file_bytes > std::numeric_limits<size_t>::max() ||
        record_alignment > std::numeric_limits<size_t>::max() ||
        component_alignment > std::numeric_limits<size_t>::max() ||
        !is_power_of_two((size_t) record_alignment) ||
        !is_power_of_two((size_t) component_alignment) ||
        component_alignment > record_alignment ||
        header_bytes < kFixedHeaderBytes ||
        header_bytes > package.mmap_size ||
        file_bytes != package.mmap_size) {
        if (err) *err = "expert package header is invalid or incompatible";
        return false;
    }
    size_t table_bytes = 0;
    size_t required_header = 0;
    if (!checked_mul((size_t) layer_count, kLayerEntryBytes, table_bytes) ||
        !checked_add(kFixedHeaderBytes, table_bytes, required_header) ||
        required_header > header_bytes) {
        if (err) *err = "expert package layer table is truncated";
        return false;
    }
    std::vector<uint8_t> header;
    try {
        header.resize(header_bytes);
    } catch (const std::bad_alloc &) {
        if (err) *err = "failed to allocate expert package header";
        return false;
    }
    if (!read_at(package, 0, header.data(), header.size(), err)) return false;

    MoeExpertPackageManifest manifest;
    manifest.version = version;
    manifest.source_layout_hash = source_hash;
    manifest.file_bytes = file_bytes;
    manifest.record_alignment = (size_t) record_alignment;
    manifest.component_alignment = (size_t) component_alignment;
    manifest.expert_counts.resize(layer_count);
    manifest.layer_regions.resize(layer_count);

    size_t previous_end = header_bytes;
    for (uint32_t i = 0; i < layer_count; ++i) {
        const uint8_t * entry = header.data() + kFixedHeaderBytes +
                                (size_t) i * kLayerEntryBytes;
        const uint32_t experts = get_u32(entry + 0);
        const bool fused = (get_u32(entry + 4) & 1U) != 0;
        for (size_t field = 8; field <= 88; field += 8) {
            if (get_u64(entry + field) >
                std::numeric_limits<size_t>::max()) {
                if (err) *err = "expert package layer field exceeds host size";
                return false;
            }
        }
        const size_t data_offset = (size_t) get_u64(entry + 8);
        const size_t data_bytes = (size_t) get_u64(entry + 16);
        const size_t stride = (size_t) get_u64(entry + 24);
        LayerExpertRegions layer;
        layer.fused_gate_up = fused;
        layer.expert_major.enabled = true;
        layer.expert_major.experts = {data_offset, data_bytes, 0};
        layer.expert_major.expert_stride = stride;
        layer.expert_major.gate_offset = (size_t) get_u64(entry + 32);
        layer.expert_bytes_gate = (size_t) get_u64(entry + 40);
        layer.expert_major.up_offset = (size_t) get_u64(entry + 48);
        layer.expert_bytes_up = (size_t) get_u64(entry + 56);
        layer.expert_major.down_offset = (size_t) get_u64(entry + 64);
        layer.expert_bytes_down = (size_t) get_u64(entry + 72);
        layer.expert_major.gate_up_offset = (size_t) get_u64(entry + 80);
        layer.expert_bytes_gate_up = (size_t) get_u64(entry + 88);

        size_t expected_bytes = 0;
        size_t data_end = 0;
        if (experts == 0 || stride == 0 ||
            stride % record_alignment != 0 ||
            data_offset % record_alignment != 0 ||
            !checked_mul(stride, (size_t) experts, expected_bytes) ||
            expected_bytes != data_bytes ||
            data_offset < previous_end ||
            !checked_add(data_offset, data_bytes, data_end) ||
            data_end > package.mmap_size ||
            layer.expert_bytes_down == 0 ||
            (fused ? layer.expert_bytes_gate_up == 0
                   : (layer.expert_bytes_gate == 0 ||
                      layer.expert_bytes_up == 0))) {
            if (err) *err = "expert package layer entry is invalid";
            return false;
        }
        auto contained = [&](size_t offset, size_t bytes) {
            return offset <= stride && bytes <= stride - offset;
        };
        if (!contained(layer.expert_major.down_offset,
                       layer.expert_bytes_down) ||
            (fused
                 ? !contained(layer.expert_major.gate_up_offset,
                              layer.expert_bytes_gate_up)
                 : (!contained(layer.expert_major.gate_offset,
                               layer.expert_bytes_gate) ||
                    !contained(layer.expert_major.up_offset,
                               layer.expert_bytes_up)))) {
            if (err) *err = "expert package component exceeds its record";
            return false;
        }
        manifest.expert_counts[i] = experts;
        manifest.layer_regions[i] = layer;
        manifest.max_record_bytes = std::max(
            manifest.max_record_bytes, stride);
        previous_end = data_end;
    }
    if (previous_end != package.mmap_size) {
        if (err) *err = "expert package contains an unaccounted trailing range";
        return false;
    }
    out = std::move(manifest);
    return true;
}

} // namespace dflash::common
