#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace dflash::common {

inline constexpr size_t COPIED_SOURCE_UPLOAD_CHUNK = 16 * 1024 * 1024;

// Upload a verified file span through one reusable heap buffer. The synchronous
// callback must finish reading each chunk before returning. Its destination
// offset is relative to the tensor, never the file. No original file pointer
// reaches the callback. The caller retains and reuses scratch across tensors.
template <typename Upload>
inline bool upload_copied_file_chunks(
        const void * mapping, size_t mapping_size, size_t source_offset,
        size_t source_size, std::vector<uint8_t> & scratch, Upload && upload) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(mapping);
    if (!mapping || mapping_size > std::numeric_limits<uintptr_t>::max() - base ||
        source_offset > mapping_size || source_size > mapping_size - source_offset)
        return false;
    if (source_size == 0) return true;
    scratch.resize(std::min(source_size, COPIED_SOURCE_UPLOAD_CHUNK));
    const auto * source = static_cast<const uint8_t *>(mapping) + source_offset;
    for (size_t offset = 0; offset < source_size;) {
        const size_t count = std::min(scratch.size(), source_size - offset);
        std::memcpy(scratch.data(), source + offset, count);
        upload(scratch.data(), offset, count);
        offset += count;
    }
    return true;
}

}  // namespace dflash::common
