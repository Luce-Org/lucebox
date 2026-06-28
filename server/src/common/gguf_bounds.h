// common/gguf_bounds.h — overflow-safe tensor bounds check for GGUF loaders.
//
// Every loader that copies tensor bytes out of an mmap'd GGUF must verify that a
// tensor's [data_off + tensor_off, + tensor_sz) region lies within the file
// before the copy. Otherwise a truncated or corrupt file makes the copy read
// past the end of the mapping and fault inside the device copy (bug #438). A
// naive `data_off + tensor_off + tensor_sz > file_size` test can itself wrap on
// malformed offsets/sizes and silently pass, so gguf_tensor_in_file() avoids any
// addition that can overflow size_t.
//
// When a *valid* file is wrongly rejected (bug #318), gguf_bounds_error() emits
// every operand (data offset, tensor offset, size, file size, type) so the
// mismatch is diagnosable instead of an opaque "overflows file".
//
// Include convention: #include "common/gguf_bounds.h"
// Never: ../common/gguf_bounds.h or absolute paths.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace dflash::common {

// True iff the tensor's data region lies fully within the mapped file.
//   data_off:   byte offset of the GGUF data section (gguf_get_data_offset)
//   tensor_off: tensor offset relative to the data section (gguf_get_tensor_offset)
//   tensor_sz:  tensor byte size (gguf_get_tensor_size)
//   file_size:  mapped file size in bytes
// Overflow-safe: never forms data_off + tensor_off + tensor_sz directly.
inline bool gguf_tensor_in_file(size_t data_off, size_t tensor_off,
                                size_t tensor_sz, size_t file_size) {
    if (data_off > file_size) return false;
    const size_t data_avail = file_size - data_off;   // bytes from data section start to EOF
    if (tensor_off > data_avail) return false;
    return tensor_sz <= data_avail - tensor_off;
}

// Build a diagnostic error for a tensor that failed gguf_tensor_in_file().
// Reports every operand so a false rejection on a genuinely valid file (#318)
// can be traced to a data-section offset/alignment or size mismatch rather than
// guessed at. `what` is the loader-specific prefix (e.g. "draft GGUF").
inline std::string gguf_bounds_error(const std::string & what,
                                     const char * tensor_name,
                                     const char * type_name,
                                     size_t data_off, size_t tensor_off,
                                     size_t tensor_sz, size_t file_size) {
    const std::string name = tensor_name ? tensor_name : "(null)";
    const std::string type = type_name   ? type_name   : "(unknown)";

    // end = data_off + tensor_off + tensor_sz, reported only when it does not
    // overflow size_t (it never does for a genuinely valid file; an overflow
    // here is itself the diagnosis).
    std::string end_str;
    if (data_off <= SIZE_MAX - tensor_off &&
        data_off + tensor_off <= SIZE_MAX - tensor_sz) {
        end_str = std::to_string(data_off + tensor_off + tensor_sz);
    } else {
        end_str = "overflow";
    }

    return what + ": tensor '" + name + "' (" + type + ") data region [data_off="
        + std::to_string(data_off) + " + tensor_off=" + std::to_string(tensor_off)
        + " + size=" + std::to_string(tensor_sz) + " = " + end_str
        + "] exceeds file size " + std::to_string(file_size)
        + ". The GGUF is truncated or corrupt, or its data-section offset/alignment "
          "does not match the writer.";
}

} // namespace dflash::common
