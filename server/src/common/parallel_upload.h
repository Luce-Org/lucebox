// Weight upload on several threads.
//
// A loader that copies each tensor from a read-only file mapping with one
// ggml_backend_tensor_set call after another is bound by page faults on that
// one thread: on R9700 + Strix Halo it read 0.52 GB/s from a file not in the
// page cache, where the NVMe gives 4.5 GB/s, and 9 GB/s from a cached file,
// where 16 threads copy 31 GB/s. Splitting the tensors into pieces and
// uploading them from a pool of threads keeps the SSD queue full and the
// staged copies overlapping. ggml-cuda's set_tensor keeps its staging buffers
// and stream per thread, so pieces of one tensor, or of tensors on different
// devices, need no ordering among themselves.

#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::common {

struct UploadPiece {
    ggml_tensor *   tensor = nullptr;
    const uint8_t * src    = nullptr;   // the bytes for [offset, offset + size) of the tensor
    size_t          offset = 0;
    size_t          size   = 0;
};

// Appends `tensor`'s bytes [0, size) from `src` as pieces of at most `piece` bytes.
void append_upload_pieces(std::vector<UploadPiece> & pieces, ggml_tensor * tensor,
                          const uint8_t * src, size_t size, size_t piece = (size_t) 64 << 20);

// Uploads every piece, then returns once all are on their devices. Each
// destination buffer type gets its own threads, all at once: `threads` each
// when > 0, else upload_threads(), capped at 4 (LUCE_LOAD_THREADS_DISCRETE)
// for a discrete GPU. Returns the bytes uploaded.
size_t upload_pieces(const std::vector<UploadPiece> & pieces, int threads = 0);

// LUCE_LOAD_THREADS, else min(16, hardware threads).
int upload_threads();

}  // namespace luce::common
