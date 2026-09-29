// Loads model weights from their file into backend tensors.
//
// A loader lists what to copy as spans (file bytes -> tensor bytes) and hands
// them over at once. Several threads read the spans, so several drive reads
// are in flight, and each read is written into its tensor as it lands. A file
// larger than half the host's available memory is read with O_DIRECT: caching
// it would evict what the host needs (on a unified-memory device, the loaded
// weights themselves). A smaller file is read through the page cache, so a
// restart finds it there.

#pragma once

#include "platform_io.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

namespace luce::common {

struct TensorFileSpan {
    ggml_tensor * tensor = nullptr;
    size_t   tensor_offset = 0;  // byte offset inside the tensor
    uint64_t file_offset = 0;
    size_t   size = 0;
};

class TensorFileReader {
public:
    // Auto: direct reads for a file larger than half the available memory.
    // Buffered / Direct force the choice (tests).
    enum class Mode { Auto, Buffered, Direct };

    bool open(const std::string & path, std::string * err = nullptr, Mode mode = Mode::Auto);
    bool is_open() const { return buffered_.is_open(); }
    bool direct() const { return direct_.is_open(); }
    uint64_t size() const { return buffered_.size(); }

    // Copies every span into its tensor (ggml_backend_tensor_set); false with
    // *err on a read error or a span outside the file or its tensor. A span
    // of a meta (tensor-parallel) buffer is written whole; other buffers take
    // it in pieces. A CUDA split buffer is not a valid target: its tensors
    // take only whole writes and it cannot be recognized here. The writes are
    // serialized.
    bool load(const std::vector<TensorFileSpan> & spans, std::string * err = nullptr) const;

private:
    ReadOnlyFile buffered_;  // always open; also the fallback of a failed direct read
    ReadOnlyFile direct_;    // open when reads bypass the page cache
};

// Loads `spans` of the file at `path`, the same file as `mapping` (mapped from
// offset zero, `mapping_size` bytes). Spans outside the mapping are refused.
// Spans already in the page cache are copied straight out of the mapping, the
// fastest warm start; otherwise a TensorFileReader reads them. The mapping is
// also the fallback where the reader cannot open the file (or LUCE_NO_PREAD is
// set).
bool load_tensor_spans(const std::string & path, const void * mapping, size_t mapping_size,
                       const std::vector<TensorFileSpan> & spans, std::string * err = nullptr);

}  // namespace luce::common
