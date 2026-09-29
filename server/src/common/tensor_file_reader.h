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
    bool open(const std::string & path, std::string * err = nullptr);
    bool is_open() const { return file_.is_open(); }
    bool direct() const { return file_.direct(); }
    uint64_t size() const { return file_.size(); }

    // Copies every span into its tensor (ggml_backend_tensor_set); false with
    // *err on a read error or a span outside the file or its tensor. Spans may
    // target any backend buffer that accepts ggml_backend_tensor_set for a
    // byte range (not a split buffer, whose tensors take only whole writes);
    // the writes are serialized.
    bool load(const std::vector<TensorFileSpan> & spans, std::string * err = nullptr) const;

private:
    ReadOnlyFile file_;
};

// Loads `spans` of the file at `path`, the same file as `mapping` (mapped from
// offset zero; the caller checked the bounds). Spans already in the page cache
// are copied straight out of the mapping, the fastest warm start; otherwise a
// TensorFileReader reads them. The mapping is also the fallback where the
// reader cannot open the file (or LUCE_NO_PREAD is set).
bool load_tensor_spans(const std::string & path, const void * mapping,
                       const std::vector<TensorFileSpan> & spans, std::string * err = nullptr);

}  // namespace luce::common
