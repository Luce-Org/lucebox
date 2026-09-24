#pragma once

#include "deepseek4_image_prompt.h"
#include <functional>
#include <memory>

namespace luce::vision {

struct ImageRaster {
    size_t rows = 0, columns = 0;
    std::vector<float> values;
};
struct ImageSentinels {
    std::vector<float> start, pad, newline, end;
};
using ImageRows = std::vector<std::vector<float>>;
using ImageEncode = std::function<bool(const PromptImage &, ImageRaster &, std::string &)>;
using ImageCancelled = std::function<bool()>;
using TextEmbed = std::function<bool(const int32_t *, size_t, float *)>;

// All outputs are replaced only on success. The permutation maps each Image
// token to one unique raster row; sentinel identities never use enum casts.
bool assemble_image_rows(const ImageLayout &, const ImageRaster &, const ImageSentinels &,
                         size_t dimension, std::vector<float> & output, std::string & error);

// Synchronous callbacks; cancellation is checked before/after every encode
// and before commit. Failure, cancellation, or callback exception leaves the
// caller's previous result intact. The caller owns GPU lifetime and scratch.
bool materialize_image_rows(const std::vector<PromptImage> &, const ImageSentinels &,
                            size_t dimension, const ImageEncode &, const ImageCancelled &,
                            ImageRows & output, std::string & error);

// Only complete image blocks may intersect a chunk. The normal embed callback
// receives validated ordinary token IDs only; mixed output commits atomically.
bool embed_image_prompt_chunk(const PreparedImagePrompt &, const ImageRows &,
                               int32_t vocabulary, size_t dimension,
                               size_t position, size_t count, const TextEmbed &,
                               std::vector<float> & output, std::string & error);

} // namespace luce::vision
