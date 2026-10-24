#pragma once

#include "../common/vision/image_decode.h"
#include "../common/vision/image_spans.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::vision {

// How an image's aligner grid becomes decoder positions.
enum class ImageLayoutKind {
    // DeepSeek-V4-Flash-Vision-Exp: row pairs interleaved by column, rows
    // padded to even, compression padding to a multiple of four positions.
    RowPairInterleave,
    // DeepSeek-V4.1-Flash: [start] + ([image] * columns + [newline]) * rows
    // + [end], aligner rows in reading order, no padding.
    ReadingOrder,
};

// The fixed recipe of one checkpoint. The defaults are DeepSeek-V4's;
// v41_preprocess_config() is DeepSeek-V4.1's. No other combination is valid.
struct PreprocessConfig {
    std::uint32_t patch_size = 14;
    std::uint32_t downsample_ratio = 3;
    std::uint32_t max_tokens = 384;
    std::uint64_t min_pixels = 147456;
    std::uint32_t max_aspect_ratio = 8;   // 0: no width limit
    std::uint32_t compress_pad_to = 4;    // 1: no compression padding
    std::uint32_t vocab_size = 129280;
    float normalization_mean = 0.5F;
    float normalization_std = 0.5F;
    ImageLayoutKind layout = ImageLayoutKind::RowPairInterleave;
};

// DeepSeek-V4.1-Flash inference/image_processor.py: up to 1024 tokens, a
// 544x544 pixel floor, no aspect limit (always ImageOps.pad), reading order.
inline PreprocessConfig v41_preprocess_config() {
    PreprocessConfig config;
    config.max_tokens = 1024;
    config.min_pixels = 295936;
    config.max_aspect_ratio = 0;
    config.compress_pad_to = 1;
    config.layout = ImageLayoutKind::ReadingOrder;
    return config;
}

struct PreprocessLimits {
    // The RGB entrypoint receives caller-owned memory, but validates these limits
    // before allocating resized or patch buffers. A decoder can use the same
    // dimension check before allocating its decoded image.
    std::uint64_t max_decoded_pixels = 64ULL * 1024ULL * 1024ULL;
    std::uint32_t max_dimension = 65'535;
    std::uint64_t max_output_pixels = 16ULL * 1024ULL * 1024ULL;
};

enum class ImageTokenType : std::int64_t {
    Start = 0,
    Pad = 1,
    Image = 2,
    Newline = 3,
    End = 4,
};

struct ResizePlan {
    std::uint32_t resized_width = 0;
    std::uint32_t resized_height = 0;
    std::uint32_t vit_rows = 0;
    std::uint32_t vit_cols = 0;
    std::uint32_t aligner_rows = 0;
    std::uint32_t aligner_cols = 0;
    bool direct_resize = false;
};

struct ImageLayout {
    std::vector<ImageTokenType> types;
    std::vector<std::int64_t> permutation;
    TokenSpan span;
};

struct PreparedImage {
    ResizePlan plan;
    std::vector<std::uint8_t> resized_rgb;
    // Raw IEEE bfloat16 words in native integer representation. The tensor
    // shape is [vit_rows * vit_cols, 3, patch_size, patch_size].
    std::vector<std::uint16_t> patches_bf16;
    ImageLayout layout;
};

enum class PreprocessError {
    None = 0,
    InvalidConfig,
    InvalidDimensions,
    InputSizeMismatch,
    InputTooLarge,
    ResizePlanFailed,
    OutputTooLarge,
    TokenBudgetExceeded,
    PositionOverflow,
};

struct PreprocessStatus {
    PreprocessError code = PreprocessError::None;
    std::string message;

    explicit operator bool() const { return code == PreprocessError::None; }
};

struct PreprocessResult {
    PreprocessStatus status;
    PreparedImage image;

    explicit operator bool() const { return static_cast<bool>(status); }
};

PreprocessStatus validate_config(const PreprocessConfig & config);

PreprocessStatus validate_decoded_dimensions(
    std::uint32_t width,
    std::uint32_t height,
    const PreprocessLimits & limits = {});

PreprocessStatus plan_image(
    std::uint32_t width,
    std::uint32_t height,
    ResizePlan & plan,
    const PreprocessConfig & config = {},
    const PreprocessLimits & limits = {});

PreprocessStatus build_image_layout(
    std::uint32_t aligner_rows,
    std::uint32_t aligner_cols,
    std::uint64_t start_position,
    ImageLayout & layout,
    const PreprocessConfig & config = {});

PreprocessResult preprocess_rgb(
    const DecodedRgbView & input,
    std::uint64_t start_position,
    const PreprocessConfig & config = {},
    const PreprocessLimits & limits = {});

const char * preprocess_error_name(PreprocessError error);

}  // namespace luce::vision
