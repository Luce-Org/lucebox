#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::vision {

struct PreprocessConfig {
    std::uint32_t patch_size = 14;
    std::uint32_t downsample_ratio = 3;
    std::uint32_t max_tokens = 384;
    std::uint64_t min_pixels = 147456;
    std::uint32_t max_aspect_ratio = 8;
    std::uint32_t compress_pad_to = 4;
    std::uint32_t vocab_size = 129280;
    float normalization_mean = 0.5F;
    float normalization_std = 0.5F;
};

struct PreprocessLimits {
    // The RGB entrypoint receives caller-owned memory, but validates these limits
    // before allocating resized or patch buffers. A decoder can use the same
    // dimension check before allocating its decoded image.
    std::uint64_t max_decoded_pixels = 64ULL * 1024ULL * 1024ULL;
    std::uint32_t max_dimension = 65'535;
    std::uint64_t max_output_pixels = 16ULL * 1024ULL * 1024ULL;
};

struct DecodedRgbView {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    const std::uint8_t * data = nullptr;
    std::size_t size = 0;
};

enum class ImageTokenType : std::int64_t {
    Start = 0,
    Pad = 1,
    Image = 2,
    Newline = 3,
    End = 4,
};

struct TokenSpan {
    // All intervals are half-open absolute token positions.
    std::uint64_t block_begin = 0;
    std::uint64_t visible_begin = 0;
    std::uint64_t visible_end = 0;
    std::uint64_t block_end = 0;
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

}  // namespace dflash::vision
