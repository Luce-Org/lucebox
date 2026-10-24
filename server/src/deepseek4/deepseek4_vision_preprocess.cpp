#include "deepseek4_vision_preprocess.h"
#include "../common/vision/image_resize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace luce::vision {
namespace {


PreprocessStatus ok() {
    return {};
}

PreprocessStatus fail(PreprocessError code, std::string message) {
    return {code, std::move(message)};
}

bool checked_mul(std::uint64_t a, std::uint64_t b, std::uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool checked_add(std::uint64_t a, std::uint64_t b, std::uint64_t & out) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

std::uint64_t round_ties_to_even(double value) {
    const double base_double = std::floor(value);
    const double fraction = value - base_double;
    const auto base = static_cast<std::uint64_t>(base_double);
    if (fraction > 0.5 || (fraction == 0.5 && (base & 1U) != 0)) {
        return base + 1;
    }
    return base;
}

struct GridTokens {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint64_t tokens = 0;
};

PreprocessStatus grid_tokens(
    std::uint32_t height,
    std::uint32_t width,
    const PreprocessConfig & config,
    GridTokens & result) {
    if (height == 0 || width == 0 || height % config.patch_size != 0 ||
        width % config.patch_size != 0) {
        return fail(PreprocessError::ResizePlanFailed,
                    "planned dimensions must be nonzero patch multiples");
    }

    const std::uint64_t vit_rows = height / config.patch_size;
    const std::uint64_t vit_cols = width / config.patch_size;
    const std::uint64_t rows =
        (vit_rows + config.downsample_ratio - 1) / config.downsample_ratio;
    const std::uint64_t cols =
        (vit_cols + config.downsample_ratio - 1) / config.downsample_ratio;
    if (rows == 0 || cols == 0 || rows > std::numeric_limits<std::uint32_t>::max() ||
        cols > std::numeric_limits<std::uint32_t>::max()) {
        return fail(PreprocessError::ResizePlanFailed, "aligner grid is out of range");
    }

    std::uint64_t row_len = 0;
    std::uint64_t tokens = 0;
    if (!checked_add(cols, 1, row_len) || !checked_mul(rows, row_len, tokens) ||
        !checked_add(tokens, 2, tokens)) {
        return fail(PreprocessError::ResizePlanFailed, "layout token count overflow");
    }
    if ((rows & 1U) != 0 && !checked_add(tokens, row_len, tokens)) {
        return fail(PreprocessError::ResizePlanFailed, "layout row padding overflow");
    }
    const std::uint64_t trailing = (((rows + 1) / 2) * row_len % 2) * 2;
    if (!checked_add(tokens, trailing, tokens)) {
        return fail(PreprocessError::ResizePlanFailed, "layout tail padding overflow");
    }

    result.rows = static_cast<std::uint32_t>(rows);
    result.cols = static_cast<std::uint32_t>(cols);
    result.tokens = tokens;
    return ok();
}

PreprocessStatus solve_resize_ratio(
    std::uint32_t height,
    std::uint32_t width,
    std::uint32_t max_tokens,
    const PreprocessConfig & config,
    std::uint32_t & best_height,
    std::uint32_t & best_width,
    GridTokens & grid) {
    if (max_tokens <= 2 || height == 0 || width == 0) {
        return fail(PreprocessError::ResizePlanFailed, "resize budget is too small");
    }

    const double ratio = static_cast<double>(height) / static_cast<double>(width);
    const double max_width_float =
        std::sqrt((static_cast<double>(max_tokens) - 2.0) / ratio + 0.25) - 0.5;
    const double max_height_float = max_width_float * ratio;
    std::uint64_t solved_width = 0;
    std::uint64_t solved_height = 0;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(config.patch_size) * config.downsample_ratio;

    if (max_width_float < 1.0) {
        const std::uint64_t max_width = 1;
        std::uint64_t max_height = (max_tokens - 2) / (max_width + 1);
        if ((max_height & 1U) != 0) {
            --max_height;
        }
        solved_width = max_width * stride;
        solved_height = max_height * stride;
    } else if (max_height_float < 2.0) {
        const std::uint64_t max_height = 2;
        const std::uint64_t max_width = (max_tokens - 2) / max_height - 1;
        if (max_width <= 1) {
            return fail(PreprocessError::ResizePlanFailed,
                        "resize budget cannot represent a wide image");
        }
        solved_width = max_width * stride;
        solved_height = max_height * stride;
    } else {
        const auto max_width = static_cast<std::uint64_t>(std::floor(max_width_float));
        auto max_height = static_cast<std::uint64_t>(std::floor(max_height_float));
        if ((max_height & 1U) != 0) {
            --max_height;
        }
        if (max_width == 0 || max_height == 0) {
            return fail(PreprocessError::ResizePlanFailed, "resize solver produced an empty grid");
        }
        const double beta = std::min(
            static_cast<double>(max_width * stride) / static_cast<double>(width),
            static_cast<double>(max_height * stride) / static_cast<double>(height));
        solved_width = static_cast<std::uint64_t>(
            std::floor(static_cast<double>(width) * beta / config.patch_size)) *
            config.patch_size;
        solved_height = static_cast<std::uint64_t>(
            std::floor(static_cast<double>(height) * beta / config.patch_size)) *
            config.patch_size;
    }

    if (solved_width == 0 || solved_height == 0 ||
        solved_width > std::numeric_limits<std::uint32_t>::max() ||
        solved_height > std::numeric_limits<std::uint32_t>::max()) {
        return fail(PreprocessError::ResizePlanFailed,
                    "resize solver produced out-of-range dimensions");
    }
    best_width = static_cast<std::uint32_t>(solved_width);
    best_height = static_cast<std::uint32_t>(solved_height);
    return grid_tokens(best_height, best_width, config, grid);
}

// V4.1 image_processor.py llm_grid / num_image_tokens.
PreprocessStatus reading_order_grid(
    std::uint32_t height,
    std::uint32_t width,
    const PreprocessConfig & config,
    GridTokens & result) {
    if (height == 0 || width == 0 || height % config.patch_size != 0 ||
        width % config.patch_size != 0) {
        return fail(PreprocessError::ResizePlanFailed,
                    "planned dimensions must be nonzero patch multiples");
    }
    const std::uint64_t rows =
        (height / config.patch_size + config.downsample_ratio - 1) / config.downsample_ratio;
    const std::uint64_t cols =
        (width / config.patch_size + config.downsample_ratio - 1) / config.downsample_ratio;
    std::uint64_t tokens = 0;
    if (rows == 0 || cols == 0 || rows > std::numeric_limits<std::uint32_t>::max() ||
        cols > std::numeric_limits<std::uint32_t>::max() ||
        !checked_mul(rows, cols + 1, tokens) || !checked_add(tokens, 2, tokens)) {
        return fail(PreprocessError::ResizePlanFailed, "aligner grid is out of range");
    }
    result.rows = static_cast<std::uint32_t>(rows);
    result.cols = static_cast<std::uint32_t>(cols);
    result.tokens = tokens;
    return ok();
}

// V4.1 image_processor.py solve_resize_ratio: the largest aspect-preserving
// size whose grid fits max_tokens, collapsing to one column or one row when
// the image is too tall or too wide for even that. The float operations and
// their order are the source's (Python floats are IEEE doubles).
PreprocessStatus solve_reading_order_resize(
    std::uint32_t height,
    std::uint32_t width,
    const PreprocessConfig & config,
    std::uint32_t & best_height,
    std::uint32_t & best_width) {
    const std::uint64_t max_tokens = config.max_tokens;
    if (max_tokens <= 3 || height == 0 || width == 0) {
        return fail(PreprocessError::ResizePlanFailed, "resize budget is too small");
    }
    const double ratio = static_cast<double>(height) / static_cast<double>(width);
    const double max_width_float =
        std::sqrt(static_cast<double>(max_tokens - 2) / ratio + 0.25) - 0.5;
    const double max_height_float = max_width_float * ratio;
    const std::uint64_t cell =
        static_cast<std::uint64_t>(config.patch_size) * config.downsample_ratio;
    std::uint64_t solved_height = 0;
    std::uint64_t solved_width = 0;
    if (max_width_float < 1.0) {
        solved_height = (max_tokens - 2) / 2 * cell;
        solved_width = cell;
    } else if (max_height_float < 1.0) {
        solved_height = cell;
        solved_width = (max_tokens - 3) * cell;
    } else {
        const double beta = std::min(
            std::floor(max_width_float) * static_cast<double>(cell) / static_cast<double>(width),
            std::floor(max_height_float) * static_cast<double>(cell) / static_cast<double>(height));
        solved_height = static_cast<std::uint64_t>(
            std::floor(static_cast<double>(height) * beta / config.patch_size)) * config.patch_size;
        solved_width = static_cast<std::uint64_t>(
            std::floor(static_cast<double>(width) * beta / config.patch_size)) * config.patch_size;
    }
    if (solved_width == 0 || solved_height == 0 ||
        solved_width > std::numeric_limits<std::uint32_t>::max() ||
        solved_height > std::numeric_limits<std::uint32_t>::max()) {
        return fail(PreprocessError::ResizePlanFailed,
                    "resize solver produced out-of-range dimensions");
    }
    best_height = static_cast<std::uint32_t>(solved_height);
    best_width = static_cast<std::uint32_t>(solved_width);
    return ok();
}

PreprocessStatus pillow_resize(
    const std::vector<std::uint8_t> & input,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    std::vector<std::uint8_t> & output) {
    std::string message;
    if (!resize_rgb_bicubic(input, input_width, input_height, output_width, output_height, output, message)) {
        return fail(PreprocessError::ResizePlanFailed, message);
    }
    return ok();
}

PreprocessStatus resize_and_pad(
    const std::vector<std::uint8_t> & input,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    bool direct_resize,
    std::vector<std::uint8_t> & output) {
    if (direct_resize) {
        return pillow_resize(
            input, input_width, input_height, output_width, output_height, output);
    }

    int contained_width = output_width;
    int contained_height = output_height;
    const double input_ratio = static_cast<double>(input_width) / input_height;
    const double destination_ratio = static_cast<double>(output_width) / output_height;
    if (input_ratio != destination_ratio) {
        if (input_ratio > destination_ratio) {
            contained_height = static_cast<int>(round_ties_to_even(
                static_cast<double>(input_height) / input_width * output_width));
        } else {
            contained_width = static_cast<int>(round_ties_to_even(
                static_cast<double>(input_width) / input_height * output_height));
        }
    }
    if (contained_width <= 0 || contained_height <= 0) {
        return fail(PreprocessError::ResizePlanFailed,
                    "ImageOps.contain produced an empty dimension");
    }

    std::vector<std::uint8_t> contained;
    if (const auto status = pillow_resize(
            input, input_width, input_height, contained_width, contained_height, contained);
        !status) {
        return status;
    }
    if (contained_width == output_width && contained_height == output_height) {
        output = std::move(contained);
        return ok();
    }

    output.assign(static_cast<std::size_t>(output_width) * output_height * 3, 127);
    const int offset_x = contained_width == output_width
        ? 0
        : static_cast<int>(round_ties_to_even((output_width - contained_width) * 0.5));
    const int offset_y = contained_width != output_width
        ? 0
        : static_cast<int>(round_ties_to_even((output_height - contained_height) * 0.5));
    for (int y = 0; y < contained_height; ++y) {
        const auto * source = contained.data() + static_cast<std::size_t>(y) * contained_width * 3;
        auto * destination = output.data() +
            (static_cast<std::size_t>(y + offset_y) * output_width + offset_x) * 3;
        std::memcpy(destination, source, static_cast<std::size_t>(contained_width) * 3);
    }
    return ok();
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

std::uint16_t normalize_pixel(std::uint8_t pixel) {
    // Keep the source operation sequence and its F32 rounding points.
    volatile float normalized = static_cast<float>(pixel);
    normalized = normalized / 255.0F;
    normalized = normalized - 0.5F;
    normalized = normalized / 0.5F;
    return float_to_bf16(normalized);
}

PreprocessStatus make_patches(
    const std::vector<std::uint8_t> & rgb,
    const ResizePlan & plan,
    const PreprocessConfig & config,
    std::vector<std::uint16_t> & patches) {
    std::uint64_t pixel_count = 0;
    std::uint64_t value_count = 0;
    if (!checked_mul(plan.resized_width, plan.resized_height, pixel_count) ||
        !checked_mul(pixel_count, 3, value_count) ||
        value_count > std::numeric_limits<std::size_t>::max()) {
        return fail(PreprocessError::OutputTooLarge, "patch tensor size overflow");
    }
    if (rgb.size() != value_count) {
        return fail(PreprocessError::ResizePlanFailed, "resized RGB buffer has the wrong size");
    }
    patches.resize(static_cast<std::size_t>(value_count));

    const std::uint32_t patch = config.patch_size;
    std::size_t destination = 0;
    for (std::uint32_t patch_y = 0; patch_y < plan.vit_rows; ++patch_y) {
        for (std::uint32_t patch_x = 0; patch_x < plan.vit_cols; ++patch_x) {
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                for (std::uint32_t y = 0; y < patch; ++y) {
                    for (std::uint32_t x = 0; x < patch; ++x) {
                        const std::uint32_t source_y = patch_y * patch + y;
                        const std::uint32_t source_x = patch_x * patch + x;
                        const std::size_t source =
                            (static_cast<std::size_t>(source_y) * plan.resized_width + source_x) *
                                3 +
                            channel;
                        patches[destination++] = normalize_pixel(rgb[source]);
                    }
                }
            }
        }
    }
    return ok();
}

}  // namespace

PreprocessStatus validate_config(const PreprocessConfig & config) {
    const PreprocessConfig expected = config.layout == ImageLayoutKind::ReadingOrder
        ? v41_preprocess_config() : PreprocessConfig{};
    if (config.layout != expected.layout || config.patch_size != expected.patch_size ||
        config.downsample_ratio != expected.downsample_ratio ||
        config.max_tokens != expected.max_tokens ||
        config.min_pixels != expected.min_pixels ||
        config.max_aspect_ratio != expected.max_aspect_ratio ||
        config.compress_pad_to != expected.compress_pad_to ||
        config.vocab_size != expected.vocab_size ||
        config.normalization_mean != expected.normalization_mean ||
        config.normalization_std != expected.normalization_std) {
        return fail(PreprocessError::InvalidConfig,
                    "preprocessing config does not match a fixed DeepSeek vision recipe");
    }
    return ok();
}

PreprocessStatus validate_decoded_dimensions(
    std::uint32_t width,
    std::uint32_t height,
    const PreprocessLimits & limits) {
    if (width == 0 || height == 0) {
        return fail(PreprocessError::InvalidDimensions, "decoded RGB dimensions must be positive");
    }
    if (limits.max_decoded_pixels == 0 || limits.max_dimension == 0 ||
        limits.max_output_pixels == 0) {
        return fail(PreprocessError::InvalidConfig, "preprocessing limits must be positive");
    }
    if (width > limits.max_dimension || height > limits.max_dimension) {
        return fail(PreprocessError::InputTooLarge, "decoded RGB dimension exceeds the limit");
    }
    std::uint64_t pixels = 0;
    if (!checked_mul(width, height, pixels) || pixels > limits.max_decoded_pixels) {
        return fail(PreprocessError::InputTooLarge, "decoded RGB pixel count exceeds the limit");
    }
    return ok();
}

PreprocessStatus plan_image(
    std::uint32_t width,
    std::uint32_t height,
    ResizePlan & plan,
    const PreprocessConfig & config,
    const PreprocessLimits & limits) {
    if (const auto status = validate_config(config); !status) {
        return status;
    }
    if (const auto status = validate_decoded_dimensions(width, height, limits); !status) {
        return status;
    }

    // V4.1 has no aspect limit (vision_max_wh_ratio is None): no clamp, and
    // every image goes through ImageOps.pad.
    const bool limited_aspect = config.max_aspect_ratio != 0;
    const bool direct_resize = limited_aspect &&
        static_cast<std::uint64_t>(width) >=
        static_cast<std::uint64_t>(height) * config.max_aspect_ratio;
    std::uint64_t planned_width = width;
    std::uint64_t planned_height = height;
    const std::uint64_t max_width =
        static_cast<std::uint64_t>(height) * config.max_aspect_ratio;
    if (limited_aspect && planned_width > max_width) {
        planned_width = max_width;
    }

    std::uint64_t planned_pixels = 0;
    if (!checked_mul(planned_width, planned_height, planned_pixels)) {
        return fail(PreprocessError::ResizePlanFailed, "planned pixel count overflow");
    }
    if (planned_pixels > 0 && planned_pixels < config.min_pixels) {
        const double ratio = std::sqrt(
            static_cast<double>(config.min_pixels) / static_cast<double>(planned_pixels));
        planned_width = static_cast<std::uint64_t>(planned_width * ratio);
        planned_height = static_cast<std::uint64_t>(planned_height * ratio);
    }
    if (planned_width == 0 || planned_height == 0 ||
        planned_width > std::numeric_limits<std::uint32_t>::max() ||
        planned_height > std::numeric_limits<std::uint32_t>::max()) {
        return fail(PreprocessError::ResizePlanFailed,
                    "minimum-pixel scaling produced invalid dimensions");
    }

    const std::uint64_t aligned_width =
        ((planned_width + config.patch_size - 1) / config.patch_size) * config.patch_size;
    const std::uint64_t aligned_height =
        ((planned_height + config.patch_size - 1) / config.patch_size) * config.patch_size;
    if (aligned_width > std::numeric_limits<std::uint32_t>::max() ||
        aligned_height > std::numeric_limits<std::uint32_t>::max()) {
        return fail(PreprocessError::ResizePlanFailed, "aligned dimensions are out of range");
    }
    std::uint32_t best_width = static_cast<std::uint32_t>(aligned_width);
    std::uint32_t best_height = static_cast<std::uint32_t>(aligned_height);
    GridTokens grid;
    if (config.layout == ImageLayoutKind::ReadingOrder) {
        // V4.1 safe_resize: one solve when the natural size costs too much.
        if (const auto status = reading_order_grid(best_height, best_width, config, grid); !status) {
            return status;
        }
        if (grid.tokens > config.max_tokens) {
            if (const auto status = solve_reading_order_resize(
                    static_cast<std::uint32_t>(planned_height),
                    static_cast<std::uint32_t>(planned_width), config, best_height, best_width);
                !status) {
                return status;
            }
            if (const auto status = reading_order_grid(best_height, best_width, config, grid); !status) {
                return status;
            }
            if (grid.tokens > config.max_tokens) {
                return fail(PreprocessError::ResizePlanFailed,
                            "could not solve image dimensions within the token budget");
            }
        }
    } else if (const auto status = grid_tokens(best_height, best_width, config, grid); !status) {
        return status;
    }

    const std::uint32_t reserved = config.compress_pad_to - 1;
    if (config.max_tokens <= reserved + 2) {
        return fail(PreprocessError::InvalidConfig, "token budget cannot hold image sentinels");
    }
    const std::uint32_t image_budget = config.max_tokens - reserved;
    std::uint32_t solver_budget = image_budget;
    std::uint32_t attempts = 0;
    while (config.layout == ImageLayoutKind::RowPairInterleave && grid.tokens > image_budget) {
        if (solver_budget <= 2 || ++attempts > config.max_tokens) {
            return fail(PreprocessError::ResizePlanFailed,
                        "could not solve image dimensions within the token budget");
        }
        if (const auto status = solve_resize_ratio(
                static_cast<std::uint32_t>(planned_height),
                static_cast<std::uint32_t>(planned_width),
                solver_budget,
                config,
                best_height,
                best_width,
                grid);
            !status) {
            return status;
        }
        --solver_budget;
    }

    std::uint64_t output_pixels = 0;
    if (!checked_mul(best_width, best_height, output_pixels) ||
        output_pixels > limits.max_output_pixels) {
        return fail(PreprocessError::OutputTooLarge,
                    "planned resized image exceeds the output pixel limit");
    }
    plan.resized_width = best_width;
    plan.resized_height = best_height;
    plan.vit_rows = best_height / config.patch_size;
    plan.vit_cols = best_width / config.patch_size;
    plan.aligner_rows = grid.rows;
    plan.aligner_cols = grid.cols;
    plan.direct_resize = direct_resize;
    return ok();
}

PreprocessStatus build_image_layout(
    std::uint32_t aligner_rows,
    std::uint32_t aligner_cols,
    std::uint64_t start_position,
    ImageLayout & layout,
    const PreprocessConfig & config) {
    if (const auto status = validate_config(config); !status) {
        return status;
    }
    if (aligner_rows == 0 || aligner_cols == 0) {
        return fail(PreprocessError::InvalidDimensions, "aligner grid dimensions must be positive");
    }
    if (config.layout == ImageLayoutKind::ReadingOrder) {
        std::uint64_t total = 0;
        if (!checked_mul(aligner_rows, static_cast<std::uint64_t>(aligner_cols) + 1, total) ||
            !checked_add(total, 2, total)) {
            return fail(PreprocessError::TokenBudgetExceeded, "layout token count overflow");
        }
        if (total > config.max_tokens) {
            std::ostringstream message;
            message << "image layout needs " << total << " tokens, limit is " << config.max_tokens;
            return fail(PreprocessError::TokenBudgetExceeded, message.str());
        }
        std::uint64_t block_end = 0;
        if (!checked_add(start_position, total, block_end)) {
            return fail(PreprocessError::PositionOverflow, "absolute image span overflow");
        }
        layout.types.clear();
        layout.permutation.clear();
        layout.types.reserve(static_cast<std::size_t>(total));
        layout.permutation.reserve(static_cast<std::size_t>(aligner_rows) * aligner_cols);
        layout.types.push_back(ImageTokenType::Start);
        for (std::uint64_t row = 0; row < aligner_rows; ++row) {
            for (std::uint64_t column = 0; column < aligner_cols; ++column) {
                layout.types.push_back(ImageTokenType::Image);
                layout.permutation.push_back(static_cast<std::int64_t>(row * aligner_cols + column));
            }
            layout.types.push_back(ImageTokenType::Newline);
        }
        layout.types.push_back(ImageTokenType::End);
        // Causal decoder: the whole block is the image, none of it pads.
        layout.span = {start_position, start_position, block_end, block_end};
        return ok();
    }

    const std::uint64_t leading_pad =
        config.compress_pad_to - 1 - start_position % config.compress_pad_to;
    const std::uint64_t padded_rows =
        static_cast<std::uint64_t>(aligner_rows) + (aligner_rows % 2);
    const std::uint64_t row_length = static_cast<std::uint64_t>(aligner_cols) + 1;
    std::uint64_t body_size = 0;
    if (!checked_mul(padded_rows, row_length, body_size)) {
        return fail(PreprocessError::TokenBudgetExceeded, "layout body size overflow");
    }
    const std::uint64_t trailing_pad = (padded_rows / 2 * row_length % 2) * 2;
    std::uint64_t total = 0;
    if (!checked_add(leading_pad, 1, total) || !checked_add(total, body_size, total) ||
        !checked_add(total, trailing_pad, total) || !checked_add(total, 1, total)) {
        return fail(PreprocessError::TokenBudgetExceeded, "layout token count overflow");
    }
    if (total > config.max_tokens) {
        std::ostringstream message;
        message << "image layout needs " << total << " tokens, limit is " << config.max_tokens;
        return fail(PreprocessError::TokenBudgetExceeded, message.str());
    }
    std::uint64_t block_end = 0;
    std::uint64_t visible_begin = 0;
    if (!checked_add(start_position, total, block_end) ||
        !checked_add(start_position, leading_pad, visible_begin)) {
        return fail(PreprocessError::PositionOverflow, "absolute image span overflow");
    }

    layout.types.clear();
    layout.permutation.clear();
    layout.types.reserve(static_cast<std::size_t>(total));
    layout.permutation.reserve(static_cast<std::size_t>(aligner_rows) * aligner_cols);
    layout.types.insert(layout.types.end(), static_cast<std::size_t>(leading_pad),
                        ImageTokenType::Pad);
    layout.types.push_back(ImageTokenType::Start);

    for (std::uint64_t pair = 0; pair < padded_rows / 2; ++pair) {
        for (std::uint64_t column = 0; column < row_length; ++column) {
            for (std::uint64_t within_pair = 0; within_pair < 2; ++within_pair) {
                const std::uint64_t row = pair * 2 + within_pair;
                if (row < aligner_rows && column < aligner_cols) {
                    layout.types.push_back(ImageTokenType::Image);
                    layout.permutation.push_back(static_cast<std::int64_t>(
                        row * aligner_cols + column));
                } else if (row < aligner_rows && column == aligner_cols) {
                    layout.types.push_back(ImageTokenType::Newline);
                } else {
                    layout.types.push_back(ImageTokenType::Pad);
                }
            }
        }
    }
    layout.types.insert(layout.types.end(), static_cast<std::size_t>(trailing_pad),
                        ImageTokenType::Pad);
    layout.types.push_back(ImageTokenType::End);
    layout.span = {start_position, visible_begin, block_end, block_end};
    return ok();
}

PreprocessResult preprocess_rgb(
    const DecodedRgbView & input,
    std::uint64_t start_position,
    const PreprocessConfig & config,
    const PreprocessLimits & limits) {
    PreprocessResult result;
    if (const auto status = validate_config(config); !status) {
        result.status = status;
        return result;
    }
    if (const auto status = validate_decoded_dimensions(input.width, input.height, limits);
        !status) {
        result.status = status;
        return result;
    }
    std::uint64_t pixels = 0;
    std::uint64_t expected_bytes = 0;
    if (!checked_mul(input.width, input.height, pixels) ||
        !checked_mul(pixels, 3, expected_bytes) || input.data == nullptr ||
        expected_bytes != input.size) {
        result.status = fail(PreprocessError::InputSizeMismatch,
                             "decoded RGB byte count must equal width * height * 3");
        return result;
    }
    if (const auto status = plan_image(
            input.width, input.height, result.image.plan, config, limits);
        !status) {
        result.status = status;
        return result;
    }
    if (const auto status = build_image_layout(
            result.image.plan.aligner_rows,
            result.image.plan.aligner_cols,
            start_position,
            result.image.layout,
            config);
        !status) {
        result.status = status;
        return result;
    }

    std::vector<std::uint8_t> owned_input(input.data, input.data + input.size);
    if (const auto status = resize_and_pad(
            owned_input,
            static_cast<int>(input.width),
            static_cast<int>(input.height),
            static_cast<int>(result.image.plan.resized_width),
            static_cast<int>(result.image.plan.resized_height),
            result.image.plan.direct_resize,
            result.image.resized_rgb);
        !status) {
        result.status = status;
        return result;
    }
    if (const auto status = make_patches(
            result.image.resized_rgb, result.image.plan, config, result.image.patches_bf16);
        !status) {
        result.status = status;
        return result;
    }
    result.status = ok();
    return result;
}

const char * preprocess_error_name(PreprocessError error) {
    switch (error) {
        case PreprocessError::None: return "none";
        case PreprocessError::InvalidConfig: return "invalid_config";
        case PreprocessError::InvalidDimensions: return "invalid_dimensions";
        case PreprocessError::InputSizeMismatch: return "input_size_mismatch";
        case PreprocessError::InputTooLarge: return "input_too_large";
        case PreprocessError::ResizePlanFailed: return "resize_plan_failed";
        case PreprocessError::OutputTooLarge: return "output_too_large";
        case PreprocessError::TokenBudgetExceeded: return "token_budget_exceeded";
        case PreprocessError::PositionOverflow: return "position_overflow";
    }
    return "unknown";
}

}  // namespace luce::vision
