#include "deepseek4_vision_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace dflash::vision {
namespace {

constexpr int kPrecisionBits = 22;

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

struct Coefficients {
    int kernel_size = 0;
    std::vector<int> bounds;
    std::vector<std::int32_t> weights;
};

// Pillow 12.3.0 Resample.c reference used for byte parity:
// https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
double bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

PreprocessStatus precompute_coefficients(int input_size, int output_size, Coefficients & out) {
    if (input_size <= 0 || output_size <= 0) {
        return fail(PreprocessError::ResizePlanFailed, "resample dimensions must be positive");
    }
    const float input_begin = 0.0F;
    const float input_end = static_cast<float>(input_size);
    double filter_scale =
        (static_cast<double>(input_end) - static_cast<double>(input_begin)) / output_size;
    const double scale = filter_scale;
    if (filter_scale < 1.0) {
        filter_scale = 1.0;
    }
    const double support = 2.0 * filter_scale;
    const int kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    std::uint64_t coefficient_count = 0;
    if (!checked_mul(static_cast<std::uint64_t>(output_size),
                     static_cast<std::uint64_t>(kernel_size), coefficient_count) ||
        coefficient_count > std::numeric_limits<std::size_t>::max()) {
        return fail(PreprocessError::OutputTooLarge, "resample coefficient count overflow");
    }

    std::vector<double> floating(static_cast<std::size_t>(coefficient_count), 0.0);
    out.bounds.resize(static_cast<std::size_t>(output_size) * 2);
    out.weights.resize(static_cast<std::size_t>(coefficient_count));
    const double inverse_filter_scale = 1.0 / filter_scale;
    for (int output = 0; output < output_size; ++output) {
        const double center = input_begin + (output + 0.5) * scale;
        int first = static_cast<int>(center - support + 0.5);
        if (first < 0) {
            first = 0;
        }
        int count = static_cast<int>(center + support + 0.5);
        if (count > input_size) {
            count = input_size;
        }
        count -= first;
        double sum = 0.0;
        const std::size_t base = static_cast<std::size_t>(output) * kernel_size;
        for (int index = 0; index < count; ++index) {
            const double weight = bicubic(
                (index + first - center + 0.5) * inverse_filter_scale);
            floating[base + index] = weight;
            sum += weight;
        }
        if (sum != 0.0) {
            for (int index = 0; index < count; ++index) {
                floating[base + index] /= sum;
            }
        }
        out.bounds[static_cast<std::size_t>(output) * 2] = first;
        out.bounds[static_cast<std::size_t>(output) * 2 + 1] = count;
    }

    constexpr double scale_to_fixed = static_cast<double>(1U << kPrecisionBits);
    for (std::size_t index = 0; index < floating.size(); ++index) {
        const double value = floating[index] * scale_to_fixed;
        out.weights[index] = static_cast<std::int32_t>(
            value < 0.0 ? value - 0.5 : value + 0.5);
    }
    out.kernel_size = kernel_size;
    return ok();
}

std::uint8_t clip_fixed(std::int32_t value) {
    const std::int32_t rounded = value >> kPrecisionBits;
    return static_cast<std::uint8_t>(std::clamp<std::int32_t>(rounded, 0, 255));
}

PreprocessStatus resize_horizontal(
    const std::vector<std::uint8_t> & input,
    int input_width,
    int input_height,
    int output_width,
    std::vector<std::uint8_t> & output) {
    Coefficients coeffs;
    if (const auto status = precompute_coefficients(input_width, output_width, coeffs); !status) {
        return status;
    }
    output.resize(static_cast<std::size_t>(output_width) * input_height * 3);
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            const int first = coeffs.bounds[static_cast<std::size_t>(x) * 2];
            const int count = coeffs.bounds[static_cast<std::size_t>(x) * 2 + 1];
            const std::int32_t * weights =
                coeffs.weights.data() + static_cast<std::size_t>(x) * coeffs.kernel_size;
            for (int channel = 0; channel < 3; ++channel) {
                std::int32_t sum = 1 << (kPrecisionBits - 1);
                for (int index = 0; index < count; ++index) {
                    const std::size_t source =
                        (static_cast<std::size_t>(y) * input_width + first + index) * 3 + channel;
                    sum += static_cast<std::int32_t>(input[source]) * weights[index];
                }
                const std::size_t destination =
                    (static_cast<std::size_t>(y) * output_width + x) * 3 + channel;
                output[destination] = clip_fixed(sum);
            }
        }
    }
    return ok();
}

PreprocessStatus resize_vertical(
    const std::vector<std::uint8_t> & input,
    int input_width,
    int input_height,
    int output_height,
    std::vector<std::uint8_t> & output) {
    Coefficients coeffs;
    if (const auto status = precompute_coefficients(input_height, output_height, coeffs); !status) {
        return status;
    }
    output.resize(static_cast<std::size_t>(input_width) * output_height * 3);
    for (int y = 0; y < output_height; ++y) {
        const int first = coeffs.bounds[static_cast<std::size_t>(y) * 2];
        const int count = coeffs.bounds[static_cast<std::size_t>(y) * 2 + 1];
        const std::int32_t * weights =
            coeffs.weights.data() + static_cast<std::size_t>(y) * coeffs.kernel_size;
        for (int x = 0; x < input_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                std::int32_t sum = 1 << (kPrecisionBits - 1);
                for (int index = 0; index < count; ++index) {
                    const std::size_t source =
                        (static_cast<std::size_t>(first + index) * input_width + x) * 3 + channel;
                    sum += static_cast<std::int32_t>(input[source]) * weights[index];
                }
                const std::size_t destination =
                    (static_cast<std::size_t>(y) * input_width + x) * 3 + channel;
                output[destination] = clip_fixed(sum);
            }
        }
    }
    return ok();
}

PreprocessStatus pillow_resize(
    const std::vector<std::uint8_t> & input,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    std::vector<std::uint8_t> & output) {
    if (input_width == output_width && input_height == output_height) {
        output = input;
        return ok();
    }

    std::vector<std::uint8_t> intermediate;
    const bool vertical_first =
        static_cast<std::uint64_t>(input_height) >
            static_cast<std::uint64_t>(input_width) * 100 &&
        output_height < input_height;
    if (vertical_first) {
        if (const auto status = resize_vertical(
                input, input_width, input_height, output_height, intermediate);
            !status) {
            return status;
        }
        if (output_width == input_width) {
            output = std::move(intermediate);
            return ok();
        }
        return resize_horizontal(
            intermediate, input_width, output_height, output_width, output);
    }

    const std::vector<std::uint8_t> * vertical_input = &input;
    int vertical_width = input_width;
    if (output_width != input_width) {
        if (const auto status = resize_horizontal(
                input, input_width, input_height, output_width, intermediate);
            !status) {
            return status;
        }
        vertical_input = &intermediate;
        vertical_width = output_width;
    }
    if (output_height != input_height) {
        return resize_vertical(
            *vertical_input, vertical_width, input_height, output_height, output);
    }
    output = *vertical_input;
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
    const PreprocessConfig expected;
    if (config.patch_size != expected.patch_size ||
        config.downsample_ratio != expected.downsample_ratio ||
        config.max_tokens != expected.max_tokens ||
        config.min_pixels != expected.min_pixels ||
        config.max_aspect_ratio != expected.max_aspect_ratio ||
        config.compress_pad_to != expected.compress_pad_to ||
        config.vocab_size != expected.vocab_size ||
        config.normalization_mean != expected.normalization_mean ||
        config.normalization_std != expected.normalization_std) {
        return fail(PreprocessError::InvalidConfig,
                    "preprocessing config does not match the fixed DeepSeek-V4 vision recipe");
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

    const bool direct_resize =
        static_cast<std::uint64_t>(width) >=
        static_cast<std::uint64_t>(height) * config.max_aspect_ratio;
    std::uint64_t planned_width = width;
    std::uint64_t planned_height = height;
    const std::uint64_t max_width =
        static_cast<std::uint64_t>(height) * config.max_aspect_ratio;
    if (planned_width > max_width) {
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
    if (const auto status = grid_tokens(best_height, best_width, config, grid); !status) {
        return status;
    }

    const std::uint32_t reserved = config.compress_pad_to - 1;
    if (config.max_tokens <= reserved + 2) {
        return fail(PreprocessError::InvalidConfig, "token budget cannot hold image sentinels");
    }
    const std::uint32_t image_budget = config.max_tokens - reserved;
    std::uint32_t solver_budget = image_budget;
    std::uint32_t attempts = 0;
    while (grid.tokens > image_budget) {
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

}  // namespace dflash::vision
