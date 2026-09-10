#include "deepseek4_vision_preprocess.h"
#ifdef DS4V_PREPROCESS_WITH_CODECS
#include "deepseek4_vision_decode.h"
#include <cstdio>
#include <jpeglib.h>
#include <lodepng.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;
using dflash::vision::DecodedRgbView;
using dflash::vision::ImageLayout;
using dflash::vision::ImageTokenType;
using dflash::vision::PreprocessConfig;
using dflash::vision::PreprocessError;
using dflash::vision::PreprocessLimits;
using dflash::vision::PreprocessResult;
using dflash::vision::ResizePlan;
#ifdef DS4V_PREPROCESS_WITH_CODECS
using dflash::vision::DecodeError;
using dflash::vision::DecodeLimits;
using dflash::vision::EncodedImageView;
#endif

namespace {

struct FixtureCase {
    std::string label;
    std::uint32_t input_width = 0;
    std::uint32_t input_height = 0;
    std::uint32_t resized_width = 0;
    std::uint32_t resized_height = 0;
    std::uint32_t vit_rows = 0;
    std::uint32_t vit_cols = 0;
    std::uint32_t aligner_rows = 0;
    std::uint32_t aligner_cols = 0;
};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<std::string> split_tabs(const std::string & line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) {
            return fields;
        }
        begin = end + 1;
    }
}

std::uint32_t parse_u32(const std::string & value, const std::string & field) {
    std::size_t used = 0;
    const unsigned long parsed = std::stoul(value, &used);
    if (used != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid " + field + ": " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

std::vector<FixtureCase> read_manifest(const fs::path & root) {
    std::ifstream input(root / "manifest.tsv");
    require(input.good(), "cannot open fixture manifest: " + (root / "manifest.tsv").string());
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "fixture manifest is empty");
    require(line ==
                "label\tinput_width\tinput_height\tresized_width\tresized_height\tvit_rows\tvit_cols\taligner_rows\taligner_cols",
            "fixture manifest header mismatch");
    std::vector<FixtureCase> cases;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_tabs(line);
        require(fields.size() == 9, "fixture manifest row must have 9 fields");
        require(!fields[0].empty() &&
                    std::all_of(fields[0].begin(), fields[0].end(), [](unsigned char value) {
                        return (value >= 'a' && value <= 'z') || value == '-';
                    }),
                "fixture label contains unsupported characters");
        FixtureCase item;
        item.label = fields[0];
        item.input_width = parse_u32(fields[1], "input_width");
        item.input_height = parse_u32(fields[2], "input_height");
        item.resized_width = parse_u32(fields[3], "resized_width");
        item.resized_height = parse_u32(fields[4], "resized_height");
        item.vit_rows = parse_u32(fields[5], "vit_rows");
        item.vit_cols = parse_u32(fields[6], "vit_cols");
        item.aligner_rows = parse_u32(fields[7], "aligner_rows");
        item.aligner_cols = parse_u32(fields[8], "aligner_cols");
        cases.push_back(std::move(item));
    }
    require(!cases.empty(), "fixture manifest has no cases");
    return cases;
}

template <typename T>
std::vector<T> read_binary(const fs::path & path) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "cannot open fixture: " + path.string());
    const std::streampos end = input.tellg();
    require(end >= 0, "cannot determine fixture size: " + path.string());
    const auto bytes = static_cast<std::uint64_t>(end);
    require(bytes % sizeof(T) == 0, "fixture byte size is invalid: " + path.string());
    std::vector<T> result(static_cast<std::size_t>(bytes / sizeof(T)));
    input.seekg(0);
    if (!result.empty()) {
        input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes));
        require(input.good(), "cannot read fixture: " + path.string());
    }
    return result;
}

template <typename T>
void require_equal(
    const std::vector<T> & actual,
    const std::vector<T> & expected,
    const std::string & label) {
    if (actual.size() != expected.size()) {
        std::ostringstream message;
        message << label << " size mismatch: actual=" << actual.size()
                << " expected=" << expected.size();
        throw std::runtime_error(message.str());
    }
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end()) {
        const std::size_t offset = static_cast<std::size_t>(mismatch.first - actual.begin());
        std::ostringstream message;
        message << label << " mismatch at element " << offset
                << ": actual=" << static_cast<std::int64_t>(*mismatch.first)
                << " expected=" << static_cast<std::int64_t>(*mismatch.second);
        throw std::runtime_error(message.str());
    }
}

std::vector<std::int64_t> layout_types(const ImageLayout & layout) {
    std::vector<std::int64_t> result;
    result.reserve(layout.types.size());
    for (const auto type : layout.types) {
        result.push_back(static_cast<std::int64_t>(type));
    }
    return result;
}

void require_plan(const ResizePlan & actual, const FixtureCase & expected) {
    std::ostringstream details;
    details << "actual resized=" << actual.resized_width << 'x' << actual.resized_height
            << " vit=" << actual.vit_rows << 'x' << actual.vit_cols
            << " aligner=" << actual.aligner_rows << 'x' << actual.aligner_cols;
    require(actual.resized_width == expected.resized_width &&
                actual.resized_height == expected.resized_height &&
                actual.vit_rows == expected.vit_rows && actual.vit_cols == expected.vit_cols &&
                actual.aligner_rows == expected.aligner_rows &&
                actual.aligner_cols == expected.aligner_cols,
            expected.label + " plan mismatch: " + details.str());
}

void require_same_result(const PreprocessResult & first, const PreprocessResult & second,
                         const std::string & label) {
    require(static_cast<bool>(first) && static_cast<bool>(second),
            label + " deterministic run failed");
    require(first.image.plan.resized_width == second.image.plan.resized_width &&
                first.image.plan.resized_height == second.image.plan.resized_height &&
                first.image.plan.vit_rows == second.image.plan.vit_rows &&
                first.image.plan.vit_cols == second.image.plan.vit_cols &&
                first.image.plan.aligner_rows == second.image.plan.aligner_rows &&
                first.image.plan.aligner_cols == second.image.plan.aligner_cols &&
                first.image.plan.direct_resize == second.image.plan.direct_resize,
            label + " plan is not deterministic");
    require_equal(first.image.resized_rgb, second.image.resized_rgb,
                  label + " deterministic resized RGB");
    require_equal(first.image.patches_bf16, second.image.patches_bf16,
                  label + " deterministic patches");
    require_equal(layout_types(first.image.layout), layout_types(second.image.layout),
                  label + " deterministic layout types");
    require_equal(first.image.layout.permutation, second.image.layout.permutation,
                  label + " deterministic permutation");
    require(first.image.layout.span.block_begin == second.image.layout.span.block_begin &&
                first.image.layout.span.visible_begin == second.image.layout.span.visible_begin &&
                first.image.layout.span.visible_end == second.image.layout.span.visible_end &&
                first.image.layout.span.block_end == second.image.layout.span.block_end,
            label + " span is not deterministic");
}

void verify_fixture(const fs::path & root, const FixtureCase & item) {
    const fs::path case_dir = root / item.label;
    const auto input = read_binary<std::uint8_t>(case_dir / "input.rgb");
#ifdef DS4V_PREPROCESS_WITH_CODECS
    const auto encoded = read_binary<std::uint8_t>(case_dir / "encoded.bin");
    const auto decoded = dflash::vision::decode_image({encoded.data(), encoded.size()});
    require(static_cast<bool>(decoded),
            item.label + " decode failed: " +
                dflash::vision::decode_error_name(decoded.status.code) + ": " +
                decoded.status.message);
    require(decoded.image.width == item.input_width && decoded.image.height == item.input_height,
            item.label + " decoded dimensions mismatch");
    require_equal(decoded.image.pixels, input, item.label + " decoded RGB");
    const auto decoded_again = dflash::vision::decode_image({encoded.data(), encoded.size()});
    require(static_cast<bool>(decoded_again), item.label + " repeated decode failed");
    require_equal(decoded_again.image.pixels, decoded.image.pixels,
                  item.label + " deterministic decode");

    DecodeLimits one_pixel;
    one_pixel.decoded.max_decoded_pixels = 1;
    const auto bounded = dflash::vision::decode_image(
        {encoded.data(), encoded.size()}, one_pixel);
    require(bounded.status.code == DecodeError::DecodedTooLarge && bounded.image.pixels.empty(),
            item.label + " decoded pixel cap did not fail before output allocation");
#endif
    const DecodedRgbView view{item.input_width, item.input_height, input.data(), input.size()};
    const PreprocessResult result = dflash::vision::preprocess_rgb(view, 0);
    require(static_cast<bool>(result),
            item.label + " preprocess failed: " +
                dflash::vision::preprocess_error_name(result.status.code) + ": " +
                result.status.message);
    require_plan(result.image.plan, item);
    require_equal(result.image.resized_rgb,
                  read_binary<std::uint8_t>(case_dir / "resized.rgb"),
                  item.label + " resized RGB");
    require_equal(result.image.patches_bf16,
                  read_binary<std::uint16_t>(case_dir / "patches.bf16"),
                  item.label + " BF16 patches");

    constexpr std::array<std::uint64_t, 5> starts = {0, 1, 2, 3, 127};
    for (const std::uint64_t start : starts) {
        ImageLayout layout;
        const auto status = dflash::vision::build_image_layout(
            item.aligner_rows, item.aligner_cols, start, layout);
        require(static_cast<bool>(status),
                item.label + " layout failed at start " + std::to_string(start) + ": " +
                    status.message);
        const std::string suffix = std::to_string(start) + ".i64";
        require_equal(layout_types(layout),
                      read_binary<std::int64_t>(case_dir / ("types-" + suffix)),
                      item.label + " types start=" + std::to_string(start));
        require_equal(layout.permutation,
                      read_binary<std::int64_t>(case_dir / ("permutation-" + suffix)),
                      item.label + " permutation start=" + std::to_string(start));
        const std::uint64_t leading = 3 - start % 4;
        const std::uint64_t end = start + layout.types.size();
        require(layout.span.block_begin == start && layout.span.visible_begin == start + leading &&
                    layout.span.visible_end == end && layout.span.block_end == end,
                item.label + " span mismatch at start " + std::to_string(start));
    }

    const PreprocessResult repeated = dflash::vision::preprocess_rgb(view, 0);
    require_same_result(result, repeated, item.label);
    std::cout << item.label << " PASS resized=" << item.resized_width << 'x'
              << item.resized_height << " vit=" << item.vit_rows << 'x' << item.vit_cols
              << " aligner=" << item.aligner_rows << 'x' << item.aligner_cols
              << " patch_words=" << result.image.patches_bf16.size() << '\n';
}

void expect_error(PreprocessError expected, PreprocessError actual, const std::string & label) {
    require(actual == expected,
            label + " returned " + dflash::vision::preprocess_error_name(actual) +
                ", expected " + dflash::vision::preprocess_error_name(expected));
}

#ifdef DS4V_PREPROCESS_WITH_CODECS
void append_u32(std::vector<std::uint8_t> & output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>(value >> 24));
    output.push_back(static_cast<std::uint8_t>(value >> 16));
    output.push_back(static_cast<std::uint8_t>(value >> 8));
    output.push_back(static_cast<std::uint8_t>(value));
}

void append_png_chunk(
    std::vector<std::uint8_t> & output,
    const std::array<char, 4> & type,
    const std::vector<std::uint8_t> & data) {
    append_u32(output, static_cast<std::uint32_t>(data.size()));
    const std::size_t crc_begin = output.size();
    output.insert(output.end(), type.begin(), type.end());
    output.insert(output.end(), data.begin(), data.end());
    append_u32(output, lodepng_crc32(output.data() + crc_begin, 4 + data.size()));
}

std::vector<std::uint8_t> excessive_idat_png() {
    std::vector<std::uint8_t> inflated(4096, 0);
    unsigned char * compressed = nullptr;
    std::size_t compressed_size = 0;
    const unsigned error = lodepng_zlib_compress(
        &compressed,
        &compressed_size,
        inflated.data(),
        inflated.size(),
        &lodepng_default_compress_settings);
    require(error == 0, "cannot create excessive-IDAT regression PNG");

    std::vector<std::uint8_t> result = {137, 80, 78, 71, 13, 10, 26, 10};
    const std::vector<std::uint8_t> ihdr = {
        0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0,
    };
    append_png_chunk(result, {'I', 'H', 'D', 'R'}, ihdr);
    const std::vector<std::uint8_t> compressed_bytes(
        compressed, compressed + compressed_size);
    append_png_chunk(result, {'I', 'D', 'A', 'T'}, compressed_bytes);
    append_png_chunk(result, {'I', 'E', 'N', 'D'}, {});
    std::free(compressed);
    return result;
}

std::vector<std::uint8_t> grey16_png() {
    constexpr std::array<std::uint16_t, 8> values = {
        0, 1, 254, 255, 256, 257, 1024, 65535,
    };
    std::vector<std::uint8_t> pixels;
    pixels.reserve(values.size() * 2 * 2);
    for (int row = 0; row < 2; ++row) {
        for (const std::uint16_t value : values) {
            pixels.push_back(static_cast<std::uint8_t>(value >> 8));
            pixels.push_back(static_cast<std::uint8_t>(value));
        }
    }
    unsigned char * encoded = nullptr;
    std::size_t encoded_size = 0;
    const unsigned error = lodepng_encode_memory(
        &encoded, &encoded_size, pixels.data(), 8, 2, LCT_GREY, 16);
    require(error == 0, "cannot create GREY16 regression PNG");
    std::vector<std::uint8_t> result(encoded, encoded + encoded_size);
    std::free(encoded);
    return result;
}

std::vector<std::uint8_t> cmyk_jpeg() {
    jpeg_compress_struct encoder{};
    jpeg_error_mgr error{};
    encoder.err = jpeg_std_error(&error);
    jpeg_create_compress(&encoder);
    unsigned char * encoded = nullptr;
    unsigned long encoded_size = 0;
    jpeg_mem_dest(&encoder, &encoded, &encoded_size);
    encoder.image_width = 2;
    encoder.image_height = 1;
    encoder.input_components = 4;
    encoder.in_color_space = JCS_CMYK;
    jpeg_set_defaults(&encoder);
    jpeg_start_compress(&encoder, TRUE);
    std::array<std::uint8_t, 8> pixels = {0, 64, 128, 16, 255, 192, 128, 32};
    JSAMPROW row = pixels.data();
    require(jpeg_write_scanlines(&encoder, &row, 1) == 1,
            "cannot create CMYK regression JPEG");
    jpeg_finish_compress(&encoder);
    std::vector<std::uint8_t> result(encoded, encoded + encoded_size);
    jpeg_destroy_compress(&encoder);
    std::free(encoded);
    return result;
}

void decoder_regression_tests() {
    std::vector<std::string> failures;
    const auto excessive = excessive_idat_png();
    const auto excessive_result =
        dflash::vision::decode_image({excessive.data(), excessive.size()});
    if (excessive_result.status.code != DecodeError::MalformedImage ||
        excessive_result.status.message.find("IDAT exceeds decoded geometry bound") ==
            std::string::npos) {
        failures.emplace_back("excess IDAT did not report the bounded-inflate outcome");
    }

    const auto grey = grey16_png();
    const auto grey_result = dflash::vision::decode_image({grey.data(), grey.size()});
    constexpr std::array<std::uint8_t, 8> expected_values = {
        0, 1, 254, 255, 255, 255, 255, 255,
    };
    std::vector<std::uint8_t> expected_rgb;
    expected_rgb.reserve(expected_values.size() * 2 * 3);
    for (int row = 0; row < 2; ++row) {
        for (const std::uint8_t value : expected_values) {
            expected_rgb.insert(expected_rgb.end(), 3, value);
        }
    }
    if (!grey_result || grey_result.image.pixels != expected_rgb) {
        failures.emplace_back("GREY16 did not match Pillow I;16 to RGB clamping");
    }

    const auto cmyk = cmyk_jpeg();
    const auto cmyk_result = dflash::vision::decode_image({cmyk.data(), cmyk.size()});
    if (cmyk_result.status.code != DecodeError::UnsupportedFormat) {
        failures.emplace_back("CMYK JPEG was not classified as unsupported_format");
    }

    if (!failures.empty()) {
        std::ostringstream message;
        message << "decoder regression failures:";
        for (const auto & failure : failures) {
            message << "\n- " << failure;
        }
        throw std::runtime_error(message.str());
    }
}
#endif

void self_test() {
    PreprocessConfig bad_config;
    bad_config.patch_size = 16;
    expect_error(PreprocessError::InvalidConfig,
                 dflash::vision::validate_config(bad_config).code,
                 "changed fixed config");

    expect_error(PreprocessError::InvalidDimensions,
                 dflash::vision::validate_decoded_dimensions(0, 1).code,
                 "zero width");
    expect_error(PreprocessError::InputTooLarge,
                 dflash::vision::validate_decoded_dimensions(65'536, 1).code,
                 "axis cap");
    expect_error(PreprocessError::InputTooLarge,
                 dflash::vision::validate_decoded_dimensions(8192, 8193).code,
                 "pixel cap");

    const std::array<std::uint8_t, 3> pixel = {0, 127, 255};
    DecodedRgbView wrong_size{1, 1, pixel.data(), 2};
    expect_error(PreprocessError::InputSizeMismatch,
                 dflash::vision::preprocess_rgb(wrong_size, 0).status.code,
                 "wrong RGB byte count");

    PreprocessLimits tiny_output_limit;
    tiny_output_limit.max_output_pixels = 1;
    ResizePlan plan;
    expect_error(PreprocessError::OutputTooLarge,
                 dflash::vision::plan_image(1, 1, plan, {}, tiny_output_limit).code,
                 "output cap");

    ImageLayout layout;
    expect_error(PreprocessError::TokenBudgetExceeded,
                 dflash::vision::build_image_layout(100, 100, 0, layout).code,
                 "layout budget");
    expect_error(PreprocessError::PositionOverflow,
                 dflash::vision::build_image_layout(
                     2, 3, std::numeric_limits<std::uint64_t>::max() - 5, layout).code,
                 "absolute span overflow");

    const auto layout_status = dflash::vision::build_image_layout(2, 3, 0, layout);
    require(static_cast<bool>(layout_status), "known layout failed");
    const std::vector<std::int64_t> expected_types = {
        1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 4,
    };
    const std::vector<std::int64_t> expected_permutation = {0, 3, 1, 4, 2, 5};
    require_equal(layout_types(layout), expected_types, "known layout types");
    require_equal(layout.permutation, expected_permutation, "known layout permutation");
    require(layout.span.block_begin == 0 && layout.span.visible_begin == 3 &&
                layout.span.visible_end == 13 && layout.span.block_end == 13,
            "known layout span mismatch");
#ifdef DS4V_PREPROCESS_WITH_CODECS
    const std::array<std::uint8_t, 6> unsupported = {'G', 'I', 'F', '8', '9', 'a'};
    require(dflash::vision::decode_image({nullptr, 0}).status.code == DecodeError::EmptyInput,
            "empty encoded image was accepted");
    require(dflash::vision::decode_image({unsupported.data(), unsupported.size()}).status.code ==
                DecodeError::UnsupportedFormat,
            "unsupported encoded format was accepted");
    const std::array<std::uint8_t, 4> truncated_jpeg = {0xFF, 0xD8, 0xFF, 0xD9};
    require(dflash::vision::decode_image(
                {truncated_jpeg.data(), truncated_jpeg.size()}).status.code ==
                DecodeError::MalformedImage,
            "truncated JPEG was accepted");
    const std::array<std::uint8_t, 8> truncated_png = {137, 80, 78, 71, 13, 10, 26, 10};
    require(dflash::vision::decode_image(
                {truncated_png.data(), truncated_png.size()}).status.code ==
                DecodeError::MalformedImage,
            "truncated PNG was accepted");
    const std::uint8_t byte = 0;
    DecodeLimits encoded_limit;
    require(dflash::vision::decode_image(
                {&byte, encoded_limit.max_encoded_bytes + 1}, encoded_limit).status.code ==
                DecodeError::EncodedTooLarge,
            "oversized encoded input was inspected");
    decoder_regression_tests();
#endif
    std::cout << "self-test PASS\n";
}

void usage(const char * program) {
    std::cerr << "Usage: " << program << " --self-test | --fixtures DIRECTORY\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--fixtures") {
            const fs::path root = argv[2];
            for (const auto & item : read_manifest(root)) {
                verify_fixture(root, item);
            }
            return 0;
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
