#include "deepseek4_vision_decode.h"

#include <cstdio>
#include <jpeglib.h>
#include <lodepng.h>

#include <algorithm>
#include <array>
#include <csetjmp>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace dflash::vision {
namespace {

DecodeResult fail(DecodeError code, std::string message) {
    DecodeResult result;
    result.status = {code, std::move(message)};
    return result;
}

DecodeStatus validate_encoded(const EncodedImageView & encoded, const DecodeLimits & limits) {
    if (encoded.data == nullptr || encoded.size == 0) {
        return {DecodeError::EmptyInput, "encoded image is empty"};
    }
    if (limits.max_encoded_bytes == 0 || encoded.size > limits.max_encoded_bytes) {
        return {DecodeError::EncodedTooLarge, "encoded image exceeds the configured byte limit"};
    }
    return {};
}

DecodeStatus validate_decoded(
    std::uint32_t width,
    std::uint32_t height,
    const DecodeLimits & limits,
    std::size_t & output_bytes) {
    const auto status = validate_decoded_dimensions(width, height, limits.decoded);
    if (!status) {
        return {DecodeError::DecodedTooLarge, status.message};
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
    if (pixels > std::numeric_limits<std::size_t>::max() / 3) {
        return {DecodeError::DecodedTooLarge, "decoded RGB byte count overflows size_t"};
    }
    output_bytes = static_cast<std::size_t>(pixels * 3);
    return {};
}

bool checked_add_size(std::size_t left, std::size_t right, std::size_t & result) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_mul_size(std::size_t left, std::size_t right, std::size_t & result) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool png_filtered_size(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t bits_per_pixel,
    std::size_t & result) {
    std::size_t whole_bytes = 0;
    if (!checked_mul_size(width / 8, bits_per_pixel, whole_bytes)) {
        return false;
    }
    const std::size_t partial_bytes =
        ((width & 7U) * static_cast<std::size_t>(bits_per_pixel) + 7) / 8;
    std::size_t row_bytes = 0;
    if (!checked_add_size(whole_bytes, partial_bytes, row_bytes) ||
        !checked_add_size(row_bytes, 1, row_bytes)) {
        return false;
    }
    return checked_mul_size(row_bytes, height, result);
}

bool png_idat_bound(
    std::uint32_t width,
    std::uint32_t height,
    const LodePNGColorMode & color,
    std::uint32_t interlace_method,
    std::size_t & result) {
    const std::uint32_t bits_per_pixel = lodepng_get_bpp(&color);
    if (bits_per_pixel == 0) {
        return false;
    }
    if (interlace_method == 0) {
        return png_filtered_size(width, height, bits_per_pixel, result);
    }
    if (interlace_method != 1) {
        return false;
    }

    constexpr std::array<std::uint32_t, 7> x_start = {0, 4, 0, 2, 0, 1, 0};
    constexpr std::array<std::uint32_t, 7> y_start = {0, 0, 4, 0, 2, 0, 1};
    constexpr std::array<std::uint32_t, 7> x_step = {8, 8, 4, 4, 2, 2, 1};
    constexpr std::array<std::uint32_t, 7> y_step = {8, 8, 8, 4, 4, 2, 2};
    result = 0;
    for (std::size_t pass = 0; pass < x_start.size(); ++pass) {
        const std::uint32_t pass_width = width <= x_start[pass]
            ? 0
            : (width - x_start[pass] + x_step[pass] - 1) / x_step[pass];
        const std::uint32_t pass_height = height <= y_start[pass]
            ? 0
            : (height - y_start[pass] + y_step[pass] - 1) / y_step[pass];
        if (pass_width == 0 || pass_height == 0) {
            continue;
        }
        std::size_t pass_size = 0;
        if (!png_filtered_size(pass_width, pass_height, bits_per_pixel, pass_size) ||
            !checked_add_size(result, pass_size, result)) {
            return false;
        }
    }
    return result != 0;
}

struct JpegErrorManager {
    jpeg_error_mgr base;
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX] = {};
    int warnings = 0;
};

extern "C" void jpeg_error_exit(j_common_ptr common) {
    auto * error = reinterpret_cast<JpegErrorManager *>(common->err);
    error->base.format_message(common, error->message);
    std::longjmp(error->jump, 1);
}

extern "C" void jpeg_emit_message(j_common_ptr common, int level) {
    auto * error = reinterpret_cast<JpegErrorManager *>(common->err);
    if (level < 0) {
        ++error->warnings;
        error->base.format_message(common, error->message);
    }
}

DecodeResult decode_jpeg(const EncodedImageView & encoded, const DecodeLimits & limits) {
    DecodeResult result;
    jpeg_decompress_struct decoder{};
    JpegErrorManager error{};
    volatile bool decoder_created = false;
    std::uint8_t * volatile output = nullptr;
    std::size_t output_bytes = 0;

    decoder.err = jpeg_std_error(&error.base);
    error.base.error_exit = jpeg_error_exit;
    error.base.emit_message = jpeg_emit_message;
    if (setjmp(error.jump) != 0) {
        if (decoder_created) {
            jpeg_destroy_decompress(&decoder);
        }
        std::free(output);
        return fail(DecodeError::MalformedImage,
                    error.message[0] == '\0' ? "malformed JPEG" : error.message);
    }

    jpeg_create_decompress(&decoder);
    decoder_created = true;
    jpeg_mem_src(&decoder, encoded.data, static_cast<unsigned long>(encoded.size));
    if (jpeg_read_header(&decoder, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::MalformedImage, "JPEG header is incomplete");
    }
    if (decoder.jpeg_color_space == JCS_CMYK || decoder.jpeg_color_space == JCS_YCCK) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::UnsupportedFormat,
                    "CMYK and YCCK JPEG images are not supported");
    }
    decoder.out_color_space = JCS_RGB;
    jpeg_calc_output_dimensions(&decoder);
    if (decoder.output_width > std::numeric_limits<std::uint32_t>::max() ||
        decoder.output_height > std::numeric_limits<std::uint32_t>::max()) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::DecodedTooLarge, "JPEG dimensions are out of range");
    }
    if (const auto status = validate_decoded(
            static_cast<std::uint32_t>(decoder.output_width),
            static_cast<std::uint32_t>(decoder.output_height),
            limits,
            output_bytes);
        !status) {
        jpeg_destroy_decompress(&decoder);
        result.status = status;
        return result;
    }
    if (!jpeg_start_decompress(&decoder) || decoder.output_components != 3) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::MalformedImage, "JPEG cannot be converted to RGB");
    }

    output = static_cast<std::uint8_t *>(std::malloc(output_bytes));
    if (output == nullptr) {
        jpeg_destroy_decompress(&decoder);
        return fail(DecodeError::AllocationFailed, "cannot allocate decoded JPEG RGB buffer");
    }
    const std::size_t stride = static_cast<std::size_t>(decoder.output_width) * 3;
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = output +
            static_cast<std::size_t>(decoder.output_scanline) * stride;
        if (jpeg_read_scanlines(&decoder, &row, 1) != 1) {
            jpeg_destroy_decompress(&decoder);
            std::free(output);
            return fail(DecodeError::MalformedImage, "JPEG scanline data is incomplete");
        }
    }
    if (!jpeg_finish_decompress(&decoder) || error.warnings != 0) {
        jpeg_destroy_decompress(&decoder);
        std::free(output);
        return fail(DecodeError::MalformedImage,
                    error.message[0] == '\0' ? "JPEG has decoding warnings" : error.message);
    }

    result.image.width = static_cast<std::uint32_t>(decoder.output_width);
    result.image.height = static_cast<std::uint32_t>(decoder.output_height);
    jpeg_destroy_decompress(&decoder);
    decoder_created = false;
    try {
        const auto * begin = output;
        result.image.pixels.assign(begin, begin + output_bytes);
    } catch (...) {
        std::free(output);
        return fail(DecodeError::AllocationFailed, "cannot own decoded JPEG RGB buffer");
    }
    std::free(output);
    result.status = {};
    return result;
}

DecodeResult decode_png(const EncodedImageView & encoded, const DecodeLimits & limits) {
    LodePNGState state;
    lodepng_state_init(&state);
    state.decoder.ignore_crc = 0;
    state.decoder.ignore_critical = 0;
    state.decoder.ignore_end = 0;
    state.decoder.zlibsettings.ignore_adler32 = 0;
    state.decoder.zlibsettings.ignore_nlen = 0;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
    state.decoder.read_text_chunks = 0;
    state.decoder.remember_unknown_chunks = 0;
    state.decoder.max_icc_size = 16ULL * 1024ULL * 1024ULL;
#endif
    unsigned width = 0;
    unsigned height = 0;
    const unsigned inspect_error =
        lodepng_inspect(&width, &height, &state, encoded.data, encoded.size);
    if (inspect_error != 0) {
        lodepng_state_cleanup(&state);
        return fail(DecodeError::MalformedImage,
                    std::string("malformed PNG: ") + lodepng_error_text(inspect_error));
    }

    std::size_t output_bytes = 0;
    if (const auto status = validate_decoded(width, height, limits, output_bytes); !status) {
        lodepng_state_cleanup(&state);
        DecodeResult result;
        result.status = status;
        return result;
    }
    std::size_t idat_bound = 0;
    if (!png_idat_bound(
            width,
            height,
            state.info_png.color,
            state.info_png.interlace_method,
            idat_bound)) {
        lodepng_state_cleanup(&state);
        return fail(DecodeError::MalformedImage, "PNG filtered scanline size is invalid");
    }
    state.decoder.zlibsettings.max_output_size = idat_bound;

    const bool grey16 =
        state.info_png.color.colortype == LCT_GREY && state.info_png.color.bitdepth == 16;
    state.info_raw.colortype = grey16 ? LCT_GREY : LCT_RGB;
    state.info_raw.bitdepth = grey16 ? 16 : 8;
    state.decoder.color_convert = grey16 ? 0 : 1;
    unsigned char * output = nullptr;
    unsigned decoded_width = 0;
    unsigned decoded_height = 0;
    const unsigned decode_error = lodepng_decode(
        &output, &decoded_width, &decoded_height, &state, encoded.data, encoded.size);
    lodepng_state_cleanup(&state);
    if (decode_error != 0) {
        std::free(output);
        if (decode_error == 109) {
            return fail(DecodeError::MalformedImage,
                        "PNG IDAT exceeds decoded geometry bound");
        }
        if (decode_error == 83) {
            return fail(DecodeError::AllocationFailed, "cannot allocate PNG decode buffer");
        }
        return fail(DecodeError::MalformedImage,
                    std::string("malformed PNG: ") + lodepng_error_text(decode_error));
    }
    if (decoded_width != width || decoded_height != height) {
        std::free(output);
        return fail(DecodeError::MalformedImage, "PNG dimensions changed during decode");
    }

    DecodeResult result;
    result.image.width = width;
    result.image.height = height;
    try {
        if (grey16) {
            const std::size_t sample_count = static_cast<std::size_t>(width) * height;
            result.image.pixels.resize(output_bytes);
            for (std::size_t sample = 0; sample < sample_count; ++sample) {
                const std::uint16_t value =
                    static_cast<std::uint16_t>(output[sample * 2]) << 8 |
                    output[sample * 2 + 1];
                const auto channel = static_cast<std::uint8_t>(std::min<std::uint16_t>(value, 255));
                result.image.pixels[sample * 3] = channel;
                result.image.pixels[sample * 3 + 1] = channel;
                result.image.pixels[sample * 3 + 2] = channel;
            }
        } else {
            result.image.pixels.assign(output, output + output_bytes);
        }
    } catch (...) {
        std::free(output);
        return fail(DecodeError::AllocationFailed, "cannot own decoded PNG RGB buffer");
    }
    std::free(output);
    result.status = {};
    return result;
}

}  // namespace

DecodeResult decode_image(const EncodedImageView & encoded, const DecodeLimits & limits) {
    if (const auto status = validate_encoded(encoded, limits); !status) {
        DecodeResult result;
        result.status = status;
        return result;
    }
    constexpr std::uint8_t png_signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (encoded.size >= sizeof(png_signature) &&
        std::memcmp(encoded.data, png_signature, sizeof(png_signature)) == 0) {
        return decode_png(encoded, limits);
    }
    if (encoded.size >= 3 && encoded.data[0] == 0xFF && encoded.data[1] == 0xD8 &&
        encoded.data[2] == 0xFF) {
        return decode_jpeg(encoded, limits);
    }
    return fail(DecodeError::UnsupportedFormat, "encoded image is neither JPEG nor PNG");
}

const char * decode_error_name(DecodeError error) {
    switch (error) {
        case DecodeError::None: return "none";
        case DecodeError::EmptyInput: return "empty_input";
        case DecodeError::EncodedTooLarge: return "encoded_too_large";
        case DecodeError::UnsupportedFormat: return "unsupported_format";
        case DecodeError::MalformedImage: return "malformed_image";
        case DecodeError::DecodedTooLarge: return "decoded_too_large";
        case DecodeError::AllocationFailed: return "allocation_failed";
    }
    return "unknown";
}

}  // namespace dflash::vision
