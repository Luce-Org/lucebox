#pragma once

#include "deepseek4_vision_preprocess.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::vision {

struct EncodedImageView {
    const std::uint8_t * data = nullptr;
    std::size_t size = 0;
};

struct DecodeLimits {
    std::size_t max_encoded_bytes = 16ULL * 1024ULL * 1024ULL;
    PreprocessLimits decoded;
};

struct DecodedRgb {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixels;

    DecodedRgbView view() const {
        return {width, height, pixels.data(), pixels.size()};
    }
};

enum class DecodeError {
    None = 0,
    EmptyInput,
    EncodedTooLarge,
    UnsupportedFormat,
    MalformedImage,
    DecodedTooLarge,
    AllocationFailed,
};

struct DecodeStatus {
    DecodeError code = DecodeError::None;
    std::string message;

    explicit operator bool() const { return code == DecodeError::None; }
};

struct DecodeResult {
    DecodeStatus status;
    DecodedRgb image;

    explicit operator bool() const { return static_cast<bool>(status); }
};

DecodeResult decode_image(
    const EncodedImageView & encoded,
    const DecodeLimits & limits = {});

const char * decode_error_name(DecodeError error);

}  // namespace dflash::vision
