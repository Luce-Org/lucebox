// JPEG/PNG decoding shared by every vision model.
#include "common/vision/image_decode.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace luce::vision;

static int failures = 0;
static void check(bool condition, const char * message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

// 3x2 PNG: red, green, blue / (10,20,30), (200,100,50), white.
const std::vector<std::uint8_t> PNG_3X2 = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 3,
    0, 0, 0, 2, 8, 2, 0, 0, 0, 18, 22, 241, 77, 0, 0, 0, 24, 73, 68, 65,
    84, 120, 156, 99, 248, 207, 192, 192, 0, 193, 92, 34, 114, 39, 82, 140, 254, 255, 255, 15,
    0, 60, 25, 7, 149, 239, 194, 198, 216, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96,
    130,
};

// 16x8 JPEG filled with (40,120,200).
const std::vector<std::uint8_t> JPEG_16X8 = {
    255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
    255, 219, 0, 67, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2,
    4, 3, 2, 2, 2, 2, 5, 4, 4, 3, 4, 6, 5, 6, 6, 6, 5, 6, 6, 6,
    7, 9, 8, 6, 7, 9, 7, 6, 6, 8, 11, 8, 9, 10, 10, 10, 10, 10, 6, 8,
    11, 12, 11, 10, 12, 9, 10, 10, 10, 255, 219, 0, 67, 1, 2, 2, 2, 2, 2, 2,
    5, 3, 3, 5, 10, 7, 6, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 255, 192,
    0, 17, 8, 0, 8, 0, 16, 3, 1, 34, 0, 2, 17, 1, 3, 17, 1, 255, 196, 0,
    31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 16, 0, 2, 1, 3, 3,
    2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33,
    49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21,
    82, 209, 240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41,
    42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86,
    87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122,
    131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198,
    199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227, 228, 229, 230, 231,
    232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 196, 0, 31, 1, 0, 3,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 17, 0, 2, 1, 2, 4, 4, 3, 4, 7,
    5, 4, 4, 0, 1, 2, 119, 0, 1, 2, 3, 17, 4, 5, 33, 49, 6, 18, 65, 81,
    7, 97, 113, 19, 34, 50, 129, 8, 20, 66, 145, 161, 177, 193, 9, 35, 51, 82, 240, 21,
    98, 114, 209, 10, 22, 36, 52, 225, 37, 241, 23, 24, 25, 26, 38, 39, 40, 41, 42, 53,
    54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89,
    90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132,
    133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166,
    167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200,
    201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234,
    242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17,
    0, 63, 0, 242, 58, 40, 162, 191, 181, 15, 228, 51, 255, 217,
};

// 8x8 greyscale JPEG filled with 128.
const std::vector<std::uint8_t> JPEG_GREY_8X8 = {
    255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
    255, 219, 0, 67, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2,
    4, 3, 2, 2, 2, 2, 5, 4, 4, 3, 4, 6, 5, 6, 6, 6, 5, 6, 6, 6,
    7, 9, 8, 6, 7, 9, 7, 6, 6, 8, 11, 8, 9, 10, 10, 10, 10, 10, 6, 8,
    11, 12, 11, 10, 12, 9, 10, 10, 10, 255, 192, 0, 11, 8, 0, 8, 0, 8, 1, 1,
    17, 0, 255, 196, 0, 31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 16,
    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0,
    4, 17, 5, 18, 33, 49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8,
    35, 66, 177, 193, 21, 82, 209, 240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26,
    37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73,
    74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117,
    118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152,
    153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186,
    194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226,
    227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 218,
    0, 8, 1, 1, 0, 0, 63, 0, 43, 255, 217,
};

// 2x1 16-bit greyscale PNG with samples 0x8000 and 0x0100.
const std::vector<std::uint8_t> PNG_GREY16_2X1 = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 1, 16, 0, 0, 0, 0, 129, 217, 252, 21, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 104, 96, 96, 100, 0, 0, 2, 7, 0, 130, 159, 82, 239, 216, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
};

static DecodeResult decode(const std::vector<std::uint8_t> & bytes, const DecodeLimits & limits = {}) {
    return decode_image({bytes.data(), bytes.size()}, limits);
}

static bool near(int value, int expected) { return std::abs(value - expected) <= 3; }

int main() {
    {
        const auto result = decode(PNG_3X2);
        check(bool(result) && result.image.width == 3 && result.image.height == 2, "PNG dimensions");
        const std::vector<std::uint8_t> expected = {
            255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30, 200, 100, 50, 255, 255, 255};
        check(result.image.pixels == expected, "PNG pixels are exact row-major RGB");
        check(result.image.view().size == expected.size(), "view covers the pixels");
    }
    {
        const auto result = decode(JPEG_16X8);
        check(bool(result) && result.image.width == 16 && result.image.height == 8, "JPEG dimensions");
        bool close = result.image.pixels.size() == 16u * 8u * 3u;
        for (size_t i = 0; close && i < result.image.pixels.size(); i += 3) {
            close = near(result.image.pixels[i], 40) && near(result.image.pixels[i + 1], 120) &&
                    near(result.image.pixels[i + 2], 200);
        }
        check(close, "JPEG pixels match the source colour");
    }
    {
        const auto result = decode(JPEG_GREY_8X8);
        bool grey = bool(result) && result.image.pixels.size() == 8u * 8u * 3u;
        for (size_t i = 0; grey && i < result.image.pixels.size(); ++i) grey = near(result.image.pixels[i], 128);
        check(grey, "greyscale JPEG expands to RGB");
    }
    {
        const auto result = decode(PNG_GREY16_2X1);
        const std::vector<std::uint8_t> expected = {128, 128, 128, 1, 1, 1};
        check(bool(result) && result.image.pixels == expected, "16-bit greyscale keeps its high byte");
    }
    {
        check(decode_image({nullptr, 0}).status.code == DecodeError::EmptyInput, "empty input");
        const std::vector<std::uint8_t> text = {'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'm', 'a', 'g', 'e'};
        check(decode(text).status.code == DecodeError::UnsupportedFormat, "unknown format");
        auto truncated = PNG_3X2;
        truncated.resize(truncated.size() / 2);
        const auto broken = decode(truncated);
        check(!broken && broken.image.pixels.empty(), "truncated PNG fails without partial output");
        auto cut = JPEG_16X8;
        cut.resize(cut.size() / 3);
        check(!decode(cut), "truncated JPEG fails");
    }
    {
        DecodeLimits limits;
        limits.max_encoded_bytes = PNG_3X2.size() - 1;
        check(decode(PNG_3X2, limits).status.code == DecodeError::EncodedTooLarge, "encoded byte limit");
        limits = {};
        limits.max_decoded_pixels = 5;
        check(decode(PNG_3X2, limits).status.code == DecodeError::DecodedTooLarge, "pixel count limit");
        limits = {};
        limits.max_dimension = 2;
        check(decode(PNG_3X2, limits).status.code == DecodeError::DecodedTooLarge, "dimension limit");
        limits = {};
        limits.max_decoded_pixels = 6;
        check(bool(decode(PNG_3X2, limits)), "exact pixel limit is accepted");
    }
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
