// PNG encoding of the generated images: RGB8, the Paeth filter on every row, zlib level 1. Large images are deflated
// in parallel strips (each ends on a byte boundary with a sync flush), joined into one zlib stream whose Adler-32 is
// combined from the strips'.

#pragma once

#include <cstdint>
#include <string>

namespace luce::moss_image {

std::string encode_png(const uint8_t * rgb, int width, int height);

} // namespace luce::moss_image
