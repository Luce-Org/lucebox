// Multimodal prompt types for native mmproj vision integration.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

// Raw image bytes (JPEG/PNG/etc.) decoded from a data: URL or API payload.
struct DecodedImage {
    std::vector<uint8_t> bytes;
};

// A prompt with mtmd media markers embedded in marked_text plus parallel
// image payloads.  marked_text must contain one mtmd_default_marker() per
// image (in order) before tokenize() is called.
struct MultimodalPrompt {
    std::string                marked_text;
    std::vector<DecodedImage>  images;
};

}  // namespace dflash::common
