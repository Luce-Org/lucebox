// Vision input parsing — base64 decode and multimodal extraction from API
// messages.  Does not depend on mtmd; safe to compile without DFLASH_HAVE_MMPROJ.

#pragma once

#include "common/vision_types.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace dflash::common {

using json = nlohmann::json;

// Decode a base64 string into raw bytes.  Throws std::runtime_error on
// invalid input.
std::vector<uint8_t> base64_decode(const std::string & encoded);

// Decode a data:image/...;base64,... URL.  Throws on unsupported schemes.
std::vector<uint8_t> decode_image_data_url(const std::string & url);

// Walk an OpenAI-style messages array, decode image_url / input_image parts,
// and build a MultimodalPrompt whose marked_text concatenates text parts with
// mtmd_default_marker() inserted for each image (llama-server-common order).
// When DFLASH_HAVE_MMPROJ is unset, markers are still inserted but callers
// must reject image requests without a loaded mmproj.
MultimodalPrompt extract_multimodal_from_messages(const json & messages);

// Returns true when `messages` contains at least one image part.
bool messages_contain_images(const json & messages);

}  // namespace dflash::common
