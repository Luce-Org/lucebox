#pragma once

#include "common/image_prompt.h"

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace luce::common {


inline constexpr size_t MAX_IMAGE_BYTES = 16 * 1024 * 1024;

struct ImageInputLimits {
    size_t image_bytes = MAX_IMAGE_BYTES;
    size_t request_bytes = 32 * 1024 * 1024;
    size_t image_count = MAX_REQUEST_IMAGES;
};

struct ImageRequestPolicy {
    bool chat_completions = false;
    bool image_capable = false;
    // Text that stands for one image in the rendered prompt; the backend's
    // chat template maps it to the model's image marker. Users may not send it.
    std::string placeholder;
};

bool parse_image_data_url(std::string_view url, EncodedImage & image,
                          std::string & error, size_t max_bytes = MAX_IMAGE_BYTES);
bool extract_chat_images(const nlohmann::json & messages,
                         std::string_view placeholder,
                         nlohmann::json & normalized,
                         std::vector<EncodedImage> & images,
                         std::string & error,
                         const ImageInputLimits & limits = {});
bool prepare_request_images(const nlohmann::json & messages,
                            const ImageRequestPolicy & policy,
                            nlohmann::json & normalized,
                            std::vector<EncodedImage> & images,
                            std::string & error,
                            const ImageInputLimits & limits = {});
void redact_image_urls(nlohmann::json & value);

} // namespace luce::common
