#pragma once

#include "common/image_prompt.h"

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace dflash::common {

inline constexpr char DS4_IMAGE_PLACEHOLDER[] = "<｜deepseek_image｜>";

struct ImageInputLimits {
    size_t image_bytes = 16 * 1024 * 1024;
    size_t request_bytes = 32 * 1024 * 1024;
    size_t image_count = 4;
};

struct ImageRequestPolicy {
    bool chat_completions = false;
    bool image_capable = false;
    bool reserve_placeholder = false;
};

bool parse_image_data_url(std::string_view url, EncodedImage & image,
                          std::string & error, size_t max_bytes = 16 * 1024 * 1024);
bool extract_chat_images(const nlohmann::json & messages,
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

} // namespace dflash::common
