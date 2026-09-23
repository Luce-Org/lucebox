// DS4V limits for the shared image span helpers.
#pragma once

#include "../common/image_prompt.h"
#include "../common/vision/image_spans.h"

namespace luce::vision {

inline constexpr size_t DS4V_MAX_IMAGES = common::MAX_REQUEST_IMAGES;
inline constexpr uint64_t DS4V_MAX_IMAGE_BLOCK_TOKENS = 384;

inline bool valid_image_spans(ImageSpanView spans, uint64_t prompt_size) {
    return valid_image_spans(spans, prompt_size, DS4V_MAX_IMAGES, DS4V_MAX_IMAGE_BLOCK_TOKENS);
}

} // namespace luce::vision
