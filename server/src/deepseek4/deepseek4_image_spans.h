// DS4V limits for the shared image span helpers.
#pragma once

#include "../common/image_prompt.h"
#include "../common/vision/image_spans.h"

namespace luce::vision {

inline constexpr size_t DS4V_MAX_IMAGES = common::MAX_REQUEST_IMAGES;
// Most tokens one image block may take: DeepSeek-V4-Flash-Vision-Exp 384
// (its compression pads included), DeepSeek-V4.1-Flash 1024.
inline constexpr uint64_t DS4V_MAX_IMAGE_BLOCK_TOKENS = 384;
inline constexpr uint64_t DS41V_MAX_IMAGE_BLOCK_TOKENS = 1024;
inline constexpr uint64_t max_image_block_tokens(bool v41) {
    return v41 ? DS41V_MAX_IMAGE_BLOCK_TOKENS : DS4V_MAX_IMAGE_BLOCK_TOKENS;
}

inline bool valid_image_spans(ImageSpanView spans, uint64_t prompt_size,
                              uint64_t max_block_tokens = DS4V_MAX_IMAGE_BLOCK_TOKENS) {
    return valid_image_spans(spans, prompt_size, DS4V_MAX_IMAGES, max_block_tokens);
}

// Rows the next staged prefill chunk takes at `position`, with `remaining`
// rows left in the prefix: about `budget` rows, whole image blocks only (a
// block, with any short text before it, may exceed the budget), and never a
// chunk or a leftover tail shorter than `min_rows`. 0 when no chunk exists.
inline int staged_prefill_chunk(ImageSpanView spans, uint64_t position, int remaining,
                                int budget, int min_rows) {
    const auto valid = [&](int n) {
        return n >= min_rows && (n == remaining || remaining - n >= min_rows);
    };
    const auto cut = [&](int proposed) {
        return atomic_image_chunk(spans, position, proposed, uint64_t(remaining), remaining);
    };
    int n = cut(std::min(remaining, std::max(budget, min_rows)));
    // Too short (text just before an image) or leaving a stub tail: grow to
    // the next boundary that is valid.
    for (int proposed = n + 1; !valid(n) && proposed <= remaining; ++proposed) n = cut(proposed);
    return valid(n) ? n : 0;
}

} // namespace luce::vision
