#pragma once

#include "deepseek4_vision_preprocess.h"
#include <algorithm>
#include <limits>

namespace dflash::vision {

struct ImageSpanView {
    const TokenSpan * data = nullptr;
    size_t size = 0;
};

inline const TokenSpan * image_block_at(ImageSpanView spans, uint64_t position) {
    for (size_t i = 0; i < spans.size; ++i) {
        const auto & span = spans.data[i];
        if (position < span.block_begin) break;
        if (position < span.block_end) return &span;
    }
    return nullptr;
}

inline bool valid_image_spans(ImageSpanView spans, uint64_t prompt_size) {
    if (spans.size > 4 || (spans.size && !spans.data)) return false;
    uint64_t previous_end = 0;
    for (size_t i = 0; i < spans.size; ++i) {
        const auto & span = spans.data[i];
        if (span.block_begin < previous_end || span.block_begin > span.visible_begin ||
            span.visible_begin >= span.visible_end || span.visible_end > span.block_end ||
            span.block_end > prompt_size || span.block_end - span.block_begin > 384) {
            return false;
        }
        previous_end = span.block_end;
    }
    return true;
}

inline int atomic_image_chunk(ImageSpanView spans, uint64_t position,
                              int proposed, uint64_t remaining, int capacity) {
    if (proposed <= 0 || capacity <= 0 || uint64_t(proposed) > remaining ||
        position > std::numeric_limits<uint64_t>::max() - remaining) return 0;
    uint64_t end = position + std::min(proposed, capacity);
    for (size_t i = 0; i < spans.size; ++i) {
        const auto & span = spans.data[i];
        if (span.block_end <= position) continue;
        if (span.block_begin < position) return 0;
        if (span.block_begin >= end) break;
        if (end < span.block_end) {
            end = span.block_begin == position ? span.block_end : span.block_begin;
            break;
        }
    }
    const uint64_t count = end - position;
    return count && count <= remaining && count <= uint64_t(capacity) ? int(count) : 0;
}

} // namespace dflash::vision
