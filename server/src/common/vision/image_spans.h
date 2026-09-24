// Where images sit inside a prompt. Model independent: a backend uses these to
// keep an image inside one prefill batch and to build its attention mask.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace luce::vision {

// Half-open absolute token positions. `block` covers every token the image
// expanded to; `visible` is the part that attends bidirectionally.
struct TokenSpan {
    std::uint64_t block_begin = 0;
    std::uint64_t visible_begin = 0;
    std::uint64_t visible_end = 0;
    std::uint64_t block_end = 0;
};

// Borrowed, sorted by position, non-overlapping.
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

// End of the last image block that overlaps [begin, end), or 0 when none does.
inline uint64_t last_image_end_in(ImageSpanView spans, uint64_t begin, uint64_t end) {
    uint64_t last = 0;
    for (size_t i = 0; i < spans.size; ++i) {
        const auto & span = spans.data[i];
        if (span.block_begin >= end) break;
        if (span.block_end > begin) last = span.block_end;
    }
    return last;
}

inline bool valid_image_spans(ImageSpanView spans, uint64_t prompt_size,
                              size_t max_images, uint64_t max_block_tokens) {
    if (spans.size > max_images || (spans.size && !spans.data)) return false;
    uint64_t previous_end = 0;
    for (size_t i = 0; i < spans.size; ++i) {
        const auto & span = spans.data[i];
        if (span.block_begin < previous_end || span.block_begin > span.visible_begin ||
            span.visible_begin >= span.visible_end || span.visible_end > span.block_end ||
            span.block_end > prompt_size || span.block_end - span.block_begin > max_block_tokens) {
            return false;
        }
        previous_end = span.block_end;
    }
    return true;
}

// A batch of about `proposed` tokens starting at `position` that does not cut
// an image in two: it stops before an image it cannot hold, and grows up to
// `capacity` to finish an image it starts with. Returns 0 when an image that
// starts here does not fit in `capacity`.
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

} // namespace luce::vision
