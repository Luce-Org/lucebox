#include "deepseek4_image_assembly.h"
#include "deepseek4_image_spans.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

namespace dflash::vision {
namespace {
void require(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(message);
}
size_t elements(size_t rows, size_t dimension) {
    require(dimension && rows <= std::numeric_limits<size_t>::max() / dimension,
            "invalid image matrix dimensions");
    return rows * dimension;
}
bool finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}
void layout_valid(const ImageLayout & layout) {
    const auto & span = layout.span;
    require(valid_image_spans({&span, 1}, span.block_end) &&
            span.block_end - span.block_begin == layout.types.size(), "invalid image layout span");
    size_t images = 0;
    for (size_t row = 0; row < layout.types.size(); ++row) {
        const uint64_t pos = span.block_begin + row;
        const auto type = layout.types[row];
        if (pos == span.visible_begin) {
            require(type == ImageTokenType::Start, "image start identity mismatch");
        } else if (pos + 1 == span.visible_end) {
            require(type == ImageTokenType::End, "image end identity mismatch");
        } else if (pos < span.visible_begin || pos >= span.visible_end) {
            require(type == ImageTokenType::Pad, "image outer padding identity mismatch");
        } else {
            require(type == ImageTokenType::Pad || type == ImageTokenType::Newline ||
                    type == ImageTokenType::Image, "invalid image token identity");
        }
        images += type == ImageTokenType::Image;
    }
    require(images > 0 && images == layout.permutation.size(), "image permutation cardinality mismatch");
    std::vector<bool> seen(images, false);
    for (int64_t index : layout.permutation) {
        require(index >= 0 && uint64_t(index) < images && !seen[size_t(index)],
                "image permutation must contain each raster row exactly once");
        seen[size_t(index)] = true;
    }
}
void sentinels_valid(const ImageSentinels & s, size_t dimension) {
    for (const auto * row : {&s.start, &s.pad, &s.newline, &s.end}) {
        require(row->size() == dimension && dimension && finite(*row), "invalid image sentinel row");
    }
}
void cancelled(const ImageCancelled & callback) {
    require(!callback || !callback(), "image materialization cancelled");
}
} // namespace

std::shared_ptr<void> ImageRequestGate::try_acquire() const {
    bool expected = false;
    if (!active_->compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return {};
    // shared_ptr invokes the deleter if control-block allocation throws too.
    return std::shared_ptr<void>(active_.get(), [active = active_](void *) {
        active->store(false, std::memory_order_release);
    });
}

bool assemble_image_rows(const ImageLayout & layout, const ImageRaster & raster,
                         const ImageSentinels & sentinels, size_t dimension,
                         std::vector<float> & output, std::string & error) {
    error.clear();
    try {
        layout_valid(layout);
        sentinels_valid(sentinels, dimension);
        require(raster.rows == layout.permutation.size() && raster.columns == dimension &&
                raster.values.size() == elements(raster.rows, dimension) && finite(raster.values),
                "invalid projected image raster");
        std::vector<float> result(elements(layout.types.size(), dimension));
        size_t index = 0;
        for (size_t row = 0; row < layout.types.size(); ++row) {
            const float * source = nullptr;
            switch (layout.types[row]) {
                case ImageTokenType::Start: source = sentinels.start.data(); break;
                case ImageTokenType::Pad: source = sentinels.pad.data(); break;
                case ImageTokenType::Newline: source = sentinels.newline.data(); break;
                case ImageTokenType::End: source = sentinels.end.data(); break;
                case ImageTokenType::Image:
                    source = raster.values.data() + size_t(layout.permutation[index++]) * dimension;
                    break;
            }
            require(source != nullptr, "unknown image token type");
            std::copy_n(source, dimension, result.data() + row * dimension);
        }
        output.swap(result);
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
      catch (...) { error = "image assembly failed"; return false; }
}

bool materialize_image_rows(const std::vector<PromptImage> & images,
                            const ImageSentinels & sentinels, size_t dimension,
                            const ImageEncode & encode, const ImageCancelled & is_cancelled,
                            ImageRows & output, std::string & error) {
    error.clear();
    try {
        require(!images.empty() && images.size() <= 4 && bool(encode), "invalid image encode request");
        cancelled(is_cancelled);
        sentinels_valid(sentinels, dimension);
        uint64_t previous_end = 0;
        for (const auto & image : images) {
            layout_valid(image.layout);
            require(image.input.plan.aligner_rows && image.input.plan.aligner_cols &&
                    uint64_t(image.input.plan.aligner_rows) * image.input.plan.aligner_cols ==
                        image.layout.permutation.size(), "prepared aligner shape does not match raster rows");
            require(image.layout.span.block_begin >= previous_end, "overlapping image blocks");
            previous_end = image.layout.span.block_end;
        }
        ImageRows result;
        result.reserve(images.size());
        for (const auto & image : images) {
            cancelled(is_cancelled);
            ImageRaster raster;
            if (!encode(image, raster, error)) {
                if (error.empty()) error = "image encode failed";
                return false;
            }
            cancelled(is_cancelled);
            std::vector<float> rows;
            if (!assemble_image_rows(image.layout, raster, sentinels, dimension, rows, error)) return false;
            result.push_back(std::move(rows));
        }
        cancelled(is_cancelled);
        output.swap(result);
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
      catch (...) { error = "image encode callback failed"; return false; }
}

bool embed_image_prompt_chunk(const PreparedImagePrompt & prompt, const ImageRows & rows,
                               int32_t vocabulary, size_t dimension, size_t position,
                               size_t count, const TextEmbed & embed,
                               std::vector<float> & output, std::string & error) {
    error.clear();
    try {
        require(bool(prompt) && vocabulary > 0 && count && position <= prompt.tokens.size() &&
                count <= prompt.tokens.size() - position && rows.size() == prompt.images.size() &&
                prompt.images.size() <= 4, "invalid mixed embedding request");
        const size_t end = position + count;
        uint64_t previous_end = 0;
        for (size_t i = 0; i < prompt.images.size(); ++i) {
            const auto & layout = prompt.images[i].layout;
            const auto & span = layout.span;
            layout_valid(layout);
            require(span.block_begin >= previous_end && span.block_end <= prompt.tokens.size(),
                    "invalid mixed embedding spans");
            previous_end = span.block_end;
            if (span.block_end <= position || span.block_begin >= end) continue;
            require(span.block_begin >= position && span.block_end <= end,
                    "mixed embedding chunk splits an image block");
            require(rows[i].size() == elements(layout.types.size(), dimension) && finite(rows[i]),
                    "invalid materialized image matrix");
            for (size_t r = 0; r < layout.types.size(); ++r) {
                const int64_t expected = int64_t(vocabulary) + int64_t(layout.types[r]);
                require(int64_t(prompt.tokens[size_t(span.block_begin) + r]) == expected,
                        "external token does not match image layout");
            }
        }
        // Validate every ordinary ID before any callback can observe this chunk.
        size_t image_index = 0;
        for (size_t p = position; p < end; ++p) {
            while (image_index < prompt.images.size() && prompt.images[image_index].layout.span.block_end <= p)
                ++image_index;
            if (image_index < prompt.images.size() && prompt.images[image_index].layout.span.block_begin <= p) continue;
            require(prompt.tokens[p] >= 0 && prompt.tokens[p] < vocabulary, "unbound external token in text range");
        }
        std::vector<float> result(elements(count, dimension));
        size_t current = position;
        image_index = 0;
        while (current < end) {
            while (image_index < prompt.images.size() && prompt.images[image_index].layout.span.block_end <= current)
                ++image_index;
            const size_t text_end = image_index < prompt.images.size()
                ? std::min(end, size_t(prompt.images[image_index].layout.span.block_begin)) : end;
            if (current < text_end) {
                require(bool(embed) && embed(prompt.tokens.data() + current, text_end - current,
                        result.data() + (current - position) * dimension), "text embedding failed");
                current = text_end;
            }
            if (current == end) break;
            const auto & block = prompt.images[image_index].layout.span;
            std::copy(rows[image_index].begin(), rows[image_index].end(),
                      result.data() + (current - position) * dimension);
            current = size_t(block.block_end);
        }
        require(finite(result), "nonfinite mixed embeddings");
        output.swap(result);
        return true;
    } catch (const std::exception & e) { error = e.what(); return false; }
      catch (...) { error = "text embedding callback failed"; return false; }
}

} // namespace dflash::vision
