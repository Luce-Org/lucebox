#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace dflash::vision {

constexpr size_t MAX_IMAGE_EXPERTS = 256;
struct ImageExpertSelection {
    size_t count = 0;
    std::array<int32_t, MAX_IMAGE_EXPERTS> indices{};
    std::array<float, MAX_IMAGE_EXPERTS> weights{};
};

// Scores are unbiased sqrt(softplus(logits)); bias affects selection only.
// Input arrays contain experts readable floats and do not alias output.
// Equal corrected scores select lower expert indices first. Torch topk does
// not specify equal-score ordering, so tie parity is not part of this contract.
bool select_image_experts(const float * scores, const float * bias, size_t experts,
                          size_t topk, ImageExpertSelection & output, std::string & error,
                          float route_scale = 1.5f);

// Only raw keys are covered. Pass (-1,-1) for no image; otherwise the range is
// [IMAGE_START, IMAGE_END+1), excluding leading compression padding. A range
// widens visibility only when it contains query. False means invalid arguments;
// visible is cleared on failure. No request span or compressed-row state is owned.
bool raw_key_visible(int64_t query, int64_t key, int64_t window,
                     int64_t image_begin, int64_t image_end, bool & visible);

} // namespace dflash::vision
