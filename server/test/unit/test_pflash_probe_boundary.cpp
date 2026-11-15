// Unit test for the segment probe's boundary smoothing (no GPU): pflash_probe_boundary_scores must equal, bit for
// bit, the loop the strict scorer ran inline (tap-count Conv1d with zero padding plus the residual logit, sigmoid),
// so hoisting it out of the per-pin segmentation changes no segment.

#include "CppUnitTestFramework.hpp"

#include "pflash_selection.h"

#include <cmath>
#include <vector>

using namespace luce::pflash;

namespace {
struct PflashProbeBoundaryFixture {};

// The scorer's inline loop, verbatim (qwen35_drafter.cpp before the hoist).
std::vector<float> reference(const std::vector<float> & raw, const std::vector<float> & conv_w, float conv_b) {
    const int S = (int) raw.size();
    const int taps = (int) conv_w.size();
    const int radius = taps / 2;
    std::vector<float> out((size_t) S, 0.0f);
    for (int t = 0; t < S; ++t) {
        float acc = raw[(size_t) t] + conv_b;
        for (int k = 0; k < taps; ++k) {
            const int u = t + k - radius;
            if (u >= 0 && u < S) acc += conv_w[(size_t) k] * raw[(size_t) u];
        }
        out[(size_t) t] = 1.0f / (1.0f + std::exp(-acc));
    }
    return out;
}

std::vector<float> ramp(int n, float scale) {
    std::vector<float> v((size_t) n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = scale * (float) ((i * 7919) % 101 - 50) / 13.0f;
    return v;
}

bool bitwise_equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    }
    return true;
}
}

TEST_CASE(PflashProbeBoundaryFixture, five_taps_match_the_inline_loop_bit_for_bit) {
    const std::vector<float> raw = ramp(1000, 0.9f);
    const std::vector<float> conv_w{0.1f, -0.3f, 0.7f, 0.2f, -0.05f};
    CHECK(bitwise_equal(pflash_probe_boundary_scores(raw, conv_w, 0.37f), reference(raw, conv_w, 0.37f)));
}
TEST_CASE(PflashProbeBoundaryFixture, edges_use_zero_padding) {
    // 3 tokens, 5 taps: every output touches the padding on at least one side.
    const std::vector<float> raw{1.5f, -2.0f, 0.25f};
    const std::vector<float> conv_w{0.5f, 0.25f, -1.0f, 0.125f, 2.0f};
    CHECK(bitwise_equal(pflash_probe_boundary_scores(raw, conv_w, -0.5f), reference(raw, conv_w, -0.5f)));
}
TEST_CASE(PflashProbeBoundaryFixture, one_tap_is_a_sigmoid_of_the_scaled_logit) {
    const std::vector<float> raw{0.0f, 2.0f};
    const std::vector<float> out = pflash_probe_boundary_scores(raw, {1.0f}, 0.0f);
    REQUIRE(out.size() == 2u);
    CHECK(out[0] == 0.5f);
    CHECK(std::fabs(out[1] - 1.0f / (1.0f + std::exp(-4.0f))) < 1e-7f);
}
TEST_CASE(PflashProbeBoundaryFixture, empty_input_gives_empty_output) {
    CHECK(pflash_probe_boundary_scores({}, {0.1f, 0.2f, 0.3f}, 0.0f).empty());
}
