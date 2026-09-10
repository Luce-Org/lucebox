#include "ggml-backend-meta-impl.h"

#include <cstdio>

int main() {
    int checks = 0;
    int failures = 0;
    const auto check = [&](bool ok, const char * label) {
        ++checks;
        if (!ok) {
            ++failures;
            std::fprintf(stderr, "[meta-split-layout] FAIL: %s\n", label);
        }
    };

    const ggml_backend_meta_split_state single = {
        GGML_BACKEND_SPLIT_AXIS_1, {2, 2}, 1, {1}};
    check(ggml_backend_meta_split_layout_equal(single, single, 2), "single segment");

    const ggml_backend_meta_split_state down = {
        GGML_BACKEND_SPLIT_AXIS_1, {1, 1, 1, 1}, 2, {1, 1}};
    auto weights = down;
    weights.axis = GGML_BACKEND_SPLIT_AXIS_0;
    check(ggml_backend_meta_split_layout_equal(down, weights, 2),
          "matching expert partition on different tensor axes");

    // Both states assign two experts per device, but down assigns [0,2]/[1,3]
    // while weights assigns [0,1]/[2,3]. Totals alone incorrectly accept this.
    const ggml_backend_meta_split_state wrong_weights = {
        GGML_BACKEND_SPLIT_AXIS_0, {2, 0, 0, 2}, 2, {1, 1}};
    check(!ggml_backend_meta_split_layout_equal(down, wrong_weights, 2),
          "equal totals with different expert identities");

    const ggml_backend_meta_split_state nonempty = {
        GGML_BACKEND_SPLIT_AXIS_1, {8, 8, 8, 8}, 2, {1, 1}};
    const ggml_backend_meta_split_state wrong_nonempty = {
        GGML_BACKEND_SPLIT_AXIS_0, {12, 4, 4, 12}, 2, {1, 1}};
    check(!ggml_backend_meta_split_layout_equal(nonempty, wrong_nonempty, 2),
          "equal totals with different nonempty expert segments");

    const ggml_backend_meta_split_state repeated = {
        GGML_BACKEND_SPLIT_AXIS_1, {1, 1, 1, 1}, 2, {2, 1}};
    check(ggml_backend_meta_split_layout_equal(repeated, repeated, 2),
          "matching repeated multi-segment layout");
    auto wrong_repeats = repeated;
    wrong_repeats.nr[0] = 1;
    wrong_repeats.nr[1] = 2;
    check(!ggml_backend_meta_split_layout_equal(repeated, wrong_repeats, 2),
          "conservatively reject different repeat encodings even when equivalent");

    const ggml_backend_meta_split_state varied_repeats = {
        GGML_BACKEND_SPLIT_AXIS_1, {1, 1, 2, 2, 1, 1}, 3, {2, 1, 1}};
    auto reordered_repeats = varied_repeats;
    reordered_repeats.nr[0] = 1;
    reordered_repeats.nr[2] = 2;
    check(!ggml_backend_meta_split_layout_equal(varied_repeats, reordered_repeats, 2),
          "equal totals but different repeated segment ordering");

    check(!ggml_backend_meta_split_layout_equal(down, single, 2),
          "different segment counts with equal totals");
    auto padded = down;
    padded.ne[4] = 123;
    padded.nr[2] = 456;
    check(ggml_backend_meta_split_layout_equal(down, padded, 2),
          "inactive storage is ignored");

    const ggml_backend_meta_split_state embedding = {
        GGML_BACKEND_SPLIT_AXIS_0, {4, 4, 4, 4}, 2, {1, 1}};
    const ggml_backend_meta_split_state wrong_shared = {
        GGML_BACKEND_SPLIT_AXIS_0, {8, 0, 0, 8}, 2, {1, 1}};
    check(ggml_backend_meta_split_layout_equal(embedding, embedding, 2),
          "matching shared embedding partition");
    check(!ggml_backend_meta_split_layout_equal(embedding, wrong_shared, 2),
          "equal totals with different shared embedding identities");

    auto invalid = down;
    invalid.n_segments = 0;
    check(!ggml_backend_meta_split_layout_equal(invalid, invalid, 2), "empty layout rejected");
    invalid.n_segments = 17;
    check(!ggml_backend_meta_split_layout_equal(invalid, invalid, 2), "oversized layout rejected");
    check(!ggml_backend_meta_split_layout_equal(down, down, 0), "zero devices rejected");
    check(!ggml_backend_meta_split_layout_equal(down, down, GGML_BACKEND_META_MAX_DEVICES + 1),
          "too many devices rejected");

    auto maximum = down;
    maximum.n_segments = 16;
    for (auto & count : maximum.ne) {
        count = 1;
    }
    for (auto & repeat : maximum.nr) {
        repeat = 1;
    }
    check(ggml_backend_meta_split_layout_equal(maximum, maximum, GGML_BACKEND_META_MAX_DEVICES),
          "maximum supported layout");
    auto wrong_last = maximum;
    ++wrong_last.ne[16 * GGML_BACKEND_META_MAX_DEVICES - 1];
    check(!ggml_backend_meta_split_layout_equal(maximum, wrong_last, GGML_BACKEND_META_MAX_DEVICES),
          "last segment and device participate in comparison");

    std::printf("[meta-split-layout] %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
