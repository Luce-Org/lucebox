#include "CppUnitTestFramework.hpp"
#include "../src/qwen35/prefill_helpers.h"

#include <cstring>
#include <vector>

namespace {
struct Qwen35MaskUploadFixture {};
}

using luce::common::align_up;
using luce::common::build_causal_mask;
using luce::common::qwen35_causal_mask_live_width;

TEST_CASE(Qwen35MaskUploadFixture, live_width_covers_the_attention_view) {
    // The FA view is the window rounded up to at most 256 keys; the live
    // width must cover it, plus one stride for a final partial KV tile.
    const int full = 131072 + 512;
    for (int kv_len : {1, 17, 255, 256, 257, 4096, 26725, 131000}) {
        const int live = qwen35_causal_mask_live_width(kv_len, full);
        CHECK(live >= align_up(kv_len, 256) + 256 || live == full);
        CHECK(live <= full);
    }
    CHECK(qwen35_causal_mask_live_width(full - 1, full) == full);
}

TEST_CASE(Qwen35MaskUploadFixture, live_rows_match_the_full_width_mask) {
    // Every column the upload writes holds the value the full-width mask had.
    const int full = 8192 + 64;
    struct Shape { int kv_start, n_tokens, win_start; };
    for (const Shape s : {Shape{0, 512, 0}, Shape{512, 512, 0}, Shape{3000, 16, 0},
                          Shape{5000, 16, 2952}, Shape{7000, 101, 0}}) {
        const int kv_len = s.kv_start + s.n_tokens - s.win_start;
        const int live = qwen35_causal_mask_live_width(kv_len, full);
        std::vector<uint16_t> wide, narrow;
        build_causal_mask(wide, kv_len, s.n_tokens, s.kv_start, 32, s.win_start, full);
        build_causal_mask(narrow, kv_len, s.n_tokens, s.kv_start, 32, s.win_start, live);
        const size_t rows = narrow.size() / (size_t)live;
        CHECK(rows * (size_t)full == wide.size());
        for (size_t r = 0; r < rows; ++r) {
            CHECK(std::memcmp(narrow.data() + r * live, wide.data() + r * full,
                              sizeof(uint16_t) * (size_t)live) == 0);
        }
    }
}
