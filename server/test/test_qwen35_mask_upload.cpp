#include "CppUnitTestFramework.hpp"
#include "../src/qwen35/prefill_helpers.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
struct Qwen35MaskUploadFixture {};
}

using luce::common::align_up;
using luce::common::build_causal_mask;
using luce::common::qwen35_causal_mask_live_width;
using luce::common::upload_qwen35_causal_mask_window;

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

TEST_CASE(Qwen35MaskUploadFixture, upload_writes_live_columns_and_keeps_the_rest) {
    // A max_ctx-wide mask tensor: the upload must write the live columns of
    // every row through the strided 2-D copy and leave the rest untouched.
    // With LUCE_QWEN35_MASK_FULL_WIDTH=1 (ctest
    // server_unit_qwen35_mask_full_width) it must write every column.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 2;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    const int full = 4096 + 64;
    const int q_pad = 32;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, full, q_pad);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    constexpr uint16_t kSentinel = 0x1234;
    std::vector<uint16_t> got((size_t)full * q_pad, kSentinel);
    ggml_backend_tensor_set(mask, got.data(), 0, sizeof(uint16_t) * got.size());

    const int kv_start = 700, n_tokens = 16;
    upload_qwen35_causal_mask_window(mask, kv_start + n_tokens, n_tokens, kv_start, 32, 0);
    ggml_backend_tensor_get(mask, got.data(), 0, sizeof(uint16_t) * got.size());

    std::vector<uint16_t> want;
    build_causal_mask(want, kv_start + n_tokens, n_tokens, kv_start, 32, 0, full);
    const char * full_env = std::getenv("LUCE_QWEN35_MASK_FULL_WIDTH");
    const bool full_width = full_env && full_env[0] == '1';
    const int live = full_width ? full : qwen35_causal_mask_live_width(kv_start + n_tokens, full);
    CHECK(full_width || live < full);
    CHECK(want.size() == got.size());
    for (int r = 0; r < q_pad; ++r) {
        CHECK(std::memcmp(got.data() + (size_t)r * full, want.data() + (size_t)r * full,
                          sizeof(uint16_t) * (size_t)live) == 0);
        bool untouched = true;
        for (int c = live; c < full; ++c) {
            untouched = untouched && got[(size_t)r * full + c] == kSentinel;
        }
        CHECK(untouched);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}
