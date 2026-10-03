// Qwen3.5 image input, the parts that need no GPU: target size, tower patch
// order, position tables, prompt expansion and rotary positions.
#include "qwen35/qwen35_image_prompt.h"
#include "qwen35/qwen35_vision.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace luce::common;
using namespace luce::vision;

static int failures = 0;
static void check(bool condition, const char * message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

static Qwen35VisionConfig config() {
    Qwen35VisionConfig c;
    c.patch_size = 16;
    c.merge = 2;
    return c;
}

// Expected sizes computed with the model's reference smart_resize
// (factor 32, 64 to 1024 image tokens).
static void test_target_size() {
    const Qwen35VisionConfig c = config();
    int w = 0, h = 0;
    std::string error;
    check(qwen35_vision_target_size(c, 640, 480, w, h, error) && w == 640 && h == 480, "aligned size is kept");
    check(qwen35_vision_target_size(c, 4000, 3000, w, h, error) && w == 1152 && h == 864, "large image shrinks under the cap");
    check(qwen35_vision_target_size(c, 100, 50, w, h, error) && w == 384 && h == 192, "small image grows to the floor");
    check(qwen35_vision_target_size(c, 304, 272, w, h, error) && w == 320 && h == 256, "halves round to even multiples");
    check(!qwen35_vision_target_size(c, 3000, 10, w, h, error) && !error.empty(), "extreme aspect ratio is rejected");
    check(!qwen35_vision_target_size(c, 0, 10, w, h, error), "empty image is rejected");
}

static void test_tower_order() {
    // 4 patches across, 2 down: two 2x2 blocks, each listed row by row.
    std::vector<int32_t> rope;
    qwen35_vision_rope_positions(4, 2, 2, rope);
    const std::vector<int32_t> y = {0, 0, 1, 1, 0, 0, 1, 1}, x = {0, 1, 0, 1, 2, 3, 2, 3};
    bool ok = rope.size() == 32;
    for (int i = 0; ok && i < 8; ++i) {
        ok = rope[i] == y[i] && rope[8 + i] == x[i] && rope[16 + i] == y[i] && rope[24 + i] == x[i];
    }
    check(ok, "rotary positions follow merged patch order");

    // A 4x4 table read on a 4x4 grid is the table itself, in merged order.
    std::vector<float> table(16), out;
    for (int i = 0; i < 16; ++i) table[i] = float(i);
    qwen35_vision_position_rows(table, 4, 1, 4, 4, 2, out);
    const std::vector<float> same = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};
    check(out == same, "matching grid reads the table unchanged");

    // On a 2x2 grid aligned corners land on the table's corners.
    qwen35_vision_position_rows(table, 4, 1, 2, 2, 2, out);
    check(out == std::vector<float>({0, 3, 12, 15}), "coarser grid samples aligned corners");

    // 4 points across a 3-wide table: 0, 2/3, 4/3, 2.
    std::vector<float> ramp = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    qwen35_vision_position_rows(ramp, 3, 1, 4, 2, 2, out);
    bool close = out.size() == 8;
    const float expected[8] = {0.0f, 2.0f / 3, 0.0f, 2.0f / 3, 4.0f / 3, 2.0f, 4.0f / 3, 2.0f};
    for (int i = 0; close && i < 8; ++i) close = out[i] > expected[i] - 1e-5f && out[i] < expected[i] + 1e-5f;
    check(close, "finer grid interpolates linearly");
}

static void test_prompt() {
    constexpr int32_t PAD = 9;
    std::vector<int32_t> tokens = {1, 2, PAD, 3, PAD, 4};
    std::vector<Qwen35ImageSlot> slots(2);
    slots[0].columns = 3; slots[0].rows = 2;
    slots[1].columns = 1; slots[1].rows = 2;
    std::string error;
    check(qwen35_expand_image_tokens(tokens, PAD, slots, 64, error), "expansion succeeds");
    check(tokens == std::vector<int32_t>({1, 2, PAD, PAD, PAD, PAD, PAD, PAD, 3, PAD, PAD, 4}), "one pad per image token");
    check(slots[0].begin == 2 && slots[1].begin == 9, "slots record where their image starts");

    const Qwen35RopePositions p = qwen35_image_rope_positions((int) tokens.size(), slots);
    // Text 0,1. Image at 2: 2 rows x 3 columns, then text resumes at 2 + 3.
    // Image at 6: 2 rows x 1 column, then text resumes at 6 + 2.
    check(p.temporal == std::vector<int32_t>({0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 6, 8}), "temporal axis holds still inside an image");
    check(p.height == std::vector<int32_t>({0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 8}), "height axis counts image rows");
    check(p.width == std::vector<int32_t>({0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 6, 8}), "width axis counts image columns");
    check(p.next == 9, "generation resumes after the widest image side");

    std::vector<int32_t> chunk(4 * 3, -1);
    p.fill(chunk.data(), 7, 3);
    check(chunk == std::vector<int32_t>({2, 5, 6, 3, 5, 6, 4, 5, 6, 0, 0, 0}), "chunk fill is axis major");

    std::vector<int32_t> mismatch = {1, PAD};
    check(!qwen35_expand_image_tokens(mismatch, PAD, slots, 64, error), "marker count must match image count");
    std::vector<int32_t> small = {1, PAD, PAD};
    check(!qwen35_expand_image_tokens(small, PAD, slots, 8, error), "expanded prompt must fit the limit");
    check(small == std::vector<int32_t>({1, PAD, PAD}), "a refused prompt is left unchanged");
}

static void test_overwrite() {
    Qwen35ImageSlot slot;
    slot.begin = 3; slot.columns = 2; slot.rows = 2;  // tokens 3..6
    const std::vector<float> rows = {10, 11, 20, 21, 30, 31, 40, 41};  // hidden = 2
    std::vector<float> chunk(3 * 2, 0.0f);
    qwen35_overwrite_image_rows(slot, rows.data(), 2, chunk.data(), 0, 3);   // tokens 0..2
    check(chunk == std::vector<float>(6, 0.0f), "chunk before the image is untouched");
    qwen35_overwrite_image_rows(slot, rows.data(), 2, chunk.data(), 2, 3);   // tokens 2..4
    check(chunk == std::vector<float>({0, 0, 10, 11, 20, 21}), "image head lands at its offset");
    chunk.assign(6, 0.0f);
    qwen35_overwrite_image_rows(slot, rows.data(), 2, chunk.data(), 5, 3);   // tokens 5..7
    check(chunk == std::vector<float>({30, 31, 40, 41, 0, 0}), "image tail continues in the next chunk");
}

int main() {
    test_target_size();
    test_tower_order();
    test_prompt();
    test_overwrite();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
