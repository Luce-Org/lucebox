#include "support/test_assert.h"
#include "qwen35/prefill_helpers.h"
#include <vector>
using namespace dflash::common;

TEST_CASE(ServerUnitFixture, test_qwen35_mrope_positions_axis_major) {
    std::vector<int32_t> standalone(4 * 5, -1);
    fill_qwen35_mrope_positions(
        standalone.data(), /*base_pos=*/7, /*n_tokens=*/5);
    const std::vector<int32_t> expected{
        7, 8, 9, 10, 11,
        7, 8, 9, 10, 11,
        7, 8, 9, 10, 11,
        0, 0, 0, 0, 0,
    };
    TEST_ASSERT(standalone == expected);

    constexpr int packed_tokens = 8;
    std::vector<int32_t> packed(4 * packed_tokens, -1);
    fill_qwen35_mrope_positions(
        packed.data(), packed_tokens, /*token_offset=*/2,
        /*base_pos=*/20, /*n_tokens=*/3);
    for (int axis = 0; axis < 4; ++axis) {
        for (int row = 0; row < packed_tokens; ++row) {
            const bool in_segment = row >= 2 && row < 5;
            const int expected_value = !in_segment
                ? -1
                : (axis < 3 ? 20 + row - 2 : 0);
            TEST_ASSERT(
                packed[(size_t)axis * packed_tokens + row] ==
                expected_value);
        }
    }
}
