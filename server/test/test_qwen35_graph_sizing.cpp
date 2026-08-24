#include "qwen35/graph_sizing.h"

#include <cstdio>

using namespace dflash::common;

namespace {
int failures = 0;

#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
} } while (0)
} // namespace

int main() {
    constexpr size_t old_capacity = 16384;
    constexpr size_t measured_eight_prefill_nodes = 15061;
    constexpr size_t measured_eight_prefill_decode_nodes = 16741;

    CHECK(qwen35_target_graph_capacity(8, false) == old_capacity);
    CHECK(measured_eight_prefill_nodes < old_capacity);

    // C16 regression: eight prompts can prefill while earlier prompts decode.
    // The compact decode recurrence is the ninth independent segment.
    const size_t mixed_c16_capacity =
        qwen35_target_graph_capacity(8, true);
    CHECK(measured_eight_prefill_decode_nodes > old_capacity);
    CHECK(mixed_c16_capacity == 18432);
    CHECK(mixed_c16_capacity > measured_eight_prefill_decode_nodes);
    CHECK(mixed_c16_capacity - measured_eight_prefill_decode_nodes == 1691);

    // Preserve linear headroom when the prefill cohort itself grows.
    CHECK(qwen35_target_graph_capacity(16, true) == 34816);
    CHECK(qwen35_target_graph_capacity(32, true) == 67584);

    return failures == 0 ? 0 : 1;
}
