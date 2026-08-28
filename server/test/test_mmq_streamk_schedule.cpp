#include "CppUnitTestFramework.hpp"
#include "mmq-streamk-schedule.h"

#include <climits>
#include <cstdint>

namespace {
constexpr int kIterK = 256;
constexpr int kNsm86 = 82;
struct MmqStreamkScheduleFixture {};
} // namespace

TEST_CASE(MmqStreamkScheduleFixture, useful_chunk_cap) {
    // Validated SM86 path: avoid empty CTAs for shallow K.
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, kNsm86, 256, kIterK, true, true), 20);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, kNsm86, 512, kIterK, true, true), 40);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, kNsm86, 5120, kIterK, true, true), kNsm86);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(1, kNsm86, 256, kIterK, true, true), 1);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(1, kNsm86, 5120, kIterK, true, true), 20);

    // qk-aligned K tails stay with the tile's final CTA. Counting ceil(K/256)
    // would produce alternating empty CTAs and a spurious fixup launch.
    const int k320_stream_blocks = mmq_stream_k_nblocks(20, kNsm86, 320, kIterK, true, true);
    REQUIRE_EQUAL(k320_stream_blocks, 20);
    REQUIRE_FALSE(mmq_stream_k_fixup_needed(20, k320_stream_blocks));
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, kNsm86, 384, kIterK, true, true), 20);
    const int k576_stream_blocks = mmq_stream_k_nblocks(20, kNsm86, 576, kIterK, true, true);
    REQUIRE_EQUAL(k576_stream_blocks, 40);
    REQUIRE_TRUE(mmq_stream_k_fixup_needed(20, k576_stream_blocks));
}

TEST_CASE(MmqStreamkScheduleFixture, unchanged_paths_and_input_guards) {
    // Existing >=90% NVIDIA tiling behavior remains unchanged.
    REQUIRE_EQUAL(mmq_stream_k_nblocks(74, kNsm86, 5120, kIterK, true, false), 74);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(kNsm86, kNsm86, 5120, kIterK, true, false), kNsm86);

    // Fail closed: non-SM86 and non-NVIDIA paths retain the old nsm schedule.
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, kNsm86, 256, kIterK, true, false), kNsm86);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, kNsm86, 256, kIterK, false, false), kNsm86);

    // Invalid inputs remain bounded, and 64-bit arithmetic avoids overflow.
    REQUIRE_EQUAL(mmq_stream_k_nblocks(0, kNsm86, 256, kIterK, true, true), 1);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, 0, 256, kIterK, true, true), 1);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, kNsm86, 0, kIterK, true, true), 1);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(20, kNsm86, 256, 0, true, true), 1);
    REQUIRE_EQUAL(mmq_stream_k_nblocks(INT_MAX / 2, kNsm86, INT64_MAX / 4, kIterK, true, true), INT_MAX / 2);

    // Fixup predicate itself remains unchanged.
    REQUIRE_FALSE(mmq_stream_k_fixup_needed(20, 20));
    REQUIRE_TRUE(mmq_stream_k_fixup_needed(20, 40));
    REQUIRE_TRUE(mmq_stream_k_fixup_needed(20, kNsm86));
}
