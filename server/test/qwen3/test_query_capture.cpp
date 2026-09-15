#include "CppUnitTestFramework.hpp"
#include "qwen3/qwen3_drafter_model.h"
#include "support/test_assert.h"

using dflash::common::query_capture_slice;

TEST_CASE(ServerUnitFixture, test_pflash_query_capture_splits_across_chunks) {
    const auto first = query_capture_slice(4093, 4101, 0, 4096);
    TEST_ASSERT(first.valid());
    TEST_ASSERT(first.chunk_offset == 4093);
    TEST_ASSERT(first.query_offset == 0);
    TEST_ASSERT(first.tokens == 3);

    const auto second = query_capture_slice(4093, 4101, 4096, 4096);
    TEST_ASSERT(second.valid());
    TEST_ASSERT(second.chunk_offset == 0);
    TEST_ASSERT(second.query_offset == 3);
    TEST_ASSERT(second.tokens == 5);

    TEST_ASSERT(!query_capture_slice(4093, 4101, 8192, 4096).valid());
}
