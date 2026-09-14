#include "CppUnitTestFramework.hpp"
#include "qwen3/qwen3_drafter_model.h"
#include "support/test_assert.h"

#include <algorithm>
#include <numeric>
#include <vector>

using dflash::common::query_capture_slice;

namespace {
struct QueryCaptureFixture {};
}

TEST_CASE(QueryCaptureFixture, clips_query_to_chunk_and_reports_both_offsets) {
    const struct {
        int query_start, query_end, chunk_start, chunk_tokens;
        int chunk_offset, query_offset, tokens;
    } cases[] = {
        {4093, 4101, 0, 4096, 4093, 0, 3},
        {4093, 4101, 4096, 4096, 0, 3, 5},
        {4088, 4096, 0, 4096, 4088, 0, 8},
        {4096, 4104, 4096, 8, 0, 0, 8},
        {4100, 4108, 4096, 20, 4, 0, 8},
        {10, 30, 16, 4, 0, 6, 4},
    };
    for (const auto & c : cases) {
        const auto slice = query_capture_slice(
            c.query_start, c.query_end, c.chunk_start, c.chunk_tokens);
        REQUIRE(slice.valid());
        CHECK(slice.chunk_offset == c.chunk_offset);
        CHECK(slice.query_offset == c.query_offset);
        CHECK(slice.tokens == c.tokens);
    }
}

TEST_CASE(QueryCaptureFixture, disjoint_or_empty_ranges_do_not_capture) {
    CHECK(!query_capture_slice(8, 16, 0, 8).valid());
    CHECK(!query_capture_slice(8, 16, 16, 8).valid());
    CHECK(!query_capture_slice(8, 16, 32, 8).valid());
    CHECK(!query_capture_slice(8, 8, 0, 16).valid());
    CHECK(!query_capture_slice(8, 16, 10, 0).valid());
}

TEST_CASE(QueryCaptureFixture, chunked_capture_reconstructs_each_query_token_once) {
    // Exercise every alignment, including queries spanning more than two
    // chunks and the short final chunk. Values identify their source position.
    std::vector<int> source(35);
    std::iota(source.begin(), source.end(), 100);
    for (int chunk_size : {1, 3, 8, 16}) {
        for (int width = 1; width <= 8; ++width) {
            for (int start = 0; start + width <= (int)source.size(); ++start) {
                std::vector<int> captured(width, -1);
                std::vector<int> writes(width, 0);
                for (int cs = 0; cs < (int)source.size(); cs += chunk_size) {
                    const int count = std::min(chunk_size, (int)source.size() - cs);
                    const auto slice = query_capture_slice(start, start + width, cs, count);
                    if (!slice.valid()) continue;
                    REQUIRE(slice.chunk_offset >= 0);
                    REQUIRE(slice.chunk_offset + slice.tokens <= count);
                    REQUIRE(slice.query_offset >= 0);
                    REQUIRE(slice.query_offset + slice.tokens <= width);
                    for (int i = 0; i < slice.tokens; ++i) {
                        captured[slice.query_offset + i] = source[cs + slice.chunk_offset + i];
                        ++writes[slice.query_offset + i];
                    }
                }
                CHECK(captured == std::vector<int>(source.begin() + start,
                                                  source.begin() + start + width));
                CHECK(writes == std::vector<int>(width, 1));
            }
        }
    }
}

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
