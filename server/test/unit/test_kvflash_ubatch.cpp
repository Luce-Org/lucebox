#include "CppUnitTestFramework.hpp"
#include "../../src/qwen35/prefill_helpers.h"

namespace {
struct KvflashUbatchFixture {};
}

using luce::common::kvflash_pooled_ubatch;

TEST_CASE(KvflashUbatchFixture, rounded_to_chunk_multiple_never_below_one_chunk) {
    CHECK(kvflash_pooled_ubatch(512, 64, 8192) == 512);
    CHECK(kvflash_pooled_ubatch(64, 64, 8192) == 64);
    CHECK(kvflash_pooled_ubatch(100, 64, 8192) == 64);   // round down
    CHECK(kvflash_pooled_ubatch(700, 128, 8192) == 640); // round down to 5*128
    CHECK(kvflash_pooled_ubatch(0, 64, 8192) == 64);     // at least one chunk
    CHECK(kvflash_pooled_ubatch(500, 0, 8192) == 500);   // no chunk -> passthrough
}

TEST_CASE(KvflashUbatchFixture, clamped_to_pool) {
    // A ubatch larger than the pool would evict its own not-yet-computed chunks.
    CHECK(kvflash_pooled_ubatch(1024, 64, 512) == 512);
    CHECK(kvflash_pooled_ubatch(512, 64, 256) == 256);
    CHECK(kvflash_pooled_ubatch(4096, 64, 512) == 512);
    // Pool smaller than one chunk: the one-chunk floor wins over the pool clamp.
    // The pager's pool is always a positive chunk multiple, so this only pins
    // the documented precedence.
    CHECK(kvflash_pooled_ubatch(1024, 64, 32) == 64);
}

TEST_CASE(KvflashUbatchFixture, clamped_to_the_unpinned_capacity_when_the_preamble_pins_rows) {
    // 2048-token pool with 1152 rows pinned: a 1024 batch (16 chunks) cannot fit the 14 unpinned blocks beside the
    // one the pager keeps free, so each batch would evict its own chunks. 13 chunks fit.
    CHECK(kvflash_pooled_ubatch(1024, 64, 2048, 1152) == 832);
    CHECK(kvflash_pooled_ubatch(512, 64, 2048, 1152) == 512);    // the default ubatch is untouched
    CHECK(kvflash_pooled_ubatch(1024, 64, 2048, 0) == 1024);     // no pins: the old clamp
    CHECK(kvflash_pooled_ubatch(1024, 64, 1280, 1152) == 64);    // the one-chunk floor still wins
}
