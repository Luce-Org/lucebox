#include "CppUnitTestFramework.hpp"
#include "../deps/llama.cpp/ggml/src/ggml-cuda/mmq-tile-selection.h"

#include <cstdint>
#include <limits>

namespace {
struct MmqTileSelection {};
bool wave32_tiles(int x) { return x % 16 == 0; }
bool skip_none(int) { return false; }
}

TEST_CASE(MmqTileSelection, default_search_checks_wider_candidates) {
    CHECK(ggml_cuda_mmq_select_x(64, 128, 0, wave32_tiles, skip_none) == 64);
    CHECK(ggml_cuda_mmq_select_x(256, 128, 0, wave32_tiles, skip_none) == 128);
    CHECK(ggml_cuda_mmq_select_x(8192, 64, 0, wave32_tiles, skip_none) == 64);
}

TEST_CASE(MmqTileSelection, admitted_override_preserves_tuned_width) {
    for (int requested : {16, 32, 48}) {
        CHECK(ggml_cuda_mmq_select_x(
            8192, 128, requested, wave32_tiles, skip_none) == requested);
    }
}

TEST_CASE(MmqTileSelection, rejected_override_runs_complete_default_search) {
    CHECK(ggml_cuda_mmq_select_x(256, 128, 24, wave32_tiles, skip_none) == 128);
    CHECK(ggml_cuda_mmq_select_x(256, 64, 128, wave32_tiles, skip_none) == 64);
    const auto fits_scratch = [](int x) { return x % 16 == 0 && x <= 64; };
    CHECK(ggml_cuda_mmq_select_x(256, 128, 96, fits_scratch, skip_none) == 64);
}

TEST_CASE(MmqTileSelection, automatic_exclusion_does_not_override_explicit_policy) {
    const auto skip_32 = [](int x) { return x == 32; };
    CHECK(ggml_cuda_mmq_select_x(32, 128, 0, wave32_tiles, skip_32) == 48);
    CHECK(ggml_cuda_mmq_select_x(32, 128, 32, wave32_tiles, skip_32) == 32);
}

TEST_CASE(MmqTileSelection, short_batches_and_ties_keep_smallest_sufficient_tile) {
    CHECK(ggml_cuda_mmq_select_x(1, 128, 0, wave32_tiles, skip_none) == 16);
    CHECK(ggml_cuda_mmq_select_x(129, 128, 0, wave32_tiles, skip_none) == 80);
}

TEST_CASE(MmqTileSelection, eligibility_and_large_column_counts_are_preserved) {
    const auto unsupported = [](int) { return false; };
    CHECK(ggml_cuda_mmq_select_x(256, 128, 32, unsupported, skip_none) == 0);
    CHECK(ggml_cuda_mmq_select_x(
        std::numeric_limits<int64_t>::max(), 128, 0,
        wave32_tiles, skip_none) == 128);
}
