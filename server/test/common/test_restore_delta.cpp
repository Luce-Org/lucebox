#include "CppUnitTestFramework.hpp"
#include "common/restore_delta.h"

#include <stdexcept>
#include <vector>

namespace {
struct RestoreDeltaFixture {};
}

TEST_CASE(RestoreDeltaFixture, restore_delta_excludes_cached_prefix) {
    using dflash::common::restore_prompt_delta;
    const std::vector<int32_t> prompt = {10, 11, 20, 21, 22};
    const std::vector<int32_t> delta = restore_prompt_delta(prompt, 2);
    CHECK(delta == std::vector<int32_t>({20, 21, 22}));
}

TEST_CASE(RestoreDeltaFixture, restore_delta_empty_on_full_hit) {
    using dflash::common::restore_prompt_delta;
    const std::vector<int32_t> prompt = {10, 11, 20, 21, 22};
    const std::vector<int32_t> full_hit_delta = restore_prompt_delta(prompt, 5);
    CHECK(full_hit_delta.empty());
}

TEST_CASE(RestoreDeltaFixture, restore_delta_rejects_long_prefix) {
    using dflash::common::restore_prompt_delta;
    const std::vector<int32_t> prompt = {10, 11, 20, 21, 22};
    CHECK_THROW(std::out_of_range, UNUSED_RETURN(restore_prompt_delta(prompt, 6)));
}
