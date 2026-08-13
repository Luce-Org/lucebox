#include "CppUnitTestFramework.hpp"
#include "qwen35_layer_split_tree_guard.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace CppUnitTestFramework;
using dflash::common::qwen35_split_run_if_root_inclusive_pure_chain;

namespace {

bool mark_executed(void * context) {
    *static_cast<bool *>(context) = true;
    return true;
}

struct Qwen35SplitTreeGuardFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};

bool runs_as_expected(const std::vector<int32_t> & parents, std::size_t n_actual,
                      bool expected) {
    bool executed = false;
    const bool result = qwen35_split_run_if_root_inclusive_pure_chain(
        parents.data(), parents.size(), n_actual, mark_executed, &executed);
    return result == expected && executed == expected;
}

}  // namespace

TEST_CASE(Qwen35SplitTreeGuardFixture, root_inclusive_pure_chain_guard) {
    CHECK(runs_as_expected({-1}, 1, true));
    CHECK(runs_as_expected({-1, 0, 1, 2, 3, 4}, 6, true));
    CHECK(runs_as_expected({0, 0, 1}, 3, false));
    CHECK(runs_as_expected({-1, 0, 0}, 3, false));
    CHECK(runs_as_expected({-1, 0, 0, 2}, 4, false));
    CHECK(runs_as_expected({-1, 2, 1}, 3, false));
    CHECK(runs_as_expected({-1, 0, 7}, 3, false));
    CHECK(runs_as_expected({-1, 0}, 3, false));

    bool null_executed = false;
    CHECK(!qwen35_split_run_if_root_inclusive_pure_chain(
        nullptr, 0, 1, mark_executed, &null_executed));
    CHECK(!null_executed);
}
