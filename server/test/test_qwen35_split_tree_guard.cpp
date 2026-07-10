#include "qwen35_layer_split_tree_guard.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using dflash::common::qwen35_split_run_if_root_inclusive_pure_chain;

namespace {

bool mark_executed(void * context) {
    *static_cast<bool *>(context) = true;
    return true;
}

bool expect(const char * name, const std::vector<int32_t> & parents,
            std::size_t n_actual, bool expected) {
    bool executed = false;
    const bool result = qwen35_split_run_if_root_inclusive_pure_chain(
        parents.data(), parents.size(), n_actual, mark_executed, &executed);
    const bool ok = result == expected && executed == expected;
    std::fprintf(stderr,
                 "qwen35_split_tree_guard case=%s result=%d sentinel_executed=%d expected=%d pass=%d\n",
                 name, result ? 1 : 0, executed ? 1 : 0, expected ? 1 : 0, ok ? 1 : 0);
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= expect("single_root", {-1}, 1, true);
    ok &= expect("pure_chain", {-1, 0, 1, 2, 3, 4}, 6, true);
    ok &= expect("malformed_root", {0, 0, 1}, 3, false);
    ok &= expect("skipped_parent", {-1, 0, 0}, 3, false);
    ok &= expect("sibling", {-1, 0, 0, 2}, 4, false);
    ok &= expect("cycle_or_forward_parent", {-1, 2, 1}, 3, false);
    ok &= expect("out_of_range", {-1, 0, 7}, 3, false);
    ok &= expect("truncated", {-1, 0}, 3, false);

    bool null_executed = false;
    const bool null_ok = !qwen35_split_run_if_root_inclusive_pure_chain(
        nullptr, 0, 1, mark_executed, &null_executed) && !null_executed;
    std::fprintf(stderr,
                 "qwen35_split_tree_guard case=null result=%d sentinel_executed=%d pass=%d\n",
                 0, null_executed ? 1 : 0, null_ok ? 1 : 0);
    ok &= null_ok;

    std::fprintf(stderr, "qwen35_split_tree_guard overall=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
