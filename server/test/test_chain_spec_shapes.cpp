#include "common/concurrency/chain_spec_shapes.h"
#include "host_check.h"

#include <cstdio>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

int main() {
    const std::vector<int32_t> draft = {10, 11, 12, 13};
    const int32_t full_posterior[] = {11, 12, 13, 14};
    CHECK(chain_verified_prefix(
        draft, full_posterior, 4) == draft.size());

    const int32_t rejected_posterior[] = {11, 99, 13, 14};
    CHECK(chain_verified_prefix(
        draft, rejected_posterior, 4) == 2);

    const int32_t root_rejected_posterior[] = {99, 12, 13, 14};
    CHECK(chain_verified_prefix(
        draft, root_rejected_posterior, 4) == 1);

    CHECK(chain_verified_prefix(
        draft, full_posterior, 0) == 1);
    CHECK(chain_verified_prefix(
        std::vector<int32_t>{}, full_posterior, 4) == 0);

    const std::vector<int> bucket_inputs = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17,
    };
    const std::vector<int> bucket_expected = {
        1, 2, 3, 4, 6, 6, 8, 8, 12, 12, 16, 16, 24,
    };
    for (size_t i = 0; i < bucket_inputs.size(); ++i) {
        CHECK(chain_decode_bucket_width(bucket_inputs[i]) ==
              bucket_expected[i]);
    }
    CHECK(chain_draft_bucket_width(4) == 4);
    CHECK(chain_draft_bucket_width(5) == 5);
    CHECK(chain_draft_bucket_width(6) == 6);
    CHECK(chain_draft_bucket_width(7) == 8);

    const auto eos = [](int32_t token) { return token == 2; };
    const std::vector<int32_t> eos_first_child = {10, 2, 11};
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child.data(), eos_first_child.size(), 0, 3, eos) == 1);
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child.data(), eos_first_child.size(), 2, 3, eos) == 2);
    const std::vector<int32_t> eos_second_child = {10, 11, 2, 12};
    CHECK(chain_min_tokens_safe_prefix(
        eos_second_child.data(), eos_second_child.size(), 0, 3, eos) == 2);
    CHECK(chain_min_tokens_safe_prefix(
        eos_second_child.data(), eos_second_child.size(), 1, 3, eos) == 3);
    const std::vector<int32_t> eos_root = {2, 11, 12};
    CHECK(chain_min_tokens_safe_prefix(
        eos_root.data(), eos_root.size(), 0, 3, eos) == eos_root.size());
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child.data(), eos_first_child.size(), 0, 0, eos) == 2);

    std::printf("chain spec shape tests passed: %d checks\n", g_checks);
    return 0;
}
