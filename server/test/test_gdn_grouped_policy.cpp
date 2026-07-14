#include "gated_delta_net_policy.h"

#include <cstdio>

namespace {

bool expect(const char * name, bool actual, bool expected) {
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "FAIL: %s returned %s, expected %s\n",
        name,
        actual ? "true" : "false",
        expected ? "true" : "false");
    return false;
}

} // namespace

int main() {
    bool pass = true;

    pass &= expect("Ampere single-token decode",
        ggml_cuda_gdn_use_grouped_columns(true, 1, false, false), false);
    pass &= expect("Ampere partial prefill",
        ggml_cuda_gdn_use_grouped_columns(true, 511, false, false), false);
    pass &= expect("Ampere prefill threshold",
        ggml_cuda_gdn_use_grouped_columns(true, 512, false, false), true);
    pass &= expect("Ampere larger prefill",
        ggml_cuda_gdn_use_grouped_columns(true, 512, false, false), true);
    pass &= expect("non-Ampere default",
        ggml_cuda_gdn_use_grouped_columns(false, 1, false, false), true);
    pass &= expect("disable override",
        ggml_cuda_gdn_use_grouped_columns(true, 512, false, true), false);
    pass &= expect("force override",
        ggml_cuda_gdn_use_grouped_columns(true, 1, true, false), true);
    pass &= expect("force wins over disable",
        ggml_cuda_gdn_use_grouped_columns(true, 1, true, true), true);

    if (!pass) {
        return 1;
    }
    std::printf("PASS: grouped GDN policy preserves decode and selects Ampere prefill\n");
    return 0;
}
