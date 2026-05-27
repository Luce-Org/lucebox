#include "common/restore_delta.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

static int failures = 0;

static void check(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

int main() {
    using dflash::common::restore_prompt_delta;

    // Regression for #216: RESTORE receives the full prompt, but the backend
    // must prefill only the suffix that was not covered by the cached snapshot.
    const std::vector<int32_t> prompt = {10, 11, 20, 21, 22};
    const std::vector<int32_t> delta = restore_prompt_delta(prompt, 2);
    check((delta == std::vector<int32_t>{20, 21, 22}),
          "RESTORE delta excludes cached prefix tokens");

    const std::vector<int32_t> full_hit_delta = restore_prompt_delta(prompt, 5);
    check(full_hit_delta.empty(),
          "RESTORE delta is empty when prompt exactly matches cached prefix");

    bool rejected_long_prefix = false;
    try {
        (void)restore_prompt_delta(prompt, 6);
    } catch (const std::out_of_range &) {
        rejected_long_prefix = true;
    }
    check(rejected_long_prefix,
          "RESTORE delta rejects cached prefixes longer than the prompt");

    return failures == 0 ? 0 : 1;
}
