#pragma once

namespace dflash::common {

// A checkpoint's declared block size is its published proposal horizon.
// Greedy chain verification keeps output byte-identical to plain decode at
// any width, so widening only risks acceptance depth, and measured on the
// Qwen3.8 DFlash2 checkpoint it extrapolates: completions stayed
// byte-identical across widths 8-24 while per-step commits grew through 16.
// Past 2x the horizon the step time cliffs with no commit gain, so that is
// where the guard now sits.
constexpr bool draft_block_size_override_supported(
        int requested, int checkpoint_block_size) {
    return requested == 0 ||
           (requested >= 2 && checkpoint_block_size >= 2 &&
            requested <= 2 * checkpoint_block_size);
}

}  // namespace dflash::common
