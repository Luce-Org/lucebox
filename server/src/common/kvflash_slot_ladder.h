// Deterministic prefill cuts (Track C, C1.3-04, D1).
// The recurrent (SSM + conv) state of a slot cannot be sliced back to an
// earlier position, so a rung (a state snapshot) has to sit on a position the
// slot reaches again: the cuts. The next prefill batch of a slot ends at the
// next multiple of U after `pos`, or at `end`: a function of position only, so
// idle, mixed and cancelled steps all produce the same boundaries and the
// rungs of kvflash_slot_turn.h always sit on them. No GPU, no IO.

#pragma once

#include <vector>

namespace luce::common {

inline int kvflash_next_cut(int pos, int end, int cut_u) {
    const int next = (pos / cut_u + 1) * cut_u;
    return next < end ? next : end;
}

}  // namespace luce::common
