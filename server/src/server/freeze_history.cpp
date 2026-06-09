// freeze_history — pure partition logic (GREEN).

#include "server/freeze_history.h"
#include "server/prefix_cache.h"  // hash_prefix

namespace dflash::common {

FreezePlan plan_freeze(const std::vector<TurnSpan> & turns, int hot_window_turns) {
    if (turns.empty()) return {0, 0, 0, false};

    const int verbatim_end = turns[0].end_tok;

    // Need: 1 system + at least 1 aged turn + hot_window_turns hot turns.
    const bool has_frozen = (int)turns.size() >= 2 + hot_window_turns;
    if (!has_frozen) return {verbatim_end, verbatim_end, verbatim_end, false};

    const int frozen_begin     = turns[1].begin_tok;
    const int last_frozen_idx  = (int)turns.size() - 2 - hot_window_turns;
    const int frozen_end       = turns[last_frozen_idx].end_tok;
    // Invariant: system never inside frozen (turns[1].begin_tok >= turns[0].end_tok).
    return {verbatim_end, frozen_begin, frozen_end, true};
}

PrefixHash frozen_block_key(const int32_t * ids, int begin, int end) {
    if (begin >= end) { PrefixHash h{}; return h; }
    return hash_prefix(ids + begin, end - begin);
}

}  // namespace dflash::common
