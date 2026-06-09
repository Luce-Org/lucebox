// freeze_history — pure partition logic for FlowKV freeze-history feature.
//
// Partitions a token stream into three regions by turn boundary:
//   VERBATIM PREFIX : turns[0] (system + tool-defs) — never compressed.
//   FROZEN region   : aged conversational/tool turns after the system prefix,
//                     up to the hot window — compressed once and cached.
//   HOT TAIL        : the last hot_window_turns turns — kept verbatim.
//
// Pure functions: no IO, no globals, no CUDA deps. Tested standalone.

#pragma once

#include "server/prefix_cache.h"  // PrefixHash

#include <cstdint>
#include <vector>

namespace dflash::common {

// ─── Data types ───────────────────────────────────────────────────────────

struct TurnSpan {
    int  begin_tok;  // first token index of this turn (inclusive)
    int  end_tok;    // one-past-last token index of this turn
    bool is_system;  // true for the leading system / tool-defs turn
};

struct FreezePlan {
    int  verbatim_prefix_end;  // = turns[0].end_tok
    int  frozen_begin;         // = turns[1].begin_tok  (0 when has_frozen=false)
    int  frozen_end;           // = turns[N-1-hot_window].end_tok (0 when has_frozen=false)
    bool has_frozen;           // false when stream is too short to freeze anything
};

// ─── Pure functions ───────────────────────────────────────────────────────

// Partition `turns` into verbatim-prefix / frozen / hot-tail regions.
//
// Rules:
//   verbatim_prefix_end = turns[0].end_tok  (system turn kept verbatim)
//   frozen              = turns[1 .. N-1-hot_window_turns]
//   hot tail            = the last hot_window_turns turns (implied by frozen_end)
//
// has_frozen = false when:
//   - turns is empty
//   - turns has fewer than (1 system + hot_window_turns + 1 aged) turns
//     i.e. turns.size() < 2 + hot_window_turns
FreezePlan plan_freeze(const std::vector<TurnSpan> & turns, int hot_window_turns);

// Compute a stable content-hash of the frozen token slice [begin, end).
// Reuses hash_prefix from prefix_cache so no SHA-1 is re-implemented here.
//
// Returns a zeroed PrefixHash when the slice is empty (begin >= end).
PrefixHash frozen_block_key(const int32_t * ids, int begin, int end);

}  // namespace dflash::common
