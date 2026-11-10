// Pure planning of a DFlash2 chain round over KVFlash slots (Stage C-4). No GPU, no engine.
//
// A spec lane verifies `width` candidate tokens in scratch rows, but the accepted prefix is appended to the
// slot's own rows afterwards, so the pager must have those rows before the graph runs: a spec lane plans
// `width` rows from its committed length, an AR peer one. The graph reads only seq_len, yet the table column
// uploaded for the round must cover every chunk the plan allocated: a chunk opened by the chain's rows leaves
// the next step with nothing new to allocate, so the pager epoch does not move and nothing would re-upload it.
#pragma once

#include "kvflash_slot_pager.h"
#include "kvflash_step_driver.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace luce::common {

struct ChainLaneIn {
    int     slot = -1;
    int64_t cur_pos = 0;     // committed length of the slot
    bool    spec = false;    // a chain lane (root + drafts) rather than a compact AR row
    int     width = 1;       // verify width of a spec lane (>= 2); ignored for an AR lane
};

struct ChainTablePlan {
    bool valid = true;
    std::vector<KvFlashStepWork> work;                  // input of KvFlashStepDriver::run_step, in lane order
    std::vector<std::pair<int, int>> cover_len;         // {slot, kv_len the uploaded column must cover}
};

inline ChainTablePlan chain_table_plan(const std::vector<ChainLaneIn> & lanes, int max_ctx) {
    ChainTablePlan p;
    for (const ChainLaneIn & l : lanes) {
        const bool dup = std::any_of(p.work.begin(), p.work.end(), [&](const KvFlashStepWork & w) { return w.slot == l.slot; });
        if (l.slot < 0 || l.cur_pos < 0 || dup || (l.spec && l.width < 2)) { p.valid = false; return p; }
        // rows past max_ctx are never accepted; one row is always planned (an exhausted context is the engine's own error)
        const int room = (int)std::max<int64_t>(1, (int64_t)max_ctx - l.cur_pos);
        const int tokens = l.spec ? std::min(l.width, room) : 1;
        p.work.push_back({l.slot, l.cur_pos, tokens});
        p.cover_len.emplace_back(l.slot, (int)(l.cur_pos + tokens));
    }
    return p;
}

enum class ChainRoundVerdict { Chain, FallbackAr };

// BlocksExhausted is an AR round, never an error: a lane the pager did not run (its page-outs are queued and the
// AR step settles them) sends the whole round to the AR path, which has its own per-slot idle handling.
inline ChainRoundVerdict chain_round_verdict(const ChainTablePlan & plan, const std::vector<int> & ran) {
    if (!plan.valid) return ChainRoundVerdict::FallbackAr;
    for (const KvFlashStepWork & w : plan.work)
        if (std::find(ran.begin(), ran.end(), w.slot) == ran.end()) return ChainRoundVerdict::FallbackAr;
    return ChainRoundVerdict::Chain;
}

}  // namespace luce::common
