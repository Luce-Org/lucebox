// KVFlash 2.0 sessions: the admit decision table, pure. The engine computes the resume pick, the resume point and which parked
// instance (if any) is the request's own sid; this turns them into one action. There is no storage tier or preamble store yet:
// a session that cannot resume is dropped by the engine and the request admits cold. Rows that need them (archive, attach, capture)
// come back with that change.
#pragma once

#include "kvflash_session_store.h"

#include <string>
#include <vector>

namespace luce::common {

enum class KvFlashAdmitAction {
    resume,          // row 1: resume the parked instance at L
    summary_cold,    // row 3: E_role == 0 under a parked sid: cold in another slot, `slot` stays
    cold,            // row 9
};

struct KvFlashAdmitIn {
    int pick_slot = -1, lcp = 0, L = -1;         // the resume pick, its shared prefix with the prompt, and kvflash_resume_point for it (-1: none)
    int e_role = 0;                              // end of the prompt's system block (0: none)
    int same_sid_slot = -1;                      // a parked instance of the request's explicit sid, or -1
};

struct KvFlashAdmitPlan {
    KvFlashAdmitAction action = KvFlashAdmitAction::cold;
    int slot = -1;                // resume: the slot; summary_cold: the slot to keep
    int L = 0;                    // resume point
};

inline KvFlashAdmitPlan kvflash_admit_plan(const KvFlashAdmitIn & in) {
    using A = KvFlashAdmitAction;
    KvFlashAdmitPlan p;
    if (in.pick_slot >= 0 && in.lcp > in.e_role && in.L > 0) {   // row 1: the prompt extends the session past the preamble and a resume point is saved
        p.action = A::resume; p.slot = in.pick_slot; p.L = in.L;
    } else if (in.same_sid_slot >= 0 && in.e_role == 0) {
        p.action = A::summary_cold; p.slot = in.same_sid_slot;
    }
    return p;
}

inline const char * kvflash_admit_row(KvFlashAdmitAction a) {
    switch (a) {
        case KvFlashAdmitAction::resume: return "1 resume";
        case KvFlashAdmitAction::summary_cold: return "3 summary-cold";
        case KvFlashAdmitAction::cold: return "9 cold";
    }
    return "?";
}

// The parked instance of an explicit (client, sid): the pick's slot when it is one, else the newest. -1 for an
// anonymous request (hash / anon identities are stateless prefix reuse, they own nothing to keep).
inline int kvflash_same_sid_slot(const std::vector<KvFlashOwnedSession> & parked, const std::string & client, const std::string & sid, int pick_slot) {
    if (!kvflash_client_explicit(client)) return -1;
    int best = -1;
    uint64_t best_tick = 0;
    for (const auto & o : parked) {
        if (o.client != client || o.sid != sid) continue;
        if (o.parked.slot == pick_slot) return pick_slot;
        if (best < 0 || o.parked.tick > best_tick) { best = o.parked.slot; best_tick = o.parked.tick; }
    }
    return best;
}

// The oldest parked session other than `keep`, for a cold admit. A summary-cold request keeps its sid's parked slot, but when that is
// the only evictable one and nothing is live, no retry can change the answer: the keep is given up rather than answered busy forever.
// evict(keep) returns the freed slot or -1.
template <class Evict>
inline int kvflash_evict_for_cold(Evict evict, int keep, bool any_live) {
    const int slot = evict(keep);
    return slot < 0 && keep >= 0 && !any_live ? evict(-1) : slot;
}

}  // namespace luce::common
