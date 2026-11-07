// KVFlash 2.0 sessions: the admit decision table, pure. The engine computes the resume pick, the resume point, the
// preamble decision and which parked instance (if any) is the request's own sid; this turns them into one action. A
// request that cannot resume its parked instance is a compaction (or a changed preamble): the instance is archived,
// never rewound, and the system+tools come from the preamble store.
#pragma once

#include "kvflash_session_store.h"

#include <string>
#include <vector>

namespace luce::common {

// What the admit does with this prompt's preamble: attach the stored entry, or cold-admit and capture one; plain cold when the
// preamble is too short. `k` is a count of chunks. The preamble store that decides it is a later change; until then the default is Cold.
struct KvFlashPreambleDecision {
    enum Kind { Cold, Capture, Attach } kind = Cold;
    int k = 0;
};

enum class KvFlashAdmitAction {
    resume,          // row 1: resume the parked instance at L
    resume_disk,     // row 2: the same, the instance was just restored from disk
    summary_cold,    // row 3: E_role == 0 under a parked sid: cold in another slot, `slot` stays, nothing archived
    archive_attach,  // row 4: archive `slot`, attach the preamble entry (L = 64k), prefill the rest
    archive_cold,    // row 5: archive `slot`, cold, capture at capture_at (0: no capture)
    attach,          // row 6: attach (a new instance), L = 64k
    capture,         // rows 7-8: cold, capture at capture_at
    cold,            // row 9
};

struct KvFlashAdmitIn {
    int pick_slot = -1, lcp = 0, L = -1;         // the resume pick, its shared prefix with the prompt, and kvflash_resume_point for it (-1: none)
    bool from_disk = false;                      // the pick was restored from disk for this request
    int e_role = 0;                              // end of the prompt's system block (0: none)
    int same_sid_slot = -1;                      // a parked instance of the request's explicit sid, or -1
    std::string same_sid_key;                    // its directory key ("" without persistence)
    bool can_archive = true;                     // persistence is on: an instance can leave for disk and be linked
    KvFlashPreambleDecision pd;                  // attach / capture / cold from the preamble store
    int ct = 64;                                 // chunk tokens
};

struct KvFlashAdmitPlan {
    KvFlashAdmitAction action = KvFlashAdmitAction::cold;
    int slot = -1;                // resume: the slot; summary_cold: the slot to keep; archive_*: the slot to archive;
                                  // without persistence: the request's own session it replaced, or -1
    int L = 0;                    // resume point, or the attach point 64k
    int capture_at = 0;           // cold with a capture cut at this token (0: none)
    std::string archive_key;      // archive_*: the instance's key; its successor links to it
    bool archives() const { return action == KvFlashAdmitAction::archive_attach || action == KvFlashAdmitAction::archive_cold; }
};

inline KvFlashAdmitPlan kvflash_admit_plan(const KvFlashAdmitIn & in) {
    using A = KvFlashAdmitAction;
    KvFlashAdmitPlan p;
    if (in.pick_slot >= 0 && in.lcp > in.e_role && in.L > 0) {   // row 1: the prompt extends the session past the preamble and a resume point is saved
        p.action = in.from_disk ? A::resume_disk : A::resume; p.slot = in.pick_slot; p.L = in.L;
        return p;
    }
    const bool same = in.same_sid_slot >= 0;
    if (same && in.e_role == 0) { p.action = A::summary_cold; p.slot = in.same_sid_slot; return p; }
    const bool archive = same && in.can_archive;
    if (archive) { p.slot = in.same_sid_slot; p.archive_key = in.same_sid_key; }
    // Without persistence nothing is archived. The same system block under the same sid, and no resume: a compaction or an edit.
    // That history will not be extended again: `slot` names it as the first to leave. Another system block (lcp < e_role) is a
    // side call: the session stays as it is.
    else if (in.pick_slot >= 0 && in.pick_slot == in.same_sid_slot && in.lcp >= in.e_role) p.slot = in.pick_slot;
    if (in.pd.kind == KvFlashPreambleDecision::Attach) {
        p.action = archive ? A::archive_attach : A::attach; p.L = in.pd.k * in.ct;
    } else {
        p.capture_at = in.pd.kind == KvFlashPreambleDecision::Capture ? in.pd.k * in.ct : 0;
        p.action = archive ? A::archive_cold : p.capture_at > 0 ? A::capture : A::cold;
    }
    return p;
}

inline const char * kvflash_admit_row(KvFlashAdmitAction a) {
    switch (a) {
        case KvFlashAdmitAction::resume: return "1 resume";
        case KvFlashAdmitAction::resume_disk: return "2 resume-from-disk";
        case KvFlashAdmitAction::summary_cold: return "3 summary-cold";
        case KvFlashAdmitAction::archive_attach: return "4 archive-attach";
        case KvFlashAdmitAction::archive_cold: return "5 archive-cold";
        case KvFlashAdmitAction::attach: return "6 attach";
        case KvFlashAdmitAction::capture: return "7-8 capture";
        case KvFlashAdmitAction::cold: return "9 cold";
    }
    return "?";
}

// The parked instance of an explicit (client, sid): the pick's slot when it is one, else the newest. -1 for an
// anonymous request (hash / anon identities are stateless prefix reuse, they own nothing to archive).
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
