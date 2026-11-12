// KVFlash host backing policy (LUCE_KVFLASH_HOST_BACKING = pinned | pageable | auto, default pinned).
//
// Evicted and restored KV chunks live in host memory. Pinned memory (cudaMallocHost, hipHostMalloc under HIP)
// gives the DMA its speed but can be refused (a locked-memory limit, a stub driver, a UMA part under combined
// pressure); pageable memory always works and is slower. `pinned` never falls back (the allocation fails and
// the page-out reports it), `pageable` never asks for pinned memory, `auto` tries pinned and, on the first
// refusal, switches that pager to pageable and says so. The reservation ledger counts bytes, not backing
// (kvflash_slot_pager.h), so a fallback leaves it unchanged. Pure: the pager owns the allocation calls.
#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace luce::common {

enum class KvFlashHostBacking { Auto, Pinned, Pageable };

// null/"" -> Pinned (the behaviour before the policy existed); an unknown value -> Pinned with *ok = false (the caller logs it).
inline KvFlashHostBacking kvflash_host_backing_parse(const char * s, bool * ok = nullptr) {
    if (ok) *ok = true;
    if (!s || !*s) return KvFlashHostBacking::Pinned;
    std::string v(s);
    for (char & c : v) c = (char)std::tolower((unsigned char)c);
    if (v == "auto") return KvFlashHostBacking::Auto;
    if (v == "pinned") return KvFlashHostBacking::Pinned;
    if (v == "pageable") return KvFlashHostBacking::Pageable;
    if (ok) *ok = false;
    return KvFlashHostBacking::Pinned;
}

inline KvFlashHostBacking kvflash_host_backing_from_env() {
    const char * e = std::getenv("LUCE_KVFLASH_HOST_BACKING");
    bool ok = true;
    const KvFlashHostBacking m = kvflash_host_backing_parse(e, &ok);
    if (!ok) std::fprintf(stderr, "[kvflash] LUCE_KVFLASH_HOST_BACKING=%s unknown (pinned|pageable|auto): using pinned\n", e);
    return m;
}

struct KvFlashBackingDecision {
    bool retry_pageable = false;   // the caller allocates the buffer pageable instead
    std::string log;               // one line for stderr, empty when nothing happened
};

class KvFlashBackingState {
public:
    explicit KvFlashBackingState(KvFlashHostBacking mode = KvFlashHostBacking::Pinned)
        : mode_(mode), pinned_(mode != KvFlashHostBacking::Pageable) {}

    KvFlashHostBacking mode() const { return mode_; }
    bool pinned() const { return pinned_; }        // what the next allocation asks for
    int fallbacks() const { return fallbacks_; }
    int failures() const { return failures_; }

    KvFlashBackingDecision on_pinned_failure(const char * why) {
        KvFlashBackingDecision d;
        if (mode_ == KvFlashHostBacking::Auto) {
            pinned_ = false;
            fallbacks_++;
            d.retry_pageable = true;
            d.log = std::string("[kvflash] host backing: pageable (pinned refused: ") + (why ? why : "?") + ")";
        } else {
            failures_++;
            d.log = std::string("[kvflash] host backing: pinned allocation failed (") + (why ? why : "?") +
                    "), LUCE_KVFLASH_HOST_BACKING=pinned: no fallback";
        }
        return d;
    }
    KvFlashBackingDecision on_pageable_failure() {
        failures_++;
        KvFlashBackingDecision d;
        d.log = "[kvflash] host backing: pageable allocation failed";
        return d;
    }

private:
    KvFlashHostBacking mode_;
    bool pinned_;
    int fallbacks_ = 0, failures_ = 0;
};

// Applies one mode to every slot pager (anything with set_host_backing), before the first restore or
// eviction allocation.
template <class PagerPtrRange>
inline void kvflash_apply_host_backing(PagerPtrRange & pagers, KvFlashHostBacking mode) {
    for (auto * p : pagers) p->set_host_backing(mode);
}

}  // namespace luce::common
