// KVFlash host backing policy (default: Auto — tries pinned DMA, falls back to pageable on refusal).
//
// Evicted and restored KV chunks live in host memory. Pinned memory (cudaMallocHost, hipHostMalloc under HIP)
// gives the DMA its speed but can be refused (a locked-memory limit, a stub driver, a UMA part under combined
// pressure); pageable memory always works and is slower. `Auto` tries pinned and, on the first refusal,
// switches that pager to pageable and logs it. Pure: the pager owns the allocation calls.
#pragma once

#include <string>

namespace luce::common {

enum class KvFlashHostBacking { Auto, Pinned, Pageable };

struct KvFlashBackingDecision {
    bool retry_pageable = false;   // the caller allocates the buffer pageable instead
    std::string log;               // one line for stderr, empty when nothing happened
};

class KvFlashBackingState {
public:
    explicit KvFlashBackingState(KvFlashHostBacking mode = KvFlashHostBacking::Auto)
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
                    "), no fallback";
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
