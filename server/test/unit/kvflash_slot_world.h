// Shared setup for the slot-pager tests: slot pagers on a notional (no-tensor) arena, joined to a KvFlashArena
// when one is given.
#pragma once
#include "kvflash_pager_test_host.h"
#include "../../src/common/kvflash_arena.h"
#include "../../src/common/kvflash_slot_pager.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace luce::common {

// The arena with a replaceable host wait: tests count or skip the stream synchronize.
struct KvFlashTestArena : KvFlashArena {
    void set_host_wait(std::function<void()> f) { host_wait_ = std::move(f); }
};

struct KvFlashTestSlot : KvFlashSlotPager {
    KvFlashTestSlot() { set_pinned_host_backing(pager_pinned_host_available()); }
};

inline KvFlashConfig kvflash_test_config(int tail_window_chunks, int chunk_tokens = kKvFlashChunkTokens) {
    KvFlashConfig c;
    c.chunk_tokens = chunk_tokens;
    c.sink_chunks = 1;
    c.tail_window_chunks = tail_window_chunks;
    return c;
}

using KvFlashTestSlots = std::vector<std::unique_ptr<KvFlashTestSlot>>;

// n slot pagers as slices of `quota` chunks each; a refusal throws.
inline KvFlashTestSlots kvflash_test_slots(int n, int quota, const KvFlashConfig & cfg, KvFlashArena * arena = nullptr) {
    KvFlashTestSlots s;
    for (int i = 0; i < n; i++) {
        s.push_back(std::make_unique<KvFlashTestSlot>());
        if (arena && !arena->add(*s.back())) throw std::runtime_error("arena.add failed");
        if (!s.back()->attach_slice(i, n, quota, cfg, {}, {})) throw std::runtime_error("attach failed");
    }
    return s;
}

inline std::vector<KvFlashSlotPager *> kvflash_test_raw(const KvFlashTestSlots & s) {
    std::vector<KvFlashSlotPager *> raw;
    for (const auto & p : s) raw.push_back(p.get());
    return raw;
}

}  // namespace luce::common
