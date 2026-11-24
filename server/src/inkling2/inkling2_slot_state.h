// Inkling2SlotStateAdapter — 4-method SlotStateAdapter packing/restoring all
// per-slot non-global-KV state (ShortConv states, SWA-1024 rings, Shared-KV W-tail
// rings, DSpark last_taps_dev, optional 2DSD sliding drafter ring) for KVFlash 2.0
// session offload, sticky sessions, and #1073 disk bundles.

#pragma once

#include "common/concurrency/slot_kv.h"
#include "inkling2_cache.h"
#include "inkling2_internal.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::common {

class Inkling2SlotStateAdapter final : public SlotStateAdapter {
public:
    static constexpr uint32_t kMagic   = 0x31424C53u; // "SLB1"
    static constexpr uint32_t kVersion = 1;

    Inkling2SlotStateAdapter(const Inkling2Weights & weights,
                             std::vector<Inkling2Cache *> caches,
                             bool include_draft_ring = false);

    size_t slab_bytes() const override;
    bool   save_slot_state(int slot, SlotStateSlab & out) const override;
    bool   restore_slot_state(int slot, const SlotStateSlab & slab) override;
    void   reset_slot_state(int slot) override;
    int    preamble_end(const std::vector<int32_t> & prompt) const override;

    int slot_count() const { return (int) caches_.size(); }
    void set_preamble_markers(int32_t message_user_id, int32_t message_system_id) {
        user_msg_id_   = message_user_id;
        system_msg_id_ = message_system_id;
    }

private:
    bool valid_slot_(int slot) const;

    const Inkling2Weights &      weights_;
    std::vector<Inkling2Cache *> caches_;
    bool                         include_draft_ring_ = false;
    int32_t                      user_msg_id_   = -1;
    int32_t                      system_msg_id_ = -1;
};

} // namespace luce::common
