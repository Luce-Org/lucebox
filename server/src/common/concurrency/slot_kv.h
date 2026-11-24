// SlotKv v2: backend-agnostic KV allocator + SlotStateAdapter contracts shared
// by SeqSlotManager, PoolSlotKv, Inkling2SlotKv, and future KvFlashSessionRuntime.

#pragma once

#include "common/concurrency/paged_kv_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::common {

// Explicit admission options replacing out-of-band one-shot setters.
struct SlotAcquireOpts {
    int     preferred_slot = -1;   // >= 0: acquire this specific slot (sticky/cold reuse)
    int64_t resume_len     = -1;   // >= 0: resume parked KV in preferred_slot at this length
    int64_t total_tokens   = 0;    // prompt + generation bound for host-ledger reservation
};

// Block-table patch emitted by any SlotKv allocator for a slot's device column.
// - PoolSlotKv:     contiguous [0, blocks.size()) logical -> physical blocks
// - Inkling2SlotKv: full logical column [0, entries) with -1 holes for evicted / peer-GPU chunks
struct SlotTablePatch {
    int                  first_block = 0;
    std::vector<int32_t> blocks;
    bool                 empty() const { return blocks.empty(); }
};

// Static allocator capabilities (replaces concrete-type null checks in SeqEngine loops).
struct SlotKvCaps {
    bool parkable_offload        = true;   // supports detach_kv / attach_kv / evict_for_recompute
    bool sticky_sessions         = false;  // supports park() + resume_len in acquire_reserved()
    bool mutates_existing_blocks = false;  // eviction/recall can mutate already-written logical blocks
    bool supports_prefix_store   = true;   // compatible with PrefixStore snapshots
    bool dual_tier_tables        = false;  // emits disjoint local (Tier 0) + remote (Tier 1.5) patches
};

// Opaque serialized per-slot non-KV state (e.g. GDN+conv slab for Qwen3.5,
// SWA-1024 + ShortConv + Shared-KV W-tail slab for Inkling 2).
using SlotStateSlab = std::vector<uint8_t>;

// Backend-agnostic 4-method per-slot state adapter for KvFlashSessionRuntime & disk bundles.
struct SlotStateAdapter {
    virtual ~SlotStateAdapter() = default;

    virtual size_t slab_bytes() const = 0;
    virtual bool   save_slot_state(int slot, SlotStateSlab & out) const = 0;
    virtual bool   restore_slot_state(int slot, const SlotStateSlab & slab) = 0;
    virtual void   reset_slot_state(int slot) = 0;
    virtual int    preamble_end(const std::vector<int32_t> & prompt) const {
        (void) prompt;
        return 0;
    }
};

class SlotKv {
public:
    virtual ~SlotKv() = default;

    virtual uint32_t   block_size() const = 0;
    virtual uint32_t   attn_block_size() const { return block_size(); }
    virtual uint32_t   physical_block_count() const = 0;
    virtual uint32_t   max_sequences() const = 0;
    virtual uint32_t   free_block_count() const = 0;
    virtual SlotKvCaps caps() const { return SlotKvCaps{}; }
    bool               parkable() const { return caps().parkable_offload; }

    virtual PagedKvStatus acquire_reserved(PagedKvRequestId request_id, uint32_t token_capacity,
                                           PagedKvSequenceHandle & out_handle) = 0;
    virtual PagedKvStatus acquire_reserved(PagedKvRequestId request_id, uint32_t token_capacity,
                                           const SlotAcquireOpts & opts,
                                           PagedKvSequenceHandle & out_handle) {
        (void) opts;
        return acquire_reserved(request_id, token_capacity, out_handle);
    }

    virtual PagedKvStatus       reserve_capacity(PagedKvSequenceHandle handle, uint32_t token_capacity) = 0;
    virtual PagedKvAppendResult append(PagedKvSequenceHandle handle, uint32_t token_count,
                                       bool only_first_last_slots = false) = 0;
    virtual PagedKvStatus       rollback_append(PagedKvSequenceHandle handle, uint32_t token_count) = 0;
    virtual PagedKvStatus       clear(PagedKvSequenceHandle handle) = 0;
    virtual PagedKvStatus       release(PagedKvSequenceHandle handle) = 0;
    virtual PagedKvStatus       park(PagedKvSequenceHandle handle) { return release(handle); }
    virtual PagedKvStatus       owned_block_count(PagedKvSequenceHandle handle, uint32_t & out_count) const = 0;

    // Unified block-table patch builder (remote=true requests Tier-1.5 disjoint table when supported).
    virtual bool build_table_patch(PagedKvSequenceHandle handle, uint32_t token_extent,
                                   uint32_t max_attn_blocks, SlotTablePatch & out,
                                   bool remote = false) const {
        (void) handle; (void) token_extent; (void) max_attn_blocks; (void) remote;
        out = {};
        return false;
    }
};

// Owns nothing: the pool must outlive the adapter.
class PoolSlotKv final : public SlotKv {
public:
    explicit PoolSlotKv(PagedKvPool & pool) : pool_(pool) {}
    uint32_t block_size() const override { return pool_.block_size(); }
    uint32_t physical_block_count() const override { return pool_.physical_block_count(); }
    uint32_t max_sequences() const override { return pool_.max_sequences(); }
    uint32_t free_block_count() const override { return pool_.free_block_count(); }
    using SlotKv::acquire_reserved;
    PagedKvStatus acquire_reserved(PagedKvRequestId id, uint32_t cap, PagedKvSequenceHandle & h) override {
        return pool_.acquire_reserved(id, cap, h);
    }
    PagedKvStatus reserve_capacity(PagedKvSequenceHandle h, uint32_t cap) override {
        return pool_.reserve_capacity(h, cap);
    }
    PagedKvAppendResult append(PagedKvSequenceHandle h, uint32_t n, bool first_last) override {
        return pool_.append(h, n, first_last);
    }
    PagedKvStatus rollback_append(PagedKvSequenceHandle h, uint32_t n) override {
        return pool_.rollback_append(h, n);
    }
    PagedKvStatus clear(PagedKvSequenceHandle h) override { return pool_.clear(h); }
    PagedKvStatus release(PagedKvSequenceHandle h) override { return pool_.release(h); }
    PagedKvStatus owned_block_count(PagedKvSequenceHandle h, uint32_t & n) const override {
        return pool_.owned_block_count(h, n);
    }
    bool build_table_patch(PagedKvSequenceHandle h, uint32_t token_extent,
                           uint32_t max_attn_blocks, SlotTablePatch & out,
                           bool remote = false) const override {
        out = {};
        if (remote) return false;
        PagedKvSequenceSnapshot snap;
        if (pool_.sequence(h, snap) != PagedKvStatus::Ok) return false;
        const uint32_t bs = std::max<uint32_t>(1, pool_.block_size());
        const uint32_t need = (token_extent + bs - 1) / bs;
        if (need > max_attn_blocks || need > snap.block_table.size()) return false;
        out.first_block = 0;
        out.blocks.resize(need);
        for (uint32_t i = 0; i < need; ++i) out.blocks[i] = (int32_t) snap.block_table[i];
        return true;
    }
private:
    PagedKvPool & pool_;
};

}  // namespace luce::common
