// SlotKv: what SeqSlotManager needs from a KV allocator, no more than the
// PagedKvPool calls it makes. PoolSlotKv is the PagedKvPool adapter, the only
// allocator today; a KVFlash arena implements the same interface over
// per-slot slices.

#pragma once

#include "common/concurrency/paged_kv_pool.h"

namespace luce::common {

class SlotKv {
public:
    virtual ~SlotKv() = default;
    virtual uint32_t block_size() const = 0;
    virtual uint32_t physical_block_count() const = 0;
    virtual uint32_t max_sequences() const = 0;
    virtual uint32_t free_block_count() const = 0;
    virtual PagedKvStatus acquire_reserved(PagedKvRequestId request_id, uint32_t token_capacity,
                                           PagedKvSequenceHandle & out_handle) = 0;
    virtual PagedKvStatus reserve_capacity(PagedKvSequenceHandle handle, uint32_t token_capacity) = 0;
    virtual PagedKvAppendResult append(PagedKvSequenceHandle handle, uint32_t token_count,
                                       bool only_first_last_slots = false) = 0;
    virtual PagedKvStatus rollback_append(PagedKvSequenceHandle handle, uint32_t token_count) = 0;
    virtual PagedKvStatus clear(PagedKvSequenceHandle handle) = 0;
    virtual PagedKvStatus release(PagedKvSequenceHandle handle) = 0;
    virtual PagedKvStatus owned_block_count(PagedKvSequenceHandle handle, uint32_t & out_count) const = 0;
};

// Owns nothing: the pool must outlive the adapter.
class PoolSlotKv final : public SlotKv {
public:
    explicit PoolSlotKv(PagedKvPool & pool) : pool_(pool) {}
    uint32_t block_size() const override { return pool_.block_size(); }
    uint32_t physical_block_count() const override { return pool_.physical_block_count(); }
    uint32_t max_sequences() const override { return pool_.max_sequences(); }
    uint32_t free_block_count() const override { return pool_.free_block_count(); }
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
private:
    PagedKvPool & pool_;
};

}  // namespace luce::common
