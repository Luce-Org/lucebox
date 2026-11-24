#include "paged_kv_pool.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace luce::common {

const char * paged_kv_status_string(PagedKvStatus status) {
    switch (status) {
        case PagedKvStatus::Ok:
            return "ok";
        case PagedKvStatus::InvalidArgument:
            return "invalid argument";
        case PagedKvStatus::DuplicateRequest:
            return "duplicate request";
        case PagedKvStatus::SequenceSlotsExhausted:
            return "sequence slots exhausted";
        case PagedKvStatus::BlocksExhausted:
            return "physical blocks exhausted";
        case PagedKvStatus::StaleHandle:
            return "stale sequence handle";
    }
    return "unknown paged KV status";
}

uint32_t PagedKvPool::take_lowest(std::vector<uint32_t> & free_list) {
    const uint32_t index = free_list.front();
    std::pop_heap(free_list.begin(), free_list.end(), std::greater<>());
    free_list.pop_back();
    return index;
}

void PagedKvPool::give_back(std::vector<uint32_t> & free_list,
                            uint32_t index) {
    free_list.push_back(index);
    std::push_heap(free_list.begin(), free_list.end(), std::greater<>());
}

void PagedKvPool::refill(std::vector<uint32_t> & free_list, uint32_t count) {
    // Ascending order is already a valid min-heap, so the refilled list needs
    // no heapify pass.
    free_list.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        free_list[i] = i;
    }
}

PagedKvPool::PagedKvPool(uint32_t physical_block_count,
                         uint32_t max_sequences,
                         uint32_t block_size)
    : block_size_(block_size),
      physical_block_count_(physical_block_count) {
    const uint64_t token_capacity =
        static_cast<uint64_t>(physical_block_count) * block_size;
    if (physical_block_count == 0 || max_sequences == 0 || block_size == 0 ||
        token_capacity > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("invalid paged KV pool dimensions");
    }

    sequences_.resize(max_sequences);
    refill(free_sequence_slots_, max_sequences);
    refill(free_blocks_, physical_block_count);
    seq_refcounts_.assign(physical_block_count, 0);
    pin_refcounts_.assign(physical_block_count, 0);
}

uint32_t PagedKvPool::take_free_block() {
    const uint32_t block = take_lowest(free_blocks_);
    seq_refcounts_[block] = 1;
    return block;
}

void PagedKvPool::add_seq_ref(uint32_t block) {
    if (seq_refcounts_[block] == 0 && pin_refcounts_[block] > 0) {
        --pinned_only_blocks_;
    }
    ++seq_refcounts_[block];
}

void PagedKvPool::release_seq_ref(uint32_t block) {
    if (seq_refcounts_[block] == 0) return;
    if (--seq_refcounts_[block] == 0) {
        if (pin_refcounts_[block] > 0) {
            ++pinned_only_blocks_;
        } else {
            give_back(free_blocks_, block);
        }
    }
}

void PagedKvPool::add_pin_ref(uint32_t block) {
    if (seq_refcounts_[block] == 0 && pin_refcounts_[block] == 0) {
        ++pinned_only_blocks_;
    }
    ++pin_refcounts_[block];
}

void PagedKvPool::release_pin_ref(uint32_t block) {
    if (pin_refcounts_[block] == 0) return;
    if (--pin_refcounts_[block] == 0 && seq_refcounts_[block] == 0) {
        --pinned_only_blocks_;
        give_back(free_blocks_, block);
    }
}

PagedKvStatus PagedKvPool::acquire(PagedKvRequestId request_id,
                                   PagedKvSequenceHandle & out_handle) {
    return acquire_reserved(request_id, 0, out_handle);
}

PagedKvStatus PagedKvPool::acquire_reserved(
        PagedKvRequestId request_id, uint32_t token_capacity,
        PagedKvSequenceHandle & out_handle) {
    if (request_to_slot_.find(request_id) != request_to_slot_.end()) {
        return PagedKvStatus::DuplicateRequest;
    }
    if (free_sequence_slots_.empty()) {
        return PagedKvStatus::SequenceSlotsExhausted;
    }

    const uint32_t reserve_blocks = blocks_for_tokens(token_capacity);
    if (reserve_blocks > free_blocks_.size()) {
        return PagedKvStatus::BlocksExhausted;
    }

    const uint32_t slot = take_lowest(free_sequence_slots_);
    SequenceState & sequence = sequences_[slot];
    uint64_t generation = sequence.generation + 1;
    if (generation == 0) generation = 1;

    // release(), reset(), and construction leave every free slot empty;
    // acquire only installs its new identity.
    sequence.request_id = request_id;
    sequence.generation = generation;
    sequence.active = true;
    take_reserved_blocks(sequence, reserve_blocks);
    request_to_slot_.emplace(request_id, slot);

    out_handle = {slot, generation};
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::reserve_capacity(
        PagedKvSequenceHandle handle, uint32_t token_capacity) {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    SequenceState & sequence = sequences_[handle.slot];
    const uint32_t target_blocks = blocks_for_tokens(token_capacity);
    const uint64_t owned_blocks =
        (uint64_t)sequence.block_table.size() +
        sequence.retry_blocks.size() +
        sequence.reserved_blocks.size();
    if (target_blocks <= owned_blocks) return PagedKvStatus::Ok;

    const uint32_t additional_blocks =
        target_blocks - static_cast<uint32_t>(owned_blocks);
    if (additional_blocks > free_blocks_.size()) {
        return PagedKvStatus::BlocksExhausted;
    }
    take_reserved_blocks(sequence, additional_blocks);
    return PagedKvStatus::Ok;
}

PagedKvAppendResult PagedKvPool::append(PagedKvSequenceHandle handle,
                                        uint32_t token_count,
                                        bool only_first_last_slots) {
    PagedKvAppendResult result;
    result.status = validate(handle);
    if (result.status != PagedKvStatus::Ok || token_count == 0) {
        return result;
    }
    SequenceState & sequence = sequences_[handle.slot];
    if (token_count > std::numeric_limits<uint32_t>::max() -
                          sequence.kv_seq_len) {
        result.status = PagedKvStatus::InvalidArgument;
        return result;
    }

    const uint32_t old_kv_seq_len = sequence.kv_seq_len;
    // Copy-on-write invariant: never append into a partially filled shared block.
    if (old_kv_seq_len % block_size_ != 0) {
        const uint32_t partial_block =
            sequence.block_table[old_kv_seq_len / block_size_];
        if (is_block_shared(partial_block)) {
            result.status = PagedKvStatus::InvalidArgument;
            return result;
        }
    }
    const uint32_t new_kv_seq_len = old_kv_seq_len + token_count;
    result.status =
        extend_block_table(sequence, blocks_for_tokens(new_kv_seq_len));
    if (result.status != PagedKvStatus::Ok) return result;

    const auto make_slot = [&](uint32_t logical_position) {
        const uint32_t logical_block = logical_position / block_size_;
        const uint32_t block_offset = logical_position % block_size_;
        const uint32_t physical_block = sequence.block_table[logical_block];
        return PagedKvWriteSlot{
            logical_position,
            physical_block,
            block_offset,
            static_cast<uint64_t>(physical_block) * block_size_ + block_offset,
        };
    };

    result.token_count = token_count;
    if (only_first_last_slots) {
        result.first = make_slot(old_kv_seq_len);
        result.last = make_slot(new_kv_seq_len - 1);
    } else {
        result.write_slots.reserve(token_count);
        for (uint32_t i = 0; i < token_count; ++i) {
            result.write_slots.push_back(make_slot(old_kv_seq_len + i));
        }
    }

    sequence.kv_seq_len = new_kv_seq_len;
    return result;
}

PagedKvStatus PagedKvPool::rollback_append(
        PagedKvSequenceHandle handle, uint32_t token_count) {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    SequenceState & sequence = sequences_[handle.slot];
    if (token_count > sequence.kv_seq_len) {
        return PagedKvStatus::InvalidArgument;
    }
    const uint32_t restored_length = sequence.kv_seq_len - token_count;
    const uint32_t restored_blocks = blocks_for_tokens(restored_length);
    // Reject rewinding across or into a shared block so shared prefix pages
    // remain immutable and never enter a sequence's private retry stack.
    for (size_t i = restored_blocks; i < sequence.block_table.size(); ++i) {
        if (is_block_shared(sequence.block_table[i])) {
            return PagedKvStatus::InvalidArgument;
        }
    }
    if (restored_length % block_size_ != 0 && restored_blocks > 0 &&
        is_block_shared(sequence.block_table[restored_blocks - 1])) {
        return PagedKvStatus::InvalidArgument;
    }
    while (sequence.block_table.size() > restored_blocks) {
        const uint32_t block = sequence.block_table.back();
        sequence.block_table.pop_back();
        sequence.retry_blocks.push_back(block);
    }
    sequence.kv_seq_len = restored_length;
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::clear(PagedKvSequenceHandle handle) {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;
    SequenceState & sequence = sequences_[handle.slot];
    for (uint32_t block : sequence.block_table) release_seq_ref(block);
    for (uint32_t block : sequence.reserved_blocks) release_seq_ref(block);
    for (uint32_t block : sequence.retry_blocks) release_seq_ref(block);
    sequence.kv_seq_len = 0;
    sequence.block_table.clear();
    sequence.reserved_blocks.clear();
    sequence.retry_blocks.clear();
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::release(PagedKvSequenceHandle handle) {
    const PagedKvStatus status = clear(handle);
    if (status != PagedKvStatus::Ok) return status;
    SequenceState & sequence = sequences_[handle.slot];
    request_to_slot_.erase(sequence.request_id);
    sequence.request_id = 0;
    sequence.active = false;
    give_back(free_sequence_slots_, handle.slot);
    return PagedKvStatus::Ok;
}


void PagedKvPool::reset() {
    request_to_slot_.clear();

    for (SequenceState & sequence : sequences_) {
        sequence.request_id = 0;
        sequence.kv_seq_len = 0;
        sequence.active = false;
        sequence.block_table.clear();
        sequence.reserved_blocks.clear();
        sequence.retry_blocks.clear();
    }
    std::fill(seq_refcounts_.begin(), seq_refcounts_.end(), 0u);
    std::fill(pin_refcounts_.begin(), pin_refcounts_.end(), 0u);
    pinned_only_blocks_ = 0;
    refill(free_sequence_slots_, static_cast<uint32_t>(sequences_.size()));
    refill(free_blocks_, physical_block_count_);
}

PagedKvStatus PagedKvPool::sequence(
    PagedKvSequenceHandle handle,
    PagedKvSequenceSnapshot & out_sequence) const {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    const SequenceState & sequence = sequences_[handle.slot];
    PagedKvSequenceSnapshot snapshot;
    snapshot.kv_seq_len = sequence.kv_seq_len;
    snapshot.block_table = sequence.block_table;
    snapshot.reserved_block_count =
        static_cast<uint32_t>(
            sequence.retry_blocks.size() + sequence.reserved_blocks.size());
    out_sequence = std::move(snapshot);
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::owned_block_count(
        PagedKvSequenceHandle handle, uint32_t & out_count) const {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    const SequenceState & sequence = sequences_[handle.slot];
    out_count = static_cast<uint32_t>(
        sequence.block_table.size() + sequence.retry_blocks.size() +
        sequence.reserved_blocks.size());
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::exclusive_block_count(
        PagedKvSequenceHandle handle, uint32_t & out_count) const {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    const SequenceState & sequence = sequences_[handle.slot];
    uint32_t exclusive = 0;
    auto count_exclusive = [&](const std::vector<uint32_t> & blocks) {
        for (uint32_t b : blocks) {
            if (b < physical_block_count_ &&
                seq_refcounts_[b] == 1 && pin_refcounts_[b] == 0) {
                ++exclusive;
            }
        }
    };
    count_exclusive(sequence.block_table);
    count_exclusive(sequence.retry_blocks);
    count_exclusive(sequence.reserved_blocks);
    out_count = exclusive;
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::pin_prefix_blocks(
        PagedKvSequenceHandle handle, uint32_t full_block_count,
        std::vector<uint32_t> & out_blocks) {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    const SequenceState & sequence = sequences_[handle.slot];
    const uint32_t available_full = sequence.kv_seq_len / block_size_;
    if (full_block_count > available_full ||
        full_block_count > sequence.block_table.size()) {
        return PagedKvStatus::InvalidArgument;
    }
    out_blocks.clear();
    out_blocks.reserve(full_block_count);
    for (uint32_t i = 0; i < full_block_count; ++i) {
        const uint32_t block = sequence.block_table[i];
        add_pin_ref(block);
        out_blocks.push_back(block);
    }
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::unpin_prefix_blocks(
        const std::vector<uint32_t> & blocks) {
    for (uint32_t block : blocks) {
        if (block >= physical_block_count_ || pin_refcounts_[block] == 0) {
            return PagedKvStatus::InvalidArgument;
        }
    }
    for (uint32_t block : blocks) {
        release_pin_ref(block);
    }
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::seed_shared_prefix(
        PagedKvSequenceHandle handle,
        const uint32_t * shared_blocks,
        uint32_t shared_block_count,
        uint32_t token_count) {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    SequenceState & sequence = sequences_[handle.slot];
    if (token_count == 0 || sequence.kv_seq_len != 0 ||
        !sequence.block_table.empty() || !sequence.retry_blocks.empty()) {
        return PagedKvStatus::InvalidArgument;
    }
    if (shared_block_count > token_count / block_size_ ||
        (shared_block_count > 0 && !shared_blocks)) {
        return PagedKvStatus::InvalidArgument;
    }
    for (uint32_t i = 0; i < shared_block_count; ++i) {
        const uint32_t b = shared_blocks[i];
        if (b >= physical_block_count_ || block_refcount(b) == 0) {
            return PagedKvStatus::InvalidArgument;
        }
    }

    const uint32_t total_blocks = blocks_for_tokens(token_count);
    const uint32_t private_tail = total_blocks - shared_block_count;
    const uint64_t avail_private =
        static_cast<uint64_t>(sequence.reserved_blocks.size()) +
        free_blocks_.size();
    if (private_tail > avail_private) {
        return PagedKvStatus::BlocksExhausted;
    }

    sequence.block_table.reserve(total_blocks);
    for (uint32_t i = 0; i < shared_block_count; ++i) {
        const uint32_t b = shared_blocks[i];
        add_seq_ref(b);
        sequence.block_table.push_back(b);
    }
    for (uint32_t i = 0; i < private_tail; ++i) {
        if (!sequence.reserved_blocks.empty()) {
            sequence.block_table.push_back(take_lowest(sequence.reserved_blocks));
        } else {
            sequence.block_table.push_back(take_free_block());
        }
    }
    sequence.kv_seq_len = token_count;
    return PagedKvStatus::Ok;
}

uint32_t PagedKvPool::blocks_for_tokens(uint32_t token_count) const {
    if (token_count == 0) return 0;
    return 1 + (token_count - 1) / block_size_;
}

PagedKvStatus PagedKvPool::validate(PagedKvSequenceHandle handle) const {
    if (handle.slot >= sequences_.size()) {
        return PagedKvStatus::StaleHandle;
    }
    const SequenceState & sequence = sequences_[handle.slot];
    if (!sequence.active || sequence.generation != handle.generation) {
        return PagedKvStatus::StaleHandle;
    }
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::extend_block_table(SequenceState & sequence,
                                              uint32_t required_blocks) {
    const uint32_t current_blocks =
        static_cast<uint32_t>(sequence.block_table.size());
    if (required_blocks <= current_blocks) return PagedKvStatus::Ok;

    const uint32_t additional_blocks = required_blocks - current_blocks;
    const uint64_t available_blocks =
        (uint64_t)sequence.retry_blocks.size() +
        sequence.reserved_blocks.size() + free_blocks_.size();
    if (additional_blocks > available_blocks) {
        return PagedKvStatus::BlocksExhausted;
    }

    sequence.block_table.reserve(required_blocks);
    for (uint32_t i = 0; i < additional_blocks; ++i) {
        if (!sequence.retry_blocks.empty()) {
            sequence.block_table.push_back(sequence.retry_blocks.back());
            sequence.retry_blocks.pop_back();
        } else if (!sequence.reserved_blocks.empty()) {
            sequence.block_table.push_back(take_lowest(sequence.reserved_blocks));
        } else {
            sequence.block_table.push_back(take_free_block());
        }
    }
    return PagedKvStatus::Ok;
}

void PagedKvPool::take_reserved_blocks(SequenceState & sequence,
                                       uint32_t additional_blocks) {
    sequence.reserved_blocks.reserve(
        sequence.reserved_blocks.size() + additional_blocks);
    for (uint32_t i = 0; i < additional_blocks; ++i) {
        give_back(sequence.reserved_blocks, take_free_block());
    }
}

}  // namespace luce::common
