#include "paged_kv_pool.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace dflash::common {

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

PagedKvStatus PagedKvPool::release(PagedKvSequenceHandle handle) {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    SequenceState & sequence = sequences_[handle.slot];
    request_to_slot_.erase(sequence.request_id);
    for (uint32_t block : sequence.block_table) {
        give_back(free_blocks_, block);
    }
    for (uint32_t block : sequence.reserved_blocks) {
        give_back(free_blocks_, block);
    }
    sequence.request_id = 0;
    sequence.kv_seq_len = 0;
    sequence.active = false;
    sequence.block_table.clear();
    sequence.reserved_blocks.clear();
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
    }
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
        static_cast<uint32_t>(sequence.reserved_blocks.size());
    out_sequence = std::move(snapshot);
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::owned_block_count(
        PagedKvSequenceHandle handle, uint32_t & out_count) const {
    const PagedKvStatus status = validate(handle);
    if (status != PagedKvStatus::Ok) return status;

    const SequenceState & sequence = sequences_[handle.slot];
    out_count = static_cast<uint32_t>(
        sequence.block_table.size() + sequence.reserved_blocks.size());
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
        (uint64_t)sequence.reserved_blocks.size() + free_blocks_.size();
    if (additional_blocks > available_blocks) {
        return PagedKvStatus::BlocksExhausted;
    }

    sequence.block_table.reserve(required_blocks);
    for (uint32_t i = 0; i < additional_blocks; ++i) {
        std::vector<uint32_t> & source = sequence.reserved_blocks.empty()
            ? free_blocks_ : sequence.reserved_blocks;
        sequence.block_table.push_back(take_lowest(source));
    }
    return PagedKvStatus::Ok;
}

void PagedKvPool::take_reserved_blocks(SequenceState & sequence,
                                       uint32_t additional_blocks) {
    sequence.reserved_blocks.reserve(
        sequence.reserved_blocks.size() + additional_blocks);
    for (uint32_t i = 0; i < additional_blocks; ++i) {
        give_back(sequence.reserved_blocks, take_lowest(free_blocks_));
    }
}

}  // namespace dflash::common
