#include "paged_kv_pool.h"

#include <algorithm>
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
        case PagedKvStatus::RequestNotFound:
            return "request not found";
        case PagedKvStatus::StaleHandle:
            return "stale sequence handle";
    }
    return "unknown paged KV status";
}

PagedKvPool::PagedKvPool(uint32_t physical_block_count,
                         uint32_t max_sequences,
                         uint32_t block_size)
    : block_size_(block_size),
      physical_block_count_(physical_block_count),
      sequences_(max_sequences) {
    const uint64_t token_capacity =
        static_cast<uint64_t>(physical_block_count) * block_size;
    if (physical_block_count == 0 || max_sequences == 0 || block_size == 0 ||
        token_capacity > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("invalid paged KV pool dimensions");
    }

    for (uint32_t slot = 0; slot < max_sequences; ++slot) {
        free_sequence_slots_.insert(slot);
    }
    for (uint32_t block = 0; block < physical_block_count; ++block) {
        free_blocks_.insert(block);
    }
}

PagedKvStatus PagedKvPool::acquire(PagedKvRequestId request_id,
                                   PagedKvSequenceHandle & out_handle) {
    if (request_to_slot_.find(request_id) != request_to_slot_.end()) {
        return PagedKvStatus::DuplicateRequest;
    }
    if (free_sequence_slots_.empty()) {
        return PagedKvStatus::SequenceSlotsExhausted;
    }

    const uint32_t slot = *free_sequence_slots_.begin();
    SequenceState & sequence = sequences_[slot];
    uint64_t generation = sequence.generation + 1;
    if (generation == 0) generation = 1;

    // release_sequence(), reset(), and construction leave every free slot
    // empty; acquire only installs its new identity.
    sequence.request_id = request_id;
    sequence.generation = generation;
    sequence.active = true;
    request_to_slot_.emplace(request_id, slot);
    free_sequence_slots_.erase(free_sequence_slots_.begin());

    out_handle = {slot, generation};
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::lookup(PagedKvRequestId request_id,
                                  PagedKvSequenceHandle & out_handle) const {
    const auto it = request_to_slot_.find(request_id);
    if (it == request_to_slot_.end()) {
        return PagedKvStatus::RequestNotFound;
    }
    const SequenceState & sequence = sequences_[it->second];
    out_handle = {it->second, sequence.generation};
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::reserve(PagedKvSequenceHandle handle,
                                   uint32_t token_capacity) {
    SequenceState * sequence = nullptr;
    const PagedKvStatus status = validate(handle, sequence);
    if (status != PagedKvStatus::Ok) return status;
    return extend_block_table(*sequence, blocks_for_tokens(token_capacity));
}

PagedKvAppendResult PagedKvPool::append(PagedKvSequenceHandle handle,
                                        uint32_t token_count) {
    PagedKvAppendResult result;
    SequenceState * sequence = nullptr;
    result.status = validate(handle, sequence);
    if (result.status != PagedKvStatus::Ok || token_count == 0) {
        return result;
    }
    if (token_count > std::numeric_limits<uint32_t>::max() -
                          sequence->kv_seq_len) {
        result.status = PagedKvStatus::InvalidArgument;
        return result;
    }

    const uint32_t old_kv_seq_len = sequence->kv_seq_len;
    const uint32_t new_kv_seq_len = old_kv_seq_len + token_count;
    const uint32_t required_blocks = blocks_for_tokens(new_kv_seq_len);
    const uint32_t current_blocks =
        static_cast<uint32_t>(sequence->block_table.size());
    const uint32_t additional_blocks =
        required_blocks > current_blocks ? required_blocks - current_blocks : 0;
    if (additional_blocks > free_blocks_.size()) {
        result.status = PagedKvStatus::BlocksExhausted;
        return result;
    }

    // Reserve before mutating the pool. Once capacity is available,
    // PagedKvWriteSlot insertion cannot allocate or throw.
    result.write_slots.reserve(token_count);

    // extend_block_table() grows the block table before consuming free blocks,
    // so a failed vector allocation leaves the pool unchanged.
    result.status = extend_block_table(*sequence, required_blocks);
    if (result.status != PagedKvStatus::Ok) return result;

    for (uint32_t i = 0; i < token_count; ++i) {
        const uint32_t logical_position = old_kv_seq_len + i;
        const uint32_t logical_block = logical_position / block_size_;
        const uint32_t block_offset = logical_position % block_size_;
        const uint32_t physical_block = sequence->block_table[logical_block];
        result.write_slots.push_back({
            logical_position,
            physical_block,
            block_offset,
            static_cast<uint64_t>(physical_block) * block_size_ + block_offset,
        });
    }

    sequence->kv_seq_len = new_kv_seq_len;
    result.status = PagedKvStatus::Ok;
    return result;
}

PagedKvStatus PagedKvPool::release(PagedKvSequenceHandle handle) {
    SequenceState * sequence = nullptr;
    const PagedKvStatus status = validate(handle, sequence);
    if (status != PagedKvStatus::Ok) return status;
    release_sequence(handle.slot);
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::cancel(PagedKvRequestId request_id) {
    const auto it = request_to_slot_.find(request_id);
    if (it == request_to_slot_.end()) {
        return PagedKvStatus::RequestNotFound;
    }
    release_sequence(it->second);
    return PagedKvStatus::Ok;
}

void PagedKvPool::reset() {
    request_to_slot_.clear();
    free_sequence_slots_.clear();
    free_blocks_.clear();

    for (uint32_t slot = 0; slot < sequences_.size(); ++slot) {
        SequenceState & sequence = sequences_[slot];
        sequence.request_id = 0;
        sequence.kv_seq_len = 0;
        sequence.active = false;
        sequence.block_table.clear();
        free_sequence_slots_.insert(slot);
    }
    for (uint32_t block = 0; block < physical_block_count_; ++block) {
        free_blocks_.insert(block);
    }
}

PagedKvStatus PagedKvPool::sequence(
    PagedKvSequenceHandle handle,
    PagedKvSequenceSnapshot & out_sequence) const {
    const SequenceState * sequence = nullptr;
    const PagedKvStatus status = validate(handle, sequence);
    if (status != PagedKvStatus::Ok) return status;

    PagedKvSequenceSnapshot snapshot;
    snapshot.request_id = sequence->request_id;
    snapshot.handle = handle;
    snapshot.kv_seq_len = sequence->kv_seq_len;
    snapshot.block_table = sequence->block_table;
    out_sequence = std::move(snapshot);
    return PagedKvStatus::Ok;
}

PagedKvMetadataSnapshot PagedKvPool::metadata_snapshot() const {
    PagedKvMetadataSnapshot snapshot;
    snapshot.block_size = block_size_;
    snapshot.physical_block_count = physical_block_count_;
    snapshot.sequences.reserve(request_to_slot_.size());
    snapshot.block_table.reserve(
        physical_block_count_ - static_cast<uint32_t>(free_blocks_.size()));

    for (uint32_t slot = 0; slot < sequences_.size(); ++slot) {
        const SequenceState & sequence = sequences_[slot];
        if (!sequence.active) continue;

        PagedKvSequenceMetadata metadata;
        metadata.request_id = sequence.request_id;
        metadata.generation = sequence.generation;
        metadata.sequence_slot = slot;
        metadata.kv_seq_len = sequence.kv_seq_len;
        metadata.block_table_offset =
            static_cast<uint32_t>(snapshot.block_table.size());
        metadata.block_count =
            static_cast<uint32_t>(sequence.block_table.size());
        snapshot.sequences.push_back(metadata);
        snapshot.block_table.insert(snapshot.block_table.end(),
                                    sequence.block_table.begin(),
                                    sequence.block_table.end());
    }
    return snapshot;
}

uint32_t PagedKvPool::blocks_for_tokens(uint32_t token_count) const {
    if (token_count == 0) return 0;
    return 1 + (token_count - 1) / block_size_;
}

PagedKvStatus PagedKvPool::validate(
    PagedKvSequenceHandle handle,
    const SequenceState *& out_state) const {
    if (handle.slot >= sequences_.size()) {
        return PagedKvStatus::StaleHandle;
    }
    const SequenceState & sequence = sequences_[handle.slot];
    if (!sequence.active || sequence.generation != handle.generation) {
        return PagedKvStatus::StaleHandle;
    }
    out_state = &sequence;
    return PagedKvStatus::Ok;
}

PagedKvStatus PagedKvPool::validate(PagedKvSequenceHandle handle,
                                    SequenceState *& out_state) {
    const SequenceState * state = nullptr;
    const PagedKvStatus status =
        static_cast<const PagedKvPool *>(this)->validate(handle, state);
    out_state = const_cast<SequenceState *>(state);
    return status;
}

PagedKvStatus PagedKvPool::extend_block_table(SequenceState & sequence,
                                              uint32_t required_blocks) {
    const uint32_t current_blocks =
        static_cast<uint32_t>(sequence.block_table.size());
    if (required_blocks <= current_blocks) return PagedKvStatus::Ok;

    const uint32_t additional_blocks = required_blocks - current_blocks;
    if (additional_blocks > free_blocks_.size()) {
        return PagedKvStatus::BlocksExhausted;
    }

    std::vector<uint32_t> allocated_blocks;
    allocated_blocks.reserve(additional_blocks);
    auto free_it = free_blocks_.begin();
    for (uint32_t i = 0; i < additional_blocks; ++i, ++free_it) {
        allocated_blocks.push_back(*free_it);
    }

    sequence.block_table.insert(sequence.block_table.end(),
                                allocated_blocks.begin(),
                                allocated_blocks.end());
    for (uint32_t block : allocated_blocks) free_blocks_.erase(block);
    return PagedKvStatus::Ok;
}

void PagedKvPool::release_sequence(uint32_t slot) {
    SequenceState & sequence = sequences_[slot];
    request_to_slot_.erase(sequence.request_id);
    for (uint32_t block : sequence.block_table) free_blocks_.insert(block);
    sequence.request_id = 0;
    sequence.kv_seq_len = 0;
    sequence.active = false;
    sequence.block_table.clear();
    free_sequence_slots_.insert(slot);
}

}  // namespace dflash::common
