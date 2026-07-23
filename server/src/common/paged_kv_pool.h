// PagedKvPool — block-table bookkeeping for paged KV-cache attention.
//
// Splits each sequence's KV cache into fixed-size blocks (vLLM-style) drawn
// from one shared physical pool, so sequences grow in block granularity
// instead of reserving max-context capacity up front. The pool tracks
// indices only — it owns no K/V storage. Callers translate the returned
// block/slot indices into offsets within their own pooled K/V tensors.
//
// The current single-request backend consumes sequence() directly. The
// request lookup, reservation, cancellation, capacity, and flattened
// metadata APIs below are intentionally retained for the continuous-batching
// scheduler, where several live request tables will be uploaded together.
//
// Not thread-safe; callers must serialize access.

#pragma once

#include <cstdint>
#include <limits>
#include <set>
#include <unordered_map>
#include <vector>

namespace dflash::common {

// Caller-chosen identifier for one inference request (one KV sequence).
using PagedKvRequestId = uint64_t;

// Result code of every pool operation. Any value other than Ok means the
// call left the pool state unchanged.
enum class PagedKvStatus : uint8_t {
    Ok = 0,
    InvalidArgument,         // size argument would overflow the sequence length
    DuplicateRequest,        // acquire() for a request id that is still active
    SequenceSlotsExhausted,  // all max_sequences slots are in use
    BlocksExhausted,         // not enough free physical blocks for the growth
    RequestNotFound,         // lookup()/cancel() for an unknown request id
    StaleHandle,             // handle refers to a released or reused slot
};

// Human-readable status name for logs and error messages.
const char * paged_kv_status_string(PagedKvStatus status);

// Ticket for one active sequence. `slot` indexes the pool's internal
// sequence array; `generation` is bumped every time the slot is re-acquired,
// so a handle kept past release()/cancel() is rejected as StaleHandle
// instead of silently aliasing the slot's next owner.
struct PagedKvSequenceHandle {
    uint32_t slot = std::numeric_limits<uint32_t>::max();
    uint64_t generation = 0;
};

// Kept as a complete comparison pair for continuous-batching scheduler state,
// where handles will be compared across admission and cancellation events.
constexpr bool operator==(const PagedKvSequenceHandle & lhs,
                          const PagedKvSequenceHandle & rhs) {
    return lhs.slot == rhs.slot && lhs.generation == rhs.generation;
}

constexpr bool operator!=(const PagedKvSequenceHandle & lhs,
                          const PagedKvSequenceHandle & rhs) {
    return !(lhs == rhs);
}

// Cache destination for one appended token.
struct PagedKvWriteSlot {
    uint32_t logical_position = 0;  // 0-based position in the sequence
    uint32_t physical_block = 0;    // block index in the shared pool
    uint32_t block_offset = 0;      // token slot within that block
    // Flat row into the pooled K/V buffer:
    // physical_block * block_size + block_offset.
    uint64_t physical_token_index = 0;
};

// Outcome of append(). On success, `write_slots` holds one entry per
// appended token in logical order; on failure it is empty.
struct PagedKvAppendResult {
    PagedKvStatus status = PagedKvStatus::Ok;
    std::vector<PagedKvWriteSlot> write_slots;

    explicit operator bool() const { return status == PagedKvStatus::Ok; }
};

// Copy of one sequence's bookkeeping state, as returned by sequence().
// The identity echo lets a future batching scheduler reject a snapshot that
// raced with cancellation and slot reuse before it submits GPU work.
struct PagedKvSequenceSnapshot {
    PagedKvRequestId request_id = 0;
    PagedKvSequenceHandle handle;
    uint32_t kv_seq_len = 0;
    std::vector<uint32_t> block_table;
};

// Per-sequence entry of a future continuous-batch metadata upload. Each entry
// addresses its own range [block_table_offset, block_table_offset +
// block_count) in PagedKvMetadataSnapshot::block_table. request_id,
// generation, and sequence_slot are CPU-side scheduler identity; only the
// lengths, offsets, counts, and flattened table are GPU routing material.
struct PagedKvSequenceMetadata {
    PagedKvRequestId request_id = 0;
    uint64_t generation = 0;
    uint32_t sequence_slot = 0;
    uint32_t kv_seq_len = 0;
    uint32_t block_table_offset = 0;
    uint32_t block_count = 0;
};

// Flattened view of every active sequence, ordered by slot index. This is
// reserved for the continuous-batching integration; the current backend does
// not upload it yet. `block_table` is the concatenation of all per-sequence
// block tables.
struct PagedKvMetadataSnapshot {
    uint32_t block_size = 0;
    uint32_t physical_block_count = 0;
    std::vector<PagedKvSequenceMetadata> sequences;
    std::vector<uint32_t> block_table;
};

// Allocator front-end: hands out sequence slots and physical block indices.
class PagedKvPool {
public:
    // `block_size` is the number of tokens per physical block. Throws
    // std::invalid_argument if any dimension is zero or the total token
    // capacity (physical_block_count * block_size) overflows uint32_t.
    explicit PagedKvPool(uint32_t physical_block_count,
                         uint32_t max_sequences,
                         uint32_t block_size = 16);

    uint32_t block_size() const { return block_size_; }
    uint32_t physical_block_count() const { return physical_block_count_; }
    // Admission capacity for the future continuous-batching scheduler.
    uint32_t max_sequences() const {
        return static_cast<uint32_t>(sequences_.size());
    }
    uint32_t active_sequence_count() const {
        return static_cast<uint32_t>(request_to_slot_.size());
    }
    uint32_t free_block_count() const {
        return static_cast<uint32_t>(free_blocks_.size());
    }

    // Claim a free sequence slot for `request_id`. The new sequence starts
    // empty; no blocks are allocated until reserve()/append().
    PagedKvStatus acquire(PagedKvRequestId request_id,
                          PagedKvSequenceHandle & out_handle);

    // Fetch the live handle of an already-acquired request. Continuous
    // batching will use this when scheduler events carry request ids.
    PagedKvStatus lookup(PagedKvRequestId request_id,
                         PagedKvSequenceHandle & out_handle) const;

    // Grow the sequence's block table until it can hold `token_capacity`
    // tokens in total (never shrinks). This does not advance kv_seq_len; a
    // later append() fills the pre-allocated blocks first. The single-request
    // backend does not need this, but batched admission can use it to fail an
    // entire scheduling round before launching any prefills.
    PagedKvStatus reserve(PagedKvSequenceHandle handle,
                          uint32_t token_capacity);

    // Advance kv_seq_len by `token_count`, allocating new blocks as needed,
    // and return the cache destination of every appended token. All-or-
    // nothing: on failure (stale handle, length overflow, BlocksExhausted)
    // no state changes. `token_count == 0` returns Ok with no write slots.
    PagedKvAppendResult append(PagedKvSequenceHandle handle,
                              uint32_t token_count);

    // Return the sequence's blocks and slot to the pool; the handle (and
    // any copy of it) becomes stale.
    PagedKvStatus release(PagedKvSequenceHandle handle);

    // release() by request id, for continuous-batching cancellation paths
    // that no longer hold the handle (e.g. client disconnect).
    PagedKvStatus cancel(PagedKvRequestId request_id);

    // Drop every sequence and reclaim all blocks. Every outstanding handle
    // becomes stale.
    void reset();

    // Copy one sequence's current length and block table.
    PagedKvStatus sequence(PagedKvSequenceHandle handle,
                           PagedKvSequenceSnapshot & out_sequence) const;

    // Flattened copy of all active sequences for the planned batched device
    // metadata upload.
    PagedKvMetadataSnapshot metadata_snapshot() const;

private:
    // Bookkeeping for one sequence slot. `generation` survives release so
    // the next acquire on this slot invalidates old handles.
    struct SequenceState {
        PagedKvRequestId request_id = 0;
        uint64_t generation = 0;
        uint32_t kv_seq_len = 0;
        bool active = false;
        std::vector<uint32_t> block_table;
    };

    // Blocks needed to hold `token_count` tokens (ceiling division).
    uint32_t blocks_for_tokens(uint32_t token_count) const;

    // Resolve a handle to its live state; StaleHandle when the slot is out
    // of range, inactive, or from an older generation.
    PagedKvStatus validate(PagedKvSequenceHandle handle,
                           const SequenceState *& out_state) const;
    PagedKvStatus validate(PagedKvSequenceHandle handle,
                           SequenceState *& out_state);

    // Grow `sequence` to own `required_blocks` blocks (no-op if it already
    // does). All-or-nothing on BlocksExhausted.
    PagedKvStatus extend_block_table(SequenceState & sequence,
                                     uint32_t required_blocks);

    // Return one slot's blocks and request mapping to the free lists.
    void release_sequence(uint32_t slot);

    uint32_t block_size_ = 0;
    uint32_t physical_block_count_ = 0;
    std::vector<SequenceState> sequences_;
    // Ordered sets make allocation deterministic: the lowest-numbered free
    // slot/block is always handed out first.
    std::set<uint32_t> free_sequence_slots_;
    std::set<uint32_t> free_blocks_;
    std::unordered_map<PagedKvRequestId, uint32_t> request_to_slot_;
};

}  // namespace dflash::common
