// PagedKvPool — block-table bookkeeping for paged KV-cache attention.
//
// Splits each sequence's KV cache into fixed-size blocks (vLLM-style) drawn
// from one shared physical pool, so sequences grow in block granularity
// instead of reserving max-context capacity up front. The pool tracks
// indices only — it owns no K/V storage. Callers translate the returned
// block/slot indices into offsets within their own pooled K/V tensors.
//
// The single-request backend consumes sequence() directly; the concurrent
// engines compose this allocator with their own per-slot lifecycle records.
//
// Not thread-safe; callers must serialize access. Pool state is unspecified
// if a std::bad_alloc escapes any call.

#pragma once

#include <cstdint>
#include <limits>
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
    StaleHandle,             // handle refers to a released or reused slot
};

// Human-readable status name for logs and error messages.
const char * paged_kv_status_string(PagedKvStatus status);

// Ticket for one active sequence. `slot` indexes the pool's internal
// sequence array; `generation` is bumped every time the slot is re-acquired,
// so a handle kept past release() is rejected as StaleHandle instead of
// silently aliasing the slot's next owner.
struct PagedKvSequenceHandle {
    uint32_t slot = std::numeric_limits<uint32_t>::max();
    uint64_t generation = 0;
};

// Cache destination for one appended token.
struct PagedKvWriteSlot {
    uint32_t logical_position = 0;  // 0-based position in the sequence
    uint32_t physical_block = 0;    // block index in the shared pool
    uint32_t block_offset = 0;      // token slot within that block
    // Flat row into the pooled K/V buffer:
    // physical_block * block_size + block_offset.
    uint64_t physical_token_index = 0;
};

// Outcome of append(). On success, `token_count` is the number of appended
// tokens. By default, `write_slots` holds one entry per token in logical
// order. With `only_first_last_slots`, it is empty and `first` and `last`
// hold the range endpoints instead. On failure all slot fields are empty or
// default-valued.
struct PagedKvAppendResult {
    PagedKvStatus status = PagedKvStatus::Ok;
    uint32_t token_count = 0;
    std::vector<PagedKvWriteSlot> write_slots;
    PagedKvWriteSlot first;
    PagedKvWriteSlot last;

    explicit operator bool() const { return status == PagedKvStatus::Ok; }
};

// Copy of one sequence's bookkeeping state, as returned by sequence().
struct PagedKvSequenceSnapshot {
    uint32_t kv_seq_len = 0;
    std::vector<uint32_t> block_table;
    // Physical blocks held for future append() calls by this sequence. They
    // are not visible in block_table until append consumes them.
    uint32_t reserved_block_count = 0;
};

// Allocator front-end: hands out sequence slots and physical block indices.
class PagedKvPool {
public:
    // `block_size` is the number of tokens per physical block. Throws
    // std::invalid_argument if any dimension is zero or the total token
    // capacity (physical_block_count * block_size) overflows uint32_t.
    explicit PagedKvPool(uint32_t physical_block_count,
                         uint32_t max_sequences,
                         uint32_t block_size);

    uint32_t block_size() const { return block_size_; }
    uint32_t physical_block_count() const { return physical_block_count_; }
    // Admission capacity: a concurrent engine sizes its slot table from this.
    uint32_t max_sequences() const {
        return static_cast<uint32_t>(sequences_.size());
    }
    uint32_t active_sequence_count() const {
        return static_cast<uint32_t>(request_to_slot_.size());
    }
    // Blocks that are neither appended nor reserved by an active sequence.
    uint32_t free_block_count() const {
        return static_cast<uint32_t>(free_blocks_.size());
    }

    // Claim a free sequence slot for `request_id`. The new sequence starts
    // empty; no blocks are allocated until append().
    PagedKvStatus acquire(PagedKvRequestId request_id,
                          PagedKvSequenceHandle & out_handle);

    // Claim a sequence slot and atomically reserve enough physical blocks for
    // `token_capacity` future tokens. The logical sequence still starts empty;
    // append() moves reserved blocks into its visible block table as needed.
    // On any status failure, no slot or block is consumed and `out_handle` is
    // unchanged. This is the admission primitive for chunked prompt prefill:
    // once it succeeds, another sequence cannot strand this prompt halfway
    // through by consuming the remainder of its capacity.
    PagedKvStatus acquire_reserved(PagedKvRequestId request_id,
                                   uint32_t token_capacity,
                                   PagedKvSequenceHandle & out_handle);

    // Atomically ensure an active sequence owns enough appended plus reserved
    // blocks to cover `token_capacity` logical tokens. This does not advance
    // kv_seq_len or expose new block-table entries; append() consumes the
    // private reservation later. Existing excess capacity is retained. On a
    // status failure no block moves and the sequence is unchanged.
    PagedKvStatus reserve_capacity(PagedKvSequenceHandle handle,
                                   uint32_t token_capacity);

    // Advance kv_seq_len by `token_count`, allocating new blocks as needed.
    // By default return every appended token's cache destination; when
    // `only_first_last_slots` is true, return only the range endpoints to
    // avoid materializing one slot per token. All-or-nothing: on failure
    // (stale handle, length overflow, BlocksExhausted) no state changes.
    // `token_count == 0` returns Ok with no slots.
    PagedKvAppendResult append(PagedKvSequenceHandle handle,
                              uint32_t token_count,
                              bool only_first_last_slots = false);

    // Return the sequence's blocks and slot to the pool; the handle (and
    // any copy of it) becomes stale.
    PagedKvStatus release(PagedKvSequenceHandle handle);

    // Drop every sequence and reclaim all blocks. Every outstanding handle
    // becomes stale.
    void reset();

    // Copy one sequence's current length and block table.
    PagedKvStatus sequence(PagedKvSequenceHandle handle,
                           PagedKvSequenceSnapshot & out_sequence) const;

    // Return appended plus reserved blocks without copying the block table.
    PagedKvStatus owned_block_count(PagedKvSequenceHandle handle,
                                    uint32_t & out_count) const;

private:
    // Bookkeeping for one sequence slot. `generation` survives release so
    // the next acquire on this slot invalidates old handles.
    struct SequenceState {
        PagedKvRequestId request_id = 0;
        uint64_t generation = 0;
        uint32_t kv_seq_len = 0;
        bool active = false;
        std::vector<uint32_t> block_table;
        // Min-heap of blocks promised to this sequence but not yet made
        // visible by append(). Reserved blocks are excluded from the global
        // free list and returned by release()/reset().
        std::vector<uint32_t> reserved_blocks;
    };

    // Blocks needed to hold `token_count` tokens (ceiling division).
    uint32_t blocks_for_tokens(uint32_t token_count) const;

    // StaleHandle when the slot is out of range, inactive, or from an older
    // generation.
    PagedKvStatus validate(PagedKvSequenceHandle handle) const;

    // Grow `sequence` to own `required_blocks` blocks (no-op if it already
    // does). All-or-nothing on BlocksExhausted.
    PagedKvStatus extend_block_table(SequenceState & sequence,
                                     uint32_t required_blocks);

    // Move exactly `additional_blocks` globally free blocks into a sequence's
    // private reservation. Caller must preflight availability.
    void take_reserved_blocks(SequenceState & sequence,
                              uint32_t additional_blocks);

    // Take the lowest free index off `free_list`, which must be non-empty.
    static uint32_t take_lowest(std::vector<uint32_t> & free_list);
    // Return one index to `free_list`.
    static void give_back(std::vector<uint32_t> & free_list, uint32_t index);
    // Refill `free_list` with every index in [0, count).
    static void refill(std::vector<uint32_t> & free_list, uint32_t count);

    uint32_t block_size_ = 0;
    uint32_t physical_block_count_ = 0;
    std::vector<SequenceState> sequences_;
    // Min-heaps (std::greater) used as lowest-index-first free lists, so a
    // released index is handed out again before any higher untouched one. A
    // heap keeps release at O(k log n) in the blocks actually returned, where
    // a sorted vector would re-sort the whole pool on every teardown. Their
    // capacities are fixed at construction, so reset/release do not allocate.
    // Contract: a fresh or reset() pool allocates physical blocks in ascending
    // order, which Qwen35Backend::begin_paged_sequence relies on for its
    // dense-prefill endpoint check.
    std::vector<uint32_t> free_sequence_slots_;
    std::vector<uint32_t> free_blocks_;
    std::unordered_map<PagedKvRequestId, uint32_t> request_to_slot_;
};

}  // namespace dflash::common
