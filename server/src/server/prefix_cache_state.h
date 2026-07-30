// Verification-friendly state core for the inline prefix cache.
//
// This header deliberately contains no tokenizer, hashing, ggml, CUDA, or
// server dependencies. PrefixCache performs boundary detection and key
// derivation, then delegates its state transitions here. The same production
// transition code is therefore usable by native unit tests and ESBMC harnesses.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace dflash::common {

using PrefixHash = std::array<uint8_t, 16>;

// Keep equality explicit instead of delegating to std::array's loop. Besides
// being cheap for a fixed 128-bit key, this gives model checkers a finite,
// fully unrolled comparison while remaining the production implementation.
inline bool prefix_hash_equal(
        const PrefixHash & lhs, const PrefixHash & rhs) {
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] &&
           lhs[2] == rhs[2] && lhs[3] == rhs[3] &&
           lhs[4] == rhs[4] && lhs[5] == rhs[5] &&
           lhs[6] == rhs[6] && lhs[7] == rhs[7] &&
           lhs[8] == rhs[8] && lhs[9] == rhs[9] &&
           lhs[10] == rhs[10] && lhs[11] == rhs[11] &&
           lhs[12] == rhs[12] && lhs[13] == rhs[13] &&
           lhs[14] == rhs[14] && lhs[15] == rhs[15];
}

inline void prefix_hash_copy(PrefixHash & dst, const PrefixHash & src) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    dst[4] = src[4];
    dst[5] = src[5];
    dst[6] = src[6];
    dst[7] = src[7];
    dst[8] = src[8];
    dst[9] = src[9];
    dst[10] = src[10];
    dst[11] = src[11];
    dst[12] = src[12];
    dst[13] = src[13];
    dst[14] = src[14];
    dst[15] = src[15];
}

namespace prefix_cache_detail {

inline bool is_strict_prefix(const std::vector<int32_t> & a,
                             const std::vector<int32_t> & b) {
    if (a.size() >= b.size()) return false;
    return std::equal(a.begin(), a.end(), b.begin());
}

}  // namespace prefix_cache_detail

// Prefix-aware inline eviction policy. Inputs are in LRU order
// (index 0 = oldest). Prefer the oldest leaf so shared ancestors remain hot.
inline int select_inline_evict_victim(
        const std::vector<const std::vector<int32_t> *> & ids_lru) {
    const int n = (int)ids_lru.size();
    if (n <= 0) return 0;
    for (int i = 0; i < n; ++i) {
        bool is_ancestor = false;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (prefix_cache_detail::is_strict_prefix(
                    *ids_lru[(size_t)i], *ids_lru[(size_t)j])) {
                is_ancestor = true;
                break;
            }
        }
        if (!is_ancestor) return i;
    }
    return 0;
}

inline int select_inline_evict_victim(
        const std::vector<std::vector<int32_t>> & ids_lru) {
    std::vector<const std::vector<int32_t> *> ptrs;
    ptrs.reserve(ids_lru.size());
    for (const auto & ids : ids_lru) ptrs.push_back(&ids);
    return select_inline_evict_victim(ptrs);
}

// Select a slot below the production PrefixCache limit of 64 slots. Keeping
// this allocation decision scalar makes the behavior independently
// model-checkable; InlinePrefixCacheState remains responsible for deriving the
// occupancy mask from its committed entries.
inline int select_inline_free_slot(
        int next_slot, int capacity, uint64_t occupied_slots) {
    if (capacity <= 0 || next_slot < 0 || next_slot >= capacity) {
        return -1;
    }
    // InlinePrefixCacheState is independently usable, while the production
    // PrefixCache clamps capacity to 64. Preserve the legacy round-robin
    // behavior for out-of-contract standalone capacities that do not fit the
    // occupancy mask.
    if (capacity > 64) return next_slot;

    for (int offset = 0; offset < capacity; ++offset) {
        const int candidate = (next_slot + offset) % capacity;
        const uint64_t candidate_bit = uint64_t{1} << candidate;
        if ((occupied_slots & candidate_bit) == 0) {
            return candidate;
        }
    }
    return -1;
}

class InlinePrefixCacheState {
public:
    struct Entry {
        PrefixHash hash{};
        int slot = -1;
        std::vector<int32_t> ids;
    };

    struct LookupResult {
        int slot = -1;
        int prefix_len = 0;
        bool stale_removed = false;
        int stale_slot = -1;
        int stale_committed_len = 0;
    };

    struct PrepareResult {
        int slot = -1;
        int target_cut = 0;
        int victim_index = -1;
        int victim_len = 0;
        int oldest_len = 0;
    };

    struct ConfirmResult {
        bool accepted = false;
        int pending_removed = 0;
        int stale_slot_entries_removed = 0;
    };

    explicit InlinePrefixCacheState(int capacity = 0)
        : capacity_(std::max(0, capacity)) {}

    int capacity() const { return capacity_; }
    int size() const { return (int)entries_.size(); }
    int next_slot() const { return next_slot_; }
    bool has_pending_eviction() const { return has_pending_evict_; }
    const PrefixHash & pending_eviction_key() const {
        return pending_evict_key_;
    }
    const std::vector<Entry> & entries() const { return entries_; }

    int find(const PrefixHash & hash) const {
        for (int i = 0; i < (int)entries_.size(); ++i) {
            if (prefix_hash_equal(entries_[(size_t)i].hash, hash)) return i;
        }
        return -1;
    }

    bool contains(const PrefixHash & hash) const { return find(hash) >= 0; }

    LookupResult lookup_candidate(const PrefixHash & hash, int cut) {
        LookupResult result;
        const int idx = find(hash);
        if (idx < 0) return result;

        const int committed = (int)entries_[(size_t)idx].ids.size();
        if (committed != cut) {
            result.stale_removed = true;
            result.stale_slot = entries_[(size_t)idx].slot;
            result.stale_committed_len = committed;
            entries_.erase(entries_.begin() + idx);
            return result;
        }

        result.slot = entries_[(size_t)idx].slot;
        result.prefix_len = cut;
        move_to_end(idx);
        return result;
    }

    PrepareResult prepare(const PrefixHash & hash, int target_cut) {
        PrepareResult result;
        if (capacity_ <= 0 || target_cut <= 0 || contains(hash)) return result;

        result.target_cut = target_cut;
        if ((int)entries_.size() >= capacity_) {
            std::vector<const std::vector<int32_t> *> ids_lru;
            ids_lru.reserve(entries_.size());
            for (const auto & entry : entries_) ids_lru.push_back(&entry.ids);

            const int victim = select_inline_evict_victim(ids_lru);
            pending_evict_key_ = entries_[(size_t)victim].hash;
            has_pending_evict_ = true;
            result.slot = entries_[(size_t)victim].slot;
            result.victim_index = victim;
            result.victim_len =
                (int)entries_[(size_t)victim].ids.size();
            result.oldest_len = (int)entries_.front().ids.size();
        } else {
            uint64_t occupied_slots = 0;
            if (capacity_ <= 64) {
                for (const auto & entry : entries_) {
                    if (entry.slot >= 0 && entry.slot < 64) {
                        occupied_slots |= uint64_t{1} << entry.slot;
                    }
                }
            }
            result.slot = select_inline_free_slot(
                next_slot_, capacity_, occupied_slots);
            next_slot_ = (result.slot + 1) % capacity_;
            has_pending_evict_ = false;
        }
        return result;
    }

    ConfirmResult confirm(int slot, const PrefixHash & hash, int target_cut,
                          const std::vector<int32_t> & prompt_ids) {
        ConfirmResult result;
        if (slot < 0 || slot >= capacity_ || target_cut <= 0 ||
            target_cut > (int)prompt_ids.size()) {
            return result;
        }

        if (has_pending_evict_) {
            const int idx = find(pending_evict_key_);
            if (idx >= 0) {
                entries_.erase(entries_.begin() + idx);
                result.pending_removed = 1;
            }
            has_pending_evict_ = false;
        }

        for (int i = (int)entries_.size() - 1; i >= 0; --i) {
            if (entries_[(size_t)i].slot == slot) {
                entries_.erase(entries_.begin() + i);
                ++result.stale_slot_entries_removed;
            }
        }

        std::vector<int32_t> ids(
            prompt_ids.begin(), prompt_ids.begin() + target_cut);
        entries_.push_back({hash, slot, std::move(ids)});
        result.accepted = true;
        return result;
    }

    int abort(int slot) {
        int removed = 0;
        for (int i = (int)entries_.size() - 1; i >= 0; --i) {
            if (entries_[(size_t)i].slot == slot) {
                entries_.erase(entries_.begin() + i);
                ++removed;
            }
        }
        has_pending_evict_ = false;
        return removed;
    }

    // Returns false only when the supplied slot does not own the pending
    // reservation. In that case state is left untouched.
    bool cancel(int slot) {
        if (has_pending_evict_) {
            const int idx = find(pending_evict_key_);
            if (idx >= 0 && entries_[(size_t)idx].slot != slot) return false;
        }
        has_pending_evict_ = false;
        return true;
    }

    void clear() {
        entries_.clear();
        next_slot_ = 0;
        has_pending_evict_ = false;
        pending_evict_key_ = {};
    }

private:
    void move_to_end(int idx) {
        if (idx < 0 || idx >= (int)entries_.size()) return;
        auto entry = std::move(entries_[(size_t)idx]);
        entries_.erase(entries_.begin() + idx);
        entries_.push_back(std::move(entry));
    }

    int capacity_ = 0;
    std::vector<Entry> entries_;
    int next_slot_ = 0;
    PrefixHash pending_evict_key_{};
    bool has_pending_evict_ = false;
};

// Exact-match full-prompt cache state. Slots are absolute backend snapshot
// slots, while allocation is relative to [slot_base, slot_base + capacity).
// The staging slot is supplied separately and configuration is rejected when
// the two ranges overlap.
//
// A reservation binds the requested key, destination slot, effective snapshot
// boundary, and (when the pool is full) the LRU victim. Confirm and
// abort must match that reservation before committed metadata can change.
// This prevents a late or partial snapshot from being published as an
// exact-prompt hit.
class FullPrefixCacheState {
public:
    struct Entry {
        PrefixHash hash{};
        int slot = -1;
        int cur_ids_len = 0;
        int raw_prompt_len = 0;
        int64_t last_used_ns = 0;
        int hits = 0;

        Entry() = default;
        Entry(const Entry & other) { copy_from(other); }
        Entry(Entry && other) noexcept { copy_from(other); }
        Entry & operator=(const Entry & other) {
            if (this != &other) copy_from(other);
            return *this;
        }
        Entry & operator=(Entry && other) noexcept {
            if (this != &other) copy_from(other);
            return *this;
        }

    private:
        void copy_from(const Entry & other) {
            prefix_hash_copy(hash, other.hash);
            slot = other.slot;
            cur_ids_len = other.cur_ids_len;
            raw_prompt_len = other.raw_prompt_len;
            last_used_ns = other.last_used_ns;
            hits = other.hits;
        }
    };

    struct LookupResult {
        int slot = -1;
        int cur_ids_len = 0;
    };

    struct PrepareResult {
        bool accepted = false;
        int slot = -1;
        int expected_snapshot_len = 0;
        bool reuses_victim = false;
        PrefixHash victim_key{};
        int victim_slot = -1;
    };

    struct MutationResult {
        bool accepted = false;
        int entries_removed = 0;
    };

    FullPrefixCacheState() = default;

    FullPrefixCacheState(int slot_base, int capacity, int staging_slot) {
        configure(slot_base, capacity, staging_slot);
    }

    bool configure(int slot_base, int capacity, int staging_slot) {
        clear();
        slot_base_ = slot_base;
        capacity_ = capacity;
        staging_slot_ = staging_slot;
        enabled_ = slot_base >= 0 && capacity > 0 && capacity <= 64 &&
                   slot_base <= std::numeric_limits<int>::max() - capacity &&
                   (staging_slot < slot_base ||
                    staging_slot >= slot_base + capacity);
        if (!enabled_) {
            capacity_ = 0;
        }
        return enabled_;
    }

    bool enabled() const { return enabled_; }
    int slot_base() const { return slot_base_; }
    int capacity() const { return capacity_; }
    int staging_slot() const { return staging_slot_; }
    int size() const { return (int)entries_.size(); }
    int next_relative_slot() const { return next_relative_slot_; }
    const std::vector<Entry> & entries() const { return entries_; }

    bool has_pending_reservation() const { return pending_.active; }
    int pending_slot() const { return pending_.active ? pending_.slot : -1; }
    const PrefixHash & pending_key() const { return pending_.key; }
    int pending_expected_snapshot_len() const {
        return pending_.active ? pending_.expected_snapshot_len : 0;
    }
    bool pending_reuses_victim() const {
        return pending_.active && pending_.has_victim;
    }
    const PrefixHash & pending_victim_key() const {
        return pending_.victim_key;
    }
    int pending_victim_slot() const {
        return pending_.has_victim ? pending_.victim_slot : -1;
    }

    int find(const PrefixHash & hash) const {
        for (int i = 0; i < (int)entries_.size(); ++i) {
            if (prefix_hash_equal(entries_[(size_t)i].hash, hash)) return i;
        }
        return -1;
    }

    bool contains(const PrefixHash & hash) const { return find(hash) >= 0; }

    LookupResult lookup(const PrefixHash & hash, int64_t now_ns) {
        LookupResult result;
        const int idx = find(hash);
        if (idx < 0) return result;

        // prepare() means the backend destination is about to be cleared.
        // Never advertise that victim as a usable hit while it is reserved.
        if (pending_.active && entries_[(size_t)idx].slot == pending_.slot) {
            return result;
        }

        Entry entry = std::move(entries_[(size_t)idx]);
        entries_.erase(entries_.begin() + idx);
        entry.hits += 1;
        entry.last_used_ns = now_ns;
        result.slot = entry.slot;
        result.cur_ids_len = entry.cur_ids_len;
        entries_.push_back(std::move(entry));
        return result;
    }

    PrepareResult prepare(const PrefixHash & hash,
                          int expected_snapshot_len) {
        PrepareResult result;
        if (!enabled_ || pending_.active || contains(hash) ||
            expected_snapshot_len <= 0) {
            return result;
        }

        int abs_slot = -1;
        if ((int)entries_.size() < capacity_) {
            uint64_t occupied = 0;
            if (capacity_ <= 64) {
                for (const auto & entry : entries_) {
                    const int relative = entry.slot - slot_base_;
                    if (relative >= 0 && relative < 64) {
                        occupied |= uint64_t{1} << relative;
                    }
                }
            }
            const int relative = select_inline_free_slot(
                next_relative_slot_, capacity_, occupied);
            if (relative < 0) return result;
            abs_slot = slot_base_ + relative;
            next_relative_slot_ = (relative + 1) % capacity_;
        } else {
            if (entries_.empty()) return result;
            pending_.has_victim = true;
            prefix_hash_copy(
                pending_.victim_key, entries_.front().hash);
            pending_.victim_slot = entries_.front().slot;
            abs_slot = pending_.victim_slot;
        }

        if (abs_slot == staging_slot_) return result;
        pending_.active = true;
        prefix_hash_copy(pending_.key, hash);
        pending_.slot = abs_slot;
        pending_.expected_snapshot_len = expected_snapshot_len;

        result.accepted = true;
        result.slot = abs_slot;
        result.expected_snapshot_len = expected_snapshot_len;
        result.reuses_victim = pending_.has_victim;
        if (pending_.has_victim) {
            prefix_hash_copy(
                result.victim_key, pending_.victim_key);
        }
        result.victim_slot = pending_.victim_slot;
        return result;
    }

    MutationResult confirm(int slot, const PrefixHash & hash,
                           int raw_prompt_len, int cur_ids_len,
                           int64_t now_ns) {
        MutationResult result;
        // raw_prompt_len describes the lookup key, but it is not necessarily
        // the effective model-input length after compression/translation.
        // Only the boundary explicitly bound by prepare() is authoritative.
        if (!reservation_matches(slot, hash) || raw_prompt_len <= 0 ||
            cur_ids_len <= 0 ||
            cur_ids_len > pending_.expected_snapshot_len) {
            return result;
        }
        if (!victim_still_matches()) return result;

        result.entries_removed = erase_slot(slot);
        Entry entry;
        prefix_hash_copy(entry.hash, hash);
        entry.slot = slot;
        entry.cur_ids_len = cur_ids_len;
        entry.raw_prompt_len = raw_prompt_len;
        entry.last_used_ns = now_ns;
        entries_.push_back(std::move(entry));
        reset_pending();
        result.accepted = true;
        return result;
    }

    MutationResult abort(int slot) {
        MutationResult result;
        if (!pending_.active || pending_.slot != slot ||
            !victim_still_matches()) {
            return result;
        }

        // The backend clears the reserved slot before attempting generation,
        // so a victim at that slot is no longer restorable after an abort.
        result.entries_removed = erase_slot(slot);
        reset_pending();
        result.accepted = true;
        return result;
    }

    // A backend snapshot may disappear independently of a reservation (for
    // example after OOM recovery). Keep that operation distinct from abort so
    // the reservation transition itself remains strictly validated.
    int invalidate_slot(int slot) {
        if (!slot_in_pool(slot)) return 0;
        if (pending_.active && pending_.slot == slot) reset_pending();
        return erase_slot(slot);
    }

    void clear() {
        entries_.clear();
        next_relative_slot_ = 0;
        reset_pending();
    }

private:
    struct PendingReservation {
        bool active = false;
        PrefixHash key{};
        int slot = -1;
        int expected_snapshot_len = 0;
        bool has_victim = false;
        PrefixHash victim_key{};
        int victim_slot = -1;
    };

    bool slot_in_pool(int slot) const {
        return enabled_ && slot >= slot_base_ &&
               slot < slot_base_ + capacity_ && slot != staging_slot_;
    }

    bool reservation_matches(int slot, const PrefixHash & hash) const {
        return pending_.active && slot_in_pool(slot) &&
               pending_.slot == slot &&
               prefix_hash_equal(pending_.key, hash);
    }

    bool victim_still_matches() const {
        if (!pending_.has_victim) return true;
        const int idx = find(pending_.victim_key);
        return idx >= 0 && entries_[(size_t)idx].slot == pending_.victim_slot &&
               pending_.victim_slot == pending_.slot;
    }

    int erase_slot(int slot) {
        int removed = 0;
        for (int i = (int)entries_.size() - 1; i >= 0; --i) {
            if (entries_[(size_t)i].slot == slot) {
                entries_.erase(entries_.begin() + i);
                ++removed;
            }
        }
        return removed;
    }

    void reset_pending() {
        // Keys are irrelevant while inactive. Reset only scalar ownership
        // fields; avoiding a whole-struct std::array memcpy also keeps this
        // transition tractable in bounded-verifier C++ frontends.
        pending_.active = false;
        pending_.slot = -1;
        pending_.expected_snapshot_len = 0;
        pending_.has_victim = false;
        pending_.victim_slot = -1;
    }

    bool enabled_ = false;
    int slot_base_ = 0;
    int capacity_ = 0;
    int staging_slot_ = -1;
    int next_relative_slot_ = 0;
    std::vector<Entry> entries_;
    PendingReservation pending_;
};

}  // namespace dflash::common
