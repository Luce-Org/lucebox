#include "server/prefix_cache_state.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using dflash::common::InlinePrefixCacheState;
using dflash::common::PrefixHash;
using dflash::common::prefix_hash_equal;
using dflash::common::select_inline_evict_victim;

namespace {

PrefixHash key(uint8_t family, uint8_t depth) {
    PrefixHash result{};
    result[0] = family;
    result[1] = depth;
    return result;
}

void assert_invariants(const InlinePrefixCacheState & state) {
    assert(state.capacity() >= 0);
    assert(state.size() >= 0);
    assert(state.size() <= state.capacity());
    if (state.capacity() > 0) {
        assert(state.next_slot() >= 0);
        assert(state.next_slot() < state.capacity());
    }

    const auto & entries = state.entries();
    for (size_t i = 0; i < entries.size(); ++i) {
        assert(entries[i].slot >= 0);
        assert(entries[i].slot < state.capacity());
        assert(!entries[i].ids.empty());
        for (size_t j = i + 1; j < entries.size(); ++j) {
            assert(entries[i].slot != entries[j].slot);
            assert(!prefix_hash_equal(entries[i].hash, entries[j].hash));
        }
    }

    if (state.has_pending_eviction()) {
        assert(state.contains(state.pending_eviction_key()));
    }
}

void test_round_robin_and_reuse() {
    InlinePrefixCacheState state(2);
    const std::vector<int32_t> a = {7, 10};
    const std::vector<int32_t> b = {8, 20};
    const std::vector<int32_t> c = {9, 30};

    auto ra = state.prepare(key(1, 2), 2);
    assert(ra.slot == 0);
    assert(state.confirm(ra.slot, key(1, 2), 2, a).accepted);

    auto rb = state.prepare(key(2, 2), 2);
    assert(rb.slot == 1);
    assert(state.confirm(rb.slot, key(2, 2), 2, b).accepted);
    assert_invariants(state);

    auto rc = state.prepare(key(3, 2), 2);
    assert(rc.slot == 0);
    assert(rc.victim_index == 0);
    auto confirmed = state.confirm(rc.slot, key(3, 2), 2, c);
    assert(confirmed.accepted);
    assert(confirmed.pending_removed == 1);
    assert(!state.contains(key(1, 2)));
    assert(state.contains(key(2, 2)));
    assert(state.contains(key(3, 2)));
    assert_invariants(state);
}

void test_abort_purges_reused_slot() {
    InlinePrefixCacheState state(2);
    const std::vector<int32_t> a = {7};
    const std::vector<int32_t> b = {8};

    auto ra = state.prepare(key(1, 1), 1);
    assert(state.confirm(ra.slot, key(1, 1), 1, a).accepted);
    auto rb = state.prepare(key(2, 1), 1);
    assert(state.confirm(rb.slot, key(2, 1), 1, b).accepted);

    auto pending = state.prepare(key(3, 1), 1);
    assert(pending.slot >= 0);
    state.abort(pending.slot);
    for (const auto & entry : state.entries()) {
        assert(entry.slot != pending.slot);
    }
    assert(!state.has_pending_eviction());
    assert_invariants(state);
}

void test_abort_reuses_hole_before_occupied_slot() {
    InlinePrefixCacheState state(2);
    const std::vector<int32_t> a = {7};

    auto committed = state.prepare(key(1, 1), 1);
    assert(committed.slot == 0);
    assert(state.confirm(
        committed.slot, key(1, 1), 1, a).accepted);

    // Reserving the second slot advances the round-robin cursor back to slot
    // zero. If that reservation aborts, slot one is a hole and slot zero still
    // owns a valid snapshot.
    auto failed = state.prepare(key(2, 1), 1);
    assert(failed.slot == 1);
    assert(state.abort(failed.slot) == 0);
    assert(state.contains(key(1, 1)));

    // The HTTP layer immediately frees the slot returned by prepare(). It must
    // therefore receive the free slot, not the occupied slot zero.
    auto replacement = state.prepare(key(3, 1), 1);
    assert(replacement.slot == failed.slot);
    for (const auto & entry : state.entries()) {
        assert(entry.slot != replacement.slot);
    }
    assert(state.contains(key(1, 1)));
    assert_invariants(state);
}

void test_cancel_preserves_entry() {
    InlinePrefixCacheState state(1);
    const std::vector<int32_t> ids = {7, 10};
    auto initial = state.prepare(key(1, 2), 2);
    assert(state.confirm(initial.slot, key(1, 2), 2, ids).accepted);

    auto pending = state.prepare(key(2, 2), 2);
    assert(pending.slot == initial.slot);
    assert(state.has_pending_eviction());
    assert(state.cancel(pending.slot));
    assert(state.contains(key(1, 2)));
    assert(!state.contains(key(2, 2)));
    assert(!state.has_pending_eviction());
    assert_invariants(state);
}

void test_stale_lookup_is_removed() {
    InlinePrefixCacheState state(2);
    const std::vector<int32_t> ids = {7, 10};
    auto reservation = state.prepare(key(1, 2), 2);
    assert(state.confirm(
        reservation.slot, key(1, 2), 2, ids).accepted);

    const auto stale = state.lookup_candidate(key(1, 2), 1);
    assert(stale.stale_removed);
    assert(stale.stale_slot == reservation.slot);
    assert(stale.stale_committed_len == 2);
    assert(state.size() == 0);
    assert_invariants(state);
}

void test_invalid_confirm_is_non_mutating() {
    InlinePrefixCacheState state(2);
    const std::vector<int32_t> ids = {7};
    assert(!state.confirm(-1, key(1, 1), 1, ids).accepted);
    assert(!state.confirm(2, key(1, 1), 1, ids).accepted);
    assert(!state.confirm(0, key(1, 2), 2, ids).accepted);
    assert(state.size() == 0);
    assert_invariants(state);
}

void test_prefix_aware_eviction() {
    const std::vector<std::vector<int32_t>> chain = {
        {7}, {7, 10}, {7, 10, 20},
    };
    assert(select_inline_evict_victim(chain) == 2);

    const std::vector<std::vector<int32_t>> branch = {
        {7}, {7, 10}, {7, 20},
    };
    assert(select_inline_evict_victim(branch) == 1);
}

void test_clear_resets_allocator() {
    InlinePrefixCacheState state(2);
    const std::vector<int32_t> ids = {7};
    auto reservation = state.prepare(key(1, 1), 1);
    assert(state.confirm(
        reservation.slot, key(1, 1), 1, ids).accepted);
    state.clear();
    assert(state.size() == 0);
    assert(state.next_slot() == 0);
    assert(!state.has_pending_eviction());
    assert_invariants(state);
}

}  // namespace

int main() {
    test_round_robin_and_reuse();
    test_abort_purges_reused_slot();
    test_abort_reuses_hole_before_occupied_slot();
    test_cancel_preserves_entry();
    test_stale_lookup_is_removed();
    test_invalid_confirm_is_non_mutating();
    test_prefix_aware_eviction();
    test_clear_resets_allocator();
    ::puts("prefix_cache_state: PASS");
    return 0;
}
