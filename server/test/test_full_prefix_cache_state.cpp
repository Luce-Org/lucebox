#include "server/prefix_cache_state.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

using dflash::common::FullPrefixCacheState;
using dflash::common::PrefixHash;
using dflash::common::prefix_hash_equal;

namespace {

PrefixHash key(uint8_t value) {
    PrefixHash result{};
    result[0] = value;
    return result;
}

void assert_invariants(const FullPrefixCacheState & state) {
    assert(state.size() >= 0);
    assert(state.size() <= state.capacity());
    for (size_t i = 0; i < state.entries().size(); ++i) {
        const auto & lhs = state.entries()[i];
        assert(lhs.slot >= state.slot_base());
        assert(lhs.slot < state.slot_base() + state.capacity());
        assert(lhs.slot != state.staging_slot());
        assert(lhs.cur_ids_len > 0);
        assert(lhs.raw_prompt_len > 0);
        for (size_t j = i + 1; j < state.entries().size(); ++j) {
            const auto & rhs = state.entries()[j];
            assert(lhs.slot != rhs.slot);
            assert(!prefix_hash_equal(lhs.hash, rhs.hash));
        }
    }
    if (state.has_pending_reservation()) {
        assert(state.pending_slot() >= state.slot_base());
        assert(state.pending_slot() <
               state.slot_base() + state.capacity());
        assert(state.pending_slot() != state.staging_slot());
    }
}

void commit(FullPrefixCacheState & state, PrefixHash hash,
            int expected_slot, int64_t now_ns) {
    const auto reservation = state.prepare(hash, 12);
    assert(reservation.accepted);
    assert(reservation.slot == expected_slot);
    assert(reservation.expected_snapshot_len == 12);
    const auto confirmed =
        state.confirm(reservation.slot, hash, 16, 12, now_ns);
    assert(confirmed.accepted);
}

void test_capacity_two_abort_hole_reuses_free_slot() {
    FullPrefixCacheState state(8, 2, 63);
    commit(state, key(1), 8, 10);

    // Allocation advances past slot 9. Aborting this free-slot reservation
    // must leave slot 9 as a hole rather than returning occupied slot 8.
    const auto failed = state.prepare(key(2), 12);
    assert(failed.accepted);
    assert(failed.slot == 9);
    assert(!failed.reuses_victim);
    assert(state.abort(failed.slot).accepted);
    assert(state.contains(key(1)));

    const auto replacement = state.prepare(key(3), 14);
    assert(replacement.accepted);
    assert(replacement.slot == 9);
    assert(!replacement.reuses_victim);
    assert(state.confirm(replacement.slot, key(3), 20, 14, 20).accepted);
    assert(state.contains(key(1)));
    assert(state.contains(key(3)));
    assert_invariants(state);
}

void test_lru_lookup_selects_oldest_victim() {
    FullPrefixCacheState state(12, 2, 63);
    commit(state, key(1), 12, 10);
    commit(state, key(2), 13, 20);

    const auto hit = state.lookup(key(1), 30);
    assert(hit.slot == 12);
    assert(hit.cur_ids_len == 12);
    assert(state.entries().back().hits == 1);
    assert(state.entries().back().last_used_ns == 30);

    const auto reservation = state.prepare(key(3), 18);
    assert(reservation.accepted);
    assert(reservation.reuses_victim);
    assert(prefix_hash_equal(reservation.victim_key, key(2)));
    assert(reservation.victim_slot == 13);
    assert(reservation.slot == 13);
    assert(prefix_hash_equal(state.pending_key(), key(3)));
    assert(state.pending_expected_snapshot_len() == 18);
    assert(prefix_hash_equal(state.pending_victim_key(), key(2)));
    assert(state.pending_victim_slot() == 13);

    // A backend slot selected for replacement is not a usable cache hit.
    assert(state.lookup(key(2), 40).slot == -1);
    assert(state.confirm(reservation.slot, key(3), 22, 18, 40).accepted);
    assert(!state.contains(key(2)));
    assert(state.contains(key(1)));
    assert(state.contains(key(3)));
    assert_invariants(state);
}

void test_aborted_victim_is_no_longer_committed() {
    FullPrefixCacheState state(20, 1, 63);
    commit(state, key(1), 20, 10);
    const auto reservation = state.prepare(key(2), 12);
    assert(reservation.accepted);
    assert(reservation.reuses_victim);
    const auto aborted = state.abort(reservation.slot);
    assert(aborted.accepted);
    assert(aborted.entries_removed == 1);
    assert(state.size() == 0);
    assert(!state.contains(key(1)));
    assert_invariants(state);
}

void test_invalid_reservations_do_not_mutate_state() {
    FullPrefixCacheState state(30, 2, 63);
    commit(state, key(1), 30, 10);

    assert(!state.confirm(31, key(2), 8, 8, 20).accepted);
    assert(!state.prepare(key(2), 0).accepted);
    const auto reservation = state.prepare(key(2), 8);
    assert(reservation.accepted);
    assert(reservation.slot == 31);

    assert(!state.prepare(key(3), 8).accepted);
    assert(!state.confirm(30, key(2), 8, 8, 20).accepted);
    assert(!state.confirm(31, key(3), 8, 8, 20).accepted);
    assert(!state.confirm(31, key(2), 0, 8, 20).accepted);
    assert(!state.confirm(31, key(2), 8, 0, 20).accepted);
    assert(!state.confirm(31, key(2), 8, 9, 20).accepted);
    assert(!state.abort(30).accepted);
    assert(state.has_pending_reservation());
    assert(prefix_hash_equal(state.pending_key(), key(2)));
    assert(state.pending_slot() == 31);
    assert(state.contains(key(1)));
    assert(state.size() == 1);

    assert(state.abort(31).accepted);
    assert(!state.has_pending_reservation());
    assert(state.contains(key(1)));
    assert(state.size() == 1);
    assert_invariants(state);
}

void test_slot_range_excludes_staging() {
    FullPrefixCacheState overlap(62, 2, 63);
    assert(!overlap.enabled());
    assert(!overlap.prepare(key(1), 8).accepted);

    FullPrefixCacheState adjacent(61, 2, 63);
    assert(adjacent.enabled());
    commit(adjacent, key(1), 61, 10);
    commit(adjacent, key(2), 62, 20);
    assert_invariants(adjacent);

    // External invalidation is deliberately separate from reservation abort.
    assert(adjacent.invalidate_slot(62) == 1);
    assert(!adjacent.contains(key(2)));
    assert(adjacent.invalidate_slot(63) == 0);
    assert_invariants(adjacent);
}

void test_effective_boundary_not_raw_length_is_authoritative() {
    FullPrefixCacheState state(40, 1, 63);
    const auto reservation = state.prepare(key(1), 10);
    assert(reservation.accepted);

    // The core does not infer the effective snapshot boundary from raw input
    // length. Compression/translation owns that mapping and binds it at
    // prepare time. A chunk-aligned backend snapshot may validly stop before
    // the bound even when its position is greater than the raw key length.
    assert(state.confirm(reservation.slot, key(1), 8, 9, 10).accepted);
    const auto hit = state.lookup(key(1), 20);
    assert(hit.slot == 40);
    assert(hit.cur_ids_len == 9);
    assert_invariants(state);
}

void test_snapshot_past_effective_boundary_is_rejected() {
    FullPrefixCacheState state(42, 1, 63);
    const auto reservation = state.prepare(key(1), 9);
    assert(reservation.accepted);
    assert(!state.confirm(reservation.slot, key(1), 20, 10, 10).accepted);
    assert(state.has_pending_reservation());
    assert(state.abort(reservation.slot).accepted);
    assert(!state.has_pending_reservation());
    assert(state.size() == 0);
}

}  // namespace

int main() {
    test_capacity_two_abort_hole_reuses_free_slot();
    test_lru_lookup_selects_oldest_victim();
    test_aborted_victim_is_no_longer_committed();
    test_invalid_reservations_do_not_mutate_state();
    test_slot_range_excludes_staging();
    test_effective_boundary_not_raw_length_is_authoritative();
    test_snapshot_past_effective_boundary_is_rejected();
    ::puts("full_prefix_cache_state: PASS");
    return 0;
}
