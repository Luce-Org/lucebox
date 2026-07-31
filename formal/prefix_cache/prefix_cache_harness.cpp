#include "server/prefix_cache_state.h"

#include <cassert>
#include <cstdint>
#include <vector>

#ifndef LUCEBOX_FORMAL_MAX_CAP
#define LUCEBOX_FORMAL_MAX_CAP 4
#endif

extern "C" unsigned int nondet_uint();
extern "C" void __ESBMC_assume(bool);

using dflash::common::InlinePrefixCacheState;
using dflash::common::PrefixHash;
using dflash::common::prefix_hash_equal;

namespace {

unsigned int bounded(unsigned int upper_exclusive) {
    const unsigned int value = nondet_uint();
    __ESBMC_assume(value < upper_exclusive);
    return value;
}

PrefixHash make_key(unsigned int branch, int depth) {
    PrefixHash key{};
    key[0] = (uint8_t)depth;
    key[1] = (uint8_t)(branch + 1);
    return key;
}

std::vector<int32_t> make_ids(unsigned int branch, int depth) {
    std::vector<int32_t> ids;
    ids.push_back(7);
    if (depth >= 2) ids.push_back((int32_t)(20 + branch));
    if (depth >= 3) ids.push_back((int32_t)(30 + branch));
    return ids;
}

void assert_invariants(const InlinePrefixCacheState & state) {
    assert(state.capacity() >= 1);
    assert(state.capacity() <= LUCEBOX_FORMAL_MAX_CAP);
    assert(state.size() >= 0);
    assert(state.size() <= state.capacity());
    assert(state.next_slot() >= 0);
    assert(state.next_slot() < state.capacity());

    const auto & entries = state.entries();
    for (size_t i = 0; i < entries.size(); ++i) {
        assert(entries[i].slot >= 0);
        assert(entries[i].slot < state.capacity());
        assert(!entries[i].ids.empty());
        assert(entries[i].ids.size() <= 3);
        for (size_t j = i + 1; j < entries.size(); ++j) {
            assert(entries[i].slot != entries[j].slot);
            assert(!prefix_hash_equal(entries[i].hash, entries[j].hash));
        }
    }

    if (state.has_pending_eviction()) {
        assert(state.contains(state.pending_eviction_key()));
    }
}

}  // namespace

int main() {
    const int capacity = (int)bounded(LUCEBOX_FORMAL_MAX_CAP) + 1;
    const unsigned int branch = bounded(3);
    const int depth = (int)bounded(3) + 1;
    InlinePrefixCacheState state(capacity);
    const PrefixHash key = make_key(branch, depth);
    const std::vector<int32_t> ids = make_ids(branch, depth);

    const auto reservation = state.prepare(key, depth);
    assert(reservation.slot >= 0);
    assert(reservation.slot < capacity);
    assert(reservation.target_cut == depth);
    assert(!state.has_pending_eviction());

    const auto confirmed =
        state.confirm(reservation.slot, key, depth, ids);
    assert(confirmed.accepted);
    assert(state.size() == 1);
    assert(state.contains(key));
    assert(state.entries()[0].ids.size() == (size_t)depth);
    for (int i = 0; i < depth; ++i) {
        assert(state.entries()[0].ids[(size_t)i] == ids[(size_t)i]);
    }

    const auto found = state.lookup_candidate(key, depth);
    assert(found.slot == reservation.slot);
    assert(found.prefix_len == depth);
    assert(!found.stale_removed);
    assert_invariants(state);
    return 0;
}
