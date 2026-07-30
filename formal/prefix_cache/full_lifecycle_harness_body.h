// Shared body for generated and legacy full prefix-cache harnesses.

#include "server/prefix_cache_state.h"

#include <cassert>
#include <cstdint>

#ifndef LUCEBOX_FORMAL_CONTRACT_SYMBOL
#error "LUCEBOX_FORMAL_CONTRACT_SYMBOL must name the production state type"
#endif

#ifndef LUCEBOX_FORMAL_CAP
#define LUCEBOX_FORMAL_CAP 2
#endif

extern "C" unsigned int nondet_uint();
extern "C" void __ESBMC_assume(bool);

using ContractState = LUCEBOX_FORMAL_CONTRACT_SYMBOL;
using dflash::common::PrefixHash;

namespace {

constexpr int kSlotBase = 8;
constexpr int kStagingSlot = 63;

unsigned int bounded(unsigned int upper_exclusive) {
    const unsigned int value = nondet_uint();
    __ESBMC_assume(value < upper_exclusive);
    return value;
}

PrefixHash key(uint8_t value) {
    PrefixHash result{};
    result[0] = value;
    return result;
}

}  // namespace

int main() {
    static_assert(LUCEBOX_FORMAL_CAP > 0);
    static_assert(LUCEBOX_FORMAL_CAP <= 8);
    static_assert(kSlotBase + LUCEBOX_FORMAL_CAP < kStagingSlot);

    const int expected_snapshot_len =
        static_cast<int>(bounded(4)) + 1;
    const int saved_pos =
        static_cast<int>(bounded(
            static_cast<unsigned int>(expected_snapshot_len))) + 1;
    const PrefixHash first = key(1);
    const PrefixHash wrong = key(2);

    ContractState state(
        kSlotBase, LUCEBOX_FORMAL_CAP, kStagingSlot);
    assert(state.enabled());
    assert(state.capacity() == LUCEBOX_FORMAL_CAP);
    assert(state.size() == 0);

    const auto prepared =
        state.prepare(first, expected_snapshot_len);
    assert(prepared.accepted);
    assert(!prepared.reuses_victim);
    assert(prepared.slot >= kSlotBase);
    assert(prepared.slot < kSlotBase + LUCEBOX_FORMAL_CAP);
    assert(prepared.slot != kStagingSlot);
    assert(prepared.expected_snapshot_len == expected_snapshot_len);
    assert(state.has_pending_reservation());
    assert(state.pending_slot() == prepared.slot);
    assert(
        state.pending_expected_snapshot_len() ==
        expected_snapshot_len);
    assert(state.size() == 0);

    // Neither a wrong key nor a saved position outside the prepared effective
    // boundary may resolve the reservation.
    assert(!state.confirm(
        prepared.slot, wrong, 1, saved_pos, 1).accepted);
    assert(!state.confirm(
        prepared.slot, first, 1, expected_snapshot_len + 1, 1).accepted);
    assert(state.has_pending_reservation());
    assert(state.size() == 0);

    // Raw prompt length is the lookup-key length and may be shorter than the
    // effective/compressed snapshot position.
    const auto confirmed =
        state.confirm(prepared.slot, first, 1, saved_pos, 2);
    assert(confirmed.accepted);
    assert(state.size() == 1);
    assert(state.contains(first));
    assert(!state.has_pending_reservation());

    const auto hit = state.lookup(first, 3);
    assert(hit.slot == prepared.slot);
    assert(hit.cur_ids_len == saved_pos);
    assert(state.size() == 1);

    state.clear();
    assert(state.size() == 0);
    assert(!state.has_pending_reservation());
    assert(state.next_relative_slot() == 0);

    ContractState overlaps_staging(
        kStagingSlot - 1, 2, kStagingSlot);
    assert(!overlaps_staging.enabled());
    assert(!overlaps_staging.prepare(first, 1).accepted);
    return 0;
}
