#include "server/prefix_cache_state.h"

#include <cassert>
#include <cstdint>

#ifndef LUCEBOX_FORMAL_MAX_CAP
#define LUCEBOX_FORMAL_MAX_CAP 4
#endif

extern "C" unsigned int nondet_uint();
extern "C" void __ESBMC_assume(bool);

using dflash::common::select_inline_free_slot;

namespace {

unsigned int bounded(unsigned int upper_exclusive) {
    const unsigned int value = nondet_uint();
    __ESBMC_assume(value < upper_exclusive);
    return value;
}

}  // namespace

int main() {
    const unsigned int capacity =
        bounded(LUCEBOX_FORMAL_MAX_CAP) + 1;
    const unsigned int next_slot = bounded(capacity);
    const uint64_t valid_slots =
        (uint64_t{1} << capacity) - 1;
    const uint64_t occupied_slots =
        (uint64_t)bounded(1u << LUCEBOX_FORMAL_MAX_CAP) & valid_slots;

    // This is the state after a below-capacity reservation aborts: at least
    // one backend slot is free, while the round-robin cursor may point at a
    // still-committed slot.
    __ESBMC_assume(occupied_slots != valid_slots);

    const int selected = select_inline_free_slot(
        (int)next_slot, (int)capacity, occupied_slots);
    assert(selected >= 0);
    assert(selected < (int)capacity);
    assert((occupied_slots & (uint64_t{1} << selected)) == 0);
    return 0;
}
