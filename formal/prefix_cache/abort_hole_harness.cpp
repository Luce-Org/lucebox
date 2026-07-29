#include "server/prefix_cache_state.h"

#include <cstdint>

#ifndef LUCEBOX_FORMAL_MAX_CAP
#define LUCEBOX_FORMAL_MAX_CAP 4
#endif

using dflash::common::select_inline_free_slot;

static_assert(
    LUCEBOX_FORMAL_MAX_CAP > 0 && LUCEBOX_FORMAL_MAX_CAP < 64);

int verify_select_inline_free_slot_contract(
        int next_slot, int capacity, uint64_t occupied_slots) {
    // Model the below-capacity state after an aborted reservation: the cursor
    // may name an occupied slot, but at least one in-range slot remains free.
    __ESBMC_requires(
        capacity > 0 && capacity <= LUCEBOX_FORMAL_MAX_CAP);
    __ESBMC_requires(next_slot >= 0 && next_slot < capacity);
    __ESBMC_requires((occupied_slots >> capacity) == 0);
    __ESBMC_requires(
        occupied_slots != ((uint64_t{1} << capacity) - 1));

    // Slot selection is scalar and must not mutate program state.
    __ESBMC_assigns();
    __ESBMC_ensures(__ESBMC_return_value >= 0);
    __ESBMC_ensures(__ESBMC_return_value < capacity);
    __ESBMC_ensures(
        (occupied_slots &
         (uint64_t{1} << __ESBMC_return_value)) == 0);

    return select_inline_free_slot(
        next_slot, capacity, occupied_slots);
}
