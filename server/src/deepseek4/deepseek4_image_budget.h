#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>

namespace dflash::vision {

inline constexpr uint64_t SCRATCH_RESERVATION = 2ULL * 1024 * 1024 * 1024;

inline uint64_t remaining_expert_budget(uint64_t total, uint64_t core,
        uint64_t kv, uint64_t warm, uint64_t safety,
        uint64_t vision_reservation, uint64_t already_resident_workspace = 0) {
    if (already_resident_workspace > vision_reservation) return 0;
    for (uint64_t charge : {core, kv, warm, safety,
                            vision_reservation - already_resident_workspace}) {
        if (charge > total) return 0;
        total -= charge;
    }
    return total;
}

} // namespace dflash::vision
