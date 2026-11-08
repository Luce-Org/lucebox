// FNV-1a 64: one step per word (kvflash ledger and question-id hashes) or per byte (blobs).
#pragma once

#include <cstddef>
#include <cstdint>

namespace luce::common {

constexpr uint64_t kFnvBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

inline uint64_t fnv1a_mix(uint64_t h, uint64_t word) { return (h ^ word) * kFnvPrime; }

inline uint64_t fnv1a_bytes(uint64_t h, const uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; i++) h = fnv1a_mix(h, p[i]);
    return h;
}

}  // namespace luce::common
