// The key of a cached prefix: a truncated SHA-1 of its token ids. The RAM
// index and the disk files use the same key.

#pragma once

#include <array>
#include <cstdint>

namespace luce::common {

// SHA-1 hash of a prefix (truncated to 16 bytes).
using PrefixHash = std::array<uint8_t, 16>;
PrefixHash hash_prefix(const int32_t * ids, int count);

}  // namespace luce::common
