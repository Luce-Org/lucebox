// The key of a cached prefix: a truncated SHA-1 of its token ids. The RAM
// index and the disk files use the same key.

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace luce::common {

// SHA-1 hash of a prefix (truncated to 16 bytes).
using PrefixHash = std::array<uint8_t, 16>;
PrefixHash hash_prefix(const int32_t * ids, int count);

// Lower-case hex of `len` bytes: the names of what a hash keys (disk files
// and their layout directories).
std::string hex_string(const uint8_t * data, int len);

}  // namespace luce::common
