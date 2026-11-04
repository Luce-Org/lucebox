#include "server/kv/prefix_hash.h"
#include "common/sha1.h"

#include <cstring>
#include <vector>

namespace luce::common {

PrefixHash hash_prefix(const int32_t * ids, int count) {
    // Build hash input: [count as LE u32] + [ids as LE i32 array]
    std::vector<uint8_t> buf(4 + count * 4);
    uint32_t n = (uint32_t)count;
    std::memcpy(buf.data(), &n, 4);
    std::memcpy(buf.data() + 4, ids, count * 4);

    uint8_t sha[20];
    sha1_hash(buf.data(), buf.size(), sha);

    PrefixHash h{};
    std::memcpy(h.data(), sha, 16);
    return h;
}

}  // namespace luce::common
