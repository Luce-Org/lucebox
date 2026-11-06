#include "server/kv/prefix_hash.h"
#include "common/sha1.h"

#include <cstring>
#include <vector>

namespace luce::common {

PrefixHash hash_prefix(const int32_t * ids, int count) {
    // Build hash input: [count as LE u32] + [ids as LE i32 array]
    if (count < 0) count = 0;
    std::vector<uint8_t> buf(4 + (size_t)count * 4);
    uint32_t n = (uint32_t)count;
    std::memcpy(buf.data(), &n, 4);
    if (count > 0) std::memcpy(buf.data() + 4, ids, (size_t)count * 4);

    uint8_t sha[20];
    sha1_hash(buf.data(), buf.size(), sha);

    PrefixHash h{};
    std::memcpy(h.data(), sha, 16);
    return h;
}

std::string hex_string(const uint8_t * data, int len) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve((size_t)len * 2);
    for (int i = 0; i < len; ++i) {
        out.push_back(digits[data[i] >> 4]);
        out.push_back(digits[data[i] & 0x0f]);
    }
    return out;
}

}  // namespace luce::common
